"""Synthetic CPU contracts only. Never imports or loads the native engine."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.encoding import candidate_suffix, identity as encoding_identity, query_suffix, token_hash
from fuzev_research.head import HeadBundle, save_bundle
from fuzev_research.native_service import NativeService, build_service, generation_timing
from fuzev_research.service_contracts import GenerationRequest, ServiceConfig, ServiceDecisionRequest
from fuzev_research.serving_head import ServingHead


class Guard:
    def start(self):
        return self

    def check(self):
        pass

    def close(self):
        pass


class Lane:
    context_limit = 20000
    chunk_size = 8192
    batch_capacity = 4

    def __init__(self, batch_capacity=4):
        self.batch_capacity = batch_capacity
        self.branches, self.calls, self.many = [], [], []
        self.output = [91, 92, 99]
        self.scratch = torch.zeros(batch_capacity, 1, 2560)
        self.params = []
        self.after_append = lambda: None

    def create(self):
        assert sum(not b.closed and not b.immutable for b in self.branches) < self.batch_capacity
        branch = SimpleNamespace(tokens=(), closed=False, immutable=False)
        self.branches.append(branch)
        return branch

    def fork(self, parent):
        assert parent.immutable and not parent.closed
        branch = self.create()
        branch.tokens = parent.tokens
        return branch

    def freeze(self, branch):
        assert not branch.closed
        branch.immutable = True
        return branch

    def release(self, branch):
        assert not branch.closed
        branch.closed = True

    def append(self, branch, tokens, *, features=False, params=None):
        assert not branch.closed and not branch.immutable
        self.params.append(params)
        self.calls.append((branch.tokens, tuple(tokens), features))
        branch.tokens += tuple(tokens)
        self.after_append()
        if features:
            self.scratch.fill_(sum(branch.tokens))
            return self.scratch[:1]
        logits = torch.full((1, 1, 512), -torch.inf)
        logits[0, 0, self.output.pop(0)] = 1.
        return logits

    def append_features_many(self, branches, tails, *, params=None):
        assert len({b.tokens for b in branches}) == 1
        assert len({len(tail) for tail in tails}) == 1
        assert 2 <= len(branches) <= self.batch_capacity
        self.params.append(params)
        self.many.append((tuple(b.tokens for b in branches), tails))
        for index, (branch, tail) in enumerate(zip(branches, tails)):
            branch.tokens += tuple(tail)
            self.scratch[index].fill_(sum(branch.tokens))
        self.after_append()
        return self.scratch[:len(branches)]


class Backend:
    candidate_batch_size = 4
    batch_capacity = 4
    readout_runtime = {"candidate_batch_size": 4, "batch_capacity": 4,
                       "model_batch_capacity": 4, "max_hybrid_slots": 4, "memory_fraction": None}

    def __init__(self, batch_capacity=4, candidate_batch_size=4, memory_fraction=None):
        self.batch_capacity, self.candidate_batch_size = batch_capacity, candidate_batch_size
        self.readout_runtime = {**self.readout_runtime, "candidate_batch_size": candidate_batch_size,
                                "batch_capacity": batch_capacity, "model_batch_capacity": batch_capacity,
                                "max_hybrid_slots": batch_capacity, "memory_fraction": memory_fraction}
        self.lane = Lane(batch_capacity)
        self.prefixes, self.images = {}, {}
        self.config = SimpleNamespace(eos_token_id_list=[99])
        self.bases, self.renders = [], []
        self.tokenizer = SimpleNamespace(decode=self.decode, hf_render_chat_template=self.render)

    @staticmethod
    def encode(text, embeddings=None):
        assert not embeddings
        return tuple(ord(char) + 1 for char in text)

    def parameters(self, _tokens, embeddings, suffix=()):
        assert not embeddings
        return {}

    def render(self, messages, **kwargs):
        self.renders.append((messages, kwargs))
        return "chat-a"

    def base_tokens(self, messages):
        self.bases.append(messages)
        return self.encode("base-a"), []

    @staticmethod
    def decode(tokens):
        return "".join({91: "ST", 92: "OPafter", 99: ""}[token] for token in tokens.tolist())

    def reserve(self, _tokens):
        pass


class Head:
    head_id = "reference"
    calibration_id = "sha256:unit-calibration"
    manifest_sha256 = "unit-manifest"

    def score(self, primitive, query, candidates):
        self.observed = (primitive, query.clone(), candidates.clone())
        logits = candidates[:, 0].tolist()
        raw = torch.softmax(candidates[:, 0], 0).tolist()
        return logits, raw, raw


@pytest.fixture
def service():
    value = NativeService(Backend(), "unit", Head(), {"runtime": "synthetic"}, Guard(),
                          admission=lambda *_: None)
    yield value
    value.close()
    assert all(branch.closed for branch in value.lane.branches)


def generation(**kwargs):
    return GenerationRequest(model="unit", prompt="A", max_tokens=3, temperature=0, **kwargs)


def decision(**kwargs):
    values = {"model": "unit", "primitive": "choice", "question": "Which?",
              "messages": [{"role": "user", "content": "A"}],
              "candidates": [{"id": "first", "text": "xx"}, {"id": "second", "text": "yy"}]}
    values.update(kwargs)
    return ServiceDecisionRequest(**values)


@pytest.mark.parametrize("stop,finish,expected_text,expected_tokens", [
    ([], "stop", "STOPafter", [91, 92, 99]),
    (["STOP"], "stop", "", [91, 92]),
])
def test_exact_accepted_tokens_include_hidden_eos_and_partial_token_stops(
        service, stop, finish, expected_text, expected_tokens):
    output = service.generate(generation(stop=stop), Event(), "generate")
    assert output["choices"][0] == {"index": 0, "finish_reason": finish, "text": expected_text}
    assert output["completion_token_ids"] == expected_tokens
    prefix = service._get_prefix(output["prefix_id"])
    assert prefix.accepted == service.backend.encode("A") + tuple(expected_tokens)
    assert prefix.branch.tokens == prefix.accepted[:-1]
    assert output["unconsumed_token_ids"] == expected_tokens[-1:]
    assert output["accepted_token_sha256"] == token_hash(prefix.accepted)

    before = len(service.lane.calls)
    request = decision(messages=None, prefix_id=prefix.id, additional_context="new C evidence")
    result = service.decide(request, Event(), "decision")
    calls = service.lane.calls[before:]
    suffix = service.backend.encode(query_suffix(
        request.question, [c.model_dump() for c in request.candidates], request.additional_context))
    assert calls[0] == (prefix.accepted[:-1], prefix.accepted[-1:], True)
    assert calls[1] == (prefix.accepted, suffix, True)
    assert all(features for _, _, features in calls)  # No vocabulary projection.
    assert result["query_token_sha256"] == token_hash(prefix.accepted + suffix)
    assert result["resolved_unconsumed_tokens"] == 1
    assert result["execution"]["reused_tokens"] == len(prefix.accepted) - 1
    assert service.backend.bases == []  # AB never rerendered or retokenized.
    again = service.decide(request, Event(), "independent")
    assert again["scores"] == result["scores"]
    assert service._get_prefix(prefix.id).branch.tokens == prefix.accepted[:-1]


def test_length_stop_and_chat_render_only_a(service):
    request = GenerationRequest(model="unit", messages=[{"role": "user", "content": "A"}],
                                max_tokens=1, temperature=0)
    output = service.generate(request, Event(), "chat")
    assert output["choices"][0]["finish_reason"] == "length"
    assert output["unconsumed_token_ids"] == [91]
    assert output["consumed_tokens"] == len(service.backend.encode("chat-a"))
    assert service.backend.renders[0][1]["add_generation_prompt"] is True
    assert service.backend.renders[0][1]["enable_thinking"] is False


@pytest.mark.parametrize("stop,expected_tokens,times,prefill,decode", [
    ([], [91, 92, 99], [100., 102., 105., 109.], 2., 7.),
    (["STOP"], [91, 92], [100., 102., 105.], 2., 3.),
])
def test_phase_timing_uses_existing_cpu_sampling_boundary(
        service, stop, expected_tokens, times, prefill, decode):
    clock = iter(times)
    service.phase_clock = lambda: next(clock)
    output = service.generate(generation(stop=stop), Event(), "phases")
    assert output["completion_token_ids"] == expected_tokens
    assert output["unconsumed_token_ids"] == expected_tokens[-1:]
    assert len(service.lane.calls) == len(expected_tokens)
    assert [call[1] for call in service.lane.calls] == [service.backend.encode("A")] + [
        (token,) for token in expected_tokens[:-1]]
    assert all(not call[2] for call in service.lane.calls)
    timing = output["phase_timing"]
    assert timing["source"] == "native-owner-cpu-observed"
    assert timing["status"] == "available" and timing["reason"] is None
    assert timing["prefill_seconds"] == prefill
    assert timing["decode_seconds"] == decode
    assert timing["decode_tokens"] == len(expected_tokens) - 1
    assert timing["completion_tokens"] == len(expected_tokens)
    assert timing["prompt_tokens"] == len(service.backend.encode("A"))
    assert timing["reported_cached_prompt_tokens"] == 0
    assert timing["queue_seconds"] is None and timing["num_preemptions"] is None
    assert next(clock, None) is None


def test_single_sample_has_no_decode_work(service):
    clock = iter([100., 102.])
    service.phase_clock = lambda: next(clock)
    request = GenerationRequest(model="unit", prompt="A", max_tokens=1, temperature=0)
    output = service.generate(request, Event(), "single-sample")
    assert output["phase_timing"]["decode_tokens"] == 0
    assert output["phase_timing"]["decode_seconds"] == 0
    assert len(service.lane.calls) == 1
    assert output["completion_token_ids"] == [91]


@pytest.mark.parametrize("started,first,last,count", [
    (0., 0., 1., 2), (2., 1., 3., 2), (0., 2., 1., 2),
    (0., 1., 1., 2), (0., 1., 2., 1),
    (float("nan"), 1., 2., 2), (0., float("inf"), 2., 2),
    (0., 1., None, 2), (False, 1., 2., 2),
])
def test_bad_phase_clock_never_invents_speed(started, first, last, count):
    result = generation_timing(10, count, started, first, last)
    assert result["status"] == "unavailable"
    assert result["prefill_seconds"] is None and result["decode_seconds"] is None


def test_unavailable_timing_does_not_change_accepted_generation(service):
    service.phase_clock = lambda: 0.
    output = service.generate(generation(), Event(), "unavailable-timing")
    assert output["phase_timing"]["status"] == "unavailable"
    assert output["completion_token_ids"] == [91, 92, 99]
    assert output["unconsumed_token_ids"] == [99]
    assert output["choices"][0]["finish_reason"] == "stop"
    assert len(service.lane.calls) == 3


def test_standalone_training_encoding_owned_features_and_max4_equal_length_groups(service):
    request = decision(candidates=[{"id": str(i), "text": "abc" if i < 5 else "xy"}
                                   for i in range(7)])
    result = service.decide(request, Event(), "standalone")
    base = service.backend.encode("base-a")
    suffix = service.backend.encode(query_suffix(request.question, [c.model_dump() for c in request.candidates]))
    assert service.backend.bases == [request.messages]
    assert result["candidate_forward_batches"] == [4, 1, 2]
    assert result["layout"] == "standalone-training-encoding"
    assert result["query_token_sha256"] == token_hash(base + suffix)
    primitive, owned_query, keys = service.head.observed
    assert primitive == "choice"
    assert owned_query[0] == sum(base + suffix)  # Scratch changed during tails.
    for index, candidate in enumerate(request.candidates):
        tail = service.backend.encode(candidate_suffix(candidate.model_dump()))
        assert keys[index, 0] == sum(base + suffix + tail)
    assert [score["id"] for score in result["scores"]] == [str(i) for i in range(7)]
    assert all(branch.closed for branch in service.lane.branches)


@pytest.mark.parametrize("capacity,size,count,batches", [
    (4, 4, 5, [4, 1]), (8, 4, 5, [4, 1]), (8, 8, 5, [5]), (8, 8, 16, [8, 8]),
])
def test_scoring_follows_pinned_physical_and_candidate_geometry(capacity, size, count, batches):
    backend = Backend(capacity, size)
    service = NativeService(backend, "unit", Head(), {}, Guard(), admission=lambda *_: None)
    try:
        request = decision(candidates=[{"id": str(i), "text": "same"} for i in range(count)])
        result = service.decide(request, Event(), "geometry")
        assert result["candidate_forward_batches"] == batches
        assert [score["id"] for score in result["scores"]] == [str(i) for i in range(count)]
    finally:
        service.close()


@pytest.mark.parametrize("field", ["candidate_batch_size", "batch_capacity", "lane", "fingerprint"])
def test_changed_geometry_rejected_before_forward(service, field):
    if field == "lane":
        service.lane.batch_capacity = 8
    elif field == "fingerprint":
        service.backend.readout_runtime["memory_fraction"] = .90
    else:
        setattr(service.backend, field, 8)
    with pytest.raises(ValueError, match="geometry changed"):
        service.decide(decision(), Event(), "bad")
    assert not service.lane.calls and not service.lane.many


def test_physical_capacity_and_fingerprint_must_agree():
    backend = Backend(8, 8)
    backend.lane.batch_capacity = 4
    with pytest.raises(ValueError, match="resource bounds"):
        NativeService(backend, "unit", Head(), {}, Guard())


@pytest.mark.parametrize("failure", ["cancel", "forward", "memory", "head"])
def test_failure_cleanup_does_not_mutate_retained_prefix(service, failure):
    prefix_id = service.generate(generation(), Event(), "a")["prefix_id"]
    prefix = service._get_prefix(prefix_id)
    original = prefix.branch.tokens
    cancel = Event()
    if failure == "cancel":
        service.lane.after_append = cancel.set
        expected = InterruptedError
    elif failure == "forward":
        def fail():
            raise RuntimeError("synthetic forward failure")
        service.lane.after_append = fail
        expected = RuntimeError
    elif failure == "memory":
        def fail(*_):
            raise MemoryError("synthetic guard")
        service.admission = fail
        expected = MemoryError
    else:
        def fail(*_):
            raise ValueError("synthetic head failure")
        service.head.score = fail
        expected = ValueError
    with pytest.raises(expected):
        service.decide(decision(messages=None, prefix_id=prefix_id), cancel, "b")
    assert prefix.branch.tokens == original and not prefix.branch.closed
    assert all(branch.closed or branch is prefix.branch for branch in service.lane.branches)


def test_cancelled_generation_releases_all_state(service):
    cancel = Event()
    service.lane.after_append = cancel.set
    with pytest.raises(InterruptedError):
        service.generate(generation(), cancel, "cancel")
    assert not service.prefixes
    assert all(branch.closed for branch in service.lane.branches)


def test_ttl_capacity_release_and_incompatible_prefix_fail_closed(service):
    now = [0.]
    service.clock, service.ttl, service.capacity = lambda: now[0], 5, 1
    first = service.generate(generation(), Event(), "one")["prefix_id"]
    first_branch = service._get_prefix(first).branch
    service.lane.output = [91, 92, 99]
    second = service.generate(generation(), Event(), "two")["prefix_id"]
    assert first_branch.closed
    with pytest.raises(KeyError):
        service._get_prefix(first)
    prefix, used = service.prefixes[second]
    service.prefixes[second] = (replace(prefix, identity="different"), used)
    with pytest.raises(ValueError, match="Incompatible"):
        service._get_prefix(second)
    service.prefixes[second] = (prefix, used)
    now[0] = 5.
    service.expire()
    assert not service.prefixes and prefix.branch.closed


@pytest.mark.parametrize("change", [
    {"model": "foreign"}, {"head_id": "untrained"},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "no"}]}]},
    {"question": "<|image_pad|>"},
])
def test_unsupported_or_incompatible_decisions_fail_before_forward(service, change):
    with pytest.raises(ValueError):
        service.decide(decision(**change), Event(), "bad")
    assert not service.lane.calls


def test_context_limits_and_owner_thread(service):
    service.lane.context_limit = 2
    with pytest.raises(ValueError, match="context window"):
        service.generate(generation(), Event(), "too-long")
    with pytest.raises(ValueError, match="context window"):
        service.decide(decision(), Event(), "too-long")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="single GPU owner"):
            executor.submit(service.expire).result(timeout=2)


@pytest.fixture
def artifact(tmp_path, request):
    identity = {"schema_version": 1, "engine": "optimized-native-exllamav3-fuzev",
                "model": "unit", "revision": "revision", "encoding": encoding_identity(),
                "cache_precision": "float16", "feature_tap": "final-mixer/post-collapse/before-lm-head",
                "hidden_size": 2560, "readout_runtime": Backend.readout_runtime,
                "runtime_sources": {"recipes": "extraction", "exllamav3-fuzev": "native"},
                "datasets": {"unit": "synthetic"}}
    if getattr(request, "param", None) == "scoped":
        from fuzev_research.scope import EXTERNAL_SOURCES, TRAINING_SOURCES, contract
        identity.update(dataset_scope=contract(),
                        datasets=dict.fromkeys(TRAINING_SOURCES + EXTERNAL_SOURCES, "synthetic"))
    membership = {"records": 1, "ordered_sha256": "a" * 64}
    metadata = {"artifact_kind": "calibrated-reference-head", "stage": "mixed",
                "feature_identity": identity, "calibration_role": "calibration",
                "preparation": {"synthetic": True}, "training_code": {"synthetic": True},
                "calibration_code": {"synthetic": True},
                **{key: membership for key in ("fit_membership", "development_membership", "calibration_membership")}}
    save_bundle(tmp_path, HeadBundle(), metadata, dict(choice=2., noul=1., ordinal=.5))
    config = ServiceConfig(
        head_directory=str(tmp_path), feature_identity=identity,
        service_sources={"recipes": "serving", "exllamav3-fuzev": "native"},
        manifest_sha256=hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256((tmp_path / "calibration.json").read_bytes()).hexdigest())
    return config


def test_fp32_calibrated_heads_preserve_candidate_mapping(artifact):
    head = ServingHead(artifact)
    assert all(parameter.dtype == torch.float32 and not parameter.requires_grad
               for parameter in head.bundle.parameters())
    logits, raw, calibrated = head.score("choice", torch.ones(2560), torch.stack([
        torch.zeros(2560), torch.ones(2560)]))
    assert torch.tensor(raw).sum() == pytest.approx(1.)
    assert calibrated == pytest.approx(torch.softmax(torch.tensor(logits) / 2, 0).tolist())
    assert head.calibration_id == "sha256:" + artifact.calibration_sha256


@pytest.mark.parametrize("file", ["head.safetensors", "manifest.json", "calibration.json"])
def test_serving_checksums_cover_all_artifacts(artifact, file):
    from pathlib import Path
    path = Path(artifact.head_directory) / file
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        ServingHead(artifact)


@pytest.mark.parametrize("key,value", [
    ("artifact_kind", "training-checkpoint"), ("precision", "float16"),
    ("calibration_role", "test"), ("fit_membership", {}), ("feature_identity", {}),
])
def test_untrained_or_incompatible_metadata_rejected_even_when_repinned(artifact, key, value):
    from pathlib import Path
    path = Path(artifact.head_directory) / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    artifact.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        ServingHead(artifact)


def test_current_runtime_and_source_receipt_required_before_head_load(artifact, monkeypatch):
    from fuzev_research import native_service
    monkeypatch.setattr(native_service, "admit", lambda *_: None)
    monkeypatch.setattr(native_service, "MemoryGuard", lambda **_: Guard())
    model = {"repo": "unit", "revision": "revision"}
    service = build_service(Backend(), "unit", model, artifact.service_sources, artifact)
    assert service.owner == get_ident()
    assert service.provenance["head_manifest_sha256"] == artifact.manifest_sha256
    assert service.provenance["calibration_sha256"] == artifact.calibration_sha256
    assert service.provenance["feature_identity"]["runtime_sources"]["recipes"] == "extraction"
    assert service.provenance["service_sources"]["recipes"] == "serving"
    service.close()
    with pytest.raises(ValueError, match="feature identity"):
        build_service(Backend(), "unit", {**model, "revision": "other"}, artifact.service_sources, artifact)
    backend = Backend()
    backend.readout_runtime = {**backend.readout_runtime, "candidate_batch_size": 1}
    with pytest.raises(ValueError, match="feature identity"):
        build_service(backend, "unit", model, artifact.service_sources, artifact)


def serving_identity(artifact, backend, model, cell="native-3.05"):
    identity = {key: deepcopy(artifact.feature_identity[key]) for key in (
        "schema_version", "engine", "encoding", "cache_precision", "feature_tap", "hidden_size")}
    return {"cell": cell, **identity, "model": model["repo"], "revision": model["revision"],
            "readout_runtime": deepcopy(backend.readout_runtime)}


@pytest.mark.parametrize("artifact", ["scoped"], indirect=True)
@pytest.mark.parametrize("cell", ["native-3.05", "native-5.05"])
def test_explicit_native_bitrate_pins_keep_scoped_reference_head_frozen(artifact, cell, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fuzev_research import server

    backend = Backend(8, 8)
    snapshot = tmp_path / "synthetic-snapshot"
    snapshot.mkdir()
    model = {"repo": "unit", "revision": "synthetic-" + cell, "local_snapshot": str(snapshot)}
    actual = serving_identity(artifact, backend, model, cell)
    config = artifact.model_copy(update={"serving_identity": actual})
    acquisition = tmp_path / "data/manifests/acquisition.json"
    acquisition.parent.mkdir(parents=True)
    acquisition.write_text(json.dumps({"models": {cell: model}}))
    selected = []

    def make_backend(directory, **kwargs):
        selected.append((directory, kwargs))
        return backend

    backend.close = lambda: selected.append("closed")
    monkeypatch.setattr(server, "project_path", lambda path: tmp_path / path)
    monkeypatch.setattr(server, "runtime_sources", lambda: artifact.service_sources)
    monkeypatch.setattr(server, "NativeBackend", make_backend)
    directory = Path(config.head_directory)
    frozen = {name: (directory / name).read_bytes() for name in (
        "head.safetensors", "manifest.json", "calibration.json")}
    app = server.create_app(cell, 8, 8, service_config=config)
    with TestClient(app) as client:
        service = app.state.service
        assert client.get("health").json()["phase"] == "shared-native-unqualified"
        assert service.provenance["feature_identity"] == artifact.feature_identity
        assert service.provenance["serving_identity"] == actual
        assert service.provenance["feature_identity"]["revision"] == "revision"
        assert service.provenance["serving_identity"]["revision"] == model["revision"]
        assert service.provenance["feature_identity"]["readout_runtime"] != actual["readout_runtime"]
        assert service.provenance["transfer_qualification"].startswith("unqualified")
        for primitive in ("choice", "noul", "ordinal"):
            candidates = [{"id": key, "text": key,
                           **({"ordinal_value": float(i)} if primitive == "ordinal" else {})}
                          for i, key in enumerate(("yes", "no") if primitive == "noul" else ("a", "b"))]
            response = client.post("/v1/decisions", json=decision(
                model=cell, primitive=primitive, candidates=candidates).model_dump(mode="json"))
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["head_manifest_sha256"] == artifact.manifest_sha256
            assert result["calibration_id"] == "sha256:" + artifact.calibration_sha256
        assert not any(p.requires_grad for p in service.head.bundle.parameters())
        actual["revision"] = "caller-mutation"
        assert service.provenance["serving_identity"]["revision"] == model["revision"]
    assert selected == [(str(snapshot), {"batch_capacity": 8, "candidate_batch_size": 8,
                                         "memory_fraction": None}), "closed"]
    assert service.guard.stop.is_set()
    assert all(branch.closed for branch in backend.lane.branches)
    assert frozen == {name: (directory / name).read_bytes() for name in frozen}


@pytest.mark.parametrize("cell", ["native-3.05", "native-5.05"])
def test_native_bitrate_transfer_remains_strict_without_actual_identity(artifact, cell, monkeypatch):
    from fuzev_research import native_service
    monkeypatch.setattr(native_service, "ServingHead", lambda _: pytest.fail("Rejected pins loaded a head"))
    assert artifact.serving_identity is None
    with pytest.raises(ValueError, match="head's feature identity"):
        build_service(Backend(), cell, {"repo": "unit", "revision": cell}, artifact.service_sources, artifact)


@pytest.mark.parametrize("field", [
    "cell", "schema_version", "engine", "model", "revision", "encoding", "cache_precision",
    "feature_tap", "hidden_size", "readout_runtime", "missing", "extra",
])
def test_explicit_actual_native_pins_fail_closed_before_head_load(artifact, field, monkeypatch):
    from fuzev_research import native_service
    monkeypatch.setattr(native_service, "ServingHead", lambda _: pytest.fail("Rejected pins loaded a head"))
    backend, model = Backend(), {"repo": "unit", "revision": "synthetic-3.05"}
    actual = serving_identity(artifact, backend, model)
    if field == "missing":
        actual.pop("revision")
    elif field == "extra":
        actual["unverified"] = True
    else:
        actual[field] = "wrong"
    config = artifact.model_copy(update={"serving_identity": actual})
    with pytest.raises(ValueError, match="operator's pins"):
        build_service(backend, "native-3.05", model, artifact.service_sources, config)


@pytest.mark.parametrize("field", ["schema_version", "engine", "encoding", "feature_tap", "hidden_size"])
def test_explicit_native_transfer_still_requires_portable_reference_features(artifact, field, monkeypatch):
    from fuzev_research import native_service
    monkeypatch.setattr(native_service, "ServingHead", lambda _: pytest.fail("Rejected reference loaded a head"))
    backend, model = Backend(), {"repo": "unit", "revision": "synthetic-5.05"}
    actual = serving_identity(artifact, backend, model, "native-5.05")
    reference = {**artifact.feature_identity, field: "wrong"}
    config = artifact.model_copy(update={"serving_identity": actual, "feature_identity": reference})
    with pytest.raises(ValueError, match="native encoding/features"):
        build_service(backend, "native-5.05", model, artifact.service_sources, config)


@pytest.mark.parametrize("field", ["runtime_sources", "datasets", "revision"])
def test_explicit_native_transfer_cannot_relabel_or_discard_reference_provenance(artifact, field):
    backend, model = Backend(), {"repo": "unit", "revision": "synthetic-5.05"}
    actual = serving_identity(artifact, backend, model, "native-5.05")
    reference = {**artifact.feature_identity, field: model["revision"] if field == "revision" else {}}
    config = artifact.model_copy(update={"serving_identity": actual, "feature_identity": reference})
    with pytest.raises(ValueError, match="extraction provenance|serving head provenance"):
        build_service(backend, "native-5.05", model, artifact.service_sources, config)


@pytest.mark.parametrize("cell", ["native-2.05", "vllm-exl3-4.05"])
def test_explicit_native_transfer_whitelists_only_required_native_cells(artifact, cell):
    backend, model = Backend(), {"repo": "unit", "revision": "revision"}
    config = artifact.model_copy(update={"serving_identity": serving_identity(artifact, backend, model, cell)})
    with pytest.raises(ValueError, match="required native bitrate cell"):
        build_service(backend, cell, model, artifact.service_sources, config)


@pytest.fixture
def http_app(tmp_path, monkeypatch):
    from fuzev_research import server
    manifest = tmp_path / "data/manifests/acquisition.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"models": {"unit": {"local_snapshot": "synthetic"}}}))
    monkeypatch.setattr(server, "project_path", lambda path: tmp_path / path)
    monkeypatch.setattr(server, "runtime_sources", lambda: {"recipes": "synthetic"})
    backends, owners = [], []

    def make_backend(_directory, **kwargs):
        owners.append(get_ident())
        backend = Backend(**kwargs)
        backend.extract = lambda *args: (torch.zeros(2560), torch.zeros(2, 2560), None, {})
        backend.close = lambda: owners.append(get_ident())
        backends.append(backend)
        return backend

    def make_service(backend, name, model, sources, config):
        owners.append(get_ident())
        return NativeService(backend, name, Head(), {"runtime": "synthetic"}, Guard(),
                             admission=lambda *_: None)

    monkeypatch.setattr(server, "NativeBackend", make_backend)
    monkeypatch.setattr(server, "build_service", make_service)
    app = server.create_app("unit", 8, 8, service_config=SimpleNamespace(prefix_ttl_seconds=30))
    return app, backends, owners


def test_http_routes_share_one_backend_and_keep_feature_service(http_app):
    from fastapi.testclient import TestClient
    app, backends, owners = http_app
    with TestClient(app) as client:
        response = client.post("/v1/completions", json=generation().model_dump())
        assert response.status_code == 200, response.text
        prefix_id = response.json()["prefix_id"]
        response = client.post("/v1/decisions", json=decision(
            messages=None, prefix_id=prefix_id).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.json()["resolved_unconsumed_tokens"] == 1
        row = {"primitive": "choice", "question": "Which?", "context": [],
               "candidates": [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}]}
        assert client.post("/v1/features", json={"row": row}).status_code == 200
        assert client.delete(f"/v1/prefixes/{prefix_id}").json()["released"]
        assert client.post("/v1/decisions", json=decision(
            messages=None, prefix_id=prefix_id).model_dump(mode="json")).status_code == 404
        health = client.get("/health").json()
        assert health["phase"] == "shared-native-unqualified"
        assert health["service_capabilities"]["trusted_still_images"]
        assert health["candidate_batch_size"] == 8
    assert len(backends) == 1
    assert len(owners) == 3 and len(set(owners)) == 1


def test_http_cancel_waits_for_forward_cleanup(http_app):
    from fastapi.testclient import TestClient
    app, backends, _ = http_app
    ready, release = Event(), Event()
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as callers:
        def forward_boundary():
            ready.set()
            assert release.wait(timeout=3)
        backends[0].lane.after_append = forward_boundary
        future = callers.submit(client.post, "/v1/completions", json=generation().model_dump(),
                                headers={"X-Fuzev-Request-ID": "cancel-me"})
        try:
            assert ready.wait(timeout=2)
            response = client.post("/v1/requests/cancel-me/cancel")
            assert response.status_code == 202
            assert not future.done()
        finally:
            release.set()
        assert future.result(timeout=2).status_code == 499
        assert not app.state.service.prefixes
        assert all(branch.closed for branch in backends[0].lane.branches)
