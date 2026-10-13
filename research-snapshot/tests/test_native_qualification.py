"""CPU fake contracts only: no native package bootstrap, data, or backbone."""

from copy import deepcopy
import json
import subprocess
import sys
from threading import Thread
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")

from fuzev_research import native_qualification as qualification
from fuzev_research.native_service import NativeService
from test_native_service import Backend, Guard, Head, Lane


class LedgerLane(Lane):
    context_limit = 262144

    def __init__(self, cache, batch_capacity=4):
        super().__init__(batch_capacity)
        self.cache = cache
        self.pool = SimpleNamespace(free=set(range(64)))
        self.corrupt_reuse = False

    def create(self):
        branch = super().create()
        branch.slot = self.cache.free_list.pop()
        branch.page = self.pool.free.pop()
        return branch

    def freeze(self, branch):
        if branch.slot is not None:
            self.cache.free_list.append(branch.slot)
            branch.slot = None
        return super().freeze(branch)

    def release(self, branch):
        super().release(branch)
        if branch.slot is not None:
            self.cache.free_list.append(branch.slot)
            branch.slot = None
        self.pool.free.add(branch.page)
        branch.tokens = ()  # Released state need not remain inspectable.

    def append(self, branch, tokens, **kwargs):
        if not branch.tokens:
            self.output = [91, 92, 99]
        result = super().append(branch, tokens, **kwargs)
        if kwargs.get("features", False) and self.corrupt_reuse and len(tokens) == 1:
            branch.tokens += (7,)
        return result


class FrozenFakeHead(Head):
    def __init__(self):
        self.bundle = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        self.temperatures = {"choice": 1., "noul": 1., "ordinal": 1.}


@pytest.fixture
def service():
    backend = Backend()
    backend.cache = SimpleNamespace(free_list=list(range(4)), num_slots=4, max_num_tokens=294912)
    backend.lane = LedgerLane(backend.cache)
    backend.readout_runtime.update(context_tokens=262144, feature_chunk_tokens=8192, cache_tokens=294912)
    value = NativeService(backend, "unit", FrozenFakeHead(), {"synthetic": True}, Guard(),
                          admission=lambda *_: None)
    yield value
    value.close()
    assert all(branch.closed for branch in value.lane.branches)
    assert len(value.lane.pool.free) == 64
    assert len(value.backend.cache.free_list) == 4


def case(name="exact", **generation):
    return {"name": name, "input_origin": "local CPU fake, no dataset", "input_role": "cpu-fake",
            "generation": {"model": "unit", "prompt": "A", "temperature": 0, "max_tokens": 3, **generation},
            "decisions": [{"primitive": "choice", "question": f"CPU fake branch {index}?",
                           "additional_context": f"C{index}",
                           "candidates": [{"id": "a", "text": "xx"}, {"id": "b", "text": "yy"}]}
                          for index in range(4)]}


def run(service, config=None, **kwargs):
    return qualification.run_qualification(service, config or {"cases": [case()]},
                                           allow_inference=True, cpu_fake=True, **kwargs)


def assert_passed(report):
    assert report.get("execution_status") == "passed", report
    assert report["status"] == "incomplete" and report["qualified"] is False
    assert report["evidence_kind"] == "cpu_fake"
    assert report["cleanup_errors"] == []
    assert report["gates"]["complete_hybrid_state"]["status"] == "unavailable"


def test_explicit_opt_in_precedes_even_service_or_config_access():
    class Untouchable:
        def __getattribute__(self, _name):
            pytest.fail("Service touched without explicit opt-in")

    for opt_in in (False, None, 1, "yes"):
        with pytest.raises(ValueError, match="explicit inference opt-in"):
            qualification.run_qualification(Untouchable(), {}, allow_inference=opt_in)


def test_import_and_disabled_entrypoint_do_not_need_torch_or_native_engine():
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
sys.modules['exllamav3'] = None
from fuzev_research.native_qualification import run_qualification
try:
    run_qualification(None, None)
except ValueError as error:
    assert 'explicit inference opt-in' in str(error)
else:
    raise AssertionError('Disabled inference did not refuse')
assert 'fuzev_research.native' not in sys.modules
assert 'fuzev_research.native_service' not in sys.modules
"""], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("change", [{"atol": 1e-4}, {"rtol": 1e-3}, {"atol": 0}, {"rtol": -1},
                                    {"rtol": float("nan")}, {"atol": float("inf")},
                                    {"run_full_window": True}, {"unrecognized": True}])
def test_invalid_config_cannot_forward(service, change):
    with pytest.raises(ValidationError):
        run(service, {"cases": [case()], **change})
    assert service.lane.calls == []


def test_exact_replay_four_branches_and_all_accepted_boundaries(service):
    report = run(service, {"cases": [case("eos"), case("partial-stop", stop=["STOP"]),
                                      case("length", max_tokens=1)]})
    assert_passed(report)
    assert report["gates"]["accepted_boundaries"] == {"status": "passed", "observed": ["eos", "length", "stop"]}
    assert [item["pending_token_ids"] for item in report["cases"]] == [[99], [92], [91]]
    assert all(len(item["branches"]) == 4 for item in report["cases"])
    assert all(branch["fresh_vs_reused"]["query_features"]["max_abs_error"] == 0
               for item in report["cases"] for branch in item["branches"])
    assert service.backend.bases == []  # No standalone substitute for exact accepted AB.
    assert any(not before and tuple(tokens) == (66, 91, 92, 99) and features
               for before, tokens, features in service.lane.calls)
    assert "_append" not in vars(service) and "score" not in vars(service.head)
    assert "weights" not in report["metadata"]
    assert report["gates"]["full_window"]["status"] == "not_requested"
    assert report["gates"]["primitives"] == {"status": "incomplete", "observed": ["choice"]}
    json.dumps(report, allow_nan=False)


def test_all_three_frozen_primitives_use_the_same_oracle(service):
    raw = case()
    raw["decisions"][1].update(primitive="noul", candidates=[{"id": "yes", "text": "yes"},
                                                           {"id": "no", "text": "no"}])
    raw["decisions"][2].update(primitive="ordinal", candidates=[
        {"id": "low", "text": "lower", "ordinal_value": -1},
        {"id": "mid", "text": "equal", "ordinal_value": 0},
        {"id": "high", "text": "higher", "ordinal_value": 1}])
    report = run(service, {"cases": [raw]})
    assert_passed(report)
    assert report["gates"]["primitives"] == {"status": "passed", "observed": ["choice", "noul", "ordinal"]}


@pytest.mark.parametrize("value", ["true", 1, None])
def test_full_window_opt_in_is_not_coerced(value):
    with pytest.raises(ValidationError):
        qualification.QualificationConfig(cases=[case()], run_full_window=value,
            full_window={"case": case("full"), "additional_peak_bytes": 1, "estimate_basis": "CPU fake"})


@pytest.mark.parametrize("field,value", [("memory_fraction", float("nan")), ("memory_fraction", True),
                                        ("context_tokens", 131072), ("feature_chunk_tokens", 64)])
def test_resource_geometry_never_silently_downgrades(service, field, value):
    service.backend.readout_runtime[field] = value
    with pytest.raises(ValueError):
        run(service)
    assert service.lane.calls == []


def test_live_nonempty_service_is_not_touched(service):
    service.backend.prefixes["foreign"] = object()
    with pytest.raises(ValueError, match="empty, exclusive"):
        run(service)
    assert "foreign" in service.backend.prefixes and service.lane.calls == []
    service.backend.prefixes.clear()


def test_wrong_owner_is_rejected_before_work(service):
    errors = []

    def other_owner():
        try:
            run(service)
        except RuntimeError as error:
            errors.append(str(error))

    thread = Thread(target=other_owner)
    thread.start()
    thread.join()
    assert errors and "single GPU owner" in errors[0]
    assert service.lane.calls == []


@pytest.mark.parametrize("value", ["heldout", "operator-non-heldout"])
def test_input_role_is_not_silently_relabelled(service, value):
    raw = case()
    raw["input_role"] = value
    with pytest.raises(ValueError):
        run(service, {"cases": [raw]})
    assert service.lane.calls == []


def test_corrupt_reuse_fails_and_cleans_owned_prefix(service):
    service.lane.corrupt_reuse = True
    report = run(service)
    assert report["status"] == "failed" and "exact accepted AB" in report["error"]
    assert not service.prefixes and report["cleanup_errors"] == []
    assert "_append" not in vars(service) and "score" not in vars(service.head)


def test_batched_candidate_cross_talk_is_detected(service):
    original = service.lane.append_features_many

    def corrupt(*args, **kwargs):
        return original(*args, **kwargs).add_(10.)

    service.lane.append_features_many = corrupt
    report = run(service)
    assert report["status"] == "failed" and "candidate_features: numerical disagreement" in report["error"]
    assert not service.prefixes


def test_fault_injection_reaches_candidate_work_and_restores_all_hooks(service):
    report = run(service)
    assert_passed(report)
    assert "candidate/head" in report["cases"][0]["fault_checks"]
    assert not {"append", "append_features_many", "fork"}.intersection(vars(service.lane))
    assert len(service.lane.many) >= 16


def test_leaked_service_branch_is_failed_and_best_effort_released(service):
    original = service._release
    skipped = []

    def leak_once(branch):
        if branch is not None and branch.immutable and not skipped:
            skipped.append(branch)
            return
        return original(branch)

    service._release = leak_once
    report = run(service)
    assert report["status"] == "failed" and "leaked native resources" in report["error"]
    assert any("left live branches" in error for error in report["cleanup_errors"])
    assert all(branch.closed for branch in service.lane.branches)


@pytest.mark.parametrize("field,value", [("accepted_token_sha256", "wrong"),
                                        ("unconsumed_token_ids", []), ("consumed_tokens", 0),
                                        ("consumed_tokens", True),
                                        ("matched_stop", "not-requested")])
def test_generation_receipt_corruption_cannot_pass(service, field, value):
    generate = service.generate

    def corrupt(*args, **kwargs):
        response = generate(*args, **kwargs)
        response[field] = value
        return response

    service.generate = corrupt
    report = run(service)
    assert report["status"] == "failed" and report["cleanup_errors"] == []


@pytest.mark.parametrize("corruption", ["nan", "probability", "order", "selected", "head",
                                        "forwarded", "replayed", "layout", "query_hash", "prefix", "boolean_counter"])
def test_response_validation_catches_bad_receipts(service, corruption):
    decide = service.decide

    def corrupt(*args, **kwargs):
        response = decide(*args, **kwargs)
        if corruption == "nan":
            response["scores"][0]["logit"] = float("nan")
        elif corruption == "probability":
            response["scores"][0]["raw_probability"] = 2.
        elif corruption == "order":
            response["scores"].reverse()
        elif corruption == "selected":
            response["selected_id"] = "not-a-candidate"
        elif corruption == "head":
            response["head_manifest_sha256"] = "changed"
        elif corruption == "forwarded":
            response["execution"]["forwarded_tokens"] += 1
        elif corruption == "replayed":
            response["execution"]["replayed_tokens"] = 1
        elif corruption == "layout":
            response["layout"] = "standalone-training-encoding"
        elif corruption == "query_hash":
            response["query_token_sha256"] = "wrong"
        elif corruption == "boolean_counter":
            response["execution"]["replayed_tokens"] = False
        else:
            response["prefix_id"] = "wrong"
        return response

    service.decide = corrupt
    report = run(service)
    assert report["status"] == "failed" and report["cleanup_errors"] == []


def test_numerical_comparison_rejects_shape_and_infinity():
    config = qualification.QualificationConfig(cases=[case()])
    for other in (torch.ones(2), torch.tensor([float("inf")]), torch.tensor([1.01])):
        with pytest.raises(ValueError):
            qualification._compare(torch.ones(1), other, config, "bad")
    assert qualification._compare(torch.ones(1), torch.tensor([1.00001]), config, "tight")["elements"] == 1


def test_tolerances_do_not_hide_a_selected_candidate_change():
    config = qualification.QualificationConfig(cases=[case()])
    left = {"query_features": torch.ones(1), "candidate_features": torch.ones(2, 1),
            "scores": torch.tensor([[1., 1.00001], [.5, .5], [.5, .5]])}
    right = {**left, "scores": left["scores"].flip(1)}
    with pytest.raises(ValueError, match="Selected candidate changed"):
        qualification._agree(left, right, config)


@pytest.mark.parametrize("corruption", [None, "directory", "revision", "cache", "head"])
def test_metadata_checks_actual_objects_not_just_supplied_labels(service, monkeypatch, corruption):
    from fuzev_research import native, paths, serving_head

    # Unit-test stand-ins only; never call the runner with a native evidence label.
    sources = {"recipes": "unit-recipes-commit", "exllamav3-fuzev": "unit-native-commit"}
    model = {"repo": "unit", "revision": "unit-revision", "local_snapshot": "/not-opened/model"}
    actual = {"model": model["repo"], "revision": model["revision"], "cell": "unit",
              "readout_runtime": deepcopy(service.backend.readout_runtime), "cache_precision": "float16"}
    service.backend.config.directory = model["local_snapshot"]
    service.backend.config.infer_params = SimpleNamespace(ngram_stream_from_disk=True)
    service.head.calibration_id = "sha256:unit-calibration"
    service.provenance.update(service_sources=deepcopy(sources), serving_identity=actual,
                              head_manifest_sha256=service.head.manifest_sha256,
                              calibration_sha256="unit-calibration")
    monkeypatch.setattr(native, "NativeBackend", Backend)
    monkeypatch.setattr(serving_head, "ServingHead", FrozenFakeHead)
    monkeypatch.setattr(native, "runtime_sources", lambda: sources)
    opened = []

    def metadata_only(path, limit):
        assert path == paths.project_path("data/manifests/acquisition.json") and limit == 1 << 20
        opened.append(path)
        return json.dumps({"models": {"unit": model}}).encode()

    monkeypatch.setattr(serving_head, "checked_bytes", metadata_only)
    if corruption == "directory":
        service.backend.config.directory = "/not-opened/other-model"
    elif corruption == "revision":
        actual["revision"] = "stale"
    elif corruption == "cache":
        service.backend.cache.max_num_tokens -= 256
    elif corruption == "head":
        service.provenance["head_manifest_sha256"] = "stale"
    if corruption:
        with pytest.raises(ValueError):
            qualification._metadata(service, cpu_fake=False)
    else:
        metadata = qualification._metadata(service, cpu_fake=False)
        assert metadata["runtime_sources"] == sources and metadata["model"] == model
        assert metadata["weight_bytes_verified"] is False
        assert metadata["cache"]["storage_dtype_verified"] is False
        assert len(opened) == 1
    assert service.lane.calls == []


def test_metadata_drift_and_frozen_head_mutation_fail_closed(service):
    original = service.head.score
    once = []

    def mutate(*args):
        if not once:
            with torch.no_grad():
                service.head.bundle.weight.add_(1.)
            service.provenance["changed"] = True
            once.append(True)
        return original(*args)

    service.head.score = mutate
    report = run(service)
    assert report["status"] == "failed"
    assert any("Frozen head changed" in error for error in report["cleanup_errors"])
    assert service.head.score is mutate


def test_provenance_change_alone_is_detected(service):
    original = service.head.score

    def mutate(*args):
        service.provenance["drift"] = True
        return original(*args)

    service.head.score = mutate
    report = run(service)
    assert report["status"] == "failed"
    assert any("provenance changed" in error for error in report["cleanup_errors"])


def test_running_head_is_not_a_frozen_head(service):
    service.head.bundle.train()
    report = run(service)
    assert report["status"] == "failed" and "frozen CPU FP32" in report["error"]
    assert service.lane.calls == []


def test_admission_failure_is_not_numerical_success(service, monkeypatch):
    def reject(*_):
        raise MemoryError("test shared-memory headroom")

    monkeypatch.setattr(qualification, "admit", reject)
    report = run(service)
    assert report["status"] == "memory_infeasible" and report["execution_status"] == "failed"
    assert service.lane.calls == []


@pytest.mark.parametrize("enabled,ready,available,status", [(False, True, 1000, "not_requested"),
    (True, False, 1000, "blocked"), (True, True, 99, "memory_infeasible"),
    (True, True, 251, "estimate_feasible")])
def test_full_window_preflight_is_only_arithmetic(enabled, ready, available, status):
    plan = qualification.FullWindowPlan(case=case("full"), additional_peak_bytes=100,
                                         estimate_basis="CPU unit-test integer budget, not a native estimate")
    result = qualification.full_window_feasibility(plan, enabled=enabled, frozen_head_ready=ready,
                                                  total=1000, available=available)
    assert result["status"] == status and result["allocated"] is False
    assert result["window_tokens"] == 262144


@pytest.mark.parametrize("memory", [{"total": 0, "available": 0}, {"total": 100, "available": 101},
                                     {"total": 100, "available": -1}])
def test_bad_full_window_budgets_are_rejected(memory):
    with pytest.raises(ValueError):
        qualification.full_window_feasibility(None, enabled=False, frozen_head_ready=False, **memory)


def test_cpu_fake_cannot_mark_full_window_ready_or_allocate(service):
    raw = {"cases": [case()], "run_full_window": True,
           "full_window": {"case": case("full"), "additional_peak_bytes": 1,
                           "estimate_basis": "deliberately false CPU budget, must never run"}}
    report = run(service, raw)
    assert_passed(report)
    assert report["gates"]["full_window"]["status"] == "blocked"
    assert report["gates"]["full_window"]["allocated"] is False
    assert len(report["cases"]) == 1


def test_short_case_cannot_be_called_full_window(service):
    config = qualification.QualificationConfig(cases=[case()])
    with pytest.raises(ValueError, match="exactly 262144"):
        qualification._case(service, config.cases[0], config, full_window=True)


def test_image_control_validation_does_not_open_files():
    from test_native_service import decision

    image = {"path": "/not-opened/a.png", "sha256": "a" * 64}
    raw = {"name": "image", "input_origin": "CPU fake", "input_role": "cpu-fake",
           "request": decision(image=image).model_dump(), "mismatched_image": image, "expected_id": "first"}
    with pytest.raises(ValidationError, match="different RGB digest"):
        qualification.ImageControl.model_validate(raw)
    raw["mismatched_image"] = {"path": "/not-opened/b.png", "sha256": "b" * 64}
    assert qualification.ImageControl.model_validate(raw).request.image.sha256 == "a" * 64


@pytest.mark.parametrize("mrope", [False, True])
def test_image_binding_replay_and_real_blind_mismatched_contracts(tmp_path, monkeypatch, mrope):
    from fuzev_research import native_service
    from test_service_images import Embedding, ImageBackend, acquired

    class FakePixels(ImageBackend):
        def vision_encode(self, _tokenizer, image):
            # Stable fake embeddings, not a native vision model or an image-quality fixture.
            return Embedding(1_000_000_000 + (100 if image.getpixel((0, 0))[2] else 0))

    backend = FakePixels(mrope=mrope)
    backend.cache = SimpleNamespace(free_list=list(range(8)), num_slots=8, max_num_tokens=294912)
    backend.lane = LedgerLane(backend.cache, batch_capacity=8)
    backend.lane.chunk_size = 6
    backend.readout_runtime.update(context_tokens=262144, feature_chunk_tokens=6, cache_tokens=294912)
    monkeypatch.setattr(native_service, "project_path", lambda name: tmp_path / name)
    red, blue = acquired(tmp_path / "data/images"), acquired(tmp_path / "data/images", "blue")
    svc = NativeService(backend, "unit", FrozenFakeHead(), {"synthetic": True}, Guard(), admission=lambda *_: None)
    raw = case(image=red.model_dump())
    for spec in raw["decisions"]:
        spec["image"] = blue.model_dump()
    control = {"name": "pixels", "input_origin": "tiny CPU fake pixels; no dataset",
               "input_role": "cpu-fake", "expected_id": "a", "mismatched_image": blue.model_dump(),
               "request": {"model": "unit", "messages": [{"role": "user", "content": "CPU fake"}],
                           **raw["decisions"][0], "image": red.model_dump()}}
    try:
        report = run(svc, {"cases": [raw], "image_controls": [control]})
        assert_passed(report)
        assert report["gates"]["real_image_controls"]["status"] == "cpu_fake"
        result = report["image_controls"][0]
        assert result["quality_qualified"] is False
        assert result["controls"]["blind"]["images"] == []
        assert result["controls"]["real"]["images"][0]["rgb_sha256"] == red.sha256
        assert result["controls"]["mismatched"]["images"][0]["rgb_sha256"] == blue.sha256
        assert backend.positions if mrope else not backend.positions
        assert not backend.images and not svc.prefixes
        assert len(backend.cache.free_list) == 8 and len(backend.lane.pool.free) == 64
    finally:
        svc.close()


def test_native_label_requires_real_running_guard(service):
    raw = case()
    raw["input_role"] = "operator-non-heldout"
    with pytest.raises(ValueError, match="running shared-memory"):
        qualification.run_qualification(service, {"cases": [raw]}, allow_inference=True)
    assert service.lane.calls == []


def test_missing_resource_ledger_is_visible(service):
    del service.lane.pool.free
    report = run(service)
    assert report["status"] == "failed" and "AttributeError" in report["error"]
    assert service.lane.calls == []
    service.lane.pool.free = set(range(64))


def test_case_count_and_uniqueness_are_bounded():
    raw = case()
    with pytest.raises(ValidationError):
        qualification.QualificationConfig(cases=[raw, deepcopy(raw)])
    raw["decisions"].pop()
    with pytest.raises(ValidationError):
        qualification.QualificationConfig(cases=[raw])
