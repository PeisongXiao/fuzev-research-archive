"""Tiny CPU state/lifetime fixtures, not native inference or qualification evidence."""

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from fuzev_research import native_qualification as qualification
from fuzev_research.native_service import NativeService
from fuzev_research.native_state_observer import NativeStateObserver
from test_native_qualification import FrozenFakeHead, case
from test_native_service import Backend, Guard, Lane
import test_native_state_observer as fixtures


observer_rig = fixtures.rig


class StateLane(Lane):
    """CPU-only implementation of the fixture's page/slot ownership contract."""
    context_limit = 262144

    def __init__(self, rig):
        super().__init__()
        self.rig, self.cache, self.model = rig, rig.cache, rig.model
        self.pool, self.lock, self.branches = rig.lane.pool, rig.lane.lock, set()
        self.created, self.released = [], []

    def create(self):
        branch, state = fixtures.Branch(), fixtures.State()
        state.cache, state.slot, state.position, state.last_history, state.exported = self.cache, self.cache.free_list.pop(), 0, 0, False
        state.checkpoint_size = sum(layer.get_checkpoint_size() for layer in self.cache.recurrent_layers.values())
        for layer in self.cache.recurrent_layers.values():
            for tensor in layer.get_state_tensors():
                tensor[state.slot].zero_()
            if isinstance(layer, fixtures.PLELayer):
                layer.id_state[state.slot].fill_(layer.module.ple_embedding.eos_token_id)
        branch.owner, branch.closed, branch.immutable = self, False, False
        branch.tokens, branch.pages, branch.recurrent, branch.checkpoint = (), [], state, None
        self.branches.add(branch)
        self.created.append(branch)
        return branch

    def fork(self, parent):
        assert not parent.closed
        branch = self.create()
        sources = ({key: tuple(item[0] for item in items) for key, items in self.rig.transfer.layout(parent.recurrent)[0]}
                   if parent.recurrent is not None else parent.checkpoint)
        for key, items in self.rig.transfer.layout(branch.recurrent)[0]:
            for (target, _, _), source in zip(items, sources[key]):
                target.copy_(source)
        branch.tokens, branch.pages, branch.recurrent.position = parent.tokens, list(parent.pages), len(parent.tokens)
        for page in branch.pages:
            self.pool.references[page] += 1
        return branch

    def freeze(self, branch):
        self.rig.freeze(branch)
        return branch

    def release(self, branch):
        assert not branch.closed
        for page in branch.pages:
            self.pool.references[page] -= 1
            if not self.pool.references[page]:
                self.pool.free.append(page)
        if branch.recurrent is not None:
            self.cache.free_list.append(branch.recurrent.slot)
        branch.closed, branch.checkpoint, branch.recurrent = True, None, None
        self.branches.remove(branch)
        self.released.append(branch)

    def _page(self):
        if not self.pool.free:
            raise MemoryError("CPU fake page pool exhausted")
        page = self.pool.free.pop(0)
        self.pool.references[page] = 1
        return page

    def _prepare(self, branch, count):
        assert not branch.immutable and not branch.closed
        start = len(branch.tokens)
        if start % 256 and self.pool.references[branch.pages[-1]] > 1:
            old, new = branch.pages[-1], self._page()
            for layer in self.cache.layers.values():
                for tensor in layer.get_tensors():
                    tensor[new].copy_(tensor[old])
            branch.pages[-1] = new
            self.pool.references[old] -= 1
        while len(branch.pages) < (start + count + 255) // 256:
            branch.pages.append(self._page())

    def _write(self, branch, tokens):
        start = len(branch.tokens)
        for offset, token in enumerate(tokens, start):
            page, row = branch.pages[offset // 256], offset % 256
            for layer in self.cache.layers.values():
                layer.k[page, row] = layer.v[page, row] = ((token + offset) % 64) / 64
                if isinstance(layer, fixtures.QSA):
                    layer.raw_k[page, row] = ((token + offset) % 32) / 32
                    if (row + 1) % 4 == 0:
                        layer.pooled[page, row // 4] = layer.raw_k[page, row - 3:row + 1].mean(0)
        branch.tokens += tuple(tokens)
        state = branch.recurrent
        state.position = len(branch.tokens)
        for _, items in self.rig.transfer.layout(state)[0]:
            for tensor, _, _ in items:
                tensor.fill_((sum(branch.tokens) % 16) / 16 if tensor.is_floating_point() else branch.tokens[-1])

    def _forward(self, branches, chunks, params, features, last):
        for branch, chunk in zip(branches, chunks):
            self._prepare(branch, len(chunk))
        page_width = max(len(branch.pages) for branch in branches)
        arguments = {**(params or {}), "attn_mode": "flash_attn", "cache": self.cache, "last_tokens_only": 1,
                     "block_table": torch.tensor([b.pages + [-1] * (page_width - len(b.pages)) for b in branches], dtype=torch.int32),
                     "cache_seqlens": torch.tensor([len(b.tokens) for b in branches], dtype=torch.int32),
                     "recurrent_states": [b.recurrent for b in branches]}
        if last and features:
            arguments["_fuzev_features"] = True
        method = self.model.forward if last else self.model.prefill
        method(torch.tensor(chunks, dtype=torch.long), arguments)
        for branch, chunk in zip(branches, chunks):
            self._write(branch, chunk)

    def append(self, branch, tokens, *, features=False, params=None):
        if not branch.tokens:
            self.output = [91, 92, 99]
        self.calls.append((branch.tokens, tuple(tokens), features))
        self.params.append(params)
        for offset in range(0, len(tokens), self.chunk_size):
            chunk = tokens[offset:offset + self.chunk_size]
            self._forward([branch], [chunk], params, features, offset + len(chunk) == len(tokens))
        self.after_append()
        if features:
            self.scratch.fill_(sum(branch.tokens))
            return self.scratch[:1]
        logits = torch.full((1, 1, 512), -torch.inf)
        logits[0, 0, self.output.pop(0)] = 1
        return logits

    def append_features_many(self, branches, tails, *, params=None):
        self.many.append((tuple(branch.tokens for branch in branches), tails))
        for offset in range(0, len(tails[0]), self.chunk_size):
            chunks = [tail[offset:offset + self.chunk_size] for tail in tails]
            self._forward(branches, chunks, params, True, offset + len(chunks[0]) == len(tails[0]))
        for index, branch in enumerate(branches):
            self.scratch[index].fill_(sum(branch.tokens))
        self.after_append()
        return self.scratch[:len(branches)]


@pytest.fixture
def service(observer_rig, monkeypatch):
    def fake_model(ids, params):
        # CPU contract fixture only; no kernels or real forward implementation are imported.
        params["recurrent_slots"] = torch.tensor([s.slot for s in params["recurrent_states"]], dtype=torch.int32)
        for state in params["recurrent_states"]:
            state.position += ids.shape[1]
            state.last_history = ids.shape[1] - 1 if params.get("recurrent_history", False) else 0

    monkeypatch.setattr(observer_rig.model, "forward", fake_model, raising=False)
    monkeypatch.setattr(observer_rig.model, "prefill", fake_model, raising=False)
    backend = Backend()
    backend.cache, backend.lane = observer_rig.cache, StateLane(observer_rig)
    backend.readout_runtime.update(context_tokens=262144, feature_chunk_tokens=8192, cache_tokens=4096)
    value = NativeService(backend, "unit", FrozenFakeHead(), {"synthetic": True}, Guard(), admission=lambda *_: None)
    value.observers = []

    def factory(service, config, cpu_fake):
        assert cpu_fake is True
        observer = NativeStateObserver(service, allow_observation=True, cpu_fake=True,
            atol=config.atol, rtol=config.rtol, _test_bindings=observer_rig.types)
        value.observers.append(observer)
        return observer

    monkeypatch.setattr(qualification, "_state_observer", factory)
    yield value
    assert not value.lane.branches and not value.prefixes
    assert len(value.backend.cache.free_list) == 4 and len(value.lane.pool.free) == 16
    assert all(branch.closed for branch in value.lane.created)
    value.close()


def run(service, *, config=None):
    return qualification.run_qualification(service, config or {"cases": [case()], "observe_native_state": True},
                                           allow_inference=True, cpu_fake=True)


def test_actual_observer_integrates_without_closing_real_gates(service):
    report = run(service)
    assert report["execution_status"] == "passed", report
    assert report["qualified"] is False and report["status"] == "incomplete"
    assert report["evidence_kind"] == "cpu_fake" and report["cleanup_errors"] == []
    gate = report["gates"]["complete_hybrid_state"]
    assert gate["status"] == "incomplete" and gate["state_observations"] == "cpu_fake_observed"
    assert report["gates"]["forward_arguments"]["status"] == "incomplete"
    flags = report["gates"]["non_speculative_forward_flags"]
    assert flags["status"] == "incomplete" and flags["flag_counts"]["absent"] > 0
    assert flags["completed_calls"]["prefill"] == 0  # Ordinary service chunking did not exercise this path.
    assert flags["source_default_verified"] is False
    for branch in report["cases"][0]["branches"]:
        assert branch["oracle_status"]["status"] == "observed"
        comparisons = branch["state_observations"]["comparisons"]
        assert [row["boundary"] for row in comparisons] == ["consumed_AB", "pending_B", "ABC", "candidate", "candidate"]
        assert branch["state_observations"]["reset"]["components"]["-4:0/id_state"]["reset_value"] == 99
        assert all(row["receipt"]["evidence_kind"] == "cpu_fake" for row in comparisons)
        assert len(comparisons[-1]["receipt"]["components"]) == 10
        assert [row["boundary"] for row in branch["reused_state_boundaries"]] == ["pending_B", "ABC"]
    assert any(not before and tokens == (66, 91, 92) for before, tokens, _ in service.lane.calls)
    assert all(name not in vars(service.lane) for name in ("freeze", "fork", "release"))
    assert "_append" not in vars(service) and "encode" not in vars(service.backend)


def test_default_partial_mode_never_constructs_observer(service, monkeypatch):
    monkeypatch.setattr(qualification, "_state_observer", lambda *_args: pytest.fail("Opt-in required"))
    report = run(service, config={"cases": [case()]})
    assert report["execution_status"] == "passed", report
    assert report["gates"]["complete_hybrid_state"]["status"] == "unavailable"


@pytest.mark.parametrize("value", [1, "true", None])
def test_state_opt_in_is_strict(service, value):
    with pytest.raises(ValueError):
        run(service, config={"cases": [case()], "observe_native_state": value})
    assert service.lane.calls == []


@pytest.mark.parametrize("where", ["constructor", "inventory"])
def test_unsupported_state_remains_incomplete_without_oracle(service, monkeypatch, where):
    if where == "constructor":
        def unavailable(*_args):
            raise NotImplementedError("CPU fake unsupported binding")
        monkeypatch.setattr(qualification, "_state_observer", unavailable)
    else:
        service.backend.cache.recurrent_layers[(-4, 0)].unreviewed = torch.zeros(1)
    report = run(service)
    assert report["execution_status"] == "incomplete", report
    assert report["status"] == "incomplete" and report["qualified"] is False
    assert report["gates"]["complete_hybrid_state"]["state_observations"] == "incomplete"
    assert report["cleanup_errors"] == []
    assert not any(not before and len(tokens) > 1 for before, tokens, _ in service.lane.calls)


@pytest.mark.parametrize("scope", ["serving", "observer_coexistence"])
def test_capacity_scopes_and_preflight_before_any_oracle_forward(service, monkeypatch, scope):
    original = NativeStateObserver.preflight
    calls = []

    def limited(observer, *args, **kwargs):
        assert len(service.lane.branches) == 1  # Only retained AB, no query/extra observer lifetimes yet.
        calls.append(len(service.lane.calls))
        report = original(observer, *args, **kwargs)
        report[scope] = {"status": "memory_infeasible", "reason": "CPU fake capacity boundary"}
        return report

    monkeypatch.setattr(NativeStateObserver, "preflight", limited)
    report = run(service)
    assert calls and report["execution_status"] == "incomplete", report
    assert report["gates"]["full_window"]["status"] == "not_requested"
    assert report["cleanup_errors"] == []
    assert not any(not before and len(tokens) > 1 for before, tokens, _ in service.lane.calls)
    for branch in report["cases"][0]["branches"]:
        assert branch["state_preflight"][scope]["status"] == "memory_infeasible"
        if scope == "observer_coexistence":
            assert branch["serving_status"] == "observed"
            assert branch["oracle_status"]["scope"] == "observer_oracle_not_serving"


@pytest.mark.parametrize("error", [RuntimeError, InterruptedError, MemoryError])
def test_oracle_fault_releases_all_retained_state(service, monkeypatch, error):
    original = service._append
    triggered = []

    def fail_fresh(branch, tokens, *args, **kwargs):
        if not branch.tokens and len(tokens) > 1:
            triggered.append(True)
            assert sum(b.immutable for b in service.lane.branches) == 5  # AB, pending B, reused ABC, two candidates.
            raise error("CPU fake oracle fault")
        return original(branch, tokens, *args, **kwargs)

    monkeypatch.setattr(service, "_append", fail_fresh)
    report = run(service)
    assert triggered and report["cleanup_errors"] == [], report
    if error is MemoryError:
        assert report["execution_status"] == "incomplete"
        assert all(row["oracle_status"]["status"] == "memory_infeasible" for row in report["cases"][0]["branches"])
        assert report["gates"]["full_window"]["status"] == "not_requested"
    else:
        assert report["status"] == ("cancelled" if error is InterruptedError else "failed")


@pytest.mark.parametrize("phase", ["reused_candidate", "reused_head", "fresh_candidate", "compare"])
def test_failure_after_retention_unwinds_every_additional_branch(service, monkeypatch, phase):
    triggered = []
    if phase == "compare":
        original = NativeStateObserver.compare
        def compare(observer, left, right):
            if len(right.tokens) > 3:
                triggered.append(True)
                raise ValueError("CPU fake state mismatch")
            return original(observer, left, right)
        monkeypatch.setattr(NativeStateObserver, "compare", compare)
    elif phase == "reused_head":
        def score(*_args):
            triggered.append(True)
            raise RuntimeError("CPU fake head fault")
        monkeypatch.setattr(service.head, "score", score)
    elif phase == "reused_candidate":
        original = service.lane.append_features_many
        def many(*args, **kwargs):
            original(*args, **kwargs)
            triggered.append(True)
            raise InterruptedError("CPU fake candidate cancellation")
        monkeypatch.setattr(service.lane, "append_features_many", many)
    else:
        original = service._append
        def append(branch, tokens, *args, **kwargs):
            before = len(branch.tokens)
            result = original(branch, tokens, *args, **kwargs)
            if len(service.lane.branches) >= 6 and before > 4:
                triggered.append(True)
                raise InterruptedError("CPU fake fresh candidate cancellation")
            return result
        monkeypatch.setattr(service, "_append", append)
    report = run(service)
    assert triggered and report["status"] in {"failed", "cancelled"}, report
    assert report["cleanup_errors"] == []


def test_parent_mutation_during_candidate_forward_is_not_a_pass(service, monkeypatch):
    original = service.lane.append_features_many

    def corrupt(*args, **kwargs):
        value = original(*args, **kwargs)
        parent = next(iter(service.prefixes.values()))[0].branch
        parent.checkpoint[(-4, 0)][1][0] += 1
        return value

    monkeypatch.setattr(service.lane, "append_features_many", corrupt)
    report = run(service)
    assert report["status"] == "failed", report
    assert report["cleanup_errors"] == []


@pytest.mark.parametrize("fail_at", [None, 2, 3, 4])
def test_every_oracle_reservation_protects_ab_including_late_capacity_failure(service, monkeypatch, fail_at):
    fresh, reserve, backend_reserve = qualification._fresh, service._reserve, service.backend.reserve
    active, calls, failures, completed = False, 0, [], []

    def replay(*args, **kwargs):
        nonlocal active, calls
        active, calls = True, 0
        try:
            return fresh(*args, **kwargs)
        finally:
            active = False
            completed.append(calls)

    def checked(tokens, protected=None):
        nonlocal calls
        prefix = next(iter(service.prefixes.values()))[0] if active else None
        if active:
            calls += 1
            assert protected == prefix.id
        try:
            return reserve(tokens, protected)  # Exercise NativeService's actual eviction loop.
        except MemoryError:
            assert active and prefix.id in service.prefixes and not prefix.branch.closed
            failures.append(calls)
            raise

    def exhaust(tokens):
        if active and calls == fail_at:
            raise MemoryError("CPU fake late oracle reservation failure")
        return backend_reserve(tokens)

    monkeypatch.setattr(qualification, "_fresh", replay)
    monkeypatch.setattr(service, "_reserve", checked)
    monkeypatch.setattr(service.backend, "reserve", exhaust)
    report = run(service)
    assert report["cleanup_errors"] == [], report
    assert completed == [fail_at or 5] * 4
    assert failures == ([] if fail_at is None else [fail_at] * 4)
    assert report["execution_status"] == ("passed" if fail_at is None else "incomplete"), report
    assert all(row["serving_status"] == "observed" for row in report["cases"][0]["branches"])
    assert report["gates"]["full_window"]["status"] == "not_requested"


@pytest.mark.parametrize("phase", ["frozen", "preflight", "compare"])
def test_unsupported_or_unwitnessed_state_stays_incomplete(service, monkeypatch, phase):
    if phase == "frozen":
        original = service.lane.freeze
        def freeze(branch):
            original(branch)
            branch.checkpoint.pop("_coalesced_slab")
        monkeypatch.setattr(service.lane, "freeze", freeze)
    else:
        def unavailable(*_args, **_kwargs):
            raise NotImplementedError("CPU fake unimplemented observation")
        monkeypatch.setattr(NativeStateObserver, phase, unavailable)
    report = run(service)
    assert report["status"] == report["execution_status"] == "incomplete", report
    assert report["gates"]["complete_hybrid_state"]["state_observations"] == "incomplete"
    assert report["cleanup_errors"] == []


def test_candidate_comparisons_restore_original_order_across_length_groups(service):
    raw = case()
    for decision in raw["decisions"]:
        decision["candidates"] = [{"id": "a", "text": "long"}, {"id": "b", "text": "x"},
                                  {"id": "c", "text": "wide"}]
    report = run(service, config={"cases": [raw], "observe_native_state": True})
    assert report["execution_status"] == "passed", report
    for branch in report["cases"][0]["branches"]:
        comparisons = branch["state_observations"]["comparisons"]
        assert [row["index"] for row in comparisons if row["boundary"] == "candidate"] == [0, 1, 2]


@pytest.mark.parametrize("flag", [True, 1, None, False])
def test_actual_forward_history_flag_not_last_history_zero(service, monkeypatch, flag):
    parameters = service.backend.parameters
    forward = service.lane.model.forward
    monkeypatch.setattr(service.backend, "parameters",
                        lambda *args: {**parameters(*args), "recurrent_history": flag})
    report = run(service)
    receipt = report["gates"]["non_speculative_forward_flags"]
    assert receipt["source_default_verified"] is False and report["qualified"] is False
    assert service.lane.model.forward is forward and report["cleanup_errors"] == []
    if flag is False:
        assert receipt["status"] == "incomplete" and receipt["flag_counts"]["explicit_false"] > 0
    else:
        assert receipt["status"] == report["execution_status"] == "incomplete", report
        assert receipt["flag_counts"] == {"absent": 0, "explicit_false": 0, "rejected": 1}
        assert len(service.lane.calls) == 1  # First one-token plan recorded, but model entry refused before computation.


@pytest.fixture
def chunk_witness(service):
    service.lane.chunk_size = 1  # Tiny CPU fixture that actually traverses prefill, never a native window downgrade.
    service.backend.readout_runtime["feature_chunk_tokens"] = service.readout_runtime["feature_chunk_tokens"] = 1
    config = qualification.QualificationConfig(cases=[case()], observe_native_state=True)
    return qualification._StateObservation(service, config, True)


@pytest.mark.parametrize("batched", [False, True])
def test_forward_and_prefill_bound_to_exact_lane_chunks(service, chunk_witness, batched):
    branches = [service.lane.create() for _ in range(2 if batched else 1)]
    originals = {name: getattr(service.lane.model, name) for name in ("forward", "prefill")}
    try:
        with chunk_witness.freezes():
            params = service.backend.parameters((), [], (10, 11, 12))
            if batched:
                service.lane.append_features_many(branches, [(10, 11, 12), (13, 14, 15)], params=params)
            else:
                service.lane.append(branches[0], (10, 11, 12), features=True, params=params)
        assert chunk_witness.forward_calls == {"forward": 1, "prefill": 2}
        assert chunk_witness.forward_flags == {"absent": 3, "explicit_false": 0, "rejected": 0}
        assert all(len(b.tokens) == b.recurrent.position == 3 for b in branches)
        assert chunk_witness.active is chunk_witness.parameter_plan is None
        assert all(getattr(service.lane.model, name) is original for name, original in originals.items())
    finally:
        qualification._release_observed(service, branches)


@pytest.mark.parametrize("entry", ["forward", "prefill"])
@pytest.mark.parametrize("field", ["ids", "block_table", "cache_seqlens", "state", "slot", "position",
                                   "images", "history", "dtype", "unknown_position", "writable_alias"])
def test_wrong_actual_entry_metadata_refused_before_fake_model(service, chunk_witness, monkeypatch, entry, field):
    model, lane = service.lane.model, service.lane
    original, called = getattr(model, entry), []

    def compute(*args):
        called.append(True)
        return original(*args)

    monkeypatch.setattr(model, entry, compute)
    branch = lane.create()
    try:
        with chunk_witness.freezes():
            witness = getattr(model, entry)

            def corrupt(ids, params):
                slot, position = branch.recurrent.slot, branch.recurrent.position
                page, references = branch.pages[-1], lane.pool.references[branch.pages[-1]]
                try:
                    if field == "ids":
                        ids[0, 0] += 1
                    elif field in {"block_table", "cache_seqlens"}:
                        params[field].view(-1)[0] += 1
                    elif field == "state":
                        params["recurrent_states"] = [SimpleNamespace()]
                    elif field == "slot":
                        branch.recurrent.slot = service.backend.cache.free_list[0]
                    elif field == "position":
                        branch.recurrent.position += 1
                    elif field == "images":
                        params["indexed_embeddings"] = [object()]
                    elif field == "history":
                        params["recurrent_history"] = True
                    elif field == "dtype":
                        ids = ids.float()
                    elif field == "writable_alias":
                        lane.pool.references[page] += 1
                    else:
                        params["past_len"] = 0
                    return witness(ids, params)
                finally:
                    branch.recurrent.slot, branch.recurrent.position = slot, position
                    lane.pool.references[page] = references

            with qualification._replace(model, entry, corrupt):
                tokens = (10,) if entry == "forward" else (10, 11)
                params = service.backend.parameters((), [], tokens)
                with pytest.raises((ValueError, qualification._StateUnavailable)):
                    lane.append(branch, tokens, features=True, params=params)
        assert not called and not branch.tokens
        assert chunk_witness.forward_failed and chunk_witness.forward_calls == {"forward": 0, "prefill": 0}
        assert chunk_witness.active is chunk_witness.parameter_plan is None
        assert getattr(model, entry) is compute
    finally:
        service._release(branch)


def test_unplanned_model_entry_never_becomes_argument_evidence(service, chunk_witness):
    with chunk_witness.freezes(), pytest.raises(qualification._StateUnavailable, match="no active"):
        service.lane.model.prefill(torch.tensor([[10]]), {})
    assert chunk_witness.forward_calls == {"forward": 0, "prefill": 0}


@pytest.mark.parametrize("replace_binding", [False, True])
def test_image_object_identity_not_just_matching_image_metadata(service, chunk_witness, monkeypatch, replace_binding):
    embedding = SimpleNamespace(first_index=10**9, last_index=10**9 + 2, text_alias="CPU-fake-image",
                                token_list=(1, 10**9, 10**9 + 1, 2), grid_thw=(1, 1, 2), mrope_merge_size=1)
    monkeypatch.setattr(service.backend, "parameters", lambda _base, images, _suffix: {"indexed_embeddings": list(images)})
    branch = service.lane.create()
    try:
        with chunk_witness.freezes():
            params = service.backend.parameters((), [embedding], (10,))
            if replace_binding:
                params["indexed_embeddings"] = [SimpleNamespace(**vars(embedding))]
                with pytest.raises(ValueError, match="image bindings"):
                    service.lane.append(branch, (10,), features=True, params=params)
            else:
                service.lane.append(branch, (10,), features=True, params=params)
                assert chunk_witness.forward_calls["forward"] == 1
    finally:
        service._release(branch)


def test_pending_boundary_uses_elementwise_compare_and_releases_early(service, monkeypatch):
    compare, append, pending, checked = NativeStateObserver.compare, service._append, [], []

    def observed(observer, left, right):
        result = compare(observer, left, right)
        if len(left.tokens) == 4:
            pending.append((left, right))
            assert result["components"]["-4:0/id_state"]["elements"] > 0
        return result

    def continued(branch, tokens, *args, **kwargs):
        for saved, fresh in pending:
            if branch is fresh and len(branch.tokens) == 4:
                assert saved.closed and saved not in service.lane.branches
                checked.append(True)
        return append(branch, tokens, *args, **kwargs)

    monkeypatch.setattr(NativeStateObserver, "compare", observed)
    monkeypatch.setattr(service, "_append", continued)
    report = run(service)
    assert report["execution_status"] == "passed" and len(pending) == len(checked) == 4, report


def test_pending_causal_difference_fails_elementwise_not_hash_only(service, monkeypatch):
    compare, checked = NativeStateObserver.compare, []

    def corrupt(observer, left, right):
        if len(left.tokens) == 4:
            service.backend.cache.recurrent_layers[(-4, 0)].id_state[right.recurrent.slot, 0] += 1
            checked.append(True)
        return compare(observer, left, right)

    monkeypatch.setattr(NativeStateObserver, "compare", corrupt)
    report = run(service)
    assert checked and report["status"] == "failed" and "integer disagreement" in report["error"], report
    assert report["cleanup_errors"] == []


def test_harness_checks_fresh_reset_before_any_oracle_forward(service, monkeypatch):
    create, corrupted = service.lane.create, []

    def stale_slot():
        branch = create()
        if len(service.lane.branches) == 6:
            service.backend.cache.recurrent_layers[(-4, 0)].id_state[branch.recurrent.slot, 0] = 0
            corrupted.append(branch)
        return branch

    monkeypatch.setattr(service.lane, "create", stale_slot)
    report = run(service)
    assert corrupted and report["status"] == "failed" and "reset disagreement" in report["error"], report
    assert report["cleanup_errors"] == []
    assert not any(not before and len(tokens) > 1 for before, tokens, _ in service.lane.calls)
