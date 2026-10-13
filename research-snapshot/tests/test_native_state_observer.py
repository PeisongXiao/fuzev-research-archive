"""Adversarial CPU contracts only. No native import/build/inference or model data."""

from contextlib import contextmanager
import json
import subprocess
import sys
from threading import RLock, get_ident
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.native_state_observer import NativeStateObserver, _tiles


class Attention:
    def __init__(self, index, qsa):
        self.layer_idx, self.caps = index, {"kv_cache": True}
        self.num_kv_heads, self.head_dim, self.cache_layers = 1, 2, []
        self.qsa_indexer = NS(compress_ratio=4, head_dim=2) if qsa else None


class GDN:
    def __init__(self):
        self.layer_idx, self.caps, self.recurrent_layers = 2, {"recurrent_cache": True}, []
        self.num_v_heads, self.k_head_dim, self.v_head_dim = 1, 2, 2
        self.fdim_qkv, self.conv_kernel_size = 4, 3


class PLE:
    def __init__(self):
        self.layer_idx, self.caps, self.recurrent_layers = -4, {"recurrent_cache": True}, []
        self.hc_mult, self.hidden_size, self.conv_state_len = 2, 2, 4
        self.conv_kernel_size, self.conv_dilation = 3, 2
        self.ple_embedding = NS(context_len=2, eos_token_id=99, _pending=[], _pins=[])


class Model:
    loaded_tp = False

    def __init__(self):
        self.modules = [Attention(0, True), Attention(1, False), GDN(), PLE()]
        self.instances = {module.layer_idx: [(module.layer_idx, 0)] for module in self.modules}

    def __iter__(self):
        return iter(self.modules)

    def get_layer_instances(self, index):
        return self.instances[index]

    def get_cache_layers(self):
        return [m for m in self if m.caps.get("kv_cache")]

    def get_recurrent_layers(self):
        return [m for m in self if m.caps.get("recurrent_cache")]


class FP16:
    def __init__(self, cache, module):
        self.attention, self.cache_id, self.max_num_tokens = module, id(cache), cache.max_num_tokens
        self.k = torch.zeros(cache.max_num_tokens // 256, 256, 1, 2, dtype=torch.float16)
        self.v = torch.zeros_like(self.k)

    def get_tensors(self):
        return [self.k, self.v]


class QSA(FP16):
    def __init__(self, cache, module):
        super().__init__(cache, module)
        self.compress_ratio, self.index_head_dim = 4, 2
        self.raw_k = torch.zeros(cache.max_num_tokens // 256, 256, 2, dtype=torch.float16)
        self.pooled = torch.zeros(cache.max_num_tokens // 256, 64, 2, dtype=torch.float16)

    def get_tensors(self):
        return super().get_tensors() + [self.raw_k, self.pooled]


class GDNLayer:
    def __init__(self, cache, module):
        self.module, self.cache_id = module, id(cache)
        self.max_history, self.max_batch_size = 3, cache.num_slots
        self.conv_state = torch.zeros(cache.num_slots, 4, 6, dtype=torch.bfloat16)
        self.recurrent_state = torch.zeros(cache.num_slots, 4, 1, 2, 2, dtype=torch.float32)

    def get_state_tensors(self):
        return self.conv_state, self.recurrent_state

    def get_checkpoint_size(self):
        return 3 * 4 * 2 + 1 * 2 * 2 * 4


class PLELayer:
    def __init__(self, cache, module):
        self.module, self.cache_id, self.win, self.ctx = module, id(cache), 4, 2
        self.max_history, self.max_batch_size = 3, cache.num_slots
        self.conv_state = torch.zeros(cache.num_slots, 4, 7, dtype=torch.float16)
        self.id_state = torch.zeros(cache.num_slots, 5, dtype=torch.int64)

    def get_state_tensors(self):
        return self.conv_state, self.id_state

    def get_checkpoint_size(self):
        return 4 * 4 * 2 + 2 * 8


class State:
    pass


class Branch:
    pass


class Transfer:
    """CPU fixture of the pinned layout contract, NOT actual engine evidence."""
    ARENA = None
    hooks = NS(CONFIG={"checkpoint_transfer": "coalesced"})

    def layout(self, state):
        groups, offset = [], 0
        for key, layer in state.cache.recurrent_layers.items():
            if type(layer) is GDNLayer:
                tensors = (layer.recurrent_state[state.slot, :1], layer.conv_state[state.slot, :, :3])
            else:
                tensors = (layer.conv_state[state.slot, :, :layer.win], layer.id_state[state.slot, :layer.ctx])
            items = []
            for tensor in tensors:
                size = tensor.numel() * tensor.element_size()
                items.append((tensor, offset, size))
                offset += size
            groups.append((key, items))
        return groups, offset


@pytest.fixture
def rig():
    model = Model()
    cache = NS(model=model, initialized=True, max_num_tokens=4096, num_slots=4, max_history=3,
               free_list=list(range(4)), layers={}, recurrent_layers={}, layer_type=FP16, recurrent_state_cls=State)
    for module in model:
        if type(module) is Attention:
            layer = (QSA if module.qsa_indexer else FP16)(cache, module)
            module.cache_layers.append(layer)
            cache.layers[(module.layer_idx, 0)] = layer
        else:
            layer = (GDNLayer if type(module) is GDN else PLELayer)(cache, module)
            module.recurrent_layers.append(layer)
            cache.recurrent_layers[(module.layer_idx, 0)] = layer
    cache.get_all_recurrent_layers = lambda: cache.recurrent_layers
    lane = NS(cache=cache, model=model, context_limit=262144, chunk_size=8192, branches=set(), lock=RLock(),
              pool=NS(references=[0] * 16, free=list(range(16))))
    runtime = {"memory_fraction": .85, "context_tokens": 262144, "cache_tokens": 4096,
               "max_hybrid_slots": 4, "feature_chunk_tokens": 8192}
    admissions, guard_calls = [], []
    owner = get_ident()

    def owned():
        assert get_ident() == owner

    service = NS(lane=lane, backend=NS(cache=cache, readout_runtime=runtime, candidate_batch_size=4),
                 guard=NS(check=lambda: guard_calls.append(True)), _owner=owned,
                 admission=lambda size, fraction: admissions.append((size, fraction)))
    transfer = Transfer()
    types = {"fp16": FP16, "qsa": QSA, "gdn": GDNLayer, "ple": PLELayer, "state": State,
             "attention": Attention, "gdn_module": GDN, "ple_module": PLE, "model": Model, "transfer": transfer}

    def branch(n=259):
        b, state = Branch(), State()
        state.cache, state.position, state.slot, state.last_history, state.exported = cache, n, cache.free_list.pop(), 0, False
        state.checkpoint_size = sum(layer.get_checkpoint_size() for layer in cache.recurrent_layers.values())
        b.owner, b.closed, b.immutable, b.recurrent, b.checkpoint = lane, False, False, state, None
        b.tokens, b.pages = tuple(range(n)), [lane.pool.free.pop(0) for _ in range((n + 255) // 256)]
        for page in b.pages:
            lane.pool.references[page] += 1
        lane.branches.add(b)
        return b

    def freeze(b):
        groups, size = transfer.layout(b.recurrent)
        slab = torch.empty(size, dtype=torch.uint8)
        saved = {"position": len(b.tokens), "checkpoint_size": size, "_coalesced_slab": slab}
        for key, items in groups:
            values = []
            for source, offset, length in items:
                target = slab[offset:offset + length].view(source.dtype).view(source.shape)
                target.copy_(source)
                values.append(target)
            saved[key] = tuple(values)
        cache.free_list.append(b.recurrent.slot)
        b.checkpoint, b.recurrent, b.immutable = saved, None, True

    def observer(**kwargs):
        return NativeStateObserver(service, allow_observation=True, cpu_fake=True, _test_bindings=types, **kwargs)

    return NS(model=model, cache=cache, lane=lane, service=service, types=types, transfer=transfer,
              branch=branch, freeze=freeze, observer=observer, admissions=admissions, guard_calls=guard_calls)


def test_disabled_import_needs_no_torch_or_native():
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = sys.modules['exllamav3'] = None
from fuzev_research.native_state_observer import NativeStateObserver
for value in (False, None, 1, 'yes'):
    try:
        NativeStateObserver(None, allow_observation=value)
    except ValueError as error:
        assert 'opt-in' in str(error)
    else:
        raise AssertionError('Missing opt-in')
"""], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kwargs", [{"atol": 1e-4}, {"rtol": 1e-3}, {"atol": float("nan")},
                                    {"rtol": 0}, {"tile_elements": True}, {"tile_elements": 16385}])
def test_declared_tolerances_and_bounds_cannot_relax(rig, kwargs):
    with pytest.raises(ValueError):
        rig.observer(**kwargs)


def test_actual_cpu_inventory_and_bounded_tiles_never_claim_native_proof(rig, monkeypatch):
    a, b, observer = rig.branch(), rig.branch(), rig.observer(tile_elements=3)
    native_cpu = torch.Tensor.cpu

    def bounded_cpu(tensor, *args, **kwargs):
        assert tensor.numel() <= 3
        return native_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", bounded_cpu)
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_a, **_k: pytest.fail("No clones needed"))
    report = observer.compare(a, b)
    assert report["evidence_kind"] == "cpu_fake" and report["qualified"] is False
    assert report["status"] == "observed_equal" and len(report["components"]) == 10
    assert report["components"]["0:0/raw_k"]["elements"] == 259 * 2
    assert report["components"]["0:0/pooled"]["elements"] == (259 // 4) * 2
    assert report["components"]["-4:0/id_state"]["dtype"] == "torch.int64"
    assert len(rig.admissions) > 10 and all(fraction == .85 and size == 384 for size, fraction in rig.admissions)
    assert observer.digest(a)["sha256"] == observer.digest(b)["sha256"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["k", "v", "raw_k", "pooled", "gdn_conv", "gdn_recurrent", "ple_conv", "ple_ids"])
def test_every_causal_component_is_compared(rig, field):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    if field in {"k", "v", "raw_k", "pooled"}:
        getattr(rig.cache.layers[(0, 0)], field)[b.pages[0], 0] = 1
    else:
        layer = rig.cache.recurrent_layers[(2, 0) if field.startswith("gdn") else (-4, 0)]
        name = {"gdn_conv": "conv_state", "gdn_recurrent": "recurrent_state",
                "ple_conv": "conv_state", "ple_ids": "id_state"}[field]
        getattr(layer, name)[b.recurrent.slot] = 1
    with pytest.raises(ValueError, match="disagreement"):
        observer.compare(a, b)
    assert a in rig.lane.branches and b in rig.lane.branches


def test_noncausal_rows_and_reserved_history_are_explicitly_ignored(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    layer = rig.cache.layers[(0, 0)]
    layer.pooled[b.pages[-1], 0] = float("nan")  # Only three raw tokens in the last page.
    layer.k[b.pages[-1], 3:] = float("nan")
    rig.cache.recurrent_layers[(2, 0)].conv_state[b.recurrent.slot, :, 3:] = float("nan")
    rig.cache.recurrent_layers[(2, 0)].recurrent_state[b.recurrent.slot, 1:] = float("nan")
    rig.cache.recurrent_layers[(-4, 0)].conv_state[b.recurrent.slot, :, 4:] = float("nan")
    rig.cache.recurrent_layers[(-4, 0)].id_state[b.recurrent.slot, 2:] = 999
    assert observer.compare(a, b)["ignored"]["qsa_partial_pool"]
    b.recurrent.last_history = 1
    with pytest.raises(ValueError, match="history"):
        observer.compare(a, b)


@pytest.mark.parametrize("change", ["missing_kv", "missing_ple", "unknown_module", "unknown_layer", "extra_tensor",
                                    "dtype", "meta", "capability", "instance", "aliased_storage", "slot", "position",
                                    "page_refs", "prefetch", "layout", "attachment", "window"])
def test_inventory_and_lifecycle_fail_closed(rig, change):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    layer = rig.cache.layers[(0, 0)]
    if change == "missing_kv":
        del rig.cache.layers[(0, 0)]
    elif change == "missing_ple":
        del rig.cache.recurrent_layers[(-4, 0)]
    elif change == "unknown_module":
        rig.model.modules.append(NS(caps={"recurrent_cache": True}))
    elif change == "unknown_layer":
        layer.__class__ = type("Unreviewed", (QSA,), {})
    elif change == "extra_tensor":
        layer.unreviewed = torch.zeros(1)
    elif change == "dtype":
        layer.k = layer.k.float()
    elif change == "meta":
        layer.k = layer.k.to("meta")
    elif change == "capability":
        rig.model.modules[-1].caps = {}
    elif change == "instance":
        rig.model.instances[-4].append((-4, 1))
    elif change == "aliased_storage":
        layer.v = layer.k
    elif change == "slot":
        b.recurrent.slot = a.recurrent.slot
    elif change == "position":
        b.recurrent.position -= 1
    elif change == "page_refs":
        rig.lane.pool.references[b.pages[0]] += 1
    elif change == "prefetch":
        rig.model.modules[-1].ple_embedding._pending.append(object())
    elif change == "layout":
        rig.transfer.layout = lambda _state: ([], 1)
    elif change == "attachment":
        rig.model.modules[0].cache_layers.append(QSA(rig.cache, rig.model.modules[0]))
    else:
        rig.lane.context_limit = 131072
    with pytest.raises(ValueError):
        observer.compare(a, b)


def test_freeze_restore_exact_digests_and_no_freed_slot_reads(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer(tile_elements=5)
    for layer in rig.cache.recurrent_layers.values():
        for tensor in layer.get_state_tensors():
            tensor[a.recurrent.slot] = 3
    digest = observer.digest(a)["sha256"]
    old_slot = a.recurrent.slot
    rig.freeze(a)
    for layer in rig.cache.recurrent_layers.values():
        for tensor in layer.get_state_tensors():
            tensor[old_slot] = 77
    for key, items in rig.transfer.layout(b.recurrent)[0]:
        for (target, _, _), source in zip(items, a.checkpoint[key]):
            target.copy_(source)
    assert observer.digest(a)["sha256"] == digest
    assert observer.compare(a, b)["status"] == "observed_equal"
    assert observer.digest(b)["sha256"] == digest
    sibling = rig.branch()
    for layer in rig.cache.recurrent_layers.values():
        for tensor in layer.get_state_tensors():
            tensor[sibling.recurrent.slot] = 44
    assert observer.digest(a)["sha256"] == observer.digest(b)["sha256"] == digest
    a.checkpoint[(2, 0)][0].fill_(1)
    with pytest.raises(ValueError, match="changed from observed live"):
        observer.digest(a)


@pytest.mark.parametrize("change", ["position", "size", "missing", "tuple_copy", "dtype", "arena", "unwitnessed"])
def test_checkpoint_slab_is_authoritative_and_owned(rig, change):
    a, observer = rig.branch(), rig.observer()
    if change != "unwitnessed":
        observer.digest(a)
    rig.freeze(a)
    saved = a.checkpoint
    if change == "position":
        saved["position"] += 1
    elif change == "size":
        saved["checkpoint_size"] += 4
    elif change == "missing":
        del saved[(-4, 0)]
    elif change == "tuple_copy":
        saved[(-4, 0)] = tuple(t.clone() for t in saved[(-4, 0)])
    elif change == "dtype":
        saved[(-4, 0)] = (saved[(-4, 0)][0].float(), saved[(-4, 0)][1])
    elif change == "arena":
        rig.transfer.ARENA = saved["_coalesced_slab"]
    with pytest.raises(ValueError):
        observer.digest(a)


def test_released_branch_rejected_before_slot_or_checkpoint_access(rig):
    a, observer = rig.branch(), rig.observer()
    a.closed = True
    del a.recurrent
    del a.checkpoint
    with pytest.raises(ValueError, match="released"):
        observer.digest(a)


def test_completed_boundary_lock_and_fault_do_not_release_or_mutate_borrowed_branches(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    active = []

    @contextmanager
    def lock():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    rig.lane.lock = lock()
    original = rig.transfer.layout

    def layout(state):
        assert active
        return original(state)

    rig.transfer.layout = layout
    rig.cache.layers[(0, 0)].k[b.pages[0], 0] = float("nan")
    before = list(rig.lane.pool.references)
    with pytest.raises(ValueError, match="Nonfinite"):
        observer.compare(a, b)
    assert not active and list(rig.lane.pool.references) == before
    assert not a.closed and not b.closed


def test_tile_walk_preserves_strided_logical_order():
    value = torch.arange(120).view(4, 5, 6)[:, :, :3]
    chunks = list(_tiles(value, 2))
    assert all(t.numel() <= 2 for t in chunks)
    assert torch.equal(torch.cat([t.contiguous().view(-1) for t in chunks]), value.contiguous().view(-1))


@pytest.mark.parametrize("prefix_length", [16, 260000])
@pytest.mark.parametrize("pending_witness", [False, True])
def test_full_window_serving_is_distinct_from_observer_lifetimes(rig, prefix_length, pending_witness):
    rig.cache.max_num_tokens = rig.service.backend.readout_runtime["cache_tokens"] = 294912
    rig.lane.pool.references, rig.lane.pool.free = [0] * 1152, list(range(1152))
    a = rig.branch(prefix_length)
    rig.freeze(a)
    accepted = a.tokens + (7,)
    suffix = (8,) * (262144 - len(accepted) - 1)
    observer = rig.observer()
    before = list(rig.lane.pool.references)
    report = observer.preflight(NS(branch=a, accepted=accepted), suffix, [(9,), (10,)], pending_witness=pending_witness)
    assert report["serving"]["status"] == "page_estimate_feasible"
    assert report["observer_coexistence"]["status"] == "memory_infeasible"
    assert report["longest_tokens"] == report["window_tokens"] == 262144
    assert report["allocated"] is False and report["qualified"] is False
    assert before == rig.lane.pool.references and a in rig.lane.branches


def test_small_coexistence_preflight_and_exact_cow_margin(rig):
    a = rig.branch(259)
    rig.freeze(a)
    prefix = NS(branch=a, accepted=a.tokens + (7,))
    report = rig.observer().preflight(prefix, (8,), [(9,), (10,)])
    assert report["serving"]["peak_pages"] == 5
    assert report["observer_coexistence"]["peak_pages"] == 8
    assert report["observer_coexistence"]["status"] == "page_estimate_feasible"
    with pytest.raises(ValueError, match="No truncation"):
        rig.observer().preflight(prefix, (8,) * 262144, [(9,)])


def test_new_slot_reset_uses_ple_eos_not_zero_and_bounded_active_views(rig, monkeypatch):
    branch, observer = rig.branch(0), rig.observer(tile_elements=3)
    ple, gdn = rig.cache.recurrent_layers[(-4, 0)], rig.cache.recurrent_layers[(2, 0)]
    slot = branch.recurrent.slot
    ple.id_state[slot].fill_(99)
    # Only active causal views are witnessed; inactive history and unused KV are not reset evidence.
    ple.id_state[slot, ple.ctx:] = -7
    gdn.recurrent_state[slot, 1:] = float("nan")
    rig.cache.layers[(0, 0)].k.fill_(float("nan"))
    original = torch.Tensor.cpu

    def bounded(tensor, *args, **kwargs):
        assert tensor.numel() <= 3
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", bounded)
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_a, **_k: pytest.fail("Reset inspection needs no clone"))
    report = observer.verify_reset(branch)
    assert report["status"] == "observed_reset" and report["qualified"] is False and report["evidence_kind"] == "cpu_fake"
    assert {key: value["reset_value"] for key, value in report["components"].items()} == {
        "2:0/recurrent_state": 0, "2:0/conv_state": 0, "-4:0/conv_state": 0, "-4:0/id_state": 99}
    assert branch.recurrent.position == 0 and branch.pages == []


@pytest.mark.parametrize("component", ["gdn_conv", "gdn_recurrent", "ple_conv", "ple_id_zero", "ple_id_wrong"])
def test_stale_fresh_slot_causal_values_fail_exact_reset(rig, component):
    branch, observer = rig.branch(0), rig.observer()
    slot = branch.recurrent.slot
    ple, gdn = rig.cache.recurrent_layers[(-4, 0)], rig.cache.recurrent_layers[(2, 0)]
    ple.id_state[slot].fill_(99)
    if component == "gdn_conv":
        gdn.conv_state[slot, 0, 0] = 1e-6  # Reset is exact, not a tolerant equivalence comparison.
    elif component == "gdn_recurrent":
        gdn.recurrent_state[slot, 0, 0, 0, 0] = 1e-6
    elif component == "ple_conv":
        ple.conv_state[slot, 0, 0] = 1
    else:
        ple.id_state[slot, 0] = 0 if component == "ple_id_zero" else 98
    with pytest.raises(ValueError, match="reset disagreement"):
        observer.verify_reset(branch)


def test_reset_refuses_consumed_and_released_branches(rig, monkeypatch):
    branch, observer = rig.branch(1), rig.observer()
    with pytest.raises(ValueError, match="fresh live slot"):
        observer.verify_reset(branch)
    branch.closed = True
    monkeypatch.setattr(observer, "_recurrent", lambda *_a: pytest.fail("Never read a freed slot"))
    with pytest.raises(ValueError, match="released branch"):
        observer.verify_reset(branch)


@pytest.mark.parametrize("batch,stage", [(4, "reused_candidates"), (1, "fresh_query_with_reused_states_retained")])
def test_pending_witness_and_split_oracle_reservation_have_real_lifetime_cost(rig, batch, stage):
    branch = rig.branch(259)
    rig.freeze(branch)
    rig.cache.max_num_tokens = rig.service.backend.readout_runtime["cache_tokens"] = 10 * 256
    rig.lane.pool.references = rig.lane.pool.references[:10]
    rig.lane.pool.free = [p for p in rig.lane.pool.free if p < 10]
    rig.service.backend.candidate_batch_size = batch
    observer, prefix = rig.observer(), NS(branch=branch, accepted=branch.tokens + (7,))
    old = observer.preflight(prefix, (8,), [(9,), (10,)])
    new = observer.preflight(prefix, (8,), [(9,), (10,)], pending_witness=True)
    assert old["observer_coexistence"]["status"] == "page_estimate_feasible"
    assert new["serving"] == old["serving"] and new["serving"]["status"] == "page_estimate_feasible"
    assert new["pending_witness"] is True and new["allocated"] is False
    assert new["observer_coexistence"]["status"] == "memory_infeasible"
    assert new["observer_coexistence"]["stage"] == stage
    assert len(rig.lane.branches) == 1 and not branch.closed


def test_pending_snapshot_requires_a_second_slot_but_serving_need_not(rig):
    branch = rig.branch(1)
    rig.freeze(branch)
    rig.cache.num_slots = rig.service.backend.readout_runtime["max_hybrid_slots"] = 1
    rig.cache.free_list = [0]
    rig.service.backend.candidate_batch_size = 1
    report = rig.observer().preflight(NS(branch=branch, accepted=branch.tokens + (7,)), (8,), [(9,)], pending_witness=True)
    assert report["serving"]["status"] == "page_estimate_feasible"
    assert report["observer_coexistence"]["status"] == "memory_infeasible"
    assert report["observer_coexistence"]["stage"] == "pending_B_checkpoint"


def test_fake_bindings_cannot_be_native_evidence(rig):
    with pytest.raises(ValueError, match="cpu_fake"):
        NativeStateObserver(rig.service, allow_observation=True, _test_bindings=rig.types)


def test_logical_page_order_not_physical_ids_controls_comparison(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    b.pages.reverse()
    layer = rig.cache.layers[(0, 0)]
    for branch in (a, b):
        for logical, page in enumerate(branch.pages):
            layer.k[page] = logical + 1
    assert observer.compare(a, b)["status"] == "observed_equal"
    b.pages.reverse()
    with pytest.raises(ValueError, match="disagreement"):
        observer.compare(a, b)


def test_completed_qsa_block_is_not_ignored(rig):
    a, b, observer = rig.branch(260), rig.branch(260), rig.observer()
    rig.cache.layers[(0, 0)].pooled[b.pages[-1], 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        observer.compare(a, b)


def test_tolerance_is_elementwise_and_policy_cannot_change_after_admission(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    state = rig.cache.recurrent_layers[(2, 0)].recurrent_state
    state[b.recurrent.slot, 0, 0, 0, 0] = 5e-6
    assert observer.compare(a, b)["components"]["2:0/recurrent_state"]["max_abs_error"] > 0
    state[b.recurrent.slot, 0, 0, 0, 0] = 2e-5
    with pytest.raises(ValueError, match="disagreement"):
        observer.compare(a, b)
    observer.atol = 1
    with pytest.raises(ValueError, match="policy changed"):
        observer.compare(a, b)


def test_guard_failure_precedes_tensor_reads_and_keeps_caller_ownership(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()

    def refused(*_args):
        raise MemoryError("admission refused")

    rig.service.admission = refused
    rig.transfer.layout = lambda *_args: pytest.fail("Read after failed admission")
    with pytest.raises(MemoryError, match="admission refused"):
        observer.compare(a, b)
    assert a in rig.lane.branches and b in rig.lane.branches


def test_same_branch_or_shared_pages_cannot_prove_independent_replay(rig):
    a, observer = rig.branch(), rig.observer()
    with pytest.raises(ValueError, match="independent replay"):
        observer.compare(a, a)
    b = rig.branch()
    for page in b.pages:
        rig.lane.pool.references[page] = 0
        rig.lane.pool.free.append(page)
    b.pages = list(a.pages)
    for page in b.pages:
        rig.lane.pool.references[page] += 1
    with pytest.raises(ValueError, match="independent replay"):
        observer.compare(a, b)


def test_int64_history_is_exact_beyond_float64_precision(rig):
    a, b, observer = rig.branch(), rig.branch(), rig.observer()
    ids = rig.cache.recurrent_layers[(-4, 0)].id_state
    ids[a.recurrent.slot, 0] = 2**60
    ids[b.recurrent.slot, 0] = 2**60 + 1
    with pytest.raises(ValueError, match="integer disagreement"):
        observer.compare(a, b)


@pytest.mark.parametrize("owner", ["arena", "frozen_checkpoint"])
@pytest.mark.parametrize("offset", [0, 8])
@pytest.mark.parametrize("operation", ["digest", "compare"])
def test_live_id_state_cannot_borrow_checkpoint_transfer_storage(rig, owner, offset, operation):
    # The real PLE ID buffer is CPU int64; staging writes and ID updates need independent owners.
    if owner == "frozen_checkpoint":
        rig.cache.num_slots = rig.service.backend.readout_runtime["max_hybrid_slots"] = 2
        rig.cache.free_list = [0, 1]
        rig.service.backend.candidate_batch_size = 2
        for layer in rig.cache.recurrent_layers.values():
            layer.max_batch_size = 2
            for name in ("conv_state", "recurrent_state", "id_state"):
                if hasattr(layer, name):
                    setattr(layer, name, getattr(layer, name)[:2].clone())
    observer = rig.observer()
    if owner == "frozen_checkpoint":
        parent = rig.branch()
        observer.digest(parent)
        rig.freeze(parent)
        storage = parent.checkpoint["_coalesced_slab"]
    else:
        storage = torch.zeros(200, dtype=torch.uint8)
        rig.transfer.ARENA = storage
    a, b = rig.branch(), rig.branch()
    report = observer.compare(a, b)
    assert report["status"] == "observed_equal" and report["qualified"] is False
    assert report["evidence_kind"] == "cpu_fake"
    layer = rig.cache.recurrent_layers[(-4, 0)]
    size = layer.id_state.numel() * layer.id_state.element_size()
    layer.id_state = storage[offset:offset + size].view(torch.int64).view(layer.id_state.shape)
    before = storage.clone()
    pages, slots = list(rig.lane.pool.references), list(rig.cache.free_list)
    with pytest.raises(ValueError, match="aliases checkpoint or staging"):
        observer.digest(a) if operation == "digest" else observer.compare(a, b)
    assert torch.equal(storage, before)
    assert pages == rig.lane.pool.references and slots == rig.cache.free_list
    assert a in rig.lane.branches and b in rig.lane.branches and not a.closed and not b.closed
