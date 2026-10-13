"""Opt-in, borrowed-state observations; never loads an engine or qualifies a run."""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import struct
import sys
from weakref import WeakKeyDictionary


ENGINE_COMMIT = "ba60611d031f673a12ebbf2f133c583b4e12df7f"  # unpublished lab commit ID
PAGE = 256
WINDOW = 262144
IGNORED = {
    "unconsumed_pages": "Unused rows and free pages are not causal state.",
    "qsa_partial_pool": "Incomplete pooled block is never selected and is rebuilt; raw keys are compared.",
    "reserved_history": "Trailing history requires witnessed non-speculative forwards; last_history=0 alone is insufficient.",
    "ple_weights": "Streamed tables/handles are weights, not slot convolution or n-gram ID history.",
    "scratch": "Prefetch staging and derived attention/RoPE workspaces are not branch checkpoints.",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bindings():
    # Only consult already loaded modules. Importing exllamav3 could bootstrap its extension.
    paths = {
        "fp16": ("exllamav3.cache.fp16", "CacheLayer_fp16"),
        "qsa": ("exllamav3.cache.qsa", "CacheLayer_qsa"),
        "gdn": ("exllamav3.modules.gated_delta_net", "GDNLayerState"),
        "ple": ("exllamav3.modules.ple", "PLELayerState"),
        "state": ("exllamav3.modules.gated_delta_net", "GDNState"),
        "attention": ("exllamav3.modules.attn", "Attention"),
        "gdn_module": ("exllamav3.modules.gated_delta_net", "GatedDeltaNet"),
        "ple_module": ("exllamav3.modules.ple", "PLELayer"),
        "model": ("exllamav3.architecture.qwen4_exp", "Qwen4ExpModel"),
    }
    _require(all(path in sys.modules for path, _ in paths.values())
             and "checkpoint_transfer" in sys.modules, "Pinned native modules must already be loaded")
    result = {key: getattr(sys.modules[path], name) for key, (path, name) in paths.items()}
    result["transfer"] = sys.modules["checkpoint_transfer"]
    return result


def _tiles(tensor, limit):
    """Split views before any contiguous/CPU conversion, including strided slot windows."""
    if tensor.numel() <= limit:
        yield tensor
    else:
        axis = next(i for i, width in enumerate(tensor.shape) if width > 1)
        half = tensor.shape[axis] // 2
        yield from _tiles(tensor.narrow(axis, 0, half), limit)
        yield from _tiles(tensor.narrow(axis, half, tensor.shape[axis] - half), limit)


class NativeStateObserver:
    def __init__(self, service, *, allow_observation=False, cpu_fake=False,
                 atol=1e-5, rtol=1e-4, tile_elements=4096, _test_bindings=None):
        _require(allow_observation is True, "Explicit observation opt-in required")
        _require(type(cpu_fake) is bool and (_test_bindings is None or cpu_fake),
                 "Test bindings require an explicit cpu_fake label")
        _require(all(type(v) in (int, float) and math.isfinite(v) and 0 < v <= cap
                     for v, cap in ((atol, 1e-5), (rtol, 1e-4))), "Invalid or relaxed tolerances")
        _require(type(tile_elements) is int and 1 <= tile_elements <= 16384, "Invalid tile bound")
        service._owner()
        self.service, self.lane, self.cache = service, service.lane, service.backend.cache
        self.cpu_fake, self.atol, self.rtol, self.tile_elements = cpu_fake, atol, rtol, tile_elements
        self._policy = (cpu_fake, atol, rtol, tile_elements)
        self.types = _test_bindings if _test_bindings is not None else _bindings()
        self._live_digests = WeakKeyDictionary()
        self.metadata = {"source_verification": "cpu_fake_not_native_provenance"}
        if not cpu_fake:
            from .native_qualification import _metadata
            self.metadata = _metadata(service, False)
            self.metadata["recorded_engine_commit"] = ENGINE_COMMIT
        self._guard()

    def _guard(self):
        service = self.service
        service._owner()
        _require((self.cpu_fake, self.atol, self.rtol, self.tile_elements) == self._policy,
                 "Observation policy changed after admission")
        runtime = service.backend.readout_runtime
        fraction = runtime["memory_fraction"]
        _require(fraction is None or (type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 1),
                 "Declared native memory fraction must be in (0, 1]")
        _require(self.lane.cache is self.cache and self.cache.model is self.lane.model
                 and self.cache.initialized and not self.lane.model.loaded_tp
                 and self.lane.context_limit == runtime["context_tokens"] == WINDOW
                 and self.cache.max_num_tokens == runtime["cache_tokens"]
                 and self.cache.num_slots == runtime["max_hybrid_slots"], "Native geometry changed")
        _require(type(self.lane.chunk_size) is int and self.lane.chunk_size > 0
                 and self.lane.chunk_size == runtime["feature_chunk_tokens"], "Native chunk geometry changed")
        if not self.cpu_fake:
            from .guard_policy import verify_memory_guard
            policy = verify_memory_guard(service.guard, fraction, message="Existing native MemoryGuard must remain running")
            previous = self.metadata.get("memory_guard_policy")
            _require(previous is None or previous == policy, "Host memory-guard policy changed after observation admission")
            self.metadata["memory_guard_policy"] = policy
        service.guard.check()
        service.admission(self.tile_elements * 128, fraction)

    def _ledger(self):
        counts, slots = Counter(), set()
        for branch in self.lane.branches:
            _require(branch.owner is self.lane and not branch.closed, "Foreign or released branch")
            n = len(branch.tokens)
            _require(type(branch.tokens) is tuple and all(type(t) is int and 0 <= t < 2**63 for t in branch.tokens)
                     and n <= WINDOW, "Invalid consumed token IDs")
            _require(len(branch.pages) == (n + PAGE - 1) // PAGE
                     and len(set(branch.pages)) == len(branch.pages)
                     and all(type(p) is int and 0 <= p < self.cache.max_num_tokens // PAGE
                             for p in branch.pages), "Invalid logical page map")
            counts.update(branch.pages)
            if branch.recurrent is not None:
                state = branch.recurrent
                _require(not branch.immutable and branch.checkpoint is None
                         and type(state) is self.types["state"] and state.cache is self.cache
                         and not state.exported and type(state.position) is int and state.position == n
                         and type(state.last_history) is int and state.last_history == 0,
                         "Unsupported recurrent position/history or ownership")
                _require(type(state.slot) is int and 0 <= state.slot < self.cache.num_slots
                         and state.slot not in slots, "Aliased or invalid recurrent slot")
                slots.add(state.slot)
            else:
                _require(branch.immutable and type(branch.checkpoint) is dict
                         and type(branch.checkpoint.get("position")) is int
                         and branch.checkpoint["position"] == n, "Invalid frozen position")
        free = list(self.cache.free_list)
        _require(all(type(v) is int for v in free) and len(free) == len(set(free))
                 and set(free) == set(range(self.cache.num_slots)) - slots,
                 "Recurrent free-list disagreement")
        pool = self.lane.pool
        _require(len(pool.references) == self.cache.max_num_tokens // PAGE
                 and all(type(v) is int and v == counts[p] for p, v in enumerate(pool.references)),
                 "Page reference disagreement")
        free = list(pool.free)
        _require(all(type(v) is int for v in free) and len(free) == len(set(free))
                 and set(free) == set(range(len(pool.references))) - counts.keys(),
                 "Page free-list disagreement")

    @contextmanager
    def _scope(self, *branches, fence=True):
        self._guard()
        with self.lane.lock:
            _require(all(b.owner is self.lane and not b.closed and b in self.lane.branches for b in branches),
                     "Cannot inspect a foreign or released branch")
            self._ledger()
            if fence and not self.cpu_fake:
                import torch
                torch.cuda.current_stream(0).synchronize()
            try:
                yield
            finally:
                self._guard()

    def _tensor(self, tensor, dtype, shape, *, cpu=False):
        import torch
        device = torch.device("cpu" if cpu or self.cpu_fake else "cuda:0")
        _require(type(tensor) is torch.Tensor and tensor.layout == torch.strided
                 and tensor.dtype == dtype and tuple(tensor.shape) == tuple(shape)
                 and tensor.device == device and not tensor.requires_grad, "Actual tensor dtype/shape/device mismatch")

    def _inventory(self):
        import torch
        cache, model, types = self.cache, self.lane.model, self.types
        _require(type(model) is types["model"] and cache.layer_type is types["fp16"]
                 and cache.recurrent_state_cls is types["state"]
                 and cache.max_history == 3 and cache.max_num_tokens % PAGE == 0,
                 "Unsupported native model/cache/history")
        expected = {"kv_cache": {}, "recurrent_cache": {}}
        module_kinds = {types["attention"]: "kv_cache", types["gdn_module"]: "recurrent_cache",
                        types["ple_module"]: "recurrent_cache"}
        for module in model:
            kind = module_kinds.get(type(module))
            if kind:
                _require(module.caps.get(kind) is True, "Stateful module registration missing")
            for cap, registry in expected.items():
                if module.caps.get(cap):
                    _require(cap == kind, "Unknown registered module")
                    instances = model.get_layer_instances(module.layer_idx)
                    _require(instances, "Registered module has no instances")
                    for key in instances:
                        _require(type(key) is tuple and len(key) == 2
                                 and all(type(v) is int for v in key) and key[0] == module.layer_idx
                                 and key[1] >= 0 and key not in registry, "Duplicate/invalid registered instance")
                        registry[key] = module
        _require(expected["kv_cache"] and expected["recurrent_cache"], "Empty hybrid inventory")
        for cap, getter, registry in (("kv_cache", model.get_cache_layers, cache.layers),
                                      ("recurrent_cache", model.get_recurrent_layers, cache.recurrent_layers)):
            modules = getter()
            _require(set(registry) == set(expected[cap])
                     and len(modules) == len({id(m) for m in modules})
                     and {id(m) for m in modules} == {id(m) for m in expected[cap].values()},
                     "Incomplete cache/module registry")
            for module in modules:
                attached = module.cache_layers if cap == "kv_cache" else module.recurrent_layers
                attached = [layer for layer in attached if layer.cache_id == id(cache)]
                wanted = [registry[key] for key, owner in expected[cap].items() if owner is module]
                _require(len(attached) == len(wanted) and {id(layer) for layer in attached} == {id(layer) for layer in wanted},
                         "Incomplete module cache attachment registry")
        _require(cache.get_all_recurrent_layers() is cache.recurrent_layers, "Recurrent registry changed")
        _require(len({id(layer) for layer in (*cache.layers.values(), *cache.recurrent_layers.values())})
                 == len(cache.layers) + len(cache.recurrent_layers), "Aliased registered layer instances")
        transfer = types["transfer"]
        _require(transfer.hooks.CONFIG.get("checkpoint_transfer") == "coalesced", "Coalesced checkpoints required")
        inventory = []
        for key, layer in cache.layers.items():
            module = expected["kv_cache"][key]
            qsa = module.qsa_indexer is not None
            _require(type(layer) is types["qsa" if qsa else "fp16"] and layer.attention is module
                     and layer.cache_id == id(cache) and layer.max_num_tokens == cache.max_num_tokens
                     and layer in module.cache_layers, "Unknown/misbound KV layer")
            shape = (cache.max_num_tokens // PAGE, PAGE, module.num_kv_heads, module.head_dim)
            fields = [("k", torch.float16, shape, None), ("v", torch.float16, shape, None)]
            if qsa:
                ratio, width = module.qsa_indexer.compress_ratio, module.qsa_indexer.head_dim
                _require(type(ratio) is int and ratio > 0 and PAGE % ratio == 0
                         and layer.compress_ratio == ratio and layer.index_head_dim == width, "Invalid QSA geometry")
                fields += [("raw_k", torch.float16, (shape[0], PAGE, width), None),
                           ("pooled", torch.float16, (shape[0], PAGE // ratio, width), None)]
            inventory.append((key, layer, fields, True))
        for key, layer in cache.recurrent_layers.items():
            module = expected["recurrent_cache"][key]
            _require(layer.module is module and layer.cache_id == id(cache)
                     and layer in module.recurrent_layers and layer.max_history == cache.max_history
                     and layer.max_batch_size == cache.num_slots, "Misbound recurrent layer")
            slots, history = cache.num_slots, cache.max_history
            if type(module) is types["gdn_module"] and type(layer) is types["gdn"]:
                fields = [("recurrent_state", torch.float32,
                           (slots, history + 1, module.num_v_heads, module.k_head_dim, module.v_head_dim),
                           (slice(0, 1), slice(None), slice(None), slice(None))),
                          ("conv_state", torch.bfloat16, (slots, module.fdim_qkv, module.conv_kernel_size + history),
                           (slice(None), slice(0, module.conv_kernel_size)))]
            elif type(module) is types["ple_module"] and type(layer) is types["ple"]:
                _require(layer.win == module.conv_state_len == (module.conv_kernel_size - 1) * module.conv_dilation
                         and layer.ctx == module.ple_embedding.context_len > 0, "Invalid PLE geometry")
                fields = [("conv_state", torch.float16,
                           (slots, module.hc_mult * module.hidden_size, layer.win + history),
                           (slice(None), slice(0, layer.win))),
                          ("id_state", torch.int64, (slots, layer.ctx + history), (slice(0, layer.ctx),))]
                embedding = module.ple_embedding
                _require(not embedding._pending and not any(p.held for p in embedding._pins),
                         "PLE staging not quiescent; observer will not drain it")
            else:
                raise ValueError("Unknown recurrent class")
            inventory.append((key, layer, fields, False))
        # Transfers rewrite the arena; live history writes must not mutate retained slabs.
        protected = [transfer.ARENA]
        protected += [b.checkpoint["_coalesced_slab"] for b in self.lane.branches
                      if b.checkpoint is not None and "_coalesced_slab" in b.checkpoint]
        _require(all(t is None or type(t) is torch.Tensor for t in protected),
                 "Invalid checkpoint or staging storage")
        storage = []
        for _, layer, fields, paged in inventory:
            tensors = [getattr(layer, name) for name, *_ in fields]
            exported = layer.get_tensors() if paged else layer.get_state_tensors()
            _require(len(exported) == len(tensors) and {id(t) for t in exported} == {id(t) for t in tensors}
                     and {name for name, value in vars(layer).items() if isinstance(value, torch.Tensor)}
                     == {name for name, *_ in fields}, "Unaccounted registered tensor")
            for name, dtype, shape, _ in fields:
                _require(all(type(n) is int and n > 0 for n in shape), "Invalid registered shape")
                tensor = getattr(layer, name)
                self._tensor(tensor, dtype, shape, cpu=name == "id_state")
                _require(tensor.is_contiguous(), "Unexpected backing tensor strides")
                _require(all(t is None or t.device != tensor.device
                             or t.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
                             for t in protected), "Live cache tensor aliases checkpoint or staging storage")
                start, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
                _require(all(tensor.device != dev or end <= lo or hi <= start for dev, lo, hi in storage),
                         "Aliased registered tensor storage")
                storage.append((tensor.device, start, end))
        return inventory

    def _recurrent(self, branch, inventory):
        import torch
        state, saved = branch.recurrent, branch.checkpoint
        result, offset, native_size = {}, 0, 0
        layout = self.types["transfer"].layout(state) if state is not None else None
        groups = []
        if saved is not None:
            _require(branch in self._live_digests and self._live_digests[branch][0] == branch.tokens,
                     "Frozen state needs an observed live non-speculative digest before freeze")
            _require(set(saved) == set(self.cache.recurrent_layers) | {"position", "checkpoint_size", "_coalesced_slab"},
                     "Incomplete checkpoint inventory")
            slab = saved["_coalesced_slab"]
            _require(type(saved["checkpoint_size"]) is int, "Invalid checkpoint byte count")
            self._tensor(slab, torch.uint8, (saved["checkpoint_size"],), cpu=True)
            _require(slab.is_contiguous() and slab._base is None and slab.storage_offset() == 0
                     and slab.untyped_storage().nbytes() == slab.numel(), "Checkpoint slab must own exact storage")
            forbidden = [self.types["transfer"].ARENA]
            forbidden += [getattr(layer, name) for _, layer, fields, _ in inventory for name, *_ in fields]
            forbidden += [b.checkpoint["_coalesced_slab"] for b in self.lane.branches
                          if b is not branch and b.checkpoint is not None and "_coalesced_slab" in b.checkpoint]
            _require(all(t is None or t.device != slab.device
                         or t.untyped_storage().data_ptr() != slab.untyped_storage().data_ptr() for t in forbidden),
                     "Checkpoint aliases slots, staging, or another checkpoint")
        for key, layer, fields, paged in inventory:
            if paged:
                continue
            native_size += layer.get_checkpoint_size()
            items = []
            for index, (name, dtype, shape, slices) in enumerate(fields):
                active_shape = tuple(len(range(*s.indices(n))) for s, n in zip(slices, shape[1:]))
                size = math.prod(active_shape) * getattr(layer, name).element_size()
                _require(offset % getattr(layer, name).element_size() == 0, "Unaccounted checkpoint padding")
                if state is not None:
                    value = getattr(layer, name)[(state.slot, *slices)]
                else:
                    value = slab[offset:offset + size].view(dtype).view(active_shape)
                    _require(type(saved[key]) is tuple and len(saved[key]) == len(fields), "Checkpoint tuple mismatch")
                    recorded = saved[key][index]
                    self._tensor(recorded, dtype, active_shape, cpu=True)
                    _require(recorded.is_contiguous() and recorded.data_ptr() == value.data_ptr()
                             and recorded.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr(),
                             "Checkpoint tuple is not an authoritative slab view")
                result[(key, name)] = value
                items.append((value, offset, size))
                offset += size
            groups.append((key, items))
        _require(0 < offset == native_size <= 128 * 2**20, "Invalid exact checkpoint budget")
        if state is None:
            _require(saved["checkpoint_size"] == offset, "Checkpoint size disagreement")
        else:
            _require(type(state.checkpoint_size) is int and state.checkpoint_size == offset
                     and layout[1] == offset and len(layout[0]) == len(groups), "Native layout size disagreement")
            for (key, items), (actual_key, actual) in zip(groups, layout[0]):
                _require(key == actual_key and len(items) == len(actual), "Native layout registry disagreement")
                for (view, start, size), (other, other_start, other_size) in zip(items, actual):
                    _require(start == other_start and size == other_size and view.dtype == other.dtype
                             and view.shape == other.shape and view.stride() == other.stride()
                             and view.device == other.device and view.data_ptr() == other.data_ptr(),
                             "Native layout view disagreement")
        return result

    def _pieces(self, branch, inventory):
        recurrent = self._recurrent(branch, inventory)
        for key, layer, fields, paged in inventory:
            for name, _, _, _ in fields:
                label = f"{key[0]}:{key[1]}/{name}"
                if paged:
                    ratio = layer.compress_ratio if name == "pooled" else 1
                    for logical, physical in enumerate(branch.pages):
                        rows = min(PAGE, len(branch.tokens) - logical * PAGE) // ratio
                        if rows:
                            yield label, getattr(layer, name)[physical, :rows]
                else:
                    yield label, recurrent[(key, name)]

    def _receipt(self, inventory, scope):
        return {"qualified": False, "scope": scope, "evidence_kind": "cpu_fake" if self.cpu_fake else "native",
                "source_verification": self.metadata["source_verification"], "ignored": dict(IGNORED),
                "provenance": {key: self.metadata[key] for key in ("runtime_sources", "model", "readout_runtime")
                               if key in self.metadata},
                "registrations": {f"{key[0]}:{key[1]}": type(layer).__module__ + "." + type(layer).__qualname__
                                  for key, layer, _, _ in inventory},
                "tolerances": {"atol": self.atol, "rtol": self.rtol}, "tile_elements": self.tile_elements,
                "components": {f"{key[0]}:{key[1]}/{name}": {"dtype": str(dtype), "elements": 0,
                                "backing_shape": list(getattr(layer, name).shape)}
                               for key, layer, fields, _ in inventory for name, dtype, _, _ in fields}}

    def compare(self, reused, fresh):
        import torch
        with self._scope(reused, fresh):
            _require(reused is not fresh and reused.tokens == fresh.tokens
                     and not set(reused.pages).intersection(fresh.pages), "Exact independent replay boundaries required")
            for branch in (reused, fresh):
                if branch.checkpoint is not None:
                    self.digest(branch)  # Verify the observed live-to-frozen transfer, not just tuple metadata.
            inventory = self._inventory()
            receipt = self._receipt(inventory, "owned_causal_state_equivalence_only")
            left, right = self._pieces(reused, inventory), self._pieces(fresh, inventory)
            for label, a in left:
                other, b = next(right)
                _require(label == other and a.shape == b.shape and a.dtype == b.dtype, "Comparison layout mismatch")
                component = receipt["components"][label]
                for x, y in zip(_tiles(a, self.tile_elements), _tiles(b, self.tile_elements)):
                    self._guard()
                    x, y = x.detach().cpu(), y.detach().cpu()
                    if x.is_floating_point():
                        _require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), "Nonfinite causal state")
                        x, y = x.double(), y.double()
                        error = (x - y).abs()
                        maximum = error.max().item()
                        _require(bool((error <= self.atol + self.rtol * y.abs()).all()), f"{label}: numerical disagreement")
                    else:
                        _require(torch.equal(x, y), f"{label}: integer disagreement")
                        maximum = 0
                    component["elements"] += x.numel()
                    component["max_abs_error"] = max(component.get("max_abs_error", 0), maximum)
            _require(next(right, None) is None, "Uncompared state")
            receipt.update(status="observed_equal", consumed_tokens=len(reused.tokens))
            return receipt

    def digest(self, branch):
        import torch
        with self._scope(branch):
            inventory = self._inventory()
            receipt = self._receipt(inventory, "byte_exact_owned_state_digest_only")
            digest = hashlib.sha256(json.dumps(receipt["components"], sort_keys=True).encode())
            digest.update(struct.pack("<Q", len(branch.tokens)))
            for token in branch.tokens:
                digest.update(struct.pack("<Q", token))
            for label, tensor in self._pieces(branch, inventory):
                digest.update(json.dumps([label, list(tensor.shape)]).encode())
                for tile in _tiles(tensor, self.tile_elements):
                    self._guard()
                    tile = tile.detach().cpu().contiguous()
                    _require(not tile.is_floating_point() or bool(torch.isfinite(tile).all()), "Nonfinite causal state")
                    digest.update(tile.view(torch.uint8).numpy().tobytes())
                    receipt["components"][label]["elements"] += tile.numel()
            value = digest.hexdigest()
            if branch.recurrent is not None:
                self._live_digests[branch] = (branch.tokens, value)
            else:
                _require(value == self._live_digests[branch][1], "Frozen state changed from observed live digest")
            receipt.update(status="observed", sha256=value, consumed_tokens=len(branch.tokens))
            return receipt

    def verify_reset(self, branch):
        """Check new-slot active views, not GDNState.reset() (which only sets position)."""
        with self._scope(branch):
            _require(branch.recurrent is not None and not branch.tokens and not branch.pages,
                     "Reset observation requires an owned fresh live slot before any append")
            inventory = self._inventory()
            views = self._recurrent(branch, inventory)
            receipt = self._receipt(inventory, "fresh_slot_active_causal_reset_only")
            receipt["components"] = {}
            for key, layer, fields, paged in inventory:
                if paged:
                    continue  # New slots own no KV pages; cleared KV storage is not the reset contract.
                for name, *_ in fields:
                    value = layer.module.ple_embedding.eos_token_id if name == "id_state" else 0
                    _require(type(value) is int and -2**63 <= value < 2**63, "Unsupported reset sentinel")
                    tensor = views[(key, name)]
                    for tile in _tiles(tensor, self.tile_elements):
                        self._guard()
                        _require(bool((tile.detach().cpu() == value).all()), f"{key}/{name}: reset disagreement")
                    receipt["components"][f"{key[0]}:{key[1]}/{name}"] = {
                        "dtype": str(tensor.dtype), "elements": tensor.numel(), "reset_value": value}
            receipt.update(status="observed_reset", consumed_tokens=0,
                contract="Cache.get_new_state(clear=True) -> GDNLayerState.clear / PLELayerState.clear; active views only")
            return receipt

    def preflight(self, prefix, suffix, tails, *, pending_witness=False):
        """Shadow actual page lifetimes only: no allocation, eviction, forward, or stream query."""
        with self._scope(prefix.branch, fence=False):
            accepted, consumed = prefix.accepted, prefix.branch.tokens
            _require(type(pending_witness) is bool, "Pending witness opt-in must be explicit")
            _require(prefix.branch.immutable and type(accepted) is tuple and accepted[:len(consumed)] == consumed
                     and 0 <= len(accepted) - len(consumed) <= 1, "Invalid accepted/consumed prefix")
            _require(not pending_witness or len(accepted) - len(consumed) == 1,
                     "Pending witness requires exactly one unresolved accepted token")
            _require(suffix and tails and all(tails), "Exact encoded suffix/tails required")
            for tokens in (accepted, suffix, *tails):
                _require(type(tokens) is tuple and all(type(t) is int and 0 <= t < 2**63 for t in tokens),
                         "Preflight requires exact encoded token IDs")
            longest = len(accepted) + len(suffix) + max(map(len, tails))
            _require(longest <= WINDOW, "No truncation of the 262144 window")
            width = self.service.backend.candidate_batch_size
            _require(type(width) is int and 1 <= width <= self.cache.num_slots, "Invalid candidate batch size")
            def simulate(comparison):
                refs = Counter({p: n for p, n in enumerate(self.lane.pool.references) if n})
                capacity, peak, serial = len(self.lane.pool.references), len(refs), -1
                stage = "reused_query"

                def reserve(tokens):
                    if capacity - len(refs) < (tokens + PAGE - 1) // PAGE + 2:
                        raise MemoryError("native_reservation_margin")

                def allocate():
                    nonlocal serial, peak
                    if len(refs) >= capacity:
                        raise MemoryError("physical_page_capacity")
                    page = serial
                    serial -= 1
                    refs[page] = 1
                    peak = max(peak, len(refs))
                    return page

                def fork(branch):
                    refs.update(branch[0])
                    return [list(branch[0]), branch[1]]

                def release(branch):
                    for page in branch[0]:
                        refs[page] -= 1
                        if not refs[page]:
                            del refs[page]

                def append(branch, count, guarded=True):
                    for start in range(0, count, self.lane.chunk_size):
                        chunk = min(count - start, self.lane.chunk_size)
                        if guarded:
                            reserve(chunk)
                        pages, n = branch
                        if n % PAGE and refs[pages[-1]] > 1:
                            old = pages[-1]
                            pages[-1] = allocate()
                            refs[old] -= 1
                        while len(pages) < (n + chunk + PAGE - 1) // PAGE:
                            pages.append(allocate())
                        branch[1] += chunk

                try:
                    query = fork([prefix.branch.pages, len(consumed)])
                    append(query, len(accepted) - len(consumed))
                    pending = None
                    if comparison and pending_witness:
                        stage = "pending_B_checkpoint"
                        if len(self.cache.free_list) < 2:
                            raise MemoryError("pending_checkpoint_needs_query_and_borrowed_slot")
                        pending = fork(query)  # Frozen until the fresh pending boundary is compared.
                    stage = "reused_query"
                    append(query, len(suffix))
                    stage = "reused_candidates"
                    for length in dict.fromkeys(map(len, tails)):
                        count = sum(len(tail) == length for tail in tails)
                        for start in range(0, count, width):
                            batch = min(width, count - start)
                            reserve(batch * (length + 512))
                            children = [fork(query) for _ in range(batch)]
                            for child in children:
                                append(child, length, False)
                            if not comparison:
                                for child in children:
                                    release(child)
                    if comparison:
                        stage = "fresh_query_with_reused_states_retained"
                        fresh = [[], 0]
                        if pending_witness:
                            append(fresh, len(consumed))
                            append(fresh, len(accepted) - len(consumed))
                            release(pending)
                        else:
                            append(fresh, len(accepted))
                        append(fresh, len(suffix))
                        stage = "fresh_candidates_with_reused_states_retained"
                        for tail in tails:
                            child = fork(fresh)
                            append(child, len(tail))
                            release(child)
                    return {"status": "page_estimate_feasible", "peak_pages": peak}
                except MemoryError as error:
                    return {"status": "memory_infeasible", "stage": stage, "reason": str(error),
                            "peak_pages_before_failure": peak}

            return {"qualified": False, "allocated": False, "window_tokens": WINDOW, "longest_tokens": longest,
                    "pending_witness": pending_witness,
                    "evidence_kind": "cpu_fake" if self.cpu_fake else "native_page_arithmetic_only",
                    "capacity_pages": len(self.lane.pool.references), "initial_pages": sum(n > 0 for n in self.lane.pool.references),
                    "serving": simulate(False), "observer_coexistence": simulate(True),
                    "scope": "Existing branches retained; reused query/all tails plus fresh replay; optional pending witness released after fresh pending comparison",
                    "limitations": "Page arithmetic only; no shared-RAM, recurrent-slot, frozen-head, or inference feasibility proof"}
