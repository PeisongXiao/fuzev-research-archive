# Native State Observer

The passive observer records branch state at explicit boundaries for
before/after and parent/child comparisons.

## Contract

`NativeStateObserver(service, allow_observation=True)` borrows an already
admitted, exclusively owned offline native service. It does not create a
backend, run forwards, freeze/fork/release branches, start workers, or write
files. Disabled construction does not even
import Torch. Native binding lookup uses already loaded modules, not an engine
import that could bootstrap extensions. The lab run recorded source revisions
(`git rev-parse` when available, otherwise package versions).
The lab audit covered unpublished lab commit `ba60611`, single-device Qwen4Exp,
FP16 cache, coalesced checkpoints and non-speculative history.

- `compare(reused, fresh)` requires identical consumed token tuples, distinct
  branches and disjoint physical pages. It compares all registered causal
  fields, returning counts, actual dtype labels and maximum absolute errors.
  This alone observes state equivalence, not fresh reset or identical forward inputs.
- `digest(branch)` streams byte-exact state and token digests for before/after
  freeze, restore and parent/sibling isolation checks. Caller compares returned
  hashes. A live digest with `last_history == 0` is mandatory before inspecting
  that branch frozen: checkpoints do not preserve `last_history`, so a frozen
  checkpoint alone cannot attest non-speculative execution. Frozen digest and
  comparison also require exact agreement with that live digest.
- `verify_reset(branch)` requires a fresh live branch with no consumed tokens or
  pages. It checks exact active causal reset contents, not an all-cache zero
  assumption; see the source contract below.
- `preflight(prefix, suffix, tails, pending_witness=False)` accepts the retained prefix and exact
  encoded token tuples. It shadows page lifetimes without touching tensors,
  allocating pages, evicting prefixes, querying a stream or changing the window.
  Opting into the pending witness includes its checkpoint lifetime/COW, temporary
  second live slot, and the fresh replay's consumed/pending append split.

All observations run on the existing service owner under `lane.lock`. Native
tensor observations fence the owner's current CUDA stream before reads. The
lab used MemoryGuard and admission checks at allocation fraction 0.85,
including comparison scratch. The observer creates no guard worker. Policy is fixed at construction: `atol <= 1e-5`, `rtol <= 1e-4`,
default 4096 and maximum 16384 elements per tile. Nonfinite values fail; integer
values, actual tensor dtypes, positions and byte-exact digests never use tolerance.

## Inventory And Ownership

Recursive module capabilities and `get_layer_instances()` must exactly match
`cache.layers` and `cache.recurrent_layers`, including negative PLE keys.
Unknown classes, missing registrations, unexpected tensor fields, misbound
modules, aliases, incompatible layouts and contradictory page/slot ledgers
fail closed. Completeness is not inferred from shapes, `num_layers`, or
`get_all_tensors()` (which omits recurrence).

| Known state | Causal views |
| --- | --- |
| FP16 KV | `k`, `v`, logical pages and consumed rows only |
| QSA | KV plus `raw_k`; `pooled` only for completed `compress_ratio` blocks |
| GDN | `recurrent_state[slot, :1]`, `conv_state[slot, :, :conv_kernel_size]` |
| PLE | `conv_state[slot, :, :win]`, CPU int64 `id_state[slot, :ctx]` |

Live recurrent views are cross-checked against the existing optimized
`checkpoint_transfer.layout(state)`. The observer never calls its `arena()`,
`snapshot()` or `restore()`. Frozen layout is derived from the explicit field
contract and backing tensor metadata, without borrowing any released slot.
`_coalesced_slab` is authoritative, with exact byte count, position, independent
owned storage and tuple-view offsets checked. Staging/slot/other-checkpoint
aliases are rejected in both directions: registered cache backing tensors must
not share root storage with the transfer arena or any retained checkpoint slab,
even through an offset view. An unwitnessed frozen checkpoint remains unsupported.

Unused page rows, incomplete QSA pooled rows and reserved speculative history
are explicitly excluded with rationales in records. QSA raw keys for the
partial block remain included. PLE tables/handles are streamed weights, not
causal history; no table is read or copied. Prefetch staging must be quiescent
(`_pending` empty, no held pins); the observer never drains it to mask cleanup
failures. Derived attention/RoPE scratch is not a checkpoint component.

Tiles are sliced before contiguous conversion or CPU transfer, including
strided convolution windows. There is no full-cache gather, cache clone, slot
snapshot, or whole-PLE copy. Reports contain no tensors. Branches remain owned
by the caller throughout; exceptions unwind the lock without releasing or
mutating them. Only weak branch keys and token/digest metadata survive calls.
Released/foreign branches are rejected before inspecting their state.

## Pinned Source Audit

The isolated audit from recipes unpublished lab commit `61954d4` checked the engine at unpublished lab commit `ba60611`:
`exllamav3/fuzev.py`, FP16/QSA caches, GDN/PLE state definitions, Qwen4Exp
registration and the optimized vendor `checkpoint_transfer.py`. Actual Branch
identity (`dataclass(eq=False)`), class/field names, tensor shapes/dtypes,
negative PLE keys, active-window slices, coalesced layout order, complete QSA
pool rows and native reservation/COW arithmetic agree with the observer.

One fail-closed ownership gap was established with CPU fixtures: a valid-shaped
live PLE CPU ID buffer could alias the mutable transfer arena or another
branch's retained checkpoint slab and still produce a successful live digest
or comparison. Frozen-state inspection already rejected this alias, but live
inventory did not. Source `PLELayerState.alloc()` normally allocates independent
CPU ID storage; the counterexamples do not demonstrate an alias in a live run.
The transfer arena is rewritten by stash/restore, and PLE history writes into
an aliased buffer would mutate a retained slab. Live inventory now rejects
shared root storage before layout/digest reads, including offset views. This
checks metadata only; no extra copies or transfer calls are introduced.

At that earlier alias-audit commit, public signatures and record keys were unchanged.
Optimized kernels, current-stream fencing, tiling and existing memory guards
remain unchanged. The checkpoint-slab alias fixture uses two small recurrent slots;
normal storage passes before the alias is installed. The current follow-up starts
from unpublished lab commit `663bea2`, which already includes this alias fix and harness integration.

`last_history == 0` remains a necessary boundary check, not a substitute for
checking actual forward parameters: pinned `advance_recurrent_states()` also
sets it to zero for a one-token `recurrent_history=True` forward. The integration
hook independently checks both actual `model.forward` and `model.prefill` entries.
The passive observer still cannot establish this precondition from a digest alone.

## Fresh Reset Contract

The pinned source chain is `NativeLane.create()` -> `Cache.get_new_state()`
(`cache/cache.py`, `clear=True`, position zero) -> `GDNState.__init__()` ->
each registered layer's `clear(slot)`. `GDNState.reset()` only assigns position
zero, and `Cache.reset_states()` only resets the free-list; neither establishes
causal reset contents and neither is invoked by this observer.

`GDNLayerState.clear` zeros convolution and recurrent storage.
`PLELayerState.clear` zeros convolution but fills IDs with
`module.ple_embedding.eos_token_id`, not necessarily zero. Qwen4Exp reads that
sentinel as an integer; the observer requires an exactly representable int64.
`verify_reset` validates only the four registered **active** causal views from
the inventory table, under the same fence, lock, dtype/layout/alias checks and
bounded tiles. Zero comparisons are exact, not tolerant. No reset is performed,
no full tensor/table is cloned, and no freed slot is read. Unused KV and reserved
speculative history are explicitly not part of this check. Source references:
`exllamav3/cache/cache.py`, `modules/gated_delta_net.py`, `modules/ple.py`, and
`architecture/qwen4_exp.py` at the fixed engine commit above.

## Harness Integration

The opt-in qualification harness witnesses live digests immediately before
freezes, retains admitted reused pending-B/query/candidate checkpoints until
matching fresh boundaries, and attempts all tracked releases in `finally`.
The pending witness is released as soon as the fresh pending comparison ends,
before fresh C; the watched set never follows it into released storage.
All oracle reservations protect original AB. No borrowed tensor view survives
a forward/freeze/release boundary. Fresh-slot reset is inspected before any
oracle append, then `compare` covers consumed AB, pending B, ABC, and each tail.

The harness also binds actual `forward` **and** `prefill` CPU entry arguments to
active `NativeLane` chunks. IDs, page tables, cache lengths, live recurrent slots/positions,
cache/feature mode and identical image objects/metadata are checked before the
model body. Prepared CPU `recurrent_slots` and position advancement are checked
on return, without inspecting device copies. Unknown position overrides remain
unsupported. `NativeLane` page COW precedes argument construction in the actual source;
these hooks do not claim validation before that earlier GPU copy. Unexercised
entry paths, native kernels, queue/real-failure cleanup, real images, weight-byte
verification and full-window execution remain incomplete. CPU fixtures cannot
qualify the runtime. See [harness details and exact commands](native-qualification.md).

## Distinct Page Budgets

Preflight reports **serving** separately from **observer_coexistence**. Both
preserve all currently owned branches and account for partial-page copy-on-write,
native chunk reservations, and candidate batching. Serving releases candidate
batches normally. The comparison plan retains reused ABC and all its frozen
candidate branches while constructing an independent fresh ABC and serial tails.
With `pending_witness=True`, it also freezes a fork at resolved AB, accounts for
the extra shared partial page and COW during reused C, holds it through reused
candidates and fresh consumed/pending replay, then releases it before fresh C.
The two fresh AB appends have separate native reservation margins. The temporary
live parent plus borrowed checkpoint slot requires two initially free slots;
failure is scoped to observer coexistence, not ordinary serving.

Default `294912 / 256 = 1152` pages can accommodate a single 1024-page window;
two near-full ABC states need about 2048 pages before margins. Consequently a
short-AB/long-C full-window serving plan may fit while the observer plan reports
`memory_infeasible`. That failure is **not** evidence against general 262144-token
serving capability. Fresh replay retaining only consumed AB of length P has the
weaker lower bound `ceil(P/256) + 1024`; P > 32768 already exceeds this pool.

`page_estimate_feasible` is arithmetic only, not inference/memory qualification.
Shared RAM, frozen-head readiness, other slot requirements and extra checkpoint/scratch
bytes still need independent admission before any future run. No protected AB eviction,
cache enlargement, truncation, head change or trial allocation is performed.

## CPU Verification

Tests use tiny explicit fake classes/layouts, labelled `cpu_fake`, not native
modules or acquired data. The test file is `tests/test_native_state_observer.py`.

Historical verification on 2026-10-03: 56 observer tests and 73 qualification
regression tests passed together (**129 passed in 6.23 seconds**).

The isolated source-audit follow-up ran only the directly affected CPU fixtures:
26 passed, 38 deselected in 1.03 seconds. Eight new cases cross arena/checkpoint
owners, root/offset views, and digest/compare operations; 18 existing controls
check owned slabs, freeze/restore, causal fields, inventory and bounded tiles.
The focused selection was:

```sh
python3 -m pytest -q -p no:cacheprovider \
  --basetemp=$FUZEV_ROOT/tmp \
  tests/test_native_state_observer.py \
  -k 'live_id_state_cannot_borrow_checkpoint_transfer_storage or checkpoint_slab_is_authoritative_and_owned or freeze_restore_exact_digests_and_no_freed_slot_reads or actual_cpu_inventory_and_bounded_tiles_never_claim_native_proof or every_causal_component_is_compared or (inventory_and_lifecycle_fail_closed and aliased_storage)'
```

These focused tests do not establish native/GPU state correctness.

The current bounded follow-up adds reset sentinel/stale-state/tile controls,
pending COW/reservation/slot accounting, and harness forward/prefill integration
fixtures. Its complete scoped CPU verification is recorded in the harness
document linked above; the earlier audit totals are historical, not its results.

[Snapshot methods and environment](../README.md)
