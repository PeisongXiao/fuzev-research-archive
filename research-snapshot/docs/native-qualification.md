# Native Correctness Harness

The harness compares accepted tokens, branch state and resource cleanup
using procedural cases. This method note includes synthetic CPU coverage.

## Scope And Gaps

`fuzev_research.native_qualification.run_qualification` operates on one already admitted
`NativeService`, synchronously on its existing owner. It refuses nonempty
service/feature/image caches or active branches. It neither constructs nor
closes a backend, starts a server, accesses an endpoint, nor launches workers.
Run it on an exclusive offline service. The caller remains responsible for `service.close()` followed by
`backend.close()`, including on errors. There is deliberately no loading CLI.

Every report has `qualified: false`. Successful checks yield
`execution_status: passed`, **not** completed qualification. Missing checks are
reported explicitly, never substituted with token counts or synthetic results:

- Default partial mode does not inspect native state. Explicit
  `observe_native_state: true` borrows the existing pinned native-state observer;
  its complete-state check cannot pass from CPU fakes or logical records.
  Actual CPU entry metadata is checked against active `NativeLane` chunks for both
  `model.forward` and `model.prefill`. Unexercised paths remain incomplete.
- Admitted state mode compares consumed AB, pending-B resolution, ABC, and
  candidate states, and checks fresh-slot active recurrent reset contents.
  This does not prove CUDA launch arguments, kernel behavior, or that unused KV
  and speculative history storage are zero (that is not the reset contract).
- Current cache geometry is checked. Successful observer records validate actual
  tensor dtypes, not just metadata; default partial mode does not. Weight bytes
  are never inspected. Model provenance comes from acquisition metadata and the
  loaded config, not a newly computed weight checksum.
- Fault injection is after completed operations, not a real kernel failure,
  physical OOM, HTTP cancellation, scheduler, or queue-drain qualification.
- Real-image controls and a full 262144-token case are optional and separately
  reported. Absence cannot pass either check. No performance, transfer accuracy,
  held-out quality, publication, or multi-backend qualification is implied.

## Exact Replay

Each supplied case generates A+B once, verifies the accepted IDs, usage,
consumed checkpoint, and final unconsumed token (including hidden EOS and
presentation-trimmed stop tokens). Four decision requests independently append
C through `NativeService.decide`. Query/candidate features are cloned on CPU
before engine scratch can change. Response identities, candidate ordering,
finite scores, normalized probabilities, and selected IDs are validated.
Every successful decision, including recovery/repeat calls, also verifies the
exact pending-token append, consumed candidate tails, layout, feature tap,
image bindings, and all integer execution counters against observed backend
boundaries. Boolean counters are not accepted as integers.

For each decision, the oracle creates an empty `NativeLane` and replays the
**same accepted token IDs**, not standalone messages, re-tokenized visible B,
or a second generated completion. It then appends exactly the recorded and
independently checked C encoding and evaluates independent serial candidate
forks with the same frozen scorer. Original image binding instances are kept
for replay, including the distinction between the pending B token's images
and a new image introduced in C. The four decisions are repeated in reverse
order to detect interference. Parent boundaries and resource ledgers must
remain unchanged.

Comparisons cover owned query features, all candidate features, logits, raw
probabilities, and calibrated probabilities. Defaults and maximum permitted
tolerances are `atol=1e-5`, `rtol=1e-4`; nonfinite values, empty/mismatched shapes,
invalid limits, and numerical disagreement fail closed. Do not loosen these
limits based on external/held-out examples. Disagreement is evidence to
investigate, not permission to retune the head or silently change semantics.

The harness injects cancellation and explicit failure after generation forward
and decision forward/candidate/head boundaries, checks branch/page/slot/image cleanup,
and performs a subsequent decision to check recovery. Temporary observation
and injection wrappers are restored even on failure. The frozen CPU FP32
head's weights, calibration temperatures, and identity must remain unchanged.

## Opt-In State Observation

State mode uses `NativeStateObserver.digest`, `compare`, and `preflight` without
modifying the engine or service. The observer adds a read-only `verify_reset`
check and an optional pending-witness preflight. Every freeze is preceded by a live
digest under the observer's existing owner lock and completed-stream fence,
then checked against its authoritative frozen slab. Fork restores, protected AB,
and retained siblings are checked with byte-exact digests at consumed AB,
pending-B resolution, ABC, and candidate boundaries. Numeric state comparisons
use the declared tight tolerances; integer state, dtypes, and positions stay
exact. The observer remains responsible for complete pinned registrations,
causal field coverage, bounded tiles, and its documented scratch/history
exclusions. No new whole-cache gather, checkpoint clone, or PLE-table copy is
introduced. See [the observer contract](native-state-observer.md).

Before creating the reused query, a hook passes the **actual** final C encoding
and independently encoded tails to the observer's page-lifetime preflight.
The committed C/tails must subsequently match that plan. If coexistence fits,
normal service release hooks freeze and temporarily retain the reused query
and all completed candidates. Immediately after resolving pending B, a witnessed
fork is frozen only when coexistence is admitted. The live-parent stash and
child freeze have separate admission checks; this requires a second live slot
temporarily. Fresh replay first verifies reset, consumes exact consumed AB,
then its pending token, and compares the retained pending state elementwise.
That extra pending branch is released **before** fresh C; it is removed from
the watched set before release and never inspected again. C and serial tails
then reach their matching comparison boundaries. Original image objects
remain strongly referenced for replay. Every oracle `_append` supplies
`protected=prefix.id`, including consumed AB, pending B, C, and each candidate:
a late `_reserve` failure must not evict AB. Every newly forked branch is tracked
before observation can throw. Additional lifetimes end in `finally` on success,
failure, or cancellation, and all releases are attempted even if one fails.

`model.forward` **and `model.prefill`** are narrowly wrapped during state-enabled cases to reject any
actual `recurrent_history` value except absent or explicit `False`, both before
and after the call. The native observer constructor verifies the unpublished lab commit `ba60611` source
pin: GDN/PLE use `params.get("recurrent_history", False)`, while
`advance_recurrent_states` records `seqlen - 1` for history-enabled forwards.
Thus **one-token speculative execution also leaves `last_history == 0`**; that
field or a digest alone cannot attest non-speculation. `prefill` directly calls
its own implementation and does not pass through `forward`; `forward_features`
does call `forward`. The flag record holds
completed entry/return observations, rejected calls, and whether source default
semantics were verified. CPU fake flag observations explicitly lack native
source verification.

Entry hooks run inside `NativeLane`'s existing RLock and require an active
`append`/`append_features_many` plan. They check every chunk's exact CPU int64
input IDs, CPU int32 block table and cache lengths, actual recurrent object
identity and live slots/positions, cache identity, last-token/feature mode,
exclusive writable pages, and identical image embedding objects and identity metadata. A bounded
`backend.parameters` witness ties image arguments to the original input
embedding instances, not merely equal image digests. Unknown position options
(including mRoPE for this linear-position scope), missing/extra entry calls,
GPU metadata tensors, or unwitnessed parameter dictionaries are rejected.
Return checks verify recurrent advancement and the preparer's actual CPU
`recurrent_slots` tensor; pinned `get_for_device` keeps that CPU source intact.
Only aggregate hashes/counts leave the hook, not token/page-map dumps or tensors.
Hooks and transient parameter/image references are restored/cleared in `finally`.

The comparison boundary is **before the model body**, after `NativeLane`'s page
allocation/COW. The pinned `NativeLane` constructs its actual arguments only after
`prepare_write`, so this does not claim validation before that earlier page
copy. Device-cache copies and kernel launch arguments are not inspected. The
normal service chunks input before calling `NativeLane`, so it may never exercise
`prefill`; its zero call count keeps the combined entry-path check incomplete.
No extra native call is fabricated merely to fill that coverage count.

`verify_reset` reads only owned active causal views after fresh `lane.create()`:
GDN recurrent/convolution and PLE convolution must be exactly zero; PLE ID
history must equal that registered layer's EOS sentinel. `Cache.get_new_state`
uses `clear=True`; `GDNState.reset()` only resets position and is not invoked as
a substitute. Registry, dtype, position, source, tiling, alias, and memory checks
remain in force. No fresh-slot contents are assumed from metadata alone.

An unsupported initial inventory, unavailable observer API, or unsupported or
unwitnessed frozen checkpoint produces `incomplete`, not a synthetic pass.
A previously observed inventory becoming invalid, changed parent/sibling
digest, or numerical disagreement fails. Unsupported initial state skips the
complete-state oracle. Successful comparisons are reported only after actual
observer calls return; every top-level and observer record stays
`qualified: false`. A native failure
inside state construction before a Branch is returned is outside `NativeLane`'s
recoverable ownership API: the harness reports a ledger failure rather than
rewriting the freelist. It only guarantees release attempts for returned/tracked
branches; genuine kernel/allocator cleanup is not established by CPU fixtures.

## Invocation

After the frozen mixed head is ready, an operator can construct the service
using the existing `NativeBackend` / `build_service` admission path in a
separate offline run. The harness itself never does so.
Pass a `QualificationConfig` and `allow_inference=True` to
`run_qualification(service, config, allow_inference=True)`. Without that exact
opt-in it refuses even to inspect the service. `cpu_fake=True` is test-only:
its records are labelled CPU fake and cannot pass the full-window check.

Configuration fields:

- `cases`: 1-8 uniquely named cases. Each includes `name`, `input_origin`,
  `input_role: operator-non-heldout`, a normal `GenerationRequest`, and exactly
  four existing `DecisionSpec` objects (`primitive`, `question`, `candidates`,
  optional `additional_context` and `image`). Include cases that actually reach
  EOS, a presentation stop, and a length stop; missing boundaries remain visible.
  Choice, Noul, and ordinal coverage is separately reported, never inferred.
- `image_controls`: optional standalone `ServiceDecisionRequest` with a real
  trusted acquired PNG, another real PNG as `mismatched_image`, and `expected_id`.
  Each also needs `name`, `input_origin`, and `input_role`. Identical RGB digests
  are rejected. Real, blind, mismatched, and repeated-real scores and image
  records are reported, including absent numerical sensitivity. Sensitivity
  measurements do not claim image quality.
- `atol`, `rtol`: only equal or tighter than the defaults.
- `observe_native_state`: strict boolean, default `false`. This does not replace
  `allow_inference=True`; both opt-ins are required for state-enabled inference.
- `full_window`: optional `case`, positive integer `additional_peak_bytes`, and
  `estimate_basis`. `run_full_window` is a separate explicit switch, default off.

Inputs must exclude held-out data. There is no built-in procedural
workload, dataset reader, tuning loop, download, padding, or truncation. Origin
is operator attestation, not automatic dataset-membership verification. The
report contains hashes/counts of accepted input, not full prompts or features.
Callers may serialize the returned JSON-safe dict to their temporary report
directory; the harness does not choose or write an output path.

## Memory And Full Window

Native `context_limit` and reported context must both remain **262144**. Actual
chunk size and cache geometry must agree with the loaded runtime metadata;
candidate geometry remains governed by the unchanged service. Actual imported
recipes/native source revisions were recorded (`git rev-parse` when available,
otherwise the package version). The
loaded model directory/revision and frozen head/calibration pins are checked
against their existing records before work and checked again at cleanup.

The lab used a 0.85 allocation fraction, MemoryGuard and admission checks.
CPU head copies were limited to 64 MiB. Observer estimates included the
128 MiB native staging arena and the branch checkpoint bytes; the observer
performed no trial arena or snapshot allocation.

The separate `full_window_feasibility` helper does arithmetic only. Without a
frozen head or explicit enablement it does not proceed. An operator's conservative
additional-peak estimate reserved 15% shared RAM in the recorded lab runs;
insufficient estimated capacity produced `memory_infeasible`, with no trial allocation or reduced window.
An accepted estimate is not measured feasibility. Only a subsequently completed
case whose actual longest ABC plus candidate tail is exactly 262144 may report
that full-window check passed. Early EOS or a shorter supplied case cannot be
relabelled full-window. State-enabled page checks distinguish two scopes:

- `state_preflight.serving`: original protected AB plus the unchanged serving
  query/candidate lifetimes. Short AB plus long C can fit at the full window.
- `state_preflight.observer_coexistence`: all those protected branches plus
  retained reused query/all tails and a separate exact-token fresh replay.
  Two independent near-full ABC states require at least 2048 pages; the default
  294912-token cache has only 1152. This makes that comparison infeasible, **not**
  general 262144-token serving. The simulation includes all retained candidates,
  page sharing/COW, and native reservation margins, not just token totals.
  The admitted pending witness holds its shared pages across reused C and all
  candidates, adds COW at C when needed, and releases after fresh pending-B
  comparison. Its temporary second slot and the fresh consumed/pending split
  are included, so a marginal former comparison plan can now be refused.

Serving preflight refusal occurs before creating its query. If only observer
coexistence is `memory_infeasible`, serving is still exercised without extra
retention and the oracle performs no forwards. A late MemoryError during fresh
replay has scope `observer_oracle_not_serving`; full-window serving and observer
results remain separate. Other state-mode full-window MemoryErrors have an
explicit unclassified phase and remain incomplete rather than being labelled
general window infeasibility. Page arithmetic is not proof of shared RAM,
recurrent-slot availability, checkpoint admission, or inference feasibility.
The opt-in preflight follows the exact consumed-AB/pending-B append split.
Runtime admission can still refuse, with AB protected and no retry/eviction to
get a pass. The observer API's default `pending_witness=False` preserves its
older page-only plan for callers that do not retain that intermediate state.

The harness will not evict its protected AB, enlarge the cache, or reduce the
context to manufacture a pass. Default partial-mode MemoryErrors retain their
existing reporting. During a failed allocation attempt, `allocated: null` means
the allocation extent is unknown, not that no allocations occurred.
No full-window allocation was attempted.

## CPU Verification

Tests use fake CPU `NativeLane` objects and tiny frozen fake heads to exercise orchestration
and adversarial failures. They do not import the native engine, train a head,
or constitute model, full-state, or image correctness evidence.

The new `tests/test_native_state_integration.py` uses the actual observer over
tiny registered CPU fixtures, not a fake observer claiming native proof. It
exercises borrowed branch cleanup, unsupported/unwitnessed observations,
parent mutation, grouped candidate ordering, both preflight refusal scopes,
every protected oracle reservation including late allocation failure, and
actual fake forward flag rejection even for the first one-token append.
Additional fixtures exercise both forward and prefill with malformed CPU entry
arguments, exact image-object identity, pending elementwise disagreement and
early release, tight page/slot budgets, and stale/nonzero reset values.

From the snapshot directory:

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_native_state_integration.py \
  tests/test_native_qualification.py \
  tests/test_native_state_observer.py
```

Historical result on 2026-10-03: **208 passed in 7.78 seconds** (59 integration,
73 partial qualification, 76 observer). This includes both pending-witness modes
for the 1152-page full-window arithmetic fixture, not a full-window native run.

[Snapshot methods and environment](../README.md)
