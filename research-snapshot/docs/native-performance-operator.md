# Native Performance Operator

The research operator compares generation, decisions and mixed workloads
using a fixed head and separately constructed control and attached engines.

## CLI

The original launcher and complete configuration are not distributed. The
module accepted `--config`, `--output`, `--allow-inference` and `--exclusive`.
Its control and attached-engine measurements were sequential.

`OperatorConfig` rejects extra fields. The lab run recorded:

- `expected_recipes_commit`: the observed source revision (`git rev-parse` when
  available, otherwise the package version).
- `expected_engine_commit`: unpublished lab commit
  `5e1ec65733f76c3b5e2f796c50daabb4d345aba2`.
- `native_tree`: absolute path to the counter-enabled native source directory.
- `native_extension`: absolute path to the recorded native extension build, for example
  `cross-cell/exl3-native-reference-pair/native/exllamav3_ext.cpython-312-aarch64-linux-gnu.so`
  under the project root. The actual resolved import and bytes must match.
- `workloads`: `{ "path": absolute_path, "sha256": actual_sha256 }` for
  `runs/native-real/qualification/native4-performance-workloads.json`.
- `plan`: the same pin shape for
  `reports/2026-10-10-kev-text-evaluation/frozen-plan-v1.json`.
- `quality`: the same pin shape for
  `reports/2026-10-10-kev-text-evaluation/native-results-v1/result.json`.
- `correctness_evidence`: the same pin shape for a terminal original native 4.05
  correctness report, including a failed report. Its status remains visible.
- `additional_peak_bytes` and `estimate_basis`: a positive, honest transient
  memory allowance for the future stock sampler/checkpoint growth/staging.
- Optional `memory_fraction` (default `None`, using engine settings) and
  `timeout_seconds` (default/max `3600`). The lab run used allocation fraction
  0.85 and disabled host-RAM and swap admission.
  An estimate is not peak-memory qualification.
  Fused deadlines signal cancellation at existing owner boundaries; they cannot
  interrupt an already running CUDA operation. Cleanup waits for the owner.

All pins are actual local bytes/commits. Plan,
quality, workloads and correctness pins are checked again at the end. Source8
stays at unpublished lab commit `9728b98b205d4f9af8ecac4e28ee678ba5ab758b`; the frozen reference head stays at
`f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c`.
The head is raw T1, `uncalibrated:T1`, never CAL. Its immutable reference feature
identity is recorded separately from the new actual performance runtime.

## Matrix And Ownership

The three existing code/prose/structured procedural workloads each produce 2K
and 32K variants: six total. Only repeated literal `ALPHA BETA GAMMA.\n` context
is prepended to the original user turn; instructions/options are retained.
The loaded native tokenizer chooses the smallest repetition count whose rendered
generation prompt meets the nominal tier, with at most 256 tokens overshoot.
Actual IDs, rendered strings, counts, query/tail hashes and conservative maximum
ABC-plus-tail length are saved in `workloads-rendered.json`. No truncation occurs.

Each of the six cases runs 1 and 4 external streams, with one warmup wave and
three measured waves, for stock generation, fused generation-only, decision-only
and mixed. Choice/Noul/ordinal rotate over the three measured waves and stream
indices. Sampling remains temperature 1, top-p .95, top-k 20, seed 17 and maximum
128 accepted tokens. Early EOS contributes its actual count. Stock and native
service use different RNG/sampling implementations, so neither stochastic byte
equality nor identical stochastic trajectories are claimed. No extra greedy
probe is launched by this operator.

One `NativeBackend` loads one native 4.05 language backbone and one 294912-token
cache, with four physical hybrid slots and candidate capacity four, retaining
the prefill optimizations. Stock `max_batch_size` remains four. No additional
persistent slot reservation or change to four-stream scheduling is needed.
The initial empty `NativeLane` is closed and detached before an original stock
`Generator` is constructed. All stock waves finish first. The helper requires
idle queues/pages/recurrent slots and genuine streaming PLE. After confirming
idle ownership, the stock filter pool is closed, its owner serial is retired and
its object is released before a fresh `NativeLane` and the canonical CUDA research head
are constructed. There is no concurrent stock/`NativeLane` cache access or second
resident backbone. No vision component is loaded.

Fused streams are asynchronous external pipelines queued to one owner thread;
`NativeService` completes each synchronous call before the next call. This is
not four simultaneous generation forwards. Equal-length candidate tails retain
their existing batching (at most four), independent branches and cleanup.
Mixed mode uses each generation's exact accepted AB checkpoint and releases it
after the decision. Generation prompt IDs and retained accepted IDs are checked
against the same rendered stock prompt. Prefixes must be absent at every wave
bookend, with all `NativeLane` pages/slots accounted for.
Actual `NativeLane.freeze()` stashes the checkpoint and returns its live slot; four
retained prefixes do not themselves occupy four live slots in this source.
`NativeService.decide()` also freezes the query before candidate forks. The
four-slot CPU fixture confirms that four retained prefixes and the frozen query
leave all four slots available for independent candidates. Native/backend,
stock and fresh fused `NativeLane` chunk sizes all remained 8192.

## Counts And Timing

The native counter is explicit per-model opt-in, after construction and before
the first wave. Stock generation uses its original completion barrier. Local
research subclasses only capture/acknowledge counter marks around the original
`NativeService._sample` and `ResearchHead.score` CPU-return boundaries. They
preserve the exact original calculations/outputs, add no CUDA reads/sync/timers,
and never wrap or monkeypatch `model.forward`, the stock Generator, or Source8.

`counter_delta` supplies genuine whole-exclusive-wave input-position totals;
padding is excluded. Failed, pending or unknown work leaves
`physical_forwarded_tokens` and its rate null, with the acknowledged lower bound
and all submission/failure coverage retained. Shared physical work is never
attributed to individual requests. Service arithmetic, logical AB reuse and
replayed tokens stay separate from the actual counter. A failed wave is retained
whole and aborts the remaining matrix; there is no success-only throughput.

Existing `summarize_samples`/`generation_phases` produce CPU-observed latency,
completion and logical-prefix rates. Fused latency includes the owner queue;
wave wall time includes prefix cleanup, not JSON writes between waves. Group
summaries aggregate exactly the three measured waves, excluding warmup. Stock
first/last accepted-token offsets and raw phase timing are retained separately:
stock `time_prefill` is not TTFT and stays unconverted. Stock per-job latency is
the last accepted-token observation relative to its common batch start. These
intervals are not GPU-kernel timings or a comparable HTTP/network benchmark.

Allocator/host/swap observations are outside requests, with no added CUDA
synchronization. Historical peak values are labelled rather than reset. The
recorded native extension build and the prefill optimizations remain the
original build; a new Python counter commit is not a C++ rebuild or execution proof.

## Verification

Targeted fixtures use actual Torch on CPU, fake loaded-owner metadata and the
pure-Python native counter class. They cover exact rendering/tier bounds,
unaltered sampling, original CPU boundaries, failed/pending counts, cache
admission, default-off opt-in, all 48 matrix groups and single-owner
sequential execution. They do not load/infer a model or qualify CUDA/performance.

Final focused proof: **21 passed** on Torch 2.13.0+cu130, CUDA
uninitialized before and after. This includes the actual stock adapter's
delegation to the unchanged helper with the genuine future native source pin.
A separate import check confirmed importing the operator loads neither Torch
nor either engine. These are CPU fixtures,
not native execution or performance evidence.
The recorded slot follow-up AST-executed actual `NativeLane`/cache allocation methods with
CPU state fixtures; all four slots and pages return after four retained prefixes,
one query and four candidate continuations. A source-level geometry fixture
checks physical capacity four, active/candidate limits four, matched 8192 chunks
and the actual service's query-freeze-before-candidate-fork ordering. Four
slots support that lifecycle.

[Snapshot methods and environment](../README.md)
