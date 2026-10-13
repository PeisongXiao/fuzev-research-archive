# Native Stock Generation Control

The generation control calls the upstream-style ExLlamaV3 Generator on
procedural inputs and records accepted tokens, timing and cleanup.

## Callable Contract

`run_native_generation_control(generator, requests, ...)` accepts an **already
loaded, exclusively owned offline stock Generator**, not NativeService or an
async wrapper. There is no loader, launcher, head or forward interception. Stock
`Job`, `ComboSampler`, `enqueue`, `iterate` and `cancel` are used directly. Normal
stock fused sampling, checkpointing, defragmentation and streaming kernels are
left unchanged; no environment/profile flags are set by the callable.

Importing the module, or leaving `allow_generation=False`, imports neither Torch
nor the engine. Enabling requires `exclusive=True`. Loaded class bindings are
looked up in `sys.modules`; the lab run recorded the source revision
(`git rev-parse` when available, otherwise the package version).
Recorded tests used one, two or four requests per wave. Each `GenerationControlRequest` explicitly
supplies its name, literal rendered prompt, tuple of prompt token IDs, `add_bos`,
`encode_special_tokens`, maximum completion count, temperature, top-p, top-k and
seed. The loaded tokenizer must reproduce exactly those IDs. The harness does
not render a chat template or silently add/heal tokens. No images/embeddings,
filters, stop strings, repetition penalties or speculative decoding are used.
`ComboSampler` specifies min-p=0, neutral penalties, temperature before filtering
and adaptive sampling disabled; it does not use DefaultSampler's min-p=.08/T=.8.

Each request's prompt-plus-maximum-completion budget is capped at 262144;
there is no separate aggregate context cap. Conservative simultaneous pages are
`sum(ceil((prompt_tokens + max_new_tokens + 1) / 256))`, matching stock
`max_rq_tokens=None` and `Sequence.prepare`. No sharing/reuse is assumed for
admission. A page-boundary request may therefore need an extra page even when
its logical token total fits. The loaded cache may have 1..8 physical recurrent
slots; the lab runs used a maximum batch of four within the available slots. All physical slots must be idle; the helper never resizes either capacity.
At an existing 1152-page (294912-token) pool, four 70000-token request budgets
need 1096 pages and fit without extending any context. Four 73727-token budgets
need exactly 1152 pages. Larger sums still reject when their rounded page budget
does not fit. These are geometry examples, not native memory qualification.

Occupied pending/active/filter queues, referenced pages, borrowed recurrent
slots, contradictory ledgers or a stale cache owner serial reject before
enqueue. Existing unreferenced warm prompt/checkpoint caches remain available;
reuse is reported, not reset. PLE must already be `trellis_disk` or `fp16_disk`,
with handles, no resident tables and no pending/held staging. No payload is read
by admission and no staging is drained to hide a failure.

An explicit positive `additional_peak_bytes` and `estimate_basis` are mandatory,
covering future native transient allocations beyond already loaded resources,
including stock recurrent checkpoint growth and staging. An additional bounded
record/token reserve was charged in the lab runs. The recorded allocation
fraction was 0.85, with 15% shared-RAM headroom and swap-out guards.
An estimate is not a peak-memory proof; loader admission is outside this helper.

## Token Records And Cleanup

Accepted completion IDs are read incrementally from the original
`Job.sequences[0].sequence_ids`, which appends the sampled token **before** EOS
presentation suppression. Exact prompts, accepted completions including EOS,
streamed IDs, token digests and prompt/completion counts are returned separately.
Job/event counts and stopping IDs must agree. Normal EOS IDs come from the
already-loaded model config; length stops retain exactly the accepted budget.

Stock phase durations are reported raw with `clock: stock time.time`. In this
pin, `time_first_token` is recorded before the first generation forward/sample,
so stock `time_prefill` is **not observed TTFT**. `ttft_seconds` stays null;
negative/nonfinite wall durations are marked unavailable rather than repaired.
Outer monotonic timing includes Job/Sampler creation, enqueue, iteration and
incremental token observation through queue drain. Tokenizer verification,
final record assembly and helper cleanup are outside that outer scope. First/
last accepted-token observation offsets are iteration-level CPU boundaries,
not individual GPU timings. No extra CUDA synchronization is added.

Stock cached pages/tokens and before/after `get_cache_stats()` are retained;
`physical_forwarded_tokens` stays null. Iterations, logical prompt/output counts
and reused-token counts cannot establish physical work performed by the engine.

Optional `native_service_completion_token_ids` records an operator-supplied
counterpart: equality/difference and greedy/stochastic mode, never an attested
NativeService execution. Same rendering/IDs/settings must be arranged externally.
Stock Job uses `random.Random(seed)` and its original sampler RNG; NativeService
uses a CPU Torch Generator/full-population multinomial. Identical seeds do not
establish identical draws, outputs or tie behavior. Greedy equality is a useful
remaining runtime control, **not established by these CPU fixtures**.

Cancellation/deadline checks occur between synchronous stock calls; they cannot
interrupt an in-flight GPU operation. On failure the helper cancels/deallocates
only its own jobs, continues cleanup after a release error and closes its local
MemoryGuard. It never clears queues, resets pages/states or closes the borrowed
generator/model. Foreign jobs remain untouched. If ownership changes, cache
cleanup is unsafe and is skipped with an error note. Failed releases may require
external owner disposal; they never produce a successful result.

## CPU Evidence

Fixtures use scripted CPU IDs, fake pages/slots and fake stock-shaped classes.
They do not execute stock sampler kernels or model math. AST inspection of the
actual pinned Job/ComboSampler constructor keywords and Generator methods runs
without importing the source. Source-pin check tests use mocked loaded bindings
and Git responses; they are not native provenance or executable identity proof.
The maximum-window fixture exercises arithmetic/records only, not GPU serving.

Coverage includes 1/2/4 jobs, EOS-only and length output, explicit BOS/token IDs,
stream suppression, cache/timing honesty, stale/occupied ownership, PLE streaming,
capacity/page margins, swap/admission/cancellation/deadline failures, partial
enqueue, foreign intrusion, takeover, cleanup errors and malformed records.
Genuine native greedy equality, stochastic behavior/quality, actual loaded-model
identity, concurrency, phase timings and peak/swap-safe runtime performance
were not established by these fixtures.
This is a text-only control: it does not support an image baseline.

Historical initial run: **85 passed in 1.00 seconds**. The admission follow-up
to unpublished lab commit `200e885` passed **97 tests in 1.31 seconds**, including
physical-5/8-slot and per-request/aggregate-page geometry regressions.
These are CPU fixture results, not native/image/greedy-equality measurements.

From the snapshot directory:

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_native_generation_control.py
```

[Snapshot methods and environment](../README.md)
