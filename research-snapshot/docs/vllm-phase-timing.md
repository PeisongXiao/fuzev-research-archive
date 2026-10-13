# vLLM Generation Phase Timing

Historical pre-1.0 research service, superseded by `fuzev-vllm-plugin` and
`fuzev-exllamav3-plugin`. It does not run against the published engine forks.
The released servers have no `--service-config`, `phase_timing` or server-local
image path.

The research adapter derives generation timing from vLLM timestamps. CPU
fixtures verify interval arithmetic and unavailable-data handling.

## Response Contract

Successful text/chat generation responses add one top-level `phase_timing`
object. Older adapters may omit it. Typed readout/decision responses and the
existing `execution` record are unchanged. The snapshot is owned while reading
the successful terminal output, before the iterator resumes, rather than after
readout or by subtracting client elapsed times.

The following is a **synthetic schema example**, not a measurement:

```json
{
  "schema_version": 1,
  "source": "vllm-engine-core-request-stats",
  "status": "available",
  "reason": null,
  "scope": "unpreempted engine-core request intervals; prefill includes first token; decode excludes first token; not GPU kernel time",
  "prefill_seconds": 3.0,
  "decode_seconds": 4.0,
  "queue_seconds": 2.0,
  "prompt_tokens": 4,
  "completion_tokens": 3,
  "decode_tokens": 2,
  "reported_cached_prompt_tokens": 2,
  "num_preemptions": 0
}
```

All keys remain present in unavailable observations: `status="unavailable"`,
`reason` is an explicit string, and **all three interval fields are null**.
`schema_version`, `source`, and `scope` remain unchanged. Counts come from the
actual submitted prompt and API-accepted completion, not estimated engine work:

- `prompt_tokens = len(prompt_token_ids)` after any existing image expansion.
- `completion_tokens = len(completion_token_ids)`, preserving every accepted ID.
- `decode_tokens = completion_tokens - 1`, excluding the first generated token.
- `reported_cached_prompt_tokens` is the existing validated record's
  `num_cached_tokens`: an exact integer from zero through `prompt_tokens`, or
  null when unknown. It is not a physical-work or hybrid-state reuse proof.
- `num_preemptions` is an exact nonnegative engine integer, or null if missing,
  malformed, or inaccessible. Available observations require exactly zero.

Validated counters can remain populated in an otherwise unavailable object;
invalid values are never forwarded as NaN, infinity, bool, float, string, or a
negative integer. The helper defensively nulls invalid cached counts. Existing
record validation rejects a malformed public `num_cached_tokens` before the
helper runs.

No rates are emitted. A one-token completion has `decode_tokens=0` and may have
`decode_seconds=0`; it cannot support a decode rate. A consumer must not treat
an absent/unavailable generation observation as a zero-duration sample or claim
a complete group rate from only the available subset. These are per-request
elapsed intervals, not concurrent aggregate wall-clock throughput. Logical
prompt and cache counts remain separate for downstream policy; the adapter does
not infer uncached compute from missing cache evidence.

## Reviewed Upstream Semantics

The source reviewed was vLLM unpublished lab commit `53a698ea9a62c6c6889df113f1bd73356fc618a7`:

- `vllm/v1/metrics/stats.py`, `RequestStateStats`, explicitly separates frontend
  wall-clock `arrival_time` from the engine-core monotonic endpoints
  `queued_ts`, `scheduled_ts`, `first_token_ts`, and `last_token_ts`.
- `IterationStats.update_from_events` records queue events, the first scheduled
  event (later scheduled events are ignored), and a preemption count.
  `update_from_output` counts `new_token_ids` and records first/last token times
  using the batch's engine-core timestamp. These are host event timestamps, not
  per-token CUDA events or isolated kernel execution times.
- `IterationStats.update_from_finished_request` defines queue as
  `scheduled_ts - queued_ts`, prefill as `first_token_ts - scheduled_ts`, and
  decode as `last_token_ts - first_token_ts`. Its mean output-token time divides
  decode by `num_generation_tokens - 1`, excluding the first token. The adapter
  uses these same interval endpoints, but requires that count to match the
  accepted API IDs and emits no rate.
- `vllm/v1/engine/__init__.py`, `EngineCoreEvent` and `EngineCoreOutputs`, obtain
  timestamps from `time.monotonic`. They must not be compared with frontend or
  client timestamps. `arrival_time` and frontend `first_token_latency` are never
  read by the adapter's phase helper.
- `vllm/v1/engine/output_processor.py`, `RequestState`, creates stats only when
  `log_stats` is enabled. `make_request_output` suppresses intermediate outputs
  for `FINAL_ONLY`; `_new_request_output` still attaches `metrics=self.stats` to
  the final `RequestOutput`. No switch to `CUMULATIVE` is required or made.
- `vllm/v1/engine/async_llm.py`, `AsyncLLM.from_vllm_config`, defaults
  `disable_log_stats=False`. The adapter does not change this configuration or
  enable stats on an attached engine; disabled/missing stats stay unavailable.

Prefill includes the first token, chunked-prefill gaps, and whatever work or
waiting occurs between scheduling and first core output. Queue begins at the
core queue event, not frontend request arrival. These phases exclude frontend
tokenization/image preprocessing and response delivery; they are not GPU-only
durations. Upstream intervals can include preemption/recomputation/rescheduling.
The adapter deliberately rejects **every request with `num_preemptions > 0`**
for this clean-phase observation, rather than presenting those spans as nominal.

Core counts are updated before detokenization/stop handling. The pinned
detokenizer normally retains a skipped stop token ID while trimming its display
text, but no count agreement is assumed across API/runtime changes. An EOS or
other engine/API count disagreement yields unavailable timing; no EOS is added,
removed, or inferred. Sampling parameters, `FINAL_ONLY`, accepted IDs, usage,
text, finish/stop reasons, AB retention/hash, readouts, worker forward stats, admission,
and abort/drain behavior remain unchanged.

## Validation And Reasons

Required timestamps are built-in int/float scalars, finite and strictly positive;
zero is an uninitialized sentinel, not a valid endpoint. Order must be
`queued <= scheduled < first <= last`. Queue duration zero is allowed. Prefill
must be positive. Multiple accepted tokens require positive decode duration;
one token requires coincident first/last endpoints. No wall-clock fallback,
client subtraction, GPU fence, tensor copy, kernel, or scheduler change is used.

| Reason | Meaning |
| --- | --- |
| `missing_request_stats` | Final output has no metrics or metrics is null. |
| `invalid_num_preemptions` | Missing, non-integer, bool, or negative count. |
| `preempted_request` | Positive validated preemption count; all intervals withheld. |
| `invalid_num_generation_tokens` | Missing, non-integer, bool, or nonpositive count. |
| `generation_token_count_mismatch` | Engine count differs from accepted API IDs. |
| `invalid_queued_ts`, `invalid_scheduled_ts`, `invalid_first_token_ts`, `invalid_last_token_ts` | Missing, wrong-type, nonfinite, zero, or negative timestamp. |
| `reversed_timestamps` | Endpoints are not ordered. |
| `nonpositive_prefill_interval` | Scheduling and first token coincide. |
| `nonpositive_decode_interval` | Multiple tokens but no decode elapsed time. |
| `single_token_decode_interval` | One token but distinct first/last endpoints. |
| `invalid_cached_prompt_tokens` | Defensive helper check failed; reported count stays null. |
| `invalid_is_corrupted`, `corrupted_request` | If supplied, upstream corruption flag must be a bool and false. |
| `malformed_request_stats` | Optional stats access/conversion raised; generation remains usable. |

Only the first validation failure is reported. No exception from reading these
optional stats is allowed to fail otherwise valid generation. Existing output
identity, success, token, record, and lifecycle checks still apply normally.

## CPU Verification

`tests/test_vllm_phase_timing.py` covers the uniform schema, first-token exclusion,
wall-clock independence, malformed/missing/zero/reversed/nonfinite data, counter
sanitization, preemption exclusion, single-token behavior, snapshot ownership,
no partial-stream inheritance, unchanged text/chat results and accepted AB,
text-stop/count disagreement, typed-readout exclusion, and HTTP serialization.

Opt-in fixtures use actual pinned `RequestStateStats`, `IterationStats`,
`EngineCoreEvent`, `EngineCoreOutput`, `RequestState`, `RequestOutput`, and
`SamplingParams`, including final-only output construction with stats enabled
and disabled. Their events/timestamps and completions are synthetic. They test
API compatibility, not actual model execution, EOS behavior, or measured phases.

Verification in the test source: the original three vLLM suites passed
**154 CPU tests**; the new timing suite passed **131**, including ten opt-in
real-API cases. The combined run passed **285 tests** in 8.22 seconds. Warnings were the existing upstream Starlette
and Torch JIT deprecations plus missing generated vLLM version metadata, not
engine/model execution failures.

Run the vLLM suites in a CPU environment with vLLM installed:

```sh
CUDA_VISIBLE_DEVICES="" FUZEV_VLLM_API_INSPECTION=1 python -m pytest -q \
  tests/test_vllm_service.py \
  tests/test_vllm_images.py \
  tests/test_vllm_image_api.py \
  tests/test_vllm_phase_timing.py
```

[Snapshot methods and environment](../README.md)
