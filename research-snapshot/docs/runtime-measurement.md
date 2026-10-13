# Research Performance Measurement

`python -m fuzev_research.runtime_measurement --config PLAN.json --output NEW_DIRECTORY
--allow-inference` reuses the existing HTTP response validators and bounded
performance pipelines.

Its config extends `HarnessConfig` with an absolute, SHA256-pinned terminal
correctness report for the same model, explicit actual engine utilization,
one or two warmup waves and at least three measured repetitions. Existing
correctness failures remain visible. `measurement-complete` never means that
correctness or quality passed; `runtime_qualified` is always false. The lab
recorded allocation fraction 0.85 and engine utilization 0.75 separately.

Generation-only, decision-only and mixed ABC modes use the existing candidate
rendering, head, token-ID records, prefix ownership and cleanup. Request
JSONL retains individual responses and elapsed intervals; measured summaries
exclude the separately recorded warmup waves. Engine phase timing is not
client-delivered TTFT because these endpoints do not stream. Physical forward
positions remain null when no genuine worker/owner observation exists. Neither
logical tokens nor submitted-minus-cached counts manufacture that observation.

This entry point measures fused paths only. The stock native `Generator` and
fusion-disabled same-fork vLLM controls remain separate, sequential owners of
the same acquired payload. A fused generation-only endpoint is not an untouched
control; saved results using other bitrates, MTP or resources are not substitutes.
The native full-window feasibility case remains separate, with total accepted
A+B+C+candidate-tail input bounded by 262144 and actual fit/failure recorded.

CPU fixtures validate scheduling, admission, failed-evidence retention and
cleanup only. They do not validate GPU math, throughput or memory feasibility.

## Forward Observations

Counted profiles used separate native unpublished lab commit `5e1ec65` and core unpublished lab commit `d6f1ee8`
source builds. Their default-off counters were enabled only after
constructing/warming an exclusive owner and before a wave.
No kernels or native extension were rebuilt. Existing completion/copy barriers
acknowledge unpadded target input positions without new synchronization.

The vLLM profile selects `fuzev_research.runtime_performance_counts.PerformanceObservationWorkerExtension`.
The trusted operator calls named `fuzev_performance_counts(enable=True)` once
while idle, then read-only snapshots before/after each drained wave. This is not
an HTTP route or a setting exposed by generation/decision requests. Native stock
controls may supply an exact `expected_engine_commit` for the counted source;
the default remains the original source, and actual clean module roots/commit
are still verified and recorded. This does not claim pristine upstream or a
new compiled-native revision.

`counter_delta` accepts only monotonic, internally consistent snapshots from
the same owner scope. Pending work, failed forwards or completion errors leave
the physical total null and retain the acknowledged lower bound. Counts are
owner-wide for an exclusive wave, not per-request attribution, accepted tokens,
kernel launches, FLOPs or proof of causal state correctness. Actual GPU counter
coverage and overhead were unverified on matched inputs, so no forwarded-position
rate is reported.

[Snapshot methods and environment](../README.md)
