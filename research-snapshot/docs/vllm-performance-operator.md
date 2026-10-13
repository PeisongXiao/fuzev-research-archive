# EXL3 Performance Operator (historical)

The EXL3 research operator compares a head-disabled control with generation,
decision and mixed workloads on the same fork.

The original launcher and exact configuration are not distributed. The
module accepted `--config`, `--output` and `--allow-inference`.

## Required Inputs

The historical operator consumed a configuration JSON. Every evidence pin is
`{"path": "/absolute/path", "sha256": "64 lowercase hex characters"}`.

- `input_role`: `operator-non-heldout`.
- `recipe_commit`: the observed source revision (`git rev-parse` when available,
  otherwise the package version).
- `engine_config`: the recorded EXL3 engine JSON with `.75`, no MTP, complete
  index/same-file payload view, all existing audited resources/PLE/image policy,
  `worker_extension_cls` equal to
  `fuzev_research.runtime_performance_counts.PerformanceObservationWorkerExtension`,
  and request statistics enabled (`disable_log_stats: false`, or its default).
- `binding_receipt`: a genuine prepared core/version/binary record for
  unpublished lab commit `d6f1ee810d3b4b3c4dffed2e39b6d2246c089712`, not the old unpublished lab commit `62fc998` binding.
- `image_build_receipt`: the genuine derived-image evidence pin when needed.
- `workloads`: predeclared JSON array of three code/prose/structured
  `Workload` templates, tagged `native-4.05`. The operator changes only the model
  alias to `vllm-exl3-4.05` and applies explicitly declared optional text padding.
- `correctness_evidence`: terminal evidence for `vllm-exl3-4.05`. An unfinished
  full panel is rejected; a failed report stays visibly failed in the result.
- `seed`: the templates' fixed seed (default `0`); it is checked, not replaced.
- `padding_text`: optional declared repeated procedural text. Without it,
  operator inputs must already meet their declared tier. `timeout_seconds`
  defaults to `600`.

Generation must remain temperature `1`, top-p `.95`, top-k `20`, max `128`, with
the fixed seed and no presentation stop. Templates retain all three primitives.
The fixed matrix consumes all three templates at nominal 2048/32768-token tiers,
streams 1/4, one excluded warmup and three measured waves. Padding never truncates
or changes decisions; records disclose actual generation/base token counts and
hashes, which can exceed the nominal tier by a padding increment. Control/fused
rendered records must match exactly. This stage is text-only; it adds no image,
heldout quality, native full-window or posthoc model-selection experiment.

## Execution And Evidence

The recorded offline profile used unpublished lab commit `d6f1ee8` for its core
source. The correctness server used unpublished lab commit `62fc998`.
The profile recorded runtime captures, loaded bindings, the model and view,
the frozen reference head at raw-T1, worker metadata, and source and resource
checks. Its plugin source was unpublished lab commit `47f4d0a`, its native
source was unpublished lab commit `ba60611`, and its extension build was
identified by the digest in the run record.

First load exactly one fusion-disabled AsyncLLM from the same engine options,
changing only `additional_config.fuzev.enabled` to `false`. Reuse
`vllm_generation_control.run_control` with exact canonical rendered inputs.
Its separate `describe_control` and named-worker metadata capture validate actual
disabled-mode imports, binaries and tasks; it never enters the fused-only
`validate_inputs` or `capture_runtime` path, and no learned head is loaded there.
Shutdown must return successfully before constructing the fused engine. Then
capture/build the genuine fused raw-T1 service on that new engine and use the
existing `MeasurementConfig`/HTTP `Harness` through actual loopback networking
for generation-only, decision-only and mixed pipelines. No ASGI transport is
used for inference measurements. API/service/engine cleanup failure is fatal,
including lifespan failures which a server might otherwise only log.

Enable the named `fuzev_performance_counts` RPC exactly once per resident engine,
after construction and before waves. Reset the idle prefix cache before every
wave on both sides; within-wave prefix reuse is allowed. Owner cache-salt values
differ between independent engines; namespace/reset policy, not literal salts,
is matched. Encoder/processor/NVMe caches are not claimed cold. RPC snapshots and
`counter_delta` cover the whole wave. Physical totals are nonnull only with
target-completion acknowledgement; pending/error/lower-bound evidence remains
null. They are not reconstructed from cached tokens, per-request records,
logical lengths, kernels or FLOPs. Measured totals exclude warmup.

Control wall time is in-process AsyncLLM client execution, not HTTP latency.
Fused wall time includes real HTTP/pipeline cleanup. These cannot yield a direct
network wall-speedup comparison. Existing request-stat phase observations retain
their explicit engine/service scope, not GPU-kernel time. The generation-only
control has no typed decision/mixed counterpart. A genuinely untouched-upstream
baseline remains unavailable: this is fusion-disabled **same fork**, never
pristine upstream. The profile passes no performance, correctness or repeatability check;
`runtime_qualified` remains false. Reports preserve completed waves on failure.

CPU tests use mocked lifecycle, capture and counter inputs. The recorded
profile identifies unpublished lab commit `d6f1ee8`, operator configuration
and full-panel observations as its historical inputs.

Historical result: **179 CPU tests and 40 subtests passed**. One
generation-control test was skipped because that image lacks the pinned vLLM
CPU development environment. These are fixtures, not a GPU performance run.

[Snapshot methods and environment](../README.md)
