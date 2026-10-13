# Recorded Integration Settings

These scrubbed records keep measured engine settings and requests on GB10.
Braced environment names stand for reader-supplied host paths, and listener
ports show the default 8000. Current launchers expose engine options
independently.

## Source-ZIP Serving Checks

- [smoke-native-exl3.json](smoke-native-exl3.json): installed ExLlamaV3
  `1.5.2+fuzev.v1.0`, runtime/native plugin `1.0.0`, source build/update
  commands, full launcher settings, HTTP 200 request/response and shutdown.
  It also keeps the input/output subset of the shared-state prewarm and
  option-checkpoint decision diagnostics.
- [smoke-vllm-nvfp4.json](smoke-vllm-nvfp4.json): installed vLLM
  `0.30.0+fuzev.v1.0.precompiled`, runtime/vLLM plugin `1.0.0`, executed
  installation commands, minimal engine JSON, HTTP 200 request/response,
  host-wide memory samples and exact SIGTERM/worker-stop outcomes.
- [smoke-vllm-exl3-4.05.json](smoke-vllm-exl3-4.05.json): installed vLLM
  `0.30.0+fuzev.v1.0.precompiled`, ExLlamaV3 `1.5.2+fuzev.v1.0`, Python
  quantization plugin `0.5.0+fuzev.v1.0` and runtime/vLLM plugin `1.0.0`.
  It also keeps the exact native-wheel build/install commands, corrected
  CPU-offload setting, disk PLE, all 19 PIECEWISE splits, HTTP 200 request/response,
  host-wide memory peak and observed shutdown outcomes.

These checks are dated 2026-10-13. Source commit IDs identify unpublished lab
exports. The JSON files are records, not direct launcher inputs:

- In each vLLM record, `engine_config` is the engine JSON.
- The native record keeps its launcher argv.

The [installation history](../component-install-smoke.md#source-zip-serving-checks)
describes measured scope. For a new environment, use the
[canonical installation guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md).

## Mixed Workloads

These settings describe the 2026-10-12 mixed-workload checks. Numeric settings
are historical measurements.

- `vllm-nvidia-mixed.json`: NVIDIA NVFP4, MTP 3, piecewise graphs and mmap PLE.
  The corrected generation-only head-loaded/control comparison uses the same
  engine settings; its control sets `additional_config.fuzev.enabled` to false.
- [service-template.json](service-template.json): the recorded vLLM service
  template, an archived research-server configuration. Environment
  placeholders replace model and head paths.
- `native-exl3-mixed.json`: turboderp EXL3 4.05 bpw, revision
  `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`, target/draft Q8 cache and MTP 3.
  The historical launcher that read this JSON is unpublished. The current
  ExLlamaV3 CLI exposes the corresponding model, cache, batch, chunk and MTP options.

Shared settings:

- Both use the raw shared decision head.
- Generation uses the bundled froggeric Qwen template v22.3. The recorded vLLM
  service template sets `preserve_thinking=true` and `reasoning_effort=medium`;
  individual test requests can override template settings.
- Decision encoding disables thinking.
- Checks request 256 and 512 output tokens, with one warmup and three measured
  requests at each length.
- Mixed-workload decisions arrive after generation begins.

The [integration record](../2026-10-12-engine-owned-integration.md) describes
the workloads, results and differences between study stages.

Omitted: the original host paths, the head-config identity check and
`smoke-expected-runtime.json` (the retired smoke wrapper's input). Model
weights remain linked to their Hugging Face repositories.

## Quality And Head Studies

Code/configuration records for the released head and the six 5,075-request
benchmark runs:

- [head-study.json](head-study.json): five-epoch refit, CAL-only scalar fit and full processed-quality settings.
- [frozen-evaluation-plan.json](frozen-evaluation-plan.json): fixed TEST plan used by the calibration/quality wrapper.
- [quality-panel.json](quality-panel.json): the 2,000-request FuZev evaluation mix and 3,075-question Kev DEV populations, with input digests.
- [quality-bindings.json](quality-bindings.json): each completed run's client binding, source hashes and observed server settings.
- [ExLlamaV3 3.05](quality-native-3.05-server.json), [4.05](quality-native-4.05-server.json) and [5.05](quality-native-5.05-server.json): retained server launch JSON.
- [NVIDIA NVFP4 engine](quality-vllm-nvidia-engine.json) and [service](quality-vllm-nvidia-service.json).
- [RadixArk NVFP4 engine](quality-vllm-radixark-engine.json) and [service](quality-vllm-radixark-service.json).
- [vLLM EXL3 engine](quality-vllm-exl3-4.05-engine.json) and [service](quality-vllm-exl3-4.05-service.json).

The `*_SOURCE` names in quality-bindings.json stand for the source trees behind
each run's training modules and HTTP client.

Launch files and observed runtime are separate records. For example, the
NVIDIA launch file has an image limit of 2 while the completed run reports 1.

Use the observed settings in quality-bindings.json to interpret a measured
result. Exact launch-file byte identity against every historical run is unconfirmed.

[Study-code records](../study-code/README.md) list the actual wrappers and
their remaining dependencies. Prepared examples, images, features and raw
journals are acquired or produced separately. Membership counts and hashes are
in the [provenance table](../research-provenance.md#released-head-provenance).
