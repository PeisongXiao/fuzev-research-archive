# vLLM Disk-PLE Admission (historical)

The research adapter validates disk-backed per-layer embedding settings
before comparison extraction.

## Explicit Operator Settings

The following are fragments to merge into reviewed engine options, not complete
launch configurations. Admission never supplies or rewrites operator flags.

Both raw cells (`vllm-nvidia`, `vllm-radixark`) require:

```json
{
  "enforce_eager": true,
  "safetensors_load_strategy": "lazy",
  "engram_config": {"cpu_offload": false},
  "additional_config": {"fuzev": {"enabled": true, "ple_storage": "mmap"}}
}
```

The research adapter supported only the standard `auto`/`safetensors` loader
and TP/DP=1. A missing,
null, numeric or string `cpu_offload` is not explicit false. Generic CPU offload
options or `VLLM_PLE_CPU_OFFLOAD=0` do not substitute for the explicit Engram
setting; an explicit false overrides the core's environment/default offload.
Resident PLE and pinned-host offload consume the same unified RAM budget.

The inspected core PLE/factory/Engram code is unchanged between unpublished lab commit `53a698e` and
cached HEAD unpublished lab commit `0efcc39`. The pinned official configs, inspected separately by the
operator, select raw FP8 PLE: NVIDIA's `MIXED_PRECISION` exact PLE layer entry is
FP8, and RadixArk's text config specifies `ple_embedding_dtype=float8_e4m3fn`.
This is not an assumption of BF16 tables or uniform FP4 model math. Core mmap
still enforces its supported raw method and file-backed CPU safetensors views
at load time; the admission helper does not inspect or instantiate tensors.

Packed `vllm-exl3-4.05` uses the independent plugin disk adapter. Explicitly set
`VLLM_EXL3_NGRAM_TABLE=disk` and pin that actual environment. Its conservative
eager engine fragment is:

```json
{
  "enforce_eager": true,
  "safetensors_load_strategy": "lazy",
  "engram_config": {"cpu_offload": false},
  "additional_config": {"fuzev": {"enabled": true}}
}
```

Do not set raw `ple_storage=mmap` on packed EXL3. Omitting the raw selector (or
using its neutral `resident` value) is valid only with the explicit plugin
`disk` environment; the plugin's environment default is resident. Lazy loading
is required because the plugin can otherwise retain a copied tensor rather
than a mapped checkpoint view. No native payload conversion is introduced;
complete canonical checksums and same-file serving-view/proof checks remain.

The existing optimized alternate remains available: `enforce_eager=false`,
`compilation_config.cudagraph_mode=PIECEWISE`, with the existing attention
splitting ops retained and `vllm::exl3_ngram_lookup_out` included. The resolved
check mirrors the plugin's exact conditions: reject any graph mode whose name
contains `FULL`; if `splitting_ops` is nonempty, require that lookup op. The
plugin also permits empty/absent splitting ops. Admission does not modify the
compilation config or assert GPU qualification of an admitted alternate.

## Boundaries

- Serving checks explicit options against the pinned runtime cell before
  starting HTTP. After metadata resolution it checks the actual quantization,
  eager/lazy/offload/parallel settings and packed graph conditions before
  importing or constructing AsyncLLM, while retaining existing resource guards.
- Extraction checks credentials first, then owns and checks options before
  frozen-reference/input validation, output records, artifact checksums or
  engine construction. `verify_artifact` independently checks storage before
  file enumeration/checksums. The real engine context rechecks resolved config.
- `modelopt_mixed` is added only for canonical `vllm-nvidia`, with the existing
  exact repo/revision pins. Its small checksummed local config must show
  `quant_method=modelopt`, `MIXED_PRECISION`, and the exact canonical
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding` FP8 entry.
  RadixArk/EXL3 reject this name. Actual runtime identity remains
  `modelopt_mixed`; no uniform-FP4 relabeling or broader loader permission.

## CPU Verification

Torch-free helper tests run in the existing host venv; launch-boundary and
comparison tests use synthetic metadata, fake engine classes and tiny CPU
fixtures. No actual AsyncEngineArgs/AsyncLLM, model backbone, download or GPU
work runs.

Verified: **101 host helper tests passed**; the combined bounded image run had
**558 passed, 20 skipped**, with one upstream Starlette/AnyIO deprecation
warning. The skips are existing opt-in real API inspections requiring the
pinned vLLM CPU dev environment, not admission failures. That inspection mode
was not enabled.

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_vllm_ple_storage.py \
  tests/test_vllm_ple_storage_launch.py \
  tests/test_vllm_comparison_extract.py \
  tests/test_vllm_service.py \
  tests/test_vllm_images.py \
  tests/test_vllm_image_api.py \
  tests/test_vllm_phase_timing.py \
  tests/test_preparation.py \
  tests/test_comparison_inputs.py \
  tests/test_comparison_seal.py \
  tests/test_telemetry.py
```

[Snapshot methods and environment](../README.md)
