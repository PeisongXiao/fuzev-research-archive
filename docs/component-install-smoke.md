# Component Installation History

Use the [installation guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md) for current commands.

## Source-ZIP Serving Checks

The 2026-10-13 checks used NVIDIA GB10, Linux aarch64, Python 3.12.3,
PyTorch 2.13.0+cu130 (CUDA 13.0), `MAX_JOBS=4`, normal dependency resolution
and a CPU decision head.

The JSON records keep scrubbed executed commands, source-ZIP release
versions, engine settings and exact HTTP requests/responses. Braced paths stand
for reader-supplied local files. Code commit IDs identify **unpublished lab exports**.

| Serving Path | Installed Versions | HTTP 200 Combined Request | Sampled Host Peak |
| --- | --- | --- | --- |
| [Native EXL3 4.05 bpw](configs/smoke-native-exl3.json) | ExLlamaV3 `1.5.2+fuzev.v1.0`; runtime/plugin `1.0.0`; FlashInfer `0.7.1`; JIT cache `0.7.1+cu130` | Noul 0.9759222269; Choice `refund` probability 0.9838241339; Score 0.8634201884 | 82,892,206,080 bytes (77.20 GiB) |
| [vLLM NVFP4](configs/smoke-vllm-nvfp4.json) | vLLM `0.30.0+fuzev.v1.0.precompiled`; runtime/plugin `1.0.0`; FlashInfer `0.6.18.post1`; JIT cache `0.6.18.post1+cu130` | Noul 0.9648171067; Choice `refund` probability 0.9877915382; Score 0.8462883234 | 105,996,451,840 bytes (98.72 GiB) |
| [vLLM EXL3 4.05 bpw](configs/smoke-vllm-exl3-4.05.json) | vLLM `0.30.0+fuzev.v1.0.precompiled`; ExLlamaV3 `1.5.2+fuzev.v1.0`; Python quantization plugin `0.5.0+fuzev.v1.0`; runtime/plugin `1.0.0`; FlashInfer `0.6.18.post1`; JIT cache `0.6.18.post1+cu130` | Noul 0.9893544912; Choice `refund` probability 0.9835783243; Score 0.8786213398 | 104,181,092,352 bytes (97.03 GiB) |

### Native EXL3

**Revisions:** native source `72408e5`, runtime `7ecb72a`, native plugin `72f8cc9`.

**Build.** The original fresh source build at `5d93a1a` compiled the full CUDA
extension. The final Python-only ZIP update reused cached, unchanged C++
objects through a normal `pip install --no-build-isolation`. The documented
header recipe keeps toolkit headers ahead of supplemental headers:

1. Clear `CPATH`, `C_INCLUDE_PATH` and `CPLUS_INCLUDE_PATH`.
2. Add the wheel's `nvidia/cu13/include` via `CPLUS_INCLUDE_PATH` and
   `nvidia/cu13/lib` via `LIBRARY_PATH`.

**Launcher settings:** context 32,768, cache 65,536 tokens, Q8 KV, batch 8,
default prefill chunk 8,192, folded-hot MoE and native BF16 GDN operands. MTP
and vision were off. Startup logged
`Prefill optimizations enabled; chunk size 8192 tokens`.

**Run.** A verification wrapper invoked the installed launcher in-process, ran
the HTTP check and decision-latency diagnostics, then awaited orderly
application shutdown. It exited 0.

**Diagnostics.** The [native JSON](configs/smoke-native-exl3.json) keeps their
input/output subset.

- **Shared-state prewarm:** the single ordered pair took 8.73646424600156 s
  without state prewarm and 5.317069264001475 s with it, a 39.139346% timing
  reduction. KV/page caches were reset before each case, with no separate
  warmup or matched first-use kernel-cache state. The pair measures timing only.
- **Option checkpoint:** one 64-word, eight-option case took 3.641698779996659 s,
  with candidate readouts using `checkpoint=False`. The single case measures
  latency below the 256-token page boundary only: the 223-token query and all
  240-token candidates stayed under it.

Native decision probabilities can differ slightly between server starts. They
depend on Triton's kernel-configuration choice, made by timing at first
compile, and on which prefix pages are already cached.

### vLLM NVFP4

**Revisions:** engine source ZIP `5e11d8fb1`, runtime `7ecb72a`, vLLM plugin
`aa2fe16`. Later engine-fork documentation changes left the measured code
unchanged. The fork installation reused the matching official vLLM 0.30.0
binary wheel.

**Settings:** the minimal 0.75/32,768/8 configuration, synchronous scheduling,
mmap PLE, PIECEWISE CUDA graphs, default BF16 KV and no MTP. No explicit
batch-token budget was supplied. Startup selected `max_num_batched_tokens=2048`:
GB10's NVML device-total-memory query raises `NVMLError_NotSupported`, so
upstream `get_batch_defaults` uses `device_memory=0` and selects 2,048.

**First attempt.** The engine loaded but API initialization failed: the
image-placeholder helper received `VllmConfig` instead of `ModelConfig`. That
server exited 3 and produced no HTTP response or host-peak measurement.

**Corrected run** (corrected plugin; HTTP 200 values in the table):

- The server exited **-15** from the wrapper's normal operator-stop SIGTERM;
  the wrapper itself exited 0.
- The log reported application shutdown complete after the upstream process
  manager's five-second force-kill fallback for its remaining worker.
- Worker shutdown was not fully graceful.

### vLLM EXL3

**Revisions:**

- Engine source `5e11d8fb1`, runtime `7ecb72a` and vLLM plugin production code
  `aa2fe16` (test-only revision `13c2f5f`).
- Quantization source ZIP `a0e5e99`, including code `f6f03a3`. It used the
  Python plugin, with no `vllm_exl3_c`.
- Compiled native wheel from source ZIP `72408e5`, reusing unchanged C++
  objects from the original fresh build.

**Build:**

- The `pip wheel --no-build-isolation --no-deps` command used the native
  environment's CUDA 13 supplemental include/library paths and `MAX_JOBS=4`.
- `--no-deps` applied to the wheel build only; every installation resolved
  dependencies normally.
- Only the older native wheel was uninstalled before the final wheel installation.
- The [JSON record](configs/smoke-vllm-exl3-4.05.json) keeps the exact scrubbed
  build and install argv.

**First attempt.** On 2026-10-13 the first vLLM EXL3 attempt failed before
loading weights, with server exit 3 and no HTTP response.

- Its original engine JSON omitted `engram_config`. The failure belongs to
  that configuration, not to the later updated engine JSON.
- The upstream `Qwen4ExpPLEPinnedHostEmbedding` wrapper expected a raw
  `.weight` tensor, but the quantization plugin owned EXL3's packed storage.
- The failed run sampled a host-wide peak of 12,161,536,000 bytes, covering
  only that pre-weight-load attempt.

**Fix.** The corrected operator configuration added only top-level
`engram_config: {"cpu_offload": false}`. It kept `VLLM_EXL3_NGRAM_TABLE=disk`,
lazy safetensors and the complete PIECEWISE split list. This bypasses the raw
pinned-host wrapper while the plugin keeps disk-backed PLE storage.

**Corrected run** (HTTP 200 values in the table): memory fraction 0.75,
context 8,192, capacity 4, synchronous scheduling, BF16 KV, MTP off, lazy
safetensors and disk-backed packed PLE with all 19 PIECEWISE splitting operations.

- The log resolved `EngramConfig(cpu_offload=False)`, completed graph capture
  and reported 19.84 GiB available for KV cache.
- Normal operator-stop SIGTERM produced server exit **-15**, with wrapper exit
  0. The log recorded engine manager stopped and application shutdown complete.
  No force-kill warning was observed.

### Memory And Coverage

- Host peaks are the maximum of one-second `MemTotal - MemAvailable` samples.
  They cover host-wide physical usage rather than isolated engine RSS.
- The native peak covers its entire verification, including the decision diagnostics.
- The NVIDIA cgroup peak of 22,113,214,464 bytes includes earlier container
  history and is not GPU memory. The native cgroup peak is in its JSON.

These checks establish installation, engine startup and one text SystemOne
request per recorded path. They do not measure image serving or decision quality.

## Earlier Source-ZIP Checks

On 2026-10-12, earlier source-ZIP installations on GB10 completed engine
startup and text SystemOne requests:

| Serving Path | Installation And Settings | HTTP 200 Responses |
| --- | --- | --- |
| ExLlamaV3 EXL3 | Fresh ZIP, compiled CUDA extension, normal dependency installation, Q8 cache and optimized prefill | Noul: 0.968095 |
| vLLM NVFP4 | Fresh ZIP plus matching official upstream binaries; GPU memory fraction 0.75, FP8 KV, memory-mapped PLE, MTP with three draft tokens and piecewise CUDA graphs | Combined request: Noul 0.980285; Choice `refund` probability 0.957217; Score 0.879695 |
| vLLM EXL3 4.05 bpw | Fresh ZIP, compiled native wheel, resolved dependencies and a newly generated complete packed serving view; context 8,192, capacity 4, GPU memory fraction 0.75, BF16 KV, memory-mapped PLE, piecewise CUDA graphs and MTP off | Noul: 0.9796603322 |

- The EXL3 source ZIP included the plugin's disk-PLE eager-boundary fix. The
  head was loaded with the turboderp 4.05 bpw pack.
- The native server shut down cleanly.
- Both vLLM server processes exited with status 0 and their containers were
  removed. In both, upstream worker teardown used the engine manager's normal
  five-second force-kill fallback, so engine shutdown was not cleanly graceful.

These checks cover installation and text SystemOne API serving. Image coverage
and decision quality are in [evaluation](research-evaluation.md).

## Earlier Wheel-Only Checks

On 2026-10-12, three wheel sets passed installation, isolated imports and
CLI-help checks:

- runtime only
- runtime plus ExLlamaV3 plugin
- runtime plus vLLM plugin

Each ran in a separate fresh virtual environment over a prepared dependency
image with inherited dependency conflicts.

### Measured Scope

The wheel-only check used these **unpublished lab commit IDs**:

| Component | Historical code revision |
| --- | --- |
| Runtime | `f4a4e64467da068819d317952920106c24572700` |
| ExLlamaV3 plugin | `e36fd72d6b08c5aabb9ee232d532c8d0eed883d6` |
| vLLM plugin | `1d9ca0905dc44d09effa194495861439d7ee1104` |

- Wheels reported version 1.0.0 and `py3-none-any`.
- Imports resolved to the installed environments with source exports removed.
- The checks inspected SystemOne and attachment symbols, executed both CLI
  `--help` paths, and read 15 head tensors containing 2,872,401 parameters.

Measured scope: installation, imports, artifact loading and CLI help.

Tested environment: Linux aarch64, Python 3.12.3, PyTorch 2.13.0+cu130,
safetensors 0.8.0, Pydantic 2.13.5, Pillow 12.3.0, FastAPI 0.136.3,
Uvicorn 0.53.0 and Transformers 5.17.0.

### Dependency Findings

- **NCCL:** Torch declared NCCL 2.29.7 while the prepared vLLM image contained
  2.30.7, an override the upstream build used for DeepEP. The metadata
  conflict remained visible. Import success did not establish binary compatibility.
- **cuSPARSELt:** the installed 0.8.1 wheel used a `manylinux2014_sbsa` tag.
  Its library was AArch64, but the platform checker rejected the tag. Ordinary
  venv `pip check` omitted this inherited diagnostic; checking all visible
  distributions kept it in every environment.

The final checks reported no new dependency diagnostics and preserved both
inherited failures. Earlier version-only checks omitted dependencies entirely
and established metadata only.

### Reproduction Boundary

The original prepared image and raw installation records are omitted. For a
new environment, follow the [current installation procedure](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md).
For route behavior, see [SystemOne CPU checks](2026-10-12-systemone-unit-compatibility.md).
