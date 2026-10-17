# Historical Generation And Decision Integration

Archived v1.0 measurements and limitations. The vLLM sections are v1.0
research results; v1.1 serves through ExLlamaV3 only. The
[v1.1 integration evidence](integration-v1.1.md) covers the later checks.

FuZev attaches a shared decision head to an ExLlamaV3 Generator or, in the
v1.0 research checks, a vLLM engine.
The engine runs scheduling and caches; each decision scores the head afresh.

To serve, see the [installation guide](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md), [API reference](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md) and
[ExLlamaV3 integration](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1).
GB10 source-installation checks are in
[verified source installations](component-install-smoke.md).

## Recorded Measurement Setup

All checks:

- **Platform:** one NVIDIA GB10 (128 GB unified memory), ARM64, Python 3.12, PyTorch 2.13.0+cu130.
- **Builds:** pre-release builds of the archived research evaluation servers and
  integration plugins.

Cross-engine research study, vLLM configuration:

- **Model:** [NVIDIA Qwen3.8-Flash-Next NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47).
- **Engine:** V1 model runner, MTP with three draft tokens, FP8 E4M3 KV,
  piecewise CUDA graphs, memory-mapped embeddings.
- **Capacity:** context 262,144, four engine sequences.
- The research configuration uses MTP and recomputes the full context for every
  decision option.

ExLlamaV3 checks:

- **Model:** [turboderp EXL3 4.05 bpw](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/55a732e0c4c3d4614bc42b68493bb930d9b02c0a).
- **Generation control:** 8-bit target and draft caches with 1,048,576 cache
  tokens each, 8,192-token prefill chunks, batch 4, MTP with three draft tokens.
- **Mixed-request record:** a separate 64-output-token check.

Run configurations: [ExLlamaV3 EXL3 mixed](configs/native-exl3-mixed.json)
and [vLLM NVIDIA mixed](configs/vllm-nvidia-mixed.json).
The [configuration notes](configs/README.md)
describe these settings and path placeholders.

## Cross-Engine Research Checks

The vLLM measurements in this section are archived research findings.
The ExLlamaV3 row records its separate mixed-request check.

### vLLM Coexistence

Generation produces coherent reasoning and Python before, during and after five
fresh Choice, Noul and Score requests, which use 15 distinct engine requests.
Recorded idle counters return to zero. Missing or released prefixes return HTTP 404.

| Generation Position | Output Tokens | Accepted / Proposed Draft Tokens | Acceptance |
| --- | ---: | ---: | ---: |
| before decisions | 90 | 63/78 | 80.77% |
| alongside decisions | 154 | 105/150 | 70.00% |
| after decisions | 81 | 54/78 | 69.23% |

The generation budget is 256 tokens; some completions end sooner on their own.

### Decisions During Generation

Three decisions arrive about 0.75, 1.00 and 1.25 seconds after generation
begins: after the observed first-token bound, before the generation HTTP response
completes. Request lifetimes overlap.

| Engine | Output Budget | First-Token Upper Bound (s) | Generation HTTP (s) | Draft Tokens |
| --- | ---: | ---: | ---: | --- |
| ExLlamaV3 EXL3 4.05 bpw | 64 | 0.29 | 3.97 | 42 accepted / 66 proposed (63.64%) |
| vLLM NVIDIA NVFP4 | 256 | 0.32 | 10.85 | 165 accepted / 279 proposed |

Both checks observe four concurrent HTTP requests. Output budgets and engine
settings differ, so they are separate implementation checks.

### Long Prefills

Generation-only vLLM probes, with no head loaded, complete exact-length
prefills and 512-token outputs:

| Prompt Tokens | Output Tokens | HTTP Wall (s) | Accepted / Proposed Draft Tokens | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | 512 | 22.23 | 370/432 | 85.65% |
| 32,768 | 512 | 29.25 | 345/498 | 69.28% |

[Long-prefill aggregates](results/integration/v1.0/mtp-diagnostics.json) also hold 256/512-token
completion checks. All are single probes with no measured repeat spread.

### Loaded But Unused Head

Matched vLLM engines from the archived research evaluation servers generate
with and without the head loaded.

- **Prompts:** two procedural Python prompts, each with one warmup and three
  measured pairs.
- **Requests:** one stream, unique prefixes, 256-token output budget.
- **Sampling:** temperature 1, top-p 0.95, top-k 20.
- **Prefix reuse:** both sides reuse the same prefix only within their pair.

Values are mean paired percentage changes, with observed pair ranges in brackets.
Positive values are slower.

| Nominal Prompt Tokens | HTTP Wall % [Range] | Prefill % [Range] | Decode % [Range] |
| --- | --- | --- | --- |
| 512 | +6.46 [3.85, 11.49] | -1.92 [-14.26, 9.98] | +7.19 [3.85, 13.44] |
| 2048 | -2.65 [-8.46, 0.70] | -3.39 [-10.17, 1.20] | -2.41 [-9.38, 1.44] |

MTP step counts and pooled decode-time change per step, over the three pairs:

| Nominal Prompt Tokens | Control MTP Steps | Head-Loaded MTP Steps | Pooled Decode-Time Change per Step |
| ---: | ---: | ---: | ---: |
| 512 | 240 | 260 | -1.11% |
| 2048 | 251 | 245 | -0.12% |

Different speculation paths and mixed timing signs make this small overhead
comparison inconclusive.

ExLlamaV3 loaded-but-unused-head overhead is unmeasured.

## ExLlamaV3 Generation Control

The control is upstream ExLlamaV3 v1.5.2, compared with the FuZev fork.
Both sides add the
[prefill optimizations from exllamav3-fuzev (`exllamav3.optimized`)](https://github.com/PeisongXiao/exllamav3-fuzev/blob/v1.0/doc/fuzev-fork.md#prefill-optimizations)
and run generation only. The comparison includes the fork's engine changes.

- **Runs:** one warmup and three measured requests per output length and side,
  with zero cached prompt tokens.
- **Time columns:** medians, with observed sample ranges in brackets.
- **Decode tok/s:** total decoded tokens after each first token, divided by total
  decode time over the three requests.

| Output Tokens | Engine | Median Decode s [Range] | Median HTTP s [Range] | Decode tok/s | Draft Acceptance |
| --- | --- | --- | --- | ---: | --- |
| 256 | Upstream + prefill optimizations | 5.12 [4.96, 5.31] | 5.44 [5.27, 5.62] | 49.70 | 513/765 (67.06%) |
| 256 | FuZev fork + prefill optimizations | 5.15 [4.86, 5.22] | 5.47 [5.16, 5.53] | 50.21 | 514/762 (67.45%) |
| 512 | Upstream + prefill optimizations | 9.42 [9.21, 10.32] | 9.73 [9.52, 10.63] | 52.96 | 1058/1434 (73.78%) |
| 512 | FuZev fork + prefill optimizations | 9.91 [9.33, 10.41] | 10.21 [9.64, 10.71] | 51.72 | 1047/1467 (71.37%) |

Median decode time increases 0.63% at 256 tokens and 5.13% at 512 tokens.
Decode throughput at 512 tokens falls 2.35%. The three-sample ranges overlap;
these observations describe this workload.

## Data

- [Integration aggregates](results/integration/v1.0/integration-results.json): coexistence and mixed requests.
- [Timing samples](results/integration/v1.0/integration-timing-samples.json): all six overhead pairs and
  the ExLlamaV3 control's twelve measured requests.
- [Source integration records](results/README.md#v10-integration-and-generation):
  timing, lifecycle and speculation measurements.

## Limitations

- Implementation checks cover the recorded configurations and short workloads.
- ExLlamaV3 measurements use a pre-release fork build. Its extension predates the
  released CUDA-graph kernel-argument fix, and its optimized prefill casts
  gated-delta-network operands to FP16. These measurements were not repeated on the
  released build ([fork limitations](https://github.com/PeisongXiao/exllamav3-fuzev/blob/v1.0/doc/fuzev-fork.md#limitations)).
- Image input is experimental. ExLlamaV3 decisions can reuse a retained prefix,
  including its image embeddings, from a generation request that included an image.
- The v1.0 published builds' image path had CPU test coverage only. The image results
  in [benchmarks](benchmarks-v1.0.md) come from the research evaluation servers.

See the [API limitations](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#limitations) for supported engine settings and
[benchmarks](benchmarks-v1.0.md) for decision quality.
