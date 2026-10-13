# Engine Integration History, 2026-10-12

Historical implementation checks on NVIDIA GB10. Curated results and measured
timing spread are on the
[index integration page](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md).
Decision quality is in the
[benchmarks](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md).

## Checkpoints

| Engine | Checkpoint | Revision |
| --- | --- | --- |
| vLLM | nvidia/Qwen3.8-Flash-Next-NVFP4 | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| ExLlamaV3 | turboderp/Qwen3.8-Flash-Next-exl3, 4.05 bpw | `55a732e0c4c3d4614bc42b68493bb930d9b02c0a` |

Tests ran one engine at a time. A shared head used the running engine's
existing LM projection, scheduler and caches. Attachment cleanup left the
injected engine available to its application.

## Pre-Fix vLLM Results

**Superseded: generation was malformed.**

- Longer NVIDIA requests reached 256 and 512 output tokens but accepted 0/765
  and 0/1,533 drafts. Output repeated exclamation marks after an initial word.
- The initial delayed-arrival test also accepted zero of 189 drafts. Its
  scheduling checks are narrower than generation correctness.

HTTP success and draft activity did not establish correct generation.

**Capture/replay defect.** Eager PLE consumers used stale captured values. The
correction replays completed producers and puts both gather and copy inside
the eager boundary.

With the same NVIDIA configuration, 256/512-token requests then accepted
168/264 and 361/459 drafts in 13.141/16.127 seconds of HTTP wall time.
Inspected output contained coherent reasoning and Python.

**Circular-cache slot-mapping defect.** A later 8,192-token input caused a CUDA
error: generic positional mapping indexed a circular cache as a sequence of
blocks. The correction honors the cache's own mapping capability. The initial
process recorder misclassified this engine failure because the API process
exited zero; failure classification was corrected.

**Stock control.** The stock vLLM 0.30.0 control rejected the configured FP8
QSA cache and lacked the mmap PLE path. It supplies no comparable acceptance
or throughput result.

## Corrected NVIDIA Checks

- **Long inputs:** exact 8,192/32,768-token inputs each produced 512 tokens,
  accepting 370/432 and 345/498 drafts in 22.228/29.246 seconds of HTTP wall
  time. The process stayed healthy and shut down without OOM or a fatal engine error.
- **Attached head:** generation ran before, alongside and after five decisions.
  The check covered all three primitives, accepted-prefix identity,
  independent query and candidate requests, released-prefix rejection and
  cleanup. Actual generation lengths were 90/154/81 tokens, accepting 63/78, 105/150
  and 54/78 drafts respectively.
- **Delayed arrival:** decisions arrived after 0.75, 1.00 and 1.25 seconds,
  following the measured first-token upper bound of 0.325 seconds. Generation
  reached 256 tokens and accepted 165/279 drafts.

Measured builds (**unpublished lab commit IDs**): vLLM `5ccd64a`, runtime
`46de616` and vLLM plugin `38e446d`.

## Unused-Head Comparison

The corrected NVIDIA comparison used one warmup and three measured pairs per
nominal input length, with 256 generated tokens per request. Decode, prefill
and HTTP wall are mean paired changes.

| Input tokens | Decode | Prefill | HTTP wall | Pooled time per step |
| ---: | ---: | ---: | ---: | ---: |
| 512 | +7.19% | -1.92% | +6.46% | -1.11% |
| 2,048 | -2.41% | -3.39% | -2.65% | -0.12% |

Generated text and MTP verification-step counts differed. The per-step
normalization does not establish the cause of the wall-time difference.

This small comparison is inconclusive about negligible generation-only overhead.
Per-pair values and ranges are on the index integration page.

## ExLlamaV3 Comparison

Recorded experiment settings: 4.05-bpw checkpoint, Q8 target/draft caches,
MTP3, mmap PLE, 262,144-token context, capacity 4 and memory fraction 0.85.
The control combined upstream ExLlamaV3 with the same prefill optimizations,
without FuZev engine hooks.

Sixteen requests reached their full 256/512-token outputs: one warmup and three
measured requests per output length and engine. Both processes used identical
encoded inputs and reported zero prefix-cache hits.

| Output tokens | Control decode tok/s | Fork decode tok/s | Median decode-time change | Acceptance, control / fork |
| --- | ---: | ---: | ---: | --- |
| 256 | 49.70 | 50.21 | +0.63% | 67.06% / 67.45% |
| 512 | 52.96 | 51.72 | +5.13% | 73.78% / 71.37% |

- Throughput is total measured `new_tokens - 1` divided by total decode time.
  The median-duration column uses a different aggregation.
- At 512 tokens, job wall times were 9.52/9.73/10.62 seconds for the control
  and 10.71/10.21/9.64 seconds for the fork. Ranges overlap, with three samples
  per side.

The control and fork are **unpublished lab commit IDs** `50faf4d` and
`b381f24`. The extension predates the
[shipped CUDA-graph kernel-argument fix](research-history.md#superseded-serving-results);
these measurements were not repeated after it.

## Reproduce

- Start the engines with the current
  [component CLIs and build instructions](modular-setup.md).
- Exercise an already running server with the small HTTP clients in the
  [evaluation guide](research-evaluation.md#running-server-checks).
- Scrubbed launch configurations are in [configs](configs/README.md). Original
  host paths and prompt journals are omitted.

Image and decision-quality scope is in [evaluation](research-evaluation.md).
