# Result Aggregates

These copies keep recorded metrics, intervals, token counts and timing samples.
They omit private paths, hash maps, examples, predictions, template bodies,
request/output text and pre-release server status fields. Settings are
historical lab measurements; current releases use engine defaults or operator settings.

## Six Quality Panels

Each configuration completes 5,075 requests: 2,000 FuZev evaluation-mix requests and
3,075 Kev DEV questions.

| Configuration | Source Directory | Aggregates |
| --- | --- | --- |
| ExLlamaV3 EXL3 3.05 bpw | `e2e-native-3.05-v1` | [Panel](panels/native-3.05/panel-report.json), [summary](panels/native-3.05/summary.json) |
| ExLlamaV3 EXL3 4.05 bpw | `e2e-native-v2` | [Panel](panels/native-4.05/panel-report.json), [summary](panels/native-4.05/summary.json) |
| ExLlamaV3 EXL3 5.05 bpw | `e2e-native-5.05-v1` | [Panel](panels/native-5.05/panel-report.json), [summary](panels/native-5.05/summary.json) |
| vLLM NVIDIA NVFP4 | `e2e-vllm-nvidia-v4` | [Panel](panels/vllm-nvidia/panel-report.json), [summary](panels/vllm-nvidia/summary.json) |
| vLLM RadixArk NVFP4 | `e2e-vllm-radixark-v1` | [Panel](panels/vllm-radixark/panel-report.json), [summary](panels/vllm-radixark/summary.json) |
| vLLM EXL3 4.05 bpw | `e2e-vllm-exl3-4.05-v1` | [Panel](panels/vllm-exl3-4.05/panel-report.json), [summary](panels/vllm-exl3-4.05/summary.json) |

- **ExLlamaV3 4.05:** the reported run is the second attempt. The first stopped
  before any decision request.
- **vLLM NVIDIA:** the reported run is the fourth attempt. Earlier attempts
  stopped on a research-server image-payload bug (request 8), a lost server
  connection (request 817) and an operator stop to lower memory (request 670).
  Their completed responses feed only the run-to-run comparison.
  - [Partial v2](panels/vllm-nvidia/partial-v2.json) and
    [partial v3](panels/vllm-nvidia/partial-v3.json) keep the stop counts.

Run-to-run records:

- The [shared 670-request comparison](repeat/nvidia-shared-670.json) records
  14.6-14.9% pairwise selection changes, probability changes up to 0.62 and
  shared-request accuracies of 73.43%, 74.78% and 75.22%.
- The [later NVIDIA synthetic trace](repeat/nvidia-later-trace.json) failed the
  2e-6 repeat tolerance, with a maximum probability delta of 0.10429 and about
  47% solo-query feature drift.

## TEST And Processed Quality

- [Native TEST result](test/result.json): `native-results-v1/result.json`;
  LM-assisted, frozen-LM and feature-only systems, with paired intervals.
- [Kev-9B v2 TEST aggregates](test/kev-9b-report.json):
  `fam-breadth-test-report/report.json`, selected Kev-9B v2 system.
- [Processed quality](processed-quality.json): 15 tracks, 27 checks per head
  variant; 23 pass and the 4 VL-RewardBench image checks fail
  ([analysis](../processed-quality.md)). Includes paired group-bootstrap intervals.

## Short Performance

Each source copy keeps its summary, warmups and measured timing samples. The
vLLM files are superseded pre-fix observations, kept only in the archive.

- [native3-small-performance-v2.json](performance/native3-small-performance-v2.json)
- [native4-small-performance-v2.json](performance/native4-small-performance-v2.json)
- [native5-small-performance-v2.json](performance/native5-small-performance-v2.json)
- [vllm-nvidia-small-performance-v2.json](performance/vllm-nvidia-small-performance-v2.json)
- [vllm-radixark-small-performance-v2.json](performance/vllm-radixark-small-performance-v2.json)
- [vllm-exl3-small-performance-v2.json](performance/vllm-exl3-small-performance-v2.json)

## Integration And Generation

`vllm-nvidia-staggered-v2` and `vllm-mrv1-mtp-long-v1` are superseded
pre-fix observations. `vllm-mrv1-long-prefill-v2` records the initial failed
long prefill. The ExLlamaV3 core control uses upstream plus the exllamav3-fuzev
prefill optimizations.

- [native405-staggered-v2.json](integration/native405-staggered-v2.json)
- [vllm-nvidia-staggered-v2.json](integration/vllm-nvidia-staggered-v2.json)
- [vllm-postfix-staggered-v1.json](integration/vllm-postfix-staggered-v1.json)
- [vllm-postfix-smoke-v1.json](integration/vllm-postfix-smoke-v1.json)
- [vllm-postfix-generation-overhead-v1.json](integration/vllm-postfix-generation-overhead-v1.json)
- [vllm-postfix-overhead-control-v1.json](integration/vllm-postfix-overhead-control-v1.json)
- [vllm-postfix-overhead-headunused-v1.json](integration/vllm-postfix-overhead-headunused-v1.json)
- [native-core-regression/upstream.json](integration/native-core-regression/upstream.json)
- [native-core-regression/fork.json](integration/native-core-regression/fork.json)
- [native-core-regression/comparison.json](integration/native-core-regression/comparison.json)
- [vllm-mrv1-mtp-long-v1.json](integration/vllm-mrv1-mtp-long-v1.json)
- [vllm-mrv1-mtp-long-v2.json](integration/vllm-mrv1-mtp-long-v2.json)
- [vllm-mrv1-long-prefill-v2.json](integration/vllm-mrv1-long-prefill-v2.json)
- [vllm-mrv1-long-prefill-v3.json](integration/vllm-mrv1-long-prefill-v3.json)
- [native405-staggered-lifecycle-v1.json](integration/native405-staggered-lifecycle-v1.json)
- [vllm-postfix-overhead-control-lifecycle-v1.json](integration/vllm-postfix-overhead-control-lifecycle-v1.json)
- [vllm-postfix-headunused-lifecycle-v1.json](integration/vllm-postfix-headunused-lifecycle-v1.json)
- [vllm-mrv1-mtp-long-lifecycle-v3.json](integration/vllm-mrv1-mtp-long-lifecycle-v3.json): container record of the failed initial long prefill
- [vllm-mrv1-mtp-long-lifecycle-v4.json](integration/vllm-mrv1-mtp-long-lifecycle-v4.json)

[Recorded configs](../configs/README.md) and
[integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md) describe the workloads.

## Capacity

[Native EXL3 4.05 capacity record](capacity/native4-serving-capacity-v1.json)
copies `native4-serving-capacity-v1.json`: a 262,144-token window and an
82,111,987,200-byte peak allocation. It measures serving capacity only.
