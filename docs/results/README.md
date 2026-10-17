# Result Aggregates

These copies keep recorded metrics, intervals, token counts and timing samples.
They omit private paths, hash maps, examples, predictions, template bodies,
request/output text and pre-release server status fields. Settings are
historical lab measurements; current releases use engine defaults or operator settings.

Speed measurements use **one NVIDIA GB10 (128 GB unified memory)**.
The served comparison records Python 3.12.3, PyTorch 2.13.0+cu130,
Triton 3.7.1, FlashInfer 0.6.18.post1 and JIT cache 0.6.18.post1+cu130.
Earlier v1.1 prefill records hold the latter four versions; their Python
version is unrecorded. Each speed record's `environment` block gives its
device and the library versions, batch, KV bits, chunk, recurrent-state
precision and MTP settings it holds. The
[prefill aggregate](performance/v1.1/final-prefill-upstream-master-results.json)
names the v1.1 prefill reference and candidate builds; the F1-F5/D1 records
use that candidate build ([recipe](../integration-v1.1.md#recipe-and-scope)).
The separate source-installation and decision-root HTTP records use
FlashInfer 0.7.2 and JIT cache 0.7.2+cu130.

Head-quality and earlier API decision results were measured on the v1.1
release candidate before the
[decision-root scheduling fix](../integration-v1.1.md#decision-root-checks);
prefill and decode do not use that path.

## v1.0 And v1.1 Comparison

[Results and measured scope](../v1.0-v1.1-comparison.md).

- [comparison/v1.1/kev-dev-paired.json](comparison/v1.1/kev-dev-paired.json).
- [comparison/v1.1/kev-dev-v1.0.json](comparison/v1.1/kev-dev-v1.0.json).
- [comparison/v1.1/kev-dev-v1.1-vs-v1.0.json](comparison/v1.1/kev-dev-v1.1-vs-v1.0.json).
- [comparison/v1.1/kev-dev-v1.1.json](comparison/v1.1/kev-dev-v1.1.json).
- [comparison/v1.1/kev-published.json](comparison/v1.1/kev-published.json).
- [comparison/v1.1/kev-test-v1.0.json](comparison/v1.1/kev-test-v1.0.json).
- [comparison/v1.1/kev-test-v1.1-vs-v1.0.json](comparison/v1.1/kev-test-v1.1-vs-v1.0.json).
- [comparison/v1.1/kev-test-v1.1.json](comparison/v1.1/kev-test-v1.1.json).
- [comparison/v1.1/speed-exllamav3-1.6.2-fp32.json](comparison/v1.1/speed-exllamav3-1.6.2-fp32.json).
- [comparison/v1.1/speed-exllamav3-1.6.2.json](comparison/v1.1/speed-exllamav3-1.6.2.json).
- [comparison/v1.1/speed-exllamav3-master-fp32.json](comparison/v1.1/speed-exllamav3-master-fp32.json).
- [comparison/v1.1/speed-exllamav3-master.json](comparison/v1.1/speed-exllamav3-master.json).
- [comparison/v1.1/speed-v1.0.json](comparison/v1.1/speed-v1.0.json).
- [comparison/v1.1/speed-v1.1.json](comparison/v1.1/speed-v1.1.json).
- [comparison/v1.1/vision-v1.0.json](comparison/v1.1/vision-v1.0.json).
- [comparison/v1.1/vision-v1.1-v1-head.json](comparison/v1.1/vision-v1.1-v1-head.json).
- [comparison/v1.1/vision-v1.1.json](comparison/v1.1/vision-v1.1.json).

## v1.1 Head

[Results and measured scope](../v1.1-head-results.md).

The [training recipe](../train-v1.1-head.md) gives WANDS preparation,
native feature extraction, replay import, fitting, calibration and evaluation
commands. Image records use SugarCrepe `replace_att` pairs on COCO val2017,
both caption orders, with 84 earlier COCO images excluded. Captions and
images are acquired separately. Image decisions are experimental.

- [v1.1-head/bitrate-api.json](v1.1-head/bitrate-api.json): Paired 4.05/3.05 bpw capacity and retained-path checks.
- [v1.1-head/cached-kev-comparison.json](v1.1-head/cached-kev-comparison.json): Raw v1.0/v1.1 heads on shared historical Kev features, including text Noul K2.
- [v1.1-head/final-frozen-v1-current-engine-kev.json](v1.1-head/final-frozen-v1-current-engine-kev.json): Final frozen-v1 head/current-engine Kev run, 2260/3089 correct.
- [v1.1-head/frozen-v1-current-engine-kev.json](v1.1-head/frozen-v1-current-engine-kev.json): Earlier frozen-v1 head/current-engine Kev run, 2253/3089 correct.
- [v1.1-head/heldout-wands.json](v1.1-head/heldout-wands.json): Held-out WANDS metrics and extraction counters by option count.
- [v1.1-head/text-calibration.json](v1.1-head/text-calibration.json): Seven fitted text calibration strata.
- [v1.1-head/training-selection.json](v1.1-head/training-selection.json): Matched training selection, both initialization arms and all epochs.
- [v1.1-head/v1.0-recipe-seed-range.json](v1.1-head/v1.0-recipe-seed-range.json): Recipe seed range (context).
- [v1.1-head/visual-eval.json](v1.1-head/visual-eval.json): Experimental SugarCrepe replace_att frozen EVAL.
- [v1.1-head/visual-selection.json](v1.1-head/visual-selection.json): Experimental SugarCrepe replace_att FIT/DEV image setting.

## v1.1 Integration

[Results and measured scope](../integration-v1.1.md).

- [integration/v1.1/api-long-prompt-checks.json](integration/v1.1/api-long-prompt-checks.json): Concurrent long-prompt completions, generation and eight-candidate Choice.
- [integration/v1.1/decision-root-concurrency.json](integration/v1.1/decision-root-concurrency.json): Four questions decided one at a time and concurrently, before and after the decision-root scheduling fix. On the final v1.1 sources, concurrent and sequential results differ by up to 0.048 raw / 0.042 calibrated probability (0.089 / 0.080 before the fix); each order repeats exactly ([method and limits](../integration-v1.1.md#decision-root-checks)).
- [integration/v1.1/decision-root-correctness.json](integration/v1.1/decision-root-correctness.json): Initial varying-geometry discrepancy and four-coexisting-root matched-geometry control ([method and limits](../integration-v1.1.md#decision-root-checks)).
- [integration/v1.1/decision-root-timing.json](integration/v1.1/decision-root-timing.json): One before/after SystemOne timing pair, grouped (first request after server start) and concurrent, with the release-candidate runtime/plugin sources held fixed ([timing scope](../performance-v1.1.md#decision-root-timing)).
- [integration/v1.1/decision-root-timing-warm.json](integration/v1.1/decision-root-timing-warm.json): Four grouped and concurrent SystemOne timing samples on the final v1.1 sources with launcher defaults; sample 0 is the first decision traffic after server start.
- [integration/v1.1/final-f1.json](integration/v1.1/final-f1.json).
- [integration/v1.1/final-f2-long.json](integration/v1.1/final-f2-long.json).
- [integration/v1.1/final-f2-short.json](integration/v1.1/final-f2-short.json).
- [integration/v1.1/final-f3.json](integration/v1.1/final-f3.json).
- [integration/v1.1/mixed-http-cancellation.json](integration/v1.1/mixed-http-cancellation.json): Final v1.1 sources; nine-candidate decision, six short chats and bounded cancellation of an uncapped streaming chat; HTTP cleanup and subsequent responsiveness.
- [integration/v1.1/source-installation.json](integration/v1.1/source-installation.json): v1.1 release-candidate source installation and API outcomes ([installation record](../component-install-smoke.md#v11-source-installation)).

## v1.1 Prefill

[Results and measured scope](../performance-v1.1.md).

- [performance/v1.1/final-fresh.json](performance/v1.1/final-fresh.json).
- [performance/v1.1/final-loo-w4.json](performance/v1.1/final-loo-w4.json).
- [performance/v1.1/final-prefill-upstream-master-results.json](performance/v1.1/final-prefill-upstream-master-results.json): Five-cell matched prefill aggregate against ExLlamaV3 master.
- [performance/v1.1/final-w5.json](performance/v1.1/final-w5.json).
- [performance/v1.1/upstream-master-fresh.json](performance/v1.1/upstream-master-fresh.json).
- [performance/v1.1/upstream-master-w4.json](performance/v1.1/upstream-master-w4.json).
- [performance/v1.1/upstream-master-w5.json](performance/v1.1/upstream-master-w5.json).

## v1.0 Six Quality Panels

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
- The [later NVIDIA synthetic trace](repeat/nvidia-later-trace.json) records
  a maximum probability delta of 0.10429 and about 47% solo-query feature drift.

## v1.0 TEST And Processed Quality

- [Native TEST result](test/result.json): `native-results-v1/result.json`;
  LM-assisted, frozen-LM and feature-only systems, with paired intervals.
- [Kev-9B v2 TEST aggregates](test/kev-9b-report.json):
  [`fam-breadth-test-report/report.json`](https://github.com/jaredpalmer/kev/blob/kev-1.0/runs/fam-breadth-test-report/report.json)
  at `kev-1.0`, selected Kev-9B v2 system. Kev: results published by the repo owner.
- [Processed quality](processed-quality.json): 15 tracks, 27 checks per head
  variant; 23 pass and the 4 VL-RewardBench image checks fail
  ([analysis](../processed-quality.md)). Includes paired group-bootstrap intervals.
  In the VL-RewardBench conversion the preferred response is first in 1,226 of
  1,247 items (98.3%), so accuracy mostly reflects how often the head answers
  Yes or picks the first response
  ([answer order](../processed-quality.md#vl-rewardbench-answer-order)).

## v1.0 Short Performance

Each source copy keeps its summary, warmups and measured timing samples. The
vLLM files are superseded pre-fix observations, kept only in the archive.

- [native3-small-performance-v2.json](performance/native3-small-performance-v2.json)
- [native4-small-performance-v2.json](performance/native4-small-performance-v2.json)
- [native5-small-performance-v2.json](performance/native5-small-performance-v2.json)
- [vllm-nvidia-small-performance-v2.json](performance/vllm-nvidia-small-performance-v2.json)
- [vllm-radixark-small-performance-v2.json](performance/vllm-radixark-small-performance-v2.json)
- [vllm-exl3-small-performance-v2.json](performance/vllm-exl3-small-performance-v2.json)
- [v1.0-short-workload-results.json](performance/v1.0-short-workload-results.json) ([performance](../performance-v1.0.md))

## v1.0 Integration And Generation

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
- [v1.0/integration-results.json](integration/v1.0/integration-results.json)
- [v1.0/integration-timing-samples.json](integration/v1.0/integration-timing-samples.json)
- [v1.0/mtp-diagnostics.json](integration/v1.0/mtp-diagnostics.json)

[Recorded configs](../configs/README.md) and
[integration results](../integration-v1.0.md) describe the workloads.

## v1.0 Capacity

[Native EXL3 4.05 capacity record](capacity/native4-serving-capacity-v1.json)
copies `native4-serving-capacity-v1.json`: a 262,144-token window and an
82,111,987,200-byte peak allocation. It measures serving capacity only.
