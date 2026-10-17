# Evaluation Methods

Covers the v1.0 head (`head.safetensors` SHA-256 prefix `f7bc42e7`);
the v1.1 head is in [v1.1 head results](v1.1-head-results.md).

This page explains the populations and reproduction boundary.

- [Benchmark results](benchmarks-v1.0.md):
  six configuration tables, per-task metrics and Kev DEV/TEST comparisons.
- [Performance page](performance-v1.0.md):
  timings and recorded spread.
- [Source aggregates](results/README.md): every quality benchmark run, TEST
  result, processed check, timing source and capacity record.

## What Was Measured

All six runs used the same 5,075 requests: 2,000 FuZev requests (1,400 text
and 600 image) and 3,075 questions from Kev's `breadth-v1` DEV split. All six
completed without response-validation errors, using the fixed v1.0 head at
raw temperature 1.

- Servers: pre-release builds of the archived research evaluation servers.
- ExLlamaV3 quality runs scored up to eight answer options per forward batch;
  the published plugin submits each option as its own Generator job.
- vLLM readout batches under its scheduler settings.

[Limitations](#limitations) collects the current serving scope.

## Populations And Metrics

Training DEV, Kev DEV, Kev TEST and FuZev's 25,599-record held-out assessment
are different sets. Kev DEV contributes to the six runs; Kev TEST is a separate
comparison on the native EXL3 4.05-bpw configuration.

Kev comparison:

- Uses the full published `breadth-v1` TEST table.
- Backbone, quantization and training budget differ.
- Marginal intervals are unpaired; the head versus its own frozen-LM control is paired.
- The benchmark page has the control, feature-only and per-dataset results.

Task notes:

- BFCL and SATA-Bench use case-exact scoring, plus question-micro accuracy as a
  descriptive metric.
- QASPER answerability, evidence and yes/no are separate tasks. Evidence
  sampling uses the recorded inclusion weights.
- The two VL-RewardBench methods use the same examples and are not independent
  populations. [Processed quality](processed-quality.md) records the visual failures.
- v1.0 research read VL-RewardBench TEST results repeatedly; the Noul
  relative-preference adapters were added afterwards with wording modelled on
  VL-RewardBench's Noul prompt. No VL-RewardBench rows or labels entered fitting,
  DEV selection or calibration.

Variation:

- Bootstrap intervals measure sampling uncertainty.
- The [benchmark repeat probes](benchmarks-v1.0.md#repeat-probes)
  give measured repeat variation, including the later NVIDIA trace and the
  shared 670-request comparison across three runs.
- Native ExLlamaV3 decision probabilities are identical for repeated requests
  within one running server. They can differ slightly between server starts,
  because Triton picks kernel configurations by timing at first compile. They can
  also differ with which prefix pages are already cached.
- Probability changes and decision-index points have different units.
- Performance comparisons use the per-run ranges published beside the timings.

## Reproduce

| Study | Retained method or command | Inputs and limits |
| --- | --- | --- |
| New-backbone fit and calibration | [Portable training commands](train-a-shared-head.md) | Caller-supplied FIT/DEV/CAL features and backbone config |
| v1.0 head refit | [Original wrappers and dependencies](study-code/README.md), [training history](research-training.md#reproduce) | Five fixed CUDA epochs; original membership rows, feature corpus and admission ledgers required |
| Full held-out quality | [Calibration/quality wrapper](study-code/calibrate_frozen_reference.py), [snapshot evaluator](../research-snapshot/docs/evaluation.md) | Fixed evaluation plan, clean FIT/DEV completion records, prepared groups, head and CAL/TEST feature corpora |
| Six HTTP runs | [Panel clients](study-code/README.md#six-http-benchmarks), [server and panel configs](configs/README.md#quality-and-head-studies) | Original request bodies, expected answers, image files and historical engine binaries required; client/server binding digests are retained |
| Serving smoke and latency | Commands below | Running component server, selected model and writable output directory |

- [historical-study-settings.json](historical-study-settings.json) records the
  available counts and settings.
- [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) has dataset revisions.
- [Study code](study-code/README.md) records retained wrappers, specialized
  dependencies and input availability.
- The index links aggregate results.

### Running-Server Checks

Start an engine with the [component setup](modular-setup.md). From the archive
root, set the served model and a writable output directory:

```sh
MODEL=your-served-model
OUT=./http-results
mkdir -p "$OUT"
python tools/smoke_head.py --url http://127.0.0.1:8000 \
  --model "$MODEL" --output "$OUT/smoke.json"
python tools/small_http_performance.py --url http://127.0.0.1:8000 \
  --model "$MODEL" --output "$OUT/latency.json"
```

- Add `--api-key` when the server requires authentication.
- The smoke test exercises generation, SystemOne decisions, retained prefixes
  and release.
- The latency tool reports end-to-end HTTP samples, median and range.

## Limitations

- The v1.0 six-configuration runs used pre-release builds; served v1.1 results
  are in [the served comparison](v1.0-v1.1-comparison.md) and
  [head results](v1.1-head-results.md).
- The three vLLM quality runs predate two vLLM-fork changes that affect decision
  features: CUDA-graph replay of per-layer embeddings, and KV-cache slot mapping.
  They were not repeated on the published build.
- ExLlamaV3 image requests use the application's vision callback. The recorded
  v1.0 GPU quality runs used an archived research image path. v1.1 image
  decisions are an experimental extension; GPU results are in
  [visual EVAL](v1.1-head-results.md#visual-eval) and
  [the served comparison](v1.0-v1.1-comparison.md#images).
- The [API reference](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#limitations)
  defines current image inputs and supported engine configurations.
