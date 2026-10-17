# FuZev Research Archive

Methods, evaluation and research history for [FuZev](https://github.com/PeisongXiao/fuzev/tree/v1.1), a
shared decision head for Choice, yes/no (Noul) and Score tasks.

- **Train a head:** [Train One Shared Decision Head](docs/train-a-shared-head.md) covers
  feature extraction, fitting, calibration and loading the bundle.
- **v1.1 head:** [training recipe](docs/train-v1.1-head.md) and [results](docs/v1.1-head-results.md).
- **Serve:** [component setup links](docs/modular-setup.md) and the
  [API reference](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md).
- **Published head:** [Hugging Face head card](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev/tree/v1.1);
  [v1.0 head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev/tree/v1.0).

**Contents**

- [What's New In v1.1](#whats-new-in-v11)
- [Results](#results)
- [Contents](#contents)
- [Limitations](#limitations)

## What's New In v1.1

- The [v1.0 and v1.1 served comparison](docs/v1.0-v1.1-comparison.md) reports Kev
  breadth-v1 DEV and TEST, experimental image decisions, and prefill and decode speed
  on one NVIDIA GB10 (128 GB unified memory).
- [Train The v1.1 Shared Head](docs/train-v1.1-head.md) and
  [v1.1 head results](docs/v1.1-head-results.md) cover head 1.1.0, including held-out
  WANDS up to 255 options and the experimental SugarCrepe `replace_att` image setting.
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) credits the added datasets.
- [Decision-root checks](docs/integration-v1.1.md#decision-root-checks) decide four
  same-prefix questions concurrently and one at a time. On the final v1.1 sources the
  two orders differ by up to 0.048 raw and 0.042 calibrated probability (0.089 and
  0.080 before the decision-root scheduling fix); each order repeats exactly.
- [v1.1 matched prefill](docs/performance-v1.1.md) compares five workloads with
  ExLlamaV3 master on one NVIDIA GB10 (128 GB unified memory).
- The [v1.1 source-installation record](docs/component-install-smoke.md#v11-source-installation)
  covers the release candidate built from unpacked source directories: 21 of 23 API
  checks pass; the 64- and 255-option Choice requests return HTTP 200 but miss the
  check's synthetic expected winners.
- Fourteen new [research tools](tools/README.md) reproduce the v1.1 head, the served
  comparison and the speed runs.
- FuZev 1.1 serves through the ExLlamaV3 components in
  [component setup](docs/modular-setup.md); the vLLM pages record v1.0 research. The
  FuZev index's v1.0 [benchmark](docs/benchmarks-v1.0.md),
  [integration](docs/integration-v1.0.md) and [performance](docs/performance-v1.0.md)
  pages are kept here.
- VL-RewardBench figures carry an
  [answer-order note](docs/processed-quality.md#vl-rewardbench-answer-order): the
  preferred response is first in 1,226 of 1,247 items (98.3%), so accuracy mostly
  reflects how often a system answers Yes or picks the first response. The
  [evaluation guide](docs/research-evaluation.md) states that v1.0 research read
  VL-RewardBench TEST results repeatedly; no VL-RewardBench rows or labels entered
  fitting, DEV selection or calibration.

[FuZev releases](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/releases.md) lists
component versions and upgrade steps.

## Results

| Study | Measured scope | Results |
| --- | --- | --- |
| v1.0 head cross-engine quality | Six configurations, 5,075 requests each, on pre-release research servers | [Benchmarks](docs/benchmarks-v1.0.md) |
| v1.0 head held-out training-source and external quality | 25,599 records; 23 of 27 checks passed, including all text checks; both external visual tasks failed | [Processed quality](docs/processed-quality.md) |
| v1.0 head generation and decision coexistence | NVIDIA NVFP4 and turboderp EXL3 4.05-bpw checkpoints | [Integration checks](docs/integration-v1.0.md) |
| v1.0 head latency and variation | Small repeated request sets on one NVIDIA GB10 (128 GB unified memory) | [Performance](docs/performance-v1.0.md) |
| v1.0 and v1.1 served comparison | Kev breadth-v1 DEV/TEST, experimental image decisions, prefill and decode on one NVIDIA GB10 (128 GB unified memory) | [Full comparison](docs/v1.0-v1.1-comparison.md) |
| v1.1 head quality and capacity | Held-out WANDS K26/64/128/255, experimental SugarCrepe EVAL, cached text retention and paired 4.05/3.05 API checks | [Head results](docs/v1.1-head-results.md) |
| v1.1 integration | Long prefills, in-process generation/decision checks and decision-root checks on one NVIDIA GB10 (128 GB unified memory) | [Integration](docs/integration-v1.1.md) |
| v1.1 matched prefill | Five workloads against ExLlamaV3 master on one NVIDIA GB10 (128 GB unified memory) | [Performance](docs/performance-v1.1.md) |

Head-quality and earlier API decision results were measured on the v1.1
release candidate before the
[decision-root scheduling fix](docs/integration-v1.1.md#decision-root-checks);
prefill and decode do not use that path.

[Evaluation](docs/research-evaluation.md) covers engine builds, batch geometry,
image coverage, populations and interpretation.

## Contents

- [v1.0 and v1.1 comparison](docs/v1.0-v1.1-comparison.md)
- [v1.1 integration](docs/integration-v1.1.md) and [prefill performance](docs/performance-v1.1.md)
- [v1.0 integration](docs/integration-v1.0.md) and [performance](docs/performance-v1.0.md)
- [Training history](docs/research-training.md) and [research history](docs/research-history.md)
- [Evaluation methods and reproduction](docs/research-evaluation.md)
- [Serving compatibility](docs/serving-compatibility.md)
- [ExLlamaV3 prefill optimizations](docs/exllamav3-prefill-optimizations.md)
- [Shared-runtime extraction](docs/shared-runtime-extraction.md)
- [vLLM attachment method (v1.0 research)](docs/vllm-adapter-extraction.md)
- [Source provenance and glossary](docs/research-provenance.md)
- [Curated research snapshot](research-snapshot/README.md)
- [Curation manifest](curation-manifest.json)

The snapshot keeps research methods with curated paths, docs and package names.
[Provenance](docs/research-provenance.md#historical-settings) explains its historical settings.

The [tool inventory](tools/README.md) lists the portable trainer, feature
extractor and clients, with their dependencies and inputs. Serving launchers
and engine builds live in the component repositories.

## Limitations

- Experimental research release.
- Tested platform: NVIDIA GB10. Other hardware needs compatible dependencies
  and enough memory for the model and cache.
- Reproduction needs equivalent hardware, data and software.
- Acquire or produce raw datasets, images, feature tensors, per-item predictions,
  optimizer state and backbone weights separately.
- Image input is experimental. Image coverage and quality: [evaluation](docs/research-evaluation.md).
- Current input limits and supported engine configurations:
  [API limitations](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#limitations).

Licenses:

- Code and docs: [Apache-2.0](LICENSE). Learned head: Apache-2.0.
- `research-snapshot/historical-native-source/`:
  [MIT](research-snapshot/historical-native-source/LICENSE).
- Backbone, dataset and other third-party terms:
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
