# FuZev Research Archive

Methods, evaluation and research history for [FuZev](https://github.com/PeisongXiao/fuzev), a
shared decision head for Choice, yes/no (Noul) and Score tasks.

- **Train a head:** [Train One Shared Decision Head](docs/train-a-shared-head.md) covers
  feature extraction, fitting, calibration and loading the bundle.
- **Serve:** [component setup links](docs/modular-setup.md) and the
  [API reference](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md).
- **Published head:** [Hugging Face head card](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev).

**Contents**

- [Results](#results)
- [Contents](#contents)
- [Limitations](#limitations)

## Results

| Study | Measured scope | Results |
| --- | --- | --- |
| Cross-engine quality | Six configurations, 5,075 requests each, on pre-release research servers | [Benchmarks](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md) |
| Held-out training-source and external quality | 25,599 records; 23 of 27 checks passed, including all text checks; both external visual tasks failed | [Processed quality](docs/processed-quality.md) |
| Generation and decision coexistence | NVIDIA NVFP4 and turboderp EXL3 4.05-bpw checkpoints | [Integration checks](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md) |
| Latency and variation | Small repeated request sets on GB10 | [Performance](https://github.com/PeisongXiao/fuzev/blob/main/docs/performance.md) |

[Evaluation](docs/research-evaluation.md) covers engine builds, batch geometry,
image coverage, populations and interpretation.

## Contents

- [Training history](docs/research-training.md) and [research history](docs/research-history.md)
- [Evaluation methods and reproduction](docs/research-evaluation.md)
- [Serving compatibility](docs/serving-compatibility.md)
- [ExLlamaV3 prefill optimizations](docs/exllamav3-prefill-optimizations.md)
- [Shared-runtime extraction](docs/shared-runtime-extraction.md)
- [vLLM attachment method](docs/vllm-adapter-extraction.md)
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
- Image coverage and quality: [evaluation](docs/research-evaluation.md).
- Current input limits and supported engine configurations:
  [API limitations](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#limitations).

Licenses:

- Code and docs: [Apache-2.0](LICENSE). Learned head: Apache-2.0.
- `research-snapshot/historical-native-source/`:
  [MIT](research-snapshot/historical-native-source/LICENSE).
- Backbone, dataset and other third-party terms:
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
