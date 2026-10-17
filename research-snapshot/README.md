# FuZev Research Snapshot

Covers the v1.0 head (`head.safetensors` SHA-256 prefix `f7bc42e7`);
the v1.1 head is in [v1.1 head results](../docs/v1.1-head-results.md).

Curated source and method notes for frozen-backbone decision-head experiments.
The research Python package is `fuzev_research`. For a new backbone, use the
archive's [training walkthrough](../docs/train-a-shared-head.md); for serving, use the
[component setup](../docs/modular-setup.md).

The snapshot keeps feature extraction, dataset adapters, training, evaluation,
runtime experiments and a read-only experiment dashboard, with curated paths and
method notes. [Provenance and historical settings](../docs/research-provenance.md)
explain the original experiments' source revisions, vocabulary and limits.

## Method Map

| Topic | Method |
| --- | --- |
| Data and split ownership | [Dataset adapters](docs/dataset-adapters-v2.md) |
| External data and weighting | [External evaluation](docs/external-evaluation-v2.md) |
| Canonical comparison inputs | [Comparison preparation](docs/comparison-inputs.md) |
| ExLlamaV3 extraction | [Held-out extraction](docs/comparison-extraction.md) |
| vLLM extraction | [Comparison extraction](docs/vllm-comparison-extraction.md) |
| Training and resume | [Training rehearsal](docs/training-rehearsal.md) |
| Offline quality metrics | [Evaluation](docs/evaluation.md) |
| Runtime timing | [Measurement definitions](docs/runtime-measurement.md) |
| HTTP recovery | [Resumable evaluation](docs/e2e-resume.md) |

## Feature Geometry

- The reference reads 2,560-dimensional final hidden states before the LM head.
  Query label logits come from the existing LM projection.
- Candidate branches share an immutable query prefix and continue independently.
- Extraction experiments used candidate batches of four and eight. The native
  quality study used batch eight; the v1.0 Generator attachment uses batch one.
- Different batch geometries select different kernels and can change features.
  Batch-1/4/8 drift was not recorded.
- Same-geometry repeats and cross-batch comparisons are distinct measurements;
  see [evaluation interpretation](../docs/research-evaluation.md).

A fitting-only extraction comparison timed 32 rows per arm. Execution order and
mixed warm repeats confound its nominal 5.529% aggregate improvement, so it does
not establish a general throughput improvement.

## Data And Training

- **Training data:** HelpSteer2, HelpSteer3, MM-RLHF and BoolQ. Related
  conversations, passages and image clusters belong to one role.
- **Roles:** FIT supplies gradients, DEV development checks, CAL temperature
  fitting and TEST evaluation.
- **External-only data:** QASPER contributes 239 yes/no, 1,450 answerability and
  2,000 sampled evidence records with explicit inclusion weights. VL-RewardBench
  supplies 1,247 examples, each adapted as Choice and as relative-preference Noul
  (two methods on the same examples).
- **Model inputs** exclude teacher explanations and gold rationales.
- **Development search:** text warmup, then mixed training with text replay;
  seeds 17/29/43, learning rates 0.0003/0.001 and Brier weights 0/0.1.

[Training history](../docs/research-training.md) records the v1.0 head's
later five-epoch refit.

## Working With The Snapshot

Use a separate Python environment. Set `SNAPSHOT` to this unpacked source
directory and `FUZEV_ROOT` to a separate experiment-data directory:

```sh
export SNAPSHOT="$PWD"
export FUZEV_ROOT="$PWD/../experiment-data"
mkdir -p "$FUZEV_ROOT"
python -m venv .venv-research
. .venv-research/bin/activate
python -m pip install "$SNAPSHOT[test,data]" torch transformers
```

Tested with torch 2.13.0 (CUDA 13.0 build).

`fuzev-research dashboard --port 8000` serves a read-only view of experiment
state. The browser-check tool needs the optional `dashboard` dependencies
(`python -m pip install "$SNAPSHOT[dashboard]"`).

In v1.0, `fuzev-vllm-plugin` and `fuzev-exllamav3-plugin` superseded the historical
pre-1.0 research service, which does not run against the published engine forks. Its
modules under `fuzev_research/` are `server.py`, `native_service.py`,
`vllm_server.py`, `vllm_service.py`, `service_contracts.py`, `service_images.py`
and `vllm_images.py`.

Some modules require unpublished lab engines:

- `fuzev_research/vllm_service.py`, `kev_vllm_capture.py` and `vllm_receipts.py`
  require the lab vLLM module `vllm/v1/fuzev_readout.py`.
- `native.py`, `native_performance_operator.py` and
  `tools/probe_extraction_concurrency.py` require the full lab ExLlamaV3 engine
  with `exllamav3/fuzev.py`.

The CPU serving fixture reads copies of `exllamav3/fuzev.py`,
`exllamav3/constants.py` and `tests/test_fuzev_lanes.py` from
[historical-native-source](historical-native-source/) (MIT); see
[trained serving](docs/trained-serving-pipeline.md#recorded-runs). The
[reproduction map](../docs/research-evaluation.md#reproduce) lists available inputs.

## Licenses

Source: [Apache-2.0](LICENSE); `historical-native-source/` is
[MIT](historical-native-source/LICENSE). Backbone, dataset and template attribution:
[third-party notices](../THIRD_PARTY_NOTICES.md).
