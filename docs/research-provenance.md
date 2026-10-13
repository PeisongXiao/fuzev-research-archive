# Research Provenance

The archive keeps methods, aggregate results and source records, with curated
paths and process notes.

- Starting research revision: unpublished lab commit
  `48c09bb6592d09702de15d16be252c851429c622`.
- Numerical scorer: unpublished lab commit `9728b98b205d4f9af8ecac4e28ee678ba5ab758b`;
  its implementation is kept in [the research snapshot](../research-snapshot/fuzev_research/shared_head.py).
- Namespaces: `fuzev_research` for the snapshot, `fuzev` for the serving runtime.
- The [curation manifest](../curation-manifest.json) links methods and results.
- Dataset contents, feature shards, optimizer state and original Git metadata
  stay with the original study inputs.

## Released Head Provenance

The [record-only provenance JSON](head-provenance.json) captures the actual
released-reference manifest, including FIT/DEV/CAL membership counts and
digests, training and calibration module identities, prepared-input manifest
digests and feature origins. All its commit IDs are **unpublished lab commit IDs**.

The original manifest belongs to the calibrated reference bundle. Serving
`config.json` and `calibration.json` use the compact runtime format.

| Record | Count / SHA-256 |
| --- | --- |
| FIT ordered membership | 185,178 records; `705923d61d63b1a0e95929110ec706ffbdfa8f1d840aaf313836e3fb3ea717bc` |
| DEVELOPMENT ordered membership | 24,937 records; `8a96b3e5b74dcc9b7fbf33ae406d6b872a4e6134c781cad7cde6456f38d665b1` |
| CALIBRATION ordered membership | 23,813 records; `74a9e672894207b2672a5219f149d3deb7273b249ecd3993cea3648e88946480` |
| Original calibrated-reference `manifest.json` | `5b463f03798f647ae377fd41dcd38add3c2532d13e974bc1684553d2ec7f078f` |
| Historical training/calibration module `features.py` | `c6583d37f865befeedc55151a9a1b37a9f945563bf9299fa65cb909d986d7579` |
| Historical training/calibration module `head.py` | `52882ccbc6009ab0d8aa090f8f0f087cc794503cfb98f7a83696139bddd163ea` |
| Historical training/calibration module `metrics.py` | `78ded3a8847ff3d10ced0fe5452bce72c2000730e48ce52ae73a1ed826380d68` |
| Historical training/calibration module `preference_head_retrain.py` | `2fcaa6c2c919527cd1164058bb36852fae732202845dfe95453b0ccd579e519b` |
| Historical training/calibration module `scope.py` | `69befb3e286726123c1f56802e9783b28e867a2a9fa612f815c463d9fc79c317` |
| Historical training/calibration module `shared_head.py` | `5435eeff154cc7494ca9a214f8a8fdf32012326902cf41da32ab7fe1dac9090c` |
| Historical training/calibration module `shared_train.py` | `f39f7b4342bec4d1b8f3616838d319fb272c3f0a45c0997f7c7558b8337fbaee` |
| Historical training/calibration module `shared_training_data.py` | `4f9364c5d0a421e75ff3cb22f636ac4cf52ac01fdae3a5c107701893234e82c8` |
| Actual refit wrapper | `e200bbfb01006f5ef58eba8f620ef8b9a67c53ad8897e86da276228ec0e3826d` |
| Actual calibration / processed-quality wrapper | `ef9d06f3e850183d3b13692d4ecf7e11b382cfa329c3a2af5896b3005a075f0c` |
| Original head tensors | `f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c` |
| Original reference calibration | `566e0f43fb66fbbd5834d34299a41f3e884f3b306c9a0040d30364c8189d01d3` |
| Historical numerical scorer | `5435eeff154cc7494ca9a214f8a8fdf32012326902cf41da32ab7fe1dac9090c` |
| Serving `head.safetensors`, selected release bytes | `f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c` |
| Serving `config.json`, selected release bytes | `33937ec66536248fc6ae4fc3eeb1df3d835aae5e908d1f8fae3f39a370a7acbe` |
| Serving `calibration.json`, selected release bytes | `24a15bd4fda54411d06ad96689f2a86317d7d259723b946a2d06b75a048a372c` |
| Serving runtime `_scoring.py`, selected release source | `a3b03255d7f9369736d3cc404b2a3c8bc24ed7c0911a03fb36d2f7cbe1f573db` |

`SHA256SUMS` is the head bundle's release checksum list. Verify it by running
`sha256sum -c SHA256SUMS` in the head directory.

## Study Code And Configurations

These records contain code and settings only:

- [Study-code records](study-code/README.md): the original refit and DEV
  wrappers, the calibration/processed-quality wrapper and the six-run panel clients.
- [Recorded study configs](configs/README.md#quality-and-head-studies): refit
  settings, CAL/quality inputs, panel manifest and each server's scrubbed configuration.

Exact dataset membership rows, feature tensors, per-item predictions and prompt
journals are outside this archive. The refit also imports historical
`capacity_train.py` and `admit.py`; the study-code index lists their status.
The [reproduction map](research-evaluation.md#reproduce) gives each study's
retained code and missing inputs.

## Historical Settings

These snapshot settings belonged to the research code on GB10:

- Model, configuration and loader whitelists.
- Exact engine commits, clean-checkout and offline-mode requirements, and fixed
  loopback endpoints.
- Numeric limits, including the 0.85 memory fraction, 15% host-RAM headroom,
  294,912-token KV pool, four-request admission, image and request size caps,
  timeouts and evaluator record limits.

The component docs give the current configurable defaults.

Historical engine hooks and private build images may be unavailable. The retired
standalone ExLlamaV3 path needs the engine revision its study used. The
[reproduction map](research-evaluation.md#reproduce) lists retained tools and
the engine revisions and inputs each historical run requires.

## Name Map

In old records, `src/`, `cross-cell/`, `reports/`, `runs/`, `weights/`,
`worktrees/`, `data/` and `state/` are unpublished lab directories. Examples use
reader-supplied `FUZEV_ROOT` (experiment data) and `SNAPSHOT` (source) instead.

| Historical name | Archive or public component |
| --- | --- |
| `recipes` | `research-snapshot/` |
| `src/exllamav3-fuzev` | [exllamav3-fuzev](https://github.com/PeisongXiao/exllamav3-fuzev) |
| `src/kev-evaluation` | [Kev](https://github.com/jaredpalmer/kev) |
| `vllm-exl3-fuzev` | [vllm-exl3-plugin](https://github.com/PeisongXiao/vllm-exl3-plugin) (AGPL-3.0-only) |

## Glossary

| Term in historical records | Meaning |
| --- | --- |
| Cell | Engine and quantization configuration |
| T1 | Temperature 1, uncalibrated |
| CAL | Calibration split or its fitted temperature scalars |
| B1/B4/B8 | Candidate batch capacity 1/4/8 |
| Source8 | Unpublished lab commit ID `9728b98` |
| MRV1/MRV2 | vLLM GPU model runner V1/V2 |
| PLE | Per-layer embeddings |
| QSA | Qwen sparse attention |
| NativeLane | Retired standalone ExLlamaV3 research execution path |
| Owner | The application's existing engine instance |
| A/B/C | Prompt / generated tokens / new decision input |
| Repeat-A | A repeated request used to measure numerical variation |
| Wave | One repeated workload run |
| Tap | Point where a model's hidden state is read |

## Licenses

- Source and documentation: Apache-2.0.
- Third-party attribution: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- Selected head: [head card](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev).
