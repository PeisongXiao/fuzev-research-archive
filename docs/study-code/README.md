# Historical Study Code

These code-only records describe the v1.0 head's original workflow.

| Name | Stands for |
| --- | --- |
| `FUZEV_ROOT`, `TRAINING_IMAGE` | Reader-supplied bases for absolute paths |
| `FEATURE_ONLY_TRAINING_SOURCE`, `LM_ASSISTED_TRAINING_SOURCE` | The two retained training sources |
| `FEATURE_ONLY_CANDIDATE_SOURCE`, `LM_ASSISTED_CANDIDATE_SOURCE` | The earlier capacity study's candidates |

- Python imports use the archive's `fuzev_research` namespace.
- The curation removes the lab's clean-worktree, exact-Torch-version,
  single-GPU, CPU-affinity and CUDA-allocation checks. Recorded commit and
  version values stay as values.
- The hash column records the original source bytes, before this curation.
- Commit IDs and historical checksum pins in these files are unpublished lab provenance.

| Study | Retained Source | Original SHA-256 |
| --- | --- | --- |
| Clean FIT refit and endpoint DEV | [refit.py](refit.py) | `e200bbfb01006f5ef58eba8f620ef8b9a67c53ad8897e86da276228ec0e3826d` |
| Component-balanced DEV confirmation | [dev_panel.py](dev_panel.py) | `825243fdbc9a55eddb1e23f6f4d3506eed23dcdfa5a97f91923726e446223a74` |
| Refit epoch implementation | [capacity_train.py](capacity_train.py) | `979fab0ec539b4d6f3c9f4b03a7282b9572a9dd4db046b0606808cb424f87531` |
| Refit weight-reference helper | [admit.py](admit.py) | `f74f81ade3c5cdbe23cdd1322d8e58f9e51ea2067ebf3a1396d728c55ca71393` |
| CAL-only temperatures and processed quality | [calibrate_frozen_reference.py](calibrate_frozen_reference.py) | `ef9d06f3e850183d3b13692d4ecf7e11b382cfa329c3a2af5896b3005a075f0c` |

The v1.0 refit used five epochs and 3,620 updates. It calls
`capacity_train.train_epoch` directly; the dependency's separate eight-epoch
CLI belongs to an earlier study. These historical wrappers use Torch (tested
with 2.13.0+cu130), safetensors, NumPy and the standard-library SQLite module.

The snapshot also keeps the curated
[calibration script](../../research-snapshot/scripts/calibrate_frozen_reference.py)
and [TEST feature/scoring wrapper](../../research-snapshot/scripts/benchmark_kev_text.py).
Their input plan is [frozen-evaluation-plan.json](../configs/frozen-evaluation-plan.json);
[head-study.json](../configs/head-study.json) records FIT/CAL/quality settings.

## Six HTTP Benchmarks

The retained snapshot contains:

- [prepare_kev_e2e_panel.py](../../research-snapshot/scripts/prepare_kev_e2e_panel.py)
- [kev_requests_adapter.py](../../research-snapshot/scripts/kev_requests_adapter.py)
- [kev_e2e_client.py](../../research-snapshot/scripts/kev_e2e_client.py)
- [kev_vllm_e2e_client.py](../../research-snapshot/scripts/kev_vllm_e2e_client.py)
- [kev_e2e_resume.py](../../research-snapshot/scripts/kev_e2e_resume.py)

These use httpx, NumPy, Pillow, Torch/safetensors and the archived research
server modules. Kev's suite and report code come from
[Kev 1.0](https://github.com/jaredpalmer/kev/tree/kev-1.0).

Related records:

- [quality-panel.json](../configs/quality-panel.json): the two populations and their input digests.
- [Each server's configs](../configs/README.md#quality-and-head-studies): the six engine/quantization settings.
- [quality-bindings.json](../configs/quality-bindings.json): client bindings and source hashes.

## Input Availability

Included: wrapper source, server and panel settings, role counts and digests,
module identities, fitted temperatures and aggregate outcomes.

Exact reproduction also needs:

- the original prepared membership rows, feature shards, image files,
  admission/declaration ledgers and historical source trees
- `controls.py`, `folds.py` and `coverage_train.py`, which the refit imports
  from its earlier-study directory, outside this code-only record

The caller supplies host-specific engine binaries and the CUDA build image.
Training-module identities are preserved in
[head provenance](../research-provenance.md#v10-head-provenance).

For a new fit, use the [portable training walkthrough](../train-a-shared-head.md).
