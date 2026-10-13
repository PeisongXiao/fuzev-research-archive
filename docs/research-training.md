# Training History

How the selected FuZev head was produced. To fit a head to another backbone, use
the [training walkthrough](train-a-shared-head.md).

## Selected Head

The LM-assisted shared scorer has 2,872,401 FP32 parameters in 15 tensors. Its
numerical implementation comes from unpublished lab commit
`9728b98b205d4f9af8ecac4e28ee678ba5ab758b`. Backbone and LM-head parameters
stayed frozen. [Provenance](research-provenance.md) lists artifact hashes.

The feature-only and LM-assisted controls each used:

- 185,178 eligible FIT records, with 12 protected-overlap rows quarantined; DEV
  contained 24,937 records.
- Five CUDA epochs, 3,620 updates, seed 43 and batch size 256.

Both controls passed 26 label-control checks. Component-balanced confirmation
passed 13/15 workloads for feature-only and 12/15 for LM-assisted.

## Data And Calibration

- **Training sources:** HelpSteer2, HelpSteer3, MM-RLHF and BoolQ. Revisions and
  licenses are in the [third-party notices](../THIRD_PARTY_NOTICES.md).
- **Evaluation-only:** Kev, QASPER and VL-RewardBench.
- **Roles:** related conversations, documents and image clusters went to disjoint
  roles. FIT supplied gradients, DEV development checks, CAL temperature fitting
  and TEST held-out reporting.

Calibration fitted three positive temperature scalars:

| Primitive | Temperature |
| --- | ---: |
| Choice | 1.18305242061615 |
| Noul | 1.273818016052246 |
| Score (`ordinal`) | 0.9596070647239685 |

Positive temperature scaling preserves argmax selections but can change expected
Score and probability metrics. The six HTTP quality runs used raw temperature 1.

The full held-out assessment passed 23/27 checks on 25,599 records. All text
checks passed; both external visual tasks failed accuracy and tie-aware accuracy
checks. See [processed quality](processed-quality.md).

## Reproduce

- **Settings:** [historical-study-settings.json](historical-study-settings.json).
- **Refit and calibration wrappers:** [study code](study-code/README.md).
- **Calibration script:** the retained
  `research-snapshot/scripts/calibrate_frozen_reference.py` reads the frozen
  reference bundle, CAL feature shards and prepared evaluation plan under
  `$FUZEV_ROOT`. [Provenance](research-provenance.md) gives paths and source identity.
- **Inputs:** acquire or produce prepared membership and feature shards separately.

For a reusable training baseline, use the new
[portable trainer](../tools/train_shared_head.py);
[train-a-shared-head.md](train-a-shared-head.md) has its executable commands,
feature schemas and a CPU example. The
[snapshot training rehearsal](../research-snapshot/docs/training-rehearsal.md)
documents the optimizer-resume and calibration methods used during development.
