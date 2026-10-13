# Offline Final Evaluation

The offline evaluator compares a fixed head and its label-token control on
complete, aligned held-out feature corpora.

## Inputs and Freeze

The **reference** uses `fuzev_research.evaluate.require_final_ready`, retaining
training readiness for HelpSteer2/3, BoolQ and MM-RLHF plus external readiness
for the QASPER sample and both VL-RewardBench adapters, zero extraction failures,
committed shards, consistent group ownership and unique shard-row locations.
Reference extraction covers all 250,480 v2 records and all roles; fitting,
development selection and calibration retain their separated roles. The existing
prepared roots/manifests remain available under `FUZEV_ROOT` for verification.

**Comparison cells** use `require_comparison_complete`. They may use either a
complete full corpus or an explicitly identified corpus containing **every
canonical final-test record only**. The full reference is the authority for
membership. This reduces comparison extraction from 250,480 records per cell to
the complete test-set count, without needing new fit/development/calibration
features for applying the same frozen head. A pilot or sample cannot qualify.

Freeze one calibrated mixed-stage reference bundle before inspecting final-test
scores. Independently record SHA-256 for `head.safetensors`, `manifest.json`,
`calibration.json` and each feature corpus's `identity.json` in an evaluation
plan. The evaluator verifies those pins; it does not silently establish new
pins from whatever files happen to be present. Keep this plan with the report.
For a test-only comparison, also independently pin `subset-extraction.json`.

The head must match the complete native-reference feature identity, preparation
identity, fitting/development/calibration memberships and recorded training and
calibration code hashes. Checkpoint-only or per-cell adapted heads are rejected.
Every cell uses the same FP32 weights and positive calibration temperatures.
Raw heads use temperature 1; calibrated heads use the frozen temperatures;
the untouched label-token control always uses temperature 1.

The selected test rows are matched by exact ID, group, source/revision, role,
primitive, modality, adaptation/attribute, ordered candidates (ID, text, ordinal
meaning) and target. Candidate order is never repaired. Shard sidecar metadata,
mask, target and ordinal tensors must agree. Missing baselines fail, rather than
using the feature store's placeholder zero logits. All rows scored are test
roles; other roles are inspected only as provenance metadata, not predictions.
Prepared-file checks verify the original input identity. Checksums of consumed
shards are recorded; concurrent corpus/artifact changes invalidate the run.

## Test-Only Comparison Producer Contract

A dedicated comparison directory uses the unchanged `FeatureStore` shard/index
format. Set `identity.json`'s `corpus_scope` to `evaluation-test-only`, retaining
the actual engine, weights/revision, execution, rendering and feature provenance.
Write only canonical test records, with their real label-token controls. Keep the
`preparations` table empty: full prepared inputs have not been extracted and must
not be marked completed. Such a corpus is **evaluation-only, not train-ready**.
`require_complete` explicitly rejects its scope; the existing training readiness
checks independently reject its missing full preparations.

After the final shard/index commit, checkpoint the SQLite index so no committed rows
remain in a nonempty WAL. Write `subset-extraction.json` with exactly these fields:

| Field | Required Value |
| --- | --- |
| `schema_version` | `1` |
| `artifact_kind` | `evaluation-test-only` |
| `roles` | `["test"]` |
| `reference.feature_identity_sha256` | SHA-256 of the full reference's `identity.json` |
| `reference.preparation_sha256` | `fuzev_research.evaluate.sha256(require_complete(reference))` |
| `reference.test_membership` | Complete `canonical_test_membership(reference)` result |
| `extraction` | `{"completed": true, "examined": N, "committed": N, "failures": 0}` |
| `ownership.feature_identity_sha256` | SHA-256 of this comparison's `identity.json` |
| `ownership.index_sha256` | SHA-256 of this comparison's checkpointed `index.sqlite3` |
| `ownership.shards` | All committed shards, in ascending shard order |

Here `N` is the authoritative complete reference test-record count. Each shard
entry contains exactly `shard`, `records`, `tensors_sha256`, and `metadata_sha256`,
pinning the matching `shard-NNNNNN.safetensors` and `shard-NNNNNN.json` files.
Neither a self-reported count nor a copied canonical digest establishes coverage:
the evaluator independently compares every ID/group/ordered candidate/target
against the full reference before predictions. It checks zero recorded failures,
absence of non-test roles, all row locations, sidecar/tensor dimensions, complete
shard ownership, and actual file checksums. Missing or extra records fail.

`canonical_test_membership` reads semantic metadata in ID order and returns record
count, unique group count, SHA-256 over canonical JSON of the `SEMANTICS` fields,
and per-track counts. This identity is independent of engine and physical shard
order. The reference's full readiness and frozen head/calibration pins remain
mandatory at evaluation time. This module performs no subset extraction itself.

Calibration adaptation is a separate future experiment requiring all canonical
calibration **and** test records, an explicit adapted artifact and its own
provenance. This test-only mode rejects calibration rows, cannot fit temperatures,
and never substitutes adaptation for frozen-reference transfer.

## Plan and CLI

Example structure, with deliberately non-executable checksum placeholders:

```json
{
  "schema_version": 1,
  "reference": {
    "features": "/absolute/path/to/reference-features",
    "feature_identity_sha256": "PIN_BEFORE_TEST",
    "head": "/absolute/path/to/frozen-reference-head",
    "sha256": {
      "head.safetensors": "PIN_BEFORE_TEST",
      "manifest.json": "PIN_BEFORE_TEST",
      "calibration.json": "PIN_BEFORE_TEST"
    }
  },
  "cells": [
    {
      "id": "native-4.05",
      "status": "evaluate",
      "features": "/absolute/path/to/reference-features",
      "feature_identity_sha256": "PIN_BEFORE_TEST"
    }
  ]
}
```

An evaluated comparison entry for test-only features has one additional pin:

```json
{
  "id": "native-3.05",
  "status": "evaluate",
  "features": "/absolute/path/to/complete-test-only-features",
  "feature_identity_sha256": "PIN_BEFORE_TEST",
  "subset_manifest_sha256": "PIN_BEFORE_TEST"
}
```

The corpus scope and subset manifest must both be explicit. Supplying a subset
pin for an unmarked/full corpus fails. Reports identify `corpus_scope` and retain
the subset manifest and its pin separately from full preparation readiness.

All six required cells are always reported: `native-3.05`, `native-4.05`,
`native-5.05`, `vllm-nvidia`, `vllm-radixark`, `vllm-exl3-4.05`. Omitted cells
are `not_run`, never inferred unsupported. Optional cells may be listed too.
An unavailable entry instead uses `id`, `status`, `reason`, and `evidence`.
Statuses `unsupported`, `memory_infeasible`, and `failed` need explicit evidence;
`not_run` needs a reason. Declared status is distinguished from an observed
offline error. An offline memory bound is not evidence of GPU infeasibility.
The actual engine, weight revision and execution identities accompany results;
plan cell names alone do not certify tensor equivalence or loader support.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m fuzev_research.evaluate --plan /absolute/path/to/frozen-evaluation-plan.json \
  --output /absolute/path/to/new-final-evaluation.json --threads 4
```

Use the [snapshot environment](../README.md#working-with-the-snapshot),
tested with Torch 2.13.0 (CUDA 13.0 build). This produces JSON plus a readable Markdown
report, refusing to overwrite existing reports. CLI success means the report
was written. Inspect
`coverage_complete` and `all_required_quality_gates_pass`; neither establishes
generation correctness, performance, full-window feasibility or publication
readiness. No dashboard/global state or source files are modified.

Functions `load_reference`, `require_comparison_complete`,
`canonical_test_membership`, `evaluate_cell`, `evaluate`, `compare_tracks` and
`markdown_report` can be reused offline. For test-only corpora, pass
`subset_manifest_sha256` to `evaluate_cell`. Callers must use a frozen CPU bundle;
the report records the PyTorch thread count.
CLI defaults: batch 256, four CPU threads,
2,000 bootstrap draws, seed 17. Bounds: 200,000 test records per cell, 128 tracks,
256 MiB per feature shard, 64 MiB per metadata file and per-reader tensor cache,
1..256 batch size, 1..10,000 draws. Cells are processed sequentially. Metrics
retain compact arrays/group sufficient statistics, not the full feature corpus
or a draws-by-records tensor. Bootstrap reductions do not invoke BLAS thread
pools. Exceeding a bound fails explicitly; no silent subsampling or truncation.

## Metrics and Criteria

Task-track keys follow training: source / primitive / modality /
attribute-or-adaptation-or-main. Every applicable `metrics.py` metric is
reported: NLL, normalized NLL, Brier, ECE, unique-gold accuracy, expected
accuracy, tie-aware accuracy, ordinal MAE and ranked-probability score.
QASPER evidence includes precision/recall/F1 and TP/FP/FN/eligible counts.
These are FuZev adaptations, not official QASPER QA scores.

Unique-gold accuracy excludes tied targets. Expected accuracy preserves soft
target mass; tie-aware accuracy accepts any maximally preferred candidate.
These are distinct metrics, not forced wins. Exactly tied evidence votes do
not supply a hard binary gold label. Evidence threshold stays fixed at 0.5.
F1 is recomputed from aggregate TP/FP/FN for each group resample, never averaged
from made-up per-record F1. ECE is also recomputed from resampled bin statistics.

All comparisons are head minus label-token. One shared draw of complete groups
is used across methods and tracks, preserving shared prompt/paper/image-cluster
dependencies. Within tracks estimates are record-weighted. Macro normalized NLL
weights tracks equally, not by dataset size. Its CI is bootstrapped jointly,
not an average of track CIs. Missing tracks in a draw are not silently dropped.
Reports include eligible groups and valid draws. Fewer than two eligible groups,
or undefined draws, make a criterion inconclusive rather than manufacture support.

The predeclared criteria are unchanged and have no CLI overrides:

- Choice/Noul accuracy and tie-aware accuracy: lower paired 95% CI >= -0.02.
- Evidence F1: lower paired 95% CI >= -0.02, using the grouped count statistic.
- Ordinal MAE: upper paired 95% CI <= +0.1.
- Macro normalized NLL improvement: upper paired 95% CI strictly < 0.

Raw and calibrated criteria are separate; frozen calibration is the primary
configuration. Every applicable track criterion must pass. An interval wholly beyond
the unacceptable boundary fails; a boundary-crossing or unsupported interval is
inconclusive. These are unadjusted two-sided percentile intervals, not a claim
of simultaneous family-wise confidence. No best-on-test configuration selection
or automatic adaptation is performed. Adaptation experiments require separate,
explicitly identified tooling and cannot replace this reference-transfer result.
## Dataset Scope Revision

The v2 scope in `external-evaluation-v2.md` is authoritative for new runs.
Training readiness covers only HelpSteer2/3, BoolQ and MM-RLHF; final evaluation
also requires all three external preparations. QASPER is original-test-only
(239 yes/no, 1,450 answerability, 2,000 weighted evidence). VL-RewardBench uses
the same 1,247 official rows in separate Choice and pairwise-preference Noul
reports; nine repeated raw image/question IDs are disambiguated by their
response-pair checksum without dropping examples. Historic corpus counts and
QASPER training references above do not apply to the active scope.

[Snapshot methods and environment](../README.md)
