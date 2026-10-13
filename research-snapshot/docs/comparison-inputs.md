# Complete Comparison Inputs

Comparison preparation selects the complete canonical test membership after
reference training and calibration.

## Purpose

The native 4.05 reference extracts the entire eligible 250,480-record v2
corpus and fits/calibrates on its separated splits. Frozen-head transfer to the
other five cells needs every canonical final-test record, not another extraction
of the fitting/development corpus. Nothing is sampled, reduced, re-ranked, or
removed from the declared final evaluation. The v2 QASPER evidence sample is
selected in advance, independent of predictions, with audited marginal
inclusion weights. External datasets never enter training or calibration.

`fuzev_research.comparison_inputs.prepare_inputs` accepts the previously frozen evaluation
plan's `reference` entry. It verifies the full reference corpus, independently
pinned head/manifest/calibration checksums, and mixed/calibrated artifact and
membership provenance before selecting prepared test rows. It loads the small
CPU head only to verify its artifact, never to make predictions or select rows.

It streams the six registered prepared roots in their original order, verifies
their hashes/counts, and copies each selected JSONL line **byte-for-byte**. Thus
context, question, image path/digest, candidates, annotations and every other
field retain the original input; no rerendering or tokenization occurs here.
Canonical test IDs/groups/ordered candidates/targets must match independently
indexed reference metadata. Missing, duplicate, extra or semantically changed
test records fail. At most 16 MiB is accepted per record; the ID set is bounded
by the evaluator's 200,000-test-record limit. The pinned prepared manifests
contain 25,767 eligible test rows: 7,603 HelpSteer, 3,270 BoolQ, 8,711 MM-RLHF,
3,689 QASPER and 2,494 paired VL-RewardBench rows (1,247 examples per adapter).
No tensor corpus is cached in RAM.

The new directory is exposed only after all checks, including the freeze's
post-checks, complete. Existing output directories are never intentionally
overwritten. Failures clean only the tool's private staging directory.

## Outputs

- `test.jsonl`: every prepared canonical test row in unchanged preparation order.
- `comparison-inputs.json`: input hash/bytes/count, canonical semantic membership,
  full-reference/preparation and frozen-head pins, producer code hash, and an
  explicit inputs-only scope.

The output contains annotations and complete input membership. It does not
contain model features or predictions.

Example CPU command after preparing the evaluation plan:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m fuzev_research.comparison_inputs \
    --plan $FUZEV_ROOT/runs/frozen-evaluation-plan.json \
    --output $FUZEV_ROOT/data/comparison-inputs
```

Pin `comparison-inputs.json` independently. Future sequential comparison
extractors consume `iter_inputs(directory, manifest_sha256, reference_spec)`,
which rechecks the input and frozen-reference identity, semantics, completeness
and file stability. A cancelled/partially consumed generator is not completion.
The runtime must still extract owned final-mixer features and genuine untouched
label-token controls for all yielded records, record its actual engine/weights,
and run the membership check on the complete feature corpus using the
separate evaluator contract in [evaluation.md](evaluation.md). Test-only comparison corpora are never train-ready.

## Check Completed Comparison Features

`fuzev_research.comparison_seal` is the separate CPU-only membership check.
First stop or drain the comparison writer. Supply its actual examined-input count; do not
infer this count from successful feature rows or use it to conceal failures.
Set `EXAMINED_COUNT` to the actual number examined by extraction.
The helper refuses the training-reference directory, an existing record,
non-test roles, fabricated preparation completion, failed or extra rows, and
incomplete canonical membership. It checkpoints the comparison SQLite WAL,
pins every owned shard and sidecar, and checks baseline presence and semantic
alignment against the complete reference.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m fuzev_research.comparison_seal \
    --plan $FUZEV_ROOT/runs/frozen-evaluation-plan.json \
    --features $FUZEV_ROOT/data/features/<comparison-cell> \
    --examined "$EXAMINED_COUNT"
```

The record is written atomically without replacing an existing file. Before
returning its SHA256, the helper reruns the evaluator's subset-ownership and
tensor-shape checks and the frozen-reference/head post-checks. On failure it
removes only its own unchanged record, never feature data or another writer's
record. Pin the returned checksum independently in the evaluation plan.

This proves **storage and complete membership only**. Actual runtime/weight
identity, numerical validity of tensor contents, model correctness, frozen-head
quality and six-cell transfer belong to separate qualification and evaluation
checks.

## Verification

Tests use exclusively synthetic inputs/features/head artifacts under the
snapshot's ignored pytest directory. They ran CPU-only (CUDA hidden, one CPU
math thread, offline Hugging Face access) and read no real datasets. The
focused producer/consumer and evaluator/subset/metric suites passed.

[Snapshot methods and environment](../README.md)
