# Train The v1.1 Shared Head

This is the completed seed-43 matched-budget recipe for the shared text head.
It reuses [train_shared_head_v2.py](../tools/train_shared_head_v2.py) on frozen,
role-separated ragged features, then fits text temperatures separately.
The backbone is never trained or loaded by these head-only commands.
[Saved selection, CAL and quality](v1.1-head-results.md).

## Inputs And Environment

Use runtime 1.1 and the archive tools with Torch, NumPy and safetensors.
The curated training record does not retain Python, NumPy or safetensors
versions.
The [general environment guide](train-a-shared-head.md#environment) covers setup;
the archive's `research-snapshot` must be on the Python module path for metrics.
The recorded run used archive unpublished lab commit ID `d7d7ee4`,
runtime unpublished lab commit ID `1fc9054`, FP32 CUDA,
autocast disabled and OMP/MKL 1.
The recorded hardware is NVIDIA GB10.

The initializer is the [v1.0 head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev/tree/v1.0):
H 2560/projection 512/query-key adapters 64/16. Keep source weights and input
features read-only. Both arms share that geometry; existing-init strictly
loads its learned weights, scratch uses the seeded initializer, and both
start fresh AdamW state.

Use the [text target config](configs/v1.1-train-config.json).
Native [WANDS](https://github.com/wayfair/WANDS) has 719 FIT / 186 DEV /
103 CAL / 189 TEST feature rows, 1197 total;
original replay retains soft targets and its original role/rendering contracts.
The final-pre-LM, last-token readouts are owned FP32 features.
WANDS uses complete product name+description, Exact=2/Partial=1/Irrelevant=0,
and a uniform target over all in-slate Exact candidates.

| Input | FIT | DEV | CAL | Observed K |
| --- | ---: | ---: | ---: | --- |
| Replay Choice | 26885 | 3425 | 3441 | 2 |
| Replay Noul | 8440 | 1047 | 1066 | 2 |
| Replay Ordinal | 81215 | 10330 | 9980 | 5 |
| WANDS Choice | 719 | 186 | 103 | 26/64/128/255 |
| Combined | 117259 | 14988 | 14590 | As above |

Replay contains HelpSteer2 Choice and Ordinal, HelpSteer3 Choice, BoolQ Noul,
HelpSteer2/3 text preference Noul and text dimension-Choice rows from the
v1.0 feature cache. The matched training uses no image rows; MM-RLHF reaches
this head only through the v1.0 initializer. The replay re-includes 12
HelpSteer2 FIT rows that the v1.0 refit quarantined for overlapping a v1.0
DEV group.

Prepared raw data/features, image files and initializer weights are acquired
or produced separately. Keep complete slates and query groups in their
original roles. TEST/EVAL never enters fitting, DEV selection or CAL.

## Prepare Inputs

Fetch the initializer and set operator paths:

```bash
hf download peisongxiao/Qwen3.8-Flash-Next-FuZev --revision v1.0 --local-dir published-v1-initializer
export ARCHIVE=/absolute/path/to/fuzev-research-archive
export BUNDLE="$PWD/published-v1-initializer"
export REPLAY=/absolute/path/to/replay-features
export WANDS_FEATURES=/absolute/path/to/wands-features
export RUNS=/absolute/path/to/head-training-output
export MODEL=/absolute/path/to/Qwen3.8-Flash-Next-exl3-4.05bpw
export WANDS_RAW=/absolute/path/to/wands-csv
export WANDS_PREPARED=/absolute/path/to/wands-prepared
export V1_FEATURE_CACHE=/absolute/path/to/v1-feature-cache
export PYTHONPATH="$ARCHIVE/research-snapshot${PYTHONPATH:+:$PYTHONPATH}"
```

`EARLIER_TEXT`, `EARLIER_BOOLQ`, `EARLIER_NOUL_PREFERENCE`,
`EARLIER_DIMENSION_CHOICE`, `EARLIER_IMAGES`, `QASPER_EXTERNAL`, `VL_CHOICE`,
`VL_NOUL`, `KEV_DEV` and `KEV_TEST` name the prepared JSONL files of the
earlier FuZev data and the Kev inputs; export them the same way.

Acquire WANDS `query.csv`, `product.csv` and `label.csv` separately.
[prepare_wands.py](../tools/prepare_wands.py) groups normalized query text,
then splits groups 60/15/10/15 percent into FIT/DEV/CAL/TEST using the recorded
seed. It excludes exact input overlaps with the supplied earlier preparations
and Kev inputs, conflicting judgments and conflicting duplicate product text.
Only judged products enter nested 26/64/128/255-option slates. Candidates use
complete product names and descriptions; Exact=2, Partial=1, Irrelevant=0,
with uniform target mass over all Exact products. Membership and presentation
use independent seeded shuffles. The output is one prepared JSONL per role,
including candidates, targets, grades and query groups.

```bash
python "$ARCHIVE/tools/prepare_wands.py" --raw "$WANDS_RAW" \
  --exclude-records "$EARLIER_TEXT" "$EARLIER_BOOLQ" "$EARLIER_IMAGES" \
    "$QASPER_EXTERNAL" "$VL_CHOICE" "$VL_NOUL" "$KEV_DEV" "$KEV_TEST" \
  --seed 20261015 --output "$WANDS_PREPARED"
python "$ARCHIVE/tools/extract_shared_head_v2.py" \
  --model "$MODEL" --model-name extraction --head "$BUNDLE" \
  --calibration raw --context-length 65536 --cache-tokens 73728 \
  --max-batch-size 4 --max-chunk-size 8192 --cache-bits 8 8 --mtp-tokens 3 \
  --use-vram 90 \
  --input "$WANDS_PREPARED/fit.jsonl" "$WANDS_PREPARED/development.jsonl" \
    "$WANDS_PREPARED/calibration.jsonl" "$WANDS_PREPARED/test.jsonl" \
  --output "$WANDS_FEATURES" --shard-bytes 33554432
```

`MODEL` is the separately acquired
`turboderp/Qwen3.8-Flash-Next-exl3@55a732e0` 4.05 bpw snapshot.
The extractor uses the public native launcher and saves the exact FP32 inputs
to the shared head. Completed shard pairs are reused on resume. The v1.0
feature cache is not published. Import its text replay with the original
prepared rows and [recorded provenance](head-provenance.json):

```bash
python "$ARCHIVE/tools/ragged_features.py" \
  --legacy "$V1_FEATURE_CACHE" \
  --prepared "$EARLIER_TEXT" "$EARLIER_BOOLQ" \
    "$EARLIER_NOUL_PREFERENCE" "$EARLIER_DIMENSION_CHOICE" \
  --bundle-config "$BUNDLE/config.json" \
  --provenance "$ARCHIVE/docs/head-provenance.json" \
  --output "$REPLAY" --shard-bytes 33554432
```

### Ragged Shards

Each `shard-NNNNNN.safetensors` has matching row-ordered JSONL metadata.
The writer buffers complete slates within `--shard-bytes` and resumes from
completed pairs.

| Tensor | Shape And Meaning |
| --- | --- |
| `query` | `[N,H]`, FP32 |
| `offsets` | `[N+1]`, int64; starts at zero, adjacent differences are K |
| `candidates` | `[sum(K),H]`, FP32, without padding |
| `baseline`, `targets` | `[sum(K)]`, FP32 |
| `grades`, `ordinal_values` | Optional aligned vectors; row metadata records presence |

Metadata preserves IDs, groups, roles, sources, ordered candidates, exact
requests and rendering. Minibatches pad whole slates; K sorting stays within
each fixed logical update.

### Loss

For a complete slate with utility `z`, normalized target `y` and grades:

```text
CE_i = -sum_j y_j * log_softmax(z)_j / log(K_i)
P_i  = {(a,b): grade_a > grade_b}
BT_i = sum_(a,b in P_i) (grade_a-grade_b) * softplus(z_b-z_a)
       / sum_(a,b in P_i) (grade_a-grade_b)
loss_i = CE_i + beta * BT_i
```

BT is zero when grades are absent or no strict pair exists. Equal-grade ties
remain ties. The recorded beta is 1; CAL fits CE only.

## Frozen Matched Budget

| Setting | Both Arms |
| --- | --- |
| Only difference | existing versus scratch initialization |
| Seed/order | 43; existing seed+epoch shuffle |
| Budget | Exactly 5 epochs, 459 updates/epoch, 2295 total |
| Logical batch | 256 rows; final batch 11 |
| Optimizer | AdamW LR 0.0003, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8 |
| Gradient norm | Existing norm-1 clipping |
| Precision | FP32, autocast disabled |
| Replay/new mass | 0.5/0.5 |
| Objective | Soft CE/log(K) +1.0 grade-gap-normalized Bradley-Terry |
| K buckets | 2:16, 17:26, 27:64, 65:128, 129:255 |
| Padded-element budget | 2611200 = 4*255*2560, complete slates retained |
| Pair block | Existing 64, nonreentrant checkpointed backward |

Weighting assigns equal lane mass, then equal populated primitive/modality
tracks, K buckets, distinct source/group units and views. Each WANDS K bucket
receives 1/8 total FIT mass; each original replay primitive receives 1/6.
Row mass sums to 1; trainer weights are N*p and every update divides by 256,
including the final 11 rows. K sorting is within each fixed logical batch.
Both arms visit 117259 FIT rows per epoch, 586295 total.

Each arm selects minimum weighted DEV normalized NLL across its five epochs,
with earlier epoch on ties. Compare those selected values across arms;
an exact tie retains existing-init. No early stopping, LR/seed grid or adaptive
extension accompanies this pair. The saved winner is **existing-init epoch 2**,
DEV NLL **0.6828698238832571** versus scratch epoch 2 **0.6865686332489497**.
Freeze the chosen weights before CAL.

## Matched Training Commands

Use the paths set in [Prepare Inputs](#prepare-inputs):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for INIT in existing scratch; do
  python "$ARCHIVE/tools/train_shared_head_v2.py" train \
    --bundle "$BUNDLE" --config "$ARCHIVE/docs/configs/v1.1-train-config.json" \
    --fit "$WANDS_FEATURES/fit" --replay-fit "$REPLAY/fit" \
    --development "$WANDS_FEATURES/development" \
    --replay-development "$REPLAY/development" \
    --initialization "$INIT" --seed 43 --epochs 5 \
    --learning-rate 0.0003 --weight-decay 0.01 \
    --batch-size 256 --batch-elements 2611200 \
    --pairwise-weight 1.0 --replay-mass 0.5 \
    --k-buckets 2:16 17:26 27:64 65:128 129:255 \
    --device cuda --output "$RUNS/$INIT"
done
```

Each arm writes its selected raw `head/`, full `training.json` and epoch DEV
reports. [Curated training aggregate](results/v1.1-head/training-selection.json)
keeps both saved arms/all epochs without per-row IDs or mass maps.
The calibrate step needs the full training.json from the train step; the
curated aggregate omits per-row masses and cannot replace it.

## Separate Text CAL

The [seven-stratum input](configs/v1.1-calibration-strata.json) uses inclusive
K intervals. Each stratum carries a placeholder temperature that the strata
format requires; the fit starts at T=1.
Fit only after frozen DEV selection. The recorded winner is existing-init:

```bash
SELECTED=existing
python "$ARCHIVE/tools/train_shared_head_v2.py" calibrate \
  --bundle "$RUNS/$SELECTED/head" \
  --training-report "$RUNS/$SELECTED/training.json" \
  --calibration "$WANDS_FEATURES/calibration" \
  --replay-calibration "$REPLAY/calibration" \
  --strata "$ARCHIVE/docs/configs/v1.1-calibration-strata.json" \
  --batch-size 256 --batch-elements 2611200 --calibration-steps 50 \
  --device cuda --output "$RUNS/calibrated-$SELECTED"
```

The deterministic fit uses initial T=1, bounds [0.03,100],
50 LBFGS iterations, strong-Wolfe search and an FP64 scalar objective.
It weights distinct groups equally, then views within group, separately from
FIT/DEV lane weighting. All 14590 CAL rows contribute to the seven fitted
text temperatures; all seven recorded solutions are interior.
Observed K values are 2/26/64/128/255 for Choice, 2 for Noul and 5 for Ordinal.
Interval interiors are not separately qualified.

The trainer's exported metadata keeps validated maxima null and Ordinal 5.
The [final metadata](configs/v1.1-final-head.json) is separate from that original
training artifact. It uses the same selected shared weights and seven text
temperatures; no tensor changes accompany final capacity metadata expansion.

## Held-Out Text Evaluation

With the final head metadata and calibration installed beside the unchanged
selected tensor, set `FINAL_HEAD` as in [Final Bundle Metadata](#final-bundle-metadata).
The held-out command is:

```bash
python "$ARCHIVE/tools/train_shared_head_v2.py" evaluate \
  --bundle "$FINAL_HEAD" --test "$WANDS_FEATURES/test" \
  --calibration calibrated --batch-size 256 --batch-elements 2611200 \
  --device cuda --output "$RUNS/heldout-wands"
```

This produces the per-K metrics and extraction counters in
[heldout-wands.json](results/v1.1-head/heldout-wands.json). Its text strata
and temperatures equal the final calibration's; its `capacities` field is the
training export's (text Ordinal 5, null validated maxima). Its original
feature inputs follow the WANDS preparation above; per-shard hashes are
outside the curated aggregate. The replay initializer and original feature
lineage are identified in [head-provenance.json](head-provenance.json).

## Image Setting And Limits

Image decisions are an experimental extension; run at your own risk.
SugarCrepe `replace_att` pairs (Hsieh et al., NeurIPS 2023) on COCO 2017
validation images, one pair per image, both orders, use seed 20261015.
The sample excludes 84 images whose COCO IDs appear in earlier FuZev image
data (including A-OKVQA metadata). Captions and images are not redistributed.
There are 48 FIT, 80 DEV and 96 held-out image groups, yielding 96, 160 and
192 decision rows. FIT/DEV selects the image residual and temperature on
the v1.1 head (bundle 1.1.0):
lambda=1, T=0.5135668608446433, frozen before the separate 192-view EVAL.
Only Choice/image_text K2 has an exported image CAL entry.
[Visual selection and EVAL](v1.1-head-results.md#visual-selection).

Regenerate the image groups and acquire images into an operator directory:

```bash
export CORPUS=/absolute/path/to/sugarcrepe-corpus
python "$ARCHIVE/tools/prepare_sugarcrepe.py" prepare --output "$CORPUS" \
  --exclude-coco-ids "$ARCHIVE/docs/configs/sugarcrepe-excluded-coco-ids.json" \
  --seed 20261015 --fit-groups 48 --dev-groups 80 --eval-groups 96
python "$ARCHIVE/tools/prepare_sugarcrepe.py" images --output "$CORPUS"
python "$ARCHIVE/tools/visual_capture.py" --model "$MODEL" --model-name extraction \
  --head "$BUNDLE" --calibration raw --vision --context-length 65536 \
  --cache-tokens 73728 --cache-bits 8 8 --max-batch-size 4 --max-chunk-size 8192 \
  --mtp-tokens 3 --use-vram 90 --image-dir "$CORPUS/images" \
  --representation-input "$CORPUS/fit.jsonl" "$CORPUS/development.jsonl" \
  --output "$RUNS/image-fit-dev-capture.json"
python "$ARCHIVE/tools/representation_probe.py" rescore \
  --capture "$RUNS/image-fit-dev-capture.json" --head "$RUNS/existing/head" \
  --fit-records "$CORPUS/fit.jsonl" --dev-records "$CORPUS/development.jsonl" \
  --device cpu --output "$RUNS/image-fit-dev-scores"
python "$ARCHIVE/tools/residual_blend.py" select \
  --fit-records "$CORPUS/fit.jsonl" --dev-records "$CORPUS/development.jsonl" \
  --fit-scores "$RUNS/image-fit-dev-scores/v2-fit-scores.jsonl" \
  --dev-scores "$RUNS/image-fit-dev-scores/v2-development-scores.jsonl" \
  --temperature-min 0.03 --temperature-max 100 --bootstrap-draws 2000 --seed 17 \
  > "$RUNS/image-selection.json"
```

The pinned source revisions and exclusion list are recorded by the preparer;
the resulting membership is fixed before capture. For EVAL, repeat the
`visual_capture.py` command with only `--representation-input
"$CORPUS/test.jsonl"` and a new capture output. Then rescore and report:

```bash
python "$ARCHIVE/tools/representation_probe.py" rescore \
  --capture "$RUNS/image-eval-capture.json" --head "$RUNS/existing/head" \
  --eval-records "$CORPUS/test.jsonl" --device cpu --output "$RUNS/image-eval-scores"
python "$ARCHIVE/tools/residual_blend.py" report \
  --eval-records "$CORPUS/test.jsonl" \
  --eval-scores "$RUNS/image-eval-scores/v2-test-scores.jsonl" \
  --selection "$RUNS/image-selection.json" --bootstrap-draws 2000 --seed 17 \
  > "$RUNS/image-eval.json"
```

## Final Bundle Metadata

The final export keeps `existing/head/head.safetensors` unchanged. Its config
sets the final capacities listed in [head results](v1.1-head-results.md#bundle-and-capacities),
residual strength 1.0 for text and image_text, head ID `fuzev-head-1.1.0`
and release date `2026-10-17`. Its calibration keeps the seven fitted
text strata and appends the selected image Choice K2 temperature. The archive's
[final config](configs/v1.1-final-head.json) and
[final calibration](configs/v1.1-final-calibration.json) are byte-identical
copies of the published bundle metadata. Acquire the final metadata with:

```bash
hf download peisongxiao/Qwen3.8-Flash-Next-FuZev --revision v1.1 --local-dir fuzev-head
export FINAL_HEAD="$PWD/fuzev-head"
```

## Limitations

Visual quality here is evaluated on SugarCrepe attribute-caption pairs.
Overlap with earlier image rows is checked through COCO image IDs only.
Text Ordinal and image Noul/Ordinal quality remain unvalidated;
uncovered admitted requests retain raw learned probabilities.
Seed variance is unmeasured (one seed). WANDS K views share
query groups, and the cached Kev comparison is conditional on its feature cache.
`validated_max` describes measured coverage. Unobserved K values inside a CAL
interval and other tasks need their own quality measurements.
