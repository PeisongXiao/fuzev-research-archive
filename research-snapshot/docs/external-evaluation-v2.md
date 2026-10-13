# External Evaluation V2

Evaluation methods for dataset scope revision `2026-10-02-v2`, including
external-only tasks and their sampling weights.

## Data Integration

- The feature identity must retain the complete strict `dataset_scope` object.
  Only identities with that field absent use the legacy synthetic-test path.
- `require_ready("calibrate")` and the frozen head's preparation provenance remain
  training-only: HelpSteer2, HelpSteer3, BoolQ and MM-RLHF. External rows are never
  fitting, development/model-selection or calibration inputs.
- Final evaluation additionally calls `require_external_ready()`. It must verify
  all three completed preparations against their pinned files, manifests, exact
  semantic memberships and committed rows, not only their reported row counts.
  The preparations are `qasper-external.jsonl`, `vl-rewardbench-choice.jsonl` and
  `vl-rewardbench-noul.jsonl`, each with its corresponding `-splits.json` manifest.
- The same full reference FeatureStore may have external test rows appended after
  training. Its identity and training preparation/membership pins must not change.
- Canonical comparison membership includes ALL reference test records, including
  the training sources' official holdouts and all external records. Comparison
  input and membership-check producers must include external preparations, not just training
  readiness's preparations. Both comparison producers use final readiness.
- Persist semantic metadata in both the SQLite index and shard sidecars. In
  particular retain `original_split`, `evaluation_scope`, `source_test_example_id`,
  `sampling_weight`, `inclusion_probability`, `sampling_stratum`,
  `source_question_id`, `evidence_id`, and image identity. Do not reconstruct
  missing external weights as one.

### Exact Metadata

All external rows additionally require `dataset_scope_revision='2026-10-02-v2'`,
`dataset_purpose='external'` and `sampling_design`. For scoped external/BoolQ/MM
rows, semantic checksums and reference/comparison alignment include **every**
field in `features.SCOPE_METADATA` plus the external identities below, not merely
record IDs or target vectors. Imported HelpSteer rows retain the legacy semantic
projection; immutable preparation hashes and producer records bind full inputs.

- QASPER: `source_paper_id`, `source_question_id`, and `source_question_ids` (the
  aliases retained when identical rendered examples are deduplicated). The group
  is `paper:<source_paper_id>`. Evidence membership distinguishes paragraph
  `evidence_id`; multiple paragraphs can share `source_test_example_id`.
- Sampled evidence: `sampling_label`, `sampling_seed`,
  `inclusion_probability_numerator`, `inclusion_probability_denominator`, and
  `sampling_stratum`, a JSON-encoded `[paper_group, question_id, vote_label]`.
- VL: `image_sha256`, `image_path`, `source_image_id`,
  `source_example_id`, `source_response_pair_sha256`, `label_semantics='relative_pairwise_preference'`,
  and `question` are required.
  Both variants must have identical image identity. The original Choice response
  texts must appear as Response A/Response B in that order in the Noul question,
  ending with the explicit A-preferred-to-B question produced by `images.py`.
  The pair checksum is verified against the original query and ordered response
  texts, using the producer's `stable_id(query, json.dumps(responses))` convention.
  `source_test_example_id=stable_id(source_example_id, source_response_pair_sha256)`
  preserves all 1,247 official response pairs despite only 1,238 distinct raw IDs.
  Repeated raw IDs/images keep their original image identity and shared group.

These fields must survive both metadata stores and preparation verification.
`query_source` is optional provenance, not a dispatch input. Imported HelpSteer
metadata need not acquire fields absent from its original extraction; comparison
input producers must not demand new external-only fields on those legacy rows.

## External Tasks

QASPER uses only official test: 239 `yes_no`, 1,450 `answerability`, and 2,000
sampled `evidence` records, 3,689 total. The first two subtypes are censuses
(`sampling_weight = inclusion_probability = 1`). Evidence retains the fixed-seed
paper/question/label sampling design and honest marginal inclusion probabilities.
Sampling weights are positive inverse inclusion probabilities; sampled evidence
must retain its stratum. The three subtypes are separate task tracks, never a
pooled QASPER accuracy or an official generative-QA score.

The supported evidence design is `paper-question-label-systematic-srs-v1`, seed
`20261002`: hierarchical equal water-filled expected quotas, exact systematic
rounding, then SRS within each stratum. Marginal inclusion probability is
**expected quota / stratum population**, not realized sample count / population.
The pinned `qasper-external-splits.json` must contain the complete `sampling`
frame from `sample_evidence()`, including zero-selected strata. Evaluation
recomputes the expected quotas, rational probabilities, systematic denominator
and fixed-seed offset, and verifies stratum/sample counts and labels. Missing or
invented design attestations, unit-weight substitutions, and realized-`n/N`
probabilities fail closed. It does not read the raw population or reselect rows;
preparation verification binds the selected identities.

Censuses declare `sampling_design='census'`. Stratum labels are optional there;
the existing producer labels `yes_no`, `answerability`, and `official_test` are
accepted, but do not imply those censuses were sampled.

VL-RewardBench retains BOTH the original Choice track
(`official_heldout_preference`) and the explicit Noul adapter
(`pairwise_preference_yes_no`) on exactly the same source test examples, groups
and images. Choice response slots are `0`, `1`; Noul Yes means response 0 is
preferred to response 1, and No means the converse. It is relative preference,
not absolute correctness. Both are censuses with weight and inclusion probability
one. Noul must never replace Choice. The adapter producer owns the explicit
pairwise question and preservation of the response texts.

## Statistics

Metrics are normalized weighted ratios within each task track. Counts remain
unweighted counts; eligible weight sums and weight-based effective sample sizes
are reported separately. Evidence precision/recall/F1 and calibration error are
recomputed from weighted sufficient statistics, including within bootstrap draws.

One shared resampling of whole groups applies to every method and task track.
QASPER's unit is the paper, not a paragraph; paired VL representations retain the
same image group. Inclusion weights remain fixed within each draw. These are
weighted paired group percentile intervals, NOT exact finite-population or
multistage survey-design confidence intervals. Effective sample size describes
weight dispersion, not a count of independent papers.

The existing methods, fixed calibration, task criteria and equal-task macro
normalized-NLL criterion remain unchanged. There is no pooled cross-task accuracy,
external calibration, model selection, temperature fitting or runtime qualification.

## Evaluation Configuration

The plan remains schema version 1, with one frozen reference bundle and pinned
feature identities; no new per-cell knobs, fitting or calibration overrides.
An identity with `dataset_scope` present must exactly match `scope.contract()`.
Reference and comparison scopes must match, including test-only comparisons.

`load_reference()` continues to use `require_complete()`, which returns the
training-only readiness object required by the frozen head manifest.
For scoped references it also requires `manifest.feature_origins` to exactly
equal `features.producer_origins()`. The latter verifies original-producer
records and byte-copy/computation provenance for reused features. This is NOT
part of preparation readiness and does not force another engine's producer
origins to match the reference engine. Sources are never relabeled.
`require_final_ready()` is the stronger final-evaluation API: it returns separate
`training` and `external` objects, combined `preparations`, counts and record count.
External readiness includes the attested sampling-frame digest and summary.
Absent-scope legacy fixtures receive the original readiness shape.

`comparison_inputs.py` and `comparison_seal.py` use
`require_final_ready(reference)` when gathering final inputs and computing
`reference.preparation_sha256`. For scoped corpora this pin is
`sha256(require_final_ready(reference))`, NOT the training-only readiness hash.
`reference.test_membership` remains `canonical_test_membership(reference)`, now
also including external subtype/pair counts and the scope-contract digest.
The test-only corpus still has no fake training preparations; the reference's
external preparations are audited explicitly and its exact semantic membership
is required in the checked comparison. The frozen head's own preparation pin is
unchanged. The frozen-reference context also rejects final-readiness or database
version changes before exposing comparison inputs or retaining a new membership record.

The JSON report retains separate task tracks, raw counts, eligible weight sums,
weight sums/squared sums, Kish effective record counts, inclusion-probability
ranges, per-stratum summaries, and the design/variance limitations. Markdown
also shows track counts, groups, weight sums and weight ESS. Neither report
claims design-exact survey uncertainty or changes the existing quality criteria.

## CPU Verification

No model or held-out predictions are needed for these synthetic fixtures:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_scope_evaluation.py tests/test_evaluation_metrics.py tests/test_metrics.py
```

Metadata/design helpers import without Torch. Tensor/head integration needs the separate tensor test environment.

[Snapshot methods and environment](../README.md)
