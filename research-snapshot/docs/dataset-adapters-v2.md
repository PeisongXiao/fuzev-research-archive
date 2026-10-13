# Dataset Adapters V2

Contract revision: `2026-10-02-v2`. This document defines the adapters; the
scope-transition report records actual preparation, auditing and feature reuse.
Adapters do not change numerical feature encoding or inspect model predictions.

## Outputs and Ownership

| Prepared file | Split manifest | Ownership |
| --- | --- | --- |
| `text.jsonl` | `text-splits.json` | HelpSteer2/3 training sources; unchanged generator |
| `boolq.jsonl` | `boolq-splits.json` | BoolQ training source |
| `images.jsonl` | `image-splits.json` | MM-RLHF training source only |
| `qasper-external.jsonl` | `qasper-external-splits.json` | QASPER official test only |
| `vl-rewardbench-choice.jsonl` | `vl-rewardbench-choice-splits.json` | VL-RewardBench official test, Choice |
| `vl-rewardbench-noul.jsonl` | `vl-rewardbench-noul-splits.json` | The same VL examples, relative-preference Noul |

Prepared files live under `data/prepared`; manifests under `data/manifests`.
All manifests include `records`, `path`, and content `sha256`. New scoped
manifests include `dataset_scope_revision` and `evaluation_scope`. The legacy
HelpSteer manifest and JSONL format are not retroactively changed.

The HelpSteer `stable_id`, conversation parsing, prompt grouping, role mapping,
record constructor, ordinal/preference adapters, and `text_records` generator
are AST-identical to unpublished lab commit `c907eb5`. A nine-row fixture also verifies byte-identical
IDs, record order, roles, targets, prompts, and JSON serialization. This preserves
the adapter side of reuse for the 115,021 extracted HelpSteer2 rows; migration,
runtime provenance and feature compatibility are separate checks.

## BoolQ

The official [Google HF card](https://huggingface.co/datasets/google/boolq/blob/35b264d03638db9f4ce671b711558bf7ff0f80d5/README.md)
declares string `question`/`passage`, Boolean `answer`, train/validation Parquet
splits, and CC-BY-SA-3.0. The default acquisition revision is the absolute commit
`35b264d03638db9f4ce671b711558bf7ff0f80d5`, not a floating branch. Existing absolute
manifest pins are preserved. See also the
[original Google dataset](https://github.com/google-research-datasets/boolean-questions).

`True` maps to `[1, 0]` over `Yes, No`; `False` maps to `[0, 1]`. The adaptation
is `boolean_truth`, never answerability. The prompt contains only the unchanged
passage and original question; `original_answer` is metadata, not prompt text.
Non-Boolean labels are rejected rather than interpreted by Python truthiness.

Groups hash the passage after NFC normalization and whitespace collapse, without
the question or label. Rendering keeps the original passage. Every validation
passage owns the test role; overlapping training passages are excluded even if
their questions differ. Remaining train groups use the existing 80/10/10
fit/development/calibration mapping and seed `20261001`.

## Images and VL Pairs

MM-RLHF retains its existing Choice adaptations, annotation ranks/ties, and
70/10/10/10 fit/development/calibration/test role mapping. All MM examples in a
cluster containing an official VL image are excluded, including MM test-role
examples. No external row enters `images.jsonl`.

Active grouping reads only image-index references owned by `mm-rlhf` or
`vl-rewardbench`. An A-OKVQA-only digest cannot become a component root, connect
two active components, or mark a component held out. Orphan image entries also
do not participate. Every indexed VL reference is protected even if its old
heldout flag is missing. Before preparing MM rows, all IDs from the pinned
official VL test Parquet files must resolve in the index. The existing five-band
perceptual-hash lookup, exact Hamming verification, distance bound of four, and
conservative connected-component grouping are unchanged.

`prepare_images()` and `prepare_vl_rewardbench()` open the existing SQLite index
read-only. They neither re-decode images nor add index entries. Explicit image
indexing is separate (`python -m fuzev_research.images index`) and includes only the two
active image sources. Historical raw acquisitions and A-OKVQA index entries are
not deleted. The old pure A-OKVQA adapter remains a historical helper, but active
acquisition, indexing, preparation, and scoped record validation reject it.

The [VL-RewardBench card](https://huggingface.co/datasets/MMInstruction/VL-RewardBench#data-fields)
defines `human_ranking` as preferred-response order. Choice keeps the original
question, response order, targets, and `official_heldout_preference` adaptation.
Noul asks whether response A is preferred to response B and uses adaptation
`pairwise_preference_yes_no`. It is not a claim that A is absolutely correct or
B is incorrect. Neither adapter includes rationale, judge, or ground-truth
fields in the prompt.

Both outputs contain one record per unique official example ID, in the same
source order. Identical renderings with distinct official IDs remain distinct;
the visual record ID includes the original example ID. The shared
`source_response_pair_sha256` hashes the query and both responses in original
order. Pair verification uses
`(source_test_example_id, image_sha256, source_response_pair_sha256)`, not the
adapter record IDs, which necessarily differ. Both manifests also store a shared
ordered-example-ID digest. No assumed count from a mutable dataset card replaces
the count of examples at the acquisition manifest's absolute revision.

### Official VL Row Identity

The pinned official test has 1,247 rows but 1,238 distinct raw image/question
IDs. Nine IDs recur with different response pairs and the same indexed pixels.
Both adapters retain every row. `source_test_example_id` hashes the raw ID and
the original query/response-pair checksum; `source_example_id` retains the raw
ID and `source_image_id` keeps the original image lookup. Repeated identical
pairs fail rather than silently collapsing or excluding official rows.

## QASPER Frame

Only the locally acquired
[original official test archive](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz)
is read. Its SHA256 must equal
`72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72`.
There is no train/dev fallback, automatic download, or heldout replacement.
The loader selects exactly one bounded `qasper-test-v0.3.json` member without
extracting archive paths onto disk.

A bounded read-only inspection found 416 papers, 239 yes/no annotations, and
1,451 raw answerability annotations. The two questions
`25e6ba07285155266c3154d3e2ca1ae05c2f7f2d` and
`b9d9803ba24127f91ba4d7cff4da11492da20f09` in paper `2004.03090` have identical
rendered answerability prompts and targets. Deduplicating rendered adaptations
therefore gives exactly 1,450 answerability records. The census unit is a unique
rendered paper/question/target adaptation, not a raw annotation occurrence.
`source_question_ids` retains aliases; `source_question_id` is their lexical
minimum. `source_test_example_id` is that original question ID. No arbitrary row
is dropped merely to reach the required count.

Answers remain annotation-vote distributions. Evidence considers distinct
full-text paragraphs for questions with at least one answerable annotation;
its target is the fraction of answerable annotations citing that exact paragraph.
The raw evidence frame has 64,454 records across 415 papers and 1,372 questions:
61,369 zero-vote, 1,931 mixed-vote, and 1,154 unanimous-positive records. Exact
render duplicates are removed before sampling. The final unique frame and all
duplicate counts are recorded in the split manifest.

Preparation fails unless the unique census has exactly 239 yes/no and 1,450
answerability records, the evidence sample has exactly 2,000 unique records,
and publication retains all 3,689 rows. Insufficient pools and duplicate final
identities fail instead of silently shortening the evaluation set.

## Evidence Sampling

`paper-question-label-systematic-srs-v1` uses fixed seed `20261002`. It uses no
model scores, predictions, features, or outcome-dependent reruns.

1. Stratify the unique population by `(paper group, source question ID, label)`.
   Labels are `no` for target Yes=0, `yes` for Yes=1, and `mixed` for 0<Yes<1.
2. Allocate an expected budget of 2,000 equally across papers, capped by each
   paper's population. Redistribute excess from capped papers. Within each paper,
   repeat across questions, then across their nonempty label strata. Quotas are
   exact rational numbers, positive for every nonempty stratum and no greater
   than stratum populations.
3. Sort strata by paper/question/label. Let `q_h` be a stratum's expected quota,
   `C_h` its cumulative quota, and `L` the least common multiple of all quota
   denominators. Draw `R` uniformly from integers `0,...,L-1`. Allocate
   `k_h = floor(C_h + R/L) - floor(C_(h-1) + R/L)` records to stratum h.
4. Uniformly sample `k_h` distinct records from its canonically ID-sorted `N_h`
   records, using the seeded RNG. Emit in deterministic stratum/ID order.

This gives exactly 2,000 rows. Each contiguous paper/question block receives the
floor or ceiling of its expected quota, so large papers or negative-label pools
cannot consume the budget simply by occurring first. Some questions or labels
can be absent in one realization when their quota is below one; they still have
positive inclusion probabilities. Realized global label counts need not match.

The exact marginal probability for every record in stratum h is
`pi_h = E[k_h] / N_h = q_h / N_h`. The weight is `1 / pi_h`, **not** the realized
`N_h / k_h`, a universal frame/sample ratio, or a claimed probability for biased
round-robin selection. The manifest includes every stratum's population, realized
count, rational quota/probability, and systematic offset/denominator, including
unsampled strata. Per-row numerator/denominator fields preserve exact pi.

For the unique evidence frame of size N, a Horvitz-Thompson mean is
`sum(sample_weight * loss) / N`. A self-normalized weighted mean is a ratio
estimator, not exactly the same estimator. Balanced unweighted results describe
this sample, not the population. Sampling is dependent and clustered; first-order
weights alone do not justify independent-row standard errors or confidence
intervals. Report the census subtypes separately from sampled evidence.

## Metadata Preservation

All external rows have `role="test"`, `original_split="test"`,
`source_original_split="test"`, `evaluation_scope="external"`, and a nonempty
`source_test_example_id`. VL and QASPER census rows have
`inclusion_probability=1`, `sampling_weight=1`, and `sampling_design="census"`;
they intentionally have **no** `sampling_stratum`. All sampled evidence rows have
a JSON-encoded stratum triple and finite reciprocal probability/weight.

The feature store must retain these adapter fields in addition to its
normal identity, group, primitive, adaptation, target, and rendering fields:

- Scope: `dataset_scope_revision`, `dataset_purpose`, `evaluation_scope`.
- Provenance: `original_split`, `source_original_split`, `source_test_example_id`,
  `source_example_id`, `source_question_id`, `source_question_ids`,
  `source_paper_id`, `evidence_id`, `original_answer`.
- Sampling: `sampling_weight`, `inclusion_probability`, `sampling_stratum`
  (evidence only), `sampling_cluster`, `sampling_design`, `sampling_seed`,
  `sampling_label`, `population_size`, `sample_size`,
  `inclusion_probability_numerator`, `inclusion_probability_denominator`.
- Images: `image_path`, `image_sha256`, `source_image_id`, `annotation_id`,
  `official_holdout_overlap`, `source_response_pair_sha256`, `query_source`,
  `label_semantics`.

`sampling_cluster` is the paper/image group. QASPER census population/sample sizes
describe each subtype's unique census. Evidence sizes describe the entire unique
evidence frame and the fixed sample, not the local stratum. VL sizes describe the
official example census. Full stratum-frame metadata lives in the split manifest,
not repeated in each feature-store row.

## Verification

Only temporary fixtures were prepared. The focused CPU command, with model
downloads, plugin autoload, bytecode writes, and pytest cache disabled, is:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider tests/test_scope_data.py tests/test_scope_images.py tests/test_external_sampling.py tests/test_data.py
```

Result: 41 passed, including `tests/test_images.py`, which the published
snapshot omits. Tests cover baseline byte parity, official BoolQ schema and
passage holdouts, acquisition history retention, exact QASPER census/sample
counts, archive pin verification, rational sampling expectations and empirical
marginals, A-OKVQA bridge/holdout exclusion, VL overlap protection, read-only
index use, duplicate-render official IDs, and paired external outputs.
[Snapshot methods and environment](../README.md)
