# Synthetic Training And Evaluation Integration

Synthetic CPU integration passed actual trained and calibrated artifacts
through the frozen loader and evaluator.

## Gap Covered

Existing offline evaluation fixtures fabricate calibrated-head metadata.
This single end-to-end test instead passes the actual output of
`train_when_ready.main -> fit(text) -> fit(mixed) -> calibrate` directly into the
frozen loader, input preparation, and evaluation. The calibrated manifest comes
from the training pipeline.

The fixture is constructed independently of `rehearse_training.build_corpus`:
that helper's rehearsal-only identity and generic Noul candidates are not the
evaluator contract.

## Fixture and Training

- 48 deterministic synthetic rows, with 2,560-dimensional query/candidate
  features, across the text, qasper, and image preparation slots. Those names
  identify schema slots, not real dataset use. Image modality is metadata on
  random vectors; no image exists or is decoded.
- All four roles have 12 distinct groups. Choice widths are 2/3, ordinal widths
  3/4 with finite increasing values, and Noul uses exactly `yes`/`no`. Noul
  evidence tracks contain both positive and negative synthetic targets.
- Real JSONL roots, split manifests with hashes/counts, SQLite ownership,
  Safetensors shards, and registered/completed preparations establish readiness.
  The complete execution identity is explicitly synthetic from the start.
- Every `FeatureStore.add` receives finite candidate-aligned baseline logits.
  These are a synthetic candidate-index ramp, **not label-token inference**.
  The metadata explicitly records this distinction before training.
- All 12 profiles run for each stage: seeds 17/29/43, learning rates 0.0003/0.001,
  Brier weights 0/0.1. The test observes the real ten-epoch caller default, then
  overrides only these synthetic trial calls to one epoch. Saved provenance
  records one epoch. Production ten epochs, patience two, batch size 256,
  optimizer, readiness optimizations, memory admission, and MemoryGuard remain
  unchanged and active.
- Tensor-row audits require each profile to visit every applicable fit and
  development row exactly once, with text-only warmup. Validation reads only
  development rows in inference mode; calibration reads every calibration row
  once. No final-test row enters fitting, selection, or calibration. Each
  selected trial is the minimum actual development loss, and every mixed trial
  pins the selected text checkpoint. This audits row selection, not physically
  separated disk reads: a shard may contain co-resident roles.

## Frozen Consumers

1. `load_reference` accepts the authentic calibrated head without edits. The
   calibrated head has the same 3,933,696 FP32 parameters as the selected mixed
   checkpoint, and all three temperatures are finite and positive.
2. `prepare_inputs` and fully consumed `iter_inputs` preserve the exact original
   JSONL bytes/order for all 12 canonical tests and keep their head/identity pins.
3. A new test-only FeatureStore copies those 12 synthetic rows, including their
   baselines, into different shard boundaries and reverse input order. There
   are 60 stored rows including copies, below the 64-row bound. The copy has no
   preparation registrations and claims only `evaluation-test-only` scope.
   `seal_features` verifies and pins its real storage and complete membership.
   Its name is `synthetic-copied-features-not-engine-transfer`; it is explicitly
   a fake feature copy, not a weight load, runtime execution, or transfer result.
4. The actual evaluator scores only test rows, with batch size three and eight
   bootstrap draws, solely to exercise the integration contract. Original/copy
   memberships and results agree. The head tensors and temperatures remain
   frozen across both calls. These tiny synthetic bootstrap results have no
   quality or statistical qualification interpretation.
5. The `native-4.05` cell is the evaluator's reference slot populated by the
   synthetic corpus, **not a native-engine run**. It and the explicitly named
   copy cell are `evaluated`; the other five required cells stay `not_run`.
   Every required cell is retained, `coverage_complete` is false, and
   `all_required_quality_gates_pass` is false. No per-cell pass is asserted as
   evidence of real quality.
6. The evaluator writes `synthetic-evaluation.json`; the test checks that the
   head files and report remain unchanged after subsequent validation.
   Appending whitespace to calibration JSON in a separate disposable head copy
   fails the frozen checksum boundary. The source bundle is never tampered with.

Feature/metadata/preparation hashes and reference training-code hashes also
remain unchanged through the whole lifecycle. The output prints only a
synthetic integration summary, not synthetic scores presented as real quality.

## Isolation and Bounds

The pytest test launches a fresh interpreter with a 180-second timeout; no
`import vllm` bootstrap is used. An import guard rejects native/vLLM engine
imports and checks the loaded-module set before and after the lifecycle,
preserving the existing rehearsal's no-engine boundary without changing it.
Torch uses one intra-op and one inter-op CPU thread, with CUDA hidden, HF
offline, and bytecode disabled. The real host memory guard may reject a run
under pressure; the test does not bypass it or claim memory qualification.

All custom files, state, weights, reports, caches, and temporary data live in a
unique `TemporaryDirectory` under the test checkout's `.pytest_cache`. `FUZEV_ROOT`
points to that owned directory before research imports; there is no export
to actual `$FUZEV_ROOT/weights`. The fixture removes its directory on success,
failure, or subprocess timeout. A final artifact-size assertion requires less
than 2 GiB; this is an end-of-run check, not a disk quota. Abrupt termination of
the outer pytest process cannot guarantee Python cleanup.

## Recorded Runs

The published snapshot omits this test.
Initial execution: **1 passed in 9.27 seconds**, 1,183,916,984 temporary artifact
bytes, with cleanup confirmed. Final focused rerun, with
`tests/test_training_orchestration.py` preceding the new test: **3 passed in
9.24 seconds**. This also checks that training imports in the outer pytest
process do not compromise the fresh worker.

[Snapshot methods and environment](../README.md)
