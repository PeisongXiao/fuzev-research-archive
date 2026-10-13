# Training Readiness Rehearsal

Synthetic CPU lifecycle checks exercised optimizer resume and calibration.

## Existing Coverage and the Actual Gap

Inspected `tests/test_training_orchestration.py`,
`tests/test_training_readiness.py`, and `tests/test_heads_and_features.py` first.
They already cover readiness hashes/counts, missing inputs, role ownership,
individual artifact integrity, variable candidate padding, completed-trial
reuse, and missing/corrupt optimizer states. Those tests were not copied.
The last module imports the native backend, so it was deliberately not executed
in this CPU-only check.

The new rehearsal fills two gaps:

1. The real `train_when_ready.main -> fit(text) -> fit(mixed) -> calibrate`
   path, through actual SQLite/Safetensors feature storage and local task state.
2. A durable mid-trial interruption followed by real optimizer restoration and
   numerical comparison against an uninterrupted control.

## What Runs

`tools/rehearse_training.py` creates 48 synthetic records across text, QASPER-like
Noul, and image-modality tracks, all with 2,560-wide feature vectors. "Image"
here is metadata on random features, not image/encoder validation. Choice, Noul,
and ordinal heads use their actual 256-wide projections and **3,933,696 FP32
parameters**. Candidate widths vary. Each role has distinct group identities.

The actual 12-profile loops run for both text and mixed stages: seeds 17/29/43,
learning rates 0.0003/0.001, and Brier weights 0/0.1. Every eligible synthetic
fit/development row remains mandatory. Mixed profiles initialize from the
development-selected text checkpoint, with its weight digest verified for all
12 profiles. Development alone selects the winners; calibration alone fits
the three positive temperatures. Head parameters and argmax predictions are
unchanged by calibration. The portable artifact is loaded using `load_bundle`.

**Only the synthetic full-matrix trial calls override `max_epochs=1`.** Their
provenance says one epoch; the harness checks that the real caller still uses
the production default of ten. Batch size 256, patience two, epoch seed rule,
optimizer, loss, selection, and membership verification are unchanged. The
paired resume control is explicitly a separate two-epoch synthetic trial.
Neither is a substitute production schedule.

The environment exposes no CUDA devices and uses one intra-op and one inter-op
Torch thread. The synthetic main replaces resource admission/MemoryGuard with
a no-sampling stub; it does not claim to qualify live memory safety. All paths
resolve under the snapshot directory's `.pytest_cache`. The CLI uses a temporary
directory with automatic cleanup; new pytest fixtures also remove their large
artifacts on success/failure. Maximum checked artifact volume is 2 GiB.

## Results

Post-fix CPU rerun: **25 passed in 10.31 seconds**, with no xfail. This includes
the now-passing patience-stop regression, all 12 text and 12 mixed profiles,
calibration/artifact checks, and the nonterminal tensor-exact resume control.
The original pre-fix run was 24 passed and one strict expected failure in
10.57 seconds (initial substantive run: 10.42 seconds).

The standalone CLI also completed and removed its temporary artifacts:

| Check | Observed |
| --- | --- |
| Runtime | Historical CUDA build image, Torch `2.13.0+cu130`, CPU only |
| Full profile orchestration | 12 text + 12 mixed profiles, all actual one-epoch head fits |
| Resume | Interrupt after committed epoch 1; restore 12 parameter states; continue epoch 2 |
| Resume equality | All final head tensors, optimizer tensors, losses and development results exact; wall time excluded |
| Calibration | All head tensors and argmax predictions unchanged; artifact reload successful |
| Feature mutation | Synthetic Safetensors shards byte-identical before/after fitting |
| Test separation | Zero test-row selections for fitting, selection, or calibration |
| CLI elapsed | 6.35 seconds, not a reference training-speed forecast |
| CLI temporary files | 1,342,801,228 bytes before cleanup; directory removed on return |

Test-row selection is the asserted boundary. Feature shards can contain
multiple roles; loading a shard may read co-resident test bytes, but those rows
are not selected into loss, optimizer updates, selection metrics, or calibration.
This rehearsal does not claim physically split-only disk I/O.

Synthetic calibration diagnostics:

| Primitive | Temperature | Calibration NLL before | After |
| --- | ---: | ---: | ---: |
| Choice | 0.337392 | 0.841140 | 0.804275 |
| Noul | 100.000008 | 1.097991 | 0.897305 |
| Ordinal | 1.062707 | 1.220237 | 1.220158 |

Noul reaches the existing 100-temperature clamp, with FP32 rounding slightly
above 100. Random synthetic features do not justify any conclusion about real
Noul quality or any change to the clamp.

## Confirmed Bug and Isolated Fix

In base unpublished lab commit `0113e3e`, `fuzev/train.py:226` commits an epoch and its `stale`
counter before result logging and the patience check at line 229. If interruption
occurs there with `stale >= 2`, `completed` remains false. On resume, line 190
enters the next epoch without first honoring the already-reached stop condition.

Reproducer: constant synthetic development score, genuine default ten-epoch
budget, patience two. Interrupt after the epoch-three commit. The saved state
contains three epochs and `stale=2`; resume executes epoch four. If a new epoch
improved sufficiently, it could also alter the eventual selected checkpoint.
This is a resume schedule-equivalence bug, not an observed failure of the
uninterrupted live run.

The original reproducer was a strict xfail accepting only the specific
`PatienceResumeRegression` exception, not unrelated fixture failures.

The fix checks the already-committed `stale >= 2` before
any resumed epoch work. Existing latest-checkpoint loading, optimizer integrity
checks, and common trial finalization remain in place. The original
end-of-epoch stop check is unchanged. No profile, membership, objective, epoch
budget, patience, or checkpoint-selection rule changes.

`test_interruption_after_patience_stop_must_not_run_another_epoch` now passes
without xfail. After the same interruption, training and validation scheduling
are forbidden by the test. Resume commits completion with the original best
checkpoint, not the latest checkpoint, and changes no epoch/checkpoint files.
The saved result matches the returned result; another resume is idempotent.
The separate nonterminal interruption still reproduces the uninterrupted
head/optimizer tensors and loss/development results exactly.

## Primary-Paper Review

### Guo et al., ICML 2017

[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html),
especially [section 4.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf):
temperature scaling fits a positive scalar against held-out NLL and preserves
the argmax. Their setup assumes train/validation/test distributions match;
empirical success is not a guarantee for a new domain.

Actual mapping: `train.fit_temperature` uses LBFGS over log-temperature,
clamped to [0.03, 100], with padded candidates masked before division.
`calibrate` fits one temperature per primitive using only calibration groups,
after mixed-checkpoint selection. This project's separate development and
calibration sets are stricter than reusing one validation set.

Risk/check: the same temperature mixes text/image sources and candidate widths.
A finite temperature does not prove calibration under runtime or quantization
shift. Keep the existing per-track and six-cell validation; do not retune on
test. Positive scaling preserves argmax, **not ordinal expected value or MAE**.
The rehearsal verifies unchanged parameters and argmax, and records NLL before/
after without adding an acceptance criterion.

### Gneiting and Raftery, JASA 2007

[Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf),
sections 2.1 and 3: logarithmic and quadratic/Brier scoring incentivize truthful
probabilities; positive constant scaling preserves propriety.

Actual mapping: `train_trial` minimizes NLL divided by log(valid candidate
count), plus nonnegative Brier weight. `validate` selects by a macro average
over source/primitive/modality/attribute tracks. `fit_temperature` instead uses
ordinary per-row NLL for each primitive, without that track macro average or
candidate-count normalization.

Risk/check: those are deliberately different weightings, not a proof that the
selected head and calibrated head optimize an identical aggregate objective.
Our inference from the code: positive normalization per fixed candidate set
preserves the conditional scoring target, but changes relative finite-model
weighting across examples. The new test proves the actual selection rule and
role boundary; retain existing NLL/Brier/per-track reporting rather than change
losses, profile counts, or selection criteria from this literature review.

### Kumar et al., ICLR 2022

[Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054),
[sections 2-4](https://arxiv.org/pdf/2202.10054): preserving useful pretrained
features can help under distribution shift, while full fine-tuning can distort
them. The analysis has feature-quality/distribution assumptions and does not
establish that frozen features are always superior.

Actual mapping: FuZev optimizes only the heads over stored feature tensors.
`CandidateHead` is a two-projection low-rank bilinear scorer, **not the paper's
ordinary linear probe**. Text warmup followed by mixed head fitting is also
not the paper's linear-probe-then-backbone-fine-tune procedure.

Risk/check: frozen weights do not make features from changed quantization,
batch geometry, or inference kernels identical. Byte-unchanged stored feature
shards and exact resume are useful lifecycle evidence, not transfer-quality
evidence. The paper does not establish cross-runtime feature equivalence.

## Reproduce

From the snapshot directory, create the cache parent before pytest:

```bash
mkdir -p .pytest_cache
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_training_readiness.py \
  tests/test_training_orchestration.py
```

The published snapshot omits `tests/test_training_rehearsal.py`.

## Limitations

One-epoch synthetic profiles test scheduling and packaging, not convergence,
real-image quality or full-corpus performance. Exact CPU resume comparisons
apply to the measured architecture and dependency versions. Timing spread was
not recorded for the individual invocations above.

[Snapshot methods and environment](../README.md)
