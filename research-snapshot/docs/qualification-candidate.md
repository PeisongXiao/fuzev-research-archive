# Historical Consumer Integration

This record summarizes synthetic CPU integration before the completed quality
study. Current outcomes are linked from the
[archive evaluation guide](../../docs/research-evaluation.md).

The tests exercised real training, calibration, feature preparation and scoring
code with synthetic features and fake engines. They covered separated training
and external evaluation roles, complete comparison membership and unchanged
head tensors through serving.

## Recorded Checks

| Check | Historical observation |
| --- | --- |
| Evaluator/comparison/export tests | 104 cases, 25.26 seconds |
| Training orchestration plus consumer lifecycle | 107 cases, 33.68 seconds |
| Trained serving and review | Two cases, 16.89 seconds; worker peak RSS 811,452 / 822,732 KiB |
| Same two cases in an integrated source tree | 15.82 seconds; worker peak RSS 816,620 / 821,948 KiB |

These are separate single invocations; timing dispersion was not recorded.
They do not measure model quality or inference-engine transfer.

## Methods

The training lifecycle used 48 original synthetic rows and 12 copied test rows,
with all 24 profile fits limited to one epoch in the fixture.
[Training and evaluation integration](trained-review-pipeline.md) describes row
selection, calibration and evaluation. [Trained serving](trained-serving-pipeline.md)
describes actual scorer use with fake engine state and rejection of synthetic
identity by the historical native builder.

The scoped follow-up used four training datasets and external-only QASPER and
VL-RewardBench adapters; see [scoped integration](qualification-scoped-integration.md).
Historical source/settings context is in the [snapshot overview](../README.md).

[Snapshot methods and environment](../README.md)
