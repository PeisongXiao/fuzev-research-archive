# Scope V2 Trained-Head Integration

The synthetic CPU fixture connects dataset scope, training, calibration and
external evaluation without a backbone.

The fixture exercises the actual v2 CPU supervisor through text fitting, mixed
fitting and calibration. All 12 profiles per stage run on 32 random-feature
training-source records. The synthetic fixture alone uses one epoch per profile;
the research ten-epoch maximum, patience, optimizer, readiness and memory
guards are unchanged. Fitting/development/calibration row selection is audited;
no test row may be consumed by those operations.

After the head is frozen, 3,693 synthetic external feature rows are appended:
239 QASPER yes/no, 1,450 answerability, 2,000 evidence selected by the real fixed
sampler, and two paired VL examples in each of Choice and Noul. The two visual
examples are a bounded schema fixture, **not** the real 1,247-example benchmark.
No image exists or is decoded; baselines are explicitly synthetic ramps.

The actual frozen loader must accept the unchanged training/calibration outputs
without repairing manifests, metadata or checksums. Training readiness remains
exactly unchanged after the external append. Final readiness must include all
six preparations and validate the full sampling frame and paired adapters.
Comparison preparation and fully consumed input validation must retain all
3,701 synthetic final-test rows (3,693 external plus eight training-source test rows). Head bytes must stay unchanged throughout.

All temporary features, 24 trial histories/checkpoints, calibration and copied
inputs live inside a disposable directory in this isolated checkout. A fresh
interpreter sets `FUZEV_ROOT` before research imports. Native/vLLM imports are
rejected, CUDA is hidden, HF is offline, and CPU math uses one thread. The real
memory guard remains active. The fixture removes its outputs on completion or
failure and has a 180-second subprocess timeout.

Verification: **one test passed in 11.30 seconds**.
This is not evidence of semantic quality, engine transfer, serving stability,
reference training completion or six-cell acceptance. The published snapshot
omits `tests/test_scope_trained_pipeline.py`.

[Snapshot methods and environment](../README.md)
