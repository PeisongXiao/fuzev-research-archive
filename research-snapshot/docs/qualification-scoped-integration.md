# Scoped Consumer Integration

Synthetic CPU checks connected the revised dataset scope to training,
evaluation and serving consumers. Training sources were HelpSteer2,
HelpSteer3, BoolQ and MM-RLHF. QASPER and VL-RewardBench remained
external-only; A-OKVQA was excluded.

## Fixture

The fixture used 32 random-feature training records and 3,693 synthetic
external rows. Each of the 12 profiles per fitting stage ran for one epoch.
All six evaluator configurations consumed the same synthetic features:
3,701 final-test rows (3,693 external plus eight training-source test rows)
and two bootstrap draws. This is schema and consumer coverage, not six
independent engine measurements.

The native and vLLM consumers scored the trained, calibrated head while
preserving reference-feature identity separately from serving identity.
Comparison extraction checked complete canonical membership before
evaluation. See [dataset adapters](dataset-adapters-v2.md) and
[external evaluation](external-evaluation-v2.md) for the underlying methods.

## Recorded Checks

| Suite | Cases passed | Time | Peak sampled process-tree RSS |
| --- | ---: | ---: | ---: |
| Native identity and serving | 77 | 8.29 s | 983,293,952 bytes |
| Scoped training and consumers | 1 | 25.48 s | 980,758,528 bytes |
| Scope, evaluation, comparison and export | 208 | 45.56 s | 765,890,560 bytes |
| Images, routes, vLLM and harness interoperability | 255 | 10.16 s | 901,152,768 bytes |

Total: 541 synthetic cases. Each row is one invocation; timing spread was
not recorded. A preparation-suite attempt could not collect because PyArrow
was absent, so these results are not an all-repository pass.

Additional extractor checks passed 105 cases in 47.60 seconds and nine
extraction cases in 1.99 seconds. Ten real-vLLM-API constructor checks passed
in 11.22 seconds without constructing an engine. These suites overlap other
records and should not be added as independent coverage.

The preserved test files include `test_scope_evaluation.py`,
`test_comparison_inputs.py`, `test_comparison_seal.py`, `test_native_service.py` and
`test_phase_interop.py`. Their setup and historical dependencies are described
in the [snapshot overview](../README.md).

[Snapshot methods and environment](../README.md)
