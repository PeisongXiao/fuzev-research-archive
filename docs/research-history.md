# Research History

Covers the v1.0 head (`head.safetensors` SHA-256 prefix `f7bc42e7`);
the v1.1 head is in [v1.1 head results](v1.1-head-results.md).

Method development and earlier outcomes. For current results, see the
[benchmarks](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/benchmarks.md).

## Head Development

Research explored independent heads, feature-only models, LM assistance,
adapters and capacity changes. The v1.0 release uses one LM-assisted shared scorer;
[training history](research-training.md) records its five-epoch refit,
incomplete component-balanced confirmation and calibration.

All six 5,075-request benchmark runs completed after earlier reports listed
configurations as not run. The complete held-out assessment passed text checks
and kept both external VL-RewardBench failures; the
[evaluation guide](research-evaluation.md) distinguishes these populations.
The [answer-order note](processed-quality.md#vl-rewardbench-answer-order)
explains VL-RewardBench's first-response imbalance.

## Superseded Serving Results

- **vLLM:** pre-fix generation was malformed. HTTP success and observed draft
  tokens in those runs did not establish ordinary generation correctness. The
  [integration history](2026-10-12-engine-owned-integration.md) records the
  graph-replay and circular-cache mapping defects, then the corrected checkpoint checks.
- **ExLlamaV3 batch:** native quality runs used the standalone research path at
  candidate batch 8; the v1.0 Generator attachment reads candidates at batch 1.
- **ExLlamaV3 graph fix:** the CUDA-graph kernel-argument ownership fix
  (unpublished lab commit `5608c62`) ships in the exllamav3-fuzev core
  extension, covered by `tests/graph/test_arguments.py`. The recorded
  measurements used an extension built before it.

The [evaluation guide](research-evaluation.md#limitations) collects current serving scope.

## Research Settings

Older experiments' host-RAM and swap admission policies are historical settings;
current serving packages use engine and operator settings.
[Provenance](research-provenance.md#historical-settings) has the shared note.

The research snapshot keeps extraction, evaluation, training and a read-only
dashboard. The separate runtime, ExLlamaV3 plugin and engine fork provide current serving.
