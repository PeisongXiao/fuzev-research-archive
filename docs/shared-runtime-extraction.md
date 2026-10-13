# Shared Scorer Extraction

Superseded by the published [fuzev-runtime](https://github.com/PeisongXiao/fuzev-runtime).
This note records how the learned scorer was separated from research training
and evaluation code.

- The runtime loads one option-conditioned head. It combines query and candidate
  final hidden states with label logits from the backbone's existing LM head.
- Training operators, datasets and engine loading live in separate packages.
- The Python API documents geometry, calibration and CPU/CUDA execution.
- The selected weights contain 2,872,401 FP32 parameters in 15 tensors. The
  [provenance table](research-provenance.md#released-head-provenance) records
  tensor, scorer and original-manifest hashes.
- [The research snapshot](../research-snapshot/fuzev_research/shared_head.py)
  retains the numerical source.

## Historical Checks

The [ExLlamaV3 capacity record](results/capacity/native4-serving-capacity-v1.json)
checked serving capacity and resource cleanup. It completed four decisions,
including one 262,144-token branch. Peak allocated/reserved CUDA memory was
82,111,987,200 / 82,208,358,400 bytes.

Current engine results are on the [integration page](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md).
