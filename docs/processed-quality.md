# Held-Out Quality

Covers the v1.0 head (`head.safetensors` SHA-256 prefix `f7bc42e7`);
the v1.1 head is in [v1.1 head results](v1.1-head-results.md).

The v1.0 LM-assisted head passed **23/27 checks on 25,599 records** in
FuZev's held-out assessment of training-source and external data. All text
checks and both in-domain MM-RLHF tasks passed. Both external VL-RewardBench
methods failed accuracy and tie-aware accuracy checks:

| Method | Head tie-aware accuracy | Frozen-LM control | Paired difference / 95% interval |
| --- | ---: | ---: | --- |
| VL preference Choice | 68.97% | 89.01% | -20.05 pp / [-22.49, -17.54] pp |
| VL relative-preference Noul | 69.85% | 82.60% | -12.75 pp / [-15.15, -10.38] pp |

### VL-RewardBench Answer Order

In this conversion the preferred response is first in 1,226 of 1,247 items
(98.3%), so accuracy mostly reflects how often a system answers Yes or picks
the first response.

- Both methods use the same 1,247 VL-RewardBench examples.
- Neither meets the predeclared 2-percentage-point noninferiority margin.
- Displayed values are rounded independently.
- Positive calibration temperatures preserve the argmax, so raw and calibrated
  heads have the same selection failures.
- Visual decisions remain experimental.

The paired intervals resample related example groups. They describe sampling
uncertainty, not repeated runtime execution.

- [processed-quality.json](results/processed-quality.json): full per-check
  aggregates and paired intervals. The [result index](results/README.md) links
  related source copies.
- [Benchmark page](benchmarks-v1.0.md):
  the six 5,075-request runs, their 100-example VL subsample and the Kev results.

## Reproduce

The [calibration and held-out assessment script](../research-snapshot/scripts/calibrate_frozen_reference.py)
is retained with the [original wrapper record](study-code/calibrate_frozen_reference.py).
It reads the fixed evaluation plan, clean-refit admission and FIT/DEV completion
records, the frozen head, and separate CAL/DEV/TEST feature stores. It fits CAL-only
positive temperatures, then assesses the full 25,599-record TEST population.

- [Study-code records](study-code/README.md) and [recorded settings](configs/head-study.json)
  identify the inputs and retained wrapper hashes.
- The [provenance table](research-provenance.md#v10-head-provenance) has the
  measured head and scorer hashes.
- The snapshot evaluator documents group resampling, tie-aware accuracy and
  noninferiority margins.
- Prepared membership rows, feature shards and per-item predictions are outside
  the archive.
