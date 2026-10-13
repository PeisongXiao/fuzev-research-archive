# Native Held-Out Extraction

The held-out extractor applies the reference encoding to complete canonical test
inputs for ExLlamaV3 comparison configurations. It writes a separate feature
corpus for the offline evaluator.

After the complete reference is trained, calibrated and frozen, prepare the
complete canonical test-only inputs with the existing `comparison_inputs`
command. The executor first completes all freeze, checksum and canonical
membership validation, not merely a first-row check;
it then loads exactly one native 3.05 or 5.05 comparison backbone, sequentially.
Native 4.05 already has complete reference features and need not be re-extracted.

The existing ExLlamaV3 prefill optimizations, max-eight physical/candidate
geometry, unpadded grouping, CPU preparation lookahead, GPU owner and ordered
writer were used in the lab, with allocation fraction 0.85 and recorded
reserve/swap and context settings. No full PLE residency, alternate image processor or dataset
filter is introduced. Only canonical test rows enter a separately identified
`evaluation-test-only` feature store; fitting preparations are never registered.

```bash
python -m fuzev_research.comparison_extract \
  --model native-3.05 --plan <frozen-plan.json> \
  --inputs <prepared-input-directory> --inputs-sha256 <manifest-sha256> \
  --output <new-comparison-directory> --allow-inference
```

The lab run recorded the source revision (`git rev-parse` when available,
otherwise the package version). There is no
automatic container launch, retry, head fitting, publication or production
promotion. The CLI requires explicit permission before reading inputs. A
failure leaves drained committed shards and a visible partial-run record.
Explicit `--resume` requires the same frozen reference and input binding and
refuses outputs that already passed the membership check. There is no silent
truncation or dropped failure. An explicit run-level failure marker prevents a
full row count from passing the membership check after failed postflight or
model/writer/guard cleanup. Only successful drainage clears that marker. A
failed cleanup remains visible in the run record.
Postflight/source or cleanup failure also forbids adopting that run's rows
through ordinary resume; those suspect outputs require a new run.

`comparison-extraction.json` keeps counts and source identity but deliberately
does not attest complete storage/membership or set runtime qualification.
After the writer/model have closed, run the independent membership check
(`comparison_seal`) in a CPU-only environment, pin its record, and evaluate
the frozen reference head and calibration with the six-cell evaluator. Any missing/extra/semantically
changed row must fail that independent check. Test targets are retained for
later held-out reporting, never used to select a kernel, configuration or head.

CPU tests use small synthetic tensors and a fake backbone only. They cover
test-only roles, pre-model freeze failure, clean resume without repeated model
readouts, owner cleanup, cancellation, partial failures, refusal of checked
outputs and the explicit permission check. They are not numerical transfer
evidence.

[Snapshot methods and environment](../README.md)
