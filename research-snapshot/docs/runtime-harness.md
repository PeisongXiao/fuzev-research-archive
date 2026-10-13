# Pinned HTTP Runtime Harness

The HTTP harness exercises generation, independent decisions, retained
prefixes and cleanup. The recorded implementation tests used mock transports.

## Checks and Workloads

The lab preflight used the 262144 window, allocation fraction 0.85, enabled
generation/typed decisions, recorded health source/provenance identity and
one matching model. Image requests additionally require the advertised trusted
image capability. A feature-only extractor is rejected before any POST.
Responses must match the head-manifest checksum, calibration ID and full prefix
identity. The pins come from the frozen head bundle.

For each procedural workload:

1. Generate AB and validate exact completion IDs, usage/budget and the accepted
   token record. Native accepted/consumed boundaries are checked. vLLM's full
   prompt+completion IDs must hash correctly; its unqualified consumed boundary
   stays unavailable.
2. Submit three independent appended C decisions, one per primitive, and repeat
   the first after the other branches. Require unchanged AB/query hashes,
   candidate order, selection and scores within declared tight tolerances.
3. Submit three standalone decisions, requiring finite normalized distributions
   and valid argmax selection. This is not rerendered AB equivalence or accuracy.
4. Release only the client's own prefix, verify idempotence and require 404 on
   subsequent use. Record cleanup failure as failure, never as a passing run.
5. Measure generation-only, standalone decision-only and generation-plus-decision
   pipelines at 1, 2 and 4 external streams, in sequential waves. Engine-owned
   candidate scheduling remains unchanged. Default repetitions are three.

The runner does not judge whether a procedural answer is intelligent. Candidate
permutations change the actual query encoding, so the runner applies no
invariance check to them.
Trusted image inputs are supported through the existing service contract, but
image-blind controls, visual quality and CPU mock images do not become GPU proof.

## Honest Measurement Scope

Record per-request client elapsed time, raw private JSON responses, pipeline
latency p50/p95, wall time including cleanup and completion tokens per pipeline
second. **That completion rate includes prefill and is not decode throughput.**
Native generation exposed no forwarded-position record, so generation
and mixed physical totals remain unavailable rather than inferred from usage.
Native decision counters are labelled service-reported input positions, not
worker acknowledgements. Patched vLLM generation uses a nested worker snapshot;
decisions aggregate independent snapshots. Missing records stay nullable and
missing acknowledgements retain lower-bound labels. No sum is emitted unless
every constituent count exists. Logical prefix lengths explicitly exclude C and
candidate tails; they are not relabelled as total executed tokens or cache rate.

The harness can consume the adapters' optional `phase_timing` records without
streaming or additional engine forwards. Native scope is existing prompt
forward through the first CPU sample, then first through last CPU sample.
vLLM scope is engine-core scheduled-to-first-token and first-to-last-token
monotonic request statistics. These are service intervals, not GPU kernel
durations, and their different CPU overheads remain explicitly labelled.
Native excludes preprocessing and queueing; vLLM does not mix its wall-clock
arrival timestamp with monotonic engine timestamps. Missing statistics,
inconsistent accepted counts, preemption or invalid boundaries stay unavailable.

Prefill's numerator is logical prompt length minus the server-reported cached
prompt length, not physical forwards, PLE traffic or recomputation. An unknown
cache count leaves the prefill rate null. Decode counts accepted output tokens
minus the first; one-token outputs have zero decode work and no decode rate.
All generation observations must have valid phase records before a group can
report a decode rate. A missing observation never becomes zero or a rate for a
silently selected partial subset. Fully cached prompts have no prefill rate.

Group rates use total phase tokens divided by the sum of per-request phase
durations. At multiple streams this is a duration-weighted per-request rate,
**not aggregate concurrent throughput**. The existing pipeline wall-rate is
reported separately and still includes prefill/decisions/cleanup. Standalone
decision-only groups have no generation phases. Raw scopes, unavailable reasons,
counts, durations and observation counts remain in private responses/reports.
The harness never infers phase times by subtracting different prompts, fitting
a curve or dividing completions by total elapsed time. The isolated CPU tests
do not establish actual throughput.

Every report keeps `runtime_qualified: false`, even when these scoped checks
pass. Fresh/reused complete KV/QSA/PLE/recurrent equivalence, cancellation and
failure boundaries, sustained scheduling, full-window feasibility, held-out
quality, transfer and the six-cell criteria remain separate evidence.

## Invocation

After extraction, fitting and calibration, supply a JSON `HarnessConfig`.
The lab run recorded the source revision (`git rev-parse` when available,
otherwise the package version). Required fields are
`engine`, `model`, `head_manifest_sha256`, `calibration_id`, `prefix_identity`,
`health_pins` and `workloads`. Native health pins must include nonempty `sources`;
vLLM pins must include nonempty `provenance`. Each workload includes `name`,
`generation` (the existing `GenerationRequest`), `standalone_messages`, and three
`decisions` covering Choice/Noul/ordinal. A decision supplies `primitive`,
`question`, `candidates`, optional `additional_context` and trusted `image`.

`streams` may contain only distinct 1/2/4 values. Tolerances default to atol=1e-5,
rtol=1e-4 and are written into the run configuration. All generation settings
are also recorded; defaults are temperature 1.0, top-p .95, top-k 20, seed 0.
No prompt/context truncation, server sampling change or live head fitting occurs.

```bash
PYTHONPATH=$SNAPSHOT \
  python -m fuzev_research.runtime_harness \
  --config <reviewed-local-config.json> --output <new-private-run-directory> \
  --allow-inference
```

Without the explicit flag, the CLI fails before reading configuration or issuing
HTTP requests. The harness connects to 127.0.0.1 on `--port` (default 8000). The output directory must be new. `config.json`,
flushed `requests.jsonl` and atomically checkpointed `report.json` retain partial
evidence after failure. No automatic model/container launch, retries, endpoint
restart, publication or deletion of another client's prefix is implemented.
Keep response/prompt artifacts private and out of Hugging Face packages.

## CPU Verification

```bash
mkdir -p .pytest_cache
python -m pytest -q \
  tests/test_runtime_harness.py --basetemp=.pytest_cache/tmp-harness-parent
```

Tests cover the complete scoped suite on both record schemas, real adapter
`forward_receipt` serialization, the recorded four-stream workloads, disabled extraction
protection, source/identity changes, token/count/NaN/order rejection, branch
interference, prefix cleanup, missing/lower-bound physical counts, image
capability admission, explicit CLI permission and existing-output refusal.
The additional phase cases check both timing schemas, duration weighting,
first-token exclusion, cached/unknown/fully-cached prompts, single-token decode,
missing-subset protection and malformed/nonfinite/count-mismatched intervals.
They are synthetic lifecycle tests, not real throughput/accuracy measurements.

[Snapshot methods and environment](../README.md)
