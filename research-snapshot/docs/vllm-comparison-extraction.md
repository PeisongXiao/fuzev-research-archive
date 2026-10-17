# Isolated vLLM Comparison Extraction

`python -m fuzev_research.vllm_comparison_extract` extracts query, candidate and
label-control features for the three historical vLLM configurations.

This work is based on recipes unpublished lab commit `6c30ef8`. The real API contract was inspected in
core unpublished lab commit `53a698ea9a62c6c6889df113f1bd73356fc618a7`, specifically
`PoolingParams.verify`, `AsyncLLM.encode`, and
`GPUModelRunner._fuzev_readout`. Verification here uses synthetic CPU inputs
and fake engines, not a GPU run or six-cell quality evidence.

## Readout Contract

- `ReadoutPreparer` owns the canonical context template, query suffix,
  independent candidate tails, and single-token ` A` through ` P` controls.
  Its small adapter uses the one engine's HF tokenizer with
  `add_special_tokens=False` and `truncation=False`, not a native backbone.
- The fresh query is the complete canonical context plus query suffix.
  `PoolingParams(task="fuzev_features",
  extra_kwargs={"baseline_token_ids": [...]})` returns one vector containing
  2,560 final-mixer features followed by exactly the requested control logits.
  The core performs the explicitly requested offline vocabulary projection;
  these are not head predictions or sampled tokens.
- Every candidate is a separate `query + candidate_tail` request, with no
  controls, padding, cross-candidate continuation, or extra generation prompt.
  Candidate order is preserved. The lab processed whole rows serially with
  four-sequence candidate waves.
- Final outputs must match their request IDs and complete token sequences.
  Tensors become finite, independent FP32 CPU copies before the next iterator
  step. Query/control and candidate widths are checked separately before a
  row can enter `FeatureStore`.
- Images use the existing acquired `digest.png` validation, engine renderer,
  expansion checks, and owned CPU payload/replay in `VllmImages`. An image is
  prepared once per row; every branch receives its own mutable payload copy.
  The root remains `FUZEV_ROOT/data/images`. No URL/upload/native alias path
  is introduced.
- "Fresh" describes the submitted canonical query, not a claim that physical
  prefix reuse is disabled. Engine caching remains engine-owned. Existing
  worker forward stats are preserved without inventing physical forward counts,
  hybrid-state reuse, or driver-free acknowledgements.

## Frozen Inputs And Identity

Only `comparison_inputs.iter_inputs` is accepted. The complete iterator is
exhausted once before model construction, including its checksum, membership,
semantic, head/calibration, and post-freeze checks. This is not merely a
first-row check. Extraction makes a second bounded streaming pass under an
outer frozen-reference context. Nothing is sampled, dropped, or truncated.

The reference must identify the canonical native-4 model/revision. EXL3 also
must match that reference's model/revision; it cannot substitute native-3/5 or
a different repack. The head files are opened only by the existing read-only
freeze validator, never by a scoring/training path here.

`--pins` is local JSON with these required fields and only the optional EXL3
`serving_view` field described below:

| Field | Required value |
| --- | --- |
| `schema_version` | `1` |
| `artifact` | `repo`, immutable `revision`, absolute resolved `local_snapshot`, and complete `files_sha256` mapping of snapshot-relative file names to SHA256s |
| `runtime_cell` | Full reviewed output of `vllm_service.describe_runtime(actual_engine)["runtime_cell"]`, not a reference-head identity |
| `runtime_sources` | Recorded recipes/core source revisions; EXL3 additionally requires `vllm-exl3-fuzev` and `exllamav3-fuzev` |
| `runtime_environment` | Exact `runtime_environment()` result: `VLLM_*`, `EXL3_*`, `FUZEV_*`, and present `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `CUDA_VISIBLE_DEVICES` variables |
| `serving_view` (EXL3 only, optional) | Separately pinned loader view of the canonical `artifact`, never an alternative HF snapshot |

Artifact repositories/revisions come from the existing acquisition constants.
The complete local snapshot, including tokenizer/config/weight files, is
checksummed before launch. HF snapshot file symlinks are supported; directory
overlay symlinks are not. The artifact file set, resolved files, sizes and
timestamps are checked again after inference. This is not a checksum of
in-memory GPU weights or loaded kernel binaries. Broken links and nonregular
files are rejected as well as directory overlays.

`--engine-config` is a pinned local `AsyncEngineArgs` JSON object. Both `model`
and `tokenizer` must explicitly equal that local snapshot unless an EXL3 view
is explicitly proved below. NVFP4 has no view exception. Remote code,
remote model paths and alternate/dummy weight loaders are rejected. Supported
standard load formats are `auto` and `safetensors`; existing EXL3 plugin
quantization still uses the standard loader. The complete options, snapshot
hashes, actual runtime sources/environment, encoding, dataset revisions and
optional dataset-scope contract enter the new feature identity.

Credential-named keys are forbidden, not redacted into a reusable pin. Before
creating output or a durable binding, the extractor rejects keys such as
`VLLM_API_KEY`, `EXL3_ACCESS_TOKEN`, `FUZEV_CLIENT_SECRET` and `HF_TOKEN` from
the relevant runtime environment. It recursively rejects sensitive option/pin
keys such as `hf_token`, `api_key`, `Authorization`, passwords and secrets,
including nested engine options and null/empty values. Errors name the key,
never its value. Remove these keys entirely for local offline execution;
normal `tokenizer`, `token_ids` and token-capacity settings remain supported.
Environment checks repeat at postflight without serializing a newly added
credential. This is credential-key rejection, not arbitrary string redaction.

Use reviewed actual runtime pins, not invented example values. The module
records each pinned source's observed revision (`git rev-parse` when
available, otherwise the package version) as `observed_runtime_sources`, beside
the pins. After construction and before any readout it compares
`describe_runtime` and requires both generation and feature tasks on that one
client. Runtime pins are rechecked after extraction. Configured KV precision such as `auto` remains a
configuration observation, not measured precision.

The output has `corpus_scope="evaluation-test-only"`. Its actual cell/model/
source identity is separate from `reference_head`. No vLLM identity is copied
onto the native reference corpus, and no per-cell head adaptation is allowed.

### Canonical EXL3 Payload And Loader View

A configuration-only loader view may use an engine model path that differs
from the HF snapshot. Keep `artifact.local_snapshot` pointing to the canonical turboderp
4.05 snapshot and keep its complete `files_sha256`, including every tensor.
Do not put the serving view in that field or replace full tensor hashes with
header checksums.

For EXL3 only, supply a `serving_view` object with exactly:

| Field | Required value |
| --- | --- |
| `directory` | Absolute resolved existing view directory, separate from the canonical snapshot |
| `files_sha256` | Exact mapping for **every non-`.safetensors` file** in the view, including inherited metadata, config backups, index, quantization config, `pack_scan.json` and `serving-view.json` |
| `canonical_pack_scan` | Object with absolute resolved `path` to the existing canonical scan record and its reviewed `sha256` |

`options.model` and `runtime_cell.model` must then equal `directory`. The
explicit tokenizer path may equal that directory or the canonical snapshot;
`options.tokenizer` and `runtime_cell.tokenizer` must agree. The artifact,
view metadata/proof and actual loader paths remain distinct in feature
identity, exact resume binding and postflight checks.

Verification follows the existing `qwen_pack_view.py` / `qwen_pack_config.py`
record formats:

- All canonical files remain present. The complete set of tensor filenames
  must match, and each must be the actual same file as its canonical payload
  (`samefile`, resolved source and byte count), not a copied or swapped tensor.
  The canonical snapshot still receives complete SHA256 verification before
  any model construction. Header records cannot bypass it.
- Only the existing config/quantization/scan/record/native-backup files and
  `model.safetensors.index.json` may be added or adapted. Other inherited
  files, including tokenizer/image metadata, must alias the canonical files.
  Every non-tensor view file must match its separately reviewed SHA256.
- `serving-view.json` must identify the canonical source/config checksum,
  every tensor, matching total bytes, `tensor_conversion=false`,
  `configuration_only=true` and `gpu_qualified=false`. Its header SHA256s are
  pinned record evidence, not independently rescanned tensor headers here.
- The pinned existing canonical scan must name the canonical snapshot. The
  view scan must name the view and otherwise equal that canonical scan,
  without n-gram problems. The extractor does not rerun a pack scan. The
  existing real source proof is `runs/native-real/canonical-pack-scan.json`.
- `config.json.native` must match canonical config bytes. The derived top
  quantization block and sidecar must agree, preserve the original block in
  `native_quantization_config`, and name `pack_scan.json` as their derivation.
  No non-quantization config change is permitted. A replaced index is
  separately pinned, must cover the scan's tensor count and reference exactly
  the canonical tensor filenames; an optional native index backup must
  preserve the original bytes. No index/header regeneration is performed.

This accepts explicitly reviewed loader adaptation, not arbitrary repacking
or automatic approval of a new quantization/index interpretation. Review of
the metadata hashes and actual plugin/core source revisions is still required;
structural provenance is not a live-loader or numerical-correctness claim.
The real records were inspected as metadata only: their scans match except
for the `pack` path, their config change is the quantization adaptation, and
all 11 tensor aliases have matching device/inode/size metadata. The canonical
config and `config.json.native` both hash to
`2ff43c0d52242195071dbb6b190f358eaf74d54ba64abdd780d4776c102ed719`;
the existing canonical scan record hashes to
`531bab0efd3b7001b0ddcfe593e52c36bb4b6d9f2695847cc399241d0a1199c2`.
These observations are not complete artifact/runtime pins.
No real tensor hashing, downloads or model launch ran for these observations.
The scan/view proofs and directories are protected from output
overlap and checked for changes before completion or resume.

## Admission And Failure

The lab used a 262,144-token context, 2,560 features, synchronous V1 execution,
four sequences, one GPU and no speculation or cache transfer. It recorded an
allocation fraction of 0.85 with host admission and swap guards. Operator
loader/offload options are pinned, not silently retuned after a failure.
Both Hugging Face and Transformers offline modes must be `1`.

Failed candidate waves cancel and drain siblings. The reused client holds
slots through abort submission, iterator close, repeated cancellation and
image preprocessing cleanup. A cleanup error closes further admission.
Abort submission is not a GPU-free acknowledgement. The owning context
shuts down its single engine before completion is reported.

The output must be separate from the reference features/head, frozen inputs,
model snapshot and executor source tree, including ancestor/descendant
overlaps. An exclusive local output lock prevents competing executors.
Storage aliases and outputs that passed the membership check are rejected on
resume. Writers flush, checkpoint their WAL, and close before storage checksums
and the completion record are returned.

`vllm-comparison-extraction.json` records exact binding, counts, storage hashes,
and one of `starting`, `extracting`, `failed`, `cancelled`, `memory_infeasible`,
or `extracted_pending_independent_seal`. Memory failure preserves the failure
and partial successful rows; it never retries with relaxed guards. Row errors
also enter `FeatureStore.failures`. A durable run-level failure marker remains
until postflight checks, outer freeze validation, and writer/client/engine
cleanup all succeed. Thus a full row count after a failed postflight cannot
pass the independent membership check.

The same atomic JSON record updates every 128 successfully handled input
records (new rows plus exact-resume skips), as well as on lifecycle transitions
and termination. It includes `observed` wall time, a copied `counts` snapshot,
`elapsed_seconds` from a monotonic run clock, and `records_per_second` equal
to new successful rows divided by that elapsed time. Skips do not inflate the
rate; wall-clock rollback cannot change its denominator. Elapsed time includes
engine startup and, at termination, cleanup. This is supervision progress,
not a benchmark or quality claim. Intermediate rows may still be buffered;
only the drained end record holds committed storage and allows resume.
No `data.features` live state is updated, and no independent membership record is produced.

`--resume` requires identical input/reference, model/source/environment and
engine-option pins, timeout, executor identity, and drained storage checksums.
Every already committed row is audited against canonical input semantics,
sidecar ownership and finite tensor/control alignment before the engine starts.
Only matching committed rows are skipped. Successful retries clear their own
row failures; the run marker clears only on final success. Extra rows,
unfinished failures, missing controls and changed files cannot be hidden.

Resume supports handled row/startup errors/cancellation with unchanged
input/reference/artifact/source/runtime identity, and completed runs awaiting the membership check.
Postflight or final cleanup failures are explicitly non-resumable: a new run
must recompute their rows rather than silently adopting suspect features.
An uncatchable kill with new unpinned commits or a nonempty WAL deliberately
fails closed; it needs a separately reviewed recovery, not automatic adoption
of unverified shards. No recovery mutates the frozen reference or recipes.

## Operator Sequence

Prepare the frozen native 4.05 reference, canonical test inputs, model acquisition
and runtime/artifact records, then run one isolated comparison configuration.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m fuzev_research.vllm_comparison_extract \
  --model vllm-nvidia \
  --plan /absolute/frozen-evaluation-plan.json \
  --inputs /absolute/complete-test-inputs \
  --inputs-sha256 <comparison-inputs.json-sha256> \
  --engine-config /absolute/reviewed-local-engine-options.json \
  --pins /absolute/reviewed-vllm-comparison-pins.json \
  --output /absolute/new-separate-comparison-features \
  --allow-inference
```

After a successful drained return, run the independent membership check in a
separate CPU-only invocation, with the actual complete examined count:

```bash
CUDA_VISIBLE_DEVICES= python -m fuzev_research.comparison_seal \
  --plan /absolute/frozen-evaluation-plan.json \
  --features /absolute/new-separate-comparison-features \
  --examined <actual-complete-examined-count>
```

Its record proves storage/membership only. Supply that identity and record to
the separate evaluation
workflow without changing the native-4 head. Missing actual cells remain
missing; fake-engine tests must never fill the six-cell matrix.

## CPU Verification

Verified locally: **92 executor tests**, and **275 tests** including the adjacent
service, image, preparation, frozen-input, membership-check and memory-guard
suites. The combined run had one upstream Starlette/AnyIO deprecation warning.

Tests reuse the repository's small synthetic freeze fixture and fake renderer,
with explicit-control engine fakes. View tests use tiny CPU safetensors and
records in the existing producer schema, check swapped/copied tensors and
unreviewed configs/indexes, and preserve the canonical/reference identity.
Credential and bounded-progress tests use mocks and temporary paths. No fitting,
real held-out content, model loading, downloads or real weights/features writes occur. Each fixture
is removed on teardown. Tests ran CPU-only (CUDA hidden, one Torch thread,
offline Hugging Face access):

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_vllm_comparison_extract.py
```

Current results are linked from the [archive evaluation guide](../../docs/research-evaluation.md). These fixture
checks establish readout and cleanup behavior, not engine transfer quality.

[Snapshot methods and environment](../README.md)
