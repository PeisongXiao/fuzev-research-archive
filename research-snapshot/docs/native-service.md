# Native Shared Service: Isolated Implementation

Historical pre-1.0 research service, superseded in FuZev v1.0 by `fuzev-vllm-plugin` and
`fuzev-exllamav3-plugin`. It does not run against the published engine forks.
The released servers have no `--service-config`, `phase_timing` or server-local
image path.

The standalone research service combines generation and independent decision
branches. The released integration instead attaches to an existing Generator.

## Ownership and Exact Tokens

The optional service receives the existing `NativeBackend`; it never constructs
another backbone. Generation, decisions, feature requests and extraction-writer
readouts shared one executor in the lab runs, with four-request admission
including writer readouts. HTTP cancellation drains executing work.

Generation renders A once, or encodes a raw completion prompt. It retains exact
accepted A+B token IDs and an immutable engine checkpoint for the consumed prefix.
The final sampled token, including EOS, can be accepted but not yet consumed.
Text stop matches affect presentation only: the matching token and any text
after the stop inside that same token remain accepted. The response discloses
the completion IDs, accepted-token hash, consumed boundary and unconsumed IDs.
Returned text must NOT be retokenized to recover that prefix.

A decision forks that checkpoint, consumes the missing accepted token through
the feature path, then appends C's additional evidence, query and candidates.
It never rerenders AB. Standalone decisions retain `NativeBackend.base_tokens`
and the existing training query/candidate encoding. Query features are owned
FP32 copies before candidate forwards can overwrite scratch storage.

Candidates fork the same immutable query, grouped by exact tail length in
first-seen order, in original-order chunks of `backend.candidate_batch_size`.
The configured size must fit the actual engine/model/cache capacity, up to eight,
and the full readout fingerprint must match the trained reference by default,
or the separately pinned actual serving identity for explicit transfer. B8 therefore
uses a natural five-way group, not `[4, 1]`; groups larger than eight are chunked
without padding. Runtime geometry mutation after head admission fails closed.
There is no padding,
causal candidate concatenation, free-form decision generation or vocabulary
projection. Scoring uses only the calibrated FP32 `HeadBundle` on CPU.

Immutable service prefixes have TTL, bounded count, explicit deletion and
memory-pressure eviction. They share the backend's KV pool with feature-cache
prefixes; a decision's source prefix is protected from service eviction while
being read. Working branches are released after success, cancellation or
failure. Expiration/deletion run on the same owner, not a separate GPU worker.
MemoryGuard and host admission remain at the backend's declared fraction.

## Opt-In Configuration

The research server accepted
`python -m fuzev_research.server --model native-4.05 --service-config <absolute-path>`.
The lab reference used `--batch-capacity 8 --candidate-batch-size 8`, allocation
fraction 0.85, context 262144, FP16 KV storage and NVMe-streamed PLE.

The operator-owned JSON is described by `ServiceConfig`:

- `head_directory`: absolute path to the calibrated reference bundle.
- `head_id`: normally `reference`.
- `manifest_sha256`, `calibration_sha256`: separately reviewed file checksums.
- `feature_identity`: the complete frozen extraction identity from the bundle.
- `service_sources`: exact current recipes and native repository revisions.
- Optional `serving_identity`: exact actual native cell/backend record for
  explicit frozen-head transfer. Omission retains strict reference matching.
- Optional `prefix_capacity` (default 8) and `prefix_ttl_seconds` (default 900).

No deployment config is fabricated before the real head exists. The loader
rejects uncalibrated training checkpoints, incompatible identities, missing
fit/development/calibration provenance, non-FP32 or nonfinite weights, changed
checksums and invalid temperatures. Safetensors only; no executable artifact
content. Inference does not fit or recalibrate a head.

Serving compares the actual model revision, encoder identity, native source,
feature tap, cache precision and full readout-runtime fingerprint. Extraction
and serving recipe revisions are separately pinned, because this serving code
necessarily differs from the frozen extraction code. Those explicit pins are
provenance, NOT a claim of numerical compatibility or a bypass of qualification.
`runtime_sources()` now identifies the imported recipe checkout, so a source directory
records its actual source revision.
The service prefix identity also includes the full head-manifest and calibration
SHA256 checksums. Neither those artifact hashes nor the extraction recipes SHA
is substituted for the actual new serving recipes/native source identity.

### Explicit Native Bitrate Transfer

After training, `native-3.05` and `native-5.05` can consume the unchanged
`native-4.05` reference by explicitly pinning `serving_identity`. Only
`native-3.05`, `native-4.05` and `native-5.05` are admitted in this mode. The
identity must contain exactly `cell`, `schema_version`, `engine`, `model`,
`revision`, `encoding`, `cache_precision`, `feature_tap`, `hidden_size` and
`readout_runtime`. These must match the selected cell, the acquisition-selected
model/revision and actual backend fingerprint. Missing, extra or mismatched
fields fail before head loading; `service_sources` remains independently exact.

`feature_identity` still pins the original frozen head, including its four
training sources, external-only scope and producer origins. Never replace it
with the actual serving identity or rewrite the head manifest. Reference native
engine/schema, encoding, feature tap, hidden size and compatible native source
remain required. The builder records both identities and
`transfer_qualification: unqualified; actual cell quality and runtime gates required`.
It never promotes that marker from operator pins or CPU tests.

Synthetic local acquisition/startup tests cover both bitrate cells and all three
scoring primitives with actual frozen CPU heads. They do not load a backbone or
establish real quantized-model, image, full-window or runtime qualification.

## Routes and Limits

- `POST /v1/completions`: raw text prompt, optional trusted image, one completion.
- `POST /v1/chat/completions`: text messages and optional trusted image.
- `POST /v1/decisions`: existing Choice/Noul/ordinal request contract.
- `DELETE /v1/prefixes/{prefix_id}`: release retained state.
- `POST /v1/requests/{request_id}/cancel`: cooperative cancellation.
- Existing feature and extraction-job routes remain available.

Set `X-Fuzev-Request-ID` before submitting to cancel an in-flight request by ID.
Cancellation is checked at target-forward boundaries, including chunk and
candidate-batch boundaries. It cannot preempt an already executing CUDA kernel.
Expired, evicted or released prefix IDs return 404, not reconstructed context.
Absent opt-in configuration returns 503 for generation/decisions.

Generation supports greedy or request-local seeded temperature/top-k/top-p
sampling. Default temperature 1, top-p .95, top-k 20, seed 0, maximum 256 new
tokens per request (client-overridable within the context window). Thinking is
disabled in the fixed chat template. No streaming, MTP/speculation, tools,
reasoning-effort interface, or general OpenAI compatibility is claimed.

### Generation Phase Timing

The research adapter returned `phase_timing`, using `time.perf_counter()` on
the existing owner. The prefill interval begins immediately before forwarding
the freshly created prompt branch and ends after the first existing CPU token
sample. Decode runs from that sample through the last CPU sample; its token
count is accepted completion length minus one, including an accepted but hidden
EOS/stop token. One-token outputs have zero decode work and no decode rate.
No additional CUDA synchronization, tensor copy, forward, stream, token, kernel
or scheduling change is introduced. CPU clocks cannot measure GPU-only time.

This scope includes existing prompt chunking/sampling and inter-token checks,
detokenization and forwards. It excludes image/token preprocessing, queueing,
branch creation and final presentation/retention. Queue duration and engine
preemption count remain unknown. A fresh generation branch has no reused prompt
KV (`reported_cached_prompt_tokens: 0`); this is not a weight/PLE-cache metric.
Invalid/nonfinite/nonmonotonic boundaries return explicit unavailable timings,
not invented rates or an altered completion. The TTL clock stays separate.
These CPU tests verify accounting and unchanged mock forwards, not live GPU
throughput.

## Trusted Still Images

Generation and decision requests accept one optional top-level field:

```json
{
  "image": {
    "path": "$FUZEV_ROOT/data/images/<digest>.png",
    "sha256": "<64-lowercase-hex-RGB-pixel-digest>"
  }
}
```

These placeholders are not real files or dataset records. The path must be an
absolute, direct child of the acquired `data/images` directory, named by the
digest. The existing ImageIndex identity is SHA256 of
`f"{width},{height}:".encode() + rgb_pixels`, **not** SHA256 of the PNG file.
The service rejects URLs, base64/uploads, symlinks, subdirectories/traversal,
nonregular files, non-PNG/animated input, files over 128 MiB and decoded images
over the acquisition limit of 32 million pixels. A bounded immutable byte
snapshot is checked before native decoding, preventing pathname replacement
after validation. This trusts the operator-controlled acquired directory; it
does not fetch or register new images or inspect dataset targets.

- Raw completion: the native image alias precedes the raw prompt.
- Chat generation: the image is first in the first user message, followed by its
  text, with the fixed generation template and the existing thinking settings.
- Standalone decision: `NativeBackend.base_tokens` and the existing preparation
  template place the image exactly as in the canonical training context. Query
  and candidate suffixes are unchanged.
- Prefix decision without `image`: exact accepted AB and its original embedding
  instances are reused. No template rerender, retokenization or image reload.
- Prefix decision with `image`: the new image becomes additional evidence in C,
  before `additional_context` text inside the existing query suffix. It does not
  replace the image in AB. The unconsumed final B token is resolved with AB's
  original modality/embeddings before introducing C's new image.

Only the backend's existing vision component and text backbone are used. Every
append, including generated-token consumption, query and candidate branches,
receives `backend.parameters` with the exact indexed embeddings. Full planned
suffixes are supplied for chunk-spanning images. Qwen4Exp stays on ordinary
linear positions; only a model advertising mRoPE receives native mRoPE
frequencies. The service never invents its own image positions or downscaling.

Retained AB owns immutable image bindings: RGB digest, exact embedding object,
native token range, token-list hash and layout identity. Image LRU eviction or
a new embedding for the same pixels cannot substitute new dynamic IDs into old
accepted tokens. Even deletion of the original file does not invalidate a live
owned prefix. Unverified digest-to-embedding cache associations are re-encoded
once from verified bytes, rather than trusted; verified instances can be reused.
The verification registry has weak values, so it cannot keep embeddings alive.
Both response types disclose image digests and native token-range/hash records.

The lab used one image per fresh generation/standalone decision, up to two active
image bindings in AB+C, retained-prefix count/TTL settings and four admitted requests. Explicit release, expiration, capacity eviction, cancellation and
failure drop service-only image/cache references. Live service/feature prefixes
remain owners; a newer LRU object with the same digest is never removed by an
older prefix's cleanup. The native tower stays warm, and its normal bounded LRU
and feature-prefix lifetimes still apply to feature-owned images. Existing RAM
guards remain enabled; file decoding additionally admits 512 MiB of headroom.

The HTTP feature route applies the same safe file/checksum boundary. Registered
corpus jobs retain their original preparation/extraction implementation and
provenance; no training rows, aliases or features were rewritten.

Structured OpenAI image content, arbitrary markers, multiple new images in one
request, videos/audio, image output and generation-from-prefix requests remain
unsupported. Prefix continuation here means accepted AB to an appended typed C
decision. All over-window requests fail without truncation or fallback.

## Focused Verification

The merge passed 58 focused CPU tests. The final extended suite passed **122
tests in 6.58 seconds**, with one upstream Starlette deprecation warning. Tests
ran CPU-only (CUDA hidden, one Torch thread, offline Hugging Face access). No request was sent to a live inference endpoint. Fixtures and
pytest temporary data stayed within the snapshot directory's ignored `.pytest_cache`.

Test modules: `test_native_service.py`, `test_service_images.py`,
`test_server.py`, `test_contracts.py`, `test_batch_capacity.py`,
`test_extraction.py`, and `test_preparation.py`. The runner asserts
`not torch.cuda.is_available()`, sets Torch intra/inter-op threads to one, and
uses test-only `PYTHONPATH`, `FUZEV_ROOT` and pytest fixture paths. No native
extension was imported/compiled and no weights or real heldout targets were read.

Coverage includes accepted/unconsumed tokens, EOS, partial-token text stops,
append-only C, standalone encoding, independent B4/B8 branches, owned scratch
features, TTL/capacity/identity checks, failure/cancellation cleanup, admission,
single-owner HTTP integration, existing extraction job behavior and calibrated
artifact checksums. Image checks use synthetic PNGs, native Python encoding and
parameter helpers, a mocked vision component and execution path, and mocked mRoPE generation.
They cover LRU replacement, shared ownership, weak-reference cleanup, new image
append ordering, safe files, cache association verification, chunk parameters
and unchanged canonical query layout. Head files are synthetic unit fixtures,
not trained results. These checks are implementation preparation only, not
additional pretraining validation experiments.

[Snapshot methods and environment](../README.md)
