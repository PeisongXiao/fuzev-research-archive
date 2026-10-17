# Shared vLLM Service Adapter

Historical pre-1.0 research service, superseded in FuZev v1.0 by `fuzev-vllm-plugin` and
`fuzev-exllamav3-plugin`. It does not run against the published engine forks.
The released servers have no `--service-config`, `phase_timing` or server-local
image path.

The research adapter shares one vLLM engine between generation and independent
decisions, retaining exact accepted prefixes and copied feature outputs.

## One Engine, Typed Decisions

`fuzev_research.vllm_service.build_service(engine_client, config)` receives an existing
fused `AsyncLLM`. It loads only the small checksum-pinned FP32 `ServingHead` on
CPU. There is no second backbone, remote inference call, cache transfer, or LM
free-form decision generation.

- Generation: `engine_client.generate(EngineInput, SamplingParams, request_id)`.
- Query and each independent candidate: `engine_client.encode(EngineInput,
  PoolingParams(task="fuzev_features"), request_id)` on the same client.
- Inputs are processed `TokensInput` or `MultiModalInput`, never native EXL3
  indexed embedding aliases. Still images use this client's existing renderer
  and existing model vision path, not another encoder model or server.
- Each final readout must be a finite 2560-element tensor; an owned FP32 CPU copy
  is taken before another iterator step can overwrite borrowed storage.
- Choice, Noul and ordinal use the existing decision request/score contracts and
  calibrated `ServingHead`, without a vocabulary projection in the decision path.

The original inspected core fork was **unpublished lab commit `0efcc39f0becd24b3e26e5f660b6a579a3a8ba57`**. Its
`vllm/v1/fuzev.py`, scheduler and `GPUModelRunner._fuzev_readout` implement
separate generation and feature task batches and the final-position tap on a
generation runner. The adapter does not replace or simulate those kernels.
The counter integration additionally inspects its descendant **unpublished lab commit `53a698e`**,
which exposes worker-acknowledged input-position snapshots through the actual
generation and pooling APIs. This establishes API constructor compatibility only.

## Exact Tokens And Honest Metrics

A generation renders/tokenizes A once, using the same portable fixed template
for chat. `RequestOutput.prompt_token_ids` must match the submitted IDs exactly.
The adapter snapshots the final completion's `token_ids` and retains immutable
`A + B` IDs. It never decodes or re-renders B. Generation presentation text may
exclude stop strings or EOS; that text is not used to construct a later prefix.

At the inspected revision, `BaseIncrementalDetokenizer.update` appends even a
stop token skipped for presentation. Both fast and slow output-token paths feed
the cumulative/final `CompletionOutput.token_ids`. An explicit integer stop ID
missing from the last output position is rejected. This is source/API evidence,
not a completed live EOS test on every supported runtime cell.

A later decision submits **the full accepted AB plus the portable C suffix**,
including the final sampled token whether or not it was consumed during
generation. Candidate requests submit immutable copies of that exact query
plus their own independent suffix, never another candidate's state. vLLM's
scheduler performs whatever replay/cache lookup is required. Cache salt includes
the reference head and actual serving provenance, not native cache state.

Retained service tokens and owned CPU processor payloads are **not**
KV/QSA/PLE/recurrent checkpoints. No native cache handles are imported. Prefix
deletion/TTL releases service records and owned payload references, not
engine-owned prefix/encoder-cache blocks. The same engine owns both request kinds,
but ownership alone does not prove complete hybrid-state reuse.

Responses distinguish:

- `logical_prefix_tokens` and `logical_query_tokens`: lengths of exact logical
  sequences, not measurements of GPU execution.
- `submitted_tokens`: sum of full prompt lengths submitted across the query and
  candidate requests; repeated shared prefixes count again.
- Per-request `num_cached_tokens`: only the engine API's actual field. Missing
  fields stay `null`; aggregate hits are summed only when all fields are known.
- `fuzev_forward_stats` in each successful request record: an owned JSON-safe
  scalar snapshot of the actual final output field, or `null` when unavailable.
- `forwarded_tokens`: alias for acknowledged unpadded target input positions,
  not a unique token count, GPU kernel/FLOP count or causal consumed-prefix.
  It is `null` on older/missing APIs. Decision `completed_token_positions` and
  `completed_forward_steps` sum independent successful query/candidate records
  only when all are available, excluding any preceding generation request.
- `forward_counts_status`: `worker-acknowledged`, `lower-bound`, or `unavailable`.
  `forwarded_tokens_lower_bound` is true if a record has
  `missing_worker_receipts > 0`, false if zero, and `null` when unavailable.
  Aggregate status and the gap count require every record; one missing record
  leaves aggregate physical counts and the lower-bound flag `null`.
- `reused_tokens`, `replayed_tokens`, `consumed_tokens`, generation
  `unconsumed_token_ids`: still `null`. Neither prompt length minus cache hits,
  output length nor a latest completed end proves restored hybrid-state coverage
  or which accepted IDs remain unconsumed.
- Mode is `fresh` or `full-token-replay`, never `checkpoint` or `reuse`.

`vllm/v1/engine/output_processor.py` populates the cache field from prefill stats.
`vllm/v1/core/kv_cache_manager.py` limits full-prefix hits to leave the last token
for readout, but does not expose complete per-request causal-state evidence to
this adapter. Unpublished lab commit `53a698e`'s core instrumentation provides acknowledged
forward-position counts, but not replay/reuse classification, restored-state
coverage or complete terminal cancellation/failure totals.

### Worker Forward-Stats Contract

`fuzev_research/vllm_receipts.py` has no core, Torch, renderer or engine imports. It copies
the nine actual `FuzevForwardStats` scalar fields into service-owned values;
no borrowed runtime object or cumulative streaming buffer is retained. Required
counts must be nonnegative Python integers, not bools, strings, floats or tensors.
Ranges must be paired, nonempty and inside the submitted sequence (including
accepted completion IDs for generation). Their width must agree with the
cumulative acknowledged steps/positions, allowing repeated recomputation.
No count is derived from a cache hit, prompt length or presumed decoder step.

For a final feature request, an available `feature_readout_position` must equal
both `last_completed_end - 1` and `len(exact_submitted_token_ids) - 1`. The
existing request/prompt ID check and `token_sha256` bind this positional evidence
to the full exact expanded token sequence. `feature_readout_position_verified`
means only this position check passed, **not** feature quality, image geometry,
native B8 agreement or restoration of the preceding hybrid state. A missing
readout position stays `null`; it is not replaced with `len(prompt) - 1`.

Only the final successful API output is considered. Earlier cumulative streaming
snapshots are neither added nor used as a fallback when the terminal snapshot
is missing. A scalar snapshot can be structurally unavailable on an older API
(missing field, `None`, or an incomplete stats schema); serving still works,
all physical counts remain `null`, and the gap/status explicitly says unavailable
and transfer remains unqualified. Populated malformed scalar/range data fails
validation instead of being coerced into evidence.

Decision aggregation requires distinct engine request IDs for its successful
query and candidates. Matching token hashes or shared prefixes do not merge
independent requests. `unavailable_forward_receipts` reports missing per-request
snapshots; cache observations and submitted logical lengths remain separately
named and follow their existing availability rules. A provided snapshot with
zero acknowledged positions and positive missing-worker count is a **zero lower
bound**, not proof of zero physical execution. An empty zero-gap final snapshot
cannot acknowledge a nonempty successful request and is rejected.

The raw optional replay/reuse/causal-prefix scalars are copied and range/type
checked, not inferred or promoted into a checkpoint or consumption claim.
At unpublished lab commit `53a698e` these values are always `null`. Any future proof/interpretation of
populated fields needs separate review and hybrid-state evidence; the legacy
consumed/replayed/reused aliases remain nullable here.

Failed/cancelled streams, missing final outputs, unsuccessful generation finish
reasons and failed candidate/head operations return no successful decision total.
Existing abort/drain cleanup is unchanged. A terminal successful older-API
response can remain usable with unavailable metrics, but a partial streaming
snapshot cannot turn a failed operation into a successful result. The core can
discard a late acknowledgement after abort; this adapter does not invent a
terminal total or driver-free acknowledgement for that case.

## Runtime Identity And Resource Bounds

`VllmServiceConfig.feature_identity` is the **reference training identity** used
to verify the existing head artifact. It is never relabeled as vLLM extraction.
Separate `runtime_cell` and `runtime_sources` must match the actual attached
client/config; source revisions were recorded with `git rev-parse` when
available, otherwise the package version. Startup checks both task
capabilities on that single client. Responses expose both identities and mark
head transfer explicitly **unqualified**. Pin agreement establishes neither
quality nor batch geometry equivalent to the B8 reference.

The runtime record includes configured model/tokenizer paths and revisions,
quantization, dtype, configured KV dtype, context, batching, memory fraction and
prefix-caching policy. `auto` KV dtype is not advertised as a measured precision.
Image records additionally include the adapter policy, effective request
override `truncation=false`, the complete actual multimodal configuration,
processor class/cache identities and model vision configuration. Graph hash
alone excludes resizing/preprocessing/cache settings; it is not a complete
identity. Pydantic serialization uses `serialize_as_any=true` so subclass
image-width/height options are retained rather than silently dropped.
Kernel binary identity is unavailable (`null`). Model path/config pins are not
an independent checksum audit of loaded weights; the acquisition/runtime
qualification records cover that check.

- Combined prompt/output or AB+C+candidate budget: **262144**, no truncation.
- The lab runs used four service operations, four-sequence candidate waves and
  `max_num_seqs=4`.
- The research adapter supported only synchronous V1 execution, TP/PP/DP one,
  and no speculative decoding or KV/encoder-cache transfer.
- The lab allocation fraction was 0.85 with a one-second MemoryGuard and 15%
  host-RAM reserve.
- Retained-prefix defaults: eight prefixes, 900-second idle TTL, 524288 total
  accepted tokens; count, token pool and TTL are configurable within fixed bounds.
  Active operations can temporarily own up to four additional borrowed prefix references.
- Each processed still image has a **512 MiB** owned CPU payload bound, including
  tensor/mask data and conservative metadata accounting. The registry defaults
  to **512 MiB** total retained multimodal payloads (configurable 1 MiB to 4 GiB).
  LRU eviction obeys count, token and image-byte bounds simultaneously. An active
  borrower keeps its own reference even when the registry deletes/expires AB;
  cleanup then drops it. At most two images can be bound to one AB+C sequence.
  Every engine request gets independent mutable CPU field/mask copies. The
  engine's existing processor/encoder caches have separate configured lifetimes;
  this registry does not manage or flush them. Host admission accounts for
  preprocessing and per-request copies, and the 15% reserve guard stays enabled.
- POST bodies at most 8 MiB. Generation defaults are temperature 1, top-p .95,
  top-k 20, seed 0 and 256 new tokens; no shared template changes. A configurable
  request deadline defaults to 3600 seconds, with abort/cleanup on expiry.

Cancellation, disconnection, readout failure, deadlines and headroom failures
abort outstanding IDs and drain/close iterators before releasing client slots.
Repeated cancellation does not detach cleanup. Abort API failure closes further
admission. **`AsyncLLM.abort` only submits core abort messages; it is not an
acknowledgement that an executing CUDA kernel or all GPU allocations are freed.**
The recorded scheduler used four sequences. Live cancellation and preemption
were untested.

## Trusted Still Images

**Implemented but GPU/head-quality unqualified.** Generation uses the existing
`GenerationRequest.image`; decisions use the merged `ServiceDecisionRequest`
extension. Both accept only this acquired-cache reference:

```json
{"image":{"path":"/absolute/FUZEV_ROOT/data/images/<rgb-sha256>.png","sha256":"<rgb-sha256>"}}
```

`checked_image` came from unpublished lab commit `f8efc0a`. The file must be a direct,
non-symlink regular child of the operator-owned acquired image root, named with
the supplied digest. It snapshots at most 128 MiB of encoded data and accepts
only still PNGs up to 32 million pixels. Identity is SHA256 of
`"<width>,<height>:" + RGB pixel bytes`, not the PNG file bytes. This service
cannot acquire images: no uploads, URLs, base64, videos, arbitrary files or
client-controlled embeddings. Text messages remain strictly text-only; client
image markers and native `<$EMB_...$>` aliases are rejected. Ordinary textual
URLs remain text and are never fetched.

`fuzev_research/vllm_images.py` calls the **existing**
`BaseRenderer.process_for_engine_async(TokensPrompt, arrival_time,
skip_mm_cache=true)` with a verified in-memory RGB PIL snapshot, a digest-bound
UUID namespaced by serving provenance, and `mm_processor_kwargs={"truncation":
false}`. It does not call `AutoProcessor.from_pretrained` or create a second
renderer, executor, vision model or generation engine.

At the pinned source revision the normal sender cache may return `None` data on
a hit, which cannot own retained image state. `skip_mm_cache=true` instead uses
the renderer's already-existing **processor-only** cache, which keeps positive
CPU preprocessing reuse while returning complete data. With caching disabled
the normal no-cache processor returns full data. Incomplete cache-only results
are explicitly rejected, never replaced with invented payloads.

The actual processor's `MultiModalInput` supplies exact expanded pad IDs,
`MultiModalKwargsItem` fields/batching descriptors, `mm_hashes`, and
`PlaceholderRange(offset,length,is_embed)`. All are captured in owned CPU
storage and bound to a payload digest and exact placeholder-token hash. Dense
CPU tensors only; one still `image_grid_thw` with temporal dimension one;
non-image modalities and native indexed IDs are not accepted. Mutable driver
inputs cannot change a retained payload or another candidate. Retained tensor,
field and range mutations are detected on reuse. The service does not expose
those objects to clients.

Placement matches frozen `TokenPreparer.render` and native service semantics:

- Raw generation: image placeholder immediately before the supplied prompt.
- Chat generation/standalone decision: image block before the original text in
  the first user message. Empty decision context uses the same fallback user;
  no-user context inserts the same `Consider the supplied image.` user.
- Portable `query_suffix` and each independent `candidate_suffix` are unchanged.
  Template identity/thinking flags remain exactly the shared encoding contract.
- Retained AB: expanded A and every API-accepted B ID, including EOS, remain
  unchanged. The original image payload/ranges survive source deletion and
  processor-cache eviction; AB is never rerendered or reprocessed.
- **New image in C is supported conditionally:** preprocess only the unchanged
  portable C suffix containing `placeholder + "\n" + additional_context`,
  verify that only its single image-pad token expanded, then shift that new
  range offset by `len(accepted_AB)`. Preserve all kwargs, hashes and masks and
  concatenate exact AB IDs with the processed suffix. Existing AB ranges stay
  fixed. The actual engine image limit must admit the combined count (normally
  `limit_mm_per_prompt={"image":2}` for image AB plus image C). A text AB needs
  only one image slot. Requests always submit complete immutable sequences.

`InputProcessor.process_inputs` accepts the processed `type="multimodal"`
input without re-rendering, and constructs `MultiModalFeatureSpec` with those
full kwargs, hashes and positions. Qwen's runner derives vision/mRoPE placement
from the complete token sequence and retained grid/ranges; this adapter does
not synthesize positions, geometry or encoder outputs.

If a future/non-Qwen processor rewrites tokens outside the single pad expansion,
the adapter fails **explicit unsupported** before any engine submission. General
support would need a core processed-suffix append contract that supplies a
lossless token edit map and position/feature bindings without retokenizing AB;
blindly rendering the full conversation would violate accepted B. Current
Qwen append semantics require no new core patch; that conclusion is static
source/API evidence plus mocks, and no real-image inference ran.
Acknowledged forward counters use the separate worker forward-stats contract above;
actual replay/reuse classification and causal consumed-prefix proof remain
unavailable. No accepted AB/C or image placement changes accompany these counters.

The lab serialized CPU image decoding/preprocessing within its four-operation
configuration. Cancellation cannot safely preempt its CPU executor;
ownership/admission are held until that worker drains and the PIL snapshot
closes, even under repeated cancellation or a reserve violation. EOS retains
the canonical payload; TTL/delete/eviction/close release registry ownership;
failed/cancelled operations cannot retain new payloads. Engine-owned caches are
not flushed by deletion, cancellation or failure.

## Separate HTTP Wiring

New module `fuzev_research.vllm_server` exposes completions/chat, typed decisions, prefix
deletion, request cancellation, models and health. Its app factory accepts one
engine context for attachment or CPU injection. The CLI constructs exactly one
engine and shuts down only the engine it owns. It binds **127.0.0.1:<port>**.
Native extraction modules and production scripts do not import this adapter.

Historical launch form:

```bash
python -m fuzev_research.vllm_server \
  --engine-config /absolute/path/to/pinned-async-engine-args.json \
  --service-config /absolute/path/to/pinned-vllm-service-config.json
```

The recorded engine JSON was an `AsyncEngineArgs` object with the local model path,
`runner="generate"`, `max_model_len=262144`, `gpu_memory_utilization=0.85`,
`max_num_seqs=4`, `async_scheduling=false`, and
`additional_config={"fuzev":{"enabled":true}}`, plus cell-specific loader/PLE
options. No model config or trained-artifact pins are fabricated here.
For the trusted image path, use CPU preprocessing and an actual image limit
that admits the desired combined count. Do not enable encoder/KV cache transfer
or hand precomputed embeddings to the public request schema.

The service JSON supplies `model_name`, head directory/ID, separate manifest and
calibration SHA256s, the complete reference `feature_identity`, and separately
reviewed actual `runtime_cell`/`runtime_sources`. `describe_runtime` reports the
actual existing client; it neither loads a model nor mutates it. Do not attach
other unbounded API routes to bypass this adapter's admission control.

## Verification

### Counter Integration

The recorded suite passed **154 CPU tests** across the existing adapter,
image mocks and opt-in real-API suites. The real API import source was the
read-only core source directory at
unpublished lab commit `53a698ea9a62c6c6889df113f1bd73356fc618a7`.
Checks exercise real `FuzevForwardStats`, `RequestOutput`,
`PoolingRequestOutput` and msgpack `EngineCoreOutput` constructors/transport,
then the actual adapter reader against those CPU-constructed outputs. No vLLM
engine is constructed; the only head predictions are the existing synthetic
CPU fake, not a trained artifact or heldout evaluation.

New coverage includes final-only cumulative snapshots, ownership despite later
runtime-object mutation, no inheritance from partial streams, successful
independent-request aggregation, missing schema/terminal records, zero/positive
lower bounds, strict scalar/range/readout checks, cache-count independence,
duplicate request rejection, failed/cancelled terminals and HTTP serialization.
The existing branch-failure test provides worker forward stats for successful
query work before a candidate fails; no successful decision total is returned.
Existing exact AB/C/image layout, lifecycle and resource-bound cases remain
unchanged and pass. Warnings are upstream Starlette/Torch JIT
deprecation and missing generated vLLM version metadata.

Reproduction, with all temporary state owned by the snapshot directory:

```sh
CUDA_VISIBLE_DEVICES="" FUZEV_VLLM_API_INSPECTION=1 python -m pytest -q \
  tests/test_vllm_service.py \
  tests/test_vllm_image_api.py \
  tests/test_vllm_images.py
```

### Earlier Adapter Evidence

unpublished lab commit `e959b39` run: 110 passed in 7.16 seconds. Original image-extension focused
run: **235 passed** (60 synthetic image cases, 53 adapter cases, 122 merged
native/extraction regression cases), plus three opt-in real-API checks run
separately, with one upstream Starlette deprecation warning in the mock run.
Tests ran CPU-only (CUDA hidden, one Torch thread, offline Hugging Face access):

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_vllm_service.py \
  tests/test_vllm_images.py \
  tests/test_vllm_image_api.py \
  tests/test_native_service.py \
  tests/test_service_images.py \
  tests/test_server.py \
  tests/test_extraction.py \
  tests/test_preparation.py \
  tests/test_batch_capacity.py \
  tests/test_contracts.py
```

Tests use synthetic tokenizers, engine outputs and CPU heads, not datasets or
trained artifacts. They cover exact AB/EOS/text-stop retention, immutable branch
layout/order, owned readout storage, typed primitives, nullable cache counters,
cross-runtime provenance separation, historical prefix/resource settings,
repeated cancellation, cleanup failure, deadlines/headroom, context bounds,
trusted RGB snapshots, exact placeholder expansion and kwargs/mask cloning,
new image C append, stale/missing/changed metadata rejection, sender-cache-only
rejection, image byte bounds/eviction, source replacement/deletion, CPU-executor
cancellation drain, unsupported inputs, HTTP routes and lifecycle shutdown.
Images are tiny synthetic fixtures created under the source directory test root, not
actual acquired dataset images. No GPU inference, model download or heldout
evaluation ran.

The real `SamplingParams`, `PoolingParams` and `TokensInput` constructors plus
the actual `AsyncLLM.generate/encode/abort` signatures were separately checked
inside the existing CPU dev container with CUDA hidden. No engine was constructed.
That import check emitted missing build-version metadata and no-CUDA-runtime
warnings; source identity remains the pinned local commit, not an invented
wheel release. It is API compatibility evidence, not loader/runtime proof.

The image extension also checks real `MultiModalInput`/`mm_input`,
`MultiModalKwargsItem(s)`, `MultiModalFieldElem`, flat/batched field descriptors,
`PlaceholderRange` and `MultiModalConfig` constructors. Actual metadata types,
dtype, masks, grid data, shifted ranges and independent CPU copies survive the
adapter. The renderer, `InputProcessor`, `AsyncLLM`, `AutoProcessor` and vLLM
processor-loader signatures were inspected without calling a loader/engine.
The configuration case verifies width/height/kwargs identity despite equal graph
hashes. Opt-in command, from the snapshot directory:

```sh
CUDA_VISIBLE_DEVICES="" FUZEV_VLLM_API_INSPECTION=1 python -m pytest -q \
  tests/test_vllm_image_api.py
```

[Snapshot methods and environment](../README.md)
