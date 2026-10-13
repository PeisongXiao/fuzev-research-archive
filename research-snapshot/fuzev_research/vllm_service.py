"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks.

The engine owns prefix caching and recomputes whatever its scheduler requires.
"""

import asyncio
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .contracts import DecisionRequest, Primitive, Score
from .encoding import TEMPLATE, candidate_suffix, identity as encoding_identity
from .encoding import query_suffix, token_hash
from .paths import project_path, source_revision
from .prefix import identity_digest
from .service_contracts import GenerationRequest, TextMessage
from .serving_head import TAP, ServingHead
from .telemetry import MemoryGuard, admit
from .vllm_images import MultimodalPayload, PLACEHOLDER, POLICY, VllmImages
from .vllm_receipts import FORWARD_SCOPE, UNAVAILABLE_GAP, aggregate_forward_receipts, forward_receipt
from .vllm_receipts import readout_receipt, readout_sources


CONTEXT = 262144
MAX_INPUT_BYTES = 8 << 20
QUALIFICATION = "unqualified cross-runtime transfer; CPU-mocked lifecycle tests only"
GENERATION_PHASE_SCOPE = (
    "unpreempted engine-core request intervals; prefill includes first token; "
    "decode excludes first token; not GPU kernel time")
IMAGE_MARKERS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                 "<|video_pad|>", "<|image|>", "<$EMB_")


class VllmServiceConfig(BaseModel):
    """Operator pins: reference feature identity is NOT the vLLM cell identity."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_name: str = Field(min_length=1)
    head_directory: str
    head_id: str = "reference"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_identity: dict = Field(min_length=1)
    runtime_cell: dict = Field(min_length=1)
    runtime_sources: dict[str, str | None] = Field(min_length=1)
    prefix_capacity: int = Field(default=8, ge=1, le=32)
    prefix_ttl_seconds: float = Field(default=900, ge=1, le=3600)
    retained_token_capacity: int = Field(default=2 * CONTEXT, ge=CONTEXT, le=8 * CONTEXT)
    retained_multimodal_bytes: int = Field(default=512 << 20, ge=1 << 20, le=4 << 30)
    request_timeout_seconds: float = Field(default=3600, ge=1, le=86400)


class VllmExecution(BaseModel):
    """Separate logical lengths, cache observations and acknowledged input rows."""

    mode: Literal["fresh", "full-token-replay"]
    logical_prefix_tokens: int
    logical_query_tokens: int
    submitted_tokens: int
    num_cached_tokens: int | None
    reused_tokens: None = None
    replayed_tokens: None = None
    forwarded_tokens: int | None = None
    completed_token_positions: int | None = None
    completed_forward_steps: int | None = None
    missing_worker_receipts: int | None = None
    forwarded_tokens_lower_bound: bool | None = None
    forward_counts_status: Literal["unavailable", "worker-acknowledged", "lower-bound"] = "unavailable"
    unavailable_forward_receipts: int | None = None
    forward_counter_scope: str = FORWARD_SCOPE
    consumed_tokens: None = None
    feature_tap: str = TAP
    checkpoint_retained: Literal[False] = False
    cache_evidence: str = "per-request num_cached_tokens only; no hybrid-state proof"
    counter_gap: str = UNAVAILABLE_GAP
    requests: list[dict]


class VllmDecisionResponse(BaseModel):
    id: str
    model: str
    primitive: Primitive
    head_id: str
    calibration_id: str
    selected_id: str
    scores: list[Score]
    execution: VllmExecution


@dataclass(frozen=True)
class RetainedTokens:
    id: str
    prompt: tuple[int, ...]
    completion: tuple[int, ...]
    accepted: tuple[int, ...]
    identity: str
    multimodal: MultimodalPayload | None = None


def _text(value):
    if not isinstance(value, str) or any(marker in value for marker in IMAGE_MARKERS):
        raise ValueError("Use the trusted image field, not image markers or structured content")
    if len(value) > MAX_INPUT_BYTES:
        raise ValueError("Text exceeds the service input resource bound")
    return value


def _tokens(value, *, nonempty=True):
    if not isinstance(value, (list, tuple)) or any(
            type(x) is not int or not 0 <= x < 1_000_000_000 for x in value):
        raise ValueError("Expected exact nonnegative integer token IDs")
    if len(value) > CONTEXT or nonempty and not value:
        raise ValueError("Empty or over-context token sequence; no truncation")
    return tuple(value)


def _generation_phase_timing(output, prompt_tokens, completion_tokens, cached_prompt_tokens):
    """Own optional final engine stats without affecting accepted generation."""
    timing = {
        "schema_version": 1, "source": "vllm-engine-core-request-stats",
        "status": "unavailable", "reason": None, "scope": GENERATION_PHASE_SCOPE,
        "prefill_seconds": None, "decode_seconds": None, "queue_seconds": None,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "decode_tokens": completion_tokens - 1,
        "reported_cached_prompt_tokens": None, "num_preemptions": None,
    }

    def unavailable(reason):
        timing["reason"] = reason
        return timing

    # Only this optional observation is fail-open; existing output checks remain.
    try:
        if cached_prompt_tokens is not None and (
                type(cached_prompt_tokens) is not int
                or not 0 <= cached_prompt_tokens <= prompt_tokens):
            return unavailable("invalid_cached_prompt_tokens")
        timing["reported_cached_prompt_tokens"] = cached_prompt_tokens
        stats = getattr(output, "metrics", None)
        if stats is None:
            return unavailable("missing_request_stats")
        preemptions = getattr(stats, "num_preemptions", None)
        if type(preemptions) is not int or preemptions < 0:
            return unavailable("invalid_num_preemptions")
        timing["num_preemptions"] = preemptions
        if preemptions:
            return unavailable("preempted_request")
        generated = getattr(stats, "num_generation_tokens", None)
        if type(generated) is not int or generated < 1:
            return unavailable("invalid_num_generation_tokens")
        if generated != completion_tokens:
            return unavailable("generation_token_count_mismatch")
        corrupted = getattr(stats, "is_corrupted", False)
        if type(corrupted) is not bool:
            return unavailable("invalid_is_corrupted")
        if corrupted:
            return unavailable("corrupted_request")

        # arrival_time/first_token_latency use frontend wall time, never these
        # engine-core monotonic endpoints. No client clock or GPU work is needed.
        timestamps = []
        for name in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"):
            value = getattr(stats, name, None)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                return unavailable("invalid_" + name)
            timestamps.append(float(value))
        queued, scheduled, first, last = timestamps
        if not queued <= scheduled <= first <= last:
            return unavailable("reversed_timestamps")
        prefill, decode, queue = first - scheduled, last - first, scheduled - queued
        if prefill <= 0:
            return unavailable("nonpositive_prefill_interval")
        if completion_tokens > 1 and decode <= 0:
            return unavailable("nonpositive_decode_interval")
        if completion_tokens == 1 and decode != 0:
            return unavailable("single_token_decode_interval")
        timing.update(status="available", prefill_seconds=prefill,
                      decode_seconds=decode, queue_seconds=queue)
    except Exception:
        # An incompatible/malformed optional metrics object cannot fail generation.
        return unavailable("malformed_request_stats")
    return timing


def _source_revision(directory):
    name = Path(directory).name
    package = {"research-snapshot": "fuzev-research", "recipes": "fuzev-research",
               "vllm-fuzev": "vllm", "exllamav3-fuzev": "exllamav3",
               "vllm-exl3-fuzev": "vllm-exl3"}.get(name, name)
    return source_revision(directory, package)


async def drain_tasks(tasks):
    """Keep ownership until child cleanup finishes, including repeated cancel."""
    pending = asyncio.gather(*tasks, return_exceptions=True)
    interrupted = False
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            interrupted = True
    pending.result()
    if interrupted:
        raise asyncio.CancelledError


def describe_runtime(engine, *, require_fusion=True):
    """Inspect the actual client/config, without native-source equivalence claims."""
    import vllm
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.fuzev import fuzev_enabled

    if (type(require_fusion) is not bool or not isinstance(engine, AsyncLLM)
            or fuzev_enabled(engine.vllm_config) is not require_fusion):
        raise ValueError("One actual AsyncLLM with the requested fusion policy is required")
    config = engine.vllm_config
    validate_engine_config(config, require_fusion=require_fusion)
    model, cache, scheduler = config.model_config, config.cache_config, config.scheduler_config
    directories = {"recipes": Path(__file__).resolve().parents[1],
                   "vllm-fuzev": Path(vllm.__file__).resolve().parent}
    for module, name in (("vllm_exl3", "vllm-exl3-fuzev"),
                         ("exllamav3", "exllamav3-fuzev")):
        if module in sys.modules or model.quantization == "exl3":
            spec = importlib.util.find_spec(module)
            if spec is not None and spec.origin:
                directories[name] = Path(spec.origin).resolve().parent
    cell = {
        "engine": "vllm-fuzev" if require_fusion else "vllm-fusion-disabled-control",
        "version": vllm.__version__,
        "model": model.model, "revision": model.revision,
        "resolved_config_revision": getattr(model.hf_config, "_commit_hash", None),
        "tokenizer": model.tokenizer, "tokenizer_revision": model.tokenizer_revision,
        "quantization": model.quantization, "dtype": str(model.dtype),
        "cache_dtype_config": cache.cache_dtype,
        "memory_fraction": cache.gpu_memory_utilization,
        "context_window": model.max_model_len, "hidden_size": 2560,
        "feature_tap": TAP, "encoding": encoding_identity(),
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "prefix_caching": cache.enable_prefix_caching,
        "runner": model.runner_type, "async_scheduling": scheduler.async_scheduling,
        "speculative_decoding": False, "cache_transfer": False,
        "additional_config": config.additional_config,
        "loaded_kernel_identity": None,
        "image_adapter_policy": POLICY,
        "image_processor_kwargs": {"truncation": False},
        "multimodal_config": (None if model.multimodal_config is None else {
            "graph_hash": model.multimodal_config.compute_hash(),
            "configuration": TypeAdapter(type(model.multimodal_config)).dump_python(
                model.multimodal_config, mode="json", serialize_as_any=True),
            "image_limit": model.multimodal_config.get_limit_per_prompt("image"),
            "processor_kwargs": model.multimodal_config.mm_processor_kwargs,
            "processor_cache_gb": model.multimodal_config.mm_processor_cache_gb,
            "processor_cache_type": model.multimodal_config.mm_processor_cache_type,
            "processor": type(engine.renderer.mm_processor).__qualname__,
            "readonly_processor": type(engine.renderer._readonly_mm_processor).__qualname__,
            "vision_config": (None if getattr(model.hf_config, "vision_config", None) is None
                              else model.hf_config.vision_config.to_dict()),
        }),
    }
    return {"runtime_cell": cell, "runtime_sources": {
        name: _source_revision(path) for name, path in directories.items()}}


def validate_engine_config(config, *, require_fusion=True):
    """Fail before launch (and again on attachment) if resource policy differs."""
    model, cache, scheduler = config.model_config, config.cache_config, config.scheduler_config
    if (model.max_model_len != CONTEXT or model.runner_type != "generate"
            or model.hf_text_config.hidden_size != 2560):
        raise ValueError("The generation runner must have native 262144 context and 2560 features")
    parallel = config.parallel_config
    if (any(getattr(parallel, key) != 1 for key in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"))
            or scheduler.async_scheduling or config.use_v2_model_runner
            or config.speculative_config is not None
            or config.kv_transfer_config is not None or config.ec_transfer_config is not None):
        raise ValueError("Requires synchronous MRV1, one GPU, no speculation or cache transfer")
    options = (config.additional_config or {}).get("fuzev", {})
    if type(require_fusion) is not bool or options.get("enabled", False) is not require_fusion:
        raise ValueError("The same engine must match the explicit fusion policy")


class VllmAPI:
    """Deferred real API imports; CPU tests inject the same small interface."""

    @staticmethod
    def sampling(request):
        from vllm.sampling_params import RequestOutputKind, SamplingParams
        return SamplingParams(
            n=1, temperature=request.temperature, top_p=request.top_p,
            top_k=request.top_k or -1, seed=request.seed, max_tokens=request.max_tokens,
            stop=list(request.stop), repetition_penalty=1., presence_penalty=0.,
            frequency_penalty=0., min_p=0., detokenize=True,
            include_stop_str_in_output=False, output_kind=RequestOutputKind.FINAL_ONLY)

    @staticmethod
    def pooling(labels=None):
        from vllm.pooling_params import PoolingParams
        return PoolingParams(task="fuzev_features", extra_kwargs=(
            None if labels is None else {"baseline_token_ids": list(labels)}))

    @staticmethod
    def readout_sources():
        try:
            from vllm.v1.fuzev_readout import readout_sources
        except ImportError as error:
            raise ImportError("Requires the unpublished lab engine module vllm/v1/fuzev_readout.py") from error
        return readout_sources()


async def build_service(engine_client, config, *, inspector=describe_runtime,
                        head_factory=ServingHead, guard_factory=MemoryGuard,
                        admission=admit, api=None, service_factory=None):
    # Operator receipts are JSON: HF id2label integer keys become strings there.
    actual = json.loads(json.dumps(inspector(engine_client), allow_nan=False))
    expected = {"runtime_cell": config.runtime_cell, "runtime_sources": config.runtime_sources}
    if (config.feature_identity.get("encoding") != encoding_identity()
            or config.feature_identity.get("feature_tap") != TAP
            or config.feature_identity.get("hidden_size") != 2560):
        raise ValueError("Reference head has incompatible portable encoding/features")
    tasks = await engine_client.get_supported_tasks()
    if not {"generate", "fuzev_features"}.issubset(tasks):
        raise ValueError("The SAME engine must support generation and fuzev_features")
    admission(128 << 20)
    head = head_factory(config)
    provenance = json.loads(json.dumps({
        "reference_head": {"head_id": config.head_id,
            "manifest_sha256": config.manifest_sha256,
            "calibration_sha256": config.calibration_sha256,
            "feature_identity": config.feature_identity},
        **actual, "recorded_runtime": expected, "transfer_qualification": QUALIFICATION}))
    guard = guard_factory()
    try:
        guard.start()
        factory = service_factory or VllmService
        return factory(engine_client, engine_client.get_tokenizer(), head, config,
                       provenance, guard, api=api, admission=admission)
    except BaseException:
        guard.close()
        raise


class VllmService:
    def __init__(self, engine, tokenizer, head, config, provenance, guard, *,
                 api=None, admission=admit, clock=time.monotonic, images=None):
        self.engine, self.tokenizer, self.head = engine, tokenizer, head
        self.config = config.model_copy(deep=True)
        self.api = api or VllmAPI()
        # Headless offline extraction keeps its existing producer identity.
        self.readout_sources = readout_sources(self.api.readout_sources()) if head is not None else None
        self.provenance = json.loads(json.dumps(provenance))
        if self.readout_sources is not None:
            self.provenance["worker_readout"] = {
                "schema_version": 1, "expected_loaded_sources": dict(self.readout_sources)}
        self.identity = identity_digest(self.provenance)
        self.guard, self.admission, self.clock = guard, admission, clock
        self.verified_readouts = {}
        self.template = TEMPLATE.read_text()
        self.loop = asyncio.get_running_loop()
        self.operations, self.engine_requests = {}, set()
        self.prefixes, self.retained_tokens = OrderedDict(), 0
        self.retained_multimodal_bytes = 0
        self.closed, self.cleanup_error = False, None
        self.images = images
        renderer = getattr(engine, "renderer", None)
        if self.images is None and renderer is not None and renderer.mm_processor is not None:
            self.images = VllmImages(renderer, tokenizer, project_path("data/images"),
                self.identity, check=self._check, admission=admission)

    def _check(self):
        if self.closed or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("The service requires its live owning event loop")
        if self.cleanup_error:
            raise RuntimeError("Engine cleanup failed; admission is closed") from self.cleanup_error
        self.guard.check()
        self.admission(0)

    @asynccontextmanager
    async def _operation(self, request_id, model):
        self._check()
        if model != self.config.model_name:
            raise ValueError("Request model does not match the existing vLLM backbone")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            raise ValueError("Invalid request ID")
        if request_id in self.operations:
            raise FileExistsError("Request ID is already active")
        self.expire()
        self.operations[request_id] = asyncio.current_task()
        try:
            async with asyncio.timeout(self.config.request_timeout_seconds):
                yield
        finally:
            self.operations.pop(request_id, None)

    def cancel(self, request_id):
        task = self.operations.get(request_id)
        if task is None:
            return False
        task.cancel()
        return True

    def release_prefix(self, key):
        value = self.prefixes.pop(key, None)
        if value is not None:
            self.retained_tokens -= len(value[0].accepted)
            if value[0].multimodal is not None:
                self.retained_multimodal_bytes -= value[0].multimodal.nbytes
        return value is not None

    def expire(self):
        now = self.clock()
        for key, (_, used) in list(self.prefixes.items()):
            if now - used >= self.config.prefix_ttl_seconds:
                self.release_prefix(key)

    def _retain(self, prompt, completion, multimodal=None):
        accepted = prompt + completion
        mm_bytes = 0 if multimodal is None else multimodal.nbytes
        if multimodal is not None:
            multimodal.validate(accepted)
        if len(accepted) > self.config.retained_token_capacity:
            raise MemoryError("Retained token pool is too small for this exact sequence")
        if mm_bytes > self.config.retained_multimodal_bytes:
            raise MemoryError("Retained multimodal byte pool is too small for this payload")
        self.admission(64 * len(accepted))
        while (len(self.prefixes) >= self.config.prefix_capacity
               or self.retained_tokens + len(accepted) > self.config.retained_token_capacity
               or self.retained_multimodal_bytes + mm_bytes > self.config.retained_multimodal_bytes):
            self.release_prefix(next(iter(self.prefixes)))
        key = "prefix_" + uuid4().hex
        prefix = RetainedTokens(key, prompt, completion, accepted, self.identity, multimodal)
        self.prefixes[key] = (prefix, self.clock())
        self.retained_tokens += len(accepted)
        self.retained_multimodal_bytes += mm_bytes
        return key

    def _get_prefix(self, key):
        self.expire()
        prefix, _ = self.prefixes[key]
        if prefix.identity != self.identity or prefix.accepted != prefix.prompt + prefix.completion:
            raise ValueError("Retained token identity is incompatible")
        if prefix.multimodal is not None:
            prefix.multimodal.validate(prefix.accepted)
        self.prefixes.move_to_end(key)
        self.prefixes[key] = (prefix, self.clock())
        return prefix

    def _encode(self, text, *, image=False):
        # Explicitly disable tokenizer truncation; no decode/rerender of B.
        _text(text.replace(PLACEHOLDER, "", 1) if image else text)
        self.admission(256 * len(text))
        return _tokens(self.tokenizer.encode(text, add_special_tokens=False,
                                             truncation=False))

    def _label_ids(self, count):
        if not 2 <= count <= 16:
            raise ValueError("LM-assisted decisions require 2..16 candidates")
        vocab_size = self.engine.vllm_config.model_config.get_vocab_size()
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("The original model must provide its vocabulary size")
        labels = [self._encode(" " + chr(ord("A") + index)) for index in range(count)]
        if (any(len(label) != 1 or label[0] >= vocab_size for label in labels)
                or len({label[0] for label in labels}) != count):
            raise ValueError("Option labels must be single, distinct in-vocabulary tokens")
        return tuple(label[0] for label in labels)

    def readout_health(self):
        required = "query-lm-assisted" if getattr(self.head, "requires_lm_scores", False) else "feature-only"
        return json.loads(json.dumps({
            "schema_version": 1, "required_mode": required,
            "status": "worker-verified" if required in self.verified_readouts else "awaiting-worker-receipt",
            "expected_loaded_sources": self.readout_sources,
            "verified_receipts": self.verified_readouts,
            "scope": "readout compatibility only; not quality or activation admission",
        }))

    def _messages(self, messages, *, generation, image=False):
        values = [TextMessage.model_validate(item).model_dump() for item in messages]
        for value in values:
            _text(value["content"])
        if not values:
            values = [{"role": "user", "content": "Context for the decision."}]
        if image:
            user = next((value for value in values if value["role"] == "user"), None)
            if user is None:
                user = {"role": "user", "content": "Consider the supplied image."}
                values.insert(0, user)
            user["content"] = [{"type": "image"}, {"type": "text", "text": user["content"]}]
        rendered = self.tokenizer.apply_chat_template(
            values, tokenize=False, chat_template=self.template,
            add_generation_prompt=generation, enable_thinking=False,
            reasoning_effort="none", preserve_thinking=True)
        return self._encode(rendered, image=image)

    async def _prepare_image(self, source, tokens):
        if self.images is None:
            raise ValueError("Trusted image preprocessing is unavailable for this vLLM cell")
        result = await self.images.prepare(source, tokens)
        self._check()
        return result

    @staticmethod
    def _receipt(output, tokens, request_id):
        if output.request_id != request_id or _tokens(output.prompt_token_ids) != tokens:
            raise ValueError("Engine returned a different request or prompt token sequence")
        cached = getattr(output, "num_cached_tokens", None)
        if cached is not None and (type(cached) is not int or not 0 <= cached <= len(tokens)):
            raise ValueError("Invalid engine prefix-cache metric")
        return {"request_id": request_id, "submitted_tokens": len(tokens),
                "token_sha256": token_hash(tokens), "num_cached_tokens": cached}

    async def _read(self, iterator, tokens, request_id, features, *, baseline_token_ids=None):
        if baseline_token_ids is not None and (
                features is not True or not isinstance(baseline_token_ids, tuple)
                or not 2 <= len(baseline_token_ids) <= 16
                or any(type(token) is not int or token < 0 for token in baseline_token_ids)
                or len(set(baseline_token_ids)) != len(baseline_token_ids)):
            raise ValueError("Expected immutable 2..16 distinct query label token IDs")
        if features and self.head is None and self.readout_sources is None:
            # Bind strict headless reads without changing the producer identity.
            provider = getattr(self.api, "readout_sources", VllmAPI.readout_sources)
            self.readout_sources = readout_sources(provider())
        result = None
        async for output in iterator:
            if result is not None:
                raise ValueError("Engine produced output after a final response")
            if not output.finished:
                continue
            receipt = self._receipt(output, tokens, request_id)
            if features:
                value = output.outputs.data
                labels = () if baseline_token_ids is None else baseline_token_ids
                if (not isinstance(value, torch.Tensor) or value.shape != (2560 + len(labels),)
                        or value.dtype != torch.float32):
                    raise ValueError("Engine must return exact FP32 final-mixer feature/control geometry")
                # Own it before awaiting another iterator step or request forward.
                value = value.detach().to(device="cpu").clone()
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite final-mixer readout")
                receipt["fuzev_readout_receipt"] = readout_receipt(
                    output, tokens, request_id, labels, self.readout_sources)
            else:
                if len(output.outputs) != 1 or output.outputs[0].index != 0:
                    raise ValueError("Exactly one generation sequence is supported")
                choice = output.outputs[0]
                if choice.finish_reason not in {"stop", "length"}:
                    raise ValueError("Generation did not finish successfully")
                completion = _tokens(choice.token_ids)
                if type(choice.stop_reason) is int and completion[-1] != choice.stop_reason:
                    raise ValueError("Stopping token is missing from the accepted sequence")
                value = {"tokens": completion, "text": choice.text,
                         "finish_reason": choice.finish_reason, "stop_reason": choice.stop_reason}
                value["phase_timing"] = _generation_phase_timing(
                    output, len(tokens), len(completion), receipt["num_cached_tokens"])
            receipt.update(forward_receipt(output, len(tokens) + (0 if features else len(completion)),
                                           features=features))
            result = value, receipt
        if result is None:
            raise ValueError("Engine stream ended without a final response")
        return result

    async def _cleanup(self, request_id, iterator, reader, failed):
        try:
            if failed:
                # This API submits abort; it is NOT a GPU-free acknowledgement.
                await self.engine.abort(request_id)
        finally:
            if reader is not None:
                if not reader.done():
                    reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if iterator is not None:
                await iterator.aclose()

    async def _engine_call(self, tokens, params, *, features, multimodal=None, baseline_token_ids=None):
        self._check()
        self.admission(64 * len(tokens) + (0 if multimodal is None else multimodal.nbytes))
        request_id = "fuzev_" + uuid4().hex
        self.engine_requests.add(request_id)
        iterator = reader = None
        failed = True
        try:
            # Processed EngineInput bypasses rendering/tokenization in AsyncLLM.
            if multimodal is None:
                prompt = {"type": "token", "prompt_token_ids": list(tokens), "cache_salt": self.identity}
            else:
                self.images.check_count(multimodal)
                prompt = multimodal.engine_input(tokens, self.identity)
            method = self.engine.encode if features else self.engine.generate
            iterator = method(prompt, params, request_id)
            read_kwargs = ({} if baseline_token_ids is None else {
                "baseline_token_ids": baseline_token_ids})
            reader = asyncio.create_task(self._read(iterator, tokens, request_id, features, **read_kwargs))
            while True:
                self._check()
                done, _ = await asyncio.wait({reader}, timeout=1.)
                if done:
                    result = reader.result()
                    if multimodal is not None:
                        result[1]["images"] = multimodal.receipt()
                    self._check()
                    if features and "fuzev_readout_receipt" in result[1]:
                        proof = result[1]["fuzev_readout_receipt"]
                        mode = "query-lm-assisted" if proof["baseline_token_ids"] else "feature-only"
                        self.verified_readouts[mode] = json.loads(json.dumps(proof))
                    failed = False
                    return result
        finally:
            cleanup = asyncio.create_task(self._cleanup(request_id, iterator, reader, failed))
            interrupted = False
            # Repeated client cancellation must not abandon a background reader
            # while cleanup is still running.
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    interrupted = True
                except Exception:
                    break
            try:
                cleanup.result()
            except BaseException as error:
                self.cleanup_error = error
                raise
            finally:
                self.engine_requests.discard(request_id)
            if interrupted:
                raise asyncio.CancelledError

    async def generate(self, request: GenerationRequest, request_id):
        async with self._operation(request_id, request.model):
            source, multimodal = request.image, None
            if source is not None and self.images is None:
                raise ValueError("Trusted image preprocessing is unavailable for this vLLM cell")
            if request.prompt is not None:
                _text(request.prompt)
                prompt = self._encode(PLACEHOLDER + request.prompt if source else request.prompt,
                                      image=source is not None)
            else:
                prompt = self._messages([x.model_dump() for x in request.messages], generation=True,
                                        image=source is not None)
            if source is not None:
                prompt, multimodal = await self._prepare_image(source, prompt)
            if len(prompt) + request.max_tokens > CONTEXT:
                raise ValueError("Prompt plus output budget exceeds native context; no truncation")
            result, receipt = await self._engine_call(prompt, self.api.sampling(request), features=False,
                                                       multimodal=multimodal)
            completion = result["tokens"]
            if len(completion) > request.max_tokens:
                raise ValueError("Engine returned more than the requested output budget")
            self._check()
            prefix_id = self._retain(prompt, completion, multimodal)
            accepted = prompt + completion
            return {
                "id": request_id, "model": self.config.model_name,
                "object": "chat.completion" if request.messages is not None else "text_completion",
                "choices": [{"index": 0, "finish_reason": result["finish_reason"],
                    **({"message": {"role": "assistant", "content": result["text"]}}
                       if request.messages is not None else {"text": result["text"]})}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                          "total_tokens": len(accepted),
                          "prompt_tokens_details": {"cached_tokens": receipt["num_cached_tokens"]}},
                "prompt_token_ids": list(prompt), "completion_token_ids": list(completion),
                "prefix_id": prefix_id, "prefix_identity": self.identity,
                "accepted_token_sha256": token_hash(accepted), "stop_reason": result["stop_reason"],
                "consumed_tokens": None, "unconsumed_token_ids": None,
                "retention": "exact API-accepted tokens only; no KV/recurrent checkpoint retained",
                "stop_policy": "engine presentation trimming only; all API output IDs retained",
                "execution": receipt, "qualification": QUALIFICATION,
                "phase_timing": result["phase_timing"],
                "images": [] if multimodal is None else multimodal.receipt(),
            }

    async def decide(self, request: DecisionRequest, request_id):
        async with self._operation(request_id, request.model):
            source = getattr(request, "image", None)
            if source is not None and self.images is None:
                raise ValueError("Trusted image preprocessing is unavailable for this vLLM cell")
            if request.head_id != self.head.head_id:
                raise ValueError("Request head does not match the calibrated reference bundle")
            _text(request.question)
            _text(request.additional_context)
            candidates = [x.model_dump() for x in request.candidates]
            for candidate in candidates:
                _text(candidate["text"])
            if request.prefix_id is not None:
                prefix = self._get_prefix(request.prefix_id)
                accepted, multimodal = prefix.accepted, prefix.multimodal
            else:
                accepted = self._messages(request.messages, generation=False, image=source is not None)
                multimodal = None
                if source is not None:
                    accepted, multimodal = await self._prepare_image(source, accepted)
            additional = request.additional_context
            if request.prefix_id is not None and source is not None:
                if self.images is None:
                    raise ValueError("Trusted image preprocessing is unavailable for this vLLM cell")
                if (0 if multimodal is None else len(multimodal.images)) + 1 > self.images.image_limit:
                    raise ValueError("AB plus new image C requires this cell's image limit >=2")
                additional = PLACEHOLDER + "\n" + additional
                raw_suffix = self._encode(query_suffix(request.question, candidates, additional), image=True)
                suffix, new_image = await self._prepare_image(source, raw_suffix)
                multimodal = (multimodal.append(new_image, len(accepted)) if multimodal is not None
                              else MultimodalPayload(tuple(image.shifted(len(accepted)) for image in new_image.images)))
            else:
                suffix = self._encode(query_suffix(request.question, candidates, additional))
            tails = [self._encode(candidate_suffix(candidate)) for candidate in candidates]
            query = accepted + suffix
            if len(query) + max(map(len, tails)) > CONTEXT:
                raise ValueError("ABC plus candidate tail exceeds native context; no truncation")
            labels = self._label_ids(len(candidates)) if getattr(self.head, "requires_lm_scores", False) else None
            params = self.api.pooling() if labels is None else self.api.pooling(labels)
            query_value, query_receipt = await self._engine_call(
                query, params, features=True, multimodal=multimodal, baseline_token_ids=labels)
            owned_query = query_value[:2560].clone()
            baseline = None if labels is None else query_value[2560:].clone()
            keys, receipts = [], [query_receipt]
            tasks = [asyncio.create_task(self._engine_call(
                query + tail, self.api.pooling(), features=True, multimodal=multimodal)) for tail in tails]
            try:
                values = await asyncio.gather(*tasks)
                for feature, receipt in values:
                    keys.append(feature)
                    receipts.append(receipt)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await drain_tasks(tasks)
            self._check()
            score_kwargs = {} if baseline is None else {"baseline": baseline}
            logits, raw, calibrated = self.head.score(
                request.primitive, owned_query, torch.stack(keys), **score_kwargs)
            self._check()
            cache_counts = [x["num_cached_tokens"] for x in receipts]
            execution = VllmExecution(
                mode="full-token-replay" if request.prefix_id is not None else "fresh",
                logical_prefix_tokens=len(accepted), logical_query_tokens=len(query),
                submitted_tokens=sum(x["submitted_tokens"] for x in receipts),
                num_cached_tokens=sum(cache_counts) if None not in cache_counts else None,
                **aggregate_forward_receipts(receipts),
                requests=receipts)
            result = VllmDecisionResponse(
                id=request_id, model=self.config.model_name, primitive=request.primitive,
                head_id=self.head.head_id, calibration_id=self.head.calibration_id,
                selected_id=candidates[max(range(len(logits)), key=logits.__getitem__)]["id"],
                scores=[Score(id=candidate["id"], logit=logit, raw_probability=p,
                              calibrated_probability=q)
                        for candidate, logit, p, q in zip(candidates, logits, raw, calibrated)],
                execution=execution)
            return {**result.model_dump(mode="json"),
                "prefix_id": request.prefix_id, "prefix_identity": self.identity,
                "accepted_token_sha256": token_hash(accepted), "query_token_sha256": token_hash(query),
                "layout": "accepted-ab-append-c" if request.prefix_id else "standalone-training-encoding",
                "head_manifest_sha256": self.head.manifest_sha256,
                "qualification": QUALIFICATION, "provenance": self.provenance,
                "images": [] if multimodal is None else multimodal.receipt()}

    async def close(self):
        if self.closed:
            return
        self.closed = True
        tasks = list(self.operations.values())
        for task in tasks:
            task.cancel()
        try:
            await drain_tasks(tasks)
        finally:
            self.prefixes.clear()
            self.retained_tokens = 0
            self.retained_multimodal_bytes = 0
            self.guard.close()
        # The caller owns the ONE engine. Never shutdown an injected engine here.
