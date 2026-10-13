"""Exclusive, fusion-disabled generation control on an already loaded AsyncLLM.

This does not construct a model, fit a head, render a prompt or establish that
the source tree is pristine upstream. The caller supplies exact rendered inputs
and owns the engine throughout this offline operation.
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from uuid import uuid4

from .encoding import token_hash
from .service_contracts import GenerationRequest
from .telemetry import MemoryGuard, admit
from .vllm_images import MAX_PAYLOAD_BYTES, MultimodalPayload
from .vllm_ple_storage import validate_ple_storage_config
from .vllm_service import (
    CONTEXT, VllmAPI, _generation_phase_timing, _tokens,
    describe_runtime, drain_tasks,
)


@dataclass(frozen=True)
class RenderedGeneration:
    name: str
    request: GenerationRequest
    prompt_token_ids: tuple[int, ...]
    multimodal: MultimodalPayload | None = None


class ControlCleanupError(RuntimeError):
    def __init__(self, faults):
        super().__init__("Stock request cleanup failed; the caller must shut down this engine")
        self.faults = tuple(faults)


def describe_control(engine):
    receipt = describe_runtime(engine, require_fusion=False)
    config = engine.vllm_config
    validate_ple_storage_config(config)
    receipt["control_policy"] = {
        "log_stats": engine.log_stats,
        "enforce_eager": config.model_config.enforce_eager,
        "safetensors_load_strategy": config.load_config.safetensors_load_strategy,
        "load_format": config.load_config.load_format,
        "ple_storage": config.additional_config.get("fuzev", {}).get("ple_storage", "resident"),
        "exl3_ple_storage": (os.environ.get("VLLM_EXL3_NGRAM_TABLE")
                             if config.model_config.quantization == "exl3" else None),
        "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
        "splitting_ops": list(config.compilation_config.splitting_ops or []),
    }
    receipt["control_kind"] = "fusion-disabled-same-fork"
    return receipt


def _idle(engine):
    if getattr(engine, "_fuzev_stock_control_requires_shutdown", False):
        raise RuntimeError("A previous stock request aborted; shut down this engine before reuse")
    count = engine.get_num_unfinished_requests()
    if type(count) is not int or count != 0:
        raise ValueError("Generation control requires an exclusive idle engine")


def _cases(cases, model, capacity, image_limit):
    if not cases or len(cases) > capacity:
        raise ValueError("A control wave must fit the engine's sequence capacity")
    owned, names = [], set()
    for case in cases:
        if not isinstance(case, RenderedGeneration) or not isinstance(case.request, GenerationRequest):
            raise ValueError("Expected an explicit rendered generation case")
        if (not isinstance(case.name, str) or not case.name or len(case.name) > 64
                or case.name in names):
            raise ValueError("Control case names must be unique and bounded")
        names.add(case.name)
        request = case.request.model_copy(deep=True)
        prompt = _tokens(case.prompt_token_ids)
        if request.model != model or len(prompt) + request.max_tokens > CONTEXT:
            raise ValueError("Rendered input does not match the pinned model/native window")
        if (request.image is None) != (case.multimodal is None):
            raise ValueError("Image requests require their exact processed multimodal payload")
        if case.multimodal is not None and not isinstance(case.multimodal, MultimodalPayload):
            raise ValueError("Expected the existing trusted processed-image payload")
        image_receipt = []
        if case.multimodal is not None:
            payload = case.multimodal
            if (image_limit < 1 or len(payload.images) != 1
                    or payload.images[0].rgb_sha256 != request.image.sha256):
                raise ValueError("Generation needs one admitted image matching the request RGB digest")
            if not 0 < payload.nbytes <= MAX_PAYLOAD_BYTES:
                raise MemoryError("Generation image payload exceeds its owned byte bound")
            payload.validate(prompt)
            image_receipt = json.loads(json.dumps(payload.receipt(), allow_nan=False))
        owned.append((RenderedGeneration(case.name, request, prompt, case.multimodal), image_receipt))
    return owned


async def _read(iterator, case, image_receipt, request_id, check):
    result = None
    async for output in iterator:
        check()
        if result is not None:
            raise ValueError("Stock engine emitted output after its final response")
        if not output.finished:
            continue
        if (output.request_id != request_id
                or _tokens(output.prompt_token_ids) != case.prompt_token_ids):
            raise ValueError("Stock engine changed the submitted request/token sequence")
        if len(output.outputs) != 1 or output.outputs[0].index != 0:
            raise ValueError("Stock control requires exactly one completion per request")
        choice = output.outputs[0]
        tokens = _tokens(choice.token_ids)
        if (choice.finish_reason not in {"stop", "length"}
                or len(tokens) > case.request.max_tokens
                or type(choice.stop_reason) is int and tokens[-1] != choice.stop_reason):
            raise ValueError("Stock completion has an invalid budget/accepted stopping boundary")
        cached = getattr(output, "num_cached_tokens", None)
        if cached is not None and (type(cached) is not int
                                   or not 0 <= cached <= len(case.prompt_token_ids)):
            raise ValueError("Stock engine returned an invalid cached-token count")
        result = {
            "name": case.name, "request_id": request_id,
            "generation_request": case.request.model_dump(mode="json"),
            "prompt_token_ids": list(case.prompt_token_ids),
            "completion_token_ids": list(tokens),
            "accepted_token_sha256": token_hash(case.prompt_token_ids + tokens),
            "text": choice.text, "finish_reason": choice.finish_reason,
            "stop_reason": choice.stop_reason,
            "usage": {"prompt_tokens": len(case.prompt_token_ids),
                      "completion_tokens": len(tokens),
                      "total_tokens": len(case.prompt_token_ids) + len(tokens)},
            "num_cached_tokens": cached,
            "phase_timing": _generation_phase_timing(
                output, len(case.prompt_token_ids), len(tokens), cached),
            "forwarded_tokens": None,
            "forward_counter_status": "unavailable-stock-path-not-instrumented",
            "images": image_receipt,
        }
    if result is None:
        raise ValueError("Stock stream ended without a final completion")
    return result


async def _request(engine, case, image_receipt, cache_salt, check, api, clock, cleanup_faults):
    request_id = "fuzev_stock_" + uuid4().hex
    iterator = reader = None
    failed = True
    start = clock()
    try:
        check()
        prompt = ({"type": "token", "prompt_token_ids": list(case.prompt_token_ids),
                   "cache_salt": cache_salt} if case.multimodal is None else
                  case.multimodal.engine_input(case.prompt_token_ids, cache_salt))
        params = api.sampling(case.request)
        submitted_sampling = {
            name: getattr(params, name) for name in (
                "n", "temperature", "top_p", "top_k", "seed", "max_tokens", "stop",
                "stop_token_ids", "ignore_eos", "repetition_penalty", "presence_penalty",
                "frequency_penalty", "min_p", "detokenize", "include_stop_str_in_output")
        }
        submitted_sampling["output_kind"] = params.output_kind.name
        submitted_sampling = json.loads(json.dumps(submitted_sampling, allow_nan=False))
        iterator = engine.generate(prompt, params, request_id)
        reader = asyncio.create_task(_read(iterator, case, image_receipt, request_id, check))
        while True:
            check()
            done, _ = await asyncio.wait({reader}, timeout=1.)
            if done:
                result = reader.result()
                break
        check()
        result["normalized_submission_sampling"] = submitted_sampling
        result["sampling_scope"] = "before engine generation-config EOS augmentation"
        result["elapsed_seconds"] = clock() - start
        if not math.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] <= 0:
            raise ValueError("Stock client interval is not positive")
        failed = False
        return result
    finally:
        async def cleanup():
            # Abort submission is not a GPU-free acknowledgement. Even a normal
            # abort closes admission until the outer caller destroys the engine.
            if failed:
                engine._fuzev_stock_control_requires_shutdown = True
            try:
                if failed:
                    await engine.abort(request_id)
            except BaseException as error:
                cleanup_faults.append((request_id, "abort", error))
            finally:
                if reader is not None:
                    if not reader.done():
                        reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                if iterator is not None:
                    try:
                        await iterator.aclose()
                    except BaseException as error:
                        engine._fuzev_stock_control_requires_shutdown = True
                        cleanup_faults.append((request_id, "iterator-close", error))

        # Cancellation must not abandon the owning reader/abort while returning.
        cleanup_task = asyncio.create_task(cleanup())
        interrupted = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                interrupted = True
        cleanup_task.result()
        if interrupted:
            raise asyncio.CancelledError


async def run_control(engine, cases, *, expected_runtime, model_name, cache_salt,
                      allow_inference=False, reset_prefix_cache=True,
                      inspector=describe_control, api=None, guard_factory=MemoryGuard,
                      admission=admit, clock=time.monotonic):
    """Run one wave; callers sequence warmup/repetitions and the fused workload.

    Both sides must match source/payload, explicit sampling, rendered IDs, cache
    policy and engine geometry before their measurements can be compared. This
    control is not a pristine-upstream result or a completed qualification gate.
    """
    if allow_inference is not True:
        raise ValueError("Offline control inference needs explicit operator admission")
    if type(reset_prefix_cache) is not bool:
        raise ValueError("Prefix-cache reset policy must be explicit")
    if not isinstance(cache_salt, str) or not 1 <= len(cache_salt) <= 128:
        raise ValueError("Control needs the explicit matched cache salt")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Control needs the explicit matched serving alias")
    _idle(engine)
    if getattr(engine, "log_stats", None) is not True:
        raise ValueError("Control must enable actual AsyncLLM request statistics")
    actual = inspector(engine)
    expected = json.loads(json.dumps(expected_runtime, allow_nan=False))
    if actual != expected or actual.get("control_kind") != "fusion-disabled-same-fork":
        raise ValueError("Stock control runtime/source receipt differs from the operator pins")
    cell = actual["runtime_cell"]
    if cell.get("engine") != "vllm-fusion-disabled-control":
        raise ValueError("A fused generation-only service is not the stock-path control")
    image_limit = (cell.get("multimodal_config") or {}).get("image_limit", 0)
    cases = _cases(cases, model_name, cell["max_num_seqs"], image_limit)
    tasks_supported = await engine.get_supported_tasks()
    if "generate" not in tasks_supported or "fuzev_features" in tasks_supported:
        raise ValueError("Stock engine must support generation without the fused feature task")
    admission(sum(64 * len(case.prompt_token_ids) +
                  (0 if case.multimodal is None else case.multimodal.nbytes)
                  for case, _ in cases))
    guard = guard_factory()
    tasks = []
    cleanup_faults = []
    try:
        guard.start()

        def check():
            guard.check()
            admission(0)

        check()
        if reset_prefix_cache and await engine.reset_prefix_cache(
                reset_running_requests=False, reset_connector=False) is not True:
            raise ValueError("Stock engine did not acknowledge an idle prefix-cache reset")
        _idle(engine)
        start = clock()
        tasks = [asyncio.create_task(_request(engine, case, image_receipt, cache_salt, check,
                                             api or VllmAPI(), clock, cleanup_faults))
                 for case, image_receipt in cases]
        results = await asyncio.gather(*tasks)
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Stock wave interval is not positive")
        check()
        _idle(engine)
        if inspector(engine) != expected:
            raise ValueError("Stock control source/runtime pins changed during the wave")
        for case, image_receipt in cases:
            if case.multimodal is not None:
                case.multimodal.validate(case.prompt_token_ids)
                if case.multimodal.receipt() != image_receipt:
                    raise ValueError("Borrowed image bindings changed after stock submission")
        if reset_prefix_cache and any(row["num_cached_tokens"] not in (None, 0) for row in results):
            # A reset does not rule out within-wave sharing of identical prefixes.
            cache_observation = "reset-acknowledged-within-wave-reuse-observed"
        else:
            cache_observation = "reset-acknowledged" if reset_prefix_cache else "retained-observed"
        return {
            "schema_version": 1, "qualified": False,
            "pristine_upstream": False,
            "control_kind": "fusion-disabled-same-fork", "runtime": actual,
            "model": model_name,
            "streams": len(cases), "elapsed_seconds": elapsed,
            "cache_policy": cache_observation, "cache_salt": cache_salt,
            "reset_covers": "prefix cache only; no claim of cold encoder/processor/NVMe caches",
            "logical_prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in results),
            "completion_tokens": sum(row["usage"]["completion_tokens"] for row in results),
            "forwarded_tokens": None, "results": results,
            "limitations": ["Caller attests procedural input and exclusive engine ownership",
                            "No pristine-upstream, fused-path, hybrid-state or quality proof",
                            "Wall intervals include scheduling/client overhead; not GPU kernel time"],
        }
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await drain_tasks(tasks)
        finally:
            guard.close()
            if cleanup_faults:
                engine._fuzev_stock_control_requires_shutdown = True
                raise ControlCleanupError(cleanup_faults)
