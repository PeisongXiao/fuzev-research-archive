"""CPU lifecycle fixtures only; no AsyncLLM/model/GPU qualification."""

import asyncio
import copy
import os
from types import SimpleNamespace as NS

import pytest
import torch

from fuzev_research.service_contracts import GenerationRequest, TrustedImage
from fuzev_research.vllm_generation_control import ControlCleanupError, RenderedGeneration, run_control
from fuzev_research.vllm_images import ImagePayload, MultimodalPayload
from fuzev_research.vllm_service import VllmAPI, validate_engine_config
from test_vllm_phase_timing import stats
from test_vllm_service import engine_config
from test_vllm_images import Elem, Position


RUNTIME = {
    "runtime_cell": {"engine": "vllm-fusion-disabled-control", "model": "snapshot",
                     "max_num_seqs": 4},
    "runtime_sources": {"vllm-fuzev": "a" * 40},
    "control_kind": "fusion-disabled-same-fork",
}


class Guard:
    def __init__(self):
        self.started = self.closed = False
        self.error = None

    def start(self):
        self.started = True

    def check(self):
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class Engine:
    def __init__(self):
        self.active, self.aborted, self.calls, self.closed = set(), [], [], []
        self.supported = ("generate",)
        self.reset_result = True
        self.resets = []
        self.patch = lambda output: None
        self.empty = self.duplicate = self.wait = False
        self.metrics = stats(num_generation_tokens=2)
        self.log_stats = True
        self.started = asyncio.Event()
        self.wait_token = None
        self.fail_abort = False
        self.abort_gate = None
        self.abort_started = asyncio.Event()

    def get_num_unfinished_requests(self):
        return len(self.active)

    async def get_supported_tasks(self):
        return self.supported

    async def reset_prefix_cache(self, **kwargs):
        self.resets.append(kwargs)
        return self.reset_result

    async def abort(self, request_id):
        self.aborted.append(request_id)
        self.active.discard(request_id)
        self.abort_started.set()
        if self.abort_gate is not None:
            await self.abort_gate.wait()
        if self.fail_abort:
            raise RuntimeError("abort failed")

    async def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))
        self.active.add(request_id)
        self.started.set()
        try:
            if self.wait or prompt["prompt_token_ids"][-1] == self.wait_token:
                await asyncio.Event().wait()
            if self.empty:
                return
            await asyncio.sleep(0)
            output = NS(request_id=request_id, finished=True,
                        prompt_token_ids=prompt["prompt_token_ids"], num_cached_tokens=0,
                        metrics=self.metrics,
                        outputs=[NS(index=0, token_ids=[3, 4], text="test",
                                    finish_reason="length", stop_reason=None)])
            self.patch(output)
            yield output
            if self.duplicate:
                yield output
        finally:
            self.active.discard(request_id)
            self.closed.append(request_id)


class API:
    @staticmethod
    def sampling(request):
        return NS(n=1, temperature=request.temperature, top_p=request.top_p,
                  top_k=request.top_k or -1, seed=request.seed, max_tokens=request.max_tokens,
                  stop=request.stop, stop_token_ids=[], ignore_eos=False,
                  repetition_penalty=1., presence_penalty=0., frequency_penalty=0.,
                  min_p=0., detokenize=True, include_stop_str_in_output=False,
                  output_kind=NS(name="FINAL_ONLY"))


def case(index=0):
    return RenderedGeneration(f"case-{index}", GenerationRequest(model="alias", prompt="test",
                              max_tokens=4, temperature=0), (1, 2))


async def execute(engine, cases=None, guard=None, **options):
    guard = guard or Guard()
    kwargs = dict(expected_runtime=copy.deepcopy(RUNTIME), model_name="alias", cache_salt="matched",
                  allow_inference=True, inspector=lambda _: copy.deepcopy(RUNTIME),
                  api=API(), guard_factory=lambda **_: guard, admission=lambda *_: None)
    kwargs.update(options)
    return await run_control(engine, [case()] if cases is None else cases, **kwargs)


@pytest.mark.parametrize("streams", [1, 2, 4])
def test_stock_wave_owns_exact_outputs_and_honest_metrics(streams):
    async def test():
        engine, guard = Engine(), Guard()
        result = await execute(engine, [case(i) for i in range(streams)], guard)
        assert result["qualified"] is False
        assert result["pristine_upstream"] is False
        assert result["streams"] == streams
        assert result["control_kind"] == "fusion-disabled-same-fork"
        assert result["forwarded_tokens"] is None
        assert result["logical_prompt_tokens"] == result["completion_tokens"] == 2 * streams
        assert result["cache_policy"] == "reset-acknowledged"
        assert engine.resets == [{"reset_running_requests": False, "reset_connector": False}]
        assert not engine.aborted and not engine.active
        assert guard.started and guard.closed
        for row in result["results"]:
            assert row["prompt_token_ids"] == [1, 2]
            assert row["completion_token_ids"] == [3, 4]
            assert row["phase_timing"]["status"] == "available"
            assert row["phase_timing"]["decode_tokens"] == 1
            assert row["normalized_submission_sampling"]["temperature"] == 0
        assert all(call[0]["cache_salt"] == "matched" for call in engine.calls)

    asyncio.run(test())


@pytest.mark.parametrize("options", [
    {"allow_inference": False}, {"allow_inference": 1}, {"reset_prefix_cache": 1},
    {"cache_salt": ""}, {"model_name": ""},
    {"expected_runtime": {**RUNTIME, "control_kind": "pristine-upstream"}},
    {"expected_runtime": {**RUNTIME, "runtime_sources": {"vllm-fuzev": "b" * 40}}},
])
def test_bad_admission_is_rejected_before_any_engine_work(options):
    async def test():
        engine = Engine()
        with pytest.raises(ValueError):
            await execute(engine, **options)
        assert not engine.calls and not engine.resets
    asyncio.run(test())


@pytest.mark.parametrize("count", [0, 5])
def test_invalid_wave_geometry(count):
    async def test():
        engine = Engine()
        with pytest.raises(ValueError):
            await execute(engine, [case(i) for i in range(count)])
        assert not engine.calls
    asyncio.run(test())


@pytest.mark.parametrize("mutation", ["occupied", "unsupported", "fused-task", "reset-failed", "no-stats"])
def test_unsafe_engine_cannot_run_control(mutation):
    async def test():
        engine = Engine()
        if mutation == "occupied":
            engine.active.add("other-owner")
        elif mutation == "unsupported":
            engine.supported = ("embed",)
        elif mutation == "fused-task":
            engine.supported = ("generate", "fuzev_features")
        elif mutation == "reset-failed":
            engine.reset_result = False
        else:
            engine.log_stats = False
        with pytest.raises(ValueError):
            await execute(engine)
        assert not engine.calls and not engine.aborted
    asyncio.run(test())


@pytest.mark.parametrize("mutation", [
    lambda out: setattr(out, "prompt_token_ids", [2, 1]),
    lambda out: setattr(out, "request_id", "wrong"),
    lambda out: setattr(out, "num_cached_tokens", True),
    lambda out: setattr(out.outputs[0], "token_ids", []),
    lambda out: setattr(out.outputs[0], "token_ids", [1] * 5),
    lambda out: setattr(out.outputs[0], "stop_reason", 9),
    lambda out: setattr(out.outputs[0], "finish_reason", "abort"),
])
def test_invalid_final_aborts_owned_request_and_closes_iterator(mutation):
    async def test():
        engine, guard = Engine(), Guard()
        engine.patch = mutation
        with pytest.raises(ValueError):
            await execute(engine, guard=guard)
        assert len(engine.aborted) == len(engine.closed) == 1
        assert not engine.active and guard.closed
    asyncio.run(test())


@pytest.mark.parametrize("mode", ["empty", "duplicate"])
def test_missing_or_duplicate_final_is_not_a_success(mode):
    async def test():
        engine = Engine()
        setattr(engine, mode, True)
        with pytest.raises(ValueError):
            await execute(engine)
        assert engine.aborted and not engine.active
    asyncio.run(test())


def test_cancel_keeps_ownership_through_abort_and_iterator_close():
    async def test():
        engine, guard = Engine(), Guard()
        engine.wait = True
        task = asyncio.create_task(execute(engine, guard=guard))
        await engine.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(engine.aborted) == len(engine.closed) == 1
        assert not engine.active and guard.closed
    asyncio.run(test())


def test_guard_failure_interrupts_final_only_wait_and_cleans_up():
    async def test():
        engine, guard = Engine(), Guard()
        engine.wait = True
        task = asyncio.create_task(execute(engine, guard=guard))
        await engine.started.wait()
        guard.error = MemoryError("reserved memory")
        with pytest.raises(MemoryError, match="reserved memory"):
            await asyncio.wait_for(task, 3)
        assert engine.aborted and not engine.active and guard.closed
    asyncio.run(test())


def test_missing_stats_and_within_wave_cache_sharing_are_honest():
    async def test():
        engine = Engine()
        engine.metrics = None
        engine.patch = lambda out: setattr(out, "num_cached_tokens", 1)
        result = await execute(engine)
        assert result["cache_policy"] == "reset-acknowledged-within-wave-reuse-observed"
        assert result["results"][0]["phase_timing"]["status"] == "unavailable"
        result = await execute(engine, reset_prefix_cache=False)
        assert result["cache_policy"] == "retained-observed"
        assert len(engine.resets) == 1
    asyncio.run(test())


def test_resource_gate_requires_correct_fusion_policy_without_relaxing_safety():
    config = engine_config()
    with pytest.raises(ValueError):
        validate_engine_config(config, require_fusion=False)
    config.additional_config["fuzev"]["enabled"] = False
    validate_engine_config(config, require_fusion=False)
    with pytest.raises(ValueError):
        validate_engine_config(config)
    config.cache_config.gpu_memory_utilization = .9
    validate_engine_config(config, require_fusion=False)


def test_failed_sibling_abort_is_preserved_and_engine_admission_closed():
    async def test():
        engine, guard = Engine(), Guard()
        engine.wait_token, engine.fail_abort = 77, True
        engine.patch = lambda out: setattr(out, "request_id", "wrong")
        cases = [case(), RenderedGeneration("sibling", case().request, (1, 77))]
        with pytest.raises(ControlCleanupError) as error:
            await execute(engine, cases, guard)
        assert len(error.value.faults) == 2
        assert {fault[1] for fault in error.value.faults} == {"abort"}
        assert not engine.active and guard.closed
        with pytest.raises(RuntimeError, match="shut down"):
            await execute(engine)
        assert len(engine.calls) == 2
    asyncio.run(test())


def test_repeated_cancellation_cannot_hide_abort_failure():
    async def test():
        engine, guard = Engine(), Guard()
        engine.wait = engine.fail_abort = True
        engine.abort_gate = asyncio.Event()
        task = asyncio.create_task(execute(engine, guard=guard))
        await engine.started.wait()
        task.cancel()
        await engine.abort_started.wait()
        task.cancel()
        engine.abort_gate.set()
        with pytest.raises(ControlCleanupError) as error:
            await task
        assert len(error.value.faults) == 1
        assert not engine.active and len(engine.closed) == 1 and guard.closed
        assert engine._fuzev_stock_control_requires_shutdown
    asyncio.run(test())


def image_case():
    source = TrustedImage(path="procedural-not-a-real-image.png", sha256="a" * 64)
    raw = (10, 99, 11)
    processed = {"type": "multimodal", "prompt_token_ids": [10, 99, 99, 11],
                 "mm_kwargs": {"image": [{"pixel_values": Elem(torch.ones(4, 3)),
                                          "image_grid_thw": Elem(torch.tensor([1, 4, 4]))}]},
                 "mm_hashes": {"image": ["procedural-namespace"]},
                 "mm_placeholders": {"image": [Position(1, 2)]}}
    tokens, image = ImagePayload.capture(source, processed, raw, 99, 1, 1 << 20)
    request = GenerationRequest(model="alias", prompt="procedural", image=source, max_tokens=4)
    return RenderedGeneration("image", request, tokens, MultimodalPayload((image,)))


async def execute_image(engine, value):
    runtime = copy.deepcopy(RUNTIME)
    runtime["runtime_cell"]["multimodal_config"] = {"image_limit": 1}
    return await execute(engine, [value], expected_runtime=runtime,
                         inspector=lambda _: copy.deepcopy(runtime))


@pytest.mark.parametrize("mutation", ["digest", "two-images", "range", "payload"])
def test_invalid_image_binding_rejected_before_cache_reset(mutation):
    async def test():
        engine, value = Engine(), image_case()
        payload = value.multimodal
        if mutation == "digest":
            request = value.request.model_copy(deep=True)
            request.image = TrustedImage(path="other.png", sha256="b" * 64)
            value = RenderedGeneration(value.name, request, value.prompt_token_ids, payload)
        elif mutation == "two-images":
            value = RenderedGeneration(value.name, value.request, value.prompt_token_ids,
                                       payload.append(payload, 2))
        elif mutation == "range":
            value = RenderedGeneration(value.name, value.request, (10, 55, 55, 11), payload)
        else:
            payload.images[0].item["pixel_values"].data.add_(1)
        with pytest.raises(ValueError):
            await execute_image(engine, value)
        assert not engine.calls and not engine.resets
    asyncio.run(test())


def test_image_receipt_witness_and_cloned_submission():
    async def test():
        engine, value = Engine(), image_case()
        expected = copy.deepcopy(value.multimodal.receipt())
        result = await execute_image(engine, value)
        assert result["results"][0]["images"] == expected
        sent = engine.calls[0][0]["mm_kwargs"]["image"][0]["pixel_values"].data
        original = value.multimodal.images[0].item["pixel_values"].data
        assert sent.data_ptr() != original.data_ptr()
    asyncio.run(test())


def test_borrowed_image_mutation_during_forward_invalidates_wave():
    async def test():
        engine, value = Engine(), image_case()
        engine.patch = lambda _: value.multimodal.images[0].item["pixel_values"].data.add_(1)
        with pytest.raises(ValueError, match="changed"):
            await execute_image(engine, value)
    asyncio.run(test())


@pytest.mark.skipif(os.environ.get("FUZEV_VLLM_API_INSPECTION") != "1",
                    reason="Requires vLLM installed (CPU); set FUZEV_VLLM_API_INSPECTION=1")
def test_real_sampling_normalization_receipt_with_cpu_fake_engine():
    assert not torch.cuda.is_available()

    async def test():
        result = await execute(Engine(), api=VllmAPI())
        sampling = result["results"][0]["normalized_submission_sampling"]
        assert sampling["temperature"] == 0
        assert sampling["top_k"] == 0 and sampling["top_p"] == 1
        assert sampling["output_kind"] == "FINAL_ONLY"
        assert sampling["ignore_eos"] is False and sampling["stop_token_ids"] == []
    asyncio.run(test())
