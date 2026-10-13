"""Opt-in real vLLM CPU constructor/signature checks, never engine construction."""

import inspect
import asyncio
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FUZEV_VLLM_API_INSPECTION") != "1",
                              reason="Requires vLLM installed (CPU); set FUZEV_VLLM_API_INSPECTION=1")


@pytest.mark.parametrize("labels", [None, [1, 2], list(range(16))])
def test_real_comparison_readout_parameters_reach_fused_task_validator(labels):
    from types import SimpleNamespace

    import torch
    from vllm.pooling_params import PoolingParams
    from fuzev_research.vllm_comparison_extract import OfflineAPI

    assert not torch.cuda.is_available()
    params = OfflineAPI.pooling(labels)
    assert isinstance(params, PoolingParams) and params.task == "fuzev_features"
    params.verify(SimpleNamespace(get_vocab_size=lambda: 32))
    assert params.skip_reading_prefix_cache is False
    assert (params.extra_kwargs or {}) == (
        {} if labels is None else {"baseline_token_ids": labels}
    )
    assert not params.requires_token_ids
    assert all(getattr(params, name) is None for name in params.all_parameters)


def test_real_forward_stats_and_public_output_constructors_reach_owned_adapter_receipts():
    import msgspec
    import torch
    from vllm.outputs import CompletionOutput, PoolingOutput, PoolingRequestOutput, RequestOutput
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.fuzev_observability import FuzevForwardStats
    from fuzev_research.vllm_service import VllmService

    assert not torch.cuda.is_available()
    for api in (RequestOutput, PoolingRequestOutput):
        assert "fuzev_forward_stats" in inspect.signature(api).parameters
    generation = FuzevForwardStats(completed_forward_steps=2, completed_token_positions=5,
        last_completed_start=4, last_completed_end=5)
    core = EngineCoreOutput("api-gen", [9, 0], fuzev_forward_stats=generation)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(core), type=EngineCoreOutput)
    assert decoded.fuzev_forward_stats == generation
    output = RequestOutput(request_id="api-gen", prompt=None, prompt_token_ids=list(range(4)),
        prompt_logprobs=None, outputs=[CompletionOutput(index=0, text="accepted", token_ids=[9, 0],
            cumulative_logprob=None, logprobs=None, finish_reason="stop", stop_reason=0)],
        finished=True, num_cached_tokens=3, fuzev_forward_stats=decoded.fuzev_forward_stats)
    readout = FuzevForwardStats(completed_forward_steps=1, completed_token_positions=1,
        last_completed_start=31, last_completed_end=32, feature_readout_position=31)
    pooling = PoolingRequestOutput(request_id="api-features", outputs=PoolingOutput(torch.ones(2560)),
        prompt_token_ids=list(range(32)), num_cached_tokens=31, finished=True,
        fuzev_forward_stats=readout)

    async def case():
        # Only the adapter reader; no initialized service, model or engine.
        service = object.__new__(VllmService)
        async def one(value):
            yield value
        accepted, gen = await service._read(one(output), tuple(range(4)), "api-gen", False)
        feature, feature_receipt = await service._read(one(pooling), tuple(range(32)), "api-features", True)
        assert accepted["tokens"] == (9, 0) and gen["forwarded_tokens"] == 5
        assert gen["fuzev_forward_stats"]["causal_consumed_prefix"] is None
        assert gen["feature_readout_position_verified"] is None
        assert feature.shape == (2560,) and feature.device.type == "cpu"
        assert feature_receipt["forwarded_tokens"] == 1
        assert feature_receipt["num_cached_tokens"] == 31
        assert feature_receipt["feature_readout_position_verified"] is True
        assert json.loads(json.dumps(feature_receipt)) == feature_receipt
        feature_receipt["fuzev_forward_stats"]["completed_token_positions"] = 99
        assert readout.completed_token_positions == 1
    asyncio.run(case())


@pytest.mark.parametrize("invalid", ["empty", "bool", "feature-position"])
def test_real_stats_constructors_do_not_turn_invalid_scalars_into_forward_evidence(invalid):
    import torch
    from vllm.outputs import PoolingOutput, PoolingRequestOutput
    from vllm.v1.fuzev_observability import FuzevForwardStats
    from fuzev_research.vllm_receipts import forward_receipt

    assert not torch.cuda.is_available()
    kwargs = {"completed_forward_steps": 1, "completed_token_positions": 1,
              "last_completed_start": 3, "last_completed_end": 4, "feature_readout_position": 3}
    if invalid == "empty":
        kwargs = {}
    elif invalid == "bool":
        kwargs["completed_token_positions"] = True
    else:
        kwargs["feature_readout_position"] = 2
    output = PoolingRequestOutput(request_id="api-features", outputs=PoolingOutput(torch.ones(2560)),
        prompt_token_ids=list(range(4)), num_cached_tokens=3, finished=True,
        fuzev_forward_stats=FuzevForwardStats(**kwargs))
    with pytest.raises(ValueError):
        forward_receipt(output, 4, features=True)


@pytest.mark.parametrize("slice_endpoints", ["plain-int", "cpu-int32", "cpu-int64"])
def test_real_multimodal_constructors_preserve_owned_field_types_and_ranges(slice_endpoints):
    import torch
    from vllm.inputs.engine import MultiModalInput, mm_input
    from vllm.multimodal.inputs import (MultiModalBatchedField, MultiModalFieldElem,
        MultiModalFlatField, MultiModalKwargsItem, MultiModalKwargsItems, PlaceholderRange)
    from fuzev_research.service_contracts import TrustedImage
    from fuzev_research.vllm_images import ImagePayload, MultimodalPayload

    assert not torch.cuda.is_available()
    raw = (17, 150001, 150002, 150003, 18)
    expanded = (17, 150001, 150002, 150002, 150002, 150002, 150003, 18)
    endpoint_dtype = {"cpu-int32": torch.int32, "cpu-int64": torch.int64}.get(slice_endpoints)
    start, stop = (0, 16) if endpoint_dtype is None else (
        torch.tensor(0, dtype=endpoint_dtype, device="cpu"),
        torch.tensor(16, dtype=endpoint_dtype, device="cpu"),
    )
    item = MultiModalKwargsItem({
        "pixel_values": MultiModalFieldElem(torch.ones(16, 12),
            MultiModalFlatField(slices=[slice(start, stop)], dim=0)),
        "image_grid_thw": MultiModalFieldElem(torch.tensor([1, 4, 4], dtype=torch.int64),
            MultiModalBatchedField(keep_on_cpu=True)),
    })
    position = PlaceholderRange(offset=2, length=4, is_embed=torch.ones(4, dtype=torch.bool))
    kwargs = MultiModalKwargsItems({"image": [item]})
    processed = mm_input(prompt_token_ids=list(expanded), mm_kwargs=kwargs,
                         mm_hashes={"image": ["synthetic-not-a-real-processor-hash"]},
                         mm_placeholders={"image": [position]})
    source = TrustedImage(path="/synthetic/not-opened/" + "1" * 64 + ".png", sha256="1" * 64)
    tokens, binding = ImagePayload.capture(source, processed, raw, 150002, 2, 512 << 20)
    assert tokens == expanded and type(binding.item) is MultiModalKwargsItem
    assert type(binding.item["pixel_values"].field) is MultiModalFlatField
    assert binding.item["pixel_values"].field.slices == [slice(0, 16)]
    assert binding.item["image_grid_thw"].field.keep_on_cpu
    for name in ("pixel_values", "image_grid_thw"):
        owned = binding.item[name].data
        assert owned.dtype == item[name].data.dtype
        assert owned.device.type == "cpu"
        assert owned.data_ptr() != item[name].data.data_ptr()
    payload = MultimodalPayload((binding,))
    output = MultiModalInput(**payload.engine_input(tokens + (0,), "synthetic-cache-salt"))
    assert output["prompt_token_ids"] == list(tokens + (0,))
    assert type(output["mm_kwargs"]["image"][0]) is MultiModalKwargsItem
    assert type(output["mm_placeholders"]["image"][0]) is PlaceholderRange
    assert output["mm_hashes"] == processed["mm_hashes"]
    emitted_item = output["mm_kwargs"]["image"][0]
    for name in ("pixel_values", "image_grid_thw"):
        emitted = emitted_item[name].data
        assert emitted.dtype == binding.item[name].data.dtype
        assert emitted.device.type == "cpu"
        assert emitted.data_ptr() != binding.item[name].data.data_ptr()
    for name, expected in (("start", 0), ("stop", 16)):
        original = getattr(item["pixel_values"].field.slices[0], name)
        owned = getattr(binding.item["pixel_values"].field.slices[0], name)
        emitted = getattr(emitted_item["pixel_values"].field.slices[0], name)
        if endpoint_dtype is None:
            assert type(owned) is type(emitted) is int
            assert owned == emitted == expected
        else:
            for endpoint in (original, owned, emitted):
                assert isinstance(endpoint, torch.Tensor)
                assert endpoint.shape == () and endpoint.dtype == endpoint_dtype
                assert endpoint.device.type == "cpu" and endpoint.item() == expected
            assert len({original.data_ptr(), owned.data_ptr(), emitted.data_ptr()}) == 3
            original.fill_(expected + 1)
            emitted.fill_(expected + 2)
            assert owned.item() == expected
    assert json.loads(json.dumps(binding.receipt(), allow_nan=False)) == binding.receipt()
    output["mm_kwargs"]["image"][0]["pixel_values"].data.fill_(9)
    output["mm_placeholders"]["image"][0].is_embed.fill_(False)
    payload.validate(tokens)
    assert torch.all(binding.item["pixel_values"].data == 1)
    shifted = payload.append(payload, len(tokens))
    combined = tokens + tokens
    assert shifted.engine_input(combined, "salt")["mm_placeholders"]["image"][1].offset == len(tokens) + 2


def test_real_preprocessing_and_autoprocessor_signatures_only():
    from transformers import AutoProcessor
    from vllm.renderers.base import BaseRenderer
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.transformers_utils.processor import get_processor

    renderer = inspect.signature(BaseRenderer.process_for_engine_async)
    assert renderer.parameters["skip_mm_cache"].default is False
    assert {"prompt", "arrival_time"}.issubset(renderer.parameters)
    input_processor = inspect.signature(InputProcessor.process_inputs)
    assert "prompt" in input_processor.parameters
    for method in (AsyncLLM.generate, AsyncLLM.encode):
        assert {"prompt", "request_id"}.issubset(inspect.signature(method).parameters)
    assert "pretrained_model_name_or_path" in inspect.signature(AutoProcessor.from_pretrained).parameters
    assert {"processor_name", "revision", "trust_remote_code"}.issubset(
        inspect.signature(get_processor).parameters)


def test_real_multimodal_config_records_preprocessing_beyond_graph_hash():
    from pydantic import TypeAdapter
    from vllm.config.multimodal import MultiModalConfig

    first = MultiModalConfig(limit_per_prompt={"image": {"count": 2, "width": 1280, "height": 1280}},
        mm_processor_kwargs={"truncation": False}, mm_processor_cache_gb=.25)
    second = MultiModalConfig(limit_per_prompt={"image": 1},
        mm_processor_kwargs={"truncation": True}, mm_processor_cache_gb=.5)
    assert first.compute_hash() == second.compute_hash()
    snapshot = TypeAdapter(MultiModalConfig).dump_python(first, mode="json", serialize_as_any=True)
    assert snapshot["limit_per_prompt"]["image"] == {"count": 2, "width": 1280, "height": 1280}
    assert snapshot["mm_processor_kwargs"] == {"truncation": False}
    assert snapshot["mm_processor_cache_gb"] == .25
    assert snapshot != TypeAdapter(MultiModalConfig).dump_python(second, mode="json", serialize_as_any=True)
