"""Actual adapter serialization into the client, with CPU-only fake engines."""

import asyncio
import json
from threading import Event
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from fuzev_research.native_service import NativeService
from fuzev_research.runtime_harness import (
    generation_phases,
    summarize_phases,
    validate_generation,
)
from test_native_service import Backend, Guard, Head, generation
from test_vllm_phase_timing import stats
from test_vllm_service import generate, service


def check_serialized(output, request, engine):
    value = json.loads(json.dumps(output, allow_nan=False))
    config = SimpleNamespace(
        model=request.model, engine=engine, prefix_identity=value["prefix_identity"]
    )
    validate_generation(value, config, request)
    timing = generation_phases(value, engine)
    result = summarize_phases(
        [{"mode": "generation-only", "generation_phase_timing": timing}]
    )
    assert result["phase_generation_observations"] == 1
    assert result["phase_available_observations"] == int(
        timing["status"] == "available"
    )
    assert result["phase_rate_scope"].startswith("duration-weighted per-request")
    return timing, result


@pytest.mark.parametrize("mode", ["normal", "single", "bad-clock"])
def test_native_adapter_serialization_and_client_phase_accounting(mode):
    request = (
        generation()
        if mode != "single"
        else generation().model_copy(update={"max_tokens": 1})
    )
    ticks = iter([100.0, 102.0, 104.0, 106.0])
    clock = (lambda: next(ticks)) if mode != "bad-clock" else (lambda: 0.0)
    svc = NativeService(
        Backend(),
        "unit",
        Head(),
        {"runtime": "synthetic"},
        Guard(),
        admission=lambda *_: None,
        phase_clock=clock,
    )
    try:
        output = svc.generate(request, Event(), "native-interop")
        timing, result = check_serialized(output, request, "native")
        if mode == "bad-clock":
            assert timing["status"] == "unavailable"
            assert result["prefill_tokens_per_second"] is None
            assert result["decode_only_tokens_per_second"] is None
        else:
            assert timing["status"] == "available"
            assert result["prefill_tokens_per_second"] == 0.5
            assert result["decode_only_tokens_per_second"] == (
                0.5 if mode == "normal" else None
            )
        assert output["unconsumed_token_ids"] == output["completion_token_ids"][-1:]
        assert len(svc.lane.calls) == len(output["completion_token_ids"])
    finally:
        svc.close()
    assert all(branch.closed for branch in svc.lane.branches)


@pytest.mark.parametrize("mode", ["normal", "missing-stats", "preempted"])
def test_vllm_adapter_serialization_and_client_phase_accounting(mode):
    async def case():
        svc = service()
        request = generate()
        value = (
            None
            if mode == "missing-stats"
            else stats(num_preemptions=1 if mode == "preempted" else 0)
        )
        svc.engine.patch_output = lambda output, _kind: setattr(
            output, "metrics", value
        )
        try:
            output = await svc.generate(request, "vllm-interop")
            timing, result = check_serialized(output, request, "vllm")
            if mode == "normal":
                assert timing["status"] == "available"
                assert (
                    result["prefill_tokens_per_second"]
                    == (output["usage"]["prompt_tokens"] - svc.engine.cached) / 3
                )
                assert result["decode_only_tokens_per_second"] == 0.5
            else:
                assert timing["status"] == "unavailable"
                assert result["prefill_tokens_per_second"] is None
                assert result["decode_only_tokens_per_second"] is None
            assert output["completion_token_ids"] == [700, 701, 0]
            assert output["consumed_tokens"] is None
            assert not svc.engine.aborted
        finally:
            await svc.close()
        assert not svc.prefixes and not svc.engine_requests

    asyncio.run(case())
