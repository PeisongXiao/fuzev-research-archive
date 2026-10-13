"""CPU fixtures with synthetic timestamps, never live phase measurements."""

import inspect
import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.vllm_service import VllmAPI, VllmService, _generation_phase_timing
from test_vllm_service import Engine, config, decision, generate, run, service


TIMESTAMPS = ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
SCOPE = ("unpreempted engine-core request intervals; prefill includes first token; "
         "decode excludes first token; not GPU kernel time")
EXPECTED = {
    "schema_version": 1, "source": "vllm-engine-core-request-stats",
    "status": "available", "reason": None, "scope": SCOPE,
    "prefill_seconds": 3., "decode_seconds": 4., "queue_seconds": 2.,
    "prompt_tokens": 4, "completion_tokens": 3, "decode_tokens": 2,
    "reported_cached_prompt_tokens": 2, "num_preemptions": 0,
}


def stats(**changes):
    return NS(**(dict(num_generation_tokens=3, num_preemptions=0,
        queued_ts=10., scheduled_ts=12., first_token_ts=15., last_token_ts=19.,
        arrival_time=1_700_000_000., first_token_latency=900., is_corrupted=False) | changes))


def timing(value, *, completion_tokens=3, cached=2):
    return _generation_phase_timing(NS(metrics=value), 4, completion_tokens, cached)


def assert_unavailable(value, reason):
    assert value.keys() == EXPECTED.keys()
    assert value["schema_version"] == 1
    assert value["source"] == EXPECTED["source"] and value["scope"] == SCOPE
    assert value["status"] == "unavailable" and value["reason"] == reason
    assert all(value[key] is None for key in ("prefill_seconds", "decode_seconds", "queue_seconds"))
    assert value["completion_tokens"] == value["decode_tokens"] + 1
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_exact_schema_uses_core_intervals_and_excludes_first_token_from_decode():
    value = timing(stats())
    assert value == EXPECTED
    assert json.loads(json.dumps(value, allow_nan=False)) == value


@pytest.mark.parametrize("wall_time", [0., -10., 1e30, float("nan"), None, "not a clock"])
def test_wall_clock_arrival_and_frontend_ttft_never_affect_phase_intervals(wall_time):
    assert timing(stats(arrival_time=wall_time, first_token_latency=wall_time)) == EXPECTED


def test_observer_never_reads_frontend_or_client_clocks(monkeypatch):
    class CoreOnly:
        def __getattr__(self, name):
            if name in {"arrival_time", "first_token_latency"}:
                raise AssertionError("Frontend clocks must not be read")
            return getattr(stats(), name)

    def forbidden():
        raise AssertionError("Client clocks must not be read")

    monkeypatch.setattr("fuzev_research.vllm_service.time.time", forbidden)
    monkeypatch.setattr("fuzev_research.vllm_service.time.monotonic", forbidden)
    assert timing(CoreOnly()) == EXPECTED


@pytest.mark.parametrize("field", TIMESTAMPS)
@pytest.mark.parametrize("invalid", [None, 0., -1., float("nan"), float("inf"),
                                     float("-inf"), True, "10", [], object()])
def test_invalid_timestamp_never_becomes_a_phase_duration(field, invalid):
    assert_unavailable(timing(stats(**{field: invalid})), "invalid_" + field)


@pytest.mark.parametrize("field", (*TIMESTAMPS, "num_preemptions", "num_generation_tokens"))
def test_missing_required_field_never_invents_defaults(field):
    value = stats()
    delattr(value, field)
    assert_unavailable(timing(value), "invalid_" + field)


@pytest.mark.parametrize("changes,reason", [
    ({"queued_ts": 13.}, "reversed_timestamps"),
    ({"scheduled_ts": 16.}, "reversed_timestamps"),
    ({"last_token_ts": 14.}, "reversed_timestamps"),
    ({"first_token_ts": 12.}, "nonpositive_prefill_interval"),
    ({"last_token_ts": 15.}, "nonpositive_decode_interval"),
    ({"num_generation_tokens": 0}, "invalid_num_generation_tokens"),
    ({"num_generation_tokens": 2}, "generation_token_count_mismatch"),
    ({"num_generation_tokens": 4}, "generation_token_count_mismatch"),
    ({"num_preemptions": 1}, "preempted_request"),
    ({"num_preemptions": 2}, "preempted_request"),
    ({"is_corrupted": True}, "corrupted_request"),
    ({"is_corrupted": 1}, "invalid_is_corrupted"),
])
def test_unqualified_stats_have_explicit_reasons_and_no_partial_timings(changes, reason):
    value = timing(stats(**changes))
    assert_unavailable(value, reason)
    assert value["reported_cached_prompt_tokens"] == 2
    assert value["num_preemptions"] == changes.get("num_preemptions", 0)


@pytest.mark.parametrize("field", ["num_preemptions", "num_generation_tokens"])
@pytest.mark.parametrize("invalid", [None, -1, True, 3., "3", float("nan"), float("inf"), []])
def test_counters_are_exact_nonnegative_integers_not_coerced(field, invalid):
    value = timing(stats(**{field: invalid}))
    assert_unavailable(value, "invalid_" + field)
    if field == "num_preemptions":
        assert value["num_preemptions"] is None


@pytest.mark.parametrize("output", [NS(), NS(metrics=None)])
def test_missing_stats_are_unavailable_not_an_estimated_client_latency(output):
    value = _generation_phase_timing(output, 4, 3, 2)
    assert_unavailable(value, "missing_request_stats")
    assert value["num_preemptions"] is None


@pytest.mark.parametrize("metrics", [42, "legacy", {}, [], object()])
def test_incompatible_metrics_objects_remain_optional(metrics):
    assert_unavailable(timing(metrics), "invalid_num_preemptions")


class BrokenMetrics:
    def __getattr__(self, _name):
        raise RuntimeError("Synthetic incompatible metrics accessor")


def test_accessor_exceptions_and_overflow_cannot_fail_generation_observation():
    assert_unavailable(timing(BrokenMetrics()), "malformed_request_stats")
    assert_unavailable(_generation_phase_timing(BrokenMetrics(), 4, 3, 2),
                       "malformed_request_stats")
    assert_unavailable(timing(stats(queued_ts=10 ** 400)), "malformed_request_stats")


def test_zero_queue_and_single_token_zero_decode_are_valid_but_emit_no_rate():
    value = timing(stats(num_generation_tokens=1, queued_ts=12., last_token_ts=15.),
                   completion_tokens=1)
    assert value == EXPECTED | {"queue_seconds": 0., "decode_seconds": 0.,
                               "completion_tokens": 1, "decode_tokens": 0}
    assert_unavailable(timing(stats(num_generation_tokens=1), completion_tokens=1),
                       "single_token_decode_interval")


@pytest.mark.parametrize("cached", [None, 0, 4])
def test_cache_count_is_reported_not_subtracted_from_logical_prompt(cached):
    assert timing(stats(), cached=cached) == EXPECTED | {"reported_cached_prompt_tokens": cached}


@pytest.mark.parametrize("cached", [-1, 5, True, 2., "2", float("nan")])
def test_helper_defensively_rejects_invalid_or_over_prompt_cache_counts(cached):
    value = timing(stats(), cached=cached)
    assert_unavailable(value, "invalid_cached_prompt_tokens")
    assert value["reported_cached_prompt_tokens"] is None


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("mode", ["valid", "missing", "mismatch", "broken", "preempted", "nan"])
def test_generation_contract_and_exact_ab_are_unchanged_except_top_level_timing(monkeypatch, chat, mode):
    monkeypatch.setattr("fuzev_research.vllm_service.uuid4", lambda: NS(hex="synthetic"))

    async def case():
        baseline, observed = service(), service()
        request = generate(**({"prompt": None, "messages": [{"role": "user", "content": "A"}]}
                              if chat else {}), temperature=.7, top_p=.8, top_k=7, seed=23,
                           stop=["STOP"])
        before = request.model_dump()

        def patch(output, _kind):
            output.metrics = {"valid": stats(), "missing": None,
                "mismatch": stats(num_generation_tokens=4), "broken": BrokenMetrics(),
                "preempted": stats(num_preemptions=1), "nan": stats(first_token_ts=float("nan"))}[mode]
        observed.engine.patch_output = patch
        try:
            expected = await baseline.generate(request, "unchanged")
            actual = await observed.generate(request, "unchanged")
            phase = actual.pop("phase_timing")
            assert_unavailable(expected.pop("phase_timing"), "missing_request_stats")
            assert actual == expected
            assert "phase_timing" not in actual["execution"]
            assert phase["status"] == ("available" if mode == "valid" else "unavailable")
            assert phase["completion_tokens"] == 3 and phase["decode_tokens"] == 2
            assert phase["prompt_tokens"] == len(actual["prompt_token_ids"])
            assert actual["completion_token_ids"] == [700, 701, 0]
            assert actual["stop_reason"] == 0
            assert actual["usage"]["completion_tokens"] == 3
            assert request.model_dump() == before
            assert observed.engine.calls == baseline.engine.calls
            assert observed.engine.inputs == baseline.engine.inputs
            assert observed.engine.tokenizer.renders == baseline.engine.tokenizer.renders
            assert observed.api.generations[0].request is request
            assert not observed.api.readouts
            assert not observed.engine.aborted and not observed.operations
            prefix = observed.prefixes[actual["prefix_id"]][0]
            assert prefix.accepted == tuple(actual["prompt_token_ids"]) + (700, 701, 0)
            assert prefix == baseline.prefixes[expected["prefix_id"]][0]
        finally:
            await observed.close()
            await baseline.close()
    run(case())


def test_text_stop_retains_all_api_ids_despite_engine_count_mismatch():
    async def case():
        svc = service()
        svc.engine.completion = [700, 701]
        def patch(output, _kind):
            output.outputs[0].stop_reason = "STOP"
            output.outputs[0].text = "trimmed"
            output.metrics = stats(num_generation_tokens=3)
        svc.engine.patch_output = patch
        try:
            result = await svc.generate(generate(stop=["STOP"]), "text-stop")
            assert result["completion_token_ids"] == [700, 701]
            assert result["choices"][0] == {"index": 0, "text": "trimmed", "finish_reason": "stop"}
            assert result["stop_reason"] == "STOP"
            assert result["usage"]["completion_tokens"] == 2
            assert result["phase_timing"]["decode_tokens"] == 1
            assert_unavailable(result["phase_timing"], "generation_token_count_mismatch")
            assert svc.prefixes[result["prefix_id"]][0].completion == (700, 701)
        finally:
            await svc.close()
    run(case())


@pytest.mark.parametrize("final_available", [False, True])
def test_only_final_stats_are_captured_before_the_iterator_can_mutate_them(final_available):
    async def case():
        value = stats()
        completion = NS(index=0, token_ids=[700, 701, 0], text="trimmed",
                        finish_reason="stop", stop_reason=0)
        output = NS(request_id="owned", prompt_token_ids=[1, 2, 3, 4], num_cached_tokens=2,
                    outputs=[completion], finished=False, metrics=value)
        async def stream():
            yield output
            output.finished = True
            output.metrics = value if final_available else None
            yield output
            value.last_token_ts = 999.
            value.num_generation_tokens = 99
            value.num_preemptions = 1
            output.num_cached_tokens = 0
            completion.token_ids[:] = [999]
        reader = object.__new__(VllmService)
        accepted, receipt = await reader._read(stream(), (1, 2, 3, 4), "owned", False)
        assert accepted["tokens"] == (700, 701, 0) and receipt["num_cached_tokens"] == 2
        if final_available:
            assert accepted["phase_timing"] == EXPECTED
        else:
            assert_unavailable(accepted["phase_timing"], "missing_request_stats")
    run(case())


def test_typed_readout_never_calls_the_generation_observer(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("Phase timing applies only to accepted generation")
    monkeypatch.setattr("fuzev_research.vllm_service._generation_phase_timing", forbidden)
    async def case():
        svc = service()
        try:
            result = await svc.decide(decision(), "no-phase-readout")
            assert "phase_timing" not in result and "phase_timing" not in result["execution"]
            assert all("phase_timing" not in row for row in result["execution"]["requests"])
            assert all(call[0] == "encode" for call in svc.engine.calls)
        finally:
            await svc.close()
    run(case())


@pytest.mark.parametrize("cached", [-1, 100, True])
def test_existing_invalid_cache_receipt_contract_is_not_relaxed(cached):
    async def case():
        svc = service()
        svc.engine.cached = cached
        try:
            with pytest.raises(ValueError, match="Invalid engine prefix-cache metric"):
                await svc.generate(generate(), "bad-cache")
            assert not svc.prefixes and svc.engine.aborted
        finally:
            await svc.close()
    run(case())


@pytest.mark.parametrize("available", [False, True])
def test_http_generation_serializes_the_optional_top_level_schema(available):
    from fastapi.testclient import TestClient
    from fuzev_research.vllm_server import create_app

    @asynccontextmanager
    async def engine_context():
        engine = Engine()
        def patch(output, _kind):
            output.metrics = stats() if available else stats(first_token_ts=float("nan"))
        engine.patch_output = patch
        yield engine
    async def builder(engine, _config):
        return service(engine)
    with TestClient(create_app(engine_context, config(), builder=builder)) as client:
        for path, request in (("/v1/completions", generate()),
                ("/v1/chat/completions", generate(prompt=None, messages=[{"role": "user", "content": "A"}]))):
            response = client.post(path, json=request.model_dump())
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["completion_token_ids"] == [700, 701, 0]
            assert result["phase_timing"]["status"] == ("available" if available else "unavailable")
            assert "phase_timing" not in result["execution"]
            json.dumps(result, allow_nan=False)


real_api = pytest.mark.skipif(os.environ.get("FUZEV_VLLM_API_INSPECTION") != "1",
                             reason="Requires vLLM installed (CPU); set FUZEV_VLLM_API_INSPECTION=1")


@real_api
@pytest.mark.parametrize("top_k", [0, 7])
def test_real_sampling_contract_stays_final_only_and_log_stats_default_is_enabled(top_k):
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    assert not torch.cuda.is_available()
    request = generate(temperature=.7, top_p=.8, top_k=top_k, seed=23, stop=["STOP"])
    before = request.model_dump()
    params = VllmAPI.sampling(request)
    expected = dict(n=1, temperature=.7, top_p=.8, top_k=top_k or -1, seed=23,
        max_tokens=8, stop=["STOP"], repetition_penalty=1., presence_penalty=0.,
        frequency_penalty=0., min_p=0., detokenize=True, include_stop_str_in_output=False,
        output_kind=RequestOutputKind.FINAL_ONLY)
    assert {key: getattr(params, key) for key in expected} == expected
    assert request.model_dump() == before
    assert inspect.signature(AsyncLLM.from_vllm_config).parameters["disable_log_stats"].default is False


@real_api
@pytest.mark.parametrize("log_stats", [False, True])
@pytest.mark.parametrize("mode", ["normal", "single", "mismatch", "preempted"])
def test_real_request_state_stats_and_final_only_output_with_synthetic_core_events(log_stats, mode):
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType, EngineCoreOutput, FinishReason
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.metrics.stats import IterationStats, LoRARequestStates, RequestStateStats

    assert not torch.cuda.is_available()
    state = RequestState(request_id="real-api", external_req_id="real-api", parent_req=None,
        request_index=0, lora_request=None, output_kind=RequestOutputKind.FINAL_ONLY,
        prompt=None, prompt_token_ids=[1, 2, 3, 4], prompt_embeds=None,
        logprobs_processor=LogprobsProcessor(tokenizer=None, logprobs=None, prompt_logprobs=None,
            cumulative_logprob=None, num_logprobs=None, num_prompt_logprobs=None),
        detokenizer=IncrementalDetokenizer(), max_tokens_param=8, arrival_time=1_700_000_000.,
        queue=None, log_stats=log_stats, stream_interval=1)
    state.num_cached_tokens = 2
    ids = [0] if mode == "single" else [700, 701, 0]
    iteration, loras = IterationStats(), LoRARequestStates()
    for index, token in enumerate(ids):
        first, final = index == 0, index == len(ids) - 1
        events = ([EngineCoreEvent(EngineCoreEventType.QUEUED, 10.),
                   EngineCoreEvent(EngineCoreEventType.SCHEDULED, 12.)] if first else [])
        if mode == "preempted" and final:
            events += [EngineCoreEvent(EngineCoreEventType.PREEMPTED, 18.),
                       EngineCoreEvent(EngineCoreEventType.SCHEDULED, 18.5)]
        core = EngineCoreOutput("real-api", [token], events=events,
            finish_reason=FinishReason.STOP if final else None, stop_reason=0 if final else None)
        if log_stats:
            assert isinstance(state.stats, RequestStateStats)
            iteration.update_from_output(core, 15. + 2 * index, first, state.stats, loras, None)
        state.detokenizer.update(core.new_token_ids, final)
        output = state.make_request_output(core.new_token_ids, None, core.finish_reason, core.stop_reason)
        if not final:
            assert output is None
    assert isinstance(output, RequestOutput) and output.finished
    assert output.metrics is state.stats
    if mode == "mismatch" and log_stats:
        # Deliberate synthetic engine/API disagreement, not an observed EOS claim.
        state.stats.num_generation_tokens += 1

    async def case():
        async def stream():
            yield output
        accepted, _ = await object.__new__(VllmService)._read(stream(), (1, 2, 3, 4), "real-api", False)
        assert accepted["tokens"] == tuple(ids)
        phase = accepted["phase_timing"]
        if not log_stats:
            assert_unavailable(phase, "missing_request_stats")
        elif mode in {"mismatch", "preempted"}:
            assert_unavailable(phase, "generation_token_count_mismatch" if mode == "mismatch"
                               else "preempted_request")
        else:
            assert phase == EXPECTED | ({"completion_tokens": 1, "decode_tokens": 0,
                                        "decode_seconds": 0.} if mode == "single" else {})
    run(case())
