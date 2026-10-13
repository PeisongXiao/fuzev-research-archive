"""Synthetic CPU-only adapter tests. No vLLM model, CUDA, or dataset access."""

import asyncio
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.contracts import DecisionRequest
from fuzev_research.encoding import candidate_suffix, identity, query_suffix, token_hash
from fuzev_research.service_contracts import GenerationRequest
from fuzev_research.serving_head import TAP
from fuzev_research.vllm_service import (CONTEXT, VllmService, VllmServiceConfig,
                               build_service, validate_engine_config)
from fuzev_research.vllm_receipts import READOUT_SOURCE_FILES, aggregate_forward_receipts


class Guard:
    def __init__(self, **kwargs):
        self.error, self.closed = None, False
        self.kwargs = kwargs

    def start(self):
        self.check()

    def check(self):
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class Tokenizer:
    def __init__(self):
        self.calls, self.renders = [], []

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        self.calls.append(text)
        return [ord(x) + 1 for x in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["enable_thinking"] is False
        assert kwargs["reasoning_effort"] == "none"
        assert kwargs["preserve_thinking"] is True
        self.renders.append((messages, kwargs))
        return "[context]" + "|".join(x["content"] for x in messages) + (
            "[assistant]" if kwargs["add_generation_prompt"] else "")

    def decode(self, *_args, **_kwargs):
        raise AssertionError("Generated B must NEVER be decoded/re-rendered by the adapter")


class Head:
    head_id = "reference"
    manifest_sha256 = "1" * 64
    calibration_id = "sha256:" + "2" * 64

    def score(self, primitive, query, candidates):
        assert query.shape == (2560,)
        assert candidates.shape[1] == 2560
        assert query.dtype == candidates.dtype == torch.float32
        assert query.device.type == candidates.device.type == "cpu"
        self.observed = (query.clone(), candidates.clone())
        logits = candidates[:, 0] / 10
        return logits.tolist(), torch.softmax(logits, 0).tolist(), torch.softmax(logits / 2, 0).tolist()


class API:
    def __init__(self):
        self.generations, self.readouts = [], []

    def sampling(self, request):
        value = NS(kind="generate", request=request)
        self.generations.append(value)
        return value

    def pooling(self, labels=None):
        value = NS(task="fuzev_features", extra_kwargs=(
            None if labels is None else {"baseline_token_ids": list(labels)}))
        self.readouts.append(value)
        return value

    @staticmethod
    def readout_sources():
        return {name: "3" * 64 for name in READOUT_SOURCE_FILES}


class Engine:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.calls, self.aborted = [], []
        self.inputs = []
        self.active, self.peak = 0, 0
        self.wait, self.abort_wait = None, None
        self.entered, self.aborting = asyncio.Event(), asyncio.Event()
        self.completion = [700, 701, 0]
        self.cached = 1
        self.patch_output = lambda output, _kind: None
        self.tasks = ("generate", "fuzev_features")
        self.fail_call = None
        self.mutate_after_yield = False
        self.vllm_config = NS(model_config=NS(get_vocab_size=lambda: 1024))
        self.synthetic_projections = []

    async def get_supported_tasks(self):
        return self.tasks

    def get_tokenizer(self):
        return self.tokenizer

    async def check_health(self):
        pass

    async def abort(self, request_id):
        self.aborted.append(request_id)
        self.aborting.set()
        if self.abort_wait:
            await self.abort_wait.wait()

    async def _run(self, kind, prompt, params, request_id):
        assert prompt["type"] in {"token", "multimodal"}
        assert isinstance(prompt["prompt_token_ids"], list)
        assert prompt["cache_salt"]
        tokens = tuple(prompt["prompt_token_ids"])
        self.calls.append((kind, tokens, params, request_id))
        self.inputs.append(deepcopy(prompt))
        call = len(self.calls)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.set()
        try:
            if self.wait:
                await self.wait.wait()
            await asyncio.sleep(0)
            if self.fail_call == call:
                raise RuntimeError("Synthetic engine failure")
            if kind == "encode":
                assert params.task == "fuzev_features"
                data = torch.full((2560,), float(sum(tokens) % 127), dtype=torch.float32)
                labels = (params.extra_kwargs or {}).get("baseline_token_ids", [])
                if labels:
                    self.synthetic_projections.append((tokens, labels.copy()))
                    data = torch.cat((data, torch.tensor(labels, dtype=torch.float32) / 4))
                payload = NS(data=data)
            else:
                payload = [NS(index=0, token_ids=list(self.completion), text="presentation NOT B tokens",
                              finish_reason="stop", stop_reason=0)]
            output = NS(request_id=request_id, outputs=payload,
                        prompt_token_ids=list(tokens), finished=True, num_cached_tokens=self.cached)
            if kind == "encode":
                output.fuzev_readout_receipt = {
                    "schema_version": 1, "worker_request_id": request_id + "-internal",
                    "request_id": request_id, "prompt_tokens": len(tokens),
                    "prompt_token_sha256": token_hash(tokens), "feature_readout_position": len(tokens) - 1,
                    "feature_width": 2560, "baseline_token_ids": labels.copy(),
                    "lm_projection_calls": 1 if labels else 0, "generated_token_count": 0,
                    "worker_loaded_sources": API.readout_sources(),
                }
            self.patch_output(output, kind)
            yield output
            if self.mutate_after_yield:
                prompt["prompt_token_ids"][:] = [999]
                output.prompt_token_ids[:] = [999]
                if kind == "encode":
                    data.fill_(999)
                    output.fuzev_readout_receipt["baseline_token_ids"][:] = [999]
                    output.fuzev_readout_receipt["worker_loaded_sources"].clear()
                else:
                    payload[0].token_ids[:] = [999]
                if prompt["type"] == "multimodal":
                    prompt["mm_kwargs"]["image"][0]["pixel_values"].data.fill_(999)
        finally:
            self.active -= 1

    def generate(self, prompt, params, request_id):
        return self._run("generate", prompt, params, request_id)

    def encode(self, prompt, params, request_id):
        return self._run("encode", prompt, params, request_id)


def config(**kwargs):
    values = dict(model_name="unit-vllm", head_directory="not-opened-by-fake",
                  manifest_sha256="1" * 64, calibration_sha256="2" * 64,
                  feature_identity={"engine": "native-reference-only", "encoding": identity(),
                                    "feature_tap": TAP, "hidden_size": 2560},
                  runtime_cell={"engine": "synthetic-vllm-cell"},
                  runtime_sources={"recipes": "synthetic", "vllm-fuzev": "synthetic"})
    return VllmServiceConfig(**(values | kwargs))


def generate(**kwargs):
    return GenerationRequest(**(dict(model="unit-vllm", prompt="prompt-A", max_tokens=8) | kwargs))


def decision(**kwargs):
    return DecisionRequest(**(dict(model="unit-vllm", primitive="choice", question="Which?",
        messages=[{"role": "user", "content": "context"}],
        candidates=[{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]) | kwargs))


def service(engine=None, **config_kwargs):
    engine = engine or Engine()
    cfg = config(**config_kwargs)
    return VllmService(engine, engine.tokenizer, Head(), cfg, {"cell": "synthetic"},
                       Guard(), api=API(), admission=lambda *_: None)


def run(coroutine):
    return asyncio.run(coroutine)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.005)


def worker_stats(output, *, features, positions=None, missing=0):
    """Synthetic scalars in the committed core receipt shape, not GPU evidence."""
    end = len(output.prompt_token_ids) + (0 if features else 2)
    positions = end if positions is None else positions
    steps = (1 if features else 3) if positions else 0
    return NS(completed_forward_steps=steps, completed_token_positions=positions,
        last_completed_start=(end - positions if features else end - 1) if positions else None,
        last_completed_end=end if positions else None,
        feature_readout_position=end - 1 if features and positions else None,
        missing_worker_receipts=missing, replayed_token_positions=None,
        reused_token_positions=None, causal_consumed_prefix=None)


@pytest.mark.parametrize("final_available", [True, False])
def test_final_generation_receipt_is_owned_not_summed_or_inherited_from_stream(final_available):
    async def case():
        svc = service()
        original = svc.engine.generate
        async def stream(prompt, params, request_id):
            async for output in original(prompt, params, request_id):
                stats = worker_stats(output, features=False)
                for _ in range(2):
                    yield NS(**(vars(output) | {"finished": False, "fuzev_forward_stats": stats}))
                output.fuzev_forward_stats = stats if final_available else None
                yield output
                # A borrowed/mutable runtime snapshot cannot mutate the JSON receipt.
                stats.completed_token_positions = 999
                stats.last_completed_end = 999
        svc.engine.generate = stream
        result = await svc.generate(generate(), "stream")
        receipt = result["execution"]
        assert result["consumed_tokens"] is result["unconsumed_token_ids"] is None
        assert receipt["submitted_tokens"] == len(result["prompt_token_ids"])
        assert receipt["num_cached_tokens"] == 1
        if final_available:
            positions = len(result["prompt_token_ids"]) + 2
            assert receipt["forwarded_tokens"] == positions  # Not three cumulative snapshots.
            assert receipt["fuzev_forward_stats"]["completed_token_positions"] == positions
            assert receipt["fuzev_forward_stats"]["last_completed_end"] == positions
            assert receipt["forwarded_tokens_lower_bound"] is False
            assert receipt["forward_counts_status"] == "worker-acknowledged"
            assert receipt["fuzev_forward_stats"]["causal_consumed_prefix"] is None
        else:
            assert receipt["fuzev_forward_stats"] is receipt["forwarded_tokens"] is None
            assert receipt["forwarded_tokens_lower_bound"] is None
            assert receipt["forward_counts_status"] == "unavailable"
            assert "unavailable" in receipt["counter_gap"]
        assert receipt["feature_readout_position_verified"] is None
        assert json.loads(json.dumps(receipt)) == receipt
        await svc.close()
    run(case())


def test_decision_totals_sum_independent_query_candidates_not_prior_generation_or_cache_hits():
    async def case():
        svc = service()
        positions = [7, 11, 13]
        def patch(output, kind):
            index = next(i for i, call in enumerate(svc.engine.calls) if call[-1] == output.request_id)
            output.fuzev_forward_stats = worker_stats(output, features=kind == "encode",
                positions=positions[index - 1] if kind == "encode" else None)
        svc.engine.patch_output = patch
        generated = await svc.generate(generate(), "gen")
        result = await svc.decide(decision(messages=None, prefix_id=generated["prefix_id"]), "score")
        execution = result["execution"]
        assert execution["completed_token_positions"] == execution["forwarded_tokens"] == 31
        assert execution["completed_forward_steps"] == 3
        assert execution["missing_worker_receipts"] == execution["unavailable_forward_receipts"] == 0
        assert execution["forward_counts_status"] == "worker-acknowledged"
        assert execution["forwarded_tokens_lower_bound"] is False
        assert execution["num_cached_tokens"] == 3
        assert execution["submitted_tokens"] == sum(len(call[1]) for call in svc.engine.calls[1:])
        assert all(execution[key] is None for key in ("replayed_tokens", "reused_tokens", "consumed_tokens"))
        assert len({item["request_id"] for item in execution["requests"]}) == 3
        for item, count, call in zip(execution["requests"], positions, svc.engine.calls[1:]):
            assert item["fuzev_forward_stats"]["completed_token_positions"] == count
            assert item["fuzev_forward_stats"]["feature_readout_position"] == len(call[1]) - 1
            assert item["feature_readout_position_verified"] is True
            assert item["token_sha256"] == token_hash(call[1])
        assert json.loads(json.dumps(execution)) == execution
        await svc.close()
    run(case())


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_one_missing_final_worker_receipt_keeps_decision_totals_null(missing_index):
    async def case():
        svc = service()
        def patch(output, _kind):
            index = next(i for i, call in enumerate(svc.engine.calls) if call[-1] == output.request_id)
            output.fuzev_forward_stats = (None if index == missing_index else
                                         worker_stats(output, features=True, positions=7))
        svc.engine.patch_output = patch
        execution = (await svc.decide(decision(), "partial"))["execution"]
        assert execution["num_cached_tokens"] == 3
        assert execution["unavailable_forward_receipts"] == 1
        assert execution["forward_counts_status"] == "unavailable"
        assert all(execution[key] is None for key in (
            "completed_forward_steps", "completed_token_positions", "missing_worker_receipts",
            "forwarded_tokens", "forwarded_tokens_lower_bound"))
        for index, item in enumerate(execution["requests"]):
            assert (item["fuzev_forward_stats"] is None) == (index == missing_index)
        await svc.close()
    run(case())


@pytest.mark.parametrize("zero_acknowledged", [False, True])
def test_missing_worker_updates_are_explicit_lower_bounds_not_zero_execution(zero_acknowledged):
    async def case():
        svc = service()
        def patch(output, _kind):
            index = next(i for i, call in enumerate(svc.engine.calls) if call[-1] == output.request_id)
            output.fuzev_forward_stats = worker_stats(output, features=True,
                positions=0 if index == 1 and zero_acknowledged else 7,
                missing=2 if index == 1 else 0)
        svc.engine.patch_output = patch
        execution = (await svc.decide(decision(), "gaps"))["execution"]
        assert execution["completed_token_positions"] == (14 if zero_acknowledged else 21)
        assert execution["forwarded_tokens"] == execution["completed_token_positions"]
        assert execution["forwarded_tokens_lower_bound"] is True
        assert execution["forward_counts_status"] == "lower-bound"
        assert execution["missing_worker_receipts"] == 2
        assert execution["unavailable_forward_receipts"] == 0
        branch = execution["requests"][1]
        assert branch["forwarded_tokens_lower_bound"] is True
        assert branch["feature_readout_position_verified"] is (None if zero_acknowledged else True)
        assert branch["fuzev_forward_stats"]["causal_consumed_prefix"] is None
        await svc.close()
    run(case())


@pytest.mark.parametrize("field", ["missing_worker_receipts", "feature_readout_position"])
def test_older_populated_counter_schema_stays_usable_but_explicitly_unavailable(field):
    async def case():
        svc = service()
        def patch(output, _kind):
            output.fuzev_forward_stats = worker_stats(output, features=True, positions=2)
            delattr(output.fuzev_forward_stats, field)
        svc.engine.patch_output = patch
        execution = (await svc.decide(decision(), "old-schema"))["execution"]
        assert execution["completed_token_positions"] is execution["forwarded_tokens"] is None
        assert execution["forward_counts_status"] == "unavailable"
        assert execution["unavailable_forward_receipts"] == 3
        assert all(item["fuzev_forward_stats"] is None for item in execution["requests"])
        assert all("incomplete" in item["counter_gap"] for item in execution["requests"])
        assert not svc.engine.aborted
        await svc.close()
    run(case())


@pytest.mark.parametrize("field,value", [
    ("completed_forward_steps", True), ("completed_forward_steps", 0),
    ("completed_token_positions", -1), ("completed_token_positions", "4"),
    ("completed_token_positions", 1.5), ("completed_token_positions", torch.tensor(4)),
    ("missing_worker_receipts", True), ("missing_worker_receipts", -1),
    ("last_completed_start", None), ("last_completed_start", "end"),
    ("last_completed_end", None), ("last_completed_end", "past-end"),
    ("feature_readout_position", "before-final"), ("feature_readout_position", True),
    ("reused_token_positions", float("nan")), ("causal_consumed_prefix", "past-end"),
    ("completed_token_positions", None),
])
def test_malformed_scalar_or_feature_range_receipts_fail_without_successful_totals(field, value):
    async def case():
        svc = service()
        def patch(output, _kind):
            stats = worker_stats(output, features=True)
            length = len(output.prompt_token_ids)
            translated = {"end": length, "past-end": length + 1, "before-final": length - 2}
            setattr(stats, field, translated.get(value, value) if isinstance(value, str) else value)
            output.fuzev_forward_stats = stats
        svc.engine.patch_output = patch
        with pytest.raises(ValueError):
            await svc.decide(decision(), "bad-receipt")
        assert svc.engine.aborted and not svc.engine_requests and not svc.operations
        assert not svc.prefixes and not hasattr(svc.head, "observed")
        await svc.close()
    run(case())


@pytest.mark.parametrize("ending", ["exception", "no-final", "cancel", "unsuccessful"])
def test_partial_worker_counts_cannot_make_failure_or_cancel_a_successful_total(ending):
    async def case():
        svc = service()
        original, entered = svc.engine.generate, asyncio.Event()
        async def stream(prompt, params, request_id):
            async for output in original(prompt, params, request_id):
                output.fuzev_forward_stats = worker_stats(output, features=False)
                yield NS(**(vars(output) | {"finished": False}))
                entered.set()
                if ending == "exception":
                    raise RuntimeError("Synthetic terminal failure")
                if ending == "cancel":
                    await asyncio.Event().wait()
                if ending == "unsuccessful":
                    output.outputs[0].finish_reason = "abort"
                    yield output
        svc.engine.generate = stream
        task = asyncio.create_task(svc.generate(generate(), "partial-failure"))
        await entered.wait()
        if ending == "cancel":
            assert svc.cancel("partial-failure")
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError if ending == "exception" else ValueError):
                await task
        assert svc.engine.aborted and not svc.prefixes and not svc.operations
        assert not svc.engine_requests and not svc.engine.active
        await svc.close()
    run(case())


def test_duplicate_request_snapshots_cannot_be_aggregated_as_independent_work():
    with pytest.raises(ValueError, match="independent"):
        aggregate_forward_receipts([{"request_id": "same"}, {"request_id": "same"}])


def test_exact_ab_replay_candidate_order_and_no_invented_cache_metrics():
    async def case():
        svc = service()
        svc.engine.mutate_after_yield = True
        generated = await svc.generate(generate(), "gen")
        prefix = svc.prefixes[generated["prefix_id"]][0]
        assert prefix.accepted == tuple(generated["prompt_token_ids"]) + (700, 701, 0)
        assert generated["consumed_tokens"] is None and generated["unconsumed_token_ids"] is None
        with pytest.raises(FrozenInstanceError):
            prefix.accepted = (999,)
        request = decision(messages=None, prefix_id=prefix.id, additional_context="new evidence")
        result = await svc.decide(request, "score")
        candidates = [x.model_dump() for x in request.candidates]
        ids = lambda text: tuple(ord(x) + 1 for x in text)
        query = prefix.accepted + ids(query_suffix(request.question, candidates, "new evidence"))
        calls = svc.engine.calls
        assert [x[0] for x in calls] == ["generate", "encode", "encode", "encode"]
        assert calls[1][1] == query
        assert [x[1] for x in calls[2:]] == [query + ids(candidate_suffix(x)) for x in candidates]
        assert prefix.accepted[-1] == 0  # Accepted EOS is replayed, never discarded.
        assert all("presentation" not in x for x in svc.tokenizer.calls)
        assert not svc.tokenizer.renders
        assert result["accepted_token_sha256"] == token_hash(prefix.accepted)
        execution = result["execution"]
        assert execution["mode"] == "full-token-replay"
        assert not execution["checkpoint_retained"]
        assert execution["num_cached_tokens"] == 3
        assert execution["submitted_tokens"] == sum(len(x[1]) for x in calls[1:])
        assert all(execution[key] is None for key in (
            "forwarded_tokens", "replayed_tokens", "reused_tokens", "consumed_tokens"))
        assert svc.head.observed[0][0] == sum(query) % 127  # Not overwritten engine scratch.
        assert not svc.engine_requests
        await svc.close()
    run(case())


@pytest.mark.parametrize("primitive", ["choice", "noul", "ordinal"])
def test_typed_primitives_use_only_pooling_and_calibrated_head(primitive):
    async def case():
        svc = service()
        candidates = [{"id": "yes", "text": "Yes", "ordinal_value": 0},
                      {"id": "no", "text": "No", "ordinal_value": 1}]
        result = await svc.decide(decision(primitive=primitive, candidates=candidates), "decision")
        assert result["primitive"] == primitive
        assert len(result["scores"]) == 2
        assert sum(x["calibrated_probability"] for x in result["scores"]) == pytest.approx(1.)
        assert all(kind == "encode" for kind, *_ in svc.engine.calls)
        assert not svc.api.generations
        assert result["execution"]["mode"] == "fresh"
        assert not svc.tokenizer.renders[0][1]["add_generation_prompt"]
        await svc.close()
    run(case())


def test_chat_and_empty_standalone_context_use_portable_template():
    async def case():
        svc = service()
        response = await svc.generate(generate(prompt=None, messages=[{"role": "user", "content": "Hi"}]), "chat")
        assert response["choices"][0]["message"]["content"] == "presentation NOT B tokens"
        await svc.decide(decision(messages=[]), "empty")
        assert svc.tokenizer.renders[0][1]["add_generation_prompt"]
        assert svc.tokenizer.renders[1][0] == [{"role": "user", "content": "Context for the decision."}]
        await svc.close()
    run(case())


def test_parallel_branches_and_requests_complete_through_engine_scheduling():
    async def case():
        svc = service()
        candidates = [{"id": str(x), "text": "candidate " + str(x)} for x in range(16)]
        results = await asyncio.gather(*[
            svc.decide(decision(candidates=candidates), f"d{x}") for x in range(4)])
        assert len(results) == 4
        assert len(svc.engine.calls) == 4 * 17
        assert not svc.operations and not svc.engine_requests
        await svc.close()
    run(case())


def test_five_requests_are_accepted_and_cancellation_waits_for_cleanup():
    async def case():
        svc = service()
        svc.engine.wait, svc.engine.abort_wait = asyncio.Event(), asyncio.Event()
        tasks = [asyncio.create_task(svc.generate(generate(), f"g{x}")) for x in range(5)]
        await until(lambda: svc.engine.active == 5)
        with pytest.raises(FileExistsError):
            await svc.generate(generate(), "g0")
        assert svc.cancel("g0")
        await svc.engine.aborting.wait()
        tasks[0].cancel()
        await asyncio.sleep(.01)
        assert len(svc.operations) == len(svc.engine_requests) == 5
        assert not tasks[0].done()
        svc.engine.abort_wait.set()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert not svc.cancel("g0")
        for task in tasks[1:]:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert len(svc.engine.aborted) == 5
        assert not svc.engine.active and not svc.operations and not svc.engine_requests
        assert not svc.prefixes
        await svc.close()
    run(case())


def test_branch_failure_drains_siblings_and_does_not_poison_valid_prefix():
    async def case():
        svc = service()
        def patch(output, kind):
            output.fuzev_forward_stats = worker_stats(output, features=kind == "encode")
        svc.engine.patch_output = patch
        response = await svc.generate(generate(), "generate")
        svc.engine.fail_call = 3
        with pytest.raises(RuntimeError, match="Synthetic"):
            await svc.decide(decision(messages=None, prefix_id=response["prefix_id"]), "fail")
        assert svc.engine.active == 0 and not svc.engine_requests and not svc.operations
        assert response["prefix_id"] in svc.prefixes
        assert svc.engine.aborted
        assert not hasattr(svc.head, "observed")
        await svc.close()
    run(case())


def test_cleanup_failure_closes_admission_without_claiming_gpu_release():
    async def case():
        svc = service()
        svc.engine.fail_call = 1
        async def failed_abort(_request_id):
            raise RuntimeError("abort unavailable")
        svc.engine.abort = failed_abort
        with pytest.raises(RuntimeError, match="abort unavailable"):
            await svc.generate(generate(), "bad")
        with pytest.raises(RuntimeError, match="admission is closed"):
            await svc.generate(generate(), "later")
        assert not svc.engine.active
        await svc.close()
    run(case())


def test_memory_guard_aborts_an_inflight_final_only_request():
    async def case():
        svc = service()
        svc.engine.wait = asyncio.Event()
        task = asyncio.create_task(svc.generate(generate(), "memory"))
        await svc.engine.entered.wait()
        svc.guard.error = MemoryError("reserved headroom")
        with pytest.raises(MemoryError, match="headroom"):
            await asyncio.wait_for(task, 3)
        assert svc.engine.aborted and not svc.engine.active
        await svc.close()
    run(case())


def test_request_deadline_aborts_and_cleans_up():
    async def case():
        svc = service(request_timeout_seconds=1)
        svc.engine.wait = asyncio.Event()
        with pytest.raises(TimeoutError):
            await svc.generate(generate(), "deadline")
        assert svc.engine.aborted and not svc.operations and not svc.engine_requests
        await svc.close()
    run(case())


def test_ttl_lru_token_pool_and_explicit_release():
    async def case():
        svc = service(prefix_capacity=2, prefix_ttl_seconds=10, retained_token_capacity=CONTEXT)
        now = [0.]
        svc.clock = lambda: now[0]
        first = svc._retain(tuple(range(CONTEXT - 2)), (0,))
        second = svc._retain((1,), (0,))
        assert first not in svc.prefixes and svc.retained_tokens == 2
        third = svc._retain((2,), (0,))
        svc._get_prefix(second)
        fourth = svc._retain((3,), (0,))
        assert third not in svc.prefixes
        assert svc.release_prefix(fourth) and not svc.release_prefix(fourth)
        now[0] = 10.
        svc.expire()
        assert not svc.prefixes and svc.retained_tokens == 0
        with pytest.raises(KeyError):
            await svc.decide(decision(messages=None, prefix_id=second), "expired")
        assert not svc.engine.calls
        await svc.close()
    run(case())


@pytest.mark.parametrize("invalid", ["prompt", "request_id", "eos", "feature_shape", "nan", "cache", "native_alias"])
def test_invalid_engine_results_fail_closed(invalid):
    async def case():
        svc = service()
        def patch(output, kind):
            if invalid == "prompt":
                output.prompt_token_ids = [999]
            elif invalid == "request_id":
                output.request_id = "different"
            elif invalid == "eos":
                output.outputs[0].token_ids = [700, 701]
            elif invalid == "native_alias":
                output.outputs[0].token_ids = [1_000_000_000, 0]
            elif invalid == "feature_shape":
                output.outputs.data = torch.zeros(1, 2560)
            elif invalid == "nan":
                output.outputs.data[0] = float("nan")
            elif invalid == "cache":
                output.num_cached_tokens = -1
        svc.engine.patch_output = patch
        with pytest.raises(ValueError):
            if invalid in {"feature_shape", "nan"}:
                await svc.decide(decision(), "invalid")
            else:
                await svc.generate(generate(), "invalid")
        # A final API response may already be cleaned by the engine; aborting
        # finished or malformed results is idempotent, not proof of GPU state.
        assert svc.engine.aborted and not svc.prefixes and not svc.operations
        await svc.close()
    run(case())


@pytest.mark.parametrize("forward_available", [False, True])
def test_unknown_cache_count_stays_unknown_not_zero_or_inferred(forward_available):
    async def case():
        svc = service()
        svc.engine.cached = None
        if forward_available:
            def patch(output, _kind):
                output.fuzev_forward_stats = worker_stats(output, features=True, positions=2)
            svc.engine.patch_output = patch
        result = await svc.decide(decision(), "uncached")
        assert result["execution"]["num_cached_tokens"] is None
        assert all(x["num_cached_tokens"] is None for x in result["execution"]["requests"])
        assert result["execution"]["forwarded_tokens"] == (6 if forward_available else None)
        await svc.close()
    run(case())


def test_text_stop_trimming_does_not_remove_any_api_accepted_token():
    async def case():
        svc = service()
        svc.engine.completion = [700, 701]
        def stop(output, kind):
            if kind == "generate":
                output.outputs[0].stop_reason = "STOP"
                output.outputs[0].text = "trimmed"
        svc.engine.patch_output = stop
        result = await svc.generate(generate(stop=["STOP"]), "stop")
        prefix = svc.prefixes[result["prefix_id"]][0]
        assert prefix.completion == (700, 701)
        assert result["choices"][0]["text"] == "trimmed"
        await svc.decide(decision(messages=None, prefix_id=prefix.id), "replay")
        assert svc.engine.calls[1][1][:len(prefix.accepted)] == prefix.accepted
        await svc.close()
    run(case())


def test_full_native_context_budget_is_not_artificially_truncated_to_2048():
    async def case():
        svc = service()
        result = await svc.generate(generate(prompt="a" * (CONTEXT - 8)), "full-budget")
        assert len(result["prompt_token_ids"]) == CONTEXT - 8
        assert svc.engine.calls[0][1] == tuple(result["prompt_token_ids"])
        await svc.close()
    run(case())


@pytest.mark.parametrize("kind", ["generation", "decision"])
def test_shared_image_field_cannot_be_silently_ignored_by_text_only_cell(kind):
    async def case():
        svc = service()
        request = generate() if kind == "generation" else decision()
        # A text-only attached client must not silently drop an image field.
        object.__setattr__(request, "image", {"path": "/tmp/not-opened", "sha256": "0" * 64})
        with pytest.raises(ValueError, match="Trusted image"):
            await (svc.generate(request, "image") if kind == "generation" else svc.decide(request, "image"))
        assert not svc.engine.calls
        await svc.close()
    run(case())


@pytest.mark.parametrize("invalid_request", [
    generate(prompt="<|image_pad|>"),
    generate(model="wrong"),
    generate(prompt="x" * CONTEXT),
    decision(head_id="wrong"),
    decision(messages=[{"role": "user", "content": [{"type": "image", "image": "/tmp/secret"}]}]),
    decision(messages=[{"role": "user", "content": "x", "image_path": "/tmp/secret"}]),
    decision(question="<|vision_start|>"),
    decision(additional_context="<|image|>"),
    decision(candidates=[{"id": "a", "text": "<|video_pad|>"}, {"id": "b", "text": "plain"}]),
    decision(question="x" * CONTEXT),
])
def test_unsupported_or_over_context_requests_rejected_before_engine(invalid_request):
    async def case():
        svc = service()
        with pytest.raises(ValueError):
            if isinstance(invalid_request, GenerationRequest):
                await svc.generate(invalid_request, "bad")
            else:
                await svc.decide(invalid_request, "bad")
        assert not svc.engine.calls
        await svc.close()
    run(case())


def test_service_close_cancels_requests_but_does_not_shutdown_injected_engine():
    async def case():
        svc = service()
        svc.engine.wait = asyncio.Event()
        task = asyncio.create_task(svc.generate(generate(), "active"))
        await svc.engine.entered.wait()
        await svc.close()
        assert task.cancelled() and svc.guard.closed
        assert not svc.operations and not svc.engine_requests and not svc.engine.active
        with pytest.raises(RuntimeError, match="live owning"):
            await svc.generate(generate(), "closed")
    run(case())


def test_builder_keeps_reference_and_actual_vllm_identity_separate():
    async def case():
        cfg, engine = config(), Engine()
        actual = {"runtime_cell": cfg.runtime_cell, "runtime_sources": cfg.runtime_sources}
        received = []
        def head_factory(config):
            received.append(config.feature_identity)
            return Head()
        svc = await build_service(engine, cfg, inspector=lambda _: actual,
            head_factory=head_factory, guard_factory=Guard, admission=lambda *_: None, api=API())
        assert svc.engine is engine
        assert received[0]["engine"] == "native-reference-only"
        assert svc.provenance["runtime_cell"]["engine"] == "synthetic-vllm-cell"
        assert svc.provenance["reference_head"]["feature_identity"] == cfg.feature_identity
        assert "unqualified" in svc.provenance["transfer_qualification"]
        cfg.runtime_cell["engine"] = "changed"
        assert svc.provenance["runtime_cell"]["engine"] == "synthetic-vllm-cell"
        await svc.close()
    run(case())


@pytest.mark.parametrize("changed_label", [False, True])
def test_builder_records_live_vision_metadata_in_receipt_json_form(changed_label):
    async def case():
        cfg, engine = config(), Engine()
        actual = {"runtime_cell": dict(cfg.runtime_cell), "runtime_sources": dict(cfg.runtime_sources)}
        actual["runtime_cell"]["multimodal_config"] = {"vision_config": {"id2label": {0: "LABEL_0"}}}
        cfg.runtime_cell = json.loads(json.dumps(actual["runtime_cell"]))
        if changed_label:
            actual["runtime_cell"]["multimodal_config"]["vision_config"]["id2label"][0] = "CHANGED"
        svc = await build_service(engine, cfg, inspector=lambda _: actual,
            head_factory=lambda _: Head(), guard_factory=Guard, admission=lambda *_: None, api=API())
        assert svc.engine is engine
        assert svc.provenance["runtime_cell"] == json.loads(json.dumps(actual["runtime_cell"]))
        assert svc.provenance["recorded_runtime"]["runtime_cell"] == cfg.runtime_cell
        await svc.close()
    run(case())


@pytest.mark.parametrize("error", ["tasks", "encoding"])
def test_builder_rejects_pin_or_capability_mismatch_before_loading_head(error):
    async def case():
        cfg, engine = config(), Engine()
        actual = {"runtime_cell": dict(cfg.runtime_cell), "runtime_sources": dict(cfg.runtime_sources)}
        if error == "sources":
            actual["runtime_sources"]["vllm-fuzev"] = "wrong"
        elif error == "cell":
            actual["runtime_cell"]["engine"] = "wrong"
        elif error == "tasks":
            engine.tasks = ("generate",)
        else:
            cfg.feature_identity["encoding"] = {}
        def no_head(_config):
            raise AssertionError("Head must not load before pins/capabilities pass")
        with pytest.raises(ValueError):
            await build_service(engine, cfg, inspector=lambda _: actual, head_factory=no_head,
                                guard_factory=Guard, admission=lambda *_: None, api=API())
    run(case())


def engine_config():
    return NS(model_config=NS(max_model_len=CONTEXT, runner_type="generate", hf_text_config=NS(hidden_size=2560)),
        cache_config=NS(gpu_memory_utilization=.85), scheduler_config=NS(max_num_seqs=4, async_scheduling=False),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        use_v2_model_runner=False, speculative_config=None, kv_transfer_config=None, ec_transfer_config=None,
        additional_config={"fuzev": {"enabled": True}})


@pytest.mark.parametrize("section,key,value", [
    ("model_config", "max_model_len", 2048), ("model_config", "runner_type", "pooling"),
    ("scheduler_config", "async_scheduling", True), ("parallel_config", "tensor_parallel_size", 2),
    (None, "use_v2_model_runner", True), (None, "kv_transfer_config", {}),
    (None, "ec_transfer_config", {}), (None, "speculative_config", {}),
    (None, "additional_config", {}),
])
def test_engine_policy_rejects_unsafe_configurations_before_launch(section, key, value):
    cfg = engine_config()
    validate_engine_config(cfg)
    setattr(getattr(cfg, section) if section else cfg, key, value)
    with pytest.raises(ValueError):
        validate_engine_config(cfg)


@pytest.mark.parametrize("forward_available", [False, True])
def test_http_routes_use_single_injected_engine_and_report_unsupported_images(forward_available):
    from fastapi.testclient import TestClient
    from fuzev_research.vllm_server import create_app
    engines, services = [], []
    @asynccontextmanager
    async def engine_context():
        engine = Engine()
        if forward_available:
            def patch(output, kind):
                output.fuzev_forward_stats = worker_stats(output, features=kind == "encode",
                    positions=2 if kind == "encode" else None)
            engine.patch_output = patch
        engines.append(engine)
        yield engine
    async def builder(engine, cfg):
        svc = service(engine)
        services.append(svc)
        return svc
    app = create_app(engine_context, config(), builder=builder)
    with TestClient(app) as client:
        response = client.post("/v1/completions", json=generate().model_dump())
        assert response.status_code == 200, response.text
        prefix_id = response.json()["prefix_id"]
        response = client.post("/v1/decisions", json=decision(messages=None, prefix_id=prefix_id).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.json()["execution"]["forwarded_tokens"] == (6 if forward_available else None)
        assert response.json()["execution"]["forward_counts_status"] == (
            "worker-acknowledged" if forward_available else "unavailable")
        assert not client.get("/health").json()["capabilities"]["images"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "unit-vllm"
        assert client.post("/v1/chat/completions", json=generate().model_dump()).status_code == 400
        assert client.post("/v1/completions", json=generate().model_dump(),
                           headers={"x-fuzev-request-id": "bad space"}).status_code == 400
        assert client.delete(f"/v1/prefixes/{prefix_id}").json()["released"]
        assert client.post("/v1/decisions", json=decision(messages=None, prefix_id=prefix_id).model_dump(mode="json")).status_code == 404
        assert client.post("/v1/requests/absent/cancel").status_code == 404
        assert client.post("/v1/completions", content=b"x" * ((8 << 20) + 1)).status_code == 413
    assert len(engines) == len(services) == 1
    assert services[0].closed and services[0].guard.closed


def test_http_cancel_waits_for_abort_submission_and_iterator_cleanup():
    from fastapi.testclient import TestClient
    from fuzev_research.vllm_server import create_app
    engines = []
    @asynccontextmanager
    async def engine_context():
        engine = Engine()
        engine.wait, engine.abort_wait = asyncio.Event(), asyncio.Event()
        engines.append(engine)
        yield engine
    async def builder(engine, cfg):
        return service(engine)
    app = create_app(engine_context, config(), builder=builder)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as callers:
        future = callers.submit(client.post, "/v1/completions", json=generate().model_dump(),
                                headers={"x-fuzev-request-id": "cancel-me"})
        engine = engines[0]
        client.portal.call(engine.entered.wait)
        try:
            assert client.post("/v1/requests/cancel-me/cancel").status_code == 202
            client.portal.call(engine.aborting.wait)
            assert not future.done()
            assert client.get("/health").json()["active_engine_requests"] == 1
        finally:
            client.portal.call(engine.abort_wait.set)
        assert future.result(timeout=3).status_code == 499
        assert not app.state.service.prefixes
        assert client.get("/health").json()["active_engine_requests"] == 0
