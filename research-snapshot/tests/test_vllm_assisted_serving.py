"""Future schema-17 compatibility only; fake engine/CPU tensors, not admission."""

import asyncio
from contextlib import asynccontextmanager

import pytest
import torch

from fuzev_research.contracts import Primitive
from fuzev_research.encoding import candidate_suffix, query_suffix, token_hash
from fuzev_research.prefix import identity_digest
from fuzev_research.vllm_receipts import aggregate_forward_receipts
from fuzev_research.vllm_server import create_app
from fuzev_research.vllm_service import VllmAPI, VllmService
from test_vllm_service import (API, Engine, Guard, Head, Tokenizer, config,
                               decision, generate, run, service)


class LabelTokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        if len(text) == 2 and text[0] == " " and "A" <= text[1] <= "P":
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            self.calls.append(text)
            return [200 + ord(text[1]) - ord("A")]
        return super().encode(text, **kwargs)


class AssistedHead(Head):
    requires_lm_scores = True

    def score(self, primitive, query, candidates, *, baseline):
        assert baseline.dtype == torch.float32 and baseline.device.type == "cpu"
        assert baseline.shape == (len(candidates),) and not baseline.requires_grad
        assert torch.isfinite(baseline).all()
        self.baseline = baseline.clone()
        return super().score(primitive, query, candidates)


def assisted_service():
    engine = Engine()
    engine.tokenizer = LabelTokenizer()
    return VllmService(engine, engine.tokenizer, AssistedHead(), config(),
                       {"cell": "synthetic"}, Guard(), api=API(), admission=lambda *_: None)


@pytest.mark.parametrize("primitive", list(Primitive))
def test_query_assistance_preserves_option_order_and_feature_only_branches(primitive):
    async def case():
        svc = assisted_service()
        candidates = ([{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]
            if primitive == Primitive.NOUL else [
                {"id": "c" + str(i), "text": "option " + str(i), "ordinal_value": i}
                for i in range(6)])
        request = decision(primitive=primitive, candidates=candidates)
        result = await svc.decide(request, "assisted")
        count = len(candidates)
        assert [score["id"] for score in result["scores"]] == [c["id"] for c in candidates]
        assert torch.equal(svc.head.baseline, torch.arange(200, 200 + count) / 4)
        assert len(svc.engine.calls) == count + 1
        assert all(call[0] == "encode" for call in svc.engine.calls)
        assert len(svc.engine.synthetic_projections) == 1
        query = svc.engine.calls[0][1]
        assert svc.engine.calls[0][2].extra_kwargs == {"baseline_token_ids": list(range(200, 200 + count))}
        for call, candidate in zip(svc.engine.calls[1:], candidates):
            assert call[1] == query + tuple(svc.tokenizer.encode(candidate_suffix(candidate),
                add_special_tokens=False, truncation=False))
            assert call[2].extra_kwargs is None
        receipts = result["execution"]["requests"]
        assert [r["fuzev_readout_receipt"]["lm_projection_calls"] for r in receipts] == [1] + [0] * count
        assert result["execution"]["forwarded_tokens"] is None
        assert result["execution"]["forward_counts_status"] == "unavailable"
        assert "unqualified" in result["qualification"]
        await svc.close()
    run(case())


def test_assisted_query_does_not_append_label_tokens_or_rerender_accepted_ab():
    async def case():
        svc = assisted_service()
        generated = await svc.generate(generate(), "generation")
        accepted = tuple(generated["prompt_token_ids"] + generated["completion_token_ids"])
        renders = len(svc.tokenizer.renders)
        request = decision(messages=None, prefix_id=generated["prefix_id"], additional_context="new C")
        result = await svc.decide(request, "accepted-ab")
        suffix = tuple(svc.tokenizer.encode(query_suffix(request.question,
            [c.model_dump() for c in request.candidates], request.additional_context),
            add_special_tokens=False, truncation=False))
        assert svc.engine.calls[1][1] == accepted + suffix
        assert len(svc.tokenizer.renders) == renders
        assert result["accepted_token_sha256"] == token_hash(accepted)
        assert result["query_token_sha256"] == token_hash(accepted + suffix)
        assert len(svc.engine.synthetic_projections) == 1
        assert [c[0] for c in svc.engine.calls] == ["generate", "encode", "encode", "encode"]
        await svc.close()
    run(case())


@pytest.mark.parametrize("count", [2, 16])
def test_feature_only_head_never_encodes_labels_or_requests_a_projection(count):
    async def case():
        svc = service()
        request = decision(candidates=[{"id": str(i), "text": "option " + str(i)} for i in range(count)])
        result = await svc.decide(request, "feature-only")
        assert all(params.extra_kwargs is None for _, _, params, _ in svc.engine.calls)
        assert not svc.engine.synthetic_projections
        assert not any(text in {" " + chr(65 + i) for i in range(16)} for text in svc.tokenizer.calls)
        assert all(r["fuzev_readout_receipt"]["baseline_token_ids"] == []
                   for r in result["execution"]["requests"])
        assert svc.readout_health()["required_mode"] == "feature-only"
        await svc.close()
    run(case())


@pytest.mark.parametrize("invalid", ["multiple", "duplicate", "outside", "bool", "negative"])
def test_bad_option_labels_fail_before_any_engine_call(invalid):
    async def case():
        svc = assisted_service()
        original = svc.tokenizer.encode
        def encode(text, **kwargs):
            if text in {" A", " B"}:
                return {"multiple": [200, 201], "duplicate": [200], "outside": [1024],
                        "bool": [True], "negative": [-1]}[invalid]
            return original(text, **kwargs)
        svc.tokenizer.encode = encode
        with pytest.raises(ValueError):
            await svc.decide(decision(), "bad-labels")
        assert not svc.engine.calls
        await svc.close()
    run(case())


@pytest.mark.parametrize("labels", [(1,), tuple(range(17)), (1, 1), (True, 2), [-1, 2]])
def test_readout_parser_requires_bounded_immutable_request_local_labels(labels):
    async def case():
        svc = assisted_service()
        async def empty():
            if False:
                yield
        with pytest.raises(ValueError, match="immutable 2..16"):
            await svc._read(empty(), (4, 7, 9), "bad-width", True, baseline_token_ids=labels)
        assert not svc.engine.calls
        await svc.close()
    run(case())


def test_headless_constructor_preserves_unbound_producer_identity():
    async def case():
        engine = Engine()
        provenance = {"producer": "synthetic", "scope": "no-head"}
        svc = VllmService(engine, engine.tokenizer, None, config(), provenance,
            Guard(), api=object(), admission=lambda *_: None)
        assert svc.provenance == provenance and svc.identity == identity_digest(provenance)
        assert svc.readout_sources is None
        await svc.close()
    run(case())


@pytest.mark.parametrize("labels", [None, (200, 201)])
@pytest.mark.parametrize("cached", [None, 1])
def test_headless_engine_calls_verify_query_and_candidates_without_changing_identity(labels, cached):
    class CountingAPI(API):
        def __init__(self):
            super().__init__()
            self.source_calls, self.sources = 0, API.readout_sources()
        def readout_sources(self):
            self.source_calls += 1
            return self.sources

    async def case():
        engine, api, cfg = Engine(), CountingAPI(), config()
        engine.cached, engine.mutate_after_yield = cached, True
        provenance = {"runtime_cell": cfg.runtime_cell, "runtime_sources": cfg.runtime_sources,
                      "reference_feature_identity": cfg.feature_identity, "scope": "synthetic-headless"}
        svc = VllmService(engine, engine.tokenizer, None, cfg, provenance,
            Guard(), api=api, admission=lambda *_: None)
        original_identity = svc.identity
        assert not api.source_calls and svc.readout_sources is None
        query, receipt = await svc._engine_call((4, 7, 9), api.pooling(labels),
            features=True, baseline_token_ids=labels)
        assert query.shape == (2560 + (0 if labels is None else len(labels)),)
        assert query.dtype == torch.float32 and query.device.type == "cpu"
        assert (query != 999).all() and torch.isfinite(query).all()
        assert api.source_calls == 1 and svc.readout_sources == API.readout_sources()
        api.sources.clear()
        candidates = await asyncio.gather(*[svc._engine_call((4, 7, 9, token),
            api.pooling(), features=True) for token in (11, 13)])
        receipts = [receipt] + [item[1] for item in candidates]
        assert api.source_calls == 1
        assert all(item[0].shape == (2560,) and (item[0] != 999).all() for item in candidates)
        assert all(r["fuzev_readout_receipt"]["worker_loaded_sources"] == API.readout_sources()
                   for r in receipts)
        assert [r["fuzev_readout_receipt"]["lm_projection_calls"] for r in receipts] == (
            [0 if labels is None else 1, 0, 0])
        assert [r["num_cached_tokens"] for r in receipts] == [cached] * 3
        totals = aggregate_forward_receipts(receipts)
        assert totals["forward_counts_status"] == "unavailable"
        assert totals["unavailable_forward_receipts"] == 3
        assert all(totals[key] is None for key in (
            "forwarded_tokens", "completed_token_positions", "completed_forward_steps",
            "missing_worker_receipts", "forwarded_tokens_lower_bound"))
        assert svc.readout_health()["required_mode"] == "feature-only"
        assert svc.readout_health()["status"] == "worker-verified"
        assert svc.provenance == provenance and svc.identity == original_identity == identity_digest(provenance)
        assert "worker_readout" not in svc.provenance
        assert svc.config.feature_identity == cfg.feature_identity
        assert all(call[0] == "encode" for call in engine.calls)
        assert len(engine.synthetic_projections) == (0 if labels is None else 1)
        assert not engine.aborted and not svc.engine_requests
        await svc.close()
    run(case())


def test_headless_pooling_only_api_binds_real_default_sources_without_publishing_them():
    from types import SimpleNamespace
    pytest.importorskip("vllm")
    sources = VllmAPI.readout_sources()
    async def case():
        engine = Engine()
        def patch(output, kind):
            assert kind == "encode"
            output.fuzev_readout_receipt["worker_loaded_sources"] = dict(sources)
        engine.patch_output = patch
        api = SimpleNamespace(pooling=API().pooling)
        provenance = {"producer": "synthetic-pooling-only"}
        svc = VllmService(engine, engine.tokenizer, None, config(), provenance,
            Guard(), api=api, admission=lambda *_: None)
        query, _ = await svc._engine_call((4, 7, 9), api.pooling((200, 201)),
            features=True, baseline_token_ids=(200, 201))
        candidate, _ = await svc._engine_call((4, 7, 9, 11), api.pooling(), features=True)
        assert query.shape == (2562,) and candidate.shape == (2560,)
        assert svc.readout_sources == sources
        assert svc.provenance == provenance and svc.identity == identity_digest(provenance)
        assert not engine.aborted
        await svc.close()
    run(case())


def test_frozen_headless_comparison_route_keeps_identity_and_never_invents_worker_counts(monkeypatch):
    from types import SimpleNamespace
    from fuzev_research.vllm_comparison_extract import ComparisonReadout
    def forbidden():
        raise AssertionError("Legacy producer pins do not bind the new receipt protocol")
    monkeypatch.setattr(VllmAPI, "readout_sources", staticmethod(forbidden))
    async def case():
        engine = Engine()
        engine.cached = None
        def patch(output, kind):
            assert kind == "encode"
            del output.fuzev_readout_receipt
        engine.patch_output = patch
        api = SimpleNamespace(pooling=API().pooling)
        provenance = {"runtime_sources": {"vllm-fuzev": "frozen-synthetic-parent"},
                      "scope": "synthetic-legacy-producer"}
        svc = ComparisonReadout(engine, engine.tokenizer, None, config(), provenance,
            Guard(), api=api, admission=lambda *_: None)
        query, query_receipt = await svc.readout((4, 7, 9), labels=(200, 201))
        candidate, candidate_receipt = await svc.readout((4, 7, 9, 11))
        assert query.shape == (2562,) and candidate.shape == (2560,)
        assert svc.readout_sources is None and not svc.verified_readouts
        assert svc.readout_health()["status"] == "awaiting-worker-receipt"
        assert "fuzev_readout_receipt" not in query_receipt
        assert query_receipt["num_cached_tokens"] is candidate_receipt["num_cached_tokens"] is None
        totals = aggregate_forward_receipts([query_receipt, candidate_receipt])
        assert totals["forward_counts_status"] == "unavailable" and totals["unavailable_forward_receipts"] == 2
        assert totals["forwarded_tokens"] is totals["completed_forward_steps"] is None
        assert svc.provenance == provenance and svc.identity == identity_digest(provenance)
        assert len(engine.synthetic_projections) == 1 and not engine.aborted
        await svc.close()
    run(case())


def test_real_pooling_api_uses_only_ordered_request_local_control_and_loaded_sources():
    pytest.importorskip("vllm")
    from vllm.v1.fuzev_readout import readout_sources
    labels = (7, 2)
    query, candidate = VllmAPI.pooling(labels), VllmAPI.pooling()
    assert query.task == candidate.task == "fuzev_features"
    assert query.extra_kwargs == {"baseline_token_ids": [7, 2]}
    assert candidate.extra_kwargs is None
    from types import SimpleNamespace
    model = SimpleNamespace(get_vocab_size=lambda: 32)
    query.verify(model)
    candidate.verify(model)
    assert query.skip_reading_prefix_cache is candidate.skip_reading_prefix_cache is False
    assert VllmAPI.readout_sources() == readout_sources()


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("assisted", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "schema_bool", "position", "width", "prompt",
    "request", "worker", "projection", "generated", "sources", "labels", "dtype", "payload_width", "nan"])
def test_strict_readout_parser_rejects_tampering_and_drains(headless, assisted, invalid):
    async def case():
        if headless:
            engine = Engine()
            svc = VllmService(engine, engine.tokenizer, None, config(), {"producer": "synthetic"},
                Guard(), api=API(), admission=lambda *_: None)
        else:
            svc = assisted_service() if assisted else service()
        original_identity, original_provenance = svc.identity, dict(svc.provenance)
        def patch(output, kind):
            if kind != "encode":
                return
            receipt = output.fuzev_readout_receipt
            if invalid == "missing":
                del output.fuzev_readout_receipt
            elif invalid == "schema_bool":
                receipt["schema_version"] = True
            elif invalid == "position":
                receipt["feature_readout_position"] -= 1
            elif invalid == "width":
                receipt["feature_width"] += 1
            elif invalid == "prompt":
                receipt["prompt_token_sha256"] = "0" * 64
            elif invalid == "request":
                receipt["request_id"] = "other"
            elif invalid == "worker":
                receipt["worker_request_id"] = ""
            elif invalid == "projection":
                receipt["lm_projection_calls"] = 0 if assisted else 1
            elif invalid == "generated":
                receipt["generated_token_count"] = 1
            elif invalid == "sources":
                receipt["worker_loaded_sources"][next(iter(receipt["worker_loaded_sources"]))] = "4" * 64
            elif invalid == "labels":
                receipt["baseline_token_ids"] = list(reversed(receipt["baseline_token_ids"])) if assisted else [200, 201]
            elif invalid == "dtype":
                output.outputs.data = output.outputs.data.half()
            elif invalid == "payload_width":
                output.outputs.data = torch.zeros(2560 if assisted else 2562)
            else:
                output.outputs.data[-1] = float("nan")
        svc.engine.patch_output = patch
        with pytest.raises(ValueError):
            if headless:
                labels = (200, 201) if assisted else None
                await svc._engine_call((4, 7, 9), svc.api.pooling(labels),
                    features=True, baseline_token_ids=labels)
            else:
                await svc.decide(decision(), "tampered")
        assert svc.engine.aborted and not svc.engine_requests and not svc.engine.active
        assert svc.readout_health()["status"] == "awaiting-worker-receipt"
        assert svc.identity == original_identity and svc.provenance == original_provenance
        await svc.close()
    run(case())


def test_request_local_widths_and_owned_controls_survive_concurrent_decisions():
    async def case():
        svc = assisted_service()
        svc.engine.mutate_after_yield = True
        results = await asyncio.gather(*[svc.decide(decision(candidates=[
            {"id": str(i), "text": "option " + str(i)} for i in range(count)]), "size-" + str(count))
            for count in (2, 16)])
        for count, result in zip((2, 16), results):
            receipt = result["execution"]["requests"][0]["fuzev_readout_receipt"]
            assert receipt["baseline_token_ids"] == list(range(200, 200 + count))
            assert receipt["worker_loaded_sources"] == API.readout_sources()
        assert len(svc.engine.synthetic_projections) == 2
        assert (svc.head.observed[0] != 999).all() and (svc.head.baseline != 999).all()
        await svc.close()
    run(case())


def test_assisted_images_keep_owned_payloads_and_candidate_cache_receipts(tmp_path):
    from test_vllm_images import acquired, image_decision, image_service
    async def case():
        svc = image_service(tmp_path)
        svc.head = AssistedHead()
        original = svc.tokenizer.encode
        svc.tokenizer.encode = lambda text, **kwargs: (
            [200 + ord(text[1]) - ord("A")] if text in {" A", " B"}
            else original(text, **kwargs))
        svc.engine.mutate_after_yield = True
        result = await svc.decide(image_decision(acquired(tmp_path)), "assisted-image")
        assert len(svc.engine.synthetic_projections) == 1
        assert len(svc.engine.renderer.calls) == 1
        assert all(prompt["type"] == "multimodal" for prompt in svc.engine.inputs)
        assert all((prompt["mm_kwargs"]["image"][0]["pixel_values"].data == 10).all()
                   for prompt in svc.engine.inputs)
        assert result["execution"]["num_cached_tokens"] == 3
        assert len(result["images"]) == 1
        assert all(len(r["images"]) == 1 for r in result["execution"]["requests"])
        assert torch.equal(svc.head.baseline, torch.tensor([50., 50.25]))
        await svc.close()
    run(case())


@pytest.mark.parametrize("phase", ["query", "branch"])
def test_assisted_cancellation_keeps_ownership_until_engine_drain(phase):
    from test_vllm_service import until
    async def case():
        svc = assisted_service()
        original = svc.engine._run
        async def stream(kind, prompt, params, request_id):
            labels = (params.extra_kwargs or {}).get("baseline_token_ids")
            if (phase == "query") == bool(labels):
                await asyncio.Event().wait()
            async for output in original(kind, prompt, params, request_id):
                yield output
        svc.engine.encode = lambda prompt, params, request_id: stream("encode", prompt, params, request_id)
        task = asyncio.create_task(svc.decide(decision(), "cancel-assisted"))
        await until(lambda: len(svc.engine_requests) >= (1 if phase == "query" else 2))
        assert svc.cancel("cancel-assisted")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert svc.engine.aborted and not svc.operations and not svc.engine_requests
        assert len(svc.engine.synthetic_projections) == (0 if phase == "query" else 1)
        await svc.close()
    run(case())


def test_integrated_fake_async_llm_lifespan_health_is_observation_not_admission():
    from fastapi.testclient import TestClient
    engines, services = [], []
    @asynccontextmanager
    async def context():
        engine = Engine()
        engine.tokenizer = LabelTokenizer()
        engines.append(engine)
        yield engine
    async def builder(engine, cfg):
        svc = VllmService(engine, engine.tokenizer, AssistedHead(), cfg,
            {"cell": "synthetic"}, Guard(), api=API(), admission=lambda *_: None)
        services.append(svc)
        return svc
    with TestClient(create_app(context, config(), builder=builder)) as client:
        initial = client.get("/health").json()
        assert initial["worker_readout"]["status"] == "awaiting-worker-receipt"
        assert initial["worker_readout"]["verified_receipts"] == {}
        response = client.post("/v1/decisions", json=decision().model_dump(mode="json"))
        assert response.status_code == 200, response.text
        health = client.get("/health").json()
        evidence = health["worker_readout"]
        assert evidence["status"] == "worker-verified"
        assert set(evidence["verified_receipts"]) == {"query-lm-assisted", "feature-only"}
        assert evidence["expected_loaded_sources"] == API.readout_sources()
        assert health["provenance"]["worker_readout"]["expected_loaded_sources"] == API.readout_sources()
        assert "not quality or activation" in evidence["scope"]
        assert "unqualified" in health["qualification"]
        assert len(engines) == len(services) == 1
        assert not engines[0].aborted and not services[0].api.generations
    assert services[0].closed and services[0].guard.closed
