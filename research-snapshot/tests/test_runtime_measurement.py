"""CPU lifecycle fixtures; no engine, model, dataset or real API requests."""

import asyncio
import hashlib
import json

import httpx
import pytest

from fuzev_research import runtime_measurement as module


def config(tmp_path, *, fraction=.75, status="failed"):
    evidence = tmp_path / "correctness.json"
    evidence.write_text(json.dumps({"status": status, "model": "fixture"}))
    candidates = [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
    decisions = [{"primitive": name, "question": "Status?", "candidates": candidates}
                 for name in ("choice", "noul")]
    decisions.append({"primitive": "ordinal", "question": "Level?", "candidates": [
        {"id": str(i), "text": str(i), "ordinal_value": i} for i in range(3)]})
    return module.MeasurementConfig.model_validate({
        "engine": "vllm", "model": "fixture", "head_manifest_sha256": "a" * 64,
        "calibration_id": "uncalibrated:T1", "prefix_identity": "b" * 64,
        "health_pins": {"provenance": {"runtime_sources": {"recipes": "fixture"}}},
        "streams": [1, 4], "expected_engine_memory_fraction": fraction,
        "correctness_evidence": {"path": str(evidence),
                                 "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()},
        "workloads": [{"name": "fixture", "generation": {"model": "fixture", "prompt": "Status"},
                       "standalone_messages": [{"role": "user", "content": "Status"}],
                       "decisions": decisions}],
    })


class FakeHarness:
    events = []
    busy = False
    cleanup_failure = False

    def __init__(self, client, config, output):
        self.client = client
        self.config = config

    async def preflight(self):
        return {"memory_fraction_limit": .85, "active_engine_requests": int(self.busy),
                "provenance": {"runtime_cell": {"memory_fraction": .75}}}

    async def pipeline(self, workload, mode, index):
        self.events.append(("warmup", mode, index))
        return {"elapsed_seconds": 1}

    async def performance(self, workload, mode, streams):
        self.events.append(("measured", mode, streams))
        return {"streams": streams, "requests": self.config.repetitions * streams}

    async def close(self):
        self.events.append(("closed",))
        return [{"error": "fixture"}] if self.cleanup_failure else []


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    FakeHarness.events, FakeHarness.busy, FakeHarness.cleanup_failure = [], False, False
    monkeypatch.setattr(module, "Harness", FakeHarness)


@pytest.mark.parametrize("port", [None, 8765])
def test_measurement_uses_configured_port(tmp_path, monkeypatch, port):
    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(200, json={"active_engine_requests": 0,
            "provenance": {"runtime_cell": {"memory_fraction": .75}}})

    async def preflight(self):
        return (await self.client.get("/health")).json()

    monkeypatch.setattr(FakeHarness, "preflight", preflight)
    result = asyncio.run(module.run_measurement(config(tmp_path), tmp_path / "out",
        transport=httpx.MockTransport(endpoint), **({} if port is None else {"port": port})))
    assert result["status"] == "measurement-complete"
    assert requests
    assert {str(request.url) for request in requests} == {
        f"http://127.0.0.1:{8000 if port is None else port}/health"}


def test_failed_correctness_remains_failed_without_becoming_measurement_qualification(tmp_path):
    result = asyncio.run(module.run_measurement(config(tmp_path), tmp_path / "out"))
    assert result["status"] == "measurement-complete"
    assert result["runtime_qualified"] is False
    assert result["existing_correctness_status"] == "failed"
    assert len(result["performance"]) == 6
    assert len(result["warmup"]) == 6
    assert sum(x["requests"] for x in result["performance"]) == 45
    assert FakeHarness.events[0][0] == "warmup"
    assert FakeHarness.events[-1] == ("closed",)


def test_allocator_ceiling_does_not_substitute_for_engine_allocation_fraction(tmp_path):
    result = asyncio.run(module.run_measurement(config(tmp_path, fraction=.85), tmp_path / "out"))
    assert result["status"] == "failed" and result["performance"] == []
    assert "allocator ceiling" in result["error"]


def test_busy_engine_refused_without_requests(tmp_path):
    FakeHarness.busy = True
    result = asyncio.run(module.run_measurement(config(tmp_path), tmp_path / "out"))
    assert result["status"] == "failed"
    assert FakeHarness.events == [("closed",)]


def test_changed_or_unfinished_correctness_evidence_refused_before_output(tmp_path):
    c = config(tmp_path, status="in_progress")
    with pytest.raises(ValueError, match="terminal correctness"):
        asyncio.run(module.run_measurement(c, tmp_path / "out"))
    c.correctness_evidence.path.write_text('{"status":"passed"}')
    with pytest.raises(ValueError, match="evidence changed"):
        asyncio.run(module.run_measurement(c, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_evidence_from_another_model_cannot_be_used(tmp_path):
    c = config(tmp_path).model_copy(update={"model": "other"})
    with pytest.raises(ValueError, match="another model"):
        asyncio.run(module.run_measurement(c, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_cleanup_failure_is_retained(tmp_path):
    FakeHarness.cleanup_failure = True
    result = asyncio.run(module.run_measurement(config(tmp_path), tmp_path / "out"))
    assert result["status"] == "failed" and result["cleanup_failures"]


def test_native_fraction_uses_actual_native_policy_not_vllm_allocator_limit():
    assert module.allocation_fraction({"memory_fraction": .85}, "native") == .85
    assert module.allocation_fraction({"memory_fraction_limit": .85}, "vllm") is None
