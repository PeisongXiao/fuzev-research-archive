import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from fuzev_research.encoding import token_hash
from fuzev_research.vllm_receipts import forward_receipt
from fuzev_research.runtime_harness import (
    HarnessConfig,
    compare_branches,
    forwarding,
    generation_phases,
    main,
    run,
    summarize_samples,
    summarize_phases,
    validate_decision,
    validate_generation,
)


def configuration(engine="vllm", **overrides):
    decisions = [
        {
            "primitive": "choice",
            "question": "Which code is reported?",
            "candidates": [
                {"id": "green", "text": "green"},
                {"id": "red", "text": "red"},
            ],
        },
        {
            "primitive": "noul",
            "question": "Is the status green?",
            "additional_context": "New evidence: green.",
            "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
        },
        {
            "primitive": "ordinal",
            "question": "How many checks passed?",
            "candidates": [
                {"id": str(value), "text": str(value), "ordinal_value": value}
                for value in range(5)
            ],
        },
    ]
    pins = (
        {"sources": {"recipes": "synthetic"}}
        if engine == "native"
        else {"provenance": {"runtime_sources": {"recipes": "synthetic"}}}
    )
    return HarnessConfig.model_validate(
        {
            "engine": engine,
            "model": "synthetic-model",
            "head_manifest_sha256": "a" * 64,
            "calibration_id": "synthetic-calibration",
            "prefix_identity": "b" * 64,
            "health_pins": pins,
            "streams": [1, 2, 4],
            "repetitions": 1,
            "workloads": [
                {
                    "name": "procedural",
                    "generation": {
                        "model": "synthetic-model",
                        "prompt": "The status is green.",
                        "max_tokens": 2,
                    },
                    "standalone_messages": [
                        {"role": "user", "content": "The status is green."}
                    ],
                    "decisions": decisions,
                }
            ],
            **overrides,
        }
    )


class Endpoint:
    """CPU-only synthetic receipt endpoint; no inference server or datasets."""

    def __init__(
        self,
        config,
        *,
        disabled=False,
        interference=False,
        unavailable=False,
        phase_timing=False,
        port=8000,
    ):
        self.config, self.disabled, self.interference, self.unavailable = (
            config,
            disabled,
            interference,
            unavailable,
        )
        self.prefixes, self.calls = {}, []
        self.phase_timing = phase_timing
        self.port = port
        self.serial, self.decisions, self.active, self.maximum = 0, 0, 0, 0

    async def __call__(self, request):
        self.calls.append(request)
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        assert request.url.port == self.port and request.url.host == "127.0.0.1"
        if path == "/health":
            cap = {
                "text_generation": not self.disabled,
                "typed_decisions": not self.disabled,
            }
            fraction = (
                "memory_fraction"
                if self.config.engine == "native"
                else "memory_fraction_limit"
            )
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "context_window": 262144,
                    fraction: 0.85,
                    "capabilities": cap,
                    **self.config.health_pins,
                },
            )
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": self.config.model}]})
        if request.method == "DELETE":
            prefix = path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json={
                    "id": prefix,
                    "released": self.prefixes.pop(prefix, None) is not None,
                },
            )
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0)
            if path.endswith("completions"):
                self.serial += 1
                prefix = f"prefix_{self.serial:032x}"
                accepted = token_hash([7, 8, 9, 10])
                self.prefixes[prefix] = accepted
                result = {
                    "model": self.config.model,
                    "prefix_identity": self.config.prefix_identity,
                    "prefix_id": prefix,
                    "accepted_token_sha256": accepted,
                    "completion_token_ids": [9, 10],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 2,
                        "total_tokens": 4,
                    },
                    "choices": [
                        {"index": 0, "finish_reason": "length", "text": "Synthetic"}
                    ],
                    "consumed_tokens": 3,
                    "unconsumed_token_ids": [10],
                }
                if self.phase_timing:
                    result["phase_timing"] = synthetic_phase(self.config.engine)
                if self.config.engine == "vllm":
                    result.update(
                        prompt_token_ids=[7, 8],
                        consumed_tokens=None,
                        unconsumed_token_ids=None,
                        execution=self.generation_execution(),
                    )
                return httpx.Response(200, json=result)
            if body.get("prefix_id") and body["prefix_id"] not in self.prefixes:
                return httpx.Response(404, json={"detail": "Released prefix"})
            self.decisions += 1
            count = len(body["candidates"])
            values = [
                {
                    "id": item["id"],
                    "logit": 0.0,
                    "raw_probability": 1 / count,
                    "calibrated_probability": 1 / count,
                }
                for item in body["candidates"]
            ]
            if self.interference and self.decisions == 4:
                values[0]["logit"] = 0.1
            prefix = body.get("prefix_id")
            result = {
                "model": self.config.model,
                "head_id": self.config.head_id,
                "primitive": body["primitive"],
                "head_manifest_sha256": self.config.head_manifest_sha256,
                "calibration_id": self.config.calibration_id,
                "prefix_identity": self.config.prefix_identity,
                "prefix_id": prefix,
                "accepted_token_sha256": self.prefixes[prefix]
                if prefix
                else token_hash([7, 8]),
                "query_token_sha256": token_hash([7, 8, len(body["question"])]),
                "scores": values,
                "selected_id": values[0]["id"],
                "execution": self.execution(12),
                "layout": "accepted-ab-append-c"
                if prefix
                else "standalone-training-encoding",
                "resolved_unconsumed_tokens": 1 if prefix else 0,
            }
            return httpx.Response(200, json=result)
        finally:
            self.active -= 1

    def execution(self, count):
        if self.config.engine == "native":
            return {"logical_prefix_tokens": 4, "forwarded_tokens": count}
        return {
            "logical_prefix_tokens": 4,
            "forwarded_tokens": None if self.unavailable else count,
            "completed_token_positions": None if self.unavailable else count,
            "forward_counts_status": "unavailable"
            if self.unavailable
            else "worker-acknowledged",
            "forwarded_tokens_lower_bound": None if self.unavailable else False,
        }

    def generation_execution(self):
        stats = (
            None
            if self.unavailable
            else SimpleNamespace(
                completed_forward_steps=1,
                completed_token_positions=3,
                last_completed_start=0,
                last_completed_end=3,
                feature_readout_position=None,
                missing_worker_receipts=0,
                replayed_token_positions=None,
                reused_token_positions=None,
                causal_consumed_prefix=None,
            )
        )
        return forward_receipt(
            SimpleNamespace(fuzev_forward_stats=stats), 4, features=False
        )


def synthetic_phase(engine="vllm", **overrides):
    """Synthetic server interval schema, not live phase measurements."""
    return {
        "schema_version": 1,
        "source": "native-owner-cpu-observed"
        if engine == "native"
        else "vllm-engine-core-request-stats",
        "status": "available",
        "reason": None,
        "scope": "synthetic per-request interval",
        "prefill_seconds": 0.5,
        "decode_seconds": 0.1,
        "queue_seconds": None if engine == "native" else 0.2,
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "decode_tokens": 1,
        "reported_cached_prompt_tokens": 0,
        "num_preemptions": None if engine == "native" else 0,
        **overrides,
    }


@pytest.mark.parametrize("engine", ["native", "vllm"])
def test_scoped_transport_reports_duration_weighted_generation_phases(tmp_path, engine):
    config = configuration(engine)
    endpoint = Endpoint(config, phase_timing=True)
    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(endpoint))
    )
    assert (
        report["status"] == "passed-scoped-checks" and not report["runtime_qualified"]
    )
    for group in report["performance"]:
        if group["mode"] == "decision-only":
            assert group["phase_generation_observations"] == 0
            assert group["prefill_tokens_per_second"] is None
            assert group["decode_only_tokens_per_second"] is None
        else:
            assert group["phase_available_observations"] == group["streams"]
            assert group["prefill_tokens_per_second"] == pytest.approx(4.0)
            assert group["decode_only_tokens_per_second"] == pytest.approx(10.0)
            assert group["phase_decode_tokens"] == group["streams"]
            assert "not concurrent wall throughput" in group["phase_rate_scope"]
            assert "not physical forward counts" in group["phase_prefill_token_scope"]
    assert endpoint.maximum == 4 and endpoint.prefixes == {}


def test_phase_weighting_excludes_first_token_and_cache_hits():
    samples = [
        {
            "mode": "mixed",
            "generation_phase_timing": synthetic_phase(
                prompt_tokens=100,
                reported_cached_prompt_tokens=20,
                completion_tokens=10,
                decode_tokens=9,
                prefill_seconds=2.0,
                decode_seconds=3.0,
            ),
        },
        {
            "mode": "mixed",
            "generation_phase_timing": synthetic_phase(
                prompt_tokens=100,
                completion_tokens=4,
                decode_tokens=3,
                prefill_seconds=1.0,
                decode_seconds=1.0,
            ),
        },
        {"mode": "decision-only", "generation_phase_timing": None},
    ]
    result = summarize_phases(samples)
    assert result["phase_prefill_uncached_prompt_tokens"] == 180
    assert result["prefill_tokens_per_second"] == 60
    assert result["phase_decode_tokens"] == 12
    assert result["decode_only_tokens_per_second"] == 3
    # Duration weighting is not an average of individual per-request rates.
    assert result["prefill_tokens_per_second"] != (40 + 100) / 2


def test_missing_generation_phase_prevents_partial_overclaim():
    samples = [
        {"mode": "generation-only", "generation_phase_timing": synthetic_phase()},
        {"mode": "mixed", "generation_phase_timing": None},
    ]
    result = summarize_phases(samples)
    assert result["phase_generation_observations"] == 2
    assert result["phase_available_observations"] == 1
    assert result["prefill_tokens_per_second"] is None
    assert result["decode_only_tokens_per_second"] is None
    assert result["phase_decode_tokens"] is None
    assert result["phase_unavailable_reasons"] == ["server_phase_timing_missing"]


def test_unknown_cache_and_single_token_do_not_invent_phase_work():
    result = summarize_phases(
        [
            {
                "mode": "generation-only",
                "generation_phase_timing": synthetic_phase(
                    reported_cached_prompt_tokens=None
                ),
            }
        ]
    )
    assert result["prefill_tokens_per_second"] is None
    assert result["decode_only_tokens_per_second"] == 10
    result = summarize_phases(
        [
            {
                "mode": "generation-only",
                "generation_phase_timing": synthetic_phase(
                    completion_tokens=1, decode_tokens=0, decode_seconds=0
                ),
            }
        ]
    )
    assert result["phase_decode_tokens"] == 0
    assert result["decode_only_tokens_per_second"] is None
    result = summarize_phases(
        [
            {
                "mode": "generation-only",
                "generation_phase_timing": synthetic_phase(
                    reported_cached_prompt_tokens=2
                ),
            }
        ]
    )
    assert result["phase_prefill_uncached_prompt_tokens"] == 0
    assert result["prefill_tokens_per_second"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": True},
        {"source": "wall-clock-guess"},
        {"prompt_tokens": 3},
        {"completion_tokens": 1},
        {"decode_tokens": 2},
        {"reported_cached_prompt_tokens": 3},
        {"reported_cached_prompt_tokens": True},
        {"prefill_seconds": float("nan")},
        {"prefill_seconds": 0},
        {"decode_seconds": float("inf")},
        {"decode_seconds": -1},
        {"decode_seconds": 0},
        {"queue_seconds": -1},
        {"scope": ""},
        {"num_preemptions": 1},
        {"num_preemptions": True},
        {"reason": "estimated"},
        {"status": "unavailable", "reason": "missing"},
    ],
)
def test_malformed_or_unqualified_phase_receipts_rejected(overrides):
    response = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        "phase_timing": synthetic_phase(**overrides),
    }
    with pytest.raises(ValueError):
        generation_phases(response, "vllm")


def test_explicit_unavailable_phase_stays_unknown():
    timing = synthetic_phase(
        status="unavailable",
        reason="missing_engine_stats",
        prefill_seconds=None,
        decode_seconds=None,
        queue_seconds=None,
        num_preemptions=None,
        reported_cached_prompt_tokens=None,
    )
    response = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        "phase_timing": timing,
    }
    assert generation_phases(response, "vllm") == timing
    result = summarize_phases([{"mode": "mixed", "generation_phase_timing": timing}])
    assert result["phase_unavailable_reasons"] == ["missing_engine_stats"]
    assert result["phase_decode_seconds_sum"] is None


@pytest.mark.parametrize("engine", ["native", "vllm"])
@pytest.mark.parametrize("port", [8000, 8765])
def test_entire_scoped_harness_on_cpu_transport(tmp_path, engine, port):
    config = configuration(engine)
    endpoint = Endpoint(config, port=port)
    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(endpoint), port=port)
    )
    assert report["status"] == "passed-scoped-checks"
    assert report["runtime_qualified"] is False
    assert len(report["correctness"]) == 1 and len(report["performance"]) == 9
    assert report["cleanup_failures"] == [] and endpoint.prefixes == {}
    assert endpoint.maximum == 4
    assert report["correctness"][0]["accepted_ab_preserved"] is True
    assert report["correctness"][0]["branch_maximum_absolute_difference"]["logit"] == 0
    for group in report["performance"]:
        assert group["prefill_tokens_per_second"] is None
        assert group["decode_only_tokens_per_second"] is None
        assert group["requests"] == group["streams"]
        if engine == "native" and group["mode"] != "decision-only":
            assert group["forwarded_tokens"] is None
        else:
            assert group["forwarded_tokens"] is not None
    records = [
        json.loads(line)
        for line in (tmp_path / "run/requests.jsonl").read_text().splitlines()
    ]
    assert all(record["request_id"].startswith("harness_") for record in records)
    assert any(record["status"] == 404 for record in records)
    assert json.loads((tmp_path / "run/report.json").read_text()) == report


def test_disabled_extractor_gets_no_inference_requests(tmp_path):
    config = configuration()
    endpoint = Endpoint(config, disabled=True)
    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(endpoint))
    )
    assert report["status"] == "failed" and "extractor" in report["error"]
    assert len(endpoint.calls) == 1 and endpoint.calls[0].method == "GET"


def test_branch_failure_is_visible_and_releases_owned_prefix(tmp_path):
    config = configuration()
    endpoint = Endpoint(config, interference=True)
    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(endpoint))
    )
    assert report["status"] == "failed" and "logit" in report["error"]
    assert endpoint.prefixes == {} and report["cleanup_failures"] == []
    assert report["performance"] == []


def test_unavailable_counts_never_become_zero_or_logical_counts(tmp_path):
    config = configuration()
    endpoint = Endpoint(config, unavailable=True)
    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(endpoint))
    )
    assert report["status"] == "passed-scoped-checks"
    assert all(item["forwarded_tokens"] is None for item in report["performance"])


@pytest.mark.parametrize("streams", [[True], [1, 1], ["4"]])
def test_invalid_external_concurrency_rejected(streams):
    with pytest.raises(ValueError):
        configuration(streams=streams)


def test_changed_model_and_missing_source_pins_rejected():
    with pytest.raises(ValueError, match="pinned model"):
        configuration(model="different")
    with pytest.raises(ValueError, match="provenance"):
        configuration(health_pins={"context_window": 262144})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(accepted_token_sha256="c" * 64),
        lambda value: value["usage"].update(completion_tokens=True),
        lambda value: value.update(consumed_tokens=4),
    ],
)
def test_invalid_generation_receipts_rejected(mutate):
    config = configuration()
    endpoint = Endpoint(config)
    response = asyncio.run(
        endpoint(
            httpx.Request("POST", "http://127.0.0.1:8000/v1/completions", json={})
        )
    ).json()
    mutate(response)
    with pytest.raises(ValueError):
        validate_generation(response, config, config.workloads[0].generation)


def test_changed_head_candidate_order_and_nonfinite_scores_rejected():
    config = configuration()
    spec = config.workloads[0].decisions[0]
    endpoint = Endpoint(config)
    body = spec.request(
        config.model,
        config.head_id,
        messages=[{"role": "user", "content": "synthetic"}],
    )
    original = asyncio.run(
        endpoint(
            httpx.Request("POST", "http://127.0.0.1:8000/v1/decisions", json=body)
        )
    ).json()
    for key, value in [
        ("head_manifest_sha256", "c" * 64),
        ("scores", original["scores"][::-1]),
    ]:
        broken = {**original, key: value}
        with pytest.raises(ValueError):
            validate_decision(broken, config, spec)
    broken = deepcopy(original)
    broken["scores"][0]["logit"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_decision(broken, config, spec)
    broken = deepcopy(original)
    broken["scores"].pop()
    with pytest.raises(ValueError, match="membership"):
        compare_branches(original, broken, config)


def test_lower_bound_and_missing_receipt_aggregation():
    assert forwarding(
        {
            "forwarded_tokens": 5,
            "completed_token_positions": 5,
            "forward_counts_status": "lower-bound",
            "forwarded_tokens_lower_bound": True,
        },
        "vllm",
    ) == (5, "lower-bound")
    with pytest.raises(ValueError, match="Unacknowledged"):
        forwarding({"forwarded_tokens": 5}, "vllm")
    with pytest.raises(ValueError, match="lower-bound"):
        forwarding(
            {
                "forwarded_tokens": 5,
                "forward_counts_status": "lower-bound",
                "forwarded_tokens_lower_bound": False,
            },
            "vllm",
        )
    samples = [
        {
            "elapsed_seconds": 0.5,
            "completion_tokens": 2,
            "logical_prefix_tokens": 7,
            "forwarded_tokens": 5,
            "forward_counts_status": "lower-bound",
        }
    ]
    assert summarize_samples(samples, 1)["forward_counts_lower_bound"] is True
    samples.append(
        {**samples[0], "forwarded_tokens": None, "forward_counts_status": None}
    )
    summary = summarize_samples(samples, 1)
    assert (
        summary["forwarded_tokens"] is None
        and summary["forward_counts_lower_bound"] is None
    )
    assert summary["completed_tokens"] == 4 and summary["logical_prefix_tokens"] == 14


def test_existing_output_directory_refused(tmp_path):
    with pytest.raises(FileExistsError):
        asyncio.run(
            run(
                configuration(),
                tmp_path,
                transport=httpx.MockTransport(Endpoint(configuration())),
            )
        )


def test_cli_requires_explicit_inference_permission_before_reading_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "runtime-harness",
            "--config",
            str(tmp_path / "absent.json"),
            "--output",
            str(tmp_path / "not-created"),
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2 and not (tmp_path / "not-created").exists()


def test_image_capability_is_required_before_inference(tmp_path):
    raw = configuration().model_dump(mode="json")
    raw["workloads"][0]["generation"]["image"] = {
        "path": "/synthetic/image.png",
        "sha256": "c" * 64,
    }
    config = HarnessConfig.model_validate(raw)
    endpoint = Endpoint(config)
    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(endpoint))
    )
    assert report["status"] == "failed" and "preprocessing" in report["error"]
    assert all(call.method == "GET" for call in endpoint.calls)


def test_cleanup_failure_is_not_scoped_success(tmp_path):
    config = configuration()
    endpoint = Endpoint(config)

    async def transport(request):
        if request.method == "DELETE":
            return httpx.Response(500, json={"detail": "synthetic cleanup failure"})
        return await endpoint(request)

    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(transport))
    )
    assert report["status"] == "failed" and len(report["cleanup_failures"]) == 1
    assert report["runtime_qualified"] is False


def test_health_source_change_stops_before_inference(tmp_path):
    config = configuration()
    endpoint = Endpoint(config)

    async def transport(request):
        response = await endpoint(request)
        if request.url.path == "/health":
            value = response.json()
            value["provenance"] = {"runtime_sources": {"recipes": "changed"}}
            return httpx.Response(200, json=value)
        return response

    report = asyncio.run(
        run(config, tmp_path / "run", transport=httpx.MockTransport(transport))
    )
    assert report["status"] == "failed" and "pin mismatch" in report["error"]
    assert all(call.method == "GET" for call in endpoint.calls)
