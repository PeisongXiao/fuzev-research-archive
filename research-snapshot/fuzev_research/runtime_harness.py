"""Pinned HTTP lifecycle checks and honestly labelled end-to-end workloads.

No backbone, training data, or engine imports. This is not a substitute for
fresh/reused hybrid-state, full-window, or held-out quality qualification.
"""

import argparse
import asyncio
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .acquire import write_json
from .contracts import Candidate, Primitive
from .encoding import token_hash
from .service_contracts import (
    GenerationRequest,
    ServiceDecisionRequest,
    TextMessage,
    TrustedImage,
)


SHA = r"^[0-9a-f]{64}$"
PREFIX = r"^prefix_[0-9a-f]{32}$"
LIMITATIONS = [
    "HTTP receipts and numerical branch-interference checks, not complete hybrid-state proof",
    "No fresh/reused state equivalence, cancellation, full-window or image-blind quality gate",
    "Phase rates require valid server timestamps; CPU/engine intervals are not GPU-kernel-only",
    "Service/worker input-position counts are not kernels, FLOPs or causal coverage",
    "No held-out accuracy, transfer qualification, training or publication",
]


class DecisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    primitive: Primitive
    question: str
    candidates: list[Candidate]
    additional_context: str = ""
    image: TrustedImage | None = None

    @model_validator(mode="after")
    def check_contract(self):
        self.request("synthetic-contract", "reference", prefix_id="synthetic-prefix")
        return self

    def request(self, model, head_id, **context):
        return ServiceDecisionRequest(
            model=model, head_id=head_id, **self.model_dump(), **context
        ).model_dump(mode="json", exclude_none=True)


class Workload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    generation: GenerationRequest
    standalone_messages: list[TextMessage] = Field(min_length=1)
    decisions: list[DecisionSpec] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def complete_primitives(self):
        if {item.primitive for item in self.decisions} != set(Primitive):
            raise ValueError(
                "Each workload needs Choice, Noul and ordinal procedural cases"
            )
        for item in self.decisions:
            item.request(
                self.generation.model,
                "reference",
                messages=[message.model_dump() for message in self.standalone_messages],
            )
        return self


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    engine: str = Field(pattern=r"^(native|vllm)$")
    model: str = Field(min_length=1)
    head_id: str = "reference"
    head_manifest_sha256: str = Field(pattern=SHA)
    calibration_id: str = Field(min_length=1)
    prefix_identity: str = Field(pattern=SHA)
    health_pins: dict = Field(min_length=1)
    workloads: list[Workload] = Field(min_length=1, max_length=8)
    streams: list[Annotated[int, Field(strict=True)]] = Field(
        default_factory=lambda: [1, 2, 4], min_length=1
    )
    repetitions: int = Field(default=3, ge=1, le=16)
    timeout_seconds: float = Field(default=600, ge=1, le=3600)
    atol: float = Field(default=1e-5, gt=0, le=1e-4)
    rtol: float = Field(default=1e-4, gt=0, le=1e-3)

    @model_validator(mode="after")
    def pins_and_bounds(self):
        if len(set(self.streams)) != len(self.streams) or any(
            type(value) is not int or value < 1 for value in self.streams
        ):
            raise ValueError("Use distinct positive stream counts")
        if len({item.name for item in self.workloads}) != len(self.workloads):
            raise ValueError("Workload names must be unique")
        if any(item.generation.model != self.model for item in self.workloads):
            raise ValueError("Every workload must target the pinned model")
        source_key = "sources" if self.engine == "native" else "provenance"
        if (
            not isinstance(self.health_pins.get(source_key), dict)
            or not self.health_pins[source_key]
        ):
            raise ValueError(f"Explicit {source_key} health pins are required")
        return self


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int and value >= 0, f"Invalid {name}")
    return value


def generation_phases(response, engine):
    timing = response.get("phase_timing")
    if timing is None:
        return {"status": "unavailable", "reason": "server_phase_timing_missing"}
    require(isinstance(timing, dict), "Invalid generation phase timing")
    source = (
        "native-owner-cpu-observed"
        if engine == "native"
        else "vllm-engine-core-request-stats"
    )
    require(
        type(timing.get("schema_version")) is int
        and timing["schema_version"] == 1
        and timing.get("source") == source,
        "Unqualified generation timing source/schema",
    )
    usage = response["usage"]
    for field, expected in (
        ("prompt_tokens", usage["prompt_tokens"]),
        ("completion_tokens", usage["completion_tokens"]),
        ("decode_tokens", usage["completion_tokens"] - 1),
    ):
        require(
            integer(timing.get(field), field) == expected, "Phase token count mismatch"
        )
    cached = timing.get("reported_cached_prompt_tokens")
    if cached is not None:
        require(
            integer(cached, "cached prompt count") <= usage["prompt_tokens"],
            "Invalid cached prompt count",
        )
    preemptions = timing.get("num_preemptions")
    if preemptions is not None:
        integer(preemptions, "preemption count")
    require(
        isinstance(timing.get("scope"), str) and timing["scope"],
        "Missing phase observation scope",
    )
    status = timing.get("status")
    if status == "unavailable":
        require(
            isinstance(timing.get("reason"), str) and timing["reason"],
            "Missing unavailable phase reason",
        )
        require(
            all(
                timing.get(key) is None
                for key in ("prefill_seconds", "decode_seconds", "queue_seconds")
            ),
            "Unavailable phases contain nominal timing",
        )
        return timing
    require(
        status == "available" and timing.get("reason") is None,
        "Invalid phase timing availability",
    )
    for key in ("prefill_seconds", "decode_seconds", "queue_seconds"):
        value = timing.get(key)
        if key == "queue_seconds" and value is None:
            continue
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Invalid monotonic phase interval",
        )
    require(timing["prefill_seconds"] > 0, "Prefill phase has no measurable interval")
    require(
        (timing["decode_tokens"] == 0 and timing["decode_seconds"] == 0)
        or (timing["decode_tokens"] > 0 and timing["decode_seconds"] > 0),
        "Decode phase count/interval mismatch",
    )
    if engine == "native":
        require(
            cached == 0 and preemptions is None and timing.get("queue_seconds") is None,
            "Native timing invented engine cache/queue/preemption metrics",
        )
    else:
        require(
            preemptions == 0, "Preempted engine timing is not a clean phase observation"
        )
    return timing


def validate_generation(response, config, request):
    require(response["model"] == config.model, "Generation model changed")
    require(
        response["prefix_identity"] == config.prefix_identity,
        "Generation prefix identity changed",
    )
    require(
        bool(re.fullmatch(SHA, response["accepted_token_sha256"])),
        "Invalid accepted-token hash",
    )
    require(
        isinstance(response["prefix_id"], str)
        and re.fullmatch(PREFIX, response["prefix_id"]),
        "Invalid retained prefix",
    )
    tokens = response["completion_token_ids"]
    require(isinstance(tokens, list) and tokens, "Missing exact completion IDs")
    for token in tokens:
        integer(token, "completion token ID")
    usage = response["usage"]
    prompt = integer(usage["prompt_tokens"], "prompt length")
    completed = integer(usage["completion_tokens"], "completion length")
    total = integer(usage["total_tokens"], "total length")
    require(
        prompt > 0 and completed == len(tokens) <= request.max_tokens,
        "Generation token budget/count changed",
    )
    require(total == prompt + completed <= 262144, "Generation context/count changed")
    choices = response["choices"]
    require(
        len(choices) == 1
        and choices[0]["index"] == 0
        and choices[0]["finish_reason"] in {"length", "stop"},
        "Invalid generation finish",
    )
    if config.engine == "native":
        consumed = integer(response["consumed_tokens"], "native consumed boundary")
        tail = response["unconsumed_token_ids"]
        require(
            isinstance(tail, list) and len(tail) <= 1 and consumed + len(tail) == total,
            "Invalid accepted/consumed boundary",
        )
        require(
            not tail or tail == tokens[-len(tail) :], "Unconsumed token was changed"
        )
    else:
        prompt_ids = response["prompt_token_ids"]
        require(
            isinstance(prompt_ids, list) and len(prompt_ids) == prompt,
            "Exact vLLM prompt IDs missing",
        )
        for token in prompt_ids:
            integer(token, "prompt token ID")
        require(
            token_hash(prompt_ids + tokens) == response["accepted_token_sha256"],
            "Accepted AB hash mismatch",
        )
        require(
            response["consumed_tokens"] is None
            and response["unconsumed_token_ids"] is None,
            "Unqualified vLLM consumption was relabelled as known",
        )
    generation_phases(response, config.engine)


def validate_decision(response, config, spec, generation=None):
    require(
        response["model"] == config.model and response["head_id"] == config.head_id,
        "Decision model/head changed",
    )
    require(
        response["head_manifest_sha256"] == config.head_manifest_sha256
        and response["calibration_id"] == config.calibration_id,
        "Artifact/calibration pin changed",
    )
    require(
        response["prefix_identity"] == config.prefix_identity,
        "Decision prefix identity changed",
    )
    require(response["primitive"] == spec.primitive, "Decision primitive changed")
    for key in ("accepted_token_sha256", "query_token_sha256"):
        require(bool(re.fullmatch(SHA, response[key])), f"Missing {key}")
    if generation is not None:
        require(
            response["prefix_id"] == generation["prefix_id"]
            and response["accepted_token_sha256"]
            == generation["accepted_token_sha256"],
            "Decision did not preserve exact accepted AB",
        )
        require(
            response["layout"] == "accepted-ab-append-c",
            "Prefix was rerendered as standalone",
        )
        if config.engine == "native":
            require(
                response["resolved_unconsumed_tokens"]
                == len(generation["unconsumed_token_ids"]),
                "Decision did not resolve the accepted final-token boundary",
            )
    scores = response["scores"]
    require(
        [item["id"] for item in scores] == [item.id for item in spec.candidates],
        "Candidate order changed",
    )
    for field in ("logit", "raw_probability", "calibrated_probability"):
        values = [item[field] for item in scores]
        require(
            all(
                type(value) in {int, float} and math.isfinite(value) for value in values
            ),
            f"Nonfinite {field}",
        )
        if field != "logit":
            require(
                all(0 <= value <= 1 for value in values)
                and math.isclose(sum(values), 1, abs_tol=1e-6),
                f"Invalid {field} distribution",
            )
    require(
        response["selected_id"] == max(scores, key=lambda item: item["logit"])["id"],
        "Selected ID disagrees with logits",
    )


def compare_branches(before, after, config):
    for key in (
        "accepted_token_sha256",
        "query_token_sha256",
        "head_manifest_sha256",
        "calibration_id",
        "prefix_identity",
        "selected_id",
    ):
        require(before[key] == after[key], f"Intervening branch changed {key}")
    require(
        [item["id"] for item in before["scores"]]
        == [item["id"] for item in after["scores"]],
        "Intervening branch changed candidate membership/order",
    )
    maximum = {}
    for field in ("logit", "raw_probability", "calibrated_probability"):
        pairs = list(zip(before["scores"], after["scores"]))
        require(
            all(
                left["id"] == right["id"]
                and math.isclose(
                    left[field], right[field], rel_tol=config.rtol, abs_tol=config.atol
                )
                for left, right in pairs
            ),
            f"Intervening branch changed {field}",
        )
        maximum[field] = max(abs(left[field] - right[field]) for left, right in pairs)
    return maximum


def forwarding(execution, engine):
    value = execution.get("forwarded_tokens")
    if engine == "native":
        return (
            (integer(value, "native forwarded count"), "native-service-reported")
            if value is not None
            else (None, None)
        )
    status, lower = (
        execution.get("forward_counts_status"),
        execution.get("forwarded_tokens_lower_bound"),
    )
    if status not in {"worker-acknowledged", "lower-bound"}:
        require(
            value is None, "Unacknowledged vLLM physical count was presented as known"
        )
        return None, None
    require(
        type(lower) is bool and lower == (status == "lower-bound"),
        "Invalid lower-bound label",
    )
    stats = execution.get("fuzev_forward_stats")
    positions = (
        stats.get("completed_token_positions")
        if isinstance(stats, dict)
        else execution.get("completed_token_positions")
    )
    require(positions == value, "Worker position totals disagree")
    return integer(value, "worker forwarded count"), status


def summarize_phases(samples):
    timings = [
        item.get("generation_phase_timing")
        for item in samples
        if item.get("mode") != "decision-only"
    ]
    available = [item for item in timings if item and item["status"] == "available"]
    complete = bool(timings) and len(available) == len(timings)
    cached_known = complete and all(
        item["reported_cached_prompt_tokens"] is not None for item in available
    )
    prefill_tokens = (
        sum(
            item["prompt_tokens"] - item["reported_cached_prompt_tokens"]
            for item in available
        )
        if cached_known
        else None
    )
    prefill_seconds = (
        sum(item["prefill_seconds"] for item in available) if complete else None
    )
    decode_tokens = (
        sum(item["decode_tokens"] for item in available) if complete else None
    )
    decode_seconds = (
        sum(item["decode_seconds"] for item in available) if complete else None
    )
    return {
        "phase_generation_observations": len(timings),
        "phase_available_observations": len(available),
        "phase_timing_sources": sorted({item["source"] for item in available}),
        "phase_unavailable_reasons": sorted(
            {
                item.get("reason", "server_phase_timing_missing")
                if item
                else "server_phase_timing_missing"
                for item in timings
                if not item or item["status"] != "available"
            }
        ),
        "phase_rate_scope": "duration-weighted per-request generation service intervals; not concurrent wall throughput or GPU-kernel-only",
        "phase_prefill_token_scope": "logical prompt tokens minus server-reported cached prompt tokens; not physical forward counts",
        "phase_prefill_uncached_prompt_tokens": prefill_tokens,
        "phase_prefill_seconds_sum": prefill_seconds,
        "phase_decode_tokens": decode_tokens,
        "phase_decode_seconds_sum": decode_seconds,
        "prefill_tokens_per_second": prefill_tokens / prefill_seconds
        if prefill_tokens and prefill_seconds
        else None,
        "decode_only_tokens_per_second": decode_tokens / decode_seconds
        if decode_tokens and decode_seconds
        else None,
    }


def summarize_samples(samples, wall_seconds):
    elapsed = [item["elapsed_seconds"] for item in samples]
    positions = [item.get("forwarded_tokens") for item in samples]
    statuses = [item.get("forward_counts_status") for item in samples]
    known = all(value is not None for value in positions)
    total = sum(positions) if known else None
    completion = sum(item["completion_tokens"] for item in samples)
    return {
        "requests": len(samples),
        "wall_seconds_including_cleanup": wall_seconds,
        "latency_p50_seconds": statistics.median(elapsed),
        "latency_p95_seconds": sorted(elapsed)[
            max(0, math.ceil(0.95 * len(elapsed)) - 1)
        ],
        "completed_tokens": completion,
        "logical_prefix_tokens": sum(item["logical_prefix_tokens"] for item in samples),
        "logical_prefix_scope": "sum of generation prompts and decision AB lengths; excludes C and candidate tails",
        "completion_tokens_per_pipeline_second_including_prefill": completion
        / wall_seconds,
        "forwarded_tokens": total,
        "forwarded_tokens_per_pipeline_second": total / wall_seconds
        if total is not None
        else None,
        "forward_counts_lower_bound": any(value == "lower-bound" for value in statuses)
        if known
        else None,
        "forward_counter_sources": sorted(
            set(value for value in statuses if value is not None)
        ),
        **summarize_phases(samples),
    }


class Harness:
    def __init__(self, client, config, output):
        self.client, self.config, self.output = client, config, output
        self.prefixes, self.samples = set(), []
        self.records = (output / "requests.jsonl").open("x")

    async def call(self, method, path, body=None):
        request_id = "harness_" + uuid4().hex
        started = time.perf_counter()
        response = await self.client.request(
            method, path, json=body, headers={"X-Fuzev-Request-ID": request_id}
        )
        elapsed = time.perf_counter() - started
        payload = response.json()
        self.records.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "request": body,
                    "status": response.status_code,
                    "elapsed_seconds": elapsed,
                    "response": payload,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.records.flush()
        # Track owned prefixes even if later response validation fails.
        if (
            path.endswith("completions")
            and response.is_success
            and payload.get("prefix_id")
        ):
            require(
                isinstance(payload["prefix_id"], str)
                and re.fullmatch(PREFIX, payload["prefix_id"]),
                "Invalid prefix identifier; cannot safely address cleanup",
            )
            self.prefixes.add(payload["prefix_id"])
        return response, payload, elapsed

    async def json(self, method, path, body=None):
        response, payload, elapsed = await self.call(method, path, body)
        response.raise_for_status()
        return payload, elapsed

    async def preflight(self):
        health, _ = await self.json("GET", "/health")
        require(
            health.get("status") == "ok" and health.get("context_window") == 262144,
            "Unready/wrong-context endpoint",
        )
        capabilities = health.get(
            "service_capabilities", health.get("capabilities", {})
        )
        require(
            capabilities.get("text_generation") is True
            and capabilities.get("typed_decisions") is True,
            "Shared calibrated service is disabled; do not interfere with the extractor",
        )
        if any(
            workload.generation.image is not None
            or any(spec.image is not None for spec in workload.decisions)
            for workload in self.config.workloads
        ):
            require(
                capabilities.get("trusted_still_images") is True,
                "Trusted image preprocessing is unavailable",
            )
        for key, value in self.config.health_pins.items():
            require(health.get(key) == value, f"Health pin mismatch: {key}")
        models, _ = await self.json("GET", "/v1/models")
        require(
            [item["id"] for item in models["data"]] == [self.config.model],
            "Loaded catalogue changed",
        )
        return health

    async def generate(self, workload):
        body = workload.generation.model_dump(mode="json", exclude_none=True)
        path = (
            "/v1/chat/completions"
            if workload.generation.messages is not None
            else "/v1/completions"
        )
        response, elapsed = await self.json("POST", path, body)
        validate_generation(response, self.config, workload.generation)
        return response, elapsed

    async def decide(self, workload, spec, generation=None):
        context = (
            {"prefix_id": generation["prefix_id"]}
            if generation is not None
            else {
                "messages": [
                    message.model_dump() for message in workload.standalone_messages
                ]
            }
        )
        body = spec.request(self.config.model, self.config.head_id, **context)
        response, elapsed = await self.json("POST", "/v1/decisions", body)
        validate_decision(response, self.config, spec, generation)
        return response, elapsed

    async def release(self, prefix):
        response, _ = await self.json("DELETE", f"/v1/prefixes/{prefix}")
        require(
            response.get("id") == prefix and response.get("released") is True,
            "Owned prefix release failed",
        )
        self.prefixes.discard(prefix)

    async def correctness(self, workload):
        generation, _ = await self.generate(workload)
        first, _ = await self.decide(workload, workload.decisions[0], generation)
        for spec in workload.decisions[1:]:
            await self.decide(workload, spec, generation)
        repeated, _ = await self.decide(workload, workload.decisions[0], generation)
        comparison = compare_branches(first, repeated, self.config)
        for spec in workload.decisions:
            await self.decide(workload, spec)
        prefix = generation["prefix_id"]
        await self.release(prefix)
        released, _ = await self.json("DELETE", f"/v1/prefixes/{prefix}")
        require(released.get("released") is False, "Released prefix remained owned")
        body = workload.decisions[0].request(
            self.config.model, self.config.head_id, prefix_id=prefix
        )
        response, _, _ = await self.call("POST", "/v1/decisions", body)
        require(response.status_code == 404, "Released AB was silently reconstructed")
        return {
            "workload": workload.name,
            "branch_maximum_absolute_difference": comparison,
            "accepted_ab_preserved": True,
            "released_prefix_rejected": True,
        }

    async def pipeline(self, workload, mode, index):
        started = time.perf_counter()
        generation, decisions, cleanup = None, None, None
        try:
            if mode != "decision-only":
                generation, _ = await self.generate(workload)
                cleanup = generation["prefix_id"]
            if mode != "generation-only":
                spec = workload.decisions[index % len(workload.decisions)]
                decisions, _ = await self.decide(workload, spec, generation)
            elapsed = time.perf_counter() - started
            executions = (
                [generation["execution"]]
                if generation and "execution" in generation
                else []
            )
            if decisions:
                executions.append(decisions["execution"])
            forwards = [forwarding(item, self.config.engine) for item in executions]
            positions = [item[0] for item in forwards]
            expected = int(generation is not None) + int(decisions is not None)
            known = len(positions) == expected and all(
                value is not None for value in positions
            )
            lower = any(item[1] == "lower-bound" for item in forwards)
            status = (
                (
                    "native-service-reported"
                    if self.config.engine == "native"
                    else "lower-bound"
                    if lower
                    else "worker-acknowledged"
                )
                if known
                else None
            )
            sample = {
                "workload": workload.name,
                "mode": mode,
                "generation_phase_timing": generation_phases(
                    generation, self.config.engine
                )
                if generation
                else None,
                "elapsed_seconds": elapsed,
                "completion_tokens": generation["usage"]["completion_tokens"]
                if generation
                else 0,
                "logical_prefix_tokens": (
                    (generation["usage"]["prompt_tokens"] if generation else 0)
                    + (
                        decisions["execution"]["logical_prefix_tokens"]
                        if decisions
                        else 0
                    )
                ),
                "forwarded_tokens": sum(positions) if known else None,
                "forward_counts_status": status,
            }
            self.samples.append(sample)
            return sample
        finally:
            if cleanup is not None:
                await self.release(cleanup)

    async def performance(self, workload, mode, streams):
        samples, started = [], time.perf_counter()
        for repetition in range(self.config.repetitions):
            # One wave of `streams` external pipelines at a time.
            tasks = [
                asyncio.create_task(
                    self.pipeline(workload, mode, repetition * streams + index)
                )
                for index in range(streams)
            ]
            try:
                samples.extend(await asyncio.gather(*tasks))
            finally:
                await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "workload": workload.name,
            "mode": mode,
            "streams": streams,
            **summarize_samples(samples, time.perf_counter() - started),
        }

    async def close(self):
        failures = []
        for prefix in list(self.prefixes):
            try:
                await self.release(prefix)
            except Exception as error:
                failures.append({"prefix_id": prefix, "error": str(error)})
        self.records.close()
        return failures


async def run(config, output, *, transport=None, port=8000):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    report = {
        "schema_version": 1,
        "scope": "pinned-http-lifecycle-and-end-to-end-performance",
        "status": "in_progress",
        "runtime_qualified": False,
        "limitations": LIMITATIONS,
        "correctness": [],
        "performance": [],
    }
    write_json(output / "report.json", report)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        transport=transport,
        timeout=config.timeout_seconds,
        follow_redirects=False,
    ) as client:
        harness = Harness(client, config, output)
        try:
            report["health"] = await harness.preflight()
            for workload in config.workloads:
                report["correctness"].append(await harness.correctness(workload))
                write_json(output / "report.json", report)
                for mode in ("generation-only", "decision-only", "mixed"):
                    for streams in config.streams:
                        report["performance"].append(
                            await harness.performance(workload, mode, streams)
                        )
                        write_json(output / "report.json", report)
            report["status"] = "passed-scoped-checks"
        except BaseException as error:
            report.update(
                status="cancelled"
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            report["cleanup_failures"] = await harness.close()
            if report["cleanup_failures"]:
                report["status"] = "failed"
            write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output", type=Path, required=True, help="New private local run directory"
    )
    parser.add_argument(
        "--allow-inference",
        action="store_true",
        help="Only after extraction/fitting/calibration; permits requests on the configured port",
    )
    args = parser.parse_args()
    if not args.allow_inference:
        parser.error(
            "Explicit --allow-inference required; do not run against the live extractor"
        )
    config = HarnessConfig.model_validate_json(args.config.read_text())
    report = asyncio.run(run(config, args.output, port=args.port))
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["status"] == "passed-scoped-checks" else 1)


if __name__ == "__main__":
    main()
