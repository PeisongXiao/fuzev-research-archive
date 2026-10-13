"""Future exclusive EXL3 same-fork control, then fused real-HTTP measurements.

No engine imports on import; no pristine-upstream, kernel-time or quality claim.
The operator must wait for the current panel and explicitly authorize inference.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import time
from types import SimpleNamespace
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .acquire import write_json
from .encoding import TEMPLATE, token_hash
from .paths import ROOT
from .runtime_harness import Harness, Workload, require, summarize_samples
from .runtime_measurement import EvidencePin, MeasurementConfig
from .runtime_performance_counts import counter_delta


MODEL = "vllm-exl3-4.05"
CORE = "d6f1ee810d3b4b3c4dffed2e39b6d2246c089712"  # unpublished lab commit ID
ORIGINAL_CORE = "62fc998457f58c5a0e6801ab9c11c9fafbcb4b8d"  # unpublished lab commit ID
EXTENSION = "fuzev_research.runtime_performance_counts.PerformanceObservationWorkerExtension"
NATIVE_BINARY = "e1398c20e1b1c4a0c91c8661790cd2543156c3556af2d5963c75dfbe4fc6c05b"
MODES = ("generation-only", "decision-only", "mixed")
TIERS = (2048, 32768)
STREAMS = (1, 4)


class OperatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_role: Literal["operator-non-heldout"]
    recipe_commit: str | None = None
    engine_config: EvidencePin
    binding_receipt: EvidencePin
    image_build_receipt: EvidencePin | None = None
    workloads: EvidencePin
    correctness_evidence: EvidencePin
    seed: int = Field(default=0, strict=True, ge=0, le=2**63 - 1)
    padding_text: str | None = Field(default=None, min_length=1, max_length=256)
    timeout_seconds: float = Field(default=600., ge=1, le=3600)


@contextmanager
def _offline_profile():
    from . import kev_vllm_research_server as server
    old_core = server.CORE_COMMIT
    old_files = server.RECIPE_FILES
    # Record the core used by the historical performance stage.
    server.CORE_COMMIT = CORE
    server.RECIPE_FILES = tuple(dict.fromkeys((*old_files, "vllm_performance_operator.py",
        "runtime_performance_counts.py", "runtime_harness.py", "runtime_measurement.py",
        "vllm_generation_control.py", "kev_vllm_capture.py")))
    try:
        yield server
    finally:
        server.CORE_COMMIT, server.RECIPE_FILES = old_core, old_files


def _templates(config):
    items = config.workloads.read()
    require(isinstance(items, list) and len(items) == 3, "Use the three predeclared procedural templates")
    result = []
    for item in items:
        item = deepcopy(item)
        require(item.get("generation", {}).get("model") == "native-4.05", "Unexpected template model tag")
        item["generation"]["model"] = MODEL
        workload = Workload.model_validate(item)
        request = workload.generation
        require((request.temperature, request.top_p, request.top_k, request.max_tokens, request.seed)
                == (1., .95, 20, 128, config.seed) and not request.stop,
                "Preserve standard generation sampling and the predeclared fixed seed")
        require(request.image is None and all(spec.image is None for spec in workload.decisions),
                "This bounded performance stage is text-only; image/full-window qualification is separate")
        result.append(workload)
    require(len({item.name for item in result}) == 3, "Template names must be distinct")
    return result


def _render(tokenizer, *, prompt=None, messages=None, generation=True):
    from .telemetry import admit
    from .vllm_service import VllmService
    renderer = SimpleNamespace(tokenizer=tokenizer, template=TEMPLATE.read_text(), admission=admit)
    renderer._encode = partial(VllmService._encode, renderer)
    return (renderer._encode(prompt) if prompt is not None else
            VllmService._messages(renderer, [item.model_dump() for item in messages], generation=generation))


def _pad(messages, text):
    messages = [item.model_copy(deep=True) for item in messages]
    user = next((item for item in reversed(messages) if item.role == "user"), None)
    require(user is not None, "Procedural chat padding needs an existing user message")
    user.content += text
    return messages


def _padded(make, render, target, padding):
    value = make("")
    count = len(render(value))
    require(count <= target, "Template exceeds its predeclared tier; no truncation or selection")
    if count == target:
        return value
    require(padding is not None, "Declare padding_text or prepare exact-tier inputs")
    low, high = 0, 1
    while len(render(make(padding * high))) < target:
        low, high = high, high * 2
        require(high <= 65536, "Padding did not reach the declared tier within its bound")
    while high - low > 1:
        middle = (low + high) // 2
        if len(render(make(padding * middle))) < target:
            low = middle
        else:
            high = middle
    value = make(padding * high)
    require(target <= len(render(value)) < 262144 - 128, "Padded input violates the native window")
    return value


def prepare_workloads(tokenizer, templates, padding):
    workloads, receipts = [], []
    for template in templates:
        for tier in TIERS:
            def generation(text):
                request = template.generation.model_copy(deep=True)
                if request.prompt is not None:
                    request.prompt += text
                else:
                    request.messages = _pad(request.messages, text)
                return request

            def tokens(request):
                return _render(tokenizer, prompt=request.prompt, messages=request.messages)

            request = _padded(generation, tokens, tier, padding)
            standalone = _padded(lambda text: _pad(template.standalone_messages, text),
                lambda messages: _render(tokenizer, messages=messages, generation=False), tier, padding)
            workload = template.model_copy(deep=True, update={"name": f"{template.name}-{tier}",
                "generation": request, "standalone_messages": standalone})
            workload = Workload.model_validate(workload.model_dump())
            prompt = tokens(request)
            base = _render(tokenizer, messages=standalone, generation=False)
            workloads.append(workload)
            receipts.append({"workload": workload.name, "declared_tier_tokens": tier,
                "actual_generation_prompt_tokens": len(prompt), "generation_prompt_sha256": token_hash(prompt),
                "actual_standalone_base_tokens": len(base), "standalone_base_sha256": token_hash(base)})
    return workloads, receipts


def control_options(fused):
    require(fused.get("worker_extension_cls") == EXTENSION and fused.get("disable_log_stats", False) is False,
            "Use the named performance worker extension and real request statistics")
    require(fused.get("additional_config", {}).get("fuzev", {}).get("enabled") is True,
            "Pinned input must be the genuinely fused profile")
    result = deepcopy(fused)
    result["additional_config"]["fuzev"]["enabled"] = False
    return result


@asynccontextmanager
async def _engine(options, *, fused):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from .telemetry import admit
    from .vllm_ple_storage import validate_ple_storage_config, validate_ple_storage_options
    from .vllm_service import validate_engine_config
    config = AsyncEngineArgs(**options).create_engine_config()
    validate_engine_config(config, require_fusion=fused)
    validate_ple_storage_options(options, quantization=config.model_config.quantization)
    validate_ple_storage_config(config)
    admit(128 << 20)
    engine = AsyncLLM.from_vllm_config(config)
    try:
        yield engine
    finally:
        engine.shutdown(timeout=120.)


async def _counts(engine, *, enable=False):
    rows = await engine.collective_rpc("fuzev_performance_counts", timeout=60., kwargs={"enable": enable})
    require(isinstance(rows, list) and len(rows) == 1, "Require the actual single counter worker")
    snapshot = deepcopy(rows[0])
    counter_delta(snapshot, snapshot)
    return snapshot


async def capture_control(engine, operator, server):
    """Separate disabled-mode capture; never call the fused-only capture/validator."""
    from .vllm_generation_control import describe_control
    actual = describe_control(engine)
    pins = operator["research"]["source_pins"]
    observed_pins = server.verify_pins(pins, pins)
    await engine.check_health()
    tasks = await engine.get_supported_tasks()
    require("generate" in tasks and "fuzev_features" not in tasks, "Disabled control must not expose fused readouts")
    workers = await engine.collective_rpc("fuzev_runtime_observation", timeout=60.,
                                        kwargs={"release_unused_startup_cache": True})
    require(isinstance(workers, list) and len(workers) == 1, "Require one actual disabled-mode worker")
    worker = workers[0]
    require(worker["rank"] == 0 and worker["cuda"]["initialized"] is True,
            "Disabled worker must initialize CUDA")
    checkout = Path(operator["operator_modules"]["vllm-fuzev"]).parent.parent
    expected = {name: pins[str(checkout / name)] for name in server.READOUT_SOURCE_FILES}
    pair = server.read(ROOT / "cross-cell/exl3-native-reference-pair/pairing.json")
    native = worker["native_extension"]
    return {"control_kind": "fusion-disabled-same-fork", "runtime": actual, "workers": workers,
            "recorded_runtime_sources": operator["operator_runtime_sources"],
            "recorded_worker_binding": {"binary_pins": operator["research"]["binary_pins"],
                "torch_version": operator["preload"]["torch_version"],
                "cuda_version": operator["preload"]["cuda_version"],
                "modules": operator["operator_modules"], "readout_sources": expected,
                "native_pair": pair, "native_binary_sha256": NATIVE_BINARY},
            "observed_source_pins": observed_pins, "observed_native_extension": native,
            "head_loaded": False, "runtime_qualified": False,
            "scope": "disabled-mode worker allocation/import metadata, not readout, quality or upstream proof"}


def _idle(engine):
    count = engine.get_num_unfinished_requests()
    require(type(count) is int and count == 0,
            "Performance waves require an exclusive idle engine")


async def observed_wave(engine, operation, *, acknowledgement_timeout=5.):
    _idle(engine)
    require(await engine.reset_prefix_cache(reset_running_requests=False, reset_connector=False) is True,
            "Idle prefix-cache reset was not acknowledged")
    before = await _counts(engine)
    start = time.perf_counter()
    result = await operation()
    elapsed = time.perf_counter() - start
    _idle(engine)
    deadline = time.monotonic() + acknowledgement_timeout
    while True:
        after = await _counts(engine)
        delta = counter_delta(before, after)
        if delta["status"] == "completion-acknowledged" or time.monotonic() >= deadline:
            break
        await asyncio.sleep(.05)
    return {"elapsed_seconds": elapsed, "counter_before": before, "counter_after": after,
        "forward_counts": delta, "result": result,
        "cache_policy": "acknowledged idle prefix reset per wave; within-wave reuse permitted",
        "wall_scope": "request execution/cleanup; excludes reset and observation RPCs; not kernel time"}


async def _pipelines(harness, workload, mode, streams, wave):
    tasks = [asyncio.create_task(harness.pipeline(workload, mode, wave * streams + index))
             for index in range(streams)]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def _network(engine, service, sock, timeout):
    import uvicorn
    from starlette.responses import JSONResponse
    from .vllm_server import create_app
    port = sock.getsockname()[1]
    lifespan_errors = []

    @asynccontextmanager
    async def borrowed():
        try:
            yield engine
        except BaseException as error:
            lifespan_errors.append(error)
            raise

    async def builder(client, config):
        require(client is engine, "HTTP service must borrow the same captured engine")
        return service

    app = create_app(borrowed, service.config, builder=builder)
    owner = secrets.token_hex(32)

    @app.middleware("http")
    async def exclusive_client(request, call_next):
        if request.headers.get("x-fuzev-performance-owner") != owner:
            return JSONResponse({"detail": "Exclusive offline performance owner required"}, status_code=403)
        return await call_next(request)

    server = uvicorn.Server(uvicorn.Config(app,
        host="127.0.0.1", port=port, log_level="warning", access_log=False,
        timeout_graceful_shutdown=30, lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        deadline = time.monotonic() + 60.
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("Owned HTTP server stopped before startup")
            require(time.monotonic() < deadline, "Owned HTTP startup timed out")
            await asyncio.sleep(.05)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", follow_redirects=False,
                                    trust_env=False, timeout=timeout,
                                    headers={"X-Fuzev-Performance-Owner": owner}) as client:
            yield client
    finally:
        server.should_exit = True
        await task
        if lifespan_errors:
            raise RuntimeError("Owned HTTP lifespan failed") from lifespan_errors[0]


async def _fused(engine, operator, control, templates, config, output, sock, report):
    from .kev_vllm_capture import capture_and_build
    from .kev_vllm_research_head import PLAN_SHA
    from .vllm_service import describe_runtime
    actual = describe_runtime(engine)
    expected_cell = deepcopy(control["runtime_cell"])
    expected_cell["engine"] = "vllm-fuzev"
    expected_cell["additional_config"]["fuzev"]["enabled"] = True
    require(actual == {"runtime_sources": control["runtime_sources"], "runtime_cell": expected_cell},
            "Sequential control/fused geometry or sources differ beyond the fusion switch")
    service = await capture_and_build(engine, operator, output / "fused-runtime-capture.json")
    try:
        workloads, receipts = prepare_workloads(service.tokenizer, templates, config.padding_text)
        require(receipts == report["inputs"], "Control/fused rendered tokens differ")
        measurement = MeasurementConfig(engine="vllm", model=MODEL, head_manifest_sha256=PLAN_SHA,
            calibration_id="uncalibrated:T1", prefix_identity=service.identity,
            health_pins={"provenance": service.provenance}, workloads=workloads, streams=list(STREAMS),
            correctness_evidence=config.correctness_evidence,
            warmup_waves=1, repetitions=3, timeout_seconds=config.timeout_seconds)
        fused_out = output / "fused-http"
        fused_out.mkdir()
        write_json(fused_out / "config.json", measurement.model_dump(mode="json"))
        await _counts(engine, enable=True)
        async with _network(engine, service, sock, config.timeout_seconds) as client:
            harness = Harness(client, measurement, fused_out)
            try:
                health = await harness.preflight()
                _idle(engine)
                report["fused_health_before"] = health
                for workload in workloads:
                    for mode in MODES:
                        for streams in STREAMS:
                            cell = {"workload": workload.name, "mode": mode, "streams": streams, "waves": []}
                            report["fused"].append(cell)
                            for wave in range(4):
                                row = await observed_wave(engine, partial(_pipelines, harness, workload, mode, streams, wave))
                                row["warmup"] = wave == 0
                                cell["waves"].append(row)
                                write_json(output / "report.json", report)
                            measured = cell["waves"][1:]
                            samples = [{**sample, "forwarded_tokens": None, "forward_counts_status": None}
                                       for row in measured for sample in row["result"]]
                            cell["request_summary"] = summarize_samples(samples, sum(row["elapsed_seconds"] for row in measured))
                            cell["physical_forwarded_tokens"] = _physical_total(measured)
                final = await harness.preflight()
                require(final["provenance"] == health["provenance"] and
                    all(final.get(key) == health.get(key) == 0 for key in (
                        "active_engine_requests", "retained_prefixes", "retained_tokens", "retained_multimodal_bytes")),
                    "Final fused provenance or ownership ledger changed")
                report["fused_health_after"] = final
            finally:
                failures = await harness.close()
                report["cleanup_failures"] = failures
                require(not failures, "Owned HTTP prefix cleanup failed")
    finally:
        await service.close()


async def run_operator(config, output, *, allow_inference=False, config_path=None, port=8000):
    require(allow_inference is True, "Explicit future-stage inference permission required")
    require(os.environ.get("FUZEV_ROOT") == str(ROOT) and not os.environ.get("FUZEV_SYNTHETIC_STATE_TRACE"),
            "Use the explicit root and disable diagnostic state tracing")
    config = OperatorConfig.model_validate(config)
    config_pin = None
    if config_path is not None:
        path = Path(config_path).resolve()
        raw = path.read_bytes()
        require(OperatorConfig.model_validate_json(raw) == config, "Operator config changed after parsing")
        config_pin = (str(path), hashlib.sha256(raw).hexdigest())
    templates = _templates(config)
    evidence = config.correctness_evidence.read()
    require(evidence.get("model") == MODEL and evidence.get("status") in {
        "passed", "passed-scoped-checks", "completed", "complete", "failed"}, "Pin terminal correctness evidence for this model")
    for pin in (config.engine_config, config.binding_receipt, config.image_build_receipt):
        if pin is not None:
            pin.read()
    output = Path(output)
    require(output.parent.resolve().is_relative_to(ROOT / "reports"), "Use a new private output under ROOT/reports")
    output.mkdir(exist_ok=False)
    write_json(output / "operator-config.json", config.model_dump(mode="json"))
    report = {"schema_version": 1, "model": MODEL, "status": "in_progress", "runtime_qualified": False,
        "profile": {"recorded_core": CORE, "original_correctness_core": ORIGINAL_CORE,
                    "recorded_memory_fraction": .75},
        "existing_correctness_status": evidence["status"], "correctness_evidence": config.correctness_evidence.model_dump(mode="json"),
        "pristine_upstream_available": False, "control_kind": "fusion-disabled-same-fork",
        "control_wall_scope": "in-process AsyncLLM generation client, NOT network latency",
        "fused_wall_scope": "real loopback HTTP API plus pipeline cleanup, NOT ASGI-only or kernel time",
        "comparison_scope": "matched generation phase observations; no direct network wall speedup versus in-process control",
        "decision_mixed_control": "unavailable: fusion-disabled engine has no typed readout task",
        "cache_salts": "separate owner namespaces; reset each wave, identical within-wave prompt sharing policy",
        "full_window": "not run; existing native full-window plan is a separate stage",
        "control": [], "fused": [], "cleanup_failures": []}
    write_json(output / "report.json", report)
    try:
        with socket.socket() as sock, _offline_profile() as server:
            sock.bind(("127.0.0.1", port))  # Reserve the port before either GPU owner can be constructed.
            from .kev_vllm_capture import capture_operator
            from .vllm_generation_control import RenderedGeneration, run_control
            operator = capture_operator(MODEL, config.engine_config.path, config.binding_receipt.path,
                image_build_receipt=None if config.image_build_receipt is None else config.image_build_receipt.path)
            operator["recorded_recipe_commit"] = config.recipe_commit
            pair = server.read(ROOT / "cross-cell/exl3-native-reference-pair/pairing.json")
            report["observed_native_pair"] = pair
            report["recorded_native_binary_sha256"] = NATIVE_BINARY
            for pin in (config.engine_config, config.binding_receipt, config.workloads, config.correctness_evidence,
                        config.image_build_receipt):
                if pin is not None:
                    operator["research"]["source_pins"][str(pin.path)] = pin.sha256
            if config_pin is not None:
                operator["research"]["source_pins"][config_pin[0]] = config_pin[1]
            server.verify_pins(operator["research"]["source_pins"], operator["research"]["source_pins"])
            options = control_options(operator["options"])
            write_json(output / "control-options.json", options)
            write_json(output / "operator-capture.json", operator)
            async with _engine(options, fused=False) as engine:
                stock_capture = await capture_control(engine, operator, server)
                write_json(output / "control-runtime-capture.json", stock_capture)
                control = stock_capture["runtime"]
                report["control_runtime"] = control
                workloads, report["inputs"] = prepare_workloads(engine.get_tokenizer(), templates, config.padding_text)
                await _counts(engine, enable=True)
                for workload in workloads:
                    prompt = _render(engine.get_tokenizer(), prompt=workload.generation.prompt, messages=workload.generation.messages)
                    for streams in STREAMS:
                        cell = {"workload": workload.name, "streams": streams, "waves": []}
                        report["control"].append(cell)
                        for wave in range(4):
                            cases = [RenderedGeneration(f"stream-{index}", workload.generation, prompt) for index in range(streams)]
                            operation = partial(run_control, engine, cases, expected_runtime=control, model_name=MODEL,
                                cache_salt="fuzev-performance-control", allow_inference=True, reset_prefix_cache=False)
                            row = await observed_wave(engine, operation)
                            row["warmup"] = wave == 0
                            cell["waves"].append(row)
                            write_json(output / "report.json", report)
                        cell["physical_forwarded_tokens"] = _physical_total(cell["waves"][1:])
            report["control_shutdown_completed"] = True
            server.verify_pins(operator["research"]["source_pins"], operator["research"]["source_pins"])
            async with _engine(operator["options"], fused=True) as engine:
                await _fused(engine, operator, control, templates, config, output, sock, report)
            server.verify_pins(operator["research"]["source_pins"], operator["research"]["source_pins"])
        config.correctness_evidence.read()
        report["status"] = "measurement-complete"
    except BaseException as error:
        report.update(status="cancelled" if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) else "failed",
                      error=f"{type(error).__name__}: {error}")
    finally:
        write_json(output / "report.json", report)
    return report


def _physical_total(waves):
    values = [row["forward_counts"]["physical_forwarded_tokens"] for row in waves]
    return sum(values) if values and all(value is not None for value in values) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_inference:
        parser.error("Explicit inference permission required; wait for the current full panel")
    config = OperatorConfig.model_validate_json(args.config.read_bytes())
    report = asyncio.run(run_operator(config, args.output, allow_inference=True,
                                     config_path=args.config, port=args.port))
    print(json.dumps({"status": report["status"], "runtime_qualified": False}))
    return 0 if report["status"] == "measurement-complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
