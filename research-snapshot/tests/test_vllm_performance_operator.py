"""Synthetic CPU orchestration only: no GPU, models, datasets or HTTP traffic."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from fuzev_research import kev_vllm_capture as capture
from fuzev_research import vllm_generation_control as control
from fuzev_research import vllm_performance_operator as operator
from test_runtime_performance_counts import snapshot


@pytest.fixture(autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized()


def pin(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def templates():
    decisions = [{"primitive": name, "question": "Select the matching option.", "candidates": [
        {"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]} for name in ("choice", "noul")]
    decisions.append({"primitive": "ordinal", "question": "Rate the option.", "candidates": [
        {"id": str(index), "text": str(index), "ordinal_value": index} for index in range(3)]})
    return [{"name": name, "generation": {"model": "native-4.05", "prompt": name,
        "temperature": 1., "top_p": .95, "top_k": 20, "max_tokens": 128, "seed": 0},
        "standalone_messages": [{"role": "user", "content": name}], "decisions": deepcopy(decisions)}
        for name in ("code", "prose", "structured")]


def config(root, *, status="failed"):
    (root / "reports").mkdir()
    options = {"gpu_memory_utilization": .75, "worker_extension_cls": operator.EXTENSION,
        "additional_config": {"fuzev": {"enabled": True, "ple_storage": "disk"}}}
    return operator.OperatorConfig.model_validate({"input_role": "operator-non-heldout",
        "recipe_commit": "a" * 40, "engine_config": pin(root / "engine.json", options),
        "binding_receipt": pin(root / "binding.json", {"source_commit": operator.CORE}),
        "workloads": pin(root / "workloads.json", templates()), "padding_text": " padding",
        "correctness_evidence": pin(root / "correctness.json", {"model": operator.MODEL, "status": status})})


class Tokenizer:
    @staticmethod
    def encode(text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return [ord(char) for char in text]

    @staticmethod
    def apply_chat_template(messages, **kwargs):
        assert kwargs["enable_thinking"] is False and kwargs["tokenize"] is False
        assert kwargs["reasoning_effort"] == "none" and kwargs["preserve_thinking"] is True
        return "chat:" + "".join(message["content"] for message in messages) + str(kwargs["add_generation_prompt"])


class Engine:
    def __init__(self, options):
        self.options, self.positions, self.enabled, self.reset = options, 0, False, 0

    def get_num_unfinished_requests(self):
        return 0

    def get_tokenizer(self):
        return Tokenizer()

    async def reset_prefix_cache(self, **kwargs):
        assert kwargs == {"reset_running_requests": False, "reset_connector": False}
        self.reset += 1
        return True

    async def collective_rpc(self, method, *, timeout, kwargs):
        assert method == "fuzev_performance_counts" and timeout == 60.
        if kwargs["enable"]:
            assert not self.enabled
            self.enabled = True
        assert self.enabled
        return [snapshot(self.positions)]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    c = config(tmp_path)
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    monkeypatch.setenv("FUZEV_ROOT", str(tmp_path))
    monkeypatch.delenv("FUZEV_SYNTHETIC_STATE_TRACE", raising=False)
    monkeypatch.setattr(operator, "TIERS", (128, 256))
    from fuzev_research import telemetry
    monkeypatch.setattr(telemetry, "admit", lambda *args: None)
    return NS(root=tmp_path, config=c)


def test_disabled_entry_has_no_engine_import_or_input_access(tmp_path):
    with pytest.raises(ValueError, match="permission"):
        asyncio.run(operator.run_operator(object(), tmp_path / "unused"))
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
sys.modules['vllm'] = None
from fuzev_research.vllm_performance_operator import main
try:
    main(['--config', '/absent', '--output', '/absent-output'])
except SystemExit as error:
    assert error.code == 2
else:
    raise AssertionError('Missing inference permission admitted')
assert 'fuzev_research.kev_vllm_research_server' not in sys.modules
"""], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_profile_changes_only_process_core_pin_and_restores_original_on_failure():
    from fuzev_research import kev_vllm_research_server as server
    files, validator = server.RECIPE_FILES, server.validate_inputs
    assert server.CORE_COMMIT == operator.ORIGINAL_CORE
    with pytest.raises(RuntimeError):
        with operator._offline_profile() as profile:
            assert profile.CORE_COMMIT == operator.CORE
            assert profile.validate_inputs is validator
            assert "vllm_performance_operator.py" in profile.RECIPE_FILES
            raise RuntimeError("fixture")
    assert server.CORE_COMMIT == operator.ORIGINAL_CORE and server.RECIPE_FILES == files


def test_control_options_only_toggle_fusion_and_leave_original_owned(rig):
    fused = rig.config.engine_config.read()
    before = deepcopy(fused)
    stock = operator.control_options(fused)
    assert stock["additional_config"]["fuzev"]["enabled"] is False
    stock["additional_config"]["fuzev"]["enabled"] = True
    assert stock == fused == before
    for field, value in (("worker_extension_cls", "unknown"), ("disable_log_stats", True)):
        with pytest.raises(ValueError):
            operator.control_options({**fused, field: value})


@pytest.mark.parametrize("field,value", [("temperature", 0.), ("top_p", 1.), ("top_k", 0),
                                        ("max_tokens", 256), ("seed", 1), ("model", "another-model")])
def test_templates_do_not_silently_change_sampling_or_select_another_model(rig, field, value):
    items = templates()
    items[0]["generation"][field] = value
    changed = pin(rig.config.workloads.path, items)
    rig.config.workloads = operator.EvidencePin.model_validate(changed)
    with pytest.raises(ValueError):
        operator._templates(rig.config)


@pytest.mark.parametrize("chat", [False, True])
def test_padding_preserves_templates_and_records_actual_ids_not_nominal_counts(rig, chat):
    items = operator._templates(rig.config)
    if chat:
        items[0].generation.messages = deepcopy(items[0].standalone_messages)
        items[0].generation.prompt = None
    before = [item.model_dump() for item in items]
    workloads, receipts = operator.prepare_workloads(Tokenizer(), items, rig.config.padding_text)
    assert [item.model_dump() for item in items] == before and len(workloads) == len(receipts) == 6
    for index, (workload, receipt) in enumerate(zip(workloads, receipts)):
        prompt = operator._render(Tokenizer(), prompt=workload.generation.prompt, messages=workload.generation.messages)
        assert receipt["actual_generation_prompt_tokens"] == len(prompt) >= receipt["declared_tier_tokens"]
        assert receipt["generation_prompt_sha256"] == operator.token_hash(prompt)
        assert workload.generation.seed == 0 and workload.generation.model == operator.MODEL
        assert workload.decisions == items[index // 2].decisions
    with pytest.raises(ValueError, match="padding"):
        operator.prepare_workloads(Tokenizer(), items, None)


@pytest.mark.parametrize("status", ["in_progress", "unknown"])
def test_unfinished_correctness_is_not_admitted(rig, status):
    changed = pin(rig.config.correctness_evidence.path, {"model": operator.MODEL, "status": status})
    rig.config.correctness_evidence = operator.EvidencePin.model_validate(changed)
    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(operator.run_operator(rig.config, rig.root / "reports/out", allow_inference=True))
    assert not (rig.root / "reports/out").exists()


@pytest.mark.parametrize("incomplete", [False, True])
def test_wave_counts_are_owner_wide_and_only_acknowledged_totals_are_nonnull(incomplete):
    engine = Engine({})
    engine.enabled = True

    async def rpc(*args, **kwargs):
        return [snapshot(engine.positions, pending=int(incomplete and engine.positions > 0))]

    engine.collective_rpc = rpc

    async def operation():
        engine.positions += 42
        return {"num_cached_tokens": 999, "streams": 4}  # Not used to infer physical positions.

    row = asyncio.run(operator.observed_wave(engine, operation, acknowledgement_timeout=0))
    assert engine.reset == 1
    assert row["forward_counts"]["physical_forwarded_tokens"] == (None if incomplete else 42)
    assert operator._physical_total([row]) == (None if incomplete else 42)
    assert "owner-wide" in row["forward_counts"]["per_request_attribution"]


def test_busy_engine_refuses_even_reset_and_counter_access():
    engine = Engine({})
    engine.get_num_unfinished_requests = lambda: 1
    with pytest.raises(ValueError, match="idle"):
        asyncio.run(operator.observed_wave(engine, None))
    assert engine.reset == 0 and not engine.enabled


@pytest.fixture
def execution(rig, monkeypatch):
    events, engines, services = [], [], []
    sources = {"recipes": rig.config.recipe_commit, "vllm-fuzev": operator.CORE,
        "vllm-exl3-fuzev": "47f4d0ad73624d08b2da7fba9f5be5b28231ee3a",  # unpublished lab commit ID
        "exllamav3-fuzev": "ba60611d031f673a12ebbf2f133c583b4e12df7f"}  # unpublished lab commit ID

    def verify(pins, required):
        for path, digest in pins.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    @contextmanager
    def profile():
        yield NS(read=lambda path: {"binary_sha256": operator.NATIVE_BINARY}, verify_pins=verify)

    class Socket:
        def __enter__(self):
            return self

        def bind(self, address):
            assert address == ("127.0.0.1", 8000)
            events.append("port-reserved")

        def __exit__(self, *args):
            events.append("port-closed")

    @asynccontextmanager
    async def engine_context(options, *, fused):
        assert events[-1] in ("port-reserved", "control-close")
        engine = Engine(options)
        engines.append(engine)
        events.append("fused-open" if fused else "control-open")
        try:
            yield engine
        finally:
            events.append("fused-close" if fused else "control-close")

    def runtime(engine):
        return {"runtime_sources": sources, "runtime_cell": {"engine": "vllm-fuzev" if
            engine.options["additional_config"]["fuzev"]["enabled"] else "vllm-fusion-disabled-control",
            "memory_fraction": .75, "additional_config": deepcopy(engine.options["additional_config"])}}

    async def capture_service(engine, intent, destination):
        assert events[-1] == "fused-open"
        service = NS(tokenizer=engine.get_tokenizer(), identity="b" * 64, provenance=runtime(engine))

        async def close():
            events.append("service-close")

        service.close = close
        services.append(service)
        return service

    @asynccontextmanager
    async def network(engine, service, sock, timeout):
        events.append("mock-network")  # No actual socket, HTTP request or timing evidence.
        service.engine = engine
        yield None

    class FakeHarness:
        def __init__(self, client, config, output):
            self.service = services[-1]

        async def preflight(self):
            return {"provenance": self.service.provenance, **dict.fromkeys((
                "active_engine_requests", "retained_prefixes", "retained_tokens", "retained_multimodal_bytes"), 0)}

        async def pipeline(self, workload, mode, index):
            self.service.engine.positions += 11
            return {"mode": mode, "elapsed_seconds": .1, "completion_tokens": 0 if mode == "decision-only" else 128,
                "logical_prefix_tokens": 128, "forwarded_tokens": 999, "forward_counts_status": "lower-bound"}

        async def close(self):
            events.append("harness-close")
            return []

    async def wave(engine, cases, **kwargs):
        assert kwargs["allow_inference"] is True and kwargs["reset_prefix_cache"] is False
        assert len(cases) in (1, 4)
        engine.positions += len(cases) * len(cases[0].prompt_token_ids)
        return {"control_kind": "fusion-disabled-same-fork", "results": []}

    def preload(*args, **kwargs):
        return {"operator_runtime_sources": sources, "research": {"source_pins": {}},
                "options": rig.config.engine_config.read()}

    async def stock_capture(engine, intent, server):
        assert not engine.options["additional_config"]["fuzev"]["enabled"]
        return {"runtime": runtime(engine), "control_kind": "fusion-disabled-same-fork", "head_loaded": False}

    monkeypatch.setattr(operator, "_offline_profile", profile)
    monkeypatch.setattr(operator, "socket", NS(socket=Socket))
    monkeypatch.setattr(operator, "_engine", engine_context)
    monkeypatch.setattr(operator, "_network", network)
    monkeypatch.setattr(operator, "Harness", FakeHarness)
    monkeypatch.setattr(operator, "capture_control", stock_capture)
    monkeypatch.setattr(capture, "capture_operator", preload)
    monkeypatch.setattr(capture, "capture_and_build", capture_service)
    monkeypatch.setattr(control, "describe_control", runtime)
    monkeypatch.setattr(control, "run_control", wave)
    from fuzev_research import vllm_service
    monkeypatch.setattr(vllm_service, "describe_runtime", runtime)
    return NS(events=events, engines=engines, services=services)


def test_sequential_control_shutdown_then_fused_real_http_glue_and_unchanged_failed_gate(rig, execution):
    out = rig.root / "reports/out"
    report = asyncio.run(operator.run_operator(rig.config, out, allow_inference=True))
    assert report["status"] == "measurement-complete", report.get("error")
    assert report["existing_correctness_status"] == "failed" and not report["runtime_qualified"]
    assert report["control_shutdown_completed"] and not report["pristine_upstream_available"]
    assert execution.events.index("control-close") < execution.events.index("fused-open")
    assert execution.events.index("harness-close") < execution.events.index("fused-close")
    assert len(report["control"]) == 12 and len(report["fused"]) == 36
    assert all(len(cell["waves"]) == 4 and [wave["warmup"] for wave in cell["waves"]] == [True, False, False, False]
               for cell in report["control"] + report["fused"])
    for cell in report["fused"]:
        assert cell["physical_forwarded_tokens"] == 3 * cell["streams"] * 11
        assert cell["request_summary"]["forwarded_tokens"] is None
    assert execution.engines[0].reset == 48 and execution.engines[1].reset == 144
    before = list(execution.events)
    with pytest.raises(FileExistsError):
        asyncio.run(operator.run_operator(rig.config, out, allow_inference=True))
    assert execution.events == before


def test_control_close_failure_stops_before_any_fused_owner_and_preserves_partial_waves(rig, execution, monkeypatch):
    original = operator._engine

    @asynccontextmanager
    async def broken(options, *, fused):
        async with original(options, fused=fused) as engine:
            yield engine
        raise RuntimeError("synthetic shutdown failed")

    monkeypatch.setattr(operator, "_engine", broken)
    report = asyncio.run(operator.run_operator(rig.config, rig.root / "reports/out", allow_inference=True))
    assert report["status"] == "failed" and len(report["control"]) == 12
    assert all(len(cell["waves"]) == 4 for cell in report["control"])
    assert report["fused"] == [] and "fused-open" not in execution.events


def test_occupied_port_refuses_before_capture_or_model_construction(rig, execution, monkeypatch):
    class Busy:
        def __enter__(self):
            return self

        def bind(self, address):
            raise OSError("synthetic occupied port")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(operator, "socket", NS(socket=Busy))
    report = asyncio.run(operator.run_operator(rig.config, rig.root / "reports/out", allow_inference=True))
    assert report["status"] == "failed" and execution.engines == []


def test_changed_operator_config_refused_before_output_or_capture(rig, execution):
    path = rig.root / "operator.json"
    document = rig.config.model_dump(mode="json")
    document["seed"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="config changed"):
        asyncio.run(operator.run_operator(rig.config, rig.root / "reports/out", allow_inference=True, config_path=path))
    assert execution.engines == [] and not (rig.root / "reports/out").exists()


@pytest.mark.parametrize("close_failure", [False, True])
@pytest.mark.parametrize("port", [8000, 8765])
def test_http_factory_uses_loopback_owner_header_and_does_not_swallow_lifespan_failure(monkeypatch, close_failure, port):
    import uvicorn
    calls = []
    service = NS(config=NS(model_name=operator.MODEL, prefix_ttl_seconds=900.), expire=lambda: None)

    async def close():
        calls.append("close")
        if close_failure:
            raise RuntimeError("synthetic service close failed")

    service.close = close

    class FakeServer:
        def __init__(self, config):
            assert config.host == "127.0.0.1" and config.port == port
            self.config, self.started, self.should_exit = config, False, False

        async def serve(self, *, sockets):
            assert len(sockets) == 1
            try:
                async with self.config.app.router.lifespan_context(self.config.app):
                    self.started = True
                    while not self.should_exit:
                        await asyncio.sleep(.001)
            except RuntimeError:
                pass  # Mimic Uvicorn logging/swallowing a failed ASGI lifespan.

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(operator.httpx, "AsyncClient", Client)

    async def check():
        sock = NS(getsockname=lambda: ("127.0.0.1", port))
        async with operator._network(object(), service, sock, 600.):
            pass

    if close_failure:
        with pytest.raises(RuntimeError, match="lifespan"):
            asyncio.run(check())
    else:
        asyncio.run(check())
    assert calls[0]["base_url"] == f"http://127.0.0.1:{port}"
    assert calls[0]["trust_env"] is False and calls[0]["timeout"] == 600.
    assert len(calls[0]["headers"]["X-Fuzev-Performance-Owner"]) == 64
    assert calls[-1] == "close"


@pytest.mark.parametrize("change", [None, "source", "task", "binary", "readout"])
def test_stock_worker_capture_records_bindings_and_requires_generation_only(rig, monkeypatch, change):
    checkout = rig.root / "synthetic-core"
    source = "vllm/v1/fuzev_readout.py"
    pins = {str(checkout / source): "b" * 64}
    sources = {"recipes": rig.config.recipe_commit, "vllm-fuzev": operator.CORE}
    modules = {"vllm-fuzev": str(checkout / "vllm/__init__.py")}
    binary = str(rig.root / "synthetic-native.so")
    worker = {"rank": 0, "cuda": {"initialized": True, "device": 0}, "torch_version": "fixture",
        "cuda_version": "fixture", "binary_pins": {"test": "c" * 64}, "runtime_sources": sources,
        "modules": modules, "readout_sources": {source: "b" * 64},
        "native_extension": {"path": binary, "sha256": operator.NATIVE_BINARY}}
    intent = {"operator_runtime_sources": sources, "operator_modules": modules,
        "preload": {"torch_version": "fixture", "cuda_version": "fixture"},
        "research": {"source_pins": pins, "binary_pins": {"test": "c" * 64}}}
    actual = {"runtime_sources": sources, "runtime_cell": {"memory_fraction": .75}}
    monkeypatch.setattr(control, "describe_control", lambda engine: actual)
    server = NS(READOUT_SOURCE_FILES={source}, verify_pins=lambda *args: dict(pins),
        read=lambda path: {"binary": binary, "binary_sha256": operator.NATIVE_BINARY})

    async def healthy():
        pass

    async def tasks():
        return ["generate", "fuzev_features"] if change == "task" else ["generate"]

    async def rpc(method, **kwargs):
        assert method == "fuzev_runtime_observation"
        assert kwargs == {"timeout": 60., "kwargs": {"release_unused_startup_cache": True}}
        return [worker]

    if change == "source":
        worker["runtime_sources"] = {"vllm-fuzev": "wrong"}
    elif change == "binary":
        worker["native_extension"]["sha256"] = "d" * 64
    elif change == "readout":
        worker["readout_sources"] = {}
    engine = NS(check_health=healthy, get_supported_tasks=tasks, collective_rpc=rpc)
    if change != "task":
        result = asyncio.run(operator.capture_control(engine, intent, server))
        assert not result["head_loaded"] and not result["runtime_qualified"]
        assert result["control_kind"] == "fusion-disabled-same-fork"
        assert result["workers"] == [worker] and result["recorded_runtime_sources"] == sources
        assert result["observed_source_pins"] == pins
    else:
        with pytest.raises(ValueError):
            asyncio.run(operator.capture_control(engine, intent, server))
