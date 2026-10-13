"""CPU-only launch-boundary fakes; no vLLM or backbone is initialized."""

import asyncio
from copy import deepcopy
import json
import shutil
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from fuzev_research import telemetry, vllm_comparison_extract as extractor, vllm_server as server
from fuzev_research.vllm_ple_storage import EXL3_LOOKUP_OP, EXL3_TABLE_ENV
from test_vllm_ple_storage import GraphMode, change, options, resolved


@pytest.fixture(autouse=True)
def cleanup_tmp(tmp_path):
    assert not torch.cuda.is_available()
    yield
    shutil.rmtree(tmp_path)


@pytest.fixture
def runtime(monkeypatch):
    state = NS(config=None, events=[])
    for name in ("vllm", "vllm.engine", "vllm.engine.arg_utils", "vllm.v1",
                 "vllm.v1.engine", "vllm.v1.engine.async_llm", "vllm.v1.fuzev"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    class Args:
        def __init__(self, **kwargs):
            state.events.append(("arguments", kwargs))

        def create_engine_config(self):
            state.events.append(("configuration",))
            return state.config

    class LLM:
        @staticmethod
        def from_vllm_config(config):
            assert config is state.config
            state.events.append(("construction",))
            return NS(shutdown=lambda: state.events.append(("shutdown",)))

    sys.modules["vllm.engine.arg_utils"].AsyncEngineArgs = Args
    sys.modules["vllm.v1.engine.async_llm"].AsyncLLM = LLM
    sys.modules["vllm.v1.fuzev"].fuzev_enabled = lambda _: True
    admission = lambda *args: state.events.append(("admission", *args))
    monkeypatch.setattr(telemetry, "admit", admission)
    monkeypatch.setattr(extractor, "admit", admission)
    return state


def launch_config(quantization):
    cfg = resolved(quantization)
    cfg.model_config.max_model_len = 262144
    cfg.model_config.runner_type = "generate"
    cfg.model_config.hf_text_config = NS(hidden_size=2560)
    cfg.cache_config = NS(gpu_memory_utilization=.85)
    cfg.scheduler_config = NS(max_num_seqs=4, async_scheduling=False)
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.use_v2_model_runner = False
    cfg.speculative_config = cfg.kv_transfer_config = cfg.ec_transfer_config = None
    return cfg


def launch(path, opts, runtime, monkeypatch, tmp_path):
    async def enter(context):
        async with context():
            runtime.events.append(("yield",))

    if path == "extractor":
        asyncio.run(enter(lambda: extractor.engine_context(opts)))
    else:
        import uvicorn
        engine_path, service_path = tmp_path / "engine.json", tmp_path / "service.json"
        engine_path.write_text(json.dumps(opts))
        service_path.write_text("{}")
        monkeypatch.setattr(server, "VllmServiceConfig", NS(model_validate_json=lambda _: NS(
            runtime_cell={"quantization": getattr(runtime, "pin_quantization", opts.get("quantization"))})))
        monkeypatch.setattr(server, "create_app", lambda context, _: context)
        monkeypatch.setattr(uvicorn, "run", lambda context, **_: asyncio.run(enter(context)))
        monkeypatch.setattr(sys, "argv", ["vllm_server", "--engine-config", str(engine_path),
                                        "--service-config", str(service_path)])
        server.main()


@pytest.mark.parametrize("path", ["server", "extractor"])
@pytest.mark.parametrize("quantization,eager", [("modelopt_mixed", True), ("exl3", True), ("exl3", False)])
def test_valid_launch_keeps_one_engine_and_packed_piecewise(runtime, monkeypatch, tmp_path, path, quantization, eager):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options(quantization)
    runtime.config = launch_config(quantization)
    opts["enforce_eager"] = runtime.config.model_config.enforce_eager = eager
    if not eager:
        opts["compilation_config"] = {"cudagraph_mode": "PIECEWISE",
                                      "splitting_ops": ["vllm::attention", EXL3_LOOKUP_OP]}
        runtime.config.compilation_config = NS(cudagraph_mode=GraphMode.PIECEWISE,
                                               splitting_ops=opts["compilation_config"]["splitting_ops"])
    before = deepcopy(opts)
    launch(path, opts, runtime, monkeypatch, tmp_path)
    names = [event[0] for event in runtime.events]
    assert names == ["arguments", "configuration", "admission", "construction", "yield", "shutdown"]
    assert ("admission", 128 << 20) in runtime.events
    assert opts == before


@pytest.mark.parametrize("path", ["server", "extractor"])
def test_packed_auto_quantization_is_rechecked_from_actual_config(runtime, monkeypatch, tmp_path, path):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options("exl3")
    opts.pop("quantization")
    runtime.config = launch_config("exl3")
    runtime.pin_quantization = "exl3"
    launch(path, opts, runtime, monkeypatch, tmp_path)
    assert [event[0] for event in runtime.events].count("construction") == 1


@pytest.mark.parametrize("path", ["server", "extractor"])
@pytest.mark.parametrize("key,value", [("engram_config", None), ("enforce_eager", "true"),
                                       ("safetensors_load_strategy", "eager"), ("additional_config", {})])
def test_unsafe_options_fail_before_llm_construction(runtime, monkeypatch, tmp_path, path, key, value):
    opts = options()
    opts[key] = value
    runtime.config = launch_config("modelopt_mixed")
    with pytest.raises(ValueError):
        launch(path, opts, runtime, monkeypatch, tmp_path)
    assert not {"admission", "construction", "yield"} & {event[0] for event in runtime.events}
    if path == "server":
        assert runtime.events == []


@pytest.mark.parametrize("path", ["server", "extractor"])
@pytest.mark.parametrize("location,value", [
    (("engram_config", "cpu_offload"), True),
    (("load_config", "safetensors_load_strategy"), "eager"),
    (("model_config", "enforce_eager"), False),
    (("additional_config", "fuzev", "ple_storage"), "resident"),
    (("model_config", "quantization"), "exl3"),
])
def test_resolved_config_is_checked_before_llm_construction(runtime, monkeypatch, tmp_path, path, location, value):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options()
    runtime.config = launch_config("modelopt_mixed")
    change(runtime.config, location, value)
    with pytest.raises(ValueError):
        launch(path, opts, runtime, monkeypatch, tmp_path)
    assert [event[0] for event in runtime.events] == ["arguments", "configuration"]


@pytest.mark.parametrize("path", ["server", "extractor"])
@pytest.mark.parametrize("mode,ops", [(GraphMode.FULL, [EXL3_LOOKUP_OP]),
                                      (GraphMode.FULL_AND_PIECEWISE, [EXL3_LOOKUP_OP]),
                                      (GraphMode.PIECEWISE, ["vllm::attention"])])
def test_packed_graph_mismatch_fails_before_llm_construction(runtime, monkeypatch, tmp_path, path, mode, ops):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options("exl3")
    opts["enforce_eager"] = False
    runtime.config = launch_config("exl3")
    runtime.config.model_config.enforce_eager = False
    runtime.config.compilation_config = NS(cudagraph_mode=mode, splitting_ops=ops)
    with pytest.raises(ValueError, match="outside capture"):
        launch(path, opts, runtime, monkeypatch, tmp_path)
    assert [event[0] for event in runtime.events] == ["arguments", "configuration"]
