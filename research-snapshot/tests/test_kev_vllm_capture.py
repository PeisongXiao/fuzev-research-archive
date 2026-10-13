"""CPU metadata fixtures only; no genuine Torch/vLLM, model, weights or GPU."""

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

import fuzev_research
from fuzev_research import kev_vllm_capture as capture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_torch(initialized=False):
    torch = ModuleType("torch")
    torch.__version__, torch.version = "2.13.0+cu130", NS(cuda="13.0")
    torch.cuda = NS(is_initialized=Mock(return_value=initialized), current_device=Mock(return_value=0),
                    mem_get_info=Mock(return_value=(100, 200)), memory_allocated=Mock(return_value=10),
                    memory_reserved=Mock(return_value=20), max_memory_allocated=Mock(return_value=30),
                    max_memory_reserved=Mock(return_value=40))
    return torch


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    server = ModuleType("fuzev_research.kev_vllm_research_server")
    server.ROOT = tmp_path
    server.__file__ = str(tmp_path / "recipes/fuzev_research/kev_vllm_research_server.py")
    server.CORE_COMMIT = "a" * 40
    server.FLAGS = {"experimental": True, "calibrated": False, "qualified": False, "servable": False}
    server.CELLS = {name: (name, "synthetic/repo", "b" * 40, quantization, "sha256:" + "1" * 64)
                    for name, quantization in (("vllm-nvidia", "modelopt_mixed"),
                                                ("vllm-radixark", "modelopt_fp4"), ("vllm-exl3-4.05", "exl3"))}
    server.EXL3_BINARIES = {"_C_stable_libtorch.abi3.so": None, "_moe_C_stable_libtorch.abi3.so": None}
    server.RECIPE_FILES = ("kev_vllm_research_server.py", "vllm_service.py")
    server.READOUT_SOURCE_FILES = {"vllm/v1/fuzev_readout.py", "vllm/v1/worker/gpu_model_runner.py"}
    package = tmp_path / "core/vllm"
    paths = [Path(server.__file__), Path(server.__file__).with_name("vllm_service.py"), package / "__init__.py",
             package / "_version.py", *(package / name for name in server.EXL3_BINARIES),
             *(package.parent / name for name in server.READOUT_SOURCE_FILES), tmp_path / "model/config.json"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic metadata/source/binary fixture")
    for name in ("PLAN", "QUALITY", "AUDIT", "ACQUISITION"):
        setattr(server, name, tmp_path / (name.lower() + ".json"))
        dump(getattr(server, name), {})
    reference = tmp_path / "reference.json"
    dump(reference, {"canonical-reference": True})
    dump(server.PLAN, {"original_feature_identity": str(reference),
                       "candidates": {"lm-assisted": {"weights": "/not-read/frozen/head.safetensors"}}})
    server.PLAN_SHA, server.QUALITY_SHA = sha(server.PLAN), sha(server.QUALITY)
    server.sha, server.read = sha, lambda path: json.loads(Path(path).read_text())

    def verify(pins, required):
        return {str(path): sha(path) for path in {*map(Path, pins), *map(Path, required)}}

    server.verify_pins = Mock(side_effect=verify)
    server.validate_inputs = Mock()
    server.loaded_binding = Mock(return_value={"torch_version": "2.13.0+cu130", "cuda_version": "13.0"})
    server.model_binding = Mock(return_value=({"repo": "synthetic/repo"}, {tmp_path / "model/config.json"}))
    server.build_research_service = AsyncMock(return_value="same-engine-service")

    def validate_config(value):
        instance = NS(**deepcopy(value))
        instance.model_dump = lambda **kwargs: deepcopy(value)
        return instance

    server.VllmServiceConfig = NS(model_validate=Mock(side_effect=validate_config))
    vllm, version = ModuleType("vllm"), ModuleType("vllm._version")
    vllm.__file__, version.__file__ = str(package / "__init__.py"), str(package / "_version.py")
    service = ModuleType("fuzev_research.vllm_service")
    sources = {"recipes": "c" * 40, "vllm-fuzev": server.CORE_COMMIT}
    modules = {"recipes": server.__file__, "vllm-fuzev": vllm.__file__}
    service.describe_runtime = Mock(return_value={"runtime_cell": {"genuinely-observed-fixture": True}, "runtime_sources": sources})
    torch = fake_torch()
    for name, module in ((server.__name__, server), (service.__name__, service), ("torch", torch),
                         ("vllm", vllm), ("vllm._version", version)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(fuzev_research, "kev_vllm_research_server", server, raising=False)
    monkeypatch.setattr(capture, "_source_observation", lambda *args: (sources, modules))
    engine_path = tmp_path / "engine.json"
    dump(engine_path, {"quantization": "modelopt_mixed"})
    binding_path = tmp_path / "binding.json"
    binding = {"source_commit": server.CORE_COMMIT, "image_id": "sha256:" + "1" * 64,
               "checkout": str(package.parent), "artifacts_sha256": {
                   name: sha(package / name) for name in ("_version.py", *server.EXL3_BINARIES)}}
    dump(binding_path, binding)
    monkeypatch.setenv("FUZEV_RESEARCH_IMAGE_ID", binding["image_id"])
    return NS(server=server, service=service, torch=torch, package=package, version=version,
              engine_path=engine_path, binding_path=binding_path, binding=binding, sources=sources, modules=modules)


def operator(runtime, cell="vllm-nvidia"):
    return capture.capture_operator(cell, runtime.engine_path, runtime.binding_path)


def engine(runtime, observed):
    worker = {"rank": 0, "pid": 123, "cuda": {"initialized": True, "device": 0, "allocated_bytes": 50},
              "torch_version": "2.13.0+cu130", "cuda_version": "13.0", "native_extension": None,
              "runtime_sources": runtime.sources, "modules": runtime.modules,
              "binary_pins": observed["research"]["binary_pins"],
              "readout_sources": {name: observed["research"]["source_pins"][str(runtime.package.parent / name)]
                                  for name in runtime.server.READOUT_SOURCE_FILES}}
    return NS(check_health=AsyncMock(), get_supported_tasks=AsyncMock(return_value=("generate", "fuzev_features")),
              collective_rpc=AsyncMock(return_value=[worker])), worker


def test_cuda_observer_never_initializes_a_hidden_api_context():
    torch = fake_torch()
    observed = capture.cuda_observation(torch)
    assert observed["initialized"] is False and observed["allocated_bytes"] is None
    torch.cuda.current_device.assert_not_called()
    torch.cuda.mem_get_info.assert_not_called()
    torch.cuda.memory_allocated.assert_not_called()
    torch.cuda.is_initialized.return_value = True
    observed = capture.cuda_observation(torch)
    assert observed["device"] == 0 and observed["allocated_bytes"] == 10
    assert observed["reserved_bytes"] == 20 and observed["peak_allocated_bytes"] == 30


def test_worker_cache_observation_reports_real_precision_without_tensor_reads():
    class Attention:
        block_size = 256
        dtype = 'torch.bfloat16'

    class Recurrent:
        block_size = 256
        dtypes = ('torch.bfloat16', 'torch.float32')

    groups = [NS(kv_cache_spec=spec, layer_names=['layer.0', 'layer.1'], host_resident=False)
              for spec in (Attention(), Recurrent())]
    worker = NS(model_runner=NS(kv_cache_config=NS(num_blocks=1234, kv_cache_groups=groups,
                                                 kv_cache_tensors=[NS(size=1024), NS(size=2048)])))
    result = capture.cache_observation(worker)
    assert result['num_blocks'] == 1234 and result['descriptor_bytes'] == 3072
    assert result['allocated_bytes'] is None
    assert result['groups'][0]['storage_dtypes'] == ['torch.bfloat16']
    assert result['groups'][1]['storage_dtypes'] == ['torch.bfloat16', 'torch.float32']
    assert capture.cache_observation(NS())['allocated_bytes'] is None


@pytest.mark.parametrize("cell", ["vllm-nvidia", "vllm-radixark"])
def test_operator_capture_is_partial_and_genuine_not_a_guessed_runtime(runtime, cell):
    observed = operator(runtime, cell)
    assert observed["stage"] == "operator-captured"
    assert "runtime_cell" not in observed["service"] and "runtime_sources" not in observed["service"]
    assert observed["operator_runtime_sources"] == runtime.sources
    assert observed["preload"]["api_cuda"]["allocated_bytes"] is None
    assert observed["worker_readout_status"] == "awaiting-worker-receipt"
    assert observed["service"]["manifest_sha256"] == observed["service"]["calibration_sha256"] == runtime.server.PLAN_SHA
    runtime.torch.cuda.current_device.assert_not_called()


@pytest.mark.parametrize("change", ["version", "binary", "checkout", "commit"])
def test_operator_records_changed_bindings(runtime, change):
    if change in ("version", "binary"):
        filename = "_version.py" if change == "version" else "_C_stable_libtorch.abi3.so"
        (runtime.package / filename).write_bytes(b"changed")
    else:
        runtime.binding["checkout" if change == "checkout" else "source_commit"] = "wrong"
        dump(runtime.binding_path, runtime.binding)
    result = operator(runtime)
    assert result["research"]["recorded_binding_receipt"] == runtime.binding
    assert result["research"]["observed_binding_artifacts"] == {
        filename: sha(runtime.package / filename) for filename in runtime.binding["artifacts_sha256"]}


def test_runtime_capture_observes_existing_engine_and_leaves_inputs_unchanged(runtime):
    observed = operator(runtime)
    before = deepcopy(observed)
    existing, _ = engine(runtime, observed)
    captured = asyncio.run(capture.capture_runtime(existing, observed))
    runtime.service.describe_runtime.assert_called_once_with(existing)
    existing.collective_rpc.assert_awaited_once_with("fuzev_runtime_observation", timeout=60.,
                                                  kwargs={'release_unused_startup_cache': True})
    assert captured["runtime_cell"] == {"genuinely-observed-fixture": True}
    assert captured["runtime_sources"] == runtime.sources
    assert observed == before
    assert captured["research"]["runtime_capture"]["worker_readout_status"] == "awaiting-worker-receipt"
    assert captured["research"]["runtime_capture"]["qualified"] is False
    runtime.server.build_research_service.assert_not_called()


@pytest.mark.parametrize("change", ["workers", "tasks"])
def test_runtime_capture_fails_closed_on_actual_worker_mismatch(runtime, change):
    observed = operator(runtime)
    existing, worker = engine(runtime, observed)
    if change == "sources":
        runtime.service.describe_runtime.return_value = {"runtime_cell": {}, "runtime_sources": {"recipes": "wrong"}}
    elif change == "binary":
        worker["binary_pins"] = {"wrong": "wrong"}
    elif change == "readout":
        worker["readout_sources"] = {}
    elif change == "origin":
        worker["modules"] = {"vllm-fuzev": "/wrong/image/module.py"}
    elif change == "device":
        worker["cuda"]["device"] = 1
    elif change == "workers":
        existing.collective_rpc.return_value = [worker, deepcopy(worker)]
    else:
        existing.get_supported_tasks.return_value = ("generate",)
    with pytest.raises(ValueError):
        asyncio.run(capture.capture_runtime(existing, observed))
    runtime.server.build_research_service.assert_not_called()


def test_changed_source_after_preload_is_recorded(runtime):
    observed = operator(runtime)
    existing, _ = engine(runtime, observed)
    runtime.engine_path.write_bytes(b"changed input")
    result = asyncio.run(capture.capture_runtime(existing, observed))
    assert result["research"]["source_pins"][str(runtime.engine_path)] == observed["research"]["source_pins"][str(runtime.engine_path)]
    runtime.service.describe_runtime.assert_called_once_with(existing)


def test_exl3_capture_records_actual_native_extension(runtime):
    root = runtime.server.ROOT
    native_source = root / "native-source"
    for key, directory in (("exllamav3-fuzev", native_source), ("vllm-exl3-fuzev", root / "plugin-source")):
        directory.mkdir()
        origin = directory / ("exllamav3/__init__.py" if key == "exllamav3-fuzev" else "__init__.py")
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_text("synthetic native/plugin source")
        runtime.modules[key] = str(origin)
        runtime.sources[key] = "d" * 40
    binary = root / "native-extension.so"
    binary.write_bytes(b"synthetic attested extension")
    pair = {"kind": "attested-native4-runtime-reuse", "source_commit": runtime.sources["exllamav3-fuzev"],
            "source_tree": str(native_source), "target_image_id": runtime.binding["image_id"],
            "binary": str(binary), "binary_sha256": sha(binary)}
    for field in ("reference_runtime_receipt", "reference_completion", "foundation"):
        proof = root / (field + ".json")
        dump(proof, {"synthetic-metadata": True})
        pair[field] = {"path": str(proof), "sha256": sha(proof)}
    pair_path = root / "cross-cell/exl3-native-reference-pair/pairing.json"
    dump(pair_path, pair)
    dump(runtime.engine_path, {"quantization": "exl3"})
    observed = operator(runtime, "vllm-exl3-4.05")
    dump(pair_path, pair)
    existing, worker = engine(runtime, observed)
    worker["native_extension"] = {"path": str(binary), "sha256": sha(binary)}
    captured = asyncio.run(capture.capture_runtime(existing, observed))
    assert str(pair_path) in captured["research"]["source_pins"]
    worker["native_extension"]["path"] = "/wrong/image/native-extension.so"
    result = asyncio.run(capture.capture_runtime(existing, observed))
    assert result["research"]["runtime_capture"]["workers"][0]["native_extension"]["path"] == "/wrong/image/native-extension.so"


def test_capture_build_writes_exclusive_receipt_and_passes_same_engine(runtime, tmp_path):
    observed = operator(runtime)
    existing, _ = engine(runtime, observed)
    destination = tmp_path / "captured-service.json"
    assert asyncio.run(capture.capture_and_build(existing, observed, destination)) == "same-engine-service"
    args = runtime.server.build_research_service.call_args
    assert args.args[0] is existing
    assert args.kwargs["options"] == observed["options"]
    assert args.kwargs["research"]["source_pins"][str(destination)] == sha(destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    raw = destination.read_bytes()
    with pytest.raises(FileExistsError):
        capture.write_capture(destination, {"replacement": True})
    assert destination.read_bytes() == raw


def test_nonfinite_capture_does_not_create_a_receipt(tmp_path):
    destination = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        capture.write_capture(destination, {"allocation": float("nan")})
    assert not destination.exists()


def test_worker_callback_only_reads_metadata(runtime, monkeypatch):
    readout = ModuleType("vllm.v1.fuzev_readout")
    readout.readout_sources = lambda: {"actual-loaded-source": "observed"}
    monkeypatch.setitem(sys.modules, "vllm.v1.fuzev_readout", readout)
    monkeypatch.delitem(sys.modules, "exllamav3_ext", raising=False)
    runtime.torch.cuda.is_initialized.return_value = True
    worker = NS(rank=0, vllm_config=NS(model_config=NS(quantization="modelopt_mixed")))
    observation = capture.worker_observation(worker)
    assert observation["rank"] == 0 and observation["cuda"]["allocated_bytes"] == 10
    assert observation["runtime_sources"] == runtime.sources
    assert observation["binary_pins"] == {name: sha(runtime.package / name) for name in runtime.server.EXL3_BINARIES}
    assert "not readout or quality validation" in observation["scope"]


def test_named_worker_extension_uses_existing_worker_without_pickle(monkeypatch):
    worker = capture.RuntimeObservationWorkerExtension()
    observer = Mock(return_value={"scope": "metadata-only"})
    monkeypatch.setattr(capture, "worker_observation", observer)
    assert worker.fuzev_runtime_observation() == {"scope": "metadata-only"}
    observer.assert_called_once_with(worker)


@pytest.mark.parametrize("live_changed", [False, True])
def test_startup_cleanup_only_releases_dead_allocator_blocks(monkeypatch, live_changed):
    torch = fake_torch(initialized=True)
    torch.cuda.synchronize = Mock()
    torch.cuda.empty_cache = Mock()
    torch.cuda.memory_allocated.side_effect = [10, 11 if live_changed else 10]
    torch.cuda.memory_reserved.side_effect = [30, 20]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(capture, "worker_observation", Mock(return_value={"scope": "metadata-only"}))
    worker = capture.RuntimeObservationWorkerExtension()
    if live_changed:
        with pytest.raises(ValueError, match="every live CUDA allocation"):
            worker.fuzev_runtime_observation(release_unused_startup_cache=True)
        capture.worker_observation.assert_not_called()
    else:
        result = worker.fuzev_runtime_observation(release_unused_startup_cache=True)
        cleanup = result['startup_allocator_cleanup']
        assert cleanup['before']['allocated_bytes'] == cleanup['after']['allocated_bytes'] == 10
        assert cleanup['before']['reserved_bytes'] == 30 and cleanup['after']['reserved_bytes'] == 20
    torch.cuda.synchronize.assert_called_once_with()
    torch.cuda.empty_cache.assert_called_once_with()
