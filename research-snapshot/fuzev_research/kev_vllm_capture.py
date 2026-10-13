"""Explicit two-phase research observation; never construct or select a model.

Imports are deferred to the pinned runtime. Preload captures no full runtime
cell; capture_runtime observes an already-owned AsyncLLM and reuses it for
strict attachment. Worker metadata is not a successful readout/quality receipt.
"""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def cuda_observation(torch):
    """Never initialize CUDA just to report allocation or device information."""
    initialized = torch.cuda.is_initialized()
    result = {"initialized": initialized, "device": None, "allocated_bytes": None,
              "reserved_bytes": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None,
              "free_bytes": None, "total_bytes": None}
    if initialized:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        result.update(device=device, allocated_bytes=torch.cuda.memory_allocated(device),
                      reserved_bytes=torch.cuda.memory_reserved(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device), free_bytes=free, total_bytes=total)
    return result


def cache_observation(worker):
    """Read resolved worker allocation metadata, not cache tensor contents."""
    config = getattr(getattr(worker, 'model_runner', None), 'kv_cache_config', None)
    if config is None:
        return {'status': 'unavailable', 'num_blocks': None, 'groups': None,
                'allocated_bytes': None, 'descriptor_bytes': None}
    groups = []
    for group in config.kv_cache_groups:
        specification = group.kv_cache_spec
        specs = getattr(specification, 'kv_cache_specs', {'group': specification})
        precision = sorted({str(dtype) for spec in specs.values()
                            for dtype in ((spec.dtype,) if hasattr(spec, 'dtype') else getattr(spec, 'dtypes', ()))})
        groups.append({'specification': type(specification).__name__, 'layers': len(group.layer_names),
                       'block_size': specification.block_size, 'storage_dtypes': precision,
                       'host_resident': group.host_resident})
    return {'status': 'observed-worker-allocation', 'num_blocks': config.num_blocks,
            'groups': groups, 'allocated_bytes': None,
            'descriptor_bytes': sum(tensor.size for tensor in config.kv_cache_tensors),
            'scope': 'Resolved hybrid-cache descriptors, not physical allocation; '
                     'blocks are not additive per-layer token capacity'}


def _source_observation(server, vllm, quantization):
    from .vllm_service import _source_revision
    directories = {"recipes": Path(server.__file__).parent.parent, "vllm-fuzev": Path(vllm.__file__).parent}
    modules = {"recipes": str(Path(server.__file__)), "vllm-fuzev": str(Path(vllm.__file__))}
    if quantization == "exl3":
        for name, key in (("vllm_exl3", "vllm-exl3-fuzev"), ("exllamav3", "exllamav3-fuzev")):
            spec = importlib.util.find_spec(name)
            _require(spec is not None and spec.origin, "Required source module is not importable: " + name)
            modules[key] = str(Path(spec.origin))
            directories[key] = Path(spec.origin).parent
    return {key: _source_revision(path) for key, path in directories.items()}, modules


def capture_operator(model_name, engine_config, binding_receipt, *, image_build_receipt=None):
    """Observe genuine files/imports before loading, without inventing a cell."""
    from . import kev_vllm_research_server as server
    import torch
    import vllm

    engine_config, binding_receipt = Path(engine_config).resolve(), Path(binding_receipt).resolve()
    raw = engine_config.read_bytes()
    options = json.loads(raw)
    _require(model_name in server.CELLS, "Unknown research cell")
    _, _, _, quantization, parent_image = server.CELLS[model_name]
    receipt = server.read(binding_receipt)
    package = Path(vllm.__file__).parent
    version = importlib.import_module("vllm._version")
    _require(Path(version.__file__).resolve() == (package / "_version.py").resolve(),
             "Version metadata must come from the genuine worker-visible core binding")
    _require(set(receipt["artifacts_sha256"]) == {"_version.py", *server.EXL3_BINARIES},
             "Missing genuine version/binary bindings")
    observed_artifacts = {filename: server.sha(package / filename) for filename in receipt["artifacts_sha256"]}
    plan = server.read(server.PLAN)
    reference = Path(plan["original_feature_identity"])
    sources, modules = _source_observation(server, vllm, quantization)
    model, model_paths = server.model_binding(options, model_name)
    paths = {server.PLAN, server.QUALITY, server.AUDIT, server.ACQUISITION, reference,
             engine_config, binding_receipt, Path(__file__), Path(version.__file__),
             *(Path(server.__file__).with_name(name) for name in server.RECIPE_FILES),
             *(package.parent / name for name in server.READOUT_SOURCE_FILES), *model_paths,
             *(Path(path) for path in modules.values())}
    from .kev_vllm_state_trace import configuration
    state_trace = configuration(server.ROOT)
    if state_trace is not None:
        paths.add(state_trace[0])
    if image_build_receipt is not None:
        paths.add(Path(image_build_receipt).resolve())
    if quantization == "exl3":
        # Preserve the recorded native pairing alongside its observed files.
        pair_path = server.ROOT / "cross-cell/exl3-native-reference-pair/pairing.json"
        pair = server.read(pair_path)
        paths.update((pair_path, Path(pair["binary"])))
        for field in ("reference_runtime_receipt", "reference_completion", "foundation"):
            proof = pair[field]
            paths.add(Path(proof["path"]))
    pins = {str(path): server.sha(path) for path in sorted(paths)}
    _require(pins[str(engine_config)] == hashlib.sha256(raw).hexdigest(), "Engine JSON changed during capture")
    research = {"image_id": os.environ.get("FUZEV_RESEARCH_IMAGE_ID"), "parent_image_id": parent_image,
                "source_pins": pins, "binary_pins": {name: observed_artifacts[name] for name in server.EXL3_BINARIES},
                "recorded_source_commit": server.CORE_COMMIT, "recorded_binding_receipt": receipt,
                "observed_binding_artifacts": observed_artifacts,
                "recorded_plan_sha256": server.PLAN_SHA, "recorded_quality_sha256": server.QUALITY_SHA}
    if image_build_receipt is not None:
        research["image_build_receipt"] = str(Path(image_build_receipt).resolve())
    service = {"model_name": model_name, "head_directory": str(Path(plan["candidates"]["lm-assisted"]["weights"]).parent),
               "head_id": "reference", "manifest_sha256": server.sha(server.PLAN),
               "calibration_sha256": server.sha(server.PLAN),
               "feature_identity": server.read(reference), "prefix_capacity": 8, "prefix_ttl_seconds": 900.}
    # This probe is declared policy only, not a complete observed ServiceConfig.
    probe = SimpleNamespace(**service, runtime_cell={"quantization": quantization}, runtime_sources=sources)
    server.validate_inputs(options, probe, research)
    actual = server.loaded_binding(research, {})
    return _json_copy({"schema_version": 1, "stage": "operator-captured", "options": options,
                      "service": service, "research": research, "operator_runtime_sources": sources,
                      "operator_modules": modules, "model_artifact": model,
                      "preload": {**actual, "api_cuda": cuda_observation(torch),
                                  "scope": "API process before model/head; not worker allocation"},
                      "worker_readout_status": "awaiting-worker-receipt", **server.FLAGS})


def worker_observation(worker):
    """Existing collective_rpc callback: metadata only, no model/forward access."""
    import torch
    import vllm
    from . import kev_vllm_research_server as server
    try:
        from vllm.v1.fuzev_readout import readout_sources
    except ImportError as error:
        raise ImportError("Requires the unpublished lab engine module vllm/v1/fuzev_readout.py") from error

    package = Path(vllm.__file__).parent
    native = sys.modules.get("exllamav3_ext")
    sources, modules = _source_observation(server, vllm, worker.vllm_config.model_config.quantization)
    return {"pid": os.getpid(), "rank": worker.rank,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cuda": cuda_observation(torch), "module": str(Path(vllm.__file__)),
            "kv_cache": cache_observation(worker),
            "runtime_sources": sources, "modules": modules,
            "readout_sources": readout_sources(),
            "binary_pins": {name: server.sha(package / name) for name in server.EXL3_BINARIES},
            "native_extension": None if native is None else {"path": str(Path(native.__file__)),
                                                              "sha256": server.sha(native.__file__)},
            "scope": "worker allocation/import metadata only; not readout or quality validation"}


class RuntimeObservationWorkerExtension:
    """Trusted named metadata RPC; no callable/pickle transport needed."""

    def fuzev_runtime_observation(self, *, release_unused_startup_cache=False):
        _require(type(release_unused_startup_cache) is bool, "Explicit startup cleanup policy required")
        cleanup = None
        if release_unused_startup_cache:
            import torch
            _require(torch.cuda.is_initialized(), "Cleanup requires the already-loaded CUDA worker")
            torch.cuda.synchronize()
            before = cuda_observation(torch)
            torch.cuda.empty_cache()
            after = cuda_observation(torch)
            _require(before['allocated_bytes'] == after['allocated_bytes'],
                     "Startup cleanup must retain every live CUDA allocation")
            cleanup = {'before': before, 'after': after,
                       'scope': 'One-time startup release of unused allocator blocks; '
                                'model, KV and kernel workspaces remain live'}
        result = worker_observation(self)
        if release_unused_startup_cache and os.environ.get('FUZEV_SYNTHETIC_STATE_TRACE'):
            from .kev_vllm_state_trace import install
            from .kev_vllm_research_server import ROOT
            trace = install(self, ROOT)
            if trace is not None:
                result['synthetic_state_trace'] = trace
        if cleanup is not None:
            result['startup_allocator_cleanup'] = cleanup
        return result


async def capture_runtime(engine, operator):
    """Observe ONE supplied engine; require real worker metadata, never reload."""
    from . import kev_vllm_research_server as server
    from .vllm_service import describe_runtime

    operator = _json_copy(operator)
    _require(operator["stage"] == "operator-captured", "Require explicit preload operator capture")
    research = operator["research"]
    server.verify_pins(research["source_pins"], research["source_pins"])
    actual = describe_runtime(engine)
    await engine.check_health()
    tasks = await engine.get_supported_tasks()
    _require({"generate", "fuzev_features"} <= set(tasks), "Same engine lacks required genuine tasks")
    workers = await engine.collective_rpc("fuzev_runtime_observation", timeout=60.,
                                         kwargs={'release_unused_startup_cache': True})
    _require(isinstance(workers, list) and len(workers) == 1, "Capture requires the actual single GPU worker")
    worker = workers[0]
    _require(worker["rank"] == 0 and worker["cuda"]["initialized"] is True,
             "Capture requires an initialized CUDA worker")
    package = Path(operator["operator_modules"]["vllm-fuzev"]).parent
    expected = {name: research["source_pins"][str(package.parent / name)] for name in server.READOUT_SOURCE_FILES}
    recorded_native_pair = None
    if operator["options"]["quantization"] == "exl3":
        pair = server.read(server.ROOT / "cross-cell/exl3-native-reference-pair/pairing.json")
        recorded_native_pair = pair
    config = server.VllmServiceConfig.model_validate({**operator["service"], **actual})
    server.validate_inputs(operator["options"], config, research)
    research["runtime_capture"] = {"schema_version": 1, "stage": "runtime-captured", "preload": operator["preload"],
                                   "workers": workers, "worker_readout_status": "awaiting-worker-receipt",
                                   "recorded_runtime_sources": operator["operator_runtime_sources"],
                                   "recorded_readout_sources": expected,
                                   "recorded_native_pair": recorded_native_pair,
                                   "scope": "actual existing engine metadata; no inference, tuning or quality admission", **server.FLAGS}
    server.verify_pins(research["source_pins"], research["source_pins"])
    return _json_copy({**config.model_dump(mode="json"), "research": research})


def write_capture(destination, envelope):
    """Exclusive private output; never overwrite operator inputs or receipts."""
    destination = Path(destination).expanduser().resolve()
    body = (json.dumps(envelope, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
        handle.write(body)
    return hashlib.sha256(body).hexdigest()


async def capture_and_build(engine, operator, destination):
    """Explicit capture mode only; strict default builder and SAME model survive."""
    from . import kev_vllm_research_server as server

    envelope = await capture_runtime(engine, operator)
    receipt_sha = write_capture(destination, envelope)
    research = envelope.pop("research")
    research["source_pins"][str(Path(destination).expanduser().resolve())] = receipt_sha
    config = server.VllmServiceConfig.model_validate(envelope)
    return await server.build_research_service(engine, config, options=operator["options"], research=research)
