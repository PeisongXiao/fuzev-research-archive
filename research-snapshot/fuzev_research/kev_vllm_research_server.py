"""Explicit raw-T1 cross-runtime research; launch permission is external.

Service JSON is the existing VllmServiceConfig plus a research envelope with
image_id, absolute source_pins and binary_pins keyed by audited library name.
runtime_cell/runtime_sources must be separately observed operator pins, never
guessed from options. Explicit capture mode records one loaded research engine
before strict attachment; normal pinned serving and production stay unchanged.
"""

import argparse
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import partial
import hashlib
import importlib
import json
import os
from pathlib import Path
import re

from .kev_vllm_research_head import (HEAD_SHA, PLAN, PLAN_SHA, QUALITY, QUALITY_SHA,
                                      KevVllmResearchHead)
from .paths import ROOT
from .vllm_ple_storage import validate_ple_storage_config, validate_ple_storage_options
from .vllm_receipts import READOUT_SOURCE_FILES
from .vllm_server import create_app
from .vllm_service import (VllmService, VllmServiceConfig, build_service,
                           describe_runtime, validate_engine_config)


AUDIT = ROOT / "cross-cell/vllm-plan.md"
ACQUISITION = ROOT / "data/manifests/acquisition.json"
EXL3_VIEW = ROOT / "runs/model-views/exl3-4.05-vllm-complete-index-20261011"
CORE_COMMIT = "62fc998457f58c5a0e6801ab9c11c9fafbcb4b8d"  # unpublished lab commit ID
MEMORY_FRACTION = .75
NV_IMAGE = os.environ.get("FUZEV_RESEARCH_CUDA_IMAGE")
EXL3_IMAGE = os.environ.get("FUZEV_RESEARCH_EXL3_IMAGE", NV_IMAGE)
CELLS = {
    "vllm-nvidia": ("vllm-nvidia", "nvidia/Qwen3.8-Flash-Next-NVFP4",
                    "fc694b54fb0174e0913e6adf86691ef85a4ead47", "modelopt_mixed", NV_IMAGE),
    "vllm-radixark": ("vllm-radixark", "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                      "7b719225242aacd3dbd3f9407468c2ee9a9d2594", "modelopt_fp4", NV_IMAGE),
    "vllm-exl3-4.05": ("native-4.05", "turboderp/Qwen3.8-Flash-Next-exl3",
                  "55a732e0c4c3d4614bc42b68493bb930d9b02c0a", "exl3", EXL3_IMAGE),
}
EXL3_BINARIES = {
    "_C_stable_libtorch.abi3.so": "0c68fbc8e8abb686e529060ccee8d0e146dd8d3c17041c4570132265e5a15d12",
    "_moe_C_stable_libtorch.abi3.so": "3c971bc6a84f3b9873faaa56e62b529d860deb27312310d3ce011dbe95bbeb2c",
}
FLAGS = {"experimental": True, "calibrated": False, "qualified": False, "servable": False}
QUALIFICATION = "raw-T1 research only; known external visual quality failure; no production qualification"
RECIPE_FILES = ("kev_vllm_research_server.py", "kev_vllm_research_head.py", "vllm_server.py",
                "vllm_service.py", "vllm_receipts.py", "vllm_images.py", "vllm_ple_storage.py",
                "contracts.py", "service_contracts.py", "encoding.py", "kev_vllm_state_trace.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_pins(pins, required):
    return {str(path): sha(path) for path in {*(Path(path) for path in pins), *required}}


def model_binding(options, name):
    require(name in CELLS, "Not an audited vLLM research cell")
    key, repo, revision, quantization, _ = CELLS[name]
    model = read(ACQUISITION)["models"][key]
    snapshot = Path(model["local_snapshot"])
    directory = EXL3_VIEW if quantization == "exl3" else snapshot
    require(Path(options["model"]).resolve() == directory.resolve()
            and Path(options["tokenizer"]).resolve() == snapshot.resolve()
            and options["quantization"] == quantization, "Engine model/tokenizer differs from acquired cell")
    metadata = read(directory / "config.json")
    text = metadata.get("text_config", metadata)
    require(metadata.get("architectures") == ["Qwen4ExpForConditionalGeneration"]
            and text.get("hidden_size") == 2560 and text.get("max_position_embeddings") == 262144
            and metadata.get("vision_config", {}).get("hidden_size") == 1152,
            "Requires the acquired multimodal Qwen4 geometry")
    paths = {directory / "config.json", snapshot / "config.json", snapshot / "tokenizer_config.json"}
    result = {"repo": model["repo"], "revision": model["revision"], "recorded_repo": repo, "recorded_revision": revision,
              "local_snapshot": str(snapshot),
              "model_directory": str(directory), "tokenizer_directory": str(snapshot), "quantization": quantization}
    if quantization == "exl3":
        view = read(directory / "serving-view.json")
        require(view["tensor_conversion"] is False and view["configuration_only"] is True
                and view["gpu_qualified"] is False and Path(view["source"]).resolve() == snapshot.resolve()
                and view["source_config_sha256"] == sha(snapshot / "config.json"), "Wrong EXL3 same-payload view")
        payload = view["payload"]
        require({"ngram_embedding.safetensors", "vision_k6.safetensors"} <= payload.keys(), "EXL3 PLE/vision payload absent")
        for filename, item in payload.items():
            require(Path(filename).name == filename and item["same_file"] is True
                    and (directory / filename).samefile(item["source"])
                    and (snapshot / filename).samefile(item["source"])
                    and (directory / filename).stat().st_size == item["bytes"], "EXL3 payload is not the original acquired file")
        paths.update(directory / filename for filename in (
            "serving-view.json", "pack_scan.json", "model.safetensors.index.json", "quantization_config.json"))
        result["serving_view"] = view
    return result, paths


def validate_inputs(options, config, research):
    require(config.model_name in CELLS and config.head_id == "reference",
            "Research requires frozen PLAN and raw-T1 policy pins, not production CAL artifacts")
    _, _, _, quantization, image = CELLS[config.model_name]
    actual_image = research["image_id"]
    parent_image = research.get("parent_image_id", actual_image)
    require(os.environ.get("FUZEV_ROOT") == str(ROOT)
            and os.environ.get("VLLM_PLUGINS") == ("vllm_exl3" if quantization == "exl3" else ""),
            "Explicit root and worker-visible plugin policy required")
    policy = {"runner": "generate", "dtype": "bfloat16", "kv_cache_dtype": "auto",
              "max_model_len": 262144,
              "max_num_batched_tokens": 16384, "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
              "data_parallel_size": 1, "async_scheduling": False, "enable_prefix_caching": True,
              "enable_chunked_prefill": True, "mamba_cache_mode": "align", "enforce_eager": True,
              "trust_remote_code": False, "limit_mm_per_prompt": {"image": 1, "video": 0},
              "mm_processor_kwargs": {"truncation": False}}
    require(all(options.get(key) == value for key, value in policy.items())
            and options.get("additional_config", {}).get("fuzev", {}).get("enabled") is True,
            "Engine differs from audited context/resources/FuZev/image policy")
    require(config.runtime_cell.get("quantization") == quantization, "Declared runtime quantization differs")
    validate_ple_storage_options(options, quantization=quantization)
    pins = research["source_pins"]
    required = {PLAN, QUALITY, AUDIT, ACQUISITION, *(Path(__file__).with_name(name) for name in RECIPE_FILES)}
    observed_pins = verify_pins(pins, required)
    reference_path = Path(read(PLAN)["original_feature_identity"])
    model, model_paths = model_binding(options, config.model_name)
    observed_pins.update(verify_pins({}, {reference_path, *model_paths}))
    require(config.feature_identity == read(reference_path), "Reference identity is not the frozen original identity")
    binaries = research["binary_pins"]
    require(set(binaries) == set(EXL3_BINARIES)
            and all(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) for digest in binaries.values()),
            "Both genuine worker-visible vLLM libraries need explicit hashes")
    image_binding = {"image_id": actual_image, "parent_image_id": parent_image,
                     "source": "operator launch environment receipt; binaries independently checked"}
    if research.get("image_build_receipt") is not None:
        receipt = Path(research["image_build_receipt"])
        verify_pins(pins, {receipt})
        build = read(receipt)
        image_binding["build_receipt"] = {"path": str(receipt), "sha256": sha(receipt), "recorded": build}
    return {"source_pins": dict(pins), "model_artifact": model,
            "observed_source_pins": observed_pins, "recorded_core_commit": CORE_COMMIT,
            "recorded_artifacts": {"plan_sha256": PLAN_SHA, "quality_sha256": QUALITY_SHA},
            "observed_artifacts": {"plan_sha256": sha(PLAN), "quality_sha256": sha(QUALITY)},
            "recorded_binary_pins": EXL3_BINARIES,
            "image_binding": image_binding,
            "raw_t1_policy_sha256": sha(PLAN), "calibration_artifact": None}


def loaded_binding(research, binding):
    import torch
    import vllm
    package = Path(vllm.__file__).parent
    version = importlib.import_module("vllm._version")
    paths = {package.parent / name for name in READOUT_SOURCE_FILES}
    paths.add(Path(version.__file__))
    verify_pins(research["source_pins"], paths)
    libraries = {}
    loaded = {str(Path(path).resolve()) for path in torch.ops.loaded_libraries}
    for filename, digest in research["binary_pins"].items():
        path = package / filename
        libraries[filename] = {"path": str(path.resolve()), "sha256": sha(path), "recorded_sha256": digest,
                               "loaded_in_frontend": str(path.resolve()) in loaded}
    versions = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "api_cuda_initialized_before_head": torch.cuda.is_initialized()}
    print(json.dumps({"phase": "research-runtime-binding", **versions}), flush=True)
    return {**binding, "binary_binding": {"libraries": libraries,
                "scope": "worker-visible package files; not a worker binary attestation"}, **versions}


class ResearchVllmService(VllmService):
    def __init__(self, engine, tokenizer, head, config, provenance, guard, *, research_binding, **kwargs):
        require(head.calibration_id == "uncalibrated:T1"
                and head.requires_lm_scores is True, "Not the immutable raw-T1 assisted head")
        provenance = {**provenance, **head.provenance, **research_binding, **FLAGS,
                      "reference_feature_identity": config.feature_identity,
                      "serving_identity": provenance["runtime_cell"], "raw_inference_temperature": 1.,
                      "reference_head": {"head_id": head.head_id, "manifest_sha256": PLAN_SHA,
                                         "raw_t1_policy_sha256": PLAN_SHA, "calibration_artifact": None},
                      "transfer_qualification": QUALIFICATION}
        trace_config = os.environ.get('FUZEV_SYNTHETIC_READOUT_TRACE')
        self._trace_fd = None
        self._trace_request = ContextVar('fuzev_synthetic_trace_request', default=None)
        self._trace_ids = frozenset()
        self._trace_claimed = set()
        self._trace_counts = {}
        if trace_config:
            path = Path(trace_config).resolve()
            require(path.is_relative_to(ROOT / 'reports'), 'Synthetic trace configuration must be private research evidence')
            trace = read(path)
            ids = trace['request_ids']
            output = Path(trace['output']).resolve()
            require(isinstance(ids, list) and len(ids) == len(set(ids)) == 4
                    and all(isinstance(value, str) and re.fullmatch(r'engine_[0-9a-f]{64}', value) for value in ids)
                    and output.parent == path.parent and not output.exists(),
                    'Trace only explicitly allowlisted synthetic IDs into a new adjacent file')
            self._trace_ids = frozenset(ids)
            provenance['source_pins'] = {**provenance['source_pins'], str(path): sha(path)}
            provenance['synthetic_readout_trace'] = {'config': str(path), 'sha256': sha(path),
                'output': str(output), 'request_ids': ids, 'scope': 'Already-owned CPU readouts only; no head or kernel changes'}
        super().__init__(engine, tokenizer, head, config, provenance, guard, **kwargs)
        if trace_config:
            self._trace_fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)

    async def decide(self, request, request_id):
        traced = request_id in self._trace_ids
        if traced:
            require(request.primitive == 'choice' and request.prefix_id is not None
                    and request.question in ('Which marker belongs to branch A?', 'Which marker belongs to branch B?')
                    and [item.id for item in request.candidates] == ['alpha', 'beta'],
                    'Synthetic trace must not record quality-panel requests')
            require(request_id not in self._trace_claimed, 'Synthetic trace request IDs are one-use only')
            self._trace_claimed.add(request_id)
            self._trace_counts[request_id] = 0
        token = self._trace_request.set(request_id if traced else None)
        try:
            result = await super().decide(request, request_id)
        finally:
            self._trace_request.reset(token)
        return {**result, **FLAGS, "head_execution_device": "cuda:0", "head_sha256": HEAD_SHA,
                "head_manifest_sha256": PLAN_SHA, "qualification": QUALIFICATION,
                "provenance": self.provenance}

    async def _engine_call(self, *args, **kwargs):
        request_id = self._trace_request.get()
        traced = request_id is not None and kwargs.get('features') is True
        if traced:
            require(request_id in self._trace_claimed and self._trace_counts[request_id] < 3,
                    'Synthetic trace allows exactly one query and two key readouts per ID')
            self._trace_counts[request_id] += 1
        value, receipt = await super()._engine_call(*args, **kwargs)
        if traced:
            import torch
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    and value.dtype == torch.float32 and value.ndim == 1,
                    'Trace only the existing owned CPU FP32 result')
            labels = kwargs.get('baseline_token_ids') or ()
            record = {'api_request_id': request_id, 'engine_receipt': receipt,
                      'stage': 'query' if labels else 'key', 'shape': list(value.shape),
                      'dtype': str(value.dtype), 'baseline_token_ids': list(labels),
                      'sha256': hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
                      'feature_sha256': hashlib.sha256(value[:2560].contiguous().numpy().tobytes()).hexdigest(),
                      'baseline_logits': value[2560:].tolist(), 'values': value.tolist()}
            data = (json.dumps(record, allow_nan=False, ensure_ascii=True) + '\n').encode('ascii')
            require(os.write(self._trace_fd, data) == len(data), 'Incomplete private synthetic trace record')
            os.fsync(self._trace_fd)
        return value, receipt

    async def close(self):
        try:
            await super().close()
        finally:
            if self._trace_fd is not None:
                os.close(self._trace_fd)
                self._trace_fd = None


async def build_research_service(engine, config, *, options, research):
    binding = validate_inputs(options, config, research)
    validate_engine_config(engine.vllm_config)
    validate_ple_storage_config(engine.vllm_config)
    binding = loaded_binding(research, binding)
    if 'runtime_capture' in research:
        binding = {**binding, 'runtime_capture': research['runtime_capture']}
    await engine.check_health()
    # Existing full runtime/source comparison and guard/admission remain intact.
    service = await build_service(engine, config, inspector=describe_runtime,
        head_factory=KevVllmResearchHead,
        service_factory=partial(ResearchVllmService, research_binding=binding))
    try:
        require(service.images is not None and callable(getattr(service, "readout_health", None)),
                "Actual trusted image adapter and strict worker readout receipts required")
        return service
    except BaseException:
        await service.close()
        raise


@asynccontextmanager
async def engine_context(options):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from .telemetry import admit
    config = AsyncEngineArgs(**options).create_engine_config()
    validate_engine_config(config)
    validate_ple_storage_options(options, quantization=config.model_config.quantization)
    validate_ple_storage_config(config)
    admit(128 << 20)
    engine = AsyncLLM.from_vllm_config(config)
    try:
        yield engine
    finally:
        engine.shutdown()


async def research_health(service):
    service._check()
    await service.engine.check_health()
    return {"status": "loaded", "model": service.config.model_name, **FLAGS,
            "provenance": service.provenance, "worker_readout": service.readout_health(),
            "capabilities": {"typed_decisions": True, "text_generation": True,
                             "shared_lm_assisted_head": True, "trusted_still_images": service.images is not None}}


def create_research_app(context, config, *, options, research, builder=None):
    from starlette.responses import JSONResponse
    if builder is None:
        builder = partial(build_research_service, options=options, research=research)
    app = create_app(context, config, builder=builder)

    @app.get("/research-health")
    async def health():
        try:
            return await research_health(app.state.service)
        except (MemoryError, RuntimeError) as error:
            return JSONResponse({"status": "unavailable", "model": config.model_name,
                                 "reason": str(error), **FLAGS}, status_code=503)

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--service-config")
    mode.add_argument("--capture-runtime-out", type=Path)
    parser.add_argument("--model", choices=tuple(CELLS))
    parser.add_argument("--binding-receipt", type=Path)
    parser.add_argument("--image-build-receipt", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="Optional override; default: engine setting")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Optional override; default: engine setting")
    args = parser.parse_args()
    if args.capture_runtime_out is not None:
        from types import SimpleNamespace
        from .kev_vllm_capture import capture_and_build, capture_operator
        if not args.model or args.binding_receipt is None:
            parser.error('Capture mode requires --model and --binding-receipt')
        destination = args.capture_runtime_out.expanduser().resolve()
        require(destination.suffix == '.json' and
                any(destination.is_relative_to(ROOT / name) for name in ('reports', 'cross-cell')),
                'Capture must use a new owned JSON under reports or cross-cell')
        require(not destination.exists() and destination.parent.is_dir(),
                'Capture destination must be new with an existing parent')
        operator = capture_operator(args.model, args.engine_config, args.binding_receipt,
                                    image_build_receipt=args.image_build_receipt)
        for key in ("gpu_memory_utilization", "max_num_seqs"):
            value = getattr(args, key)
            if value is not None:
                operator['options'][key] = value
        config = SimpleNamespace(**operator['service'])

        async def captured_builder(engine, _intent):
            return await capture_and_build(engine, operator, destination)

        app = create_research_app(partial(engine_context, operator['options']), config,
            options=operator['options'], research=operator['research'], builder=captured_builder)
        uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=30)
        return
    if any(value is not None for value in (args.model, args.binding_receipt, args.image_build_receipt)):
        parser.error('Capture-only arguments must not modify the pinned serving path')
    documents = {Path(path).resolve(): Path(path).read_bytes()
                 for path in (args.engine_config, args.service_config)}
    options = json.loads(documents[Path(args.engine_config).resolve()])
    for key in ("gpu_memory_utilization", "max_num_seqs"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    envelope = json.loads(documents[Path(args.service_config).resolve()])
    research = envelope.pop("research")
    config = VllmServiceConfig.model_validate(envelope)
    research["source_pins"] = dict(research["source_pins"])
    for path, body in documents.items():
        # Bind launch inputs without inserting a self-hash into the service JSON.
        research["source_pins"][str(path)] = hashlib.sha256(body).hexdigest()
    validate_inputs(options, config, research)
    uvicorn.run(create_research_app(partial(engine_context, options), config, options=options, research=research),
                host=args.host, port=args.port, timeout_graceful_shutdown=30)


if __name__ == "__main__":
    main()
