"""Exclusive native4 procedural checks with the frozen CUDA raw-T1 research head.

No HTTP attachment, legacy loader relaxation, or GPU work on import. The caller
must explicitly authorize a new offline owner; CPU fixtures are not native proof.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from .kev_vllm_research_head import HEAD_SHA, PLAN, PLAN_SHA, QUALITY, QUALITY_SHA, PARAMETER_COUNT, _frozen_artifacts
from .paths import ROOT


NATIVE_COMMIT = "ba60611d031f673a12ebbf2f133c583b4e12df7f"  # unpublished lab commit ID
# Manifest sha256 of the unpublished lab `fuzev-optimized` bundle used by the recorded runs.
FOUNDATION_SHA = "59d2d45db19d642e457717bbf96d9523ca00fc1005315ff37b90dddc28448144"
EXTENSION_SHA = "e1398c20e1b1c4a0c91c8661790cd2543156c3556af2d5963c75dfbe4fc6c05b"
FLAGS = {"experimental": True, "calibrated": False, "qualified": False, "servable": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def _cuda_parameters(bundle):
    import torch
    parameters = list(bundle.parameters())
    require(not bundle.training and parameters and all(not p.requires_grad and p.dtype == torch.float32
            for p in parameters), "Research qualification requires the frozen FP32 head")
    require(sum(p.numel() for p in parameters) == PARAMETER_COUNT, "Frozen shared head parameter geometry changed")


def head_snapshot(head):
    import torch
    from .kev_research_server import ResearchHead
    require(type(head) is ResearchHead and head.head_id == "reference"
            and head.calibration_id == "uncalibrated:T1"
            and getattr(head.score, "__func__", None) is ResearchHead.score,
            "Require the actual unchanged native CUDA raw-T1 research scorer")
    _cuda_parameters(head.bundle)
    state = {name: tensor.detach().cpu().clone() for name, tensor in head.bundle.state_dict().items()}
    require(len(state) == 15 and sum(t.numel() for t in state.values()) == PARAMETER_COUNT
            and all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in state.values()),
            "Invalid frozen shared head storage")
    return {"weights": state, "temperatures": {name: 1. for name in ("choice", "noul", "ordinal")},
            "head_id": head.head_id, "manifest": head.manifest_sha256, "calibration": head.calibration_id}


def research_metadata(service):
    import torch
    from safetensors.torch import load
    from . import kev_research_server as research
    from .encoding import identity
    from .native import NativeBackend, runtime_sources
    from exllamav3.ext import exllamav3_ext

    require(type(service) is research.ResearchService and type(service.backend) is NativeBackend
            and service.model_name == "native-4.05" and service.head.backend is service.backend,
            "Research evidence requires the actual exclusive native4 service/backend/head")
    case, checkpoint = _frozen_artifacts()
    snapshot = head_snapshot(service.head)
    expected = load(checkpoint)
    actual = snapshot["weights"]
    checkpoint_matches = actual.keys() == expected.keys() and all(torch.equal(actual[name], expected[name]) for name in actual)
    module = sys.modules.get(type(service.head.bundle).__module__)
    source_module = getattr(module, "__file__", None)
    lm = next(module for module, _, _ in service.backend.model.fwd_modules if module.caps.get("logits_output"))
    encoded_labels = [service.backend.preparer.encode(" " + chr(65 + i)) for i in range(16)]
    require(service.head.lm is lm and all(len(ids) == 1 for ids in encoded_labels)
            and service.head.labels == [ids[0] for ids in encoded_labels]
            and len(set(service.head.labels)) == 16,
            "Actual native query LM projection or label mapping changed")
    provenance = service.provenance
    require(all(provenance.get(key) is value for key, value in FLAGS.items())
            and all(provenance.get(key) == value for key, value in {
                "head_lane": "lm-assisted", "learned_head_count": 1,
                "calibration_id": "uncalibrated:T1", "raw_inference_temperature": 1.}.items()),
            "Frozen CUDA raw-T1 research provenance changed")
    sources = runtime_sources()
    plan = research.read(PLAN)
    reference = Path(plan["original_feature_identity"])
    original = research.read(reference)
    model = research.read(research.ACQUISITION)["models"][service.model_name]
    runtime = service.backend.readout_runtime
    serving = research.serving_identity_for(service.model_name, model, original, runtime)
    require(provenance.get("feature_identity") == provenance.get("reference_feature_identity") == original
            and provenance.get("serving_identity") == serving and provenance.get("readout_runtime") == runtime
            and Path(service.backend.config.directory).resolve() == Path(model["local_snapshot"]).resolve()
            and service.backend.config.infer_params.ngram_stream_from_disk is True,
            "Original native4 geometry, payload or disk PLE binding changed")
    require(original["encoding"] == identity() and service.lane.context_limit == 262144
            and service.lane.chunk_size == runtime["feature_chunk_tokens"]
            and service.backend.cache.max_num_tokens == runtime["cache_tokens"]
            and service.backend.cache.num_slots == runtime["max_hybrid_slots"], "Actual cache or encoding changed")
    pins = provenance.get("source_pins", {})
    paths = {PLAN, QUALITY, reference, research.ACQUISITION, Path(__file__),
             Path(research.__file__), *(Path(__file__).with_name(name) for name in (
                 "native_qualification.py", "native_state_observer.py", "native.py", "native_service.py",
                 "server.py", "contracts.py", "service_contracts.py", "service_images.py", "encoding.py", "preparation.py"))}
    observed_pins = {str(path): research.sha(path) for path in paths | {Path(path) for path in pins}}
    return {"service_provenance": deepcopy(provenance), "readout_runtime": deepcopy(runtime),
            "model_name": service.model_name, "window_tokens": 262144,
            "head_manifest_sha256": service.head.manifest_sha256, "calibration_id": "uncalibrated:T1",
            "source_verification": "observed_sources", "runtime_sources": sources,
            "recorded_native_commit": NATIVE_COMMIT, "recorded_foundation_sha256": FOUNDATION_SHA,
            "recorded_extension_sha256": EXTENSION_SHA, "native_extension_sha256": research.sha(exllamav3_ext.__file__),
            "recorded_source_pins": deepcopy(pins), "observed_source_pins": observed_pins, "source_module": source_module,
            "head_checkpoint_verified": checkpoint_matches,
            "head_execution_device": str(next(service.head.bundle.parameters()).device),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "backbone_weight_bytes_verified": False}


def _run(args):
    require(args.allow_inference is True and os.environ.get("FUZEV_ROOT") == str(ROOT),
            "Explicit inference opt-in and project root required")
    from . import kev_research_server as research
    from .native_qualification import QualificationConfig, run_qualification
    from .native_qualification_cli import _backend_options

    raw = args.cases.read_bytes()
    document = json.loads(raw)
    if args.mode == "procedural":
        document.update(full_window=None, run_full_window=False)
    else:
        require(document.get("full_window") is not None, "Full-window mode requires the existing explicit plan")
        document["run_full_window"] = True
    cases = QualificationConfig.model_validate(document)
    require(cases.observe_native_state is True and all(case.input_role == "operator-non-heldout" for case in
            cases.cases + cases.image_controls + ([cases.full_window.case] if cases.full_window else [])),
            "Require explicit native state observation and non-heldout procedural inputs")
    _frozen_artifacts()
    plan = research.read(PLAN)
    original = research.read(plan["original_feature_identity"])
    model = research.read(research.ACQUISITION)["models"]["native-4.05"]
    options = _backend_options(SimpleNamespace(feature_identity=original))
    output = args.output.parent.resolve() / args.output.name
    require(output.suffix == ".json" and output.is_relative_to(ROOT / "reports") and output.parent.is_dir(),
            "Use a new exclusive JSON in an existing private reports directory")
    paths = {PLAN, QUALITY, research.ACQUISITION, Path(plan["original_feature_identity"]), args.cases.resolve(),
             Path(__file__), Path(research.__file__), *(Path(__file__).with_name(name) for name in (
                 "native_qualification.py", "native_state_observer.py", "native_qualification_cli.py",
                 "kev_vllm_research_head.py", "server.py", "native_service.py", "native.py", "contracts.py",
                 "service_contracts.py", "service_images.py", "encoding.py", "preparation.py"))}
    pins = {str(path): research.sha(path) for path in paths}
    require(pins[str(args.cases.resolve())] == hashlib.sha256(raw).hexdigest(), "Procedural cases changed")
    config = SimpleNamespace(prefix_capacity=8, prefix_ttl_seconds=900., source_pins=pins,
                             reference_feature_identity=None, serving_identity=None)
    # Reserve exclusive evidence before constructing an offline backbone owner.
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        from .native import NativeBackend, runtime_sources
        report = None
        try:
            sources = runtime_sources()
            with ExitStack() as resources:
                backend = NativeBackend(model["local_snapshot"], **options)
                resources.callback(backend.close)
                service = research.build_research_service(backend, "native-4.05", model, sources, config)
                resources.callback(service.close)
                report = run_qualification(service, cases, allow_inference=True, research_cuda=True)
        except (Exception, KeyboardInterrupt) as error:
            report = report or {"qualified": False, "evidence_kind": "native", "cleanup_errors": []}
            report.update(status="failed", execution_status="failed", error=f"{type(error).__name__}: {error}")
        report.update(operator_mode=args.mode, input_cases_sha256=pins[str(args.cases.resolve())])
        handle.write(json.dumps(report, allow_nan=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0 if report.get("execution_status") == "passed" and not report["cleanup_errors"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--allow-inference", action="store_true", required=True)
    parser.add_argument("--mode", choices=("procedural", "full-window"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (Exception, KeyboardInterrupt) as error:
        print("Native research qualification failed: " + type(error).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
