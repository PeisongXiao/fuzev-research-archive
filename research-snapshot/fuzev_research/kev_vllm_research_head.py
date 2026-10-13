"""Frozen A on CUDA FP32, assisted only by engine-supplied label logits.

This is an unqualified raw-T1 research head, not a production loader or a
cross-runtime quality approval. No backbone or native LM module is accessed.
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

from .paths import ROOT


REPORT = ROOT / "reports/2026-10-10-kev-text-evaluation"
PLAN = REPORT / "frozen-plan-v1.json"
QUALITY = REPORT / "native-results-v1/result.json"
PLAN_SHA = "252f43ac3d7eab82c3d3be8db468da17798f4766e6b083e505da72c98bffcf3e"
QUALITY_SHA = "83464ee250883e186876195c4bd5421d27bce07a1968cd13a39b5243da34eb9b"
HEAD_SHA = "f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c"
SOURCE_COMMIT = "9728b98b205d4f9af8ecac4e28ee678ba5ab758b"  # unpublished lab commit ID
SOURCE8 = frozenset(("features.py", "head.py", "metrics.py", "preference_head_retrain.py",
                     "scope.py", "shared_head.py", "shared_train.py", "shared_training_data.py"))
PARAMETER_COUNT = 2_872_401


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked_bytes(path, limit, digest):
    with Path(path).open("rb") as handle:
        value = handle.read(limit + 1)
    _require(len(value) <= limit, "Research artifact exceeds size limit: " + Path(path).name)
    return value


def _frozen_artifacts():
    plan = json.loads(_checked_bytes(PLAN, 1 << 20, PLAN_SHA))
    quality = json.loads(_checked_bytes(QUALITY, 1 << 20, QUALITY_SHA))
    _require(plan["primary"] == quality["primary"] == "lm-assisted"
             and quality["status"] == "pass" and quality["comparison"]["passed"] is True
             and quality["test_sha256"] == plan["test_sha256"]
             and quality["systems"]["lm-assisted"]["overall"]["answered_questions"] == 3089
             and quality["uncertainty"]["index"]["lm-assisted"][0] > plan["baseline_index_ci"][1],
             "Frozen primary KEV evidence does not match")
    _require(plan["temperature"] == 1. and plan["training_allowed"] is False
             and plan["calibration_allowed"] is False and quality["held_data_used_for_tuning"] is False
             and all(quality[key] is False for key in ("calibrated", "qualified", "servable")),
             "Frozen research evidence is not untuned, unqualified raw T1")
    case = plan["candidates"]["lm-assisted"]
    _require(case["parameters"] == PARAMETER_COUNT and case["profile"]["seed"] == 43
             and case["profile"]["key_gain"] == .001 and set(case["source8"]) == SOURCE8,
             "Frozen A checkpoint/class/source identity changed")
    directory = Path(case["tree"]) / "fuzev_research"
    for filename, digest in case["source8"].items():
        path = directory / filename
        _checked_bytes(path, 1 << 20, digest)
    case = {**case, "recorded_source8": dict(case["source8"]),
            "source8": {filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                        for filename in case["source8"]}}
    # Decode exactly the bytes that were checked, not a second filesystem read.
    return case, _checked_bytes(case["weights"], 64 << 20, HEAD_SHA)


def _source_module(case):
    directory = Path(case["tree"]) / "fuzev_research"
    name = "_kev_vllm_research_a"
    package = sys.modules.get(name)
    if package is None:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    _require(package.__path__ == [str(directory)], "Frozen A import namespace differs from pinned tree")
    module = importlib.import_module(name + ".shared_head")
    _require(Path(module.__file__).resolve() == (directory / "shared_head.py").resolve(),
             "Frozen A class imported from an unpinned source")
    return module


class KevVllmResearchHead:
    requires_lm_scores = True
    head_id = "reference"
    calibration_id = "uncalibrated:T1"
    manifest_sha256 = PLAN_SHA
    weights_sha256 = HEAD_SHA
    head_execution_device = "cuda:0"
    experimental = True
    calibrated = False
    qualified = False
    servable = False

    def __init__(self, config=None):
        case, checkpoint = _frozen_artifacts()
        import torch
        from safetensors.torch import load

        _require(torch.cuda.is_available(), "Research head requires CUDA; no CPU fallback")
        tensors = load(checkpoint)
        _require(tensors and all(value.dtype == torch.float32 and torch.isfinite(value).all()
                                 for value in tensors.values()), "Frozen A must contain finite FP32 tensors")
        module = _source_module(case)
        _require(module.SharedDecisionHead.requires_lm_scores is True
                 and module.HIDDEN_SIZE == 2560 and module.PARAMETER_COUNT == PARAMETER_COUNT,
                 "Wrong immutable assisted head class/geometry")
        self.bundle = module.SharedDecisionHead(seed=43, key_gain=case["profile"]["key_gain"]).to(
            device="cuda:0", dtype=torch.float32)
        self.bundle.load_state_dict(tensors, strict=True)
        self.bundle.eval().requires_grad_(False)
        self.manifest_sha256 = hashlib.sha256(PLAN.read_bytes()).hexdigest()
        self.weights_sha256 = hashlib.sha256(checkpoint).hexdigest()
        parameters = list(self.bundle.parameters())
        _require(sum(parameter.numel() for parameter in parameters) == PARAMETER_COUNT
                 and all(parameter.dtype == torch.float32 and parameter.device == torch.device("cuda:0")
                         and torch.isfinite(parameter).all() for parameter in parameters),
                 "Frozen A parameters must remain finite CUDA FP32")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._torch = torch
        self.provenance = {
            "experimental": True, "calibrated": False, "qualified": False, "servable": False,
            "head_id": self.head_id, "calibration_id": self.calibration_id,
            "head_execution_device": self.head_execution_device, "head_sha256": self.weights_sha256,
            "plan_sha256": self.manifest_sha256, "quality_sha256": hashlib.sha256(QUALITY.read_bytes()).hexdigest(),
            "recorded_artifacts": {"head_sha256": HEAD_SHA, "plan_sha256": PLAN_SHA, "quality_sha256": QUALITY_SHA},
            "recorded_source_commit": SOURCE_COMMIT, "recorded_source8": case["recorded_source8"],
            "head_lane": "lm-assisted", "learned_head_count": 1, "source_commit": case.get("commit"),
            "source8": dict(case["source8"]), "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "requires_lm_scores": True,
            "readout_math": {"hidden_size": 2560, "head_dtype": "float32", "raw_temperature": 1.,
                             "baseline_source": "engine-supplied-label-logits", "tf32": False},
            "transfer_qualification": "unqualified research; frozen KEV pass is reference evidence only",
        }

    def score(self, primitive, query, candidates, *, baseline):
        torch = self._torch
        _require(primitive in ("choice", "noul", "ordinal"), "Unknown research decision primitive")
        _require(isinstance(query, torch.Tensor) and isinstance(candidates, torch.Tensor)
                 and isinstance(baseline, torch.Tensor) and query.shape == (2560,)
                 and candidates.ndim == 2 and candidates.shape[1] == 2560
                 and 2 <= len(candidates) <= 16 and baseline.shape == (len(candidates),)
                 and (primitive != "noul" or len(candidates) == 2),
                 "Invalid query/candidate/engine-baseline geometry")
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            # Engine outputs can already be FP32/CUDA; still own independent storage.
            query = query.detach().to(device="cuda:0", dtype=torch.float32, copy=True).unsqueeze(0)
            candidates = candidates.detach().to(device="cuda:0", dtype=torch.float32, copy=True).unsqueeze(0)
            baseline = baseline.detach().to(device="cuda:0", dtype=torch.float32, copy=True).unsqueeze(0)
            _require(all(torch.isfinite(value).all() for value in (query, candidates, baseline)),
                     "Nonfinite owned query/candidate/engine-baseline values")
            logits = self.bundle(primitive, query, candidates, baseline=baseline)[0]
            _require(logits.shape == (candidates.shape[1],) and logits.dtype == torch.float32
                     and logits.device == torch.device("cuda:0") and torch.isfinite(logits).all(),
                     "Invalid/nonfinite CUDA FP32 decision logits")
            probabilities = torch.softmax(logits, dim=0)
            _require(torch.isfinite(probabilities).all(), "Nonfinite raw T1 probabilities")
            raw = probabilities.cpu().tolist()
            return logits.cpu().tolist(), raw, raw[:]
