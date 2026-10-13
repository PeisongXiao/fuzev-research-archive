"""Read-only, checksum-pinned serving of calibrated FP32 head artifacts."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load

from .head import HeadBundle


TAP = "final-mixer/post-collapse/before-lm-head"


def checked_bytes(path, limit, digest=None):
    with Path(path).open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Serving artifact exceeds its resource bound")
    if digest is not None and hashlib.sha256(value).hexdigest() != digest:
        raise ValueError(f"Serving artifact checksum mismatch: {Path(path).name}")
    return value


class ServingHead:
    def __init__(self, config):
        directory = Path(config.head_directory)
        manifest = json.loads(checked_bytes(
            directory / "manifest.json", 1 << 20, config.manifest_sha256))
        if manifest.get("architecture") == "shared-candidate-dot-product-v1":
            self._load_shared(config, directory)
            return
        calibration = json.loads(checked_bytes(
            directory / "calibration.json", 1 << 20, config.calibration_sha256))
        expected = {
            "schema_version": 1, "architecture": "candidate-dot-product-v1",
            "hidden_size": 2560, "projection_size": 256, "precision": "float32",
            "feature_tap": TAP, "artifact_kind": "calibrated-reference-head",
            "stage": "mixed", "calibration_role": "calibration",
            "feature_identity": config.feature_identity,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible or uncalibrated serving head provenance")
        for key in ("preparation", "training_code", "calibration_code"):
            if not isinstance(manifest.get(key), dict) or not manifest[key]:
                raise ValueError(f"Missing head provenance: {key}")
        for key in ("fit_membership", "development_membership", "calibration_membership"):
            membership = manifest.get(key, {})
            if (not isinstance(membership, dict)
                    or type(membership.get("records")) is not int
                    or membership["records"] <= 0
                    or len(membership.get("ordered_sha256", "")) != 64):
                raise ValueError(f"Missing head membership provenance: {key}")
        tensors = load(checked_bytes(directory / "head.safetensors", 64 << 20,
                                     manifest["weights_sha256"]))
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
               for value in tensors.values()):
            raise ValueError("Serving heads must contain finite FP32 tensors")
        bundle = HeadBundle()
        bundle.load_state_dict(tensors, strict=True)
        temperatures = calibration.get("temperatures", {})
        if (calibration.get("schema_version") != 1
                or set(temperatures) != set(bundle.heads)
                or any(type(value) not in (float, int) or not math.isfinite(value)
                       or value <= 0 for value in temperatures.values())):
            raise ValueError("Invalid serving calibration")
        from .composite_head import validate_composite
        validate_composite(bundle, manifest, temperatures)
        self.bundle = bundle.eval().requires_grad_(False)
        self.temperatures = temperatures
        self.head_id = config.head_id
        self.calibration_id = "sha256:" + config.calibration_sha256
        self.manifest_sha256 = config.manifest_sha256

    def _load_shared(self, config, directory):
        from .shared_head import SharedDecisionHead, load_shared_bundle

        for pin in (config.manifest_sha256, config.calibration_sha256):
            if (not isinstance(pin, str) or len(pin) != 64
                    or any(char not in "0123456789abcdef" for char in pin)):
                raise ValueError("Shared serving requires explicit manifest and calibration pins")
        bundle, manifest, temperatures = load_shared_bundle(
            directory, expected_feature_tap=TAP,
            expected_manifest_sha256=config.manifest_sha256,
            expected_calibration_sha256=config.calibration_sha256)
        expected = {"artifact_kind": "calibrated-reference-head", "stage": "mixed",
                    "calibration_role": "calibration", "feature_identity": config.feature_identity}
        if (type(bundle) is not SharedDecisionHead
                or any(manifest.get(key) != value for key, value in expected.items())):
            raise ValueError("Incompatible or uncalibrated shared serving head provenance")
        gate = manifest.get("shared_development_gate", {})
        checks = gate.get("checks") if isinstance(gate, dict) else None
        if (not isinstance(gate, dict) or gate.get("eligible") is not True
                or gate.get("selection_role") != "development"
                or gate.get("external_selection") is not False
                or gate.get("calibration_selection") is not False
                or gate.get("membership") != manifest["development_membership"]
                or gate.get("temperature_by_primitive") != temperatures
                or not isinstance(checks, list) or not checks
                or any(not isinstance(check, dict) or check.get("passed") is not True for check in checks)):
            raise ValueError("Shared serving requires accepted, matching calibrated DEV evidence")
        # Runtime proof is bounded and self-contained; live corpus qualification stays offline.
        try:
            from .shared_qualification import validate_shared_reference
        except ImportError as exc:
            raise ValueError("Shared serving qualification validator is unavailable") from exc
        validate_shared_reference(bundle, manifest, temperatures)
        self.bundle = bundle.cpu().eval().requires_grad_(False)
        self.temperatures = temperatures
        self.head_id = config.head_id
        self.calibration_id = "sha256:" + config.calibration_sha256
        self.manifest_sha256 = config.manifest_sha256

    @torch.inference_mode()
    def score(self, primitive, query, candidates):
        query = query.detach().to(device="cpu", dtype=torch.float32)
        candidates = candidates.detach().to(device="cpu", dtype=torch.float32)
        if (query.shape != (2560,) or candidates.ndim != 2
                or candidates.shape[1] != 2560 or not 2 <= len(candidates) <= 16
                or not torch.isfinite(query).all() or not torch.isfinite(candidates).all()):
            raise ValueError("Invalid owned final-mixer features")
        logits = self.bundle(primitive, query[None], candidates[None])[0]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite decision logits")
        raw = torch.softmax(logits, dim=0)
        calibrated = torch.softmax(logits / self.temperatures[primitive], dim=0)
        if not torch.isfinite(calibrated).all():
            raise ValueError("Nonfinite calibrated probabilities")
        return logits.tolist(), raw.tolist(), calibrated.tolist()
