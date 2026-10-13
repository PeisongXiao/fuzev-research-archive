import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


class CandidateHead(nn.Module):
    def __init__(self, hidden_size=2560, projection_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.query = nn.Linear(hidden_size, projection_size)
        self.key = nn.Linear(hidden_size, projection_size)

    def forward(self, query, candidates, mask=None):
        if query.shape[-1] != self.hidden_size or candidates.shape[-1] != self.hidden_size:
            raise ValueError("Head feature dimension does not match the declared post-collapse tap")
        q = self.query(query.float())
        k = self.key(candidates.float())
        logits = torch.einsum("bd,bkd->bk", q, k) / math.sqrt(self.projection_size)
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf)
        return logits


class HeadBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({name: CandidateHead() for name in ("choice", "noul", "ordinal")})

    def forward(self, primitive, query, candidates, mask=None):
        return self.heads[primitive](query, candidates, mask)


def save_bundle(directory, bundle, metadata, temperatures):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "head.safetensors"
    save_file({name: value.detach().float().cpu().contiguous()
               for name, value in bundle.state_dict().items()}, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {**metadata, "schema_version": 1, "feature_tap": "final-mixer/post-collapse/before-lm-head",
                "hidden_size": 2560, "projection_size": 256, "precision": "float32",
                "weights_sha256": digest, "architecture": "candidate-dot-product-v1"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "calibration.json").write_text(json.dumps({"schema_version": 1, "temperatures": temperatures}, indent=2) + "\n")


def load_bundle(directory, expected_feature_tap="final-mixer/post-collapse/before-lm-head"):
    directory = Path(directory)
    if (directory / "manifest.json").stat().st_size > 1 << 20:
        raise ValueError("Head manifest exceeds its resource bound")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("architecture") in {"shared-candidate-dot-product-v1", "shared-lm-assisted-candidate-dot-product-v1",
                                       "shared-lm-assisted-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-context-trust-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-adapter-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-channel-gate-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-channel-gate-adapter64-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1"}:
        from .shared_head import load_shared_bundle
        return load_shared_bundle(directory, expected_feature_tap)
    if (manifest.get("architecture"), manifest.get("hidden_size"), manifest.get("projection_size"), manifest.get("feature_tap")) != (
            "candidate-dot-product-v1", 2560, 256, expected_feature_tap):
        raise ValueError("Incompatible head bundle")
    weights = directory / "head.safetensors"
    if weights.stat().st_size > 64 << 20:
        raise ValueError("Head artifact exceeds its resource bound")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Head checksum mismatch")
    bundle = HeadBundle()
    bundle.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    if any(not torch.isfinite(value).all() for value in bundle.state_dict().values()):
        raise ValueError("Nonfinite head weights")
    temperatures = json.loads((directory / "calibration.json").read_text())["temperatures"]
    if set(temperatures) != set(bundle.heads) or any(not math.isfinite(t) or t <= 0 for t in temperatures.values()):
        raise ValueError("Invalid head calibration")
    from .composite_head import validate_composite
    validate_composite(bundle, manifest, temperatures)
    return bundle.eval(), manifest, temperatures
