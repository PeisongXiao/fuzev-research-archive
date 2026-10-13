"""Self-contained reference readout and bounded residual scaffolding, not admission.

Cached controls require separately qualified corpus/producer bindings. These local
guards do not replace corpus qualification, DEV selection, or calibrated retention.
Native imports and CUDA allocation happen only on uncached forward, never on load.
"""

import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import threading
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from .head import CandidateHead
from .scope import TRAINING_SOURCES


HIDDEN = 2560
VOCAB = 248320
RANK = 256
BITS = 6
MAX_BATCH = 256
ANCHOR_LIMIT = 512 << 20
STATE_LIMIT = 64 << 20
JSON_LIMIT = 64 << 10
TAP = "final-mixer/post-collapse/before-lm-head"
ANCHOR_PROFILE = "full-native-exl3-label-anchor-v1"
HEAD_PROFILE = "bounded-native-anchor-residual-v1"
FORMULA = "bound*tanh((raw-valid_mean(raw))/bound);mask_after"
ENV_KEYS = ("EXL3_INT8_GEMV", "EXL3_INT8_GEMV_MAX_K", "EXL3_NO_FUSED_RECONSTRUCT")
CODEBOOKS = {"mcg": 0xCBAC1FED - (1 << 32), "mul1": 0x83DCD12D - (1 << 32)}
DTYPES = {"I16": torch.int16, "I32": torch.int32, "F16": torch.float16, "F32": torch.float32}


def _digest(value, length=64):
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("Invalid provenance digest")
    return value


def _sha(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < limit:
        raise ValueError("Missing, symlinked or oversized artifact")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        read = 0
        while chunk := handle.read(1 << 20):
            read += len(chunk)
            if read >= limit:
                raise ValueError("Artifact grew past its resource bound")
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, digest):
    _digest(digest)
    if _sha(path, JSON_LIMIT) != digest:
        raise ValueError("Manifest checksum mismatch")
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Manifest must be an object")
    return value


def _write_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(data) >= JSON_LIMIT:
        raise ValueError("Manifest exceeds its resource bound")
    with Path(path).open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def _anchor_spec(codebook):
    spec = {"trellis": {"shape": [HIDDEN // 16, VOCAB // 16, 16 * BITS], "dtype": "I16"},
            "suh": {"shape": [HIDDEN], "dtype": "F16"},
            "svh": {"shape": [VOCAB], "dtype": "F16"}}
    for name, enabled in codebook.items():
        if enabled:
            spec[name] = {"shape": [], "dtype": "I32"}
    return spec


def _anchor_metadata(metadata):
    required = {"schema_version", "architecture", "feature_tap", "hidden_size", "vocab_size", "bits",
                "input_dtype", "output_dtype", "label_text", "label_ids", "codebook", "tensors",
                "native_commit", "extension_sha256", "native_environment", "source_readout_sha256",
                "tokenizer_sha256", "encoding_sha256", "operator_sha256", "model_repo", "model_revision"}
    if set(metadata) != required:
        raise ValueError("Unknown or incomplete frozen readout metadata")
    expected = {"schema_version": 1, "architecture": ANCHOR_PROFILE, "feature_tap": TAP,
                "hidden_size": HIDDEN, "vocab_size": VOCAB, "bits": BITS,
                "input_dtype": "float16", "output_dtype": "float16",
                "label_text": [" " + chr(ord("A") + i) for i in range(16)]}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported reference readout geometry or semantics")
    if any(type(metadata[key]) is not int for key in ("schema_version", "hidden_size", "vocab_size", "bits")):
        raise ValueError("Invalid numeric reference metadata")
    labels = metadata["label_ids"]
    if (not isinstance(labels, list) or len(labels) != 16
            or any(type(label) is not int or not 0 <= label < VOCAB for label in labels)
            or len(set(labels)) != 16):
        raise ValueError("Expected sixteen distinct A..P label-token IDs")
    codebook = metadata["codebook"]
    if (not isinstance(codebook, dict) or set(codebook) != set(CODEBOOKS)
            or any(type(value) is not bool for value in codebook.values())
            or metadata["tensors"] != _anchor_spec(codebook)):
        raise ValueError("Invalid packed readout tensor/codebook schema")
    environment = metadata["native_environment"]
    if (not isinstance(environment, dict) or set(environment) != set(ENV_KEYS)
            or any(value is not None and (not isinstance(value, str) or len(value) > 64)
                   for value in environment.values())):
        raise ValueError("Missing producer native environment")
    _digest(metadata["native_commit"], 40)
    for key in ("extension_sha256", "source_readout_sha256", "tokenizer_sha256", "encoding_sha256", "operator_sha256"):
        _digest(metadata[key])
    for key in ("model_repo", "model_revision"):
        if not isinstance(metadata[key], str) or not 0 < len(metadata[key]) <= 256:
            raise ValueError("Missing reference model provenance")
    return metadata


def _check_tensors(tensors, spec):
    if set(tensors) != set(spec):
        raise ValueError("Unexpected state tensor keys")
    for name, expected in spec.items():
        value = tensors[name]
        if (not isinstance(value, torch.Tensor) or list(value.shape) != expected["shape"]
                or value.dtype != DTYPES[expected["dtype"]] or value.requires_grad):
            raise ValueError("Wrong packed/state tensor shape, dtype or ownership")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite artifact tensor")
        if name in CODEBOOKS and value.item() != CODEBOOKS[name]:
            raise ValueError("Unsupported native codebook identifier")


def _inspect_file(path, digest, limit, spec):
    _digest(digest)
    if _sha(path, limit) != digest:
        raise ValueError("State checksum mismatch")
    # Inspect headers before allocating. Integer trellis pages need not be scanned.
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if handle.metadata() or set(handle.keys()) != set(spec):
            raise ValueError("Unexpected safetensors metadata or tensor keys")
        for name, expected in spec.items():
            tensor = handle.get_slice(name)
            if tensor.get_shape() != expected["shape"] or tensor.get_dtype() != expected["dtype"]:
                raise ValueError("State header disagrees with pinned tensor schema")
            if expected["dtype"] in ("F16", "F32") or name in CODEBOOKS:
                _check_tensors({name: handle.get_tensor(name)}, {name: expected})


def save_anchor(directory, tensors, metadata):
    """Export full packed readout tensors; never construct or load a model."""
    metadata = copy.deepcopy(metadata)
    _anchor_metadata(metadata)
    _check_tensors(tensors, metadata["tensors"])
    size = sum(value.numel() * value.element_size() for value in tensors.values())
    if size >= ANCHOR_LIMIT - JSON_LIMIT:
        raise ValueError("Frozen readout exceeds its resource bound")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "anchor.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, str(path))
    metadata["weights_sha256"] = _sha(path, ANCHOR_LIMIT)
    manifest_sha = _write_json(directory / "anchor.json", metadata)
    return load_anchor(directory, manifest_sha256=manifest_sha)


def load_anchor(directory, *, manifest_sha256):
    directory = Path(directory)
    metadata = _read_json(directory / "anchor.json", manifest_sha256)
    weights_sha = metadata.pop("weights_sha256", None)
    _digest(weights_sha)
    _anchor_metadata(metadata)
    _inspect_file(directory / "anchor.safetensors", weights_sha, ANCHOR_LIMIT, metadata["tensors"])
    return FrozenNativeAnchor(directory, metadata, weights_sha, manifest_sha256)


class FrozenNativeAnchor:
    """One lazy, reference-pinned CUDA readout shared by independent heads."""

    def __init__(self, directory, metadata, weights_sha256, manifest_sha256):
        _anchor_metadata(metadata)
        self.directory = Path(directory)
        self._metadata = copy.deepcopy(metadata)
        self.weights_sha256 = _digest(weights_sha256)
        self.manifest_sha256 = _digest(manifest_sha256)
        self._runtime = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @property
    def producer_identity_sha256(self):
        keys = ("native_commit", "extension_sha256", "native_environment", "source_readout_sha256",
                "tokenizer_sha256", "encoding_sha256", "model_repo", "model_revision")
        return hashlib.sha256(json.dumps({key: self._metadata[key] for key in keys},
                                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _check_environment(self):
        expected = self._metadata["native_environment"]
        if {key: os.environ.get(key) for key in ENV_KEYS} != expected:
            raise ValueError("Native environment differs from the frozen reference producer")

    def _load_runtime(self, device):
        self._check_environment()
        if not torch.cuda.is_available():
            raise RuntimeError("The exact reference anchor requires CUDA, not a CPU approximation")
        extension = importlib.import_module("exllamav3.ext").exllamav3_ext
        if _sha(extension.__file__, 1 << 30) != self._metadata["extension_sha256"]:
            raise ValueError("Native extension differs from the frozen reference producer")
        module = importlib.import_module("exllamav3.modules.quant.exl3")
        expected_fused = (os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0")
        if module.no_fused_reconstruct != expected_fused:
            raise ValueError("Native module was imported under a different environment")
        path = self.directory / "anchor.safetensors"
        _inspect_file(path, self.weights_sha256, ANCHOR_LIMIT, self._metadata["tensors"])
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key).to(device) for key in handle.keys()}
        self._runtime = module.LinearEXL3(None, in_features=HIDDEN, out_features=VOCAB,
                                         **tensors, key="frozen_native4_reference_anchor")
        if self._runtime.default_out_dtype != torch.float16:
            raise ValueError("Native reference output precision changed")
        self._device = device

    @torch.no_grad()
    def logits(self, query, mask):
        _query_mask(query, mask)
        # Native B1 scratch buffers are shared; serialize initialization and forward.
        with self._lock:
            return self._logits_locked(query, mask)

    def _logits_locked(self, query, mask):
        self._check_environment()
        device = query.device if query.device.type == "cuda" else torch.device("cuda:0")
        if self._runtime is None:
            self._load_runtime(device)
        elif self._device != device:
            raise ValueError("One frozen anchor may not silently migrate CUDA devices")
        outputs = []
        for row, valid in zip(query, mask):
            # B1 preserves the producer's GEMV/SQ path, even in a cached-training batch.
            feature = row.detach().to(device=device, dtype=torch.float16).reshape(1, 1, HIDDEN).contiguous()
            if not torch.isfinite(feature).all():
                raise ValueError("Feature cannot be represented by the producer FP16 cast")
            vocabulary = self._runtime.forward(feature, {})
            if vocabulary.dtype != torch.float16 or vocabulary.shape != (1, 1, VOCAB):
                raise ValueError("Native full readout output changed")
            count = int(valid.sum())
            values = vocabulary[0, 0, self._metadata["label_ids"][:count]].float().to(query.device)
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite reference anchor logits")
            output = torch.zeros(mask.shape[1], device=query.device, dtype=torch.float32)
            output[:count] = values
            outputs.append(output)
        return torch.stack(outputs)


def _query_mask(query, mask, hidden_size=HIDDEN):
    if (not isinstance(query, torch.Tensor) or query.dtype != torch.float32 or query.ndim != 2
            or query.shape[1] != hidden_size or not 1 <= query.shape[0] <= MAX_BATCH
            or query.requires_grad or not torch.isfinite(query).all()):
        raise ValueError("Require finite owned FP32 query features")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2
            or mask.device != query.device or mask.shape[0] != query.shape[0]
            or not 2 <= mask.shape[1] <= 16):
        raise ValueError("Invalid candidate mask")
    counts = mask.sum(dim=1)
    prefix = torch.arange(mask.shape[1], device=mask.device)[None] < counts[:, None]
    if (counts < 2).any() or not torch.equal(prefix, mask):
        raise ValueError("Candidates must be a contiguous valid prefix of at least two")


def _inputs(query, candidates, mask, hidden_size=HIDDEN):
    if not isinstance(candidates, torch.Tensor) or candidates.ndim != 3:
        raise ValueError("Invalid candidate feature dimensions")
    if mask is None:
        mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
    _query_mask(query, mask, hidden_size)
    if (candidates.shape != (*mask.shape, hidden_size) or candidates.dtype != torch.float32
            or candidates.device != query.device or candidates.requires_grad
            or not torch.isfinite(candidates).all()):
        raise ValueError("Require finite owned FP32 candidate features")
    return mask


def _bound(value):
    if (type(value) is not float or not math.isfinite(value)
            or not torch.finfo(torch.float32).tiny <= value <= torch.finfo(torch.float32).max / 4):
        raise ValueError("Residual bound must be a positive, representable fixed float")
    return value


def _seed(value):
    if type(value) is not int or not 0 <= value < (1 << 63):
        raise ValueError("Invalid residual initialization seed")
    return value


def fresh_residual(seed, hidden_size=HIDDEN, projection_size=RANK):
    if (type(hidden_size) is not int or type(projection_size) is not int
            or not 1 <= hidden_size <= HIDDEN or not 1 <= projection_size <= RANK):
        raise ValueError("Invalid synthetic/production residual dimensions")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(_seed(seed))
        residual = CandidateHead(hidden_size, projection_size)
    nn.init.zeros_(residual.key.weight)
    nn.init.zeros_(residual.key.bias)
    return residual.float()


def retain_ordinal(head):
    """Copy without reinitializing, anchoring, or changing temperature ownership."""
    if not isinstance(head, CandidateHead) or (head.hidden_size, head.projection_size) != (HIDDEN, RANK):
        raise ValueError("Unsupported retained ordinal architecture")
    return copy.deepcopy(head).requires_grad_(False)


def _cache_rows(rows, primitive, batch, training):
    if not isinstance(rows, (list, tuple)) or len(rows) != batch or not rows:
        raise ValueError("Cached anchor needs aligned owned row metadata")
    roles, ids = set(), set()
    for row in rows:
        if (not isinstance(row, dict) or row.get("source") not in TRAINING_SOURCES
                or row.get("primitive") != primitive or row.get("role") not in {"fit", "development"}
                or row.get("evaluation_scope", "training") != "training"
                or row.get("dataset_purpose", "training") != "training"
                or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids
                or not isinstance(row.get("group_"), str) or not row["group_"]):
            raise ValueError("Cached anchors are FIT/DEV owned training-source controls only")
        roles.add(row["role"])
        ids.add(row["id"])
    if len(roles) != 1 or ("development" in roles and (training or torch.is_grad_enabled())):
        raise ValueError("DEV cached controls may only be used in no-gradient evaluation")
    return next(iter(roles))


class AnchorResidualHead(nn.Module):
    def __init__(self, anchor, *, primitive, residual_bound, seed, cache_feature_identity_sha256=None,
                 hidden_size=HIDDEN, projection_size=RANK):
        super().__init__()
        if not isinstance(anchor, FrozenNativeAnchor) or primitive not in {"choice", "noul"}:
            raise ValueError("Only Choice/Noul may use a frozen reference anchor")
        self.anchor = anchor
        self.primitive = primitive
        self._residual_bound = _bound(residual_bound)
        self.seed = _seed(seed)
        self.cache_feature_identity_sha256 = (None if cache_feature_identity_sha256 is None
                                              else _digest(cache_feature_identity_sha256))
        # Smaller dimensions are only for synthetic cached CPU checks, not artifacts/runtime.
        self.residual = fresh_residual(seed, hidden_size, projection_size)

    @property
    def residual_bound(self):
        return self._residual_bound

    def residual_logits(self, query, candidates, mask=None):
        mask = _inputs(query, candidates, mask, self.residual.hidden_size)
        if any(parameter.dtype != torch.float32 or parameter.device != query.device
               for parameter in self.residual.parameters()):
            raise ValueError("Residual parameters must remain FP32 on the feature device")
        raw = self.residual(query, candidates)
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite raw residual logits")
        mean = raw.masked_fill(~mask, 0).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        centered = raw - mean
        if not torch.isfinite(centered).all():
            raise ValueError("Nonfinite centered residual logits")
        bounded = self.residual_bound * torch.tanh(centered / self.residual_bound)
        return bounded.masked_fill(~mask, -torch.inf)

    def forward(self, query, candidates, mask=None):
        if (self.residual.hidden_size, self.residual.projection_size) != (HIDDEN, RANK):
            raise ValueError("Synthetic residual dimensions may not reach the native runtime")
        mask = _inputs(query, candidates, mask)
        residual = self.residual_logits(query, candidates, mask)
        anchor = self.anchor.logits(query, mask)
        return self._combine(anchor, residual, mask)

    def _combine(self, baseline, residual, mask):
        if (not isinstance(baseline, torch.Tensor) or baseline.shape != mask.shape
                or baseline.dtype != torch.float32 or baseline.device != mask.device
                or baseline.requires_grad or not torch.isfinite(baseline).all()):
            raise ValueError("Missing or invalid detached real cached/reference anchor")
        logits = (baseline + residual).masked_fill(~mask, -torch.inf)
        if not torch.isfinite(logits[mask]).all():
            raise ValueError("Nonfinite combined logits")
        return logits

    def forward_cached(self, query, candidates, mask, *, baseline, rows,
                       anchor_sha256, feature_identity_sha256, producer_identity_sha256):
        """Explicit offline FIT/DEV path; callers must prove cache-to-query binding."""
        mask = _inputs(query, candidates, mask, self.residual.hidden_size)
        _cache_rows(rows, self.primitive, len(query), self.training)
        if (_digest(anchor_sha256) != self.anchor.weights_sha256
                or _digest(producer_identity_sha256) != self.anchor.producer_identity_sha256
                or self.cache_feature_identity_sha256 is None
                or _digest(feature_identity_sha256) != self.cache_feature_identity_sha256):
            raise ValueError("Cached controls lack the pinned reference/corpus identity")
        return self._combine(baseline, self.residual_logits(query, candidates, mask), mask)

    def residual_sha256(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.residual.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def margin_diagnostics(self, query, candidates, mask=None):
        mask = _inputs(query, candidates, mask, self.residual.hidden_size)
        delta = self.residual_logits(query, candidates, mask).masked_fill(~mask, 0)
        centered = (delta - delta.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)).masked_fill(~mask, 0)
        margins = delta.masked_fill(~mask, -torch.inf).max(dim=1).values - delta.masked_fill(~mask, torch.inf).min(dim=1).values
        return {"residual_bound": self.residual_bound, "maximum_pairwise_bound": 2 * self.residual_bound,
                "valid_candidates": int(mask.sum()), "nonzero_centered_candidates": int((centered[mask] != 0).sum()),
                "centered_rms": float(centered[mask].double().square().mean().sqrt()),
                "max_abs_residual": float(delta[mask].abs().max()), "max_candidate_margin": float(margins.max())}

    def gradient_diagnostics(self):
        result = {}
        for name, parameter in self.residual.named_parameters():
            gradient = parameter.grad
            result[name] = {"present": gradient is not None,
                            "finite": gradient is None or bool(torch.isfinite(gradient).all()),
                            "nonzero": gradient is not None and bool((gradient != 0).any())}
        return result

    def require_learned_residual(self, initial_sha256, *, optimizer_steps, supervised_gradient_steps,
                                 query, candidates, mask, rows):
        _digest(initial_sha256)
        _cache_rows(rows, self.primitive, len(query), False)
        current = self.residual_sha256()
        margins = self.margin_diagnostics(query, candidates, mask)
        if (type(optimizer_steps) is not int or type(supervised_gradient_steps) is not int
                or not 0 < supervised_gradient_steps <= optimizer_steps
                or current == initial_sha256 or margins["max_candidate_margin"] == 0
                or any(not item["finite"] for item in self.gradient_diagnostics().values())):
            raise ValueError("No effective supervised residual evidence; control-only is not repair")
        return {"initial_residual_sha256": initial_sha256, "residual_sha256": current,
                "optimizer_steps": optimizer_steps, "supervised_gradient_steps": supervised_gradient_steps,
                "margins": margins, "quality_pass": False, "dev_anchor_ablation_required": True,
                "full_corpus_proof_required": True}


def _state_spec():
    return {"query.weight": {"shape": [RANK, HIDDEN], "dtype": "F32"},
            "query.bias": {"shape": [RANK], "dtype": "F32"},
            "key.weight": {"shape": [RANK, HIDDEN], "dtype": "F32"},
            "key.bias": {"shape": [RANK], "dtype": "F32"}}


def save_head(directory, head):
    """Exclusive, self-contained scaffold export, not a calibrated serving bundle."""
    if not isinstance(head, AnchorResidualHead):
        raise ValueError("Expected an anchored Choice/Noul head")
    if (head.residual.hidden_size, head.residual.projection_size) != (HIDDEN, RANK):
        raise ValueError("Synthetic residuals may not be published as production artifacts")
    state = {key: value.detach().cpu().contiguous() for key, value in head.residual.state_dict().items()}
    _check_tensors(state, _state_spec())
    # Recheck the source before copying instead of trusting a previously loaded object.
    anchor = load_anchor(head.anchor.directory, manifest_sha256=head.anchor.manifest_sha256)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("anchor.json", "anchor.safetensors"):
        shutil.copyfile(anchor.directory / name, directory / name)
    copied = load_anchor(directory, manifest_sha256=anchor.manifest_sha256)
    weights = directory / "residual.safetensors"
    save_file(state, str(weights))
    metadata = {"schema_version": 1, "architecture": HEAD_PROFILE, "artifact_kind": "unqualified-head-scaffold",
                "feature_tap": TAP, "hidden_size": HIDDEN, "projection_size": RANK, "primitive": head.primitive,
                "residual_bound": head.residual_bound, "residual_formula": FORMULA, "seed": head.seed,
                "anchor_coefficient": 1.0, "anchor_manifest_sha256": copied.manifest_sha256,
                "anchor_sha256": copied.weights_sha256, "weights_sha256": _sha(weights, STATE_LIMIT),
                "cache_feature_identity_sha256": head.cache_feature_identity_sha256}
    return _write_json(directory / "head.json", metadata)


def load_head(directory, *, manifest_sha256):
    directory = Path(directory)
    metadata = _read_json(directory / "head.json", manifest_sha256)
    expected = {"schema_version": 1, "architecture": HEAD_PROFILE, "artifact_kind": "unqualified-head-scaffold",
                "feature_tap": TAP, "hidden_size": HIDDEN, "projection_size": RANK,
                "residual_formula": FORMULA, "anchor_coefficient": 1.0}
    if (set(metadata) != set(expected) | {"primitive", "residual_bound", "seed", "anchor_manifest_sha256",
                                         "anchor_sha256", "weights_sha256", "cache_feature_identity_sha256"}
            or any(metadata.get(key) != value for key, value in expected.items())):
        raise ValueError("Unsupported residual scaffold profile")
    if (any(type(metadata[key]) is not int for key in ("schema_version", "hidden_size", "projection_size"))
            or type(metadata["anchor_coefficient"]) is not float):
        raise ValueError("Invalid numeric residual metadata")
    anchor = load_anchor(directory, manifest_sha256=metadata["anchor_manifest_sha256"])
    if anchor.weights_sha256 != metadata["anchor_sha256"]:
        raise ValueError("Residual refers to a different frozen reference")
    _inspect_file(directory / "residual.safetensors", metadata["weights_sha256"], STATE_LIMIT, _state_spec())
    head = AnchorResidualHead(anchor, primitive=metadata["primitive"], residual_bound=metadata["residual_bound"],
                              seed=metadata["seed"], cache_feature_identity_sha256=metadata["cache_feature_identity_sha256"])
    with safe_open(str(directory / "residual.safetensors"), framework="pt", device="cpu") as handle:
        head.residual.load_state_dict({key: handle.get_tensor(key) for key in handle.keys()}, strict=True)
    return head.eval()
