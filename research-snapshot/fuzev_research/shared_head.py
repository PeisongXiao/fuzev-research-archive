"""One learned scorer assisted by supplied frozen LM scores; no LM is loaded."""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save_file
from torch import nn

from .head import CandidateHead


ARCHITECTURE = "shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1"
SCHEMA_VERSION = 25
PRIMITIVES = ("choice", "noul", "ordinal")
HIDDEN_SIZE = 2560
PROJECTION_SIZE = 512
ADAPTER_SIZE = 64
KEY_ADAPTER_SIZE = 16
KEY_ADAPTER_PARENT_COMMIT = "da1a527d18d1e053e3a72da31fc0cbd72258fcda"  # unpublished lab commit ID
KEY_ADAPTER_REFERENCE_COMMIT = "6721adaf03b0598839c2d59c6059632912bc883a"  # unpublished lab commit ID
CANDIDATE_GATE_INPUT_SCALE = 1.
STATE_TENSOR_COUNT = 15
PARAMETER_COUNT = (2 * PROJECTION_SIZE * (HIDDEN_SIZE + 1) + HIDDEN_SIZE + 1
                   + ADAPTER_SIZE * (HIDDEN_SIZE + 1) + PROJECTION_SIZE * ADAPTER_SIZE
                   + 3 * PROJECTION_SIZE + KEY_ADAPTER_SIZE * (HIDDEN_SIZE + 1 + PROJECTION_SIZE))
FEATURE_TAP = "final-mixer/post-collapse/before-lm-head"
WEIGHT_LIMIT = 64 << 20
JSON_LIMIT = 1 << 20
CODE_FILES = {"shared_train.py", "shared_head.py", "shared_training_data.py",
              "features.py", "metrics.py", "scope.py", "preference_head_retrain.py", "head.py"}
INITIALIZATION = "xavier-query-small-key-zero-context-gate-zero-up-query-adapter64-zero-channel-gate-zero-up-key-adapter16-v1"
ASSISTANCE_FORMULA = "baseline.detach()+center_valid(raw)+(2*sigmoid(gate(query))-1)*center_valid(baseline.detach())"
PROJECTION_FORMULA = "u=Wq*q+bq+U*SiLU(V*q+d); v_i=Wk*k_i+bk+Uk*SiLU(Vk*k_i+dk)"
RAW_FORMULA = "sum_j(u_j*v_ij*2*sigmoid(A_j*u_j+B_j*v_ij+c_j))/sqrt(projection_size)"
EVALUATION_FORMULA = "einsum('bd,bkd->bk',u,v)/sqrt(projection_size)+sum_j(u_j*v_ij*(2*sigmoid(A_j*u_j+B_j*v_ij+c_j)-1))/sqrt(projection_size)"
PARAMETER_SHARING = "one-query-key-scorer-with-shared-query-and-key-adapters-channel-gates-and-query-context-gate"


def _key_adapter():
    return {"rank": KEY_ADAPTER_SIZE, "activation": "SiLU", "input": "candidate-only",
            "output": "additive-key-projection", "down_bias": True, "up_bias": False,
            "down_gain": 1., "initial_down_bias": 0., "initial_up_weight": 0.,
            "construction": "after-all-A41-parent-RNG-draws", "parent_commit": KEY_ADAPTER_PARENT_COMMIT,
            "source_only_reference_commit": KEY_ADAPTER_REFERENCE_COMMIT}


def _validate_key_adapter(value):
    if (not isinstance(value, dict) or value != _key_adapter()
            or type(value.get("rank")) is not int or value.get("down_bias") is not True
            or value.get("up_bias") is not False
            or any(type(value.get(key)) is not float for key in
                   ("down_gain", "initial_down_bias", "initial_up_weight"))):
        raise ValueError("Incompatible shared key-adapter metadata or parent provenance")


def _key_adapter_geometry(bundle):
    down, up = getattr(bundle, "key_adapter_down", None), getattr(bundle, "key_adapter_up", None)
    if (not isinstance(down, nn.Linear) or not isinstance(up, nn.Linear)
            or (down.in_features, down.out_features) != (bundle.hidden_size, KEY_ADAPTER_SIZE)
            or (up.in_features, up.out_features) != (KEY_ADAPTER_SIZE, bundle.projection_size)
            or down.weight.shape != (KEY_ADAPTER_SIZE, bundle.hidden_size)
            or down.bias is None or down.bias.shape != (KEY_ADAPTER_SIZE,)
            or up.weight.shape != (bundle.projection_size, KEY_ADAPTER_SIZE) or up.bias is not None):
        raise ValueError("Require one shared rank-16 key adapter with bias-free up projection")


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < (1 << 63):
        raise ValueError("Expected an explicit nonnegative shared-scorer seed")
    return seed


def _gain(gain):
    if (type(gain) is not float or not math.isfinite(gain)
            or not torch.finfo(torch.float32).tiny <= gain <= 1.):
        raise ValueError("Key Xavier gain must be a small positive finite float")
    return gain


def _assistance():
    return {"source": "existing-frozen-lm-head", "scores": "native4-owned-cached-label-controls",
            "lm_head_trained": False, "extra_lm_copy": False,
            "formula": ASSISTANCE_FORMULA, "bounded": False,
            "raw_formula": RAW_FORMULA,
            "projection_formula": PROJECTION_FORMULA, "evaluation_formula": EVALUATION_FORMULA,
            "candidate_gate_input_scale": CANDIDATE_GATE_INPUT_SCALE,
            "query_adapter_size": ADAPTER_SIZE,
            "key_adapter_size": KEY_ADAPTER_SIZE,
            "context_trust": "query-only linear; 2*sigmoid; shared across candidates/primitives/modalities",
            "centering": "valid-candidate mean only"}


def _initialization(value):
    if (not isinstance(value, dict)
            or set(value) != {"scheme", "seed", "query_gain", "key_gain", "gate_weight", "gate_bias",
                              "adapter_down_gain", "adapter_down_bias", "adapter_up_weight", "adapter_up_bias",
                              "channel_a", "channel_b", "channel_c", "initial_channel_multiplier",
                              "parent_function_preserved", "query_adapter_size", "key_adapter_size",
                              "key_adapter_down_gain", "key_adapter_down_bias", "key_adapter_up_weight",
                              "key_adapter_up_bias"}
            or value["scheme"] != INITIALIZATION
            or type(value["query_gain"]) is not float or value["query_gain"] != 1.
            or type(value["adapter_down_gain"]) is not float or value["adapter_down_gain"] != 1.
            or value["adapter_up_bias"] is not False
            or value["parent_function_preserved"] is not True
            or type(value["query_adapter_size"]) is not int or value["query_adapter_size"] != ADAPTER_SIZE
            or type(value["key_adapter_size"]) is not int or value["key_adapter_size"] != KEY_ADAPTER_SIZE
            or type(value["key_adapter_down_gain"]) is not float or value["key_adapter_down_gain"] != 1.
            or value["key_adapter_up_bias"] is not False
            or type(value["initial_channel_multiplier"]) is not float or value["initial_channel_multiplier"] != 1.
            or any(type(value[key]) is not float or value[key] != 0.
                   for key in ("gate_weight", "gate_bias", "adapter_down_bias", "adapter_up_weight",
                               "channel_a", "channel_b", "channel_c", "key_adapter_down_bias", "key_adapter_up_weight"))):
        raise ValueError("Unknown shared-scorer initialization provenance")
    _seed(value["seed"])
    _gain(value["key_gain"])


def _validate_assistance(value):
    if (not isinstance(value, dict) or value != _assistance()
            or value.get("lm_head_trained") is not False or value.get("extra_lm_copy") is not False
            or value.get("bounded") is not False
            or type(value.get("candidate_gate_input_scale")) is not float
            or type(value.get("query_adapter_size")) is not int or type(value.get("key_adapter_size")) is not int):
        raise ValueError("Incompatible frozen LM assistance formula/source/profile")


def _adapter_geometry(bundle):
    down, up = getattr(bundle, "adapter_down", None), getattr(bundle, "adapter_up", None)
    if (not isinstance(down, nn.Linear) or not isinstance(up, nn.Linear)
            or (down.in_features, down.out_features) != (bundle.hidden_size, ADAPTER_SIZE)
            or (up.in_features, up.out_features) != (ADAPTER_SIZE, bundle.projection_size)
            or down.weight.shape != (ADAPTER_SIZE, bundle.hidden_size)
            or down.bias is None or down.bias.shape != (ADAPTER_SIZE,)
            or up.weight.shape != (bundle.projection_size, ADAPTER_SIZE) or up.bias is not None):
        raise ValueError("Require one shared rank-64 query adapter with bias-free up projection")


def _channel_geometry(bundle):
    for name in ("channel_a", "channel_b", "channel_c"):
        value = getattr(bundle, name, None)
        if not isinstance(value, nn.Parameter) or value.shape != (bundle.projection_size,):
            raise ValueError("Require exactly three shared channel-gate vectors")


class SharedDecisionHead(nn.Module):
    requires_lm_scores = True

    def __init__(self, *, seed, hidden_size=HIDDEN_SIZE, projection_size=PROJECTION_SIZE, key_gain=1e-3):
        super().__init__()
        if (type(hidden_size) is not int or type(projection_size) is not int
                or not 1 <= hidden_size <= HIDDEN_SIZE or not 1 <= projection_size <= PROJECTION_SIZE):
            raise ValueError("Invalid scorer dimensions")
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.seed = _seed(seed)
        self.key_gain = _gain(key_gain)
        # CPU initialization is reproducible without touching a CUDA RNG or global RNG state.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(self.seed)
            self.scorer = CandidateHead(hidden_size, projection_size).float()
            nn.init.xavier_uniform_(self.scorer.query.weight, gain=1.)
            nn.init.xavier_uniform_(self.scorer.key.weight, gain=self.key_gain)
            nn.init.zeros_(self.scorer.query.bias)
            nn.init.zeros_(self.scorer.key.bias)
            # Construct after Q/K initialization to preserve the exact R12 scorer draws.
            self.gate = nn.Linear(hidden_size, 1).float()
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            # Append to the parent RNG sequence; zero U preserves its exact starting function.
            self.adapter_down = nn.Linear(hidden_size, ADAPTER_SIZE).float()
            self.adapter_up = nn.Linear(ADAPTER_SIZE, projection_size, bias=False).float()
            nn.init.xavier_uniform_(self.adapter_down.weight, gain=1.)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            # Zero channel gates preserve the remaining common parent tensors without RNG draws.
            self.channel_a = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
            self.channel_b = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
            self.channel_c = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
            # Append after every A41 draw; zero Uk preserves scores, not the new Jacobian.
            self.key_adapter_down = nn.Linear(hidden_size, KEY_ADAPTER_SIZE).float()
            self.key_adapter_up = nn.Linear(KEY_ADAPTER_SIZE, projection_size, bias=False).float()
            nn.init.xavier_uniform_(self.key_adapter_down.weight, gain=1.)
            nn.init.zeros_(self.key_adapter_down.bias)
            nn.init.zeros_(self.key_adapter_up.weight)

    @property
    def initialization(self):
        return {"scheme": INITIALIZATION, "seed": self.seed, "query_gain": 1., "key_gain": self.key_gain,
                "gate_weight": 0., "gate_bias": 0., "adapter_down_gain": 1., "adapter_down_bias": 0.,
                "adapter_up_weight": 0., "adapter_up_bias": False,
                "channel_a": 0., "channel_b": 0., "channel_c": 0.,
                "initial_channel_multiplier": 1., "parent_function_preserved": True,
                "query_adapter_size": ADAPTER_SIZE, "key_adapter_size": KEY_ADAPTER_SIZE,
                "key_adapter_down_gain": 1., "key_adapter_down_bias": 0.,
                "key_adapter_up_weight": 0., "key_adapter_up_bias": False}

    @property
    def lm_assistance(self):
        return _assistance()

    @property
    def key_adapter(self):
        return _key_adapter()

    def forward(self, primitive, query, candidates, mask=None, *, baseline):
        if primitive not in PRIMITIVES:
            raise ValueError("Unknown shared-scoring primitive")
        if (not isinstance(query, torch.Tensor) or not isinstance(candidates, torch.Tensor)
                or query.ndim != 2 or candidates.ndim != 3
                or query.shape != (candidates.shape[0], self.hidden_size)
                or candidates.shape[-1] != self.hidden_size or query.shape[0] < 1 or candidates.shape[1] < 1
                or query.dtype != torch.float32 or candidates.dtype != torch.float32
                or query.device != candidates.device):
            raise ValueError("Invalid FP32 query/candidate features")
        if not torch.isfinite(query).all() or not torch.isfinite(candidates).all():
            raise ValueError("Nonfinite shared-scorer features")
        if mask is not None:
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != candidates.shape[:2] or mask.device != query.device
                    or not mask.any(dim=1).all()):
                raise ValueError("Invalid option mask or entirely masked decision")
        if (not isinstance(baseline, torch.Tensor) or baseline.dtype != torch.float32
                or baseline.shape != candidates.shape[:2] or baseline.device != query.device
                or baseline.requires_grad or not torch.isfinite(baseline).all()):
            raise ValueError("Require finite, aligned FP32 frozen LM baseline scores")
        if any(parameter.dtype != torch.float32 or parameter.device != query.device
               for parameter in self.parameters()):
            raise ValueError("Shared scorer parameters must remain FP32 on the feature device")
        if (not isinstance(self.gate, nn.Linear) or self.gate.weight.shape != (1, self.hidden_size)
                or self.gate.bias is None or self.gate.bias.shape != (1,)):
            raise ValueError("Require one query-only scalar context gate")
        _adapter_geometry(self)
        _channel_geometry(self)
        _key_adapter_geometry(self)
        hidden = torch.nn.functional.silu(self.adapter_down(query))
        projected_query = self.scorer.query(query) + self.adapter_up(hidden)
        if not torch.isfinite(hidden).all() or not torch.isfinite(projected_query).all():
            raise ValueError("Nonfinite query-adapter features")
        key_hidden = torch.nn.functional.silu(self.key_adapter_down(candidates))
        projected_keys = self.scorer.key(candidates) + self.key_adapter_up(key_hidden)
        if not torch.isfinite(key_hidden).all() or not torch.isfinite(projected_keys).all():
            raise ValueError("Nonfinite key-adapter features")
        channel_logits = (self.channel_a * projected_query[:, None, :]
                          + self.channel_b * projected_keys + self.channel_c)
        if not torch.isfinite(projected_keys).all() or not torch.isfinite(channel_logits).all():
            raise ValueError("Nonfinite channel-gate features or logits")
        # Keep the parent's FP32 reduction and add an exactly zero residual at initialization.
        base_raw = torch.einsum("bd,bkd->bk", projected_query, projected_keys) / math.sqrt(self.projection_size)
        channel_delta = (projected_query[:, None, :] * projected_keys
                         * (2 * torch.sigmoid(channel_logits) - 1)).sum(dim=-1) / math.sqrt(self.projection_size)
        raw = base_raw + channel_delta
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite shared-scorer logits")
        valid = torch.ones_like(raw, dtype=torch.bool) if mask is None else mask
        counts = valid.sum(dim=1, keepdim=True)
        raw = raw.masked_fill(~valid, 0)
        mean = raw.sum(dim=1, keepdim=True) / counts
        delta = (raw - mean).masked_fill(~valid, 0)
        base = baseline.detach().masked_fill(~valid, 0)
        base_mean = base.sum(dim=1, keepdim=True) / counts
        centered_base = (base - base_mean).masked_fill(~valid, 0)
        gate_logits = self.gate(query)
        if not torch.isfinite(gate_logits).all():
            raise ValueError("Nonfinite context-gate logits")
        trust = 2 * torch.sigmoid(gate_logits)
        logits = base + delta + (trust - 1) * centered_base
        if (not torch.isfinite(mean).all() or not torch.isfinite(base_mean).all()
                or not torch.isfinite(logits).all()):
            raise ValueError("Nonfinite LM-assisted decision logits")
        return logits if mask is None else logits.masked_fill(~mask, -torch.inf)


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("Invalid shared-head provenance checksum")
    return value


def _membership(value):
    if (not isinstance(value, dict) or type(value.get("records")) is not int or value["records"] <= 0):
        raise ValueError("Missing nonempty shared-head role membership")
    _sha(value.get("ordered_sha256"))


def _code(value):
    if not isinstance(value, dict) or set(value) != CODE_FILES:
        raise ValueError("Missing actual shared training/calibration code provenance")
    for digest in value.values():
        _sha(digest)


def _provenance(metadata):
    from .composite_head import COMPOSITE_MARKERS

    if COMPOSITE_MARKERS.intersection(metadata):
        raise ValueError("Independent composite provenance cannot describe one shared scorer")
    if any(key in metadata for key in ("anchor_sha256", "anchor_manifest_sha256", "per_head_weights", "residual_bound")):
        raise ValueError("Anchor/residual/per-head provenance is not a shared decision scorer")
    if (type(metadata.get("learned_head_count", 1)) is not int or metadata.get("learned_head_count", 1) != 1
            or metadata.get("frozen_lm_head_trained", False) is not False):
        raise ValueError("Require one learned scorer and an untrained existing LM head")
    if metadata.get("artifact_kind") not in {"training-checkpoint", "calibrated-reference-head"}:
        raise ValueError("An explicit training or calibrated-reference qualifier is required")
    if metadata.get("stage") not in {"text", "mixed"}:
        raise ValueError("Missing shared-head training stage")
    for field in ("feature_identity", "preparation"):
        if not isinstance(metadata.get(field), dict) or not metadata[field]:
            raise ValueError("Missing shared-head feature/preparation provenance")
    for field in ("fit_membership", "development_membership"):
        _membership(metadata.get(field))
    _code(metadata.get("training_code"))
    _code(metadata.get("shared_training_code"))
    if metadata["training_code"] != metadata["shared_training_code"]:
        raise ValueError("Shared training source pins disagree")
    _sha(metadata.get("operator_sha256"))
    if metadata["operator_sha256"] != metadata["training_code"]["shared_train.py"]:
        raise ValueError("Shared operator checksum does not match its training source")
    if metadata["feature_identity"].get("dataset_scope") is not None:
        if not isinstance(metadata.get("feature_origins"), dict) or not metadata["feature_origins"]:
            raise ValueError("Missing scoped feature producer origins")
    if metadata["artifact_kind"] == "calibrated-reference-head":
        if metadata["stage"] != "mixed" or metadata.get("calibration_role") != "calibration":
            raise ValueError("Reference qualifier requires mixed training and CAL-only calibration")
        _membership(metadata.get("calibration_membership"))
        _code(metadata.get("calibration_code"))
    elif any(key in metadata for key in ("calibration_role", "calibration_code", "calibration_membership")):
        raise ValueError("A training checkpoint cannot claim fitted calibration")


def _temperatures(values):
    if not isinstance(values, dict) or set(values) != set(PRIMITIVES):
        raise ValueError("Require one typed temperature for each primitive")
    if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("Primitive temperatures must be positive finite numbers, not booleans")
    return {key: float(values[key]) for key in PRIMITIVES}


def _bytes(path, limit, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Missing, symlinked or oversized shared-head artifact")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Shared-head artifact exceeds its resource bound")
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != _sha(expected_sha256):
        raise ValueError("Shared-head checksum mismatch")
    return data


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate shared-head JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Nonfinite shared-head JSON constant: " + value)


def _json(data):
    value = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("Shared-head metadata must be an object")
    return value


def _json_bytes(value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(data) > JSON_LIMIT:
        raise ValueError("Shared-head metadata exceeds its resource bound")
    return data


def _state(state):
    shapes = {"scorer.query.weight": (PROJECTION_SIZE, HIDDEN_SIZE),
              "scorer.query.bias": (PROJECTION_SIZE,),
              "scorer.key.weight": (PROJECTION_SIZE, HIDDEN_SIZE),
              "scorer.key.bias": (PROJECTION_SIZE,),
              "gate.weight": (1, HIDDEN_SIZE), "gate.bias": (1,),
              "adapter_down.weight": (ADAPTER_SIZE, HIDDEN_SIZE), "adapter_down.bias": (ADAPTER_SIZE,),
              "adapter_up.weight": (PROJECTION_SIZE, ADAPTER_SIZE),
              "channel_a": (PROJECTION_SIZE,), "channel_b": (PROJECTION_SIZE,),
              "channel_c": (PROJECTION_SIZE,),
              "key_adapter_down.weight": (KEY_ADAPTER_SIZE, HIDDEN_SIZE),
              "key_adapter_down.bias": (KEY_ADAPTER_SIZE,),
              "key_adapter_up.weight": (PROJECTION_SIZE, KEY_ADAPTER_SIZE)}
    if set(state) != set(shapes):
        raise ValueError("Require exactly fifteen parameters: one query/key scorer, query/key adapters, context gate and channel gates")
    if any(value.dtype != torch.float32 or tuple(value.shape) != shapes[key]
           or not torch.isfinite(value).all() for key, value in state.items()):
        raise ValueError("Invalid FP32 shared-scorer state shape or nonfinite weights")


def _profile(metadata, expected_feature_tap):
    expected = {"schema_version": SCHEMA_VERSION, "architecture": ARCHITECTURE,
                "hidden_size": HIDDEN_SIZE, "projection_size": PROJECTION_SIZE,
                "query_adapter_size": ADAPTER_SIZE,
                "key_adapter_size": KEY_ADAPTER_SIZE,
                "candidate_gate_input_scale": CANDIDATE_GATE_INPUT_SCALE,
                "state_tensor_count": STATE_TENSOR_COUNT, "parameter_count": PARAMETER_COUNT,
                "raw_formula": RAW_FORMULA, "projection_formula": PROJECTION_FORMULA,
                "evaluation_formula": EVALUATION_FORMULA,
                "precision": "float32", "feature_tap": FEATURE_TAP,
                "primitives": list(PRIMITIVES), "parameter_sharing": PARAMETER_SHARING}
    if (expected_feature_tap != FEATURE_TAP
            or any(metadata.get(key) != value for key, value in expected.items())
            or any(type(metadata.get(key)) is not int
                   for key in ("schema_version", "hidden_size", "projection_size", "query_adapter_size", "key_adapter_size",
                               "state_tensor_count", "parameter_count"))
            or type(metadata.get("candidate_gate_input_scale")) is not float):
        raise ValueError("Incompatible shared-head architecture, geometry or feature tap")
    initialization = metadata.get("initialization")
    _initialization(initialization)
    _validate_assistance(metadata.get("lm_assistance"))
    _validate_key_adapter(metadata.get("key_adapter"))
    if metadata.get("requires_lm_scores") is not True:
        raise ValueError("Incompatible frozen LM assistance formula/source/profile")
    for field in ("weights_sha256", "calibration_sha256", "shared_head_code_sha256"):
        _sha(metadata.get(field))
    _provenance(metadata)
    if metadata["shared_head_code_sha256"] != metadata["training_code"]["shared_head.py"]:
        raise ValueError("Scorer implementation checksum disagrees with training provenance")
    if "seed" in metadata and (type(metadata["seed"]) is not int or metadata["seed"] != initialization["seed"]):
        raise ValueError("Shared initialization seed disagrees with caller provenance")


def save_shared_bundle(directory, bundle, metadata, temperatures):
    """Serialize a checkpoint or caller-qualified reference; never invent qualification."""
    if not isinstance(bundle, SharedDecisionHead) or (bundle.hidden_size, bundle.projection_size) != (
            HIDDEN_SIZE, PROJECTION_SIZE):
        raise ValueError("Synthetic geometry may not be published as a shared-head artifact")
    if not isinstance(metadata, dict):
        raise ValueError("Require explicit shared-head provenance")
    metadata = copy.deepcopy(metadata)
    _provenance(metadata)
    if "initialization" in metadata:
        _initialization(metadata["initialization"])
    if "lm_assistance" in metadata:
        _validate_assistance(metadata["lm_assistance"])
    if "key_adapter" in metadata:
        _validate_key_adapter(metadata["key_adapter"])
    if (any(key in metadata and type(metadata[key]) is not int
            for key in ("schema_version", "hidden_size", "projection_size", "query_adapter_size", "key_adapter_size",
                        "state_tensor_count", "parameter_count"))
            or ("candidate_gate_input_scale" in metadata and type(metadata["candidate_gate_input_scale"]) is not float)
            or ("requires_lm_scores" in metadata and metadata["requires_lm_scores"] is not True)):
        raise ValueError("Caller provenance disagrees with the actual shared scorer")
    calibration = {"schema_version": SCHEMA_VERSION, "architecture": ARCHITECTURE,
                   "query_adapter_size": ADAPTER_SIZE,
                   "key_adapter_size": KEY_ADAPTER_SIZE,
                   "candidate_gate_input_scale": CANDIDATE_GATE_INPUT_SCALE,
                   "method": "positive-temperature-per-primitive", "temperatures": _temperatures(temperatures)}
    calibration_data = _json_bytes(calibration)
    profile = {"schema_version": SCHEMA_VERSION, "architecture": ARCHITECTURE, "feature_tap": FEATURE_TAP,
               "hidden_size": HIDDEN_SIZE, "projection_size": PROJECTION_SIZE, "precision": "float32",
               "query_adapter_size": ADAPTER_SIZE,
               "key_adapter_size": KEY_ADAPTER_SIZE, "key_adapter": bundle.key_adapter,
               "candidate_gate_input_scale": CANDIDATE_GATE_INPUT_SCALE,
               "state_tensor_count": STATE_TENSOR_COUNT, "parameter_count": PARAMETER_COUNT,
               "raw_formula": RAW_FORMULA, "projection_formula": PROJECTION_FORMULA,
               "evaluation_formula": EVALUATION_FORMULA,
               "primitives": list(PRIMITIVES), "parameter_sharing": PARAMETER_SHARING,
               "initialization": bundle.initialization,
               "requires_lm_scores": True, "lm_assistance": bundle.lm_assistance,
               "shared_head_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "calibration_sha256": hashlib.sha256(calibration_data).hexdigest()}
    if any(key in metadata and metadata[key] != value for key, value in profile.items()):
        raise ValueError("Caller provenance disagrees with the actual shared scorer")
    if "seed" in metadata and (type(metadata["seed"]) is not int or metadata["seed"] != bundle.seed):
        raise ValueError("Caller seed differs from shared initialization")
    _adapter_geometry(bundle)
    _channel_geometry(bundle)
    _key_adapter_geometry(bundle)
    state = {key: value.detach().cpu().contiguous() for key, value in bundle.state_dict().items()}
    _state(state)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "head.safetensors"
    save_file(state, str(weights))
    profile["weights_sha256"] = hashlib.sha256(_bytes(weights, WEIGHT_LIMIT)).hexdigest()
    if "weights_sha256" in metadata and metadata["weights_sha256"] != profile["weights_sha256"]:
        raise ValueError("Caller supplied a different shared weight identity")
    manifest = {**metadata, **profile}
    _profile(manifest, FEATURE_TAP)
    with (directory / "calibration.json").open("xb") as handle:
        handle.write(calibration_data)
    with (directory / "manifest.json").open("xb") as handle:
        handle.write(_json_bytes(manifest))
    return manifest


def load_shared_bundle(directory, expected_feature_tap=FEATURE_TAP, *,
                       expected_manifest_sha256=None, expected_calibration_sha256=None):
    """Bounded artifact validation only; evaluator corpus/readiness proof is still mandatory."""
    directory = Path(directory)
    manifest = _json(_bytes(directory / "manifest.json", JSON_LIMIT, expected_manifest_sha256))
    _profile(manifest, expected_feature_tap)
    calibration_data = _bytes(directory / "calibration.json", JSON_LIMIT, manifest["calibration_sha256"])
    if expected_calibration_sha256 is not None:
        if hashlib.sha256(calibration_data).hexdigest() != _sha(expected_calibration_sha256):
            raise ValueError("Shared-head pinned calibration checksum mismatch")
    calibration = _json(calibration_data)
    if (set(calibration) != {"schema_version", "architecture", "query_adapter_size", "key_adapter_size", "candidate_gate_input_scale", "method", "temperatures"}
            or type(calibration.get("schema_version")) is not int or calibration["schema_version"] != SCHEMA_VERSION
            or calibration.get("architecture") != ARCHITECTURE
            or type(calibration.get("query_adapter_size")) is not int or calibration["query_adapter_size"] != ADAPTER_SIZE
            or type(calibration.get("key_adapter_size")) is not int or calibration["key_adapter_size"] != KEY_ADAPTER_SIZE
            or type(calibration.get("candidate_gate_input_scale")) is not float
            or calibration["candidate_gate_input_scale"] != CANDIDATE_GATE_INPUT_SCALE
            or calibration.get("method") != "positive-temperature-per-primitive"):
        raise ValueError("Incompatible typed primitive calibration")
    temperatures = _temperatures(calibration["temperatures"])
    try:
        state = load(_bytes(directory / "head.safetensors", WEIGHT_LIMIT, manifest["weights_sha256"]))
    except SafetensorError as exc:
        raise ValueError("Invalid shared-head safetensors artifact") from exc
    _state(state)
    initialization = manifest["initialization"]
    bundle = SharedDecisionHead(seed=initialization["seed"], key_gain=initialization["key_gain"])
    bundle.load_state_dict(state, strict=True)
    _key_adapter_geometry(bundle)
    return bundle.eval(), manifest, temperatures
