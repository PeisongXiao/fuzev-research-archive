"""Owned scalar snapshots of successful vLLM worker acknowledgements.

No core imports, schedule/cache inference, or cumulative-stream summation.
These input-position counts are not kernel/FLOP or hybrid-state coverage counts.
"""


from .encoding import token_hash


READOUT_SOURCE_FILES = {
    "vllm/v1/fuzev_readout.py", "vllm/v1/fuzev.py", "vllm/pooling_params.py",
    "vllm/v1/worker/gpu_model_runner.py", "vllm/v1/outputs.py",
    "vllm/v1/engine/__init__.py", "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/output_processor.py", "vllm/outputs.py",
}
READOUT_FIELDS = {
    "schema_version", "worker_request_id", "request_id", "prompt_tokens",
    "prompt_token_sha256", "feature_readout_position", "feature_width",
    "baseline_token_ids", "lm_projection_calls", "generated_token_count",
    "worker_loaded_sources",
}


def readout_sources(value):
    if (not isinstance(value, dict) or set(value) != READOUT_SOURCE_FILES
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(char not in "0123456789abcdef" for char in digest)
                   for digest in value.values())):
        raise ValueError("Invalid loaded FuZev readout source binding")
    return dict(value)


def readout_receipt(output, tokens, request_id, labels, expected_sources):
    """Require worker facts and the actual frontend request-ID mapping."""
    raw = getattr(output, "fuzev_readout_receipt", None)
    if not isinstance(raw, dict) or set(raw) != READOUT_FIELDS:
        raise ValueError("Missing or unsupported worker readout receipt")
    expected = {
        "schema_version": 1, "prompt_tokens": len(tokens),
        "feature_readout_position": len(tokens) - 1, "feature_width": 2560,
        "lm_projection_calls": 1 if labels else 0, "generated_token_count": 0,
    }
    if any(type(raw[name]) is not int or raw[name] != value
           for name, value in expected.items()):
        raise ValueError("Worker readout geometry/projection receipt mismatch")
    if (raw["request_id"] != request_id
            or not isinstance(raw["worker_request_id"], str)
            or not 1 <= len(raw["worker_request_id"]) <= 1024
            or raw["prompt_token_sha256"] != token_hash(tokens)):
        raise ValueError("Worker readout request/prompt binding mismatch")
    actual_labels = raw["baseline_token_ids"]
    if (not isinstance(actual_labels, list) or tuple(actual_labels) != tuple(labels)
            or any(type(token) is not int for token in actual_labels)):
        raise ValueError("Worker readout label order mismatch")
    sources = readout_sources(raw["worker_loaded_sources"])
    if sources != expected_sources:
        raise ValueError("Worker loaded readout sources differ from the pinned frontend")
    return {**raw, "baseline_token_ids": actual_labels.copy(), "worker_loaded_sources": sources}


FORWARD_SCOPE = "worker-acknowledged unpadded target input positions; not kernels/FLOPs or causal coverage"
COVERAGE_GAP = "replay/reuse, causal consumed-prefix and terminal abort/failure totals remain unavailable"
UNAVAILABLE_GAP = "final worker receipt unavailable; no physical total; " + COVERAGE_GAP
FIELDS = (
    "completed_forward_steps", "completed_token_positions",
    "last_completed_start", "last_completed_end", "feature_readout_position",
    "missing_worker_receipts", "replayed_token_positions", "reused_token_positions",
    "causal_consumed_prefix",
)


def _unavailable(reason=UNAVAILABLE_GAP):
    return {
        "fuzev_forward_stats": None, "forwarded_tokens": None,
        "forwarded_tokens_lower_bound": None, "forward_counts_status": "unavailable",
        "feature_readout_position_verified": None, "forward_counter_scope": FORWARD_SCOPE,
        "counter_gap": reason,
    }


def forward_receipt(output, sequence_length, *, features):
    """Copy/validate a final output's scalars, never borrow its receipt object."""
    raw = getattr(output, "fuzev_forward_stats", None)
    if raw is None:
        return _unavailable()
    try:
        stats = {name: getattr(raw, name) for name in FIELDS}
    except AttributeError:
        return _unavailable("unsupported/incomplete worker receipt schema; " + UNAVAILABLE_GAP)
    for name, value in stats.items():
        required = name in {"completed_forward_steps", "completed_token_positions",
                            "missing_worker_receipts"}
        if value is None and not required:
            continue
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid engine worker receipt scalar: {name}")
    steps, positions = stats["completed_forward_steps"], stats["completed_token_positions"]
    start, end = stats["last_completed_start"], stats["last_completed_end"]
    readout, missing = stats["feature_readout_position"], stats["missing_worker_receipts"]
    if steps == 0:
        if positions != 0 or start is not None or end is not None or readout is not None or not missing:
            raise ValueError("Invalid empty final worker receipt")
    elif (start is None or end is None or not 0 <= start < end <= sequence_length
          or not end - start + steps - 1 <= positions <= steps * sequence_length
          or steps == 1 and positions != end - start):
        raise ValueError("Invalid completed worker input-position range/counts")
    if readout is not None:
        if not features or readout != end - 1 or readout != sequence_length - 1:
            raise ValueError("Final feature readout does not match the exact submitted sequence")
    prefix = stats["causal_consumed_prefix"]
    if prefix is not None and prefix > sequence_length:
        raise ValueError("Invalid engine-reported causal prefix bound")
    return {
        "fuzev_forward_stats": stats, "forwarded_tokens": positions,
        "forwarded_tokens_lower_bound": missing > 0,
        "forward_counts_status": "lower-bound" if missing else "worker-acknowledged",
        "feature_readout_position_verified": True if readout is not None else None,
        "forward_counter_scope": FORWARD_SCOPE, "counter_gap": COVERAGE_GAP,
    }


def aggregate_forward_receipts(receipts):
    """Sum distinct successful engine requests only when every snapshot exists."""
    ids = [item["request_id"] for item in receipts]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Forward totals require independent successful request receipts")
    stats = [item["fuzev_forward_stats"] for item in receipts]
    unavailable = sum(value is None for value in stats)
    result = {
        "completed_forward_steps": None, "completed_token_positions": None,
        "missing_worker_receipts": None, "forwarded_tokens": None,
        "forwarded_tokens_lower_bound": None, "forward_counts_status": "unavailable",
        "unavailable_forward_receipts": unavailable, "forward_counter_scope": FORWARD_SCOPE,
        "counter_gap": UNAVAILABLE_GAP if unavailable else COVERAGE_GAP,
    }
    if not unavailable:
        for field in ("completed_forward_steps", "completed_token_positions", "missing_worker_receipts"):
            result[field] = sum(value[field] for value in stats)
        result["forwarded_tokens"] = result["completed_token_positions"]
        result["forwarded_tokens_lower_bound"] = result["missing_worker_receipts"] > 0
        result["forward_counts_status"] = (
            "lower-bound" if result["forwarded_tokens_lower_bound"] else "worker-acknowledged")
    return result
