"""Verify a live host guard or an explicitly approved operational removal."""

import math
import re


def verify_memory_guard(guard, fraction, *, message="Native service must retain a running shared-memory guard"):
    from .telemetry import MemoryGuard

    def require(condition, reason=message):
        if not condition:
            raise ValueError(reason)

    require(fraction is None or (type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 1))
    require(type(guard) is MemoryGuard)
    declared = getattr(guard, "fraction", None)
    require(declared is None or (type(declared) in (int, float) and math.isfinite(declared)
                                and 0 < declared <= (fraction if fraction is not None else 1)))
    enabled = getattr(guard, "enabled", True)
    require(type(enabled) is bool, "Host memory-guard enabled flag must be explicit boolean")
    if enabled:
        require(type(guard).__name__ != "NoMemoryGuard", "NoMemoryGuard cannot claim an active host guard")
        stop = getattr(guard, "stop", None)
        require(getattr(guard, "thread", None) is not None and callable(getattr(stop, "is_set", None)) and not stop.is_set())
        return {"enabled": True, "fraction": declared,
                "scope": "host RAM headroom and sustained swap watchdog"}

    require(type(guard).__name__ == "NoMemoryGuard" and getattr(guard, "thread", object()) is None,
            "Disabled host guard must be an explicit NoMemoryGuard without a pretend running thread")
    require(getattr(guard, "user_approved_oom_risk", None) is True,
            "Disabled host guard requires explicit user-approved OOM/host-crash risk")
    digest = getattr(guard, "operator_sha256", None)
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "Disabled host guard requires the operational wrapper SHA256")
    snapshot = guard.snapshot()
    require(isinstance(snapshot, dict) and snapshot.get("enabled") is False
            and snapshot.get("user_approved_oom_risk") is True and snapshot.get("operator_sha256") == digest,
            "Disabled host guard snapshot disagrees with the approved operational policy")
    require(all(isinstance(snapshot.get(key), str) and snapshot[key].strip() for key in ("policy", "scope")),
            "Disabled host guard snapshot must name its policy and scope")
    return {key: snapshot[key] for key in ("enabled", "user_approved_oom_risk", "operator_sha256", "policy", "scope")}
