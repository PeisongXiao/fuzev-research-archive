"""Trusted offline worker RPC for opt-in, owner-wide forward observations.

No model load or request endpoint. The operator must exclusively own the engine;
enable once while idle, after warmup construction and before measured waves.
"""

from .kev_vllm_capture import RuntimeObservationWorkerExtension


class PerformanceObservationWorkerExtension(RuntimeObservationWorkerExtension):
    def fuzev_performance_counts(self, *, enable=False):
        if type(enable) is not bool:
            raise ValueError("Explicit boolean counter policy required")
        if enable:
            return self.model_runner.enable_performance_counters()
        return self.model_runner.get_performance_counters()


def counter_delta(before, after):
    """Whole exclusive wave only; shared forwards cannot be split per request."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Genuine enabled counter snapshots required")
    if before.get("scope") != after.get("scope") or not before.get("scope"):
        raise ValueError("Counter observation scope changed")
    keys = ("submitted_token_positions", "host_returned_token_positions",
            "completion_acknowledged_token_positions", "failed_forward_token_positions",
            "completion_errors")
    delta = {}
    for key in keys:
        a, b = before.get(key), after.get(key)
        if type(a) is not int or type(b) is not int or not 0 <= a <= b:
            raise ValueError("Counter reset, invalid count or mixed owner: " + key)
        delta[key] = b - a
    for snapshot in (before, after):
        pending = snapshot.get("pending_token_positions")
        if type(pending) is not int or pending < 0:
            raise ValueError("Invalid pending forward-position count")
        if type(snapshot.get("completion_counts_lower_bound")) is not bool:
            raise ValueError("Explicit completion-count status required")
        if not (snapshot["completion_acknowledged_token_positions"]
                <= snapshot["host_returned_token_positions"] <= snapshot["submitted_token_positions"]
                and snapshot["submitted_token_positions"] ==
                snapshot["completion_acknowledged_token_positions"]
                + snapshot["failed_forward_token_positions"] + pending):
            raise ValueError("Contradictory counter ownership totals")
    complete = (before["pending_token_positions"] == after["pending_token_positions"] == 0
                and not before["completion_counts_lower_bound"]
                and not after["completion_counts_lower_bound"]
                and delta["failed_forward_token_positions"] == delta["completion_errors"] == 0
                and delta["submitted_token_positions"] == delta["completion_acknowledged_token_positions"])
    return {"scope": after["scope"], **delta,
            "genuinely_acknowledged_forwarded_positions": delta["completion_acknowledged_token_positions"],
            "physical_forwarded_tokens": delta["completion_acknowledged_token_positions"] if complete else None,
            "status": "completion-acknowledged" if complete else "incomplete-lower-bound",
            "per_request_attribution": "unavailable; shared work is owner-wide"}
