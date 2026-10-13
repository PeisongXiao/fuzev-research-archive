"""CPU-only metadata fixtures, not CUDA completion or throughput evidence."""

from types import SimpleNamespace

import pytest

from fuzev_research.runtime_performance_counts import PerformanceObservationWorkerExtension, counter_delta


def snapshot(submitted=0, completed=None, *, failed=0, pending=0, errors=0):
    completed = submitted - failed - pending if completed is None else completed
    return {"scope": "fixture owner-wide positions", "submitted_token_positions": submitted,
            "host_returned_token_positions": submitted - failed,
            "completion_acknowledged_token_positions": completed,
            "failed_forward_token_positions": failed, "pending_token_positions": pending,
            "completion_errors": errors,
            "completion_counts_lower_bound": bool(failed or pending or errors)}


def test_complete_exclusive_wave_is_not_per_request_or_logical_work():
    result = counter_delta(snapshot(10), snapshot(42))
    assert result["physical_forwarded_tokens"] == 32
    assert result["status"] == "completion-acknowledged"
    assert "owner-wide" in result["per_request_attribution"]


@pytest.mark.parametrize("before,after", [
    (snapshot(), snapshot(10, pending=2)),
    (snapshot(), snapshot(10, failed=2)),
    (snapshot(10, pending=2), snapshot(20)),
    (snapshot(), snapshot(10, errors=1)),
])
def test_incomplete_boundary_does_not_become_physical_total(before, after):
    result = counter_delta(before, after)
    assert result["physical_forwarded_tokens"] is None
    assert result["status"] == "incomplete-lower-bound"


@pytest.mark.parametrize("before,after", [
    (None, snapshot()),
    (snapshot(20), snapshot(10)),
    (snapshot(), {**snapshot(10), "scope": "another scope"}),
    (snapshot(), {**snapshot(10), "submitted_token_positions": True}),
    (snapshot(), {**snapshot(10), "host_returned_token_positions": 9}),
    (snapshot(), {**snapshot(10), "pending_token_positions": 1}),
])
def test_disabled_reset_or_inconsistent_counts_refused(before, after):
    with pytest.raises(ValueError):
        counter_delta(before, after)


def test_rpc_is_named_and_defaults_to_observation_only():
    extension = PerformanceObservationWorkerExtension()
    calls = []
    extension.model_runner = SimpleNamespace(
        enable_performance_counters=lambda: calls.append("enable") or snapshot(),
        get_performance_counters=lambda: calls.append("read") or None)
    assert extension.fuzev_performance_counts() is None
    assert calls == ["read"]
    assert extension.fuzev_performance_counts(enable=True) == snapshot()
    with pytest.raises(ValueError):
        extension.fuzev_performance_counts(enable=1)
