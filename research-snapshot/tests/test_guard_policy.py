"""CPU policy/receipt fixtures only; never execute the operational wrapper."""

from copy import deepcopy
from hashlib import sha256
import subprocess
import sys
from threading import Event
from types import SimpleNamespace as NS

import pytest

from fuzev_research import telemetry
from fuzev_research.guard_policy import verify_memory_guard


DIGEST = sha256(b"synthetic CPU policy fixture, not operator execution").hexdigest()


def test_policy_modules_import_without_torch_or_native_bootstrap():
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = sys.modules['exllamav3'] = None
from fuzev_research.guard_policy import verify_memory_guard
from fuzev_research.native_qualification import run_qualification
from fuzev_research.native_state_observer import NativeStateObserver
"""], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


class NoMemoryGuard:
    enabled = False
    user_approved_oom_risk = True
    operator_sha256 = DIGEST

    def __init__(self, fraction=.85):
        self.fraction, self.thread = fraction, None
        self.checks, self.closed = 0, False

    def start(self):
        return self

    def check(self):
        self.checks += 1

    def snapshot(self):
        return {"enabled": self.enabled, "user_approved_oom_risk": self.user_approved_oom_risk,
                "operator_sha256": self.operator_sha256, "policy": "explicit CPU fixture removal",
                "scope": "host RAM headroom and sustained swap watchdog"}

    def close(self):
        self.closed = True


@pytest.fixture
def removed(monkeypatch):
    monkeypatch.setattr(telemetry, "MemoryGuard", NoMemoryGuard)
    return NoMemoryGuard()


def test_explicit_removed_policy_never_fabricates_thread(removed):
    policy = verify_memory_guard(removed, .85)
    assert policy == removed.snapshot()
    assert policy["enabled"] is False and policy["user_approved_oom_risk"] is True
    assert removed.thread is None and removed.checks == 0


def test_removed_class_cannot_pretend_to_be_active(removed):
    removed.enabled, removed.thread, removed.stop = True, object(), Event()
    with pytest.raises(ValueError, match="cannot claim an active"):
        verify_memory_guard(removed, .85)


@pytest.mark.parametrize("key,value", [
    ("user_approved_oom_risk", False), ("user_approved_oom_risk", 1),
    ("user_approved_oom_risk", "yes"), ("operator_sha256", None),
    ("operator_sha256", "a" * 63), ("operator_sha256", "A" * 64),
    ("operator_sha256", "g" * 64), ("operator_sha256", "a" * 64 + "\n"),
    ("operator_sha256", 42), ("thread", object()), ("enabled", None),
    ("enabled", 0), ("fraction", .9), ("fraction", 0), ("fraction", True),
])
def test_ambiguous_removed_policy_rejected(removed, key, value):
    setattr(removed, key, value)
    with pytest.raises(ValueError):
        verify_memory_guard(removed, .85)


@pytest.mark.parametrize("key,value", [("enabled", True), ("user_approved_oom_risk", False),
                                      ("operator_sha256", "b" * 64), ("policy", ""), ("scope", None)])
def test_snapshot_must_match_approved_policy(removed, key, value):
    snapshot = removed.snapshot()
    snapshot[key] = value
    removed.snapshot = lambda: snapshot
    with pytest.raises(ValueError):
        verify_memory_guard(removed, .85)


def test_foreign_guard_or_subclass_not_admitted(removed):
    class Foreign(NoMemoryGuard):
        pass

    for guard in (Foreign(), NS(**vars(removed))):
        with pytest.raises(ValueError, match="running shared-memory"):
            verify_memory_guard(guard, .85)


def test_matching_alias_without_explicit_no_memory_class_rejected(monkeypatch):
    class Unnamed(NoMemoryGuard):
        pass

    monkeypatch.setattr(telemetry, "MemoryGuard", Unnamed)
    with pytest.raises(ValueError, match="explicit NoMemoryGuard"):
        verify_memory_guard(Unnamed(), .85)


def test_normal_guard_checks_unchanged_without_starting_watchdog():
    guard = telemetry.MemoryGuard()
    with pytest.raises(ValueError, match="running shared-memory"):
        verify_memory_guard(guard, .85)
    # Symbolic thread field tests the original contract; no monitor is spawned.
    guard.thread = object()
    assert verify_memory_guard(guard, .85)["enabled"] is True
    guard.stop.set()
    with pytest.raises(ValueError, match="running shared-memory"):
        verify_memory_guard(guard, .85)


@pytest.mark.parametrize("fraction", [0, 1.1, True, float("nan"), float("inf")])
def test_removed_policy_does_not_relax_declared_fraction(removed, fraction):
    with pytest.raises(ValueError):
        verify_memory_guard(removed, fraction)


def observer_fixture(guard):
    from fuzev_research.native_state_observer import NativeStateObserver

    model = NS(loaded_tp=False)
    cache = NS(model=model, initialized=True, max_num_tokens=4096, num_slots=4)
    lane = NS(model=model, cache=cache, context_limit=262144, chunk_size=8192)
    runtime = {"memory_fraction": .85, "context_tokens": 262144, "cache_tokens": 4096,
               "max_hybrid_slots": 4, "feature_chunk_tokens": 8192}
    admissions = []
    service = NS(lane=lane, backend=NS(cache=cache, readout_runtime=runtime), guard=guard,
                 _owner=lambda: None, admission=lambda *args: admissions.append(args))
    # Exercise only policy plumbing, not native bindings or tensor observation.
    observer = object.__new__(NativeStateObserver)
    observer.service, observer.lane, observer.cache = service, lane, cache
    observer.cpu_fake, observer.atol, observer.rtol, observer.tile_elements = False, 1e-5, 1e-4, 4096
    observer._policy = (False, 1e-5, 1e-4, 4096)
    observer.metadata = {"source_verification": "synthetic_guard_plumbing_not_native_evidence"}
    return observer, admissions


def test_observer_records_removed_policy_and_rejects_drift(removed):
    observer, admissions = observer_fixture(removed)
    observer._guard()
    assert observer.metadata["memory_guard_policy"]["enabled"] is False
    assert admissions == [(4096 * 128, .85)] and removed.checks == 1
    removed.operator_sha256 = "b" * 64
    with pytest.raises(ValueError, match="policy changed"):
        observer._guard()
    assert len(admissions) == 1


def test_observer_normal_guard_error_still_precedes_admission():
    guard = telemetry.MemoryGuard()
    guard.thread = object()
    guard.error = MemoryError("synthetic latched guard error")
    observer, admissions = observer_fixture(guard)
    with pytest.raises(MemoryError, match="synthetic latched"):
        observer._guard()
    assert admissions == []


def qualification_fixture(monkeypatch, guard):
    from fuzev_research import native_qualification as qualification

    runtime = {"memory_fraction": .85, "context_tokens": 262144, "feature_chunk_tokens": 8192}
    service = NS(_owner=lambda: None, prefixes={}, model_name="cpu-policy-fixture", guard=guard,
                 backend=NS(prefixes={}, images={}, readout_runtime=runtime),
                 lane=NS(branches=set(), context_limit=262144, chunk_size=8192))
    # No source/model/head evidence is generated: these are receipt-only stand-ins.
    monkeypatch.setattr(qualification, "MemoryGuard", NoMemoryGuard)
    monkeypatch.setattr(qualification, "admit", lambda *_: None)
    monkeypatch.setattr(qualification, "_metadata", lambda *_: {"synthetic_guard_plumbing": True})
    monkeypatch.setattr(qualification, "_resources", lambda *_: {})
    monkeypatch.setattr(qualification, "_frozen_head", lambda *_: {})
    monkeypatch.setattr(qualification, "_unchanged_head", lambda *_: None)
    monkeypatch.setattr(qualification, "_case", lambda *_, **__: {"boundary": "length", "branches": [{"primitive": "choice"}]})
    case = {"name": "guard_policy", "input_origin": "synthetic policy plumbing only",
            "input_role": "operator-non-heldout", "generation": {"model": service.model_name, "prompt": "A"},
            "decisions": [{"primitive": "choice", "question": "Q", "candidates": [
                {"id": "a", "text": "A"}, {"id": "b", "text": "B"}]} for _ in range(4)]}
    return qualification, service, {"cases": [case]}


def test_qualification_reports_both_removed_guards_without_qualifying(removed, monkeypatch):
    qualification, service, config = qualification_fixture(monkeypatch, removed)
    report = qualification.run_qualification(service, config, allow_inference=True)
    assert report["qualified"] is False and report["execution_status"] == "passed", report
    assert report["metadata"] == {"synthetic_guard_plumbing": True}
    for key in ("service_memory_guard", "memory_guard"):
        assert report[key]["enabled"] is False
        assert report[key]["operator_sha256"] == DIGEST
        assert report[key]["user_approved_oom_risk"] is True
    assert removed.thread is None and not removed.closed


def test_qualification_rejects_missing_approval_before_case(removed, monkeypatch):
    qualification, service, config = qualification_fixture(monkeypatch, removed)
    removed.user_approved_oom_risk = False
    monkeypatch.setattr(qualification, "_case", lambda *_: pytest.fail("Policy failure must precede case work"))
    with pytest.raises(ValueError, match="user-approved"):
        qualification.run_qualification(service, config, allow_inference=True)


def test_qualification_policy_drift_cannot_pass(removed, monkeypatch):
    qualification, service, config = qualification_fixture(monkeypatch, removed)

    def drift(*_, **__):
        removed.operator_sha256 = "b" * 64
        return {"boundary": "length", "branches": [{"primitive": "choice"}]}

    monkeypatch.setattr(qualification, "_case", drift)
    before = deepcopy(config)
    report = qualification.run_qualification(service, config, allow_inference=True)
    assert config == before and report["qualified"] is False
    assert report["status"] == "failed" and report["execution_status"] == "failed"
    assert any("policy changed" in error for error in report["cleanup_errors"])


def full_window_config(config):
    config = deepcopy(config)
    case = deepcopy(config["cases"][0])
    case["name"] = "full_window_policy"
    config.update(run_full_window=True, full_window={"case": case, "additional_peak_bytes": 20,
                                                   "estimate_basis": "arbitrary synthetic RAM values, not a GPU estimate"})
    return config


def test_full_window_arithmetic_respects_operator_headroom():
    from fuzev_research.native_qualification import full_window_feasibility

    result = full_window_feasibility(NS(additional_peak_bytes=20, estimate_basis="CPU arithmetic only"),
                                     enabled=True, frozen_head_ready=True, total=100, available=30, fraction=.85)
    assert result["status"] == "memory_infeasible" and result["reason"] == "would_breach_shared_headroom"
    assert result["allocated"] is False


def test_removed_host_reserve_is_reported_but_does_not_block_full_window(removed, monkeypatch):
    import psutil

    qualification, service, config = qualification_fixture(monkeypatch, removed)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: NS(total=100, available=30))
    calls = []

    def case(*_, full_window=False, **__):
        calls.append(full_window)
        return {"boundary": "length", "branches": [{"primitive": "choice"}]}

    monkeypatch.setattr(qualification, "_case", case)
    report = qualification.run_qualification(service, full_window_config(config), allow_inference=True)
    full = report["gates"]["full_window"]
    assert calls == [False, True] and report["qualified"] is False
    assert full["host_ram_admission"] == {"enabled": False, "operator_sha256": DIGEST,
                                          "estimate_status": "memory_infeasible",
                                          "estimate_reason": "would_breach_shared_headroom"}
    assert full["allocation_attempted"] is True and full["status"] == "passed"
    assert report["metadata"] == {"synthetic_guard_plumbing": True}


@pytest.mark.parametrize("failure", [MemoryError("physical page plan cannot fit"),
                                     MemoryError("no hybrid-state slots"), ValueError("context exceeds 262144")])
def test_removed_host_policy_does_not_override_physical_failure(removed, monkeypatch, failure):
    import psutil

    qualification, service, config = qualification_fixture(monkeypatch, removed)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: NS(total=100, available=30))

    def case(*_, full_window=False, **__):
        if full_window:
            raise failure
        return {"boundary": "length", "branches": [{"primitive": "choice"}]}

    monkeypatch.setattr(qualification, "_case", case)
    report = qualification.run_qualification(service, full_window_config(config), allow_inference=True)
    assert report["qualified"] is False and report["execution_status"] != "passed"
    assert report["gates"]["full_window"]["status"] in {"memory_infeasible", "failed"}


def test_removed_policy_does_not_override_cpu_fake_frozen_head_gate(removed, monkeypatch):
    qualification, service, config = qualification_fixture(monkeypatch, removed)
    config = full_window_config(config)
    config["cases"][0]["input_role"] = config["full_window"]["case"]["input_role"] = "cpu-fake"
    report = qualification.run_qualification(service, config, allow_inference=True, cpu_fake=True)
    assert report["qualified"] is False and report["evidence_kind"] == "cpu_fake"
    assert report["gates"]["full_window"]["status"] == "blocked"
    assert report["gates"]["full_window"]["reason"] == "frozen_head_not_ready"
