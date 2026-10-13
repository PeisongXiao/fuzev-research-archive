from types import SimpleNamespace

import pytest

from fuzev_research.telemetry import MemoryGuard


RAM = SimpleNamespace(total=1000, available=300)


def test_isolated_reclaim_bursts_are_not_sustained_swapping():
    guard = MemoryGuard()
    guard.observe(RAM, 0, 0.)
    guard.observe(RAM, 100 << 20, 1.)
    guard.observe(RAM, 100 << 20, 2.)
    guard.observe(RAM, 200 << 20, 100.)
    guard.check()
    assert guard.snapshot()["consecutive_high_swap_seconds"] == 0.


def test_consecutive_swap_pressure_stops_work_and_remains_latched():
    guard = MemoryGuard()
    for second in range(4):
        guard.observe(RAM, second * (2 << 20), float(second))
    with pytest.raises(MemoryError, match="Sustained"):
        guard.check()
    guard.observe(RAM, 6 << 20, 4.)
    with pytest.raises(MemoryError, match="Sustained"):
        guard.check()


def test_headroom_guard_is_immediate_even_without_swap():
    guard = MemoryGuard(fraction=.85)
    guard.observe(SimpleNamespace(total=1000, available=149), 0, 0.)
    with pytest.raises(MemoryError, match="headroom"):
        guard.check()


def test_guard_thread_stops_cleanly(monkeypatch):
    guard = MemoryGuard(interval=.001)
    monkeypatch.setattr("fuzev_research.telemetry.psutil.virtual_memory", lambda: RAM)
    monkeypatch.setattr("fuzev_research.telemetry.psutil.swap_memory", lambda: SimpleNamespace(sout=0))
    guard.start()
    guard.close()
    assert not guard.thread.is_alive()


def test_sampler_failure_fails_closed(monkeypatch):
    guard = MemoryGuard(interval=.001)
    calls = 0
    def memory():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("telemetry unavailable")
        return RAM
    monkeypatch.setattr("fuzev_research.telemetry.psutil.virtual_memory", memory)
    monkeypatch.setattr("fuzev_research.telemetry.psutil.swap_memory", lambda: SimpleNamespace(sout=0))
    guard.start()
    guard.thread.join(timeout=2)
    try:
        with pytest.raises(MemoryError, match="Cannot verify"):
            guard.check()
    finally:
        guard.close()
