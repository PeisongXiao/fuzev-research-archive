import subprocess
import time
from threading import Event, RLock, Thread

import psutil


class Sampler:
    def __init__(self):
        psutil.cpu_percent(percpu=True)
        self.last_swap = psutil.swap_memory()
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.device = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def gpu(self):
        result = {"utilization": None, "memory_used": None, "memory_total": None, "temperature": None}
        if self.nvml:
            queries = [("utilization", lambda: self.nvml.nvmlDeviceGetUtilizationRates(self.device).gpu),
                       ("temperature", lambda: self.nvml.nvmlDeviceGetTemperature(self.device, 0)),
                       ("memory_used", lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.device).used),
                       ("memory_total", lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.device).total)]
            for key, query in queries:
                try:
                    result[key] = query()
                except Exception:
                    pass
        else:
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=.8)
                values = output.stdout.splitlines()[0].split(",")
                result["utilization"], result["temperature"] = map(float, values)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
        return result

    def sample(self):
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        result = {"timestamp": time.time(), "cpu": psutil.cpu_percent(percpu=True),
                  "ram_total": memory.total, "ram_available": memory.available,
                  "ram_used": memory.total - memory.available, "swap_used": swap.used,
                  "swap_in_delta": max(0, swap.sin - self.last_swap.sin),
                  "swap_out_delta": max(0, swap.sout - self.last_swap.sout),
                  "gpu": self.gpu(), "load": psutil.getloadavg()}
        self.last_swap = swap
        return result


def admit(required_bytes=0, fraction=None):
    memory = psutil.virtual_memory()
    if memory.total - memory.available + required_bytes > memory.total * (fraction if fraction is not None else 1):
        raise MemoryError("Admission rejected: operation would consume the configured host headroom")


class MemoryGuard:
    """Enforce headroom and consecutive swap pressure, not isolated reclaim bursts."""

    def __init__(self, fraction=None, interval=1., sustained_seconds=3., swap_rate=1 << 20):
        self.fraction, self.interval = fraction, interval
        self.sustained_seconds, self.swap_rate = sustained_seconds, swap_rate
        self.lock, self.stop = RLock(), Event()
        self.previous, self.high_seconds, self.error, self.thread = None, 0., None, None
        self.observation = {}

    def observe(self, memory, swap_out, now):
        with self.lock:
            elapsed = now - self.previous[0] if self.previous else 0.
            rate = max(0, swap_out - self.previous[1]) / elapsed if elapsed > 0 else 0.
            continuous = 0 < elapsed <= self.interval * 2.5
            self.high_seconds = self.high_seconds + elapsed if continuous and rate >= self.swap_rate else 0.
            self.previous = (now, swap_out)
            self.observation = {"ram_available": memory.available, "ram_total": memory.total,
                                "swap_out_bytes_per_second": rate,
                                "consecutive_high_swap_seconds": self.high_seconds}
            if self.fraction is not None and memory.available < memory.total * (1 - self.fraction):
                self.error = MemoryError("Shared RAM fell below the configured headroom")
            elif self.high_seconds >= self.sustained_seconds:
                self.error = MemoryError("Sustained swap-out pressure exceeded the memory-safety guard")

    def start(self):
        if self.thread is not None:
            raise ValueError("Memory guard is already started")
        self.observe(psutil.virtual_memory(), psutil.swap_memory().sout, time.monotonic())
        self.check()
        self.thread = Thread(target=self._watch, name="fuzev-memory-guard", daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        while not self.stop.wait(self.interval):
            try:
                self.observe(psutil.virtual_memory(), psutil.swap_memory().sout, time.monotonic())
            except Exception as error:
                with self.lock:
                    self.error = MemoryError(f"Cannot verify memory safety: {error}")
            if self.error is not None:
                return

    def check(self):
        with self.lock:
            if self.error is not None:
                raise self.error

    def snapshot(self):
        with self.lock:
            return dict(self.observation)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
