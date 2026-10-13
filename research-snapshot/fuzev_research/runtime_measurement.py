"""Research-only fused-path measurements; never a correctness qualification.

The existing correctness result stays pinned and visible, including failures.
Stock controls and full-window feasibility are separate required experiments.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .acquire import write_json
from .runtime_harness import Harness, HarnessConfig, LIMITATIONS, require


class EvidencePin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def read(self):
        require(self.path.is_absolute(), "Evidence path must be absolute")
        raw = self.path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.sha256,
                "Existing correctness evidence changed")
        return json.loads(raw)


class MeasurementConfig(HarnessConfig):
    expected_engine_memory_fraction: float | None = Field(default=None, gt=0, le=1)
    correctness_evidence: EvidencePin
    warmup_waves: int = Field(default=1, ge=1, le=2)
    repetitions: int = Field(default=3, ge=3, le=16)


def allocation_fraction(health, engine):
    if engine == "native":
        return health.get("memory_fraction")
    return health.get("provenance", {}).get("runtime_cell", {}).get("memory_fraction")


async def run_measurement(config, output, *, transport=None, port=8000):
    evidence = config.correctness_evidence.read()
    require(isinstance(evidence, dict) and evidence.get("status") in {
        "passed", "passed-scoped-checks", "completed", "complete", "failed"},
        "Pin a terminal correctness report, not an unfinished run")
    require(evidence.get("model") == config.model, "Correctness evidence belongs to another model")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.model_dump(mode="json"))
    report = {
        "schema_version": 1, "status": "in_progress", "runtime_qualified": False,
        "scope": "fused-path research measurement; stock comparison and full-window fit remain separate",
        "correctness_evidence": config.correctness_evidence.model_dump(mode="json"),
        "existing_correctness_status": evidence["status"],
        "limitations": LIMITATIONS,
        "warmup": [], "performance": [],
        "expected_engine_memory_fraction": config.expected_engine_memory_fraction,
    }
    write_json(output / "report.json", report)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", transport=transport,
                                 timeout=config.timeout_seconds, follow_redirects=False) as client:
        harness = Harness(client, config, output)
        try:
            health = await harness.preflight()
            require(config.expected_engine_memory_fraction is None or allocation_fraction(health, config.engine) == config.expected_engine_memory_fraction,
                    "Actual engine allocation fraction differs; allocator ceiling is not the KV target")
            require(health.get("active_engine_requests", health.get("active_requests", 0)) == 0,
                    "Do not benchmark an occupied inference engine")
            report["health"] = health
            for workload in config.workloads:
                for mode in ("generation-only", "decision-only", "mixed"):
                    for streams in config.streams:
                        for warmup in range(config.warmup_waves):
                            tasks = [asyncio.create_task(harness.pipeline(workload, mode, index))
                                     for index in range(streams)]
                            try:
                                rows = await asyncio.gather(*tasks)
                            finally:
                                for task in tasks:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.gather(*tasks, return_exceptions=True)
                            report["warmup"].append({"workload": workload.name, "mode": mode,
                                                     "streams": streams, "wave": warmup, "samples": rows})
                        report["performance"].append(await harness.performance(workload, mode, streams))
                        write_json(output / "report.json", report)
            final = await harness.preflight()
            require(config.expected_engine_memory_fraction is None or allocation_fraction(final, config.engine) == config.expected_engine_memory_fraction,
                    "Actual engine allocation fraction changed during measurement")
            for key in ("provenance", "sources", "context_window", "retained_prefixes",
                        "retained_tokens", "retained_multimodal_bytes", "active_engine_requests"):
                require(final.get(key) == health.get(key), f"Runtime or ownership ledger changed: {key}")
            report["health_after"] = final
            config.correctness_evidence.read()
            report["status"] = "measurement-complete"
        except BaseException as error:
            report.update(status="cancelled" if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
                          else "failed", error=f"{type(error).__name__}: {error}")
        finally:
            report["cleanup_failures"] = await harness.close()
            if report["cleanup_failures"]:
                report["status"] = "failed"
            write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args()
    if not args.allow_inference:
        parser.error("Explicit inference permission required; never overlap a live validation panel")
    config = MeasurementConfig.model_validate_json(args.config.read_text())
    report = asyncio.run(run_measurement(config, args.output, port=args.port))
    print(json.dumps({"status": report["status"], "runtime_qualified": False}))
    raise SystemExit(0 if report["status"] == "measurement-complete" else 1)


if __name__ == "__main__":
    main()
