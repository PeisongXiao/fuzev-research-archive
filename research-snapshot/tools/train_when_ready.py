"""CPU-only full-corpus head fitting with five-minute completion polling."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from fuzev_research.acquire import write_json
from fuzev_research.features import FeatureStore
from fuzev_research.paths import project_path
from fuzev_research.state import Store
from fuzev_research.telemetry import MemoryGuard, admit
from fuzev_research.train import calibrate, fit
from fuzev_research.scope import required_preparations


class GuardedReader(FeatureStore):
    def __init__(self, *args, guard, **kwargs):
        self.guard = guard
        super().__init__(*args, **kwargs)

    def batch_tensors(self, rows):
        self.guard.check()
        yield from super().batch_tensors(rows)
        self.guard.check()


def retry_readiness_change(features, stage, operation):
    while True:
        features.guard.check()
        try:
            return operation()
        except ValueError as error:
            if str(error) != "Preparation corpus changed during readiness verification; retry":
                raise
            print(json.dumps({"stage": stage, "status": "waiting_for_stable_corpus",
                              "reason": str(error), "poll_interval_seconds": 300}), flush=True)
            time.sleep(300)


def wait_for_corpus(features, stage):
    required = (set(required_preparations(stage)) if getattr(features, "scoped", False) else
                {"text.jsonl", "qasper.jsonl"} | ({"images.jsonl"} if stage != "text" else set()))
    while True:
        features.guard.check()
        complete = {Path(row["path"]).name for row in features.preparations()
                    if row["completed"]}
        if required <= complete:
            return retry_readiness_change(features, stage, lambda: features.require_ready(stage))
        print(json.dumps({"stage": stage, "status": "waiting_for_full_corpus",
                          "missing": sorted(required - complete),
                          "poll_interval_seconds": 300}), flush=True)
        time.sleep(300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if torch.cuda.is_available():
        raise RuntimeError("Launch this fitting orchestrator without GPU access")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    admit(1 << 30)
    guard = MemoryGuard().start()
    state = Store()
    task = "training.text"
    identity = json.loads((args.features / "identity.json").read_text())
    features = GuardedReader(args.features, identity, read_only=True, guard=guard)
    if not features.scoped:
        features.close()
        guard.close()
        raise RuntimeError("The active CPU supervisor requires the revised dataset scope")
    handle = project_path("runs/head-training.json")
    progress = {"status": "waiting", "device": "cpu", "threads": args.threads,
                "feature_directory": str(args.features.resolve()),
                "feature_identity": identity,
                "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "started": time.time(), "publication": "not_requested"}
    write_json(handle, progress)
    try:
        for stage in ("text", "mixed"):
            task = "training.text" if stage == "text" else "training.multimodal"
            readiness = wait_for_corpus(features, stage)
            progress.update(status="fitting", stage=stage, preparation=readiness)
            write_json(handle, progress)
            selected = retry_readiness_change(
                features, stage, lambda: fit(features, stage, progress.get("text_checkpoint")))
            progress[f"{stage}_checkpoint"] = selected
            write_json(handle, progress)
        task = "training.calibration"
        progress.update(status="calibrating")
        write_json(handle, progress)
        progress["reference"] = retry_readiness_change(
            features, "calibrate", lambda: calibrate(features, progress["mixed_checkpoint"]))
        progress.update(status="completed", finished=time.time())
        write_json(handle, progress)
        print(json.dumps(progress, allow_nan=False), flush=True)
    except BaseException as error:
        progress.update(status="failed", error=str(error), finished=time.time())
        write_json(handle, progress)
        state.update(task, "failed", f"Head fitting stopped: {error}")
        raise
    finally:
        features.close()
        guard.close()


if __name__ == "__main__":
    main()
