"""Bounded read-only fitting-row collation comparison; no fitting or GPU work."""

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch

from fuzev_research.acquire import write_json
from fuzev_research.features import FeatureStore, collate
from fuzev_research.paths import ROOT, project_path
from fuzev_research.telemetry import MemoryGuard, admit


def main():
    if torch.cuda.is_available():
        raise RuntimeError("Hide CUDA devices for this CPU-only comparison")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    admit(1 << 30)
    original = Path(os.environ["FUZEV_CONTROL_SOURCE"]).expanduser().resolve()
    revision = "035184be283b0bc30b69b38022eeb73b11a5f9bc"  # unpublished lab commit ID
    control_digest = "f09c5b2e6dbff9710bfcfbdb6ce5218eb96724031ab8dfb2f08d7c63f2564187"
    observed_control_digest = hashlib.sha256((original / "fuzev_research/features.py").read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("fuzev_research._original_features", original / "fuzev_research/features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = project_path("data/features/native-4.05")
    identity = json.loads((directory / "identity.json").read_text())
    guard = MemoryGuard().start()
    control = candidate = None
    try:
        control = module.FeatureStore(directory, identity, read_only=True)
        candidate = FeatureStore(directory, identity, read_only=True)
        # Small row locators only. Never select heldouts or load private prompts.
        groups = {}
        cursor = control.db.execute("SELECT id,shard,row_,role FROM records "
                                    "WHERE role='fit' AND row_<4 ORDER BY shard,row_")
        for row in cursor:
            shard = row["shard"]
            if shard not in groups and len(groups) == 32:
                break
            groups.setdefault(shard, []).append(dict(row))
        cursor.close()
        rows = [row for values in groups.values() for row in values]
        if len(groups) != 32 or not rows or any(row["role"] != "fit" for row in rows):
            raise ValueError("Need 32 committed fitting shards for the bounded comparison")
        random.Random(17).shuffle(rows)
        guard.check()
        expected = module.collate(control, rows)
        actual = collate(candidate, rows)
        if expected.keys() != actual.keys() or any(not torch.equal(expected[key], actual[key]) for key in expected):
            raise ValueError("Candidate collation differs from the original training control")
        del actual, expected
        measurements = {arm: {phase: [] for phase in ("mask_width", "collation")}
                        for arm in ("control", "candidate")}
        stores = {"control": control, "candidate": candidate}
        for _ in range(3):
            for arm in ("control", "candidate", "candidate", "control"):
                store = stores[arm]
                guard.check()
                started = time.perf_counter()
                width = store.mask_width(rows)
                measurements[arm]["mask_width"].append(time.perf_counter() - started)
                started = time.perf_counter()
                result = (module.collate if arm == "control" else collate)(store, rows)
                measurements[arm]["collation"].append(time.perf_counter() - started)
                if result["candidates"].shape[1] != width:
                    raise ValueError("Mask width and collated tensor width disagree")
                del result
                guard.check()
        medians = {arm: {phase: statistics.median(values) for phase, values in phases.items()}
                   for arm, phases in measurements.items()}
        report = {"scope": "warm CPU4 collation only; not extraction or training throughput",
                  "observed": time.time(), "rows": len(rows), "shards": len(groups), "role": "fit",
                  "row_locator_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                  "feature_identity_sha256": hashlib.sha256((directory / "identity.json").read_bytes()).hexdigest(),
                  "control_revision": revision,
                  "control_source_sha256": observed_control_digest, "recorded_control_source_sha256": control_digest,
                  "candidate_source_sha256": hashlib.sha256(Path(inspect.getfile(FeatureStore)).read_bytes()).hexdigest(),
                  "torch": torch.__version__, "threads": 4, "batch_tensors_exact": True,
                  "cuda_initialized": torch.cuda.is_initialized(), "measurements_seconds": measurements,
                  "medians_seconds": medians,
                  "warm_collation_speed_ratio": medians["control"]["collation"] / medians["candidate"]["collation"],
                  "descriptor_payload_bytes": candidate._mask_descriptor_bytes,
                  "feature_files_modified": False}
        write_json(project_path("runs/cpu-mask-descriptor-comparison.json"), report)
        print(json.dumps(report, allow_nan=False))
    finally:
        if control is not None:
            control.close()
        if candidate is not None:
            candidate.close()
        guard.close()


if __name__ == "__main__":
    main()
