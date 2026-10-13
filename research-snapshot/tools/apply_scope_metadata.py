"""One-time, reversible metadata handoff; raw datasets and old features stay intact."""

import json
import os
from pathlib import Path
import subprocess
import time

import httpx

from fuzev_research.acquire import dataset_purpose, pin_all, update_manifest, write_json
from fuzev_research.features import file_sha256
from fuzev_research.paths import project_path
from fuzev_research.scope import EXTERNAL_SOURCES, TRAINING_SOURCES, contract
from fuzev_research.state import Store, TASKS


def main():
    receipt_path = project_path("runs/dataset-scope-v2/before.json")
    if receipt_path.exists():
        raise ValueError("Scope metadata handoff already exists; inspect it rather than rerunning")
    handle = json.loads(project_path("runs/native-real/extraction-job.json").read_text())
    trainer = json.loads(project_path("runs/head-training-container.json").read_text())
    progress = json.loads(project_path("runs/head-training.json").read_text())
    with httpx.Client(timeout=30) as client:
        response = client.get(handle["base_url"].rstrip("/") + "/v1/jobs/" + handle["id"])
        response.raise_for_status()
        job = response.json()
    if job.get("id") != handle["id"] or job.get("status") != "cancelled":
        raise ValueError("Safely cancel and drain the exact old feature job before changing scope")
    state = json.loads(subprocess.check_output(["docker", "inspect", "--format", "{{json .State}}", trainer["container_id"]], text=True))
    if state["Running"] or state["OOMKilled"] or progress.get("stage") is not None or any(
            progress.get(name + "_checkpoint") for name in ("text", "mixed")):
        raise ValueError("Require a stopped, unfitted old CPU supervisor")
    archived = project_path("data/archive/scope-before-v2")
    archived.mkdir(parents=True, exist_ok=True)
    backups = []
    for relative in ("data/prepared/qasper.jsonl", "data/prepared/images.jsonl",
                     "data/manifests/qasper-splits.json", "data/manifests/image-splits.json",
                     "data/manifests/acquisition.json"):
        source = project_path(relative)
        target = archived / ("prepared" if "prepared" in relative else "manifests") / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("An archive target already exists; refuse to overwrite history")
        os.link(source, target)
        backups.append({"source": str(source), "archive": str(target), "sha256": file_sha256(target)})
    saved = {"observed": time.time(), "old_job": job, "old_handle": handle, "old_trainer": trainer,
             "old_progress": progress, "old_trainer_state": state, "derived_backups": backups,
             "new_scope": contract(), "raw_datasets_deleted": False, "safety_guards_changed": False,
             "status": "preparing_revised_scope"}
    write_json(receipt_path, saved)
    pin_all()

    def revise(manifest):
        manifest["dataset_scope"] = contract()
        for name, info in manifest["datasets"].items():
            info["active"] = name in TRAINING_SOURCES + EXTERNAL_SOURCES
            info["purpose"] = dataset_purpose(name) if info["active"] else "archived"
        if not manifest["datasets"]["boolq"].get("revision"):
            raise ValueError("BoolQ acquisition pin unresolved")
    update_manifest(revise)
    store = Store()
    store.initialize()
    with store.connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS archived_jobs (id TEXT PRIMARY KEY, retired REAL, reason TEXT, payload TEXT)")
        for row in db.execute("SELECT * FROM jobs WHERE validator='json:data/manifests/qasper-splits.json'").fetchall():
            db.execute("INSERT INTO archived_jobs VALUES(?,?,?,?)", (row["id"], time.time(), "scope-v2: QASPER external-only", json.dumps(dict(row))))
            db.execute("DELETE FROM jobs WHERE id=?", (row["id"],))
        for key, _, title in TASKS:
            if key in {"data.splits.qasper", "data.splits.images"}:
                db.execute("UPDATE tasks SET title=? WHERE id=?", (title, key))
    store.update("data.scope", "in_progress", "Revising active sources; 115,021 compatible HelpSteer features preserved. Raw data and old corpus retained.", 0)
    store.update("data.features", "pending", "Safely drained for the dataset-scope transition; no old QASPER or A-OKVQA processing will resume.", 0)
    store.update("training.text", "pending", "Waiting for revised HelpSteer/BoolQ readiness; external examples cannot select or calibrate heads.", 0)
    store.update("data.splits.qasper", "pending", "Prepare only the original-test 239/1450/2000 external QASPER sample.", 0)
    store.update("data.splits.images", "pending", "Rebuild MM-RLHF training and paired VL heldouts without A-OKVQA.", 0)
    write_json(project_path("data/manifests/active-scope.json"), contract())
    print(json.dumps({"receipt": str(receipt_path), "raw_datasets_deleted": False, "scope": contract()["revision"]}))


if __name__ == "__main__":
    main()
