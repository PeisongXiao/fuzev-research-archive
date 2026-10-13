"""Pin the revised corpus identity or activate its guarded extraction/CPU trainer."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import httpx

from fuzev_research.acquire import write_json
from fuzev_research.features import FeatureStore, file_sha256
from fuzev_research.paths import project_path
from fuzev_research.scope import PREPARATIONS, contract, validate_contract


IMAGE = os.environ.get("FUZEV_RESEARCH_CUDA_IMAGE")
CHECKOUT = Path(__file__).resolve().parents[1]
DIRECTORY = project_path("data/features/native-4.05-scope-v2")


def identity():
    from fuzev_research.native import scoped_feature_identity
    old = json.loads(project_path("data/features/native-4.05/identity.json").read_text())
    value = scoped_feature_identity("native-4.05", old["readout_runtime"])
    write_json(project_path("runs/dataset-scope-v2/new-identity.json"), value)
    print(json.dumps({"sources": value["runtime_sources"], "scope": value["dataset_scope"]["revision"]}))


def activate(port=8000, *, image, trainer_container, runtime_container, cpuset_cpus=None, user=None):
    value = json.loads((DIRECTORY / "identity.json").read_text())
    validate_contract(value["dataset_scope"])
    audit = json.loads(project_path("runs/dataset-scope-v2/preparation-audit.json").read_text())
    if audit["dataset_scope"] != contract():
        raise ValueError("Preparation audit belongs to a different scope")
    for filename, entry in audit["preparations"].items():
        if (file_sha256(project_path("data/prepared/" + filename)) != entry["sha256"]
                or file_sha256(project_path("data/manifests/" + PREPARATIONS[filename][0])) != entry["manifest_sha256"]):
            raise ValueError("Preparations changed after scope audit")
    features = FeatureStore(DIRECTORY, value, read_only=True)
    try:
        origins = features.producer_origins()
        reused = origins["reused"]["records"]
    finally:
        features.close()
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
        response = client.get("/health")
        response.raise_for_status()
        health = response.json()
        if (health["status"] != "ok" or health["model"] != "native-4.05"
                or health["sources"] != value["runtime_sources"]
                or health["readout_runtime"] != value["readout_runtime"]):
            raise ValueError("Runtime source/geometry differs from migrated corpus")
        if subprocess.run(["docker", "container", "inspect", trainer_container],
                          capture_output=True).returncode == 0:
            raise ValueError("Refuse to replace an existing CPU trainer")
        response = client.post("/v1/jobs/extract", json={"prepared": list(PREPARATIONS)})
        response.raise_for_status()
        job = response.json()
        runtime_id = subprocess.check_output(["docker", "inspect", "--format", "{{.Id}}", runtime_container], text=True).strip()
        handle = {**job, "prepared": list(PREPARATIONS), "started": time.time(),
                  "container_id": runtime_id, "runtime": health, "base_url": str(client.base_url).rstrip("/"),
                  "feature_directory": str(DIRECTORY), "committed_before_resume": reused,
                  "scope_transition": str(DIRECTORY / "scope-transition.json")}
        write_json(project_path("runs/native-real/extraction-job.json"), handle)
        try:
            root, home = str(project_path("")), str(Path.home())
            options = []
            for flag, value in (("--cpuset-cpus", cpuset_cpus), ("--user", user)):
                if value is not None:
                    options.extend((flag, value))
            trainer_id = subprocess.check_output([
                "docker", "run", "--detach", "--name", trainer_container, "--label", "fuzev.experiment=true", *options,
                "--volume", root + ":" + root,
                "--volume", home + "/.cache/huggingface/hub:" + home + "/.cache/huggingface/hub:ro",
                "--env", "HOME=" + home, "--env", "FUZEV_ROOT=" + root,
                "--env", "PYTHONPATH=" + str(CHECKOUT), "--env", "CUDA_VISIBLE_DEVICES=",
                "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=4",
                "--env", "MKL_NUM_THREADS=4", "--env", "HF_HUB_OFFLINE=1",
                "--entrypoint", "python3", image, "-B", "-u", str(CHECKOUT / "tools/train_when_ready.py"),
                "--features", str(DIRECTORY), "--threads", "4"], text=True).strip()
        except BaseException:
            cancelled = client.post("/v1/jobs/" + job["id"] + "/cancel")
            cancelled.raise_for_status()
            raise
        trainer = {"container": trainer_container, "container_id": trainer_id, "native_job": job["id"],
                   "feature_identity": value, "device": "cpu", "threads": 4, "poll_interval_seconds": 300,
                   "launched": time.time(), "status": "waiting_for_full_active_corpus",
                   "sequence": ["text", "mixed", "calibration"],
                   "supervisor_source": {"directory": str(CHECKOUT), "commit": value["runtime_sources"]["recipes"],
                                         "orchestrator_sha256": file_sha256(CHECKOUT / "tools/train_when_ready.py")},
                   "reader_optimization": "bounded-mask-descriptors", "dataset_scope": contract()}
        write_json(project_path("runs/head-training-container.json"), trainer)
        write_json(project_path("runs/dataset-scope-v2/activation.json"), {"extraction": handle, "trainer": trainer})
        print(json.dumps({"job": job["id"], "model_container": runtime_id, "trainer_container": trainer_id,
                          "reused_records": reused, "prepared": list(PREPARATIONS)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["identity", "activate"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image", default=IMAGE, help="The lab's CUDA build image (Torch 2.13.0+cu130)")
    parser.add_argument("--trainer-container", help="CPU trainer container name")
    parser.add_argument("--runtime-container", help="Existing native server container name")
    parser.add_argument("--cpuset-cpus", help="Optional Docker CPU set")
    parser.add_argument("--user", help="Optional Docker user")
    args = parser.parse_args()
    if args.command == "identity":
        identity()
    else:
        if not args.image or not args.trainer_container or not args.runtime_container:
            parser.error("activate requires --image, --trainer-container and --runtime-container")
        activate(port=args.port, image=args.image, trainer_container=args.trainer_container,
                 runtime_container=args.runtime_container, cpuset_cpus=args.cpuset_cpus, user=args.user)
