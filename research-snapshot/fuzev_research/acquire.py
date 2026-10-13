import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time

from huggingface_hub import HfApi, snapshot_download

from .paths import ROOT, project_path
from .scope import EXTERNAL_SOURCES, TRAINING_SOURCES
from .state import Store


MODELS = {
    "native-4.05": ("turboderp/Qwen3.8-Flash-Next-exl3", "55a732e0c4c3d4614bc42b68493bb930d9b02c0a"),
    "native-3.05": ("turboderp/Qwen3.8-Flash-Next-exl3", "3.05bpw_h5_ng5"),
    "native-5.05": ("turboderp/Qwen3.8-Flash-Next-exl3", "5.05bpw_h6_ng6"),
    "vllm-nvidia": ("nvidia/Qwen3.8-Flash-Next-NVFP4", "fc694b54fb0174e0913e6adf86691ef85a4ead47"),
    "vllm-radixark": ("RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594"),
}
DATASETS = {
    "helpsteer2": "nvidia/HelpSteer2",
    "helpsteer3": "nvidia/HelpSteer3",
    "boolq": "google/boolq",
    "qasper": "allenai/qasper",
    "mm-rlhf": "yifanzhang114/MM-RLHF",
    "vl-rewardbench": "MMInstruction/VL-RewardBench",
}
DATASET_REVISIONS = {"boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5"}


def dataset_purpose(name):
    if name in TRAINING_SOURCES:
        return "training"
    if name in EXTERNAL_SOURCES:
        return "external"
    raise ValueError(f"Dataset is outside the active scope: {name}")


def validate_dataset(name, info, purpose=None):
    expected = dataset_purpose(name)
    if purpose is not None and purpose != expected:
        raise ValueError(f"Dataset purpose mismatch: {name} is {expected}, not {purpose}")
    if info.get("purpose") not in (None, expected):
        raise ValueError(f"Acquisition purpose mismatch: {name}")
    if info.get("repo") != DATASETS[name] or not re.fullmatch(r"[0-9a-f]{40}", info.get("revision") or ""):
        raise ValueError(f"Dataset requires its official repository and absolute revision: {name}")
    return info


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name,
                                     suffix=".tmp", delete=False) as handle:
        temporary = type(path)(handle.name)
        try:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def update_manifest(callback):
    path = project_path("data/manifests/acquisition.json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(path.read_text())
        callback(manifest)
        write_json(path, manifest)
        return manifest


def source_manifest():
    rows = []
    for name in ("recipes", "exllamav3-fuzev", "vllm-fuzev", "vllm-exl3-fuzev"):
        path = ROOT / "src" / name
        from .paths import source_revision
        package = {"recipes": "fuzev-research", "exllamav3-fuzev": "exllamav3",
                   "vllm-fuzev": "vllm", "vllm-exl3-fuzev": "vllm-exl3"}[name]
        rows.append({"name": name, "path": str(path), "revision": source_revision(path, package)})
    return rows


def pin_all():
    api = HfApi()
    path = project_path("data/manifests/acquisition.json")
    existing = json.loads(path.read_text()) if path.exists() else {"models": {}, "datasets": {}}
    existing["sources"] = source_manifest()
    for kind, entries in (("models", MODELS), ("datasets", DATASETS)):
        for name, entry in entries.items():
            if name in existing[kind] and existing[kind][name].get("revision"):
                continue
            repo, ref = entry if kind == "models" else (entry, DATASET_REVISIONS.get(name, "main"))
            print(f"Pinning {kind}/{name}", flush=True)
            try:
                info = api.repo_info(repo_id=repo, repo_type="model" if kind == "models" else "dataset",
                                     revision=ref, files_metadata=True)
                card = getattr(info, "card_data", None)
                license_id = card.get("license") if card else None
                existing[kind][name] = {"repo": repo, "requested_revision": ref, "revision": info.sha,
                                        "license_declared": license_id,
                                        "files": [{"path": f.rfilename, "size": f.size} for f in info.siblings],
                                        "pinned_at": time.time()}
                if kind == "datasets":
                    existing[kind][name]["purpose"] = dataset_purpose(name)
            except Exception as error:
                existing[kind][name] = {"repo": repo, "requested_revision": ref,
                                        "error": type(error).__name__, "revision": None}
                print(f"Unavailable: {name} ({type(error).__name__})", flush=True)
            write_json(path, existing)
    return existing


def download_model(name):
    manifest = pin_all()
    model = manifest["models"][name]
    if not model.get("revision"):
        raise RuntimeError(f"Model revision unresolved: {name}")
    total = sum(f.get("size") or 0 for f in model["files"])
    if shutil.disk_usage(ROOT).free < total + (50 << 30):
        raise OSError("Insufficient disk space for model snapshot and 50 GiB reserve")
    snapshot = snapshot_download(model["repo"], revision=model["revision"], max_workers=4)
    update_manifest(lambda latest: latest["models"][name].update(
        local_snapshot=snapshot, downloaded_at=time.time()))
    print(json.dumps({"model": name, "revision": model["revision"], "snapshot": snapshot}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS))
    args = parser.parse_args()
    store = Store()
    store.initialize()
    if args.model:
        download_model(args.model)
    else:
        store.update("data.audit", "running", "Pinning model and dataset revisions; no training eligibility inferred from repository license alone.")
        pin_all()


if __name__ == "__main__":
    main()
