"""CPU metadata audit of the real preparations; no inference or label scoring."""

from collections import Counter
import hashlib
import json
import sqlite3
from types import SimpleNamespace

from fuzev_research.acquire import write_json
from fuzev_research.evaluate import external_test_membership, validate_sampling_design
from fuzev_research.features import SCOPE_METADATA, file_sha256
from fuzev_research.paths import project_path
from fuzev_research.scope import PREPARATIONS, contract, validate_manifest, validate_row


def audit():
    roles, seen, sources, summaries = {}, set(), Counter(), {}
    external = sqlite3.connect(":memory:")
    external.row_factory = sqlite3.Row
    external.execute("CREATE TABLE records(id TEXT PRIMARY KEY,group_ TEXT,role TEXT,primitive TEXT,"
                     "modality TEXT,source TEXT,metadata TEXT)")
    try:
        for filename, (manifest_name, _, purpose, _) in PREPARATIONS.items():
            path = project_path("data/prepared/" + filename)
            manifest_path = project_path("data/manifests/" + manifest_name)
            manifest = validate_manifest(filename, json.loads(manifest_path.read_text()))
            digest, count, tracks = hashlib.sha256(), 0, Counter()
            with path.open("rb") as handle:
                for raw in handle:
                    digest.update(raw)
                    row = validate_row(json.loads(raw), filename)
                    if row["id"] in seen:
                        raise ValueError("Duplicate active prepared record identity")
                    seen.add(row["id"])
                    previous = roles.setdefault(row["group"], row["role"])
                    if previous != row["role"]:
                        raise ValueError("Fit/development/calibration/test group leakage")
                    if row.get("official_holdout_overlap"):
                        raise ValueError("Official external image overlaps active MM training")
                    count += 1
                    sources[row["source"]] += 1
                    tracks[f"{row['source']}/{row['primitive']}/{row['role']}"] += 1
                    if purpose == "external":
                        keys = ("id", "group", "role", "primitive", "modality", "source", "revision",
                                "candidates", "target", "adaptation", *SCOPE_METADATA)
                        meta = {key: row[key] for key in keys if key in row}
                        external.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)", (
                            row["id"], row["group"], row["role"], row["primitive"], row["modality"],
                            row["source"], json.dumps(meta, allow_nan=False)))
            expected = {key: value for key, value in manifest["counts"].items()
                        if not key.startswith("excluded_")}
            if (count != manifest["records"] or digest.hexdigest() != manifest["sha256"]
                    or tracks != expected):
                raise ValueError("Preparation hash/count/role manifest mismatch: " + filename)
            summaries[filename] = {"records": count, "sha256": digest.hexdigest(),
                                   "manifest_sha256": file_sha256(manifest_path),
                                   "counts": dict(tracks), "purpose": purpose}
        membership = external_test_membership(SimpleNamespace(db=external, identity={"dataset_scope": contract()}))
        qasper = json.loads(project_path("data/manifests/qasper-external-splits.json").read_text())
        rows = (json.loads(row[0]) for row in external.execute(
            "SELECT metadata FROM records WHERE source='qasper' "
            "AND json_extract(metadata,'$.adaptation')='evidence' ORDER BY id"))
        design = validate_sampling_design(qasper["sampling"], rows)
        result = {"dataset_scope": contract(), "records": len(seen), "groups": len(roles),
                  "source_records": dict(sources), "preparations": summaries,
                  "external_membership": membership, "sampling_design": design,
                  "scope": "metadata-only; no predictions, fitting, selection or calibration",
                  "raw_datasets_deleted": False, "safety_guards_bypassed": False}
        write_json(project_path("runs/dataset-scope-v2/preparation-audit.json"), result)
        print(json.dumps({key: result[key] for key in ("records", "groups", "source_records", "external_membership")},
                         indent=2))
        return result
    finally:
        external.close()


if __name__ == "__main__":
    audit()
