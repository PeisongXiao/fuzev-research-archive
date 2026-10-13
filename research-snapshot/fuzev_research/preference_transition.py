"""Extend a complete v2 corpus without rewriting original records or tensors."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3

from .acquire import write_json
from .features import file_sha256, verify_preparation
from .paths import project_path
from .scope import LEGACY_REVISION, PREFERENCE_FILES, PREPARATIONS, contract, validate_contract, validate_manifest
from .scope_transition import COMPUTATION_FIELDS, computation_proof


def legacy_contract():
    return contract()


def migrate(source, destination, identity, old_checkout, new_checkout):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Refuse to overwrite an existing preference transition")
    validate_contract(identity.get("dataset_scope"))
    if identity["dataset_scope"] != contract(augmentation=True):
        raise ValueError("Preference transition requires the explicit augmented scope")
    old = json.loads((source / "identity.json").read_text())
    if old.get("dataset_scope") != legacy_contract():
        raise ValueError("Preference transition requires the exact original v2 scope")
    if (any(old.get(key) != identity.get(key) for key in COMPUTATION_FIELDS + ("datasets",))
            or old["runtime_sources"]["exllamav3-fuzev"] != identity["runtime_sources"]["exllamav3-fuzev"]):
        raise ValueError("Numerical model/encoding/runtime identity changed; features are incompatible")
    proof = computation_proof(old_checkout, new_checkout)
    db = sqlite3.connect((source / "index.sqlite3").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        version = db.execute("PRAGMA data_version").fetchone()[0]
        if db.execute("SELECT count(*) FROM failures").fetchone()[0]:
            raise ValueError("Unresolved source extraction failures")
        if db.execute("SELECT group_ FROM records GROUP BY group_ HAVING count(DISTINCT role)>1 LIMIT 1").fetchone():
            raise ValueError("Leakage in the original corpus")
        allowed = tuple(identity["datasets"])
        if db.execute("SELECT id FROM records WHERE source IS NULL OR source NOT IN (" +
                      ",".join("?" for _ in allowed) + ") LIMIT 1", allowed).fetchone():
            raise ValueError("Inactive source in the original corpus")
        expected, preparations = Counter(), []
        for name, (manifest_name, modality, _, _) in PREPARATIONS.items():
            if name in PREFERENCE_FILES:
                continue
            path = project_path("data/prepared/" + name)
            manifest_path = project_path("data/manifests/" + manifest_name)
            manifest = validate_manifest(name, json.loads(manifest_path.read_text()))
            registered = db.execute("SELECT * FROM preparations WHERE path=?", (str(path.resolve()),)).fetchone()
            if (not registered or registered["completed"] != 1
                    or registered["examined"] != manifest["records"]
                    or Path(manifest["path"]).resolve() != path.resolve()
                    or registered["sha256"] != manifest["sha256"]):
                raise ValueError("Original preparation is not complete and pinned: " + name)
            prepared = {"sha256": manifest["sha256"], "records": manifest["records"]}
            verify_preparation(db, path, prepared, modality, identity["datasets"], scoped=True)
            tracks = Counter()
            for key, count in manifest["counts"].items():
                if key.startswith("excluded_"):
                    continue
                source_, primitive, role = key.split("/")
                tracks[(modality, source_, primitive, role)] += count
            if sum(tracks.values()) != manifest["records"]:
                raise ValueError("Original preparation counts are incomplete")
            expected.update(tracks)
            preparations.append({"path": str(path), **prepared, "manifest_sha256": file_sha256(manifest_path)})
        actual = Counter({tuple(row[:4]): row[4] for row in db.execute(
            "SELECT modality,source,primitive,role,count(*) FROM records GROUP BY modality,source,primitive,role")})
        if actual != expected or not sum(actual.values()):
            raise ValueError("Unaccounted original records in the transition")
        prior_receipt = None
        has_origins = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_origins'").fetchone()
        if has_origins:
            prior_receipt = json.loads((source / "scope-transition.json").read_text())
            prior_count = db.execute("SELECT count(*) FROM records r JOIN feature_origins o ON r.shard=o.shard").fetchone()[0]
            if (prior_receipt["records_reused"] != prior_count
                    or prior_receipt["destination_identity_sha256"] != file_sha256(source / "identity.json")
                    or not prior_receipt.get("computation_proof")):
                raise ValueError("Original producer-origin chain is incomplete")
        shards = []
        for row in db.execute("SELECT shard,count(*) AS n,min(row_) AS first,max(row_) AS last FROM records GROUP BY shard ORDER BY shard"):
            if row["first"] != 0 or row["last"] != row["n"] - 1:
                raise ValueError("Incomplete original shard index")
            for suffix in ("json", "safetensors"):
                path = source / f"shard-{row['shard']:06d}.{suffix}"
                shards.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        if db.execute("PRAGMA data_version").fetchone()[0] != version:
            raise ValueError("Source corpus changed during verification")
        destination.mkdir(parents=True)
        copied = sqlite3.connect(destination / "index.sqlite3")
        try:
            db.backup(copied)
            copied.execute("CREATE TABLE IF NOT EXISTS feature_origins (shard INTEGER PRIMARY KEY, source_identity_sha256 TEXT, source_directory TEXT, source_recipe_commit TEXT)")
            copied.executemany("INSERT OR IGNORE INTO feature_origins VALUES(?,?,?,?)", (
                (row[0], file_sha256(source / "identity.json"), str(source), old["runtime_sources"]["recipes"])
                for row in db.execute("SELECT DISTINCT shard FROM records ORDER BY shard")))
            copied.commit()
        finally:
            copied.close()
        for shard in shards:
            os.link(source / shard["name"], destination / shard["name"])
        if db.execute("PRAGMA data_version").fetchone()[0] != version:
            raise ValueError("Source changed during copy; do not use the incomplete destination")
        write_json(destination / "identity.json", identity)
        receipt = {"schema_version": 1, "artifact_kind": "compatible-noul-preference-extension",
                   "source": str(source), "destination": str(destination),
                   "source_identity_sha256": file_sha256(source / "identity.json"),
                   "destination_identity_sha256": file_sha256(destination / "identity.json"),
                   "source_runtime": old["runtime_sources"], "new_runtime": identity["runtime_sources"],
                   "computation_proof": proof, "records_reused": sum(actual.values()),
                   "original_preparations": preparations, "prior_scope_transition": prior_receipt,
                   "shards": shards, "storage": "immutable hard-linked original shards; independent SQLite index",
                   "raw_datasets_deleted": False, "external_records_changed": False}
        write_json(destination / "scope-transition.json", receipt)
        return receipt
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "identity", "old-checkout", "new-checkout"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = migrate(args.source, args.destination, json.loads(args.identity.read_text()),
                      args.old_checkout, args.new_checkout)
    print(json.dumps({"records_reused": receipt["records_reused"], "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
