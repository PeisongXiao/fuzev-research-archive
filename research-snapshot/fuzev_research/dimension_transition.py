"""Extend a pinned v3 corpus with dimension views, without rewriting old features."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from .acquire import write_json
from .features import file_sha256, verify_preparation
from .paths import project_path
from .scope import PREPARATIONS, contract, validate_contract, validate_manifest
from .scope_transition import COMPUTATION_FIELDS, computation_proof


def test_snapshot(database):
    """Hash opaque original TEST index rows; do not parse examples or targets."""
    digest, count = hashlib.sha256(), 0
    for row in database.execute("SELECT * FROM records WHERE role='test' ORDER BY id"):
        digest.update(json.dumps(tuple(row), separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
        count += 1
    return {"records": count, "ordered_sha256": digest.hexdigest()}


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def migrate(source, destination, identity, old_checkout, new_checkout):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Refuse to overwrite an existing dimension transition")
    validate_contract(identity.get("dataset_scope"))
    if identity["dataset_scope"] != contract(augmentation=True, dimensions=True):
        raise ValueError("Dimension transition requires the explicit v4 scope")
    source_sha = file_sha256(source / "identity.json")
    old = json.loads((source / "identity.json").read_text())
    if old.get("dataset_scope") != contract(augmentation=True):
        raise ValueError("Dimension transition requires the exact complete v3 scope")
    if (any(old.get(key) != identity.get(key) for key in COMPUTATION_FIELDS + ("datasets",))
            or {key: value for key, value in old.items() if key not in {"dataset_scope", "runtime_sources"}}
            != {key: value for key, value in identity.items() if key not in {"dataset_scope", "runtime_sources"}}
            or {key: value for key, value in old["runtime_sources"].items() if key != "recipes"}
            != {key: value for key, value in identity["runtime_sources"].items() if key != "recipes"}):
        raise ValueError("Numerical model/encoding/runtime identity changed; features are incompatible")
    proof = computation_proof(old_checkout, new_checkout)
    db = sqlite3.connect((source / "index.sqlite3").as_uri() + "?mode=ro", uri=True, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        db.execute("SELECT count(*) FROM records").fetchone()
        version = db.execute("PRAGMA data_version").fetchone()[0]
        db.execute("BEGIN")
        if db.execute("SELECT count(*) FROM failures").fetchone()[0]:
            raise ValueError("Unresolved source extraction failures")
        if db.execute("SELECT group_ FROM records GROUP BY group_ HAVING count(DISTINCT role)>1 LIMIT 1").fetchone():
            raise ValueError("Leakage in the original corpus")
        if db.execute("SELECT r.id FROM records r LEFT JOIN groups_ g ON r.group_=g.id "
                      "WHERE g.role IS NULL OR g.role!=r.role LIMIT 1").fetchone():
            raise ValueError("Original group ownership is missing or inconsistent")
        expected, preparations, stamps = Counter(), [], {}
        for name in old["dataset_scope"]["preparations"]:
            manifest_name, modality, purpose, _ = PREPARATIONS[name]
            path = project_path("data/prepared/" + name).resolve()
            manifest_path = project_path("data/manifests/" + manifest_name).resolve()
            stamps[path], stamps[manifest_path] = _stamp(path), _stamp(manifest_path)
            manifest = validate_manifest(name, json.loads(manifest_path.read_text()))
            registered = db.execute("SELECT * FROM preparations WHERE path=?", (str(path),)).fetchone()
            if (not registered or registered["completed"] != 1
                    or registered["examined"] != manifest["records"]
                    or Path(manifest["path"]).resolve() != path
                    or registered["sha256"] != manifest["sha256"]
                    or file_sha256(path) != manifest["sha256"]):
                raise ValueError("Original preparation is not complete and pinned: " + name)
            prepared = {"sha256": manifest["sha256"], "records": manifest["records"]}
            if purpose == "training":
                verify_preparation(db, path, prepared, modality, identity["datasets"], scoped=True)
            tracks = Counter()
            for key, count in manifest["counts"].items():
                if key.startswith("excluded_"):
                    continue
                if type(count) is not int or count < 0:
                    raise ValueError("Invalid original preparation counts")
                source_, primitive, role = key.split("/")
                tracks[(modality, source_, primitive, role)] += count
            if sum(tracks.values()) != manifest["records"]:
                raise ValueError("Original preparation counts are incomplete")
            expected.update(tracks)
            preparations.append({"path": str(path), **prepared,
                                 "manifest_sha256": file_sha256(manifest_path),
                                 "verification": "canonical training rows" if purpose == "training"
                                 else "descriptors, completed binding and file SHA only"})
        if {row[0] for row in db.execute("SELECT path FROM preparations")} != {p["path"] for p in preparations}:
            raise ValueError("Unaccounted original preparation binding")
        actual = Counter({tuple(row[:4]): row[4] for row in db.execute(
            "SELECT modality,source,primitive,role,count(*) FROM records GROUP BY modality,source,primitive,role")})
        if actual != expected or not sum(actual.values()):
            raise ValueError("Unaccounted original records in the transition")
        snapshot = test_snapshot(db)
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_origins'").fetchone():
            raise ValueError("Missing original v3 producer-origin chain")
        prior_path = source / "scope-transition.json"
        stamps[prior_path] = _stamp(prior_path)
        prior_sha = file_sha256(prior_path)
        prior = json.loads(prior_path.read_text())
        origin_rows = [tuple(row) for row in db.execute("SELECT * FROM feature_origins ORDER BY shard")]
        prior_count = db.execute("SELECT count(*) FROM records r JOIN feature_origins o ON r.shard=o.shard").fetchone()[0]
        if (not origin_rows or prior.get("records_reused") != prior_count
                or prior.get("destination_identity_sha256") != source_sha or not prior.get("computation_proof")
                or db.execute("SELECT o.shard FROM feature_origins o LEFT JOIN records r ON r.shard=o.shard "
                              "WHERE r.id IS NULL LIMIT 1").fetchone()):
            raise ValueError("Original producer-origin chain is incomplete")
        shards = []
        for row in db.execute("SELECT shard,count(*) AS n,count(DISTINCT row_) AS positions,min(row_) AS first,max(row_) AS last "
                              "FROM records GROUP BY shard ORDER BY shard"):
            if row["first"] != 0 or row["last"] != row["n"] - 1 or row["positions"] != row["n"]:
                raise ValueError("Incomplete original shard index")
            for suffix in ("json", "safetensors"):
                path = source / f"shard-{row['shard']:06d}.{suffix}"
                stamps[path] = _stamp(path)
                shards.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        destination.mkdir(parents=True)
        copied = sqlite3.connect(destination / "index.sqlite3")
        try:
            # A separate reader avoids sqlite3.backup stalling on our verification transaction.
            with sqlite3.connect((source / "index.sqlite3").as_uri() + "?mode=ro", uri=True) as reader:
                reader.backup(copied)
            if test_snapshot(copied) != snapshot:
                raise ValueError("Original TEST snapshot changed during backup")
            if [tuple(row) for row in copied.execute("SELECT * FROM feature_origins ORDER BY shard")] != origin_rows:
                raise ValueError("Original producer origins changed during backup")
            copied.executemany("INSERT OR IGNORE INTO feature_origins VALUES(?,?,?,?)", (
                (row[0], source_sha, str(source), old["runtime_sources"]["recipes"])
                for row in db.execute("SELECT DISTINCT shard FROM records ORDER BY shard")))
            copied.commit()
        finally:
            copied.close()
        for shard in shards:
            path = source / shard["name"]
            if _stamp(path) != stamps[path]:
                raise ValueError("Source shard changed before linking")
            os.link(path, destination / shard["name"])
            if file_sha256(destination / shard["name"]) != shard["sha256"]:
                raise ValueError("Source shard changed during linking")
            # Creating our hard link changes inode ctime, but not the pinned bytes.
            stamps[path] = _stamp(path)
        db.execute("ROLLBACK")
        if (db.execute("PRAGMA data_version").fetchone()[0] != version
                or source_sha != file_sha256(source / "identity.json")
                or any(_stamp(path) != stamp for path, stamp in stamps.items())):
            raise ValueError("Source changed during copy; do not use the incomplete destination")
        write_json(destination / "identity.json", identity)
        receipt = {"schema_version": 1, "artifact_kind": "compatible-dimension-choice-extension",
                   "source": str(source), "destination": str(destination),
                   "source_identity_sha256": source_sha,
                   "destination_identity_sha256": file_sha256(destination / "identity.json"),
                   "source_runtime": old["runtime_sources"], "new_runtime": identity["runtime_sources"],
                   "computation_proof": proof, "records_reused": sum(actual.values()),
                   "original_preparations": preparations, "prior_scope_transition": prior,
                   "prior_scope_transition_sha256": prior_sha, "original_test_snapshot": snapshot,
                   "shards": shards, "storage": "immutable hard-linked original shards; independent SQLite index",
                   "raw_datasets_deleted": False, "external_records_changed": False,
                   "external_examples_parsed": False, "original_test_records_changed": False}
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
