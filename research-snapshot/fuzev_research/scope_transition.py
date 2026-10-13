"""Reuse only semantically and numerically compatible frozen shards, with origins."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from .acquire import write_json
from .features import file_sha256
from .scope import TRAINING_SOURCES, validate_contract


COMPUTATION_FIELDS = ("engine", "model", "revision", "encoding", "cache_precision", "feature_tap",
                      "hidden_size", "readout_runtime", "extraction_pipeline")


def canonical_ast(source, definition):
    node = next((node for node in ast.parse(source).body if getattr(node, "name", None) == definition), None)
    if node is None:
        raise ValueError("Missing pinned computation definition: " + definition)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def computation_proof(old_checkout, new_checkout):
    old_checkout, new_checkout = Path(old_checkout), Path(new_checkout)
    proof = {}
    for filename, definition in (("native.py", "NativeBackend"), ("extraction.py", "write_rows"),
                                 ("extraction.py", "PreparedRows"), ("extraction.py", "PIPELINE_POLICY")):
        old = (old_checkout / "fuzev_research" / filename).read_text()
        new = (new_checkout / "fuzev_research" / filename).read_text()
        if definition == "PIPELINE_POLICY":
            def policy(source):
                node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                            and any(getattr(target, "id", None) == definition for target in node.targets))
                return hashlib.sha256(ast.dump(node.value, include_attributes=False).encode()).hexdigest()
            a, b = policy(old), policy(new)
        else:
            a, b = canonical_ast(old, definition), canonical_ast(new, definition)
        if a != b:
            raise ValueError("Frozen computation changed; reuse requires separate numerical qualification")
        proof[filename + ":" + definition] = a
    for filename in ("encoding.py", "preparation.py", "templates/chat_template.jinja"):
        old = (old_checkout / "fuzev_research" / filename).read_bytes()
        digest = hashlib.sha256(old).hexdigest()
        if digest != file_sha256(new_checkout / "fuzev_research" / filename):
            raise ValueError("Frozen encoding/preparation changed; refuse feature reuse")
        proof[filename] = digest
    return proof


def migrate(source, destination, identity, prepared_text, old_checkout, new_checkout):
    source, destination, prepared_text = Path(source), Path(destination), Path(prepared_text)
    if destination.exists():
        raise ValueError("Refuse to overwrite an existing feature transition")
    validate_contract(identity.get("dataset_scope"))
    old = json.loads((source / "identity.json").read_text())
    if any(old.get(name) != identity.get(name) for name in COMPUTATION_FIELDS):
        raise ValueError("Numerical model/encoding/runtime geometry changed; features are incompatible")
    if old["runtime_sources"]["exllamav3-fuzev"] != identity["runtime_sources"]["exllamav3-fuzev"]:
        raise ValueError("Native engine source changed; refuse feature reuse")
    proof = computation_proof(old_checkout, new_checkout)
    database = sqlite3.connect((source / "index.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        version = database.execute("PRAGMA data_version").fetchone()[0]
        if database.execute("SELECT count(*) FROM failures").fetchone()[0]:
            raise ValueError("Unresolved failures in the old corpus")
        if database.execute("SELECT source FROM records WHERE source NOT IN ('helpsteer2','helpsteer3') LIMIT 1").fetchone():
            raise ValueError("This transition reuses only unchanged HelpSteer preparations")
        registered = database.execute("SELECT sha256 FROM preparations WHERE path=?", (str(prepared_text.resolve()),)).fetchone()
        if not registered or file_sha256(prepared_text) != registered[0]:
            raise ValueError("HelpSteer preparation changed; refuse to relabel cached features")
        fields = ("id", "group", "role", "primitive", "modality", "source", "revision", "candidates", "target", "adaptation", "attribute")
        matched, examined = 0, 0
        with prepared_text.open() as handle:
            for line in handle:
                expected = json.loads(line)
                examined += 1
                if expected["source"] not in TRAINING_SOURCES or identity["datasets"].get(expected["source"]) != expected["revision"]:
                    raise ValueError("Reused record is outside the revised source/revision contract")
                actual = database.execute("SELECT * FROM records WHERE id=?", (expected["id"],)).fetchone()
                if actual is None:
                    continue
                meta = json.loads(actual["metadata"])
                if (any(meta.get(key) != expected.get(key) for key in fields)
                        or actual["group_"] != expected["group"]
                        or any(actual[key] != expected[key] for key in ("role", "source", "primitive", "modality"))):
                    raise ValueError("Reused record semantics/group ownership differ from preparation")
                matched += 1
        count = database.execute("SELECT count(*) FROM records").fetchone()[0]
        if matched != count or not count:
            raise ValueError("Unaccounted old records or empty reuse corpus")
        if database.execute("SELECT group_ FROM records GROUP BY group_ HAVING count(DISTINCT role)>1 LIMIT 1").fetchone():
            raise ValueError("Leakage in the reused corpus")
        shards = []
        for row in database.execute("SELECT shard,count(*) AS n,min(row_) AS first,max(row_) AS last FROM records GROUP BY shard ORDER BY shard"):
            if row["first"] != 0 or row["last"] != row["n"] - 1:
                raise ValueError("Incomplete reused shard index")
            for suffix in ("json", "safetensors"):
                path = source / f"shard-{row['shard']:06d}.{suffix}"
                shards.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        destination.mkdir(parents=True)
        copied = sqlite3.connect(destination / "index.sqlite3")
        try:
            database.backup(copied)
            copied.execute("DELETE FROM preparations WHERE path!=?", (str(prepared_text.resolve()),))
            copied.execute("CREATE TABLE feature_origins (shard INTEGER PRIMARY KEY, source_identity_sha256 TEXT, source_directory TEXT, source_recipe_commit TEXT)")
            copied.executemany("INSERT INTO feature_origins VALUES(?,?,?,?)", (
                (row[0], file_sha256(source / "identity.json"), str(source.resolve()), old["runtime_sources"]["recipes"])
                for row in database.execute("SELECT DISTINCT shard FROM records ORDER BY shard")))
            copied.commit()
        finally:
            copied.close()
        for shard in shards:
            os.link(source / shard["name"], destination / shard["name"])
        if database.execute("PRAGMA data_version").fetchone()[0] != version:
            raise ValueError("Source corpus changed during transition; do not resume the incomplete destination")
        write_json(destination / "identity.json", identity)
        receipt = {"schema_version": 1, "artifact_kind": "compatible-feature-scope-transition",
                   "source": str(source.resolve()), "destination": str(destination.resolve()),
                   "source_identity_sha256": file_sha256(source / "identity.json"),
                   "destination_identity_sha256": file_sha256(destination / "identity.json"),
                   "source_runtime": old["runtime_sources"], "new_runtime": identity["runtime_sources"],
                   "computation_proof": proof, "records_reused": count, "prepared_records_examined": examined,
                   "prepared_sha256": registered[0], "shards": shards, "storage": "immutable hard-linked original shards; independent SQLite index",
                   "raw_datasets_deleted": False, "safety_guards_changed": False}
        write_json(destination / "scope-transition.json", receipt)
        return receipt
    finally:
        database.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "identity", "prepared-text", "old-checkout", "new-checkout"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = migrate(args.source, args.destination, json.loads(args.identity.read_text()),
                      args.prepared_text, args.old_checkout, args.new_checkout)
    print(json.dumps({"records_reused": receipt["records_reused"], "shard_files": len(receipt["shards"]),
                      "receipt": str(args.destination / "scope-transition.json")}))


if __name__ == "__main__":
    main()
