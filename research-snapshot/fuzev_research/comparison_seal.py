"""Seal completed comparison features; no model inference or quality scoring."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .comparison_inputs import frozen_reference
from .evaluate import (SUBSET_FILE, TEST_SCOPE, aligned_test_rows,
                       canonical_test_membership, read_json,
                       require_comparison_complete, require_final_ready,
                       require_feature_ownership, sha256, update_membership)
from .features import FeatureStore, file_sha256


def checkpoint_index(directory):
    """Require the writer to be drained; its WAL must not own committed rows."""
    path = directory / "index.sqlite3"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=1) as db:
        busy, _, _ = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise RuntimeError("Comparison index is busy; stop/drain its writer before sealing")
    wal = directory / "index.sqlite3-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Comparison index still has an unsealed WAL")


def seal_features(directory, spec, examined):
    """Independently check all canonical test rows and pin only actual storage."""
    directory = Path(directory).expanduser().resolve()
    path = directory / SUBSET_FILE
    if directory == Path(spec["features"]).expanduser().resolve():
        raise ValueError("Never seal or mutate the full training reference as a comparison")
    if path.exists():
        raise FileExistsError("Refuse to overwrite an existing subset extraction receipt")
    if type(examined) is not int or examined <= 0:
        raise ValueError("Report the actual positive examined input count")
    written = None
    try:
        with frozen_reference(spec) as reference:
            canonical = canonical_test_membership(reference)
            if examined != canonical["records"]:
                raise ValueError("Examined inputs do not account for the complete canonical holdout")
            identity = read_json(directory / "identity.json")
            if identity.get("corpus_scope") != TEST_SCOPE:
                raise ValueError("Require explicit evaluation-test-only scope; never relabel an existing corpus")
            checkpoint_index(directory)
            features = FeatureStore(directory, identity, read_only=True)
            try:
                if features.preparations():
                    raise ValueError("Test-only features cannot claim full preparation completion")
                if features.db.execute("SELECT COUNT(*) FROM records WHERE role!='test' OR role IS NULL").fetchone()[0]:
                    raise ValueError("Test-only features may contain exclusively test roles")
                shards = require_feature_ownership(features)
                count = features.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                if count != examined:
                    raise ValueError("Comparison features are incomplete or contain extra records")
                digest = hashlib.sha256()
                for _, meta in aligned_test_rows(reference, features):
                    update_membership(digest, meta)
                if digest.hexdigest() != canonical["ordered_sha256"]:
                    raise ValueError("Comparison canonical semantic membership mismatch")
                ownership = [{**entry,
                              "tensors_sha256": file_sha256(directory / f"shard-{entry['shard']:06d}.safetensors"),
                              "metadata_sha256": file_sha256(directory / f"shard-{entry['shard']:06d}.json")}
                             for entry in shards]
                manifest = {
                    "schema_version": 1, "artifact_kind": TEST_SCOPE, "roles": ["test"],
                    "reference": {
                        "feature_identity_sha256": spec["feature_identity_sha256"],
                        "preparation_sha256": sha256(require_final_ready(reference)),
                        "test_membership": canonical,
                    },
                    "extraction": {"completed": True, "examined": examined,
                                   "committed": count, "failures": 0},
                    "ownership": {
                        "feature_identity_sha256": file_sha256(directory / "identity.json"),
                        "index_sha256": file_sha256(directory / "index.sqlite3"),
                        "shards": ownership,
                    },
                }
                encoded = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
                with tempfile.NamedTemporaryFile(dir=directory, prefix="." + SUBSET_FILE + "-",
                                                 suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    try:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                        # Link is atomic and exclusive: no partial receipt or
                        # replacement of an operator's concurrently added file.
                        os.link(temporary, path)
                        written = encoded
                    finally:
                        temporary.unlink(missing_ok=True)
                pin = hashlib.sha256(encoded).hexdigest()
                # The evaluator checks exact sidecar/index ownership and tensor
                # shapes as well as semantic alignment; this is not a self-count.
                require_comparison_complete(reference, features, pin)
            finally:
                features.close()
        return {"directory": str(directory), "subset_manifest_sha256": pin,
                "manifest": manifest,
                "scope": "storage and membership only; no runtime or quality qualification"}
    except BaseException:
        # Remove only this invocation's unchanged receipt, never another edit.
        if written is not None and path.exists() and path.read_bytes() == written:
            path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Previously frozen evaluation plan")
    parser.add_argument("--features", type=Path, required=True, help="Completed, drained test-only comparison")
    parser.add_argument("--examined", type=int, required=True)
    args = parser.parse_args()
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Run comparison sealing without GPU access")
    torch.set_num_threads(1)
    plan = read_json(args.plan)
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported frozen evaluation plan")
    print(json.dumps(seal_features(args.features, plan["reference"], args.examined), allow_nan=False))


if __name__ == "__main__":
    main()
