import json
from pathlib import Path
import sqlite3

import pytest

from fuzev_research.acquire import write_json
from fuzev_research.features import file_sha256
from fuzev_research.preference_transition import legacy_contract, migrate
from fuzev_research.scope import contract
from test_scope_transition import transition


@pytest.fixture
def extension(transition, monkeypatch):
    source, destination, identity, text, checkout = transition
    old = json.loads((source / "identity.json").read_text())
    old["dataset_scope"] = legacy_contract()
    write_json(source / "identity.json", old)
    identity["dataset_scope"] = contract(augmentation=True)
    root = source.parent
    manifest_path = root / "text-splits.json"
    write_json(manifest_path, {"path": str(text), "sha256": file_sha256(text), "records": 1,
                               "counts": {"helpsteer2/choice/fit": 1}})
    with sqlite3.connect(source / "index.sqlite3") as db:
        db.execute("UPDATE preparations SET completed=1,examined=1")
    # This one-shard copy fixture is not a numerical or real-corpus qualification.
    monkeypatch.setattr("fuzev_research.preference_transition.PREPARATIONS", {
        "text.jsonl": ("text-splits.json", "text", "training", ("helpsteer2",))})
    monkeypatch.setattr("fuzev_research.preference_transition.project_path", lambda path: root / Path(path).name)
    return source, destination, identity, checkout


def perform(extension):
    source, destination, identity, checkout = extension
    return migrate(source, destination, identity, checkout, checkout)


def test_retains_original_index_and_hardlinks_payloads(extension):
    source, destination, identity, _ = extension
    before = file_sha256(source / "index.sqlite3"), file_sha256(source / "identity.json")
    receipt = perform(extension)
    assert receipt["records_reused"] == 1
    assert json.loads((destination / "identity.json").read_text()) == identity
    assert before == (file_sha256(source / "index.sqlite3"), file_sha256(source / "identity.json"))
    assert (source / "index.sqlite3").stat().st_ino != (destination / "index.sqlite3").stat().st_ino
    assert (source / "shard-000000.safetensors").stat().st_ino == (destination / "shard-000000.safetensors").stat().st_ino
    with sqlite3.connect(destination / "index.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM feature_origins").fetchone()[0] == 1
    with pytest.raises(ValueError, match="overwrite"):
        perform(extension)


@pytest.mark.parametrize("change", ["failed", "incomplete", "changed-preparation", "inactive", "geometry", "scope"])
def test_failed_or_changed_corpus_never_admitted(extension, change):
    source, destination, identity, _ = extension
    with sqlite3.connect(source / "index.sqlite3") as db:
        if change == "failed":
            db.execute("INSERT INTO failures VALUES('bad','error')")
        if change == "incomplete":
            db.execute("UPDATE preparations SET completed=0")
        if change == "inactive":
            db.execute("UPDATE records SET source='aokvqa'")
    if change == "changed-preparation":
        path = source.parent / "text.jsonl"
        path.write_text(path.read_text() + "\n")
    if change == "geometry":
        identity["cache_precision"] = "changed"
    if change == "scope":
        identity["dataset_scope"] = contract()
    with pytest.raises(ValueError):
        perform(extension)
    assert not destination.exists()


def test_existing_origin_chain_retained_without_relabelling(extension):
    source, _, _, _ = extension
    with sqlite3.connect(source / "index.sqlite3") as db:
        db.execute("CREATE TABLE feature_origins (shard INTEGER PRIMARY KEY, source_identity_sha256 TEXT, source_directory TEXT, source_recipe_commit TEXT)")
        db.execute("INSERT INTO feature_origins VALUES(0,'previous-identity','previous-directory','original-producer')")
    prior = {"records_reused": 1, "destination_identity_sha256": file_sha256(source / "identity.json"),
             "computation_proof": {"earlier": "proof"}}
    write_json(source / "scope-transition.json", prior)
    receipt = perform(extension)
    assert receipt["prior_scope_transition"] == prior
    with sqlite3.connect(extension[1] / "index.sqlite3") as db:
        assert db.execute("SELECT source_recipe_commit FROM feature_origins WHERE shard=0").fetchone()[0] == "original-producer"
