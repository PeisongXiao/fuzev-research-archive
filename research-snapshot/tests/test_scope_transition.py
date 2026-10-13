import json
from pathlib import Path
import sqlite3
import shutil

import pytest

from fuzev_research.features import FeatureStore, file_sha256
from fuzev_research.scope import contract
from fuzev_research.scope_transition import COMPUTATION_FIELDS, migrate


@pytest.fixture
def transition(tmp_path):
    checkout = tmp_path / "checkout"
    source_code = checkout / "fuzev_research"
    (source_code / "templates").mkdir(parents=True)
    (source_code / "native.py").write_text("class NativeBackend:\n    pass\n")
    (source_code / "extraction.py").write_text("PIPELINE_POLICY = {'test': True}\nclass PreparedRows:\n    pass\ndef write_rows():\n    pass\n")
    for name in ("encoding.py", "preparation.py", "templates/chat_template.jinja"):
        (source_code / name).write_text("# Synthetic source only\n")
    shutil.copytree(source_code, checkout / "old-source/fuzev_research")
    commit = "synthetic-source"
    identity = {name: "fixture" for name in COMPUTATION_FIELDS}
    identity.update(datasets={"helpsteer2": "fixture"}, runtime_sources={"recipes": commit, "exllamav3-fuzev": "native"})
    source = tmp_path / "source"
    store = FeatureStore(source, identity)
    row = {"id": "example", "group": "group", "role": "fit", "source": "helpsteer2", "revision": "fixture",
           "primitive": "choice", "modality": "text", "candidates": [{"id": "a"}, {"id": "b"}], "target": [1., 0.]}
    text = tmp_path / "text.jsonl"
    text.write_text(json.dumps(row) + "\n")
    store.register_preparation(text, file_sha256(text))
    store.db.execute("INSERT INTO records VALUES('example',0,0,'group','fit','choice','text','helpsteer2',?)", (json.dumps(row),))
    store.db.execute("INSERT INTO groups_ VALUES('group','fit')")
    store.db.commit()
    store.close()
    # These are explicitly synthetic file-copy fixtures, not qualified tensors.
    (source / "shard-000000.json").write_text(json.dumps([row]))
    (source / "shard-000000.safetensors").write_bytes(b"synthetic-copy-fixture")
    revised = {**identity, "dataset_scope": contract(), "runtime_sources": {**identity["runtime_sources"], "recipes": "new-source"}}
    return source, tmp_path / "destination", revised, text, checkout


def perform(values):
    source, destination, identity, text, checkout = values
    return migrate(source, destination, identity, text, checkout / "old-source", checkout)


def test_reuse_has_independent_index_immutable_shards_and_original_producer(transition):
    source, destination, _, _, _ = transition
    old = (source / "identity.json").read_bytes()
    receipt = perform(transition)
    assert receipt["records_reused"] == 1
    assert receipt["source_runtime"]["recipes"] != receipt["new_runtime"]["recipes"]
    assert (source / "identity.json").read_bytes() == old
    assert (source / "shard-000000.safetensors").stat().st_ino == (destination / "shard-000000.safetensors").stat().st_ino
    assert (source / "index.sqlite3").stat().st_ino != (destination / "index.sqlite3").stat().st_ino
    with sqlite3.connect(destination / "index.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM feature_origins").fetchone()[0] == 1
    with pytest.raises(ValueError, match="overwrite"):
        perform(transition)


@pytest.mark.parametrize("field", COMPUTATION_FIELDS)
def test_changed_geometry_or_numerical_identity_cannot_reuse(transition, field):
    transition[2][field] = "changed"
    with pytest.raises(ValueError, match="incompatible"):
        perform(transition)
    assert not transition[1].exists()


def test_changed_native_backbone_computation_cannot_reuse(transition):
    (transition[4] / "fuzev_research/native.py").write_text("class NativeBackend:\n    value = 2\n")
    with pytest.raises(ValueError, match="computation changed"):
        perform(transition)


def test_changed_prepared_input_cannot_relabel_features(transition):
    transition[3].write_text(transition[3].read_text() + "\n")
    with pytest.raises(ValueError, match="preparation changed"):
        perform(transition)


def test_inactive_cached_source_is_not_silently_imported(transition):
    with sqlite3.connect(transition[0] / "index.sqlite3") as db:
        db.execute("UPDATE records SET source='aokvqa'")
    with pytest.raises(ValueError, match="only unchanged HelpSteer"):
        perform(transition)
