import hashlib
import json
import sqlite3

import pytest

from fuzev_research.features import FeatureStore


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("fuzev_research.features.project_path", lambda path: tmp_path / path)
    identity = {"datasets": {"text": "fixture", "qasper": "fixture", "images": "fixture"}}
    store = FeatureStore(tmp_path / "features", identity)
    for source in ("text", "qasper", "images"):
        modality = "image" if source == "images" else "text"
        primitive = "noul" if source == "qasper" else "choice"
        rows = [{"id": f"{source}-{role}", "group": f"{source}-{role}", "role": role,
                 "primitive": primitive, "modality": modality, "source": source, "revision": "fixture",
                 "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
                 "target": [1., 0.]} for role in ("fit", "development", "calibration", "test")]
        path = tmp_path / f"data/prepared/{source}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = tmp_path / f"data/manifests/{'image' if source == 'images' else source}-splits.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"path": str(path), "records": 4, "sha256": digest,
                                       "counts": {**{f"{source}/{primitive}/{row['role']}": 1 for row in rows},
                                                  "excluded_exact_duplicates": 3}}))
        store.register_preparation(path, digest)
        with store.db:
            for index, row in enumerate(rows):
                store.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)", (
                    row["id"], 0, index, row["group"], row["role"], primitive, modality, source, json.dumps(row)))
        store.finish_preparation(path, 4)
    yield store, tmp_path
    store.close()


def test_stage_readiness_counts_and_manifest_identity(corpus):
    store, _ = corpus
    assert store.require_ready("text")["records"] == 8
    assert store.require_ready("mixed")["records"] == 12
    assert store.require_ready("calibrate") == store.require_ready("mixed")
    assert len(store.require_ready("text")["preparations"]) == 2


@pytest.mark.parametrize("mutation,match", [
    ("DELETE FROM preparations WHERE path LIKE '%qasper.jsonl'", "not completed"),
    ("UPDATE preparations SET completed=0", "not completed"),
    ("UPDATE preparations SET examined=3", "not fully examined"),
    ("UPDATE preparations SET sha256='changed'", "identity/hash"),
    ("DELETE FROM records WHERE id='qasper-fit'", "counts"),
    ("UPDATE records SET id='substitute' WHERE id='qasper-fit'", "not accounted"),
    ("INSERT INTO failures VALUES('bad', 'context overflow')", "Extraction failures"),
])
def test_incomplete_or_changed_corpora_fail_closed(corpus, mutation, match):
    store, _ = corpus
    store.require_ready("text")
    with store.db:
        store.db.execute(mutation)
    with pytest.raises(ValueError, match=match):
        store.require_ready("text")


def test_text_does_not_require_images_but_mixed_and_calibration_do(corpus):
    store, _ = corpus
    with store.db:
        store.db.execute("UPDATE preparations SET completed=0 WHERE path LIKE '%images.jsonl'")
    store.require_ready("text")
    for stage in ("mixed", "calibrate"):
        with pytest.raises(ValueError, match="not completed"):
            store.require_ready(stage)


def test_prepared_hash_is_rechecked_after_file_change(corpus):
    store, root = corpus
    store.require_ready("text")
    path = root / "data/prepared/text.jsonl"
    path.write_text(path.read_text().replace('"fixture"', '"changed"'))
    with pytest.raises(ValueError, match="not accounted|hash"):
        store.require_ready("text")


def test_missing_manifest_cannot_reuse_cached_readiness(corpus):
    store, root = corpus
    store.require_ready("text")
    (root / "data/manifests/qasper-splits.json").unlink()
    with pytest.raises(ValueError, match="Missing required"):
        store.require_ready("text")


def test_reader_is_read_only_and_observes_only_committed_changes(corpus):
    writer, _ = corpus
    reader = FeatureStore(writer.directory, writer.identity, read_only=True)
    try:
        assert reader.require_ready("text")["records"] == 8
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.db.execute("DELETE FROM records")
        writer.db.execute("DELETE FROM records WHERE id='text-fit'")
        assert len(reader.records("fit", "text")) == 2
        writer.db.commit()
        assert len(reader.records("fit", "text")) == 1
        with pytest.raises(ValueError, match="counts"):
            reader.require_ready("text")
    finally:
        reader.close()


def test_read_only_missing_store_is_not_created(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(ValueError, match="Missing feature identity"):
        FeatureStore(path, {}, read_only=True)
    assert not path.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_image_commits_do_not_reparse_completed_text_roots(corpus, monkeypatch, legacy):
    writer, _ = corpus
    if legacy:
        for suffix in ("insert", "delete", "update"):
            writer.db.execute(f"DROP TRIGGER records_version_{suffix}")
    reader = FeatureStore(writer.directory, writer.identity, read_only=True)
    try:
        expected = reader.require_ready("text")
        def forbidden(*args):
            pytest.fail("Unrelated image commits must not reparse text/QASPER roots")
        monkeypatch.setattr("fuzev_research.features.verify_preparation", forbidden)
        with writer.db:
            writer.db.execute("UPDATE records SET shard=shard+1 WHERE modality='image'")
            writer.db.execute("UPDATE preparations SET completed=0 WHERE path LIKE '%images.jsonl'")
        assert reader.require_ready("text") == expected
        with writer.db:
            writer.db.execute("UPDATE records SET role='test' WHERE id='text-fit'")
        with pytest.raises(ValueError, match="counts"):
            reader.require_ready("text")
    finally:
        reader.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_same_count_text_metadata_mutation_invalidates_readiness(corpus, legacy):
    writer, _ = corpus
    if legacy:
        for suffix in ("insert", "delete", "update"):
            writer.db.execute(f"DROP TRIGGER records_version_{suffix}")
    reader = FeatureStore(writer.directory, writer.identity, read_only=True)
    try:
        reader.require_ready("text")
        with writer.db:
            writer.db.execute("UPDATE records SET metadata=json_set(metadata,'$.target',json('[0,1]')) WHERE id='text-fit'")
        with pytest.raises(ValueError, match="not accounted"):
            reader.require_ready("text")
    finally:
        reader.close()


def test_image_commit_during_text_verification_is_allowed(corpus, monkeypatch):
    import fuzev_research.features as module

    writer, _ = corpus
    reader = FeatureStore(writer.directory, writer.identity, read_only=True)
    original = module.verify_preparation
    def verify(*args):
        original(*args)
        with writer.db:
            writer.db.execute("UPDATE records SET shard=shard+1 WHERE modality='image'")
    monkeypatch.setattr(module, "verify_preparation", verify)
    try:
        assert reader.require_ready("text")["records"] == 8
    finally:
        reader.close()


def test_replace_that_moves_text_into_images_invalidates_text_cache(corpus):
    store, _ = corpus
    store.require_ready("text")
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO records SELECT id,shard,row_,group_,role,primitive,"
                         "'image',source,metadata FROM records WHERE id='text-fit'")
    with pytest.raises(ValueError, match="counts"):
        store.require_ready("text")
