import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from fuzev_research import acquire, data, images
from fuzev_research.scope import validate_row


@pytest.fixture(autouse=True)
def isolated_sources(tmp_path, monkeypatch):
    pins = {name: {"repo": repo, "revision": "a" * 40, "files": []}
            for name, repo in acquire.DATASETS.items()}
    for module in (data, images):
        monkeypatch.setattr(module, "acquisition", lambda: pins)
        monkeypatch.setattr(module, "project_path", lambda path: tmp_path / path)
        monkeypatch.setattr(module, "Store", lambda: SimpleNamespace(
            record_result=lambda *args: None, update=lambda *args: None))


def add_reference(index, digest, phash, source, source_id, heldout=False):
    with index.db:
        index.db.execute("INSERT OR IGNORE INTO images VALUES(?,?,?,?,?)",
                         (digest, f"{phash:016x}", 64, 64, str(index.image_root / (digest + ".png"))))
        index.db.execute("INSERT INTO references_ VALUES(?,?,?,?)", (source, source_id, digest, int(heldout)))


def test_active_index_excludes_aok_bridges_roots_and_holdout_flags(tmp_path):
    index = images.ImageIndex(tmp_path / "index.sqlite", tmp_path / "pixels")
    try:
        add_reference(index, "mm", 0, "mm-rlhf", "mm")
        add_reference(index, "aok-bridge", 15, "aokvqa", "bridge", True)
        add_reference(index, "vl", 255, "vl-rewardbench", "vl")
        add_reference(index, "mm", 0, "aokvqa", "same-pixels", True)
        legacy, _ = index.groups()
        assert legacy["mm"] == legacy["vl"]
        groups, heldout = index.groups(active_sources=images.ACTIVE_IMAGE_SOURCES)
        assert groups == {"mm": "image:mm", "vl": "image:vl"}
        assert heldout == {"image:vl"}
        assert images.image_roles(index, groups, heldout)["image:mm"] == data.group_role("image:mm", "mm-rlhf", "train")
        add_reference(index, "mm-near-vl", 511, "mm-rlhf", "overlap")
        groups, heldout = index.groups(active_sources=images.ACTIVE_IMAGE_SOURCES)
        assert groups["mm-near-vl"] == groups["vl"]
        assert groups["mm-near-vl"] in heldout
        with pytest.raises(ValueError, match="active image sources"):
            index.groups(active_sources=("mm-rlhf", "aokvqa"))
        with pytest.raises(ValueError, match="Inactive"):
            images.source_files("aokvqa")
    finally:
        index.close()


def test_readonly_index_does_not_initialize_or_allow_writes(tmp_path):
    path = tmp_path / "index.sqlite"
    index = images.ImageIndex(path, tmp_path / "pixels")
    index.close()
    index = images.ImageIndex(path, tmp_path / "not-created", readonly=True)
    try:
        assert not (tmp_path / "not-created").exists()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            index.db.execute("DELETE FROM references_")
    finally:
        index.close()


def emit(source, split, source_id, question, candidates, target, primitive="choice", **metadata):
    return {"source": source, "original_split": split, "source_image_id": source_id,
            "question": question, "candidates": candidates, "target": target, "primitive": primitive, **metadata}


@pytest.mark.parametrize("ranking,target", [([0, 1], [1., 0.]), ([1, 0], [0., 1.])])
def test_vl_choice_preserved_noul_relative_preference_and_shared_metadata(ranking, target):
    raw = {"id": "test-1", "query": "What color?", "response": ["red", "blue"], "human_ranking": ranking,
           "rationale": "secret rationale", "ground_truth": "secret answer", "query_source": "source"}
    choice = next(images.vl_reward_records(raw, emit))
    noul = next(images.vl_reward_noul_records(raw, emit))
    assert choice["question"] == "Which response best answers the question about the image?\nWhat color?"
    assert choice["candidates"] == [{"id": "0", "text": "red"}, {"id": "1", "text": "blue"}]
    assert choice["adaptation"] == "official_heldout_preference"
    assert noul["primitive"] == "noul" and noul["adaptation"] == "pairwise_preference_yes_no"
    assert noul["candidates"] == [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
    assert choice["target"] == noul["target"] == target
    assert "Response A: red\nResponse B: blue" in noul["question"]
    assert "preferred to response B" in noul["question"]
    assert "correct" not in noul["question"]
    for row in (choice, noul):
        assert row["source_test_example_id"] == "test-1"
        assert row["original_split"] == row["source_original_split"] == "test"
        assert row["evaluation_scope"] == "external"
        assert row["sampling_weight"] == row["inclusion_probability"] == 1
        assert row["label_semantics"] == "relative_pairwise_preference"
        assert "sampling_stratum" not in row and "secret" not in json.dumps(row)
    assert choice["source_response_pair_sha256"] == noul["source_response_pair_sha256"]
    swapped = next(images.vl_reward_records({**raw, "response": list(reversed(raw["response"]))}, emit))
    assert swapped["source_response_pair_sha256"] != choice["source_response_pair_sha256"]


def test_preparations_keep_mm_only_and_paired_vl_outputs(tmp_path, monkeypatch):
    index_path = tmp_path / "index.sqlite"
    image_root = tmp_path / "pixels"
    cls = images.ImageIndex
    index = cls(index_path, image_root)
    add_reference(index, "kept", 0, "mm-rlhf", "keep.png")
    add_reference(index, "official", (1 << 64) - 1, "vl-rewardbench", "vl1", True)
    add_reference(index, "official", (1 << 64) - 1, "vl-rewardbench", "vl2", True)
    add_reference(index, "overlap", (1 << 64) - 2, "mm-rlhf", "overlap.png")
    add_reference(index, "aok", 15, "aokvqa", "aok", True)
    index.close()
    original_digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    monkeypatch.setattr(images, "ImageIndex", lambda **kwargs: cls(index_path, image_root, **kwargs))
    raw_vl = [{"id": key, "query": "Q", "response": ["a", "b"], "human_ranking": [1, 0]} for key in ("vl1", "vl2")]
    parquet_calls = []

    def parquet_rows(name, *, include_images):
        parquet_calls.append((name, include_images))
        assert name == "vl-rewardbench" and not include_images
        return iter(("test", row) for row in raw_vl)

    monkeypatch.setattr(images, "parquet_rows", parquet_rows)
    monkeypatch.setattr(images, "download", lambda source, filename: (source, filename))
    monkeypatch.setattr(images, "json_lines", lambda path: iter([
        {"id": key, "image": key + ".png", "question": "Q", "models_output": ["a", "b", "c"],
         "final_ranking": [1, 2, 3]} for key in ("keep", "overlap")]))
    training = images.prepare_images()
    train_path = tmp_path / "data/prepared/images.jsonl"
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len(train) == 4 and {row["source"] for row in train} == {"mm-rlhf"}
    assert {row["source_image_id"] for row in train} == {"keep.png"}
    assert training["exclusions"]["official_holdout_overlap"] == 4
    assert training["dataset_scope_revision"] == "2026-10-02-v2"
    assert training["evaluation_scope"] == "training"
    assert training["sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    external = images.prepare_vl_rewardbench()
    pair = []
    for primitive in ("choice", "noul"):
        path = tmp_path / f"data/prepared/vl-rewardbench-{primitive}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == len({row["id"] for row in rows}) == 2
        for row in rows:
            validate_row(row, path.name)
            assert row["role"] == "test"
            assert row["image_sha256"] == "official"
            assert row["sample_size"] == row["population_size"] == 2
        assert external[primitive]["records"] == 2
        assert external[primitive]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        pair.append([(row["source_test_example_id"], row["image_sha256"], row["source_response_pair_sha256"])
                     for row in rows])
    assert pair[0] == pair[1]
    assert external["choice"]["source_test_example_ids_sha256"] == external["noul"]["source_test_example_ids_sha256"]
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == original_digest
    assert parquet_calls == [("vl-rewardbench", False), ("vl-rewardbench", False)]


def test_vl_index_coverage_and_split_checks_precede_training(tmp_path, monkeypatch):
    index = images.ImageIndex(tmp_path / "index.sqlite", tmp_path / "pixels")
    try:
        monkeypatch.setattr(images, "parquet_rows", lambda *args, **kwargs: iter([("test", {"id": "missing"})]))
        with pytest.raises(ValueError, match="Unindexed"):
            list(images.indexed_vl_rows(index))
        monkeypatch.setattr(images, "parquet_rows", lambda *args, **kwargs: iter([("validation", {"id": "x"})]))
        with pytest.raises(ValueError, match="original test"):
            list(images.indexed_vl_rows(index))
        monkeypatch.setattr(images, "parquet_rows", lambda *args, **kwargs: iter([]))
        with pytest.raises(ValueError, match="without official"):
            list(images.indexed_vl_rows(index))
    finally:
        index.close()


def test_vl_repeated_source_ids_keep_distinct_official_pairs(tmp_path, monkeypatch):
    index = images.ImageIndex(tmp_path / "index.sqlite", tmp_path / "pixels")
    try:
        add_reference(index, "image", 0, "vl-rewardbench", "repeat", True)
        raw = {"id": "repeat", "query": "Q", "response": ["a", "b"], "human_ranking": [0, 1]}
        other = {**raw, "response": ["c", "d"]}
        monkeypatch.setattr(images, "parquet_rows", lambda *args, **kwargs: iter([
            ("test", raw), ("test", other)]))
        rows = list(images.indexed_vl_rows(index))
        assert len({row["source_test_example_id"] for row in rows}) == 2
        for row in rows:
            choice = next(images.vl_reward_records(row, emit))
            noul = next(images.vl_reward_noul_records(row, emit))
            assert choice["source_test_example_id"] == noul["source_test_example_id"]
            assert choice["source_example_id"] == noul["source_example_id"] == "repeat"
        assert [row["source_test_example_id"] for row in images.indexed_vl_rows(index)] == [
            row["source_test_example_id"] for row in rows]
        monkeypatch.setattr(images, "parquet_rows", lambda *args, **kwargs: iter([
            ("test", raw), ("test", raw)]))
        with pytest.raises(ValueError, match="Duplicate official"):
            list(images.indexed_vl_rows(index))
    finally:
        index.close()
