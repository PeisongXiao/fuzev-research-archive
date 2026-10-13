import copy
import hashlib
import io
import json
import tarfile
from collections import Counter
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from fuzev_research import acquire, data
from fuzev_research.scope import validate_row


@pytest.fixture(autouse=True)
def isolated_sources(tmp_path, monkeypatch):
    pins = {name: {"repo": repo, "revision": "a" * 40, "files": []}
            for name, repo in acquire.DATASETS.items()}
    monkeypatch.setattr(data, "acquisition", lambda: pins)
    monkeypatch.setattr(data, "project_path", lambda path: tmp_path / path)
    monkeypatch.setattr(data, "Store", lambda: SimpleNamespace(record_result=lambda *args: None))
    return pins


def test_helpsteer_generator_matches_c907eb5_fixture_bytes(monkeypatch):
    fixture = {
        ("helpsteer2", "train.jsonl.gz"): [{"prompt": "Question", "response": "Answer", "helpfulness": 2.5,
                                            "correctness": 4, "coherence": 1, "complexity": 0, "verbosity": 3}],
        ("helpsteer2", "validation.jsonl.gz"): [{"prompt": "Heldout", "response": "Other", "helpfulness": 3}],
        ("helpsteer2", "preference/preference.jsonl.gz"): [
            {"prompt": "Question", "response_1": "A", "response_2": "B", "preference_strength": -2}],
        ("helpsteer3", "preference/train.jsonl.gz"): [
            {"context": [{"role": "user", "content": "Three"}], "response1": "A", "response2": "B",
             "overall_preference": 0}],
        ("helpsteer3", "preference/validation.jsonl.gz"): [
            {"context": [{"role": "user", "content": "Heldout three"}], "response1": "A", "response2": "B",
             "overall_preference": 1}],
    }
    calls = []

    def download(source, filename):
        calls.append((source, filename))
        return source, filename

    monkeypatch.setattr(data, "download", download)
    monkeypatch.setattr(data, "json_lines", lambda path: iter(fixture[path]))
    rows = list(data.text_records())
    encoded = "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows)
    assert len(rows) == 9
    assert hashlib.sha256(encoded.encode()).hexdigest() == "0c7fa20f422df2046cde7f3f470804f5fae503ff5512abfe8ecc1d699359668c"
    assert calls == list(fixture)
    assert all("dataset_scope_revision" not in row for row in rows)


@pytest.mark.parametrize("answer,target", [(True, [1., 0.]), (False, [0., 1.])])
def test_boolq_truth_not_answerability(answer, target):
    raw = {"question": "is the sample blue", "passage": "The sample is red.", "answer": answer}
    row = next(data.boolq_records(raw, "train", "example:0"))
    assert row["question"] == raw["question"]
    assert row["context"] == [{"role": "user", "content": raw["passage"]}]
    assert row["target"] == target
    assert row["candidates"] == [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
    assert row["original_answer"] is answer
    assert row["source_example_id"] == "example:0"
    assert row["adaptation"] == "boolean_truth"
    validate_row(row, "boolq.jsonl")


@pytest.mark.parametrize("answer", ["False", "true", 0, 1, None])
def test_boolq_rejects_non_boolean_answers(answer):
    with pytest.raises(ValueError, match="schema"):
        list(data.boolq_records({"question": "Q", "passage": "P", "answer": answer}, "train"))


def test_boolq_passage_group_and_official_validation_priority(tmp_path):
    raw = {"question": "First?", "passage": "Passage\n content", "answer": False}
    train = next(data.boolq_records(raw, "train"))
    validation = next(data.boolq_records({**raw, "question": "Second?", "passage": " Passage content "}, "validation"))
    assert train["group"] == validation["group"]
    assert validation["role"] == "test"
    path = tmp_path / "boolq.jsonl"
    summary = data.prepared_records(lambda: iter([train, validation]), path, tmp_path / "boolq-splits.json",
                                    filename="boolq.jsonl")
    assert summary["counts"]["excluded_holdout_overlap"] == 1
    assert [json.loads(line) for line in path.read_text().splitlines()] == [validation]


def test_boolq_reads_only_pinned_official_parquet_splits(tmp_path, monkeypatch, isolated_sources):
    files = [f"data/{split}-00000-of-00001.parquet" for split in ("train", "validation", "test")]
    isolated_sources["boolq"]["files"] = [{"path": filename} for filename in reversed(files)]
    for filename in files[:2]:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(pa.Table.from_pylist([{"question": "Q", "passage": "P", "answer": False}]), path)
    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(data, "hf_hub_download", download)
    rows = list(data.boolq_source_rows())
    assert [row[0] for row in rows] == ["train", "validation"]
    assert [row[1] for row in rows] == [f"{name}:0" for name in files[:2]]
    assert all(call[0] == "google/boolq" and call[2]["revision"] == "a" * 40 for call in calls)


def qasper_fixture():
    documents = []
    for paper_number in range(29):
        qas = []
        for number in range(50):
            qid = paper_number * 50 + number
            qas.append({"question_id": f"q{qid:04}", "question": f"Question {qid}?", "answers": [
                {"answer": {"unanswerable": False, "yes_no": True if qid < 239 else None,
                            "evidence": ["Evidence", "Disputed"]}},
                {"answer": {"unanswerable": False, "yes_no": False if qid < 239 else None,
                            "evidence": ["Evidence"]}},
            ]})
        documents.append(("test", f"paper{paper_number:02}", {
            "title": "Title", "abstract": "Abstract", "full_text": [
                {"paragraphs": ["Evidence", "Disputed", "Other", "Evidence"]}], "qas": qas}))
    duplicate = copy.deepcopy(documents[-1][2]["qas"][-1])
    duplicate["question_id"] = "duplicate-question"
    documents[-1][2]["qas"].append(duplicate)
    return documents


def test_qasper_exact_counts_soft_targets_aliases_and_test_only_metadata():
    documents = qasper_fixture()
    rows, summary = data.qasper_external_records(documents)
    assert len(rows) == 3689
    assert Counter(row["adaptation"] for row in rows) == {"yes_no": 239, "answerability": 1450, "evidence": 2000}
    assert summary["raw_counts"]["answerability"] == 1451
    assert summary["excluded_exact_duplicates"] == {"answerability": 1, "evidence": 3}
    assert summary["sampling"]["population"] == 4350
    for row in rows:
        validate_row(row, "qasper-external.jsonl")
        assert row["role"] == row["original_split"] == row["source_original_split"] == "test"
        assert row["evaluation_scope"] == "external"
        assert row["source_test_example_id"] == row["source_question_id"]
        assert row["sampling_seed"] == 20261002
        assert row["sampling_weight"] * row["inclusion_probability"] == pytest.approx(1)
        if row["adaptation"] == "yes_no":
            assert row["target"] == [.5, .5]
        if row["adaptation"] != "evidence":
            assert row["sampling_weight"] == row["inclusion_probability"] == 1
            assert "sampling_stratum" not in row
    assert any(row["source_question_ids"] == ["duplicate-question", "q1449"] for row in rows)
    reordered = [(split, key, {**paper, "qas": list(reversed(paper["qas"]))})
                 for split, key, paper in reversed(documents)]
    assert data.qasper_external_records(reordered) == (rows, summary)


def test_qasper_rejects_wrong_split_and_census_before_sampling():
    documents = qasper_fixture()
    with pytest.raises(ValueError, match="original test"):
        data.qasper_external_records([("dev", documents[0][1], documents[0][2])])
    with pytest.raises(ValueError, match="census count"):
        data.qasper_external_records(documents[:1])
    with pytest.raises(ValueError, match="Duplicate QASPER paper"):
        data.qasper_external_records([documents[0], documents[0]])


def test_qasper_preparation_paths_and_manifests_only_in_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "qasper_documents", lambda: iter(qasper_fixture()))
    summary = data.prepare_qasper()
    path = tmp_path / "data/prepared/qasper-external.jsonl"
    assert summary["records"] == 3689
    assert summary["path"] == str(path)
    assert summary["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "data/manifests/qasper-external-splits.json").read_text()) == summary
    assert not (tmp_path / "data/prepared/qasper.jsonl").exists()


def test_qasper_reader_uses_only_verified_local_test_archive(tmp_path, monkeypatch):
    path = tmp_path / "data/raw/qasper/test.tgz"
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w:gz") as archive:
        for split in ("train", "dev", "test"):
            payload = json.dumps({split: {"marker": split}}).encode()
            member = tarfile.TarInfo(f"qasper-{split}-v0.3.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(data, "QASPER_TEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert list(data.qasper_documents()) == [("test", "test", {"marker": "test"})]
    monkeypatch.setattr(data, "QASPER_TEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        list(data.qasper_documents())


def test_prepared_exact_count_failure_does_not_publish(tmp_path):
    row = next(data.boolq_records({"question": "Q", "passage": "P", "answer": False}, "train"))
    path = tmp_path / "boolq.jsonl"
    with pytest.raises(ValueError, match="count mismatch"):
        data.prepared_records(lambda: iter([row, row]), path, tmp_path / "summary.json", expected_records=2)
    assert not path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_acquisition_scope_absolute_pins_and_historical_manifest_retention(tmp_path, monkeypatch):
    assert "aokvqa" not in acquire.DATASETS
    assert acquire.DATASETS["boolq"] == "google/boolq"
    revision = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
    assert acquire.DATASET_REVISIONS["boolq"] == revision
    with pytest.raises(ValueError, match="outside"):
        acquire.dataset_purpose("aokvqa")
    with pytest.raises(ValueError, match="purpose mismatch"):
        acquire.validate_dataset("qasper", {}, "training")
    with pytest.raises(ValueError, match="absolute revision"):
        acquire.validate_dataset("boolq", {"repo": "google/boolq", "revision": "main"})
    path = tmp_path / "data/manifests/acquisition.json"
    path.parent.mkdir(parents=True)
    manifest = {"models": {name: {"revision": "b" * 40} for name in acquire.MODELS},
                "datasets": {name: {"repo": repo, "revision": "a" * 40}
                             for name, repo in acquire.DATASETS.items() if name != "boolq"}}
    historical = {"repo": "historical/aok", "revision": "c" * 40, "retained": True}
    manifest["datasets"]["aokvqa"] = historical
    path.write_text(json.dumps(manifest))
    calls = []

    def repo_info(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sha=revision, card_data={"license": "cc-by-sa-3.0"}, siblings=[])

    monkeypatch.setattr(acquire, "HfApi", lambda: SimpleNamespace(repo_info=repo_info))
    monkeypatch.setattr(acquire, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(acquire, "source_manifest", lambda: [])
    result = acquire.pin_all()
    assert len(calls) == 1 and calls[0]["revision"] == revision
    assert result["datasets"]["aokvqa"] == historical
    assert result["datasets"]["boolq"]["purpose"] == "training"
