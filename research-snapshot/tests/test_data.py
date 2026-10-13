import json

import pytest

from fuzev_research.data import (
    conversation, group_role, ordinal_records, preference_records,
    prepared_records, qasper_records, stable_id,
)


@pytest.fixture(autouse=True)
def pinned_sources(monkeypatch):
    monkeypatch.setattr("fuzev_research.data.acquisition", lambda: {
        name: {"revision": "pinned-test-revision"}
        for name in ("helpsteer2", "helpsteer3", "qasper")
    })


def test_splits_are_deterministic_and_official_heldout_cannot_train():
    group = "image:" + stable_id("pixels")
    assert group_role(group, "aokvqa", "train") == group_role(group, "aokvqa", "train")
    assert group_role(group, "aokvqa", "validation") == "test"
    assert group_role(group, "helpsteer2", "validation") == "test"
    assert group_role(group, "mm-rlhf", "train") in {"fit", "development", "calibration", "test"}


def test_signed_preferences_and_embedded_split():
    hs2 = {"prompt": "question", "response_1": "a", "response_2": "b",
           "preference_strength": -2, "split": "validation"}
    hs3 = {"context": [{"role": "user", "content": "question"}],
           "response1": "a", "response2": "b", "overall_preference": 2}
    a = next(preference_records(hs2, "helpsteer2", "train"))
    b = next(preference_records(hs3, "helpsteer3", "train"))
    assert a["target"] == [1, 0]
    assert b["target"] == [0, 1]
    assert a["group"] == b["group"]
    assert a["role"] == "test"
    assert a["original_split"] == "validation"
    hs3["overall_preference"] = 0
    assert next(preference_records(hs3, "helpsteer3", "train"))["target"] == [.5, .5]


def test_ordinal_group_excludes_response_and_preserves_rating_semantics():
    row = {"prompt": "question", "response": "answer", "helpfulness": 2}
    quality = next(ordinal_records(row, "train"))
    pref = {"prompt": "question", "response_1": "a", "response_2": "b",
            "preference_strength": -1}
    assert quality["group"] == next(preference_records(pref, "helpsteer2", "train"))["group"]
    assert quality["target"] == [0, 0, 1, 0, 0]
    assert quality["context"][-1] == {"role": "assistant", "content": "answer"}


def test_multi_turn_source_prompt_matches_structured_conversation():
    raw = "<extra_id_1>User\nA<extra_id_1>Assistant\nB<extra_id_1>User\nC"
    assert conversation(raw) == [{"role": "user", "content": "A"},
                                 {"role": "assistant", "content": "B"},
                                 {"role": "user", "content": "C"}]
    assert conversation("A\n<extra_id_1>Assistant\nB<extra_id_1>User\nC") == conversation(raw)


def test_train_duplicate_cannot_consume_heldout_record_id(tmp_path, monkeypatch):
    pref = {"prompt": "question", "response_1": "a", "response_2": "b",
            "preference_strength": -1}
    train = next(preference_records(pref, "helpsteer2", "train"))
    heldout = next(preference_records(pref, "helpsteer2", "validation"))
    assert train["id"] == heldout["id"]
    monkeypatch.setattr("fuzev_research.data.Store", lambda: type("Store", (), {
        "record_result": lambda *args: None,
    })())
    path = tmp_path / "data.jsonl"
    prepared_records(lambda: iter([train, heldout]), path, tmp_path / "summary.json")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["role"] == "test"


def test_qasper_votes_are_soft_and_every_adaptation_is_paper_grouped():
    paper = {"title": "Study", "abstract": "Abstract", "full_text": [
        {"paragraphs": ["Evidence", "Other"]}], "qas": [{
            "question": "Is it supported?", "question_id": "q1", "answers": [
                {"answer": {"unanswerable": False, "yes_no": True, "evidence": ["Evidence"]}},
                {"answer": {"unanswerable": False, "yes_no": False, "evidence": ["Evidence"]}},
                {"answer": {"unanswerable": True, "yes_no": None, "evidence": []}},
            ]}]}
    rows = list(qasper_records("dev", "paper1", paper))
    assert len(rows) == 4
    assert rows[0]["target"] == pytest.approx([2 / 3, 1 / 3])
    assert rows[1]["target"] == [.5, .5]
    assert rows[2]["target"] == [1, 0]
    assert rows[3]["target"] == [0, 1]
    assert {row["group"] for row in rows} == {"paper:paper1"}
    assert {row["role"] for row in rows} == {"test"}
