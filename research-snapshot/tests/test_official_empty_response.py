from copy import deepcopy
import json

import pytest

from fuzev_research.evaluate import metadata


def response_row():
    meta = {"id": "official-response-pair", "group": "prompt:fixture", "role": "test",
            "source": "helpsteer3", "revision": "pinned-fixture", "primitive": "choice",
            "modality": "text", "adaptation": None, "attribute": None,
            "candidates": [{"id": "a", "text": ""}, {"id": "b", "text": "An answer"}],
            "target": [0., 1.]}
    return {**meta, "group_": meta["group"], "metadata": json.dumps(meta)}


@pytest.mark.parametrize("index", [0, 1])
def test_official_empty_response_is_preserved_without_changing_target(index):
    row = response_row()
    meta = json.loads(row["metadata"])
    meta["candidates"][0]["text"] = "An answer"
    meta["candidates"][index]["text"] = ""
    row["metadata"] = json.dumps(meta)
    before = deepcopy(row)
    assert metadata(row) == meta
    assert row == before


@pytest.mark.parametrize("field,value", [
    ("source", "helpsteer2"), ("source", "vl-rewardbench"),
    ("primitive", "ordinal"), ("modality", "image"),
    ("adaptation", "evidence"), ("attribute", "correctness"),
])
def test_empty_label_is_not_admitted_on_other_tracks(field, value):
    row = response_row()
    meta = json.loads(row["metadata"])
    meta[field] = value
    row[field] = value
    row["metadata"] = json.dumps(meta)
    with pytest.raises(ValueError, match="Invalid candidate semantics"):
        metadata(row)


@pytest.mark.parametrize("value", [None, 1, [], {}])
def test_official_response_must_still_be_a_string(value):
    row = response_row()
    meta = json.loads(row["metadata"])
    meta["candidates"][0]["text"] = value
    row["metadata"] = json.dumps(meta)
    with pytest.raises(ValueError, match="Invalid candidate semantics"):
        metadata(row)


@pytest.mark.parametrize("ids", [["b", "a"], ["a", "a"], ["yes", "no"], ["a", "b", "c"]])
def test_exception_requires_original_response_pair_identifiers_and_order(ids):
    row = response_row()
    meta = json.loads(row["metadata"])
    meta["candidates"] = [{"id": key, "text": "" if i == 0 else "answer"} for i, key in enumerate(ids)]
    row["metadata"] = json.dumps(meta)
    with pytest.raises(ValueError, match="Invalid candidate semantics"):
        metadata(row)
