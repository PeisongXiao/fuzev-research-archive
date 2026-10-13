import copy
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzev_research.data import ordinal_records, stable_id
from fuzev_research.dimension_augmentation import hs2_correctness_view, mm_faithfulness_views
from fuzev_research.scope import (
    ALL_PREPARATIONS, DIMENSION_FILES, DIMENSION_METADATA, DIMENSION_REVISION,
    PREPARATIONS, contract, required_preparations, validate_contract, validate_manifest, validate_row,
)


def dimension_view(source="mm-rlhf", scores=(5, 2)):
    """Synthetic annotations and parents only; no real corpus or acquisition files."""
    if source == "mm-rlhf":
        raw = {"id": "synthetic", "image": "synthetic.png", "question": "What is visible?",
               "models_output": ["First response", "Second response"], "faithfulness": list(scores),
               "final_ranking": [1, 2]}
        parent = {"source": source, "revision": "synthetic-revision", "group": "synthetic-group",
                  "role": "fit", "original_split": "train", "modality": "image", "primitive": "choice",
                  "adaptation": "best_response", "context": [],
                  "question": "Which response best answers the question about the image?\n" + raw["question"],
                  "candidates": [{"id": str(i), "text": text} for i, text in enumerate(raw["models_output"])],
                  "target": [1., 0.], "image_sha256": "f" * 64, "image_path": "synthetic.png",
                  "source_image_id": raw["image"], "annotation_id": raw["id"], "official_holdout_overlap": False}
        parent["id"] = stable_id(source, parent["group"], parent["context"], parent["question"],
                                 json.dumps(parent["candidates"]), "choice", parent["target"])
        provenance = {"source": source, "revision": parent["revision"], "filename": "data.jsonl", "sha256": "e" * 64}
        return mm_faithfulness_views(parent, raw, provenance, group_roles={parent["group"]: "fit"})[0]
    raws = [{"prompt": "A synthetic request", "response": text, "correctness": score}
            for text, score in zip(("First response", "Second response"), scores)]
    with patch("fuzev_research.data.acquisition", return_value={source: {"revision": "synthetic-revision"}}):
        parents = [next(ordinal_records(raw, "train")) for raw in raws]
    for parent in parents:
        parent["role"] = "fit"
    provenance = {"source": source, "revision": parents[0]["revision"],
                  "filename": "train.jsonl.gz", "sha256": "e" * 64}
    return hs2_correctness_view(parents, raws, provenance, row_indices=[4, 7],
                               group_roles={parents[0]["group"]: "fit"})




def test_v4_is_explicit_and_only_adds_training_preparations():
    with pytest.raises(ValueError, match="explicit"):
        contract(dimensions=True)
    active = contract(augmentation=True, dimensions=True)
    assert validate_contract(active) is active
    assert active["revision"] == DIMENSION_REVISION
    assert len(active["preparations"]) == 10
    assert set(ALL_PREPARATIONS) - set(PREPARATIONS) == set(DIMENSION_FILES)
    assert "dimension-choice-text.jsonl" in required_preparations("text", active)
    assert "dimension-choice-images.jsonl" not in required_preparations("text", active)
    assert set(DIMENSION_FILES) <= set(required_preparations("mixed", active))
    assert required_preparations("external", active) == required_preparations("external")
    active["dimension_choice"]["role_budgets"]["fit"] = 1
    with pytest.raises(ValueError, match="scope"):
        validate_contract(active)


@pytest.mark.parametrize("source,scores,filename", [
    ("mm-rlhf", (5, 2), "dimension-choice-images.jsonl"),
    ("mm-rlhf", (3, 3), "dimension-choice-images.jsonl"),
    ("helpsteer2", (4, 1), "dimension-choice-text.jsonl"),
    ("helpsteer2", (2.5, 2.5), "dimension-choice-text.jsonl"),
])
def test_valid_relative_ratings_orientation_and_ties(source, scores, filename):
    row = dimension_view(source, scores)
    before = copy.deepcopy(row)
    assert validate_row(row, filename) is row
    assert validate_row(row) is row
    assert row == before
    assert row["label_semantics"].startswith("relative_human_")
    assert "context" in DIMENSION_METADATA and "rating_record_ids" in DIMENSION_METADATA


@pytest.mark.parametrize("key,value", [
    ("role", "test"), ("source", "qasper"), ("source", "helpsteer3"),
    ("original_split", "validation"), ("primitive", "noul"), ("context", [{"role": "user", "content": "changed"}]),
    ("dataset_scope_revision", "2026-10-05-v3-noul-preference"), ("dataset_purpose", "external"),
    ("orientation_seed", True), ("label_semantics", "absolute_correctness"),
    ("candidate_permutation", [True, False]), ("dimension_scores", [float("nan"), 2]),
    ("source_provenance", {}), ("source_annotations", []), ("parent_ids", ["f" * 64]),
    ("parent_record_sha256s", ["f" * 64]), ("rating_record_ids", ["f" * 64, "e" * 64]),
])
def test_rejects_changed_ownership_inputs_and_rating_provenance(key, value):
    row = dimension_view()
    row[key] = value
    with pytest.raises(ValueError):
        validate_row(row)


def test_parent_role_and_image_eligibility_are_not_relaxed():
    for change in ("role", "official_holdout_overlap", "target"):
        row = dimension_view()
        row["parent_records"][0][change] = {"role": "test", "official_holdout_overlap": True, "target": [0., 1.]}[change]
        with pytest.raises(ValueError):
            validate_row(row)


def test_dimension_views_cannot_be_registered_as_original_preparations():
    for source, filename in (("mm-rlhf", "images.jsonl"), ("helpsteer2", "text.jsonl")):
        with pytest.raises(ValueError, match="training-only"):
            validate_row(dimension_view(source, (4, 2)), filename)


def test_dimension_manifest_roles_and_revisions_are_training_only():
    good = {"dataset_scope_revision": DIMENSION_REVISION, "evaluation_scope": "training",
            "dataset_purpose": "training", "counts": {"mm-rlhf/choice/fit": 1}}
    assert validate_manifest("dimension-choice-images.jsonl", good) is good
    for changes in ({"dataset_scope_revision": "2026-10-02-v2"}, {"dataset_purpose": "external"},
                    {"counts": {"mm-rlhf/choice/test": 1}}, {"counts": {"mm-rlhf/noul/fit": 1}},
                    {"counts": {"helpsteer2/choice/fit": 1}}, {"counts": {"mm-rlhf/choice/fit": True}}):
        with pytest.raises(ValueError):
            validate_manifest("dimension-choice-images.jsonl", {**good, **changes})
