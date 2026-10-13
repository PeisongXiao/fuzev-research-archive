import copy

import pytest

from fuzev_research.scope import LEGACY_REVISION, PREFERENCE_REVISION, contract, required_preparations, validate_manifest, validate_row
from fuzev_research.preference_transition import legacy_contract


def view():
    return {"id": "b" * 64, "group": "group", "source": "helpsteer2", "revision": "revision",
            "role": "fit", "modality": "text", "primitive": "noul", "question": "Is A preferred to B?",
            "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
            "target": [0., 1.], "adaptation": "pairwise_preference_yes_no",
            "dataset_scope_revision": PREFERENCE_REVISION, "dataset_purpose": "training", "evaluation_scope": "training",
            "adapter_version": "noul-relative-preference-v1", "label_semantics": "relative_pairwise_preference",
            "parent_id": "a" * 64, "parent_role": "fit", "parent_source": "helpsteer2",
            "parent_primitive": "choice", "parent_candidate_ids": ["a", "b"],
            "parent_target": [1., 0.], "candidate_permutation": [1, 0]}


def test_correctly_remapped_preference_views_and_ties():
    row = view()
    assert validate_row(row, "noul-preference-text.jsonl") is row
    row["parent_target"] = row["target"] = [.5, .5]
    assert validate_row(row) is row


@pytest.mark.parametrize("key,value", [
    ("role", "test"), ("parent_role", "development"), ("parent_source", "mm-rlhf"),
    ("source", "qasper"), ("source", "boolq"), ("primitive", "choice"),
    ("label_semantics", "absolute_correctness"), ("parent_primitive", "noul"),
    ("parent_id", "missing"), ("target", [1., 0.]), ("parent_target", [.2, .8]),
    ("dataset_scope_revision", LEGACY_REVISION), ("candidate_permutation", [True, False]),
])
def test_rejects_leakage_or_relabelled_semantics(key, value):
    row = view()
    row[key] = value
    with pytest.raises(ValueError):
        validate_row(row, "noul-preference-text.jsonl")


def test_new_preparations_required_without_changing_external_membership():
    assert "noul-preference-text.jsonl" not in required_preparations("text")
    assert "noul-preference-text.jsonl" in required_preparations("text", contract(augmentation=True))
    assert "noul-preference-images.jsonl" not in required_preparations("text", contract(augmentation=True))
    assert set(required_preparations("external")) == {"qasper-external.jsonl", "vl-rewardbench-choice.jsonl", "vl-rewardbench-noul.jsonl"}
    assert contract()["training_sources"] == legacy_contract()["training_sources"]
    assert contract()["qasper"] == legacy_contract()["qasper"]


def test_byte_identical_old_manifest_accepted_only_for_original_preparations():
    manifest = {"dataset_scope_revision": LEGACY_REVISION, "evaluation_scope": "training",
                "counts": {"mm-rlhf/choice/fit": 1}}
    original = copy.deepcopy(manifest)
    assert validate_manifest("images.jsonl", manifest) == original
    assert manifest == original
    with pytest.raises(ValueError, match="superseded"):
        validate_manifest("noul-preference-images.jsonl", manifest)


def test_augmented_manifest_cannot_claim_test_or_wrong_primitive():
    for track in ("mm-rlhf/noul/test", "mm-rlhf/choice/fit"):
        manifest = {"dataset_scope_revision": PREFERENCE_REVISION, "evaluation_scope": "training", "counts": {track: 1}}
        with pytest.raises(ValueError, match="Inactive"):
            validate_manifest("noul-preference-images.jsonl", manifest)
