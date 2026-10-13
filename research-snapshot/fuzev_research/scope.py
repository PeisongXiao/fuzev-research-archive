"""Versioned training ownership and strictly external evaluation contracts."""

import hashlib
import json
import math

from .dimension_augmentation import PROVENANCE_FIELDS


REVISION = "2026-10-02-v2"
LEGACY_REVISION = REVISION
PREFERENCE_REVISION = "2026-10-05-v3-noul-preference"
PREFERENCE_ADAPTATION = "pairwise_preference_yes_no"
PREFERENCE_FILES = ("noul-preference-text.jsonl", "noul-preference-images.jsonl")
DIMENSION_REVISION = "2026-10-05-v4-dimension-choice"
DIMENSION_FILES = ("dimension-choice-text.jsonl", "dimension-choice-images.jsonl")
DIMENSION_METADATA = ("context",) + PROVENANCE_FIELDS
TRAINING_SOURCES = ("helpsteer2", "helpsteer3", "boolq", "mm-rlhf")
EXTERNAL_SOURCES = ("qasper", "vl-rewardbench")
PREPARATIONS = {
    "text.jsonl": ("text-splits.json", "text", "training", ("helpsteer2", "helpsteer3")),
    "boolq.jsonl": ("boolq-splits.json", "text", "training", ("boolq",)),
    "images.jsonl": ("image-splits.json", "image", "training", ("mm-rlhf",)),
    "qasper-external.jsonl": ("qasper-external-splits.json", "text", "external", ("qasper",)),
    "vl-rewardbench-choice.jsonl": ("vl-rewardbench-choice-splits.json", "image", "external", ("vl-rewardbench",)),
    "vl-rewardbench-noul.jsonl": ("vl-rewardbench-noul-splits.json", "image", "external", ("vl-rewardbench",)),
    "noul-preference-text.jsonl": ("noul-preference-text-splits.json", "text", "training", ("helpsteer2", "helpsteer3")),
    "noul-preference-images.jsonl": ("noul-preference-images-splits.json", "image", "training", ("mm-rlhf",)),
}
DIMENSION_PREPARATIONS = {
    "dimension-choice-text.jsonl": ("dimension-choice-text-splits.json", "text", "training", ("helpsteer2",)),
    "dimension-choice-images.jsonl": ("dimension-choice-images-splits.json", "image", "training", ("mm-rlhf",)),
}
ALL_PREPARATIONS = {**PREPARATIONS, **DIMENSION_PREPARATIONS}


def contract(augmentation=False, dimensions=False):
    if dimensions and not augmentation:
        raise ValueError("Dimension Choice requires the explicit preference-augmented scope")
    revision = DIMENSION_REVISION if dimensions else PREFERENCE_REVISION if augmentation else REVISION
    value = {"revision": revision, "training_sources": list(TRAINING_SOURCES),
            "external_sources": list(EXTERNAL_SOURCES), "external_roles": ["test"],
            "boolq_semantics": "boolean truth: True -> Yes, False -> No; not answerability",
            "qasper": {"original_split": "test", "yes_no": 239, "answerability": 1450,
                       "evidence": 2000, "seed": 20261002, "sampling_weights_required": True},
            "vl_rewardbench": {"same_examples": True, "methods": ["choice", "noul"],
                               "noul_semantics": "pairwise preference, not absolute correctness"},
            "noul_preference": {"adapter_version": "noul-relative-preference-v1",
                                "semantics": "relative pairwise preference, not absolute correctness or answerability",
                                "parent_roles": ["fit", "development", "calibration"],
                                "seed": 20261005, "role_budgets": {"fit": 4096, "development": 512, "calibration": 512},
                                "source_shares": {"helpsteer2": .25, "helpsteer3": .25, "mm-rlhf": .5}},
            "preparations": {name: {"manifest": value[0], "modality": value[1],
                                     "purpose": value[2], "sources": list(value[3])}
                             for name, value in ALL_PREPARATIONS.items()}}
    if not augmentation:
        value.pop("noul_preference")
        value["preparations"] = {name: spec for name, spec in value["preparations"].items()
                                 if name not in PREFERENCE_FILES}
    if dimensions:
        value["dimension_choice"] = {
            "adapter_version": "human-dimension-choice-v1",
            "semantics": "relative human dimension ratings; not absolute correctness or answerability",
            "parent_roles": ["fit", "development", "calibration"], "seed": 20261005,
            "role_budgets": {"fit": 4096, "development": 1024, "calibration": 1024},
            "source_shares": {"mm-rlhf": .75, "helpsteer2": .25},
            "dimensions": {"mm-rlhf": "faithfulness", "helpsteer2": "correctness"},
        }
    else:
        value["preparations"] = {name: spec for name, spec in value["preparations"].items()
                                 if name not in DIMENSION_FILES}
    return value


def validate_contract(value):
    if value not in (contract(), contract(augmentation=True), contract(augmentation=True, dimensions=True)):
        raise ValueError("Unknown or changed active dataset scope")
    return value


def contract_sha256(augmentation=False, dimensions=False):
    return hashlib.sha256(json.dumps(contract(augmentation, dimensions), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def required_preparations(stage, dataset_scope=None):
    if stage not in {"text", "mixed", "calibrate", "external"}:
        raise ValueError("Unknown preparation stage")
    active = validate_contract(dataset_scope) if dataset_scope is not None else contract()
    return tuple(name for name, (_, modality, purpose, _) in ALL_PREPARATIONS.items()
                 if name in active["preparations"] and (purpose == "external" if stage == "external" else
                     purpose == "training" and (stage != "text" or modality == "text")))


def validate_manifest(filename, manifest):
    if filename not in ALL_PREPARATIONS:
        raise ValueError("Unregistered active preparation")
    _, _, purpose, sources = ALL_PREPARATIONS[filename]
    # The byte-identical HelpSteer preparation predates the scope transition.
    revisions = ((DIMENSION_REVISION,) if filename in DIMENSION_FILES else
                 (PREFERENCE_REVISION,) if filename in PREFERENCE_FILES else (REVISION,))
    if filename != "text.jsonl" and (manifest.get("dataset_scope_revision") not in revisions
                                      or manifest.get("evaluation_scope") != purpose):
        raise ValueError("Preparation belongs to a superseded dataset scope")
    if filename in DIMENSION_FILES and manifest.get("dataset_purpose") != "training":
        raise ValueError("Dimension Choice manifests must remain training-only")
    if purpose == "external" and manifest.get("original_split") != "test":
        raise ValueError("External manifest is not the original official test split")
    for track, count in manifest.get("counts", {}).items():
        if track.startswith("excluded_"):
            continue
        if filename in DIMENSION_FILES and (type(count) is not int or count < 0):
            raise ValueError("Invalid dimension preparation counts")
        parts = track.split("/")
        if (len(parts) != 3 or parts[0] not in sources or purpose == "external" and parts[-1] != "test"
                or filename in PREFERENCE_FILES and (parts[1] != "noul" or parts[-1] not in {"fit", "development", "calibration"})
                or filename in DIMENSION_FILES and (parts[1] != "choice" or parts[-1] not in {"fit", "development", "calibration"})):
            raise ValueError("Inactive source/role in preparation manifest")
    return manifest


def validate_row(row, filename=None):
    source, role = row.get("source"), row.get("role")
    if source not in TRAINING_SOURCES + EXTERNAL_SOURCES:
        raise ValueError("Dataset is outside the active scope")
    if role not in {"fit", "development", "calibration", "test"}:
        raise ValueError("Unknown active split role")
    if filename is not None:
        if filename not in ALL_PREPARATIONS or source not in ALL_PREPARATIONS[filename][3]:
            raise ValueError("Source does not belong to this registered preparation")
        if row.get("modality") != ALL_PREPARATIONS[filename][1]:
            raise ValueError("Preparation modality mismatch")
    if source in EXTERNAL_SOURCES:
        if role != "test" or row.get("evaluation_scope") != "external" or row.get("original_split") != "test":
            raise ValueError("External examples may only use the original official test split")
        probability, weight = row.get("inclusion_probability"), row.get("sampling_weight")
        if (isinstance(probability, bool) or isinstance(weight, bool)
                or not isinstance(probability, (int, float)) or not isinstance(weight, (int, float))
                or not math.isfinite(probability) or not math.isfinite(weight)
                or not 0 < probability <= 1 or weight <= 0
                or not math.isclose(weight * probability, 1., rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("External evaluation requires valid inverse-probability sampling weights")
        if not isinstance(row.get("source_test_example_id"), str) or not row["source_test_example_id"]:
            raise ValueError("External records require an original official-test example ID")
        if source == "qasper" and (row.get("primitive") != "noul" or row.get("adaptation") not in
                                   {"yes_no", "answerability", "evidence"}):
            raise ValueError("Unknown external QASPER subtype")
        if source == "vl-rewardbench" and (row.get("primitive"), row.get("adaptation")) not in {
                ("choice", "official_heldout_preference"), ("noul", "pairwise_preference_yes_no")}:
            raise ValueError("VL-RewardBench Choice and Noul contracts must remain separate")
    elif row.get("evaluation_scope") == "external":
        raise ValueError("A training source cannot claim the external evaluation contract")
    if (filename in DIMENSION_FILES or row.get("dataset_scope_revision") == DIMENSION_REVISION
            or row.get("adapter_version") == "human-dimension-choice-v1"
            or row.get("adaptation") in {"faithfulness_pairwise_choice", "correctness_pairwise_choice"}):
        if (source not in {"helpsteer2", "mm-rlhf"} or role not in {"fit", "development", "calibration"}
                or filename is not None and filename not in DIMENSION_FILES
                or row.get("primitive") != "choice" or row.get("original_split") != "train"
                or row.get("dataset_scope_revision") != DIMENSION_REVISION
                or row.get("evaluation_scope") != "training" or row.get("dataset_purpose") != "training"
                or type(row.get("orientation_seed")) is not int or row["orientation_seed"] != 20261005):
            raise ValueError("Invalid training-only human-dimension Choice contract")
        from .dimension_augmentation import validate_dimension_view

        validate_dimension_view(row, group_roles={row.get("group"): role}, seed=20261005)
        for parent in row["parent_records"]:
            identifier = parent.get("id")
            if (not isinstance(identifier, str) or len(identifier) != 64
                    or any(char not in "0123456789abcdef" for char in identifier)
                    or source == "mm-rlhf" and (parent.get("primitive"), parent.get("adaptation")) != ("choice", "best_response")
                    or source == "helpsteer2" and (parent.get("primitive"), parent.get("attribute")) != ("ordinal", "correctness")):
                raise ValueError("Dimension view requires canonical original rating parents")
    if source in TRAINING_SOURCES and (row.get("adaptation") == PREFERENCE_ADAPTATION
                                      or filename in PREFERENCE_FILES):
        if (source not in {"helpsteer2", "helpsteer3", "mm-rlhf"}
                or row.get("primitive") != "noul" or row.get("adaptation") != PREFERENCE_ADAPTATION
                or role not in {"fit", "development", "calibration"}
                or row.get("dataset_scope_revision") != PREFERENCE_REVISION
                or row.get("dataset_purpose") != "training" or row.get("evaluation_scope") != "training"
                or row.get("adapter_version") != "noul-relative-preference-v1"
                or row.get("label_semantics") != "relative_pairwise_preference"
                or row.get("parent_role") != role or row.get("parent_source") != source
                or row.get("parent_primitive") != "choice"
                or row.get("candidate_permutation") not in ([0, 1], [1, 0])
                or row.get("candidates") != [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
                or row.get("target") not in ([1., 0.], [0., 1.], [.5, .5])):
            raise ValueError("Invalid training-only relative-preference Noul contract")
        parent = row.get("parent_id")
        if (not isinstance(parent, str) or len(parent) != 64
                or any(char not in "0123456789abcdef" for char in parent)):
            raise ValueError("Preference view requires an original parent identity")
        parent_target = row.get("parent_target")
        parent_ids = row.get("parent_candidate_ids")
        if (parent_target not in ([1., 0.], [0., 1.], [.5, .5])
                or not isinstance(parent_ids, list) or len(parent_ids) != 2
                or not all(isinstance(value, str) and value for value in parent_ids)
                or len(set(parent_ids)) != 2
                or any(type(index) is not int for index in row["candidate_permutation"])
                or row["target"] != [parent_target[index] for index in row["candidate_permutation"]]):
            raise ValueError("Preference orientation/target provenance mismatch")
    if source == "boolq" and (row.get("primitive") != "noul" or row.get("adaptation") != "boolean_truth"):
        raise ValueError("BoolQ must represent boolean truth, not answerability")
    if source == "boolq" and (row.get("target") not in ([1., 0.], [0., 1.]) or
                              row.get("candidates") != [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]):
        raise ValueError("BoolQ True/False must map exactly to Yes/No candidates")
    return row
