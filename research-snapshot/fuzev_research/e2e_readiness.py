"""Metadata-only main-quality and unchanged-input gates; never perform inference.

The caller must checksum-pin the fixed subset before passing its parsed data.
This gate does not qualify an engine, a quantization, or an API response.
"""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path

from .evaluate import (HEAD_FILES, canonical_test_membership, check_hashes,
                       metadata, open_features, quality_gates, read_json,
                       require_final_ready, require_reference_origins, sha256,
                       task_track, test_rows)
from .features import file_sha256


SCOPE = "offline final-test quality only; not runtime qualification"


def require(condition, message):
    if not condition:
        raise ValueError("Jev E2E readiness: " + message)


def checked_quality(path, digest, config):
    """Bind a passing full native4 report to the actual one-head files."""
    require(file_sha256(path) == digest, "quality report checksum mismatch")
    report = read_json(path)
    sha256(report)
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1
            and report.get("scope") == SCOPE
            and report.get("track") == "frozen-reference-head"
            and report.get("primary_method") == "head_calibrated",
            "require the main processed-record calibrated-head report")
    reference = report["reference"]
    pins = reference["sha256"]
    require(set(pins) == set(HEAD_FILES), "pin all three frozen head files")
    require(config.get("head_id") == "reference"
            and config.get("manifest_sha256") == pins["manifest.json"]
            and config.get("calibration_sha256") == pins["calibration.json"],
            "service configuration uses another head/calibration")
    directory = Path(config["head_directory"])
    check_hashes(directory, pins)
    manifest = read_json(directory / "manifest.json")
    temperatures = read_json(directory / "calibration.json")["temperatures"]
    require(manifest == reference["manifest"] and temperatures == reference["temperatures"]
            and manifest.get("weights_sha256") == pins["head.safetensors"]
            and manifest.get("calibration_sha256") == pins["calibration.json"],
            "quality report belongs to another frozen artifact")
    require(manifest.get("architecture") == "shared-candidate-dot-product-v1"
            and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 2
            and type(manifest.get("learned_head_count")) is int and manifest["learned_head_count"] == 1
            and manifest.get("frozen_lm_head_trained") is False
            and manifest.get("artifact_kind") == "calibrated-reference-head"
            and manifest.get("stage") == "mixed" and manifest.get("calibration_role") == "calibration",
            "require one shared calibrated decision head with the LM head unchanged")
    cell = report["cells"]["native-4.05"]
    require(cell.get("id") == "native-4.05" and cell.get("status") == "evaluated"
            and cell.get("corpus_scope") == "full"
            and Path(cell["features"]).resolve() == Path(reference["features"]).resolve()
            and cell.get("feature_identity_sha256") == reference["feature_identity_sha256"]
            and cell.get("feature_identity") == manifest.get("feature_identity") == config.get("feature_identity"),
            "require the complete native4 reference cell, not a subset/transfer cell")
    require(cell.get("gates", {}).get("head_calibrated", {}).get("status") == "pass",
            "main native4 processed-record quality must pass before E2E")
    require(quality_gates(cell, cell["track_metadata"]) == cell["gates"],
            "reported quality flags disagree with the canonical per-track confidence gates")
    require(report.get("execution", {}).get("device") == "cpu",
            "require head-only processed-record validation")
    return report


def unchanged_inputs(original, reference, inputs, original_ready, reference_ready):
    """Prove compatibility from full TEST metadata and immutable prepared inputs."""
    require(inputs.get("feature_identity") == original.identity,
            "fixed inputs belong to another original corpus")
    ignored = {"dataset_scope", "runtime_sources"}
    fields = (set(original.identity) | set(reference.identity)) - ignored
    require(all(original.identity.get(key) == reference.identity.get(key) for key in fields),
            "original and trained-reference feature computation identities differ")
    before, after = canonical_test_membership(original), canonical_test_membership(reference)
    membership_keys = set(before) | set(after)
    require(all(before.get(key) == after.get(key) for key in membership_keys - {"dataset_scope_sha256"}),
            "the complete original TEST membership/semantics changed")
    require(inputs.get("external_membership") == before.get("external"),
            "fixed external evaluation membership changed")
    roots = {item["path"]: item for item in reference_ready["preparations"]}
    require(all(roots.get(item["path"]) == item for item in original_ready["preparations"]),
            "original prepared inputs/manifests were replaced")
    selected = {}
    for item in inputs["rows"]:
        row = item["row"]
        require(row.get("role") == "test" and row.get("id") and row["id"] not in selected,
                "fixed subset contains a non-TEST or duplicate record")
        selected[row["id"]] = row
    require(selected, "fixed subset must be nonempty")
    remaining = dict(selected)
    # Features omit some full prompt fields; compare whole prepared rows too.
    for prepared in original_ready["preparations"]:
        with Path(prepared["path"]).open("rb") as handle:
            for raw in handle:
                require(len(raw) <= 16 << 20, "prepared input exceeds its row bound")
                row = json.loads(raw)
                if row.get("id") in remaining:
                    require(row == remaining.pop(row["id"]),
                            "fixed input changes original context/options/labels/images")
    require(not remaining, "fixed inputs are missing from the original prepared TEST corpus")
    descriptions = {}
    for row in test_rows(reference):
        meta = metadata(row, scoped=reference.scoped)
        track = task_track(meta)
        description = {key: meta.get(key) for key in (
            "source", "primitive", "modality", "attribute", "adaptation", "evaluation_scope")}
        require(track not in descriptions or descriptions[track] == description,
                "ambiguous canonical task/metric semantics")
        descriptions[track] = description
    return {"test_membership": {key: before[key] for key in ("records", "ordered_sha256")},
            "track_counts": before["counts"], "subset_records": len(selected),
            "inputs_unchanged": True, "track_metadata": descriptions}


@contextmanager
def ready_reference(report_path, report_sha256, config, inputs):
    """Finish P0 and input preflight before a caller opens any HTTP/output path."""
    report = checked_quality(report_path, report_sha256, config)
    spec = report["reference"]
    with ExitStack() as stack:
        original = open_features(inputs["corpus"], inputs["feature_identity_sha256"])
        stack.callback(original.close)
        reference = open_features(spec["features"], spec["feature_identity_sha256"])
        stack.callback(reference.close)
        versions = [store.db.execute("PRAGMA data_version").fetchone()[0]
                    for store in (original, reference)]
        original_ready, reference_ready = require_final_ready(original), require_final_ready(reference)
        require(reference_ready == spec["evaluation_preparation"]
                and reference_ready == report["cells"]["native-4.05"]["preparation"],
                "reference readiness changed since processed-record validation")
        require(spec["manifest"].get("preparation") == reference_ready.get("training", reference_ready),
                "frozen head training readiness differs from the validated corpus")
        require_reference_origins(reference, spec["manifest"])
        if inputs["feature_identity_sha256"] != spec["feature_identity_sha256"]:
            reused = spec["manifest"].get("feature_origins", {}).get("reused") or {}
            require(reused.get("source_identity_sha256") == inputs["feature_identity_sha256"],
                    "augmented reference lacks the original fixed-input producer lineage")
        proof = unchanged_inputs(original, reference, inputs, original_ready, reference_ready)
        cell = report["cells"]["native-4.05"]
        require(cell["test_membership"] == proof["test_membership"]
                and set(cell["tracks"]) == set(cell["track_metadata"]) == set(proof["track_counts"]),
                "main quality omitted original heldout records/tracks")
        checks = {"macro_normalized_nll"}
        for track, description in proof["track_metadata"].items():
            require(type(cell["tracks"][track].get("records")) is int
                    and cell["tracks"][track]["records"] == proof["track_counts"][track],
                    "main quality omitted records within an original heldout track")
            require(all(cell["track_metadata"][track].get(key) == value for key, value in description.items()),
                    "quality report changes canonical task/metric semantics")
            metrics = ["ordinal_mae"] if description["primitive"] == "ordinal" else ["accuracy", "tie_aware_accuracy"]
            if description["adaptation"] == "evidence":
                metrics.append("evidence_f1")
            checks.update(track + "/" + metric for metric in metrics)
        require(set(cell["gates"]["head_calibrated"]["checks"]) == checks,
                "main quality omitted required confidence/evidence-F1 checks")
        receipt = {"scope": "P0 head quality and unchanged-input readiness only; not runtime qualification",
                   "quality_report_sha256": report_sha256, "main_quality": "pass",
                   "head_sha256": spec["sha256"], "learned_head_count": 1,
                   "original_feature_identity_sha256": inputs["feature_identity_sha256"],
                   "reference_feature_identity_sha256": spec["feature_identity_sha256"], **proof}

        def check_frozen():
            require(file_sha256(report_path) == report_sha256, "quality report changed during E2E")
            check_hashes(config["head_directory"], spec["sha256"])
            require_reference_origins(reference, spec["manifest"])
            for store, version, readiness, identity_pin in zip(
                    (original, reference), versions, (original_ready, reference_ready),
                    (inputs["feature_identity_sha256"], spec["feature_identity_sha256"])):
                require(store.db.execute("PRAGMA data_version").fetchone()[0] == version
                        and file_sha256(store.directory / "identity.json") == identity_pin
                        and require_final_ready(store) == readiness,
                        "corpus identity, ownership, or preparation changed during E2E")

        check_frozen()
        try:
            yield receipt
        finally:
            check_frozen()
