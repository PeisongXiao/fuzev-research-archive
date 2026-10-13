"""Composite admission: bounded snapshots at runtime, live proofs at freeze/eval."""

import ast
import hashlib
import json
import math
import struct
from pathlib import Path

from .features import file_sha256


PROCEDURE = "composite-independent-heads-v1"
PRIMITIVES = ("choice", "noul", "ordinal")
ROLES = ("fit", "development", "calibration")
FIELDS = dict(zip(ROLES, ("fit_membership", "development_membership", "calibration_membership")))
ARTIFACTS = ("head.safetensors", "manifest.json", "calibration.json")
CODE_FILES = {"train.py", "features.py", "head.py", "metrics.py"}
COMPOSITE_MARKERS = {"training_procedure", "per_head_provenance", "membership_semantics", "lineage_snapshot",
                     "migration_binding", "parent_binding", "corpus_binding", "final_dev_audit", "unchanged_membership_proof"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def require(condition, message):
    if not condition:
        raise ValueError("Composite provenance: " + message)


def membership(value):
    require(isinstance(value, dict) and type(value.get("records")) is int and value["records"] > 0
            and is_sha(value.get("ordered_sha256")), "invalid role membership")


def accepted_gate(value):
    return (isinstance(value, dict) and value.get("eligible") is True
            and value.get("selection_role") == "development"
            and value.get("external_selection") is False and value.get("calibration_selection") is False
            and bool(value.get("checks")) and all(item.get("passed") is True for item in value["checks"]))


def temperature_bits(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, "invalid temperature")
    return struct.pack("!d", value).hex()


def head_sha(bundle, primitive):
    result = hashlib.sha256()
    for name, tensor in bundle.heads[primitive].state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def artifact_sha(directory):
    directory = Path(directory)
    for name in ARTIFACTS:
        require((directory / name).stat().st_size <= (64 << 20 if name == "head.safetensors" else 1 << 20),
                "source artifact exceeds bound")
    return {name: file_sha256(directory / name) for name in ARTIFACTS}


def snapshot(directory, bundle):
    """Parent bytes and tensor digests, verified live before qualification."""
    return {"sha256": artifact_sha(directory),
            "manifest_json": (Path(directory) / "manifest.json").read_text(),
            "calibration_json": (Path(directory) / "calibration.json").read_text(),
            "head_sha256": {primitive: head_sha(bundle, primitive) for primitive in PRIMITIVES}}


def snapshot_values(value):
    require(isinstance(value, dict) and set(value.get("sha256", {})) == set(ARTIFACTS)
            and all(is_sha(item) for item in value["sha256"].values()), "missing source artifact pins")
    decoded = {}
    for name in ("manifest", "calibration"):
        raw = value.get(name + "_json")
        require(isinstance(raw, str) and len(raw.encode()) <= 1 << 20, "unbounded source snapshot")
        value.setdefault("observed_sha256", {})[name + ".json"] = hashlib.sha256(raw.encode()).hexdigest()
        decoded[name] = json.loads(raw)
    require(decoded["manifest"].get("weights_sha256") == value["sha256"]["head.safetensors"], "parent weight pin mismatch")
    require("training_procedure" not in decoded["manifest"] and "per_head_provenance" not in decoded["manifest"], "unexpected recursive parent policy")
    require(set(value.get("head_sha256", {})) == set(PRIMITIVES)
            and all(is_sha(item) for item in value["head_sha256"].values()), "missing source tensor digests")
    temperatures = decoded["calibration"].get("temperatures", {})
    require(decoded["calibration"].get("schema_version") == 1 and set(temperatures) == set(PRIMITIVES), "incomplete source calibration")
    for temperature in temperatures.values():
        temperature_bits(temperature)
    return decoded["manifest"], temperatures


def compact_ast(node):
    """Match Python 3.13's parsed-source dump on the Python 3.12 CUDA runtime."""
    if isinstance(node, ast.AST):
        fields = []
        for name, value in ast.iter_fields(node):
            if value is None and getattr(type(node), name, ...) is None:
                continue
            if isinstance(value, list) and not value:
                continue
            fields.append(name + "=" + compact_ast(value))
        return type(node).__name__ + "(" + ", ".join(fields) + ")"
    if isinstance(node, list):
        return "[" + ", ".join(compact_ast(value) for value in node) + "]"
    return repr(node)


def current_computation_proof():
    root = Path(__file__).parent
    proof = {}
    for filename, definition in (("native.py", "NativeBackend"), ("extraction.py", "write_rows"),
                                 ("extraction.py", "PreparedRows"), ("extraction.py", "PIPELINE_POLICY")):
        source = (root / filename).read_text()
        if definition == "PIPELINE_POLICY":
            node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                        and any(getattr(target, "id", None) == definition for target in node.targets))
            node = node.value
        else:
            node = next(node for node in ast.parse(source).body if getattr(node, "name", None) == definition)
        proof[filename + ":" + definition] = hashlib.sha256(compact_ast(node).encode()).hexdigest()
    for filename in ("encoding.py", "preparation.py", "templates/chat_template.jinja"):
        proof[filename] = file_sha256(root / filename)
    return proof


def bind_transition(previous, current):
    """Qualify the immediate parent and immutable shard links, never regenerate."""
    from .scope_transition import COMPUTATION_FIELDS

    old, new = previous.identity, current.identity
    require(all(old.get(key) == new.get(key) for key in COMPUTATION_FIELDS + ("datasets",)), "computation/dataset identity changed")
    require({key: value for key, value in old["runtime_sources"].items() if key != "recipes"}
            == {key: value for key, value in new["runtime_sources"].items() if key != "recipes"}, "native runtime changed")
    path = current.directory / "scope-transition.json"
    require(path.stat().st_size <= 32 << 20, "unbounded transition receipt")
    receipt = json.loads(path.read_text())
    old_sha, new_sha = (file_sha256(store.directory / "identity.json") for store in (previous, current))
    require(receipt.get("artifact_kind") == "compatible-noul-preference-extension"
            and receipt.get("schema_version") == 1
            and Path(receipt.get("source", "")).resolve() == previous.directory.resolve()
            and Path(receipt.get("destination", "")).resolve() == current.directory.resolve()
            and receipt.get("source_identity_sha256") == old_sha
            and receipt.get("destination_identity_sha256") == new_sha
            and receipt.get("source_runtime") == old["runtime_sources"]
            and receipt.get("new_runtime") == new["runtime_sources"], "transition is not bound to the immediate parent")
    require(receipt.get("computation_proof") == current_computation_proof(), "incomplete or changed computation proof")
    require(receipt.get("records_reused") == previous.db.execute("SELECT count(*) FROM records").fetchone()[0], "reused record count mismatch")
    prior = previous.directory / "scope-transition.json"
    require(receipt.get("prior_scope_transition") == (json.loads(prior.read_text()) if prior.exists() else None), "producer-origin ancestry changed")
    expected = {f"shard-{row[0]:06d}.{suffix}" for row in previous.db.execute("SELECT DISTINCT shard FROM records")
                for suffix in ("json", "safetensors")}
    shards = receipt.get("shards", [])
    require(isinstance(shards, list) and len(shards) == len(expected)
            and {item.get("name") for item in shards} == expected, "incomplete original shard inventory")
    for item in shards:
        source, destination = (store.directory / item["name"] for store in (previous, current))
        a, b = source.stat(), destination.stat()
        require((a.st_dev, a.st_ino, a.st_size) == (b.st_dev, b.st_ino, item.get("bytes"))
                and a.st_size == item["bytes"] and is_sha(item.get("sha256")), "original shard is not the pinned hardlink")
        item["observed_sha256"] = file_sha256(source)
    return {"receipt_sha256": file_sha256(path), "source_identity_sha256": old_sha,
            "destination_identity_sha256": new_sha, "computation_proof": receipt["computation_proof"],
            "records_reused": receipt["records_reused"], "shards_verified": len(shards)}


def validate_composite(bundle, manifest, temperatures, *, features=None, previous_features=None, corpus=False):
    """Runtime uses snapshots only. Freeze/evaluator must also set corpus=True."""
    if not COMPOSITE_MARKERS.intersection(manifest):
        return
    require(manifest.get("training_procedure") == PROCEDURE, "unknown or missing composite policy")
    require(manifest.get("artifact_kind") == "calibrated-reference-head", "composite is not a qualified reference")
    require(len(json.dumps(manifest, allow_nan=False).encode()) <= 1 << 20, "unbounded composite manifest")
    heads = manifest.get("per_head_provenance", {})
    require(set(heads) == set(PRIMITIVES) and set(temperatures) == set(PRIMITIVES), "partial head lineage")
    require(is_sha(manifest.get("operator_sha256")), "missing actual operator hash")
    snapshots = manifest.get("lineage_snapshot", {})
    require(snapshots.get("verification") == "publication-and-corpus-aware-evaluation", "missing qualification snapshot")
    old_manifest, old_temperatures = snapshot_values(snapshots.get("frozen"))
    require(old_manifest == manifest.get("frozen_reference", {}).get("manifest")
            and snapshots["frozen"]["sha256"] == manifest["frozen_reference"].get("sha256"), "frozen lineage snapshot mismatch")
    require(old_manifest.get("artifact_kind") == "calibrated-reference-head" and old_manifest.get("stage") == "mixed"
            and old_manifest.get("calibration_role") == "calibration", "unqualified frozen parent snapshot")
    changed = {primitive for primitive, item in heads.items() if item.get("status") == "retrained"}
    require("noul" in changed and changed <= {"noul", "choice"}
            and set(snapshots.get("selected", {})) == changed, "invalid changed-head coverage")
    for role, field in FIELDS.items():
        membership(manifest.get(field))
        require(sum(heads[primitive].get("memberships", {}).get(role, {}).get("records", 0)
                    for primitive in PRIMITIVES) == manifest[field]["records"], "global/per-head union count mismatch")
    for primitive, item in heads.items():
        require(item.get("head_sha256") == head_sha(bundle, primitive), "current head tensor digest mismatch")
        require(set(item.get("memberships", {})) == set(ROLES), "partial per-head role coverage")
        for field in ("training_code", "calibration_code"):
            require(set(item.get(field, {})) == CODE_FILES and all(is_sha(value) for value in item[field].values()), "missing per-head code provenance")
        for role in ROLES:
            membership(item.get("memberships", {}).get(role))
        calibration = item.get("calibration", {})
        require(calibration.get("temperature_bits") == temperature_bits(temperatures[primitive]), "per-head calibration binding mismatch")
        if primitive not in changed:
            require(item.get("status") == "retained-frozen" and item.get("new_fit_visits") == 0
                    and item.get("source") == {key: manifest["frozen_reference"][key] for key in ("directory", "sha256")}
                    and item["head_sha256"] == snapshots["frozen"]["head_sha256"][primitive]
                    and item["training_code"] == old_manifest.get("training_code")
                    and item["calibration_code"] == old_manifest.get("calibration_code")
                    and calibration.get("status") == "retained"
                    and temperature_bits(temperatures[primitive]) == temperature_bits(old_temperatures[primitive]), "retained head/temperature lineage changed")
        else:
            checkpoint, checkpoint_temperatures = snapshot_values(snapshots["selected"][primitive])
            gate = item.get("development_acceptance", {})
            require(item.get("source", {}).get("sha256") == snapshots["selected"][primitive]["sha256"]
                    and item["head_sha256"] == snapshots["selected"][primitive]["head_sha256"][primitive]
                    and checkpoint.get("artifact_kind") == "training-checkpoint"
                    and checkpoint.get("primitive") == primitive
                    and checkpoint.get("operator_sha256") == manifest["operator_sha256"]
                    and accepted_gate(gate)
                    and checkpoint.get("frozen_reference") == {key: manifest["frozen_reference"][key] for key in ("directory", "sha256")}
                    and item["training_code"] == checkpoint.get("training_code")
                    and gate == checkpoint.get("development_acceptance"), "selected checkpoint lineage/acceptance mismatch")
            for role, field in (("fit", "fit_membership"), ("development", "development_membership")):
                require(item["memberships"][role] == checkpoint.get(field), "selected primitive membership mismatch")
            visits = item.get("actual_fit_visitation", {})
            require(visits == checkpoint.get("actual_fit_visitation") and type(item.get("new_fit_visits")) is int
                    and item["new_fit_visits"] == visits.get("visits", 0) and item["new_fit_visits"] > 0
                    and visits.get("membership") == item["memberships"]["fit"], "actual FIT visitation mismatch")
            require(calibration.get("status") == "refit" and calibration.get("input_role") == "calibration"
                    and calibration.get("membership") == item["memberships"]["calibration"], "changed-head CAL provenance missing")
            for other in set(PRIMITIVES) - {primitive}:
                require(snapshots["selected"][primitive]["head_sha256"][other] == snapshots["frozen"]["head_sha256"][other]
                        and temperature_bits(checkpoint_temperatures[other]) == temperature_bits(old_temperatures[other]), "stage did not freeze other heads")
    audit = manifest.get("final_dev_audit", {})
    require(audit.get("status") == "passed" and audit.get("input_role") == "development"
            and audit.get("selection_locked") is True and audit.get("candidate_search") is False
            and audit.get("membership") == manifest["development_membership"]
            and audit.get("head_sha256") == {primitive: head_sha(bundle, primitive) for primitive in PRIMITIVES}
            and audit.get("temperature_bits") == {primitive: temperature_bits(value) for primitive, value in temperatures.items()}
            and set(audit.get("heads", {})) == changed, "final calibrated DEV audit is missing or unbound")
    for primitive in changed:
        evidence = audit["heads"][primitive]
        require(evidence.get("membership") == heads[primitive]["memberships"]["development"]
                and accepted_gate(evidence.get("acceptance"))
                and evidence.get("checkpoint_sha256") == heads[primitive]["source"]["sha256"], "final calibrated DEV gate failed")
    require(manifest.get("membership_semantics") == "union of independent per-head role coverage, not joint fitting"
            and isinstance(manifest.get("migration_binding"), dict) and is_sha(manifest["migration_binding"].get("receipt_sha256"))
            and isinstance(manifest.get("parent_binding"), dict) and manifest["parent_binding"].get("verified_parents") == 5120,
            "missing publication union/migration/parent proof")
    if not corpus:
        return
    from .features import FeatureStore
    from .head import load_bundle
    from .preference_head_retrain import bind_frozen, check_extension, coverage_identity, read_roles

    opened = []
    try:
        bindings = manifest.get("corpus_binding", {})
        stores = []
        for label, supplied in (("previous", previous_features), ("current", features)):
            binding = bindings.get(label, {})
            directory = Path(binding.get("directory", "")).resolve()
            binding["observed_identity_sha256"] = file_sha256(directory / "identity.json")
            if supplied is None:
                supplied = FeatureStore(directory, json.loads((directory / "identity.json").read_text()), read_only=True)
                opened.append(supplied)
            require(supplied.read_only and supplied.directory.resolve() == directory, "wrong live corpus supplied")
            stores.append(supplied)
        old, current = stores
        require(current.identity == manifest.get("feature_identity"), "current feature computation identity differs")
        previous, roles = read_roles(old), read_roles(current)
        require(bind_transition(old, current) == manifest["migration_binding"], "live migration binding differs")
        require(check_extension(current, previous, roles) == manifest["parent_binding"], "live original parent semantics differ")
        global_ids, per_head, proof = coverage_identity(previous, roles, changed)
        require(all(manifest[field] == value for field, value in global_ids.items())
                and all(heads[primitive]["memberships"] == values for primitive, values in per_head.items())
                and manifest.get("unchanged_membership_proof") == proof["unchanged"], "exact ordered per-head/global union mismatch")
        for label, source, saved in (("frozen", manifest["frozen_reference"], snapshots["frozen"]), *(
                (primitive, heads[primitive]["source"], snapshots["selected"][primitive]) for primitive in sorted(changed))):
            directory = Path(source["directory"])
            require(artifact_sha(directory) == saved["sha256"], "live source artifact pin mismatch")
            original, original_manifest, _ = load_bundle(directory)
            require(snapshot(directory, original) == saved, "live source tensors/snapshot differ")
            if label == "frozen":
                bind_frozen(old, original_manifest, previous)
            else:
                visits = heads[label]["actual_fit_visitation"]
                counts_file = directory / "fit-visits.json"
                require(Path(visits.get("counts_file", "")).resolve() == counts_file.resolve()
                        and counts_file.stat().st_size <= 32 << 20
                        and file_sha256(counts_file) == visits.get("counts_file_sha256"), "FIT visitation file binding differs")
                counts = json.loads(counts_file.read_text()).get("records", [])
                rows = [row for row in roles["fit"] if row["primitive"] == label]
                from .train import membership_identity
                require([item.get("id") for item in counts] == [row["id"] for row in rows]
                        and all(type(item.get("visits")) is int and item["visits"] > 0 for item in counts)
                        and membership_identity(counts) == visits.get("counts_identity")
                        and sum(item["visits"] for item in counts) == visits["visits"], "actual visited FIT records differ")
    finally:
        for store in opened:
            store.close()
