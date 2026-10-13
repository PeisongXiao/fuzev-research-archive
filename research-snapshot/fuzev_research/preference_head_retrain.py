"""Predeclared, independent head repairs on frozen features; never load an LLM.

The CLI is CUDA-only. Internal device arguments allow tiny synthetic CPU tests,
not a production CPU fallback. Rejected stages retain the frozen primitive.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .acquire import write_json
from .composite_head import (artifact_sha, bind_transition, digest, head_sha, snapshot,
                             temperature_bits, validate_composite)
from .features import FeatureStore, collate, file_sha256
from .head import CandidateHead, load_bundle, save_bundle
from .metrics import paired_group_interval, per_record
from .paths import project_path
from .scope import PREFERENCE_ADAPTATION, TRAINING_SOURCES, contract, validate_row
from .train import (epoch_batches, fit_temperature, membership_identity,
                    training_code_identity, validate)


ROLES = ("fit", "development", "calibration")
PRIMITIVES = ("choice", "noul", "ordinal")
FIELDS = dict(zip(ROLES, ("fit_membership", "development_membership", "calibration_membership")))
PROCEDURE = "composite-independent-heads-v1"
OUTPUT = "weights/cuda-preference-repair-20261005"
PROFILE = {
    "noul": {"seed": 43, "max_epochs": 10, "patience": 2, "fresh": True},
    "choice": {"seed": 29, "max_epochs": 3, "patience": None, "fresh": False},
}
CHOICE_TRACKS = {
    "helpsteer2/choice/text/main", "helpsteer3/choice/text/main",
    "mm-rlhf/choice/image/best_response", "mm-rlhf/choice/image/pairwise_preference",
}
NOUL_TRACKS = {"boolq/noul/text/boolean_truth", *(
    f"{source}/noul/{modality}/{PREFERENCE_ADAPTATION}"
    for source, modality in (("helpsteer2", "text"), ("helpsteer3", "text"), ("mm-rlhf", "image"))
)}


def track(row):
    return "/".join((row["source"], row["primitive"], row["modality"],
                     row.get("attribute") or row.get("adaptation") or "main"))


def check_rows(rows, role, primitive=None):
    if role not in ROLES or not rows:
        raise ValueError("Need nonempty training-source FIT/DEV/CAL rows")
    ids = set()
    for row in rows:
        if (row.get("role") != role or row.get("source") not in TRAINING_SOURCES
                or row.get("primitive") not in PRIMITIVES
                or row.get("modality") not in {"text", "image"}
                or (primitive is not None and row["primitive"] != primitive)
                or not row.get("group_") or not row.get("id") or row["id"] in ids
                or row.get("evaluation_scope") == "external"
                or row.get("dataset_purpose") not in (None, "training")):
            raise ValueError("Invalid role, source, primitive, or duplicate training record")
        ids.add(row["id"])


def check_roles(roles):
    if set(roles) != set(ROLES):
        raise ValueError("FIT, DEV and CAL ownership is required")
    ownership, ids = {}, set()
    for role, rows in roles.items():
        check_rows(rows, role)
        for row in rows:
            group = row["group_"]
            if ownership.setdefault(group, role) != role or row["id"] in ids:
                raise ValueError("Group/record leakage across FIT, DEV or CAL")
            ids.add(row["id"])


def read_roles(features):
    if not features.read_only:
        raise ValueError("Repairs require a read-only feature store")
    features.require_ready("calibrate")
    # Audit ownership without reading heldout tensors, answers or predictions.
    if features.db.execute(
            "SELECT group_ FROM records GROUP BY group_ HAVING COUNT(DISTINCT role)>1 LIMIT 1").fetchone():
        raise ValueError("Feature group leakage, including heldout ownership")
    if features.db.execute(
            "SELECT r.id FROM records r LEFT JOIN groups_ g ON r.group_=g.id "
            "WHERE g.id IS NULL OR r.role!=g.role LIMIT 1").fetchone():
        raise ValueError("Feature group ownership mismatch")
    roles = {role: features.records(role) for role in ROLES}
    check_roles(roles)
    for role in ROLES:
        for indexed in features.iter_records(role, include_metadata=True):
            metadata = json.loads(indexed["metadata"])
            validate_row(metadata)
            if (any(metadata.get(key) != indexed[key] for key in
                    ("id", "role", "primitive", "modality", "source"))
                    or metadata.get("group") != indexed["group_"]):
                raise ValueError("Indexed training metadata differs from its parent record")
    return roles


def prove_memberships(previous, current):
    """Exact old locations/order remain reusable; only admitted Noul rows grow."""
    check_roles(previous)
    check_roles(current)
    proof = {"unchanged": {}, "per_head": {}, "added": {}}
    for primitive in PRIMITIVES:
        proof["per_head"][primitive] = {"previous": {}, "current": {}}
        for role in ROLES:
            old = [row for row in previous[role] if row["primitive"] == primitive]
            new = [row for row in current[role] if row["primitive"] == primitive]
            check_rows(old, role, primitive)
            check_rows(new, role, primitive)
            before, after = membership_identity(old), membership_identity(new)
            proof["per_head"][primitive]["previous"][role] = before
            proof["per_head"][primitive]["current"][role] = after
            if primitive != "noul":
                if before != after:
                    raise ValueError(f"Unchanged {primitive}/{role} membership differs")
                proof["unchanged"][f"{primitive}/{role}"] = before
            else:
                if any(row["source"] != "boolq" for row in old):
                    raise ValueError("The frozen Noul control must contain only original BoolQ")
                old_ids = {row["id"] for row in old}
                if membership_identity(row for row in new if row["id"] in old_ids) != before:
                    raise ValueError("Original BoolQ membership changed")
                added = [row for row in new if row["id"] not in old_ids]
                if not added or any(row.get("adaptation") != PREFERENCE_ADAPTATION
                                    or row["source"] not in {"helpsteer2", "helpsteer3", "mm-rlhf"}
                                    for row in added):
                    raise ValueError("Only the approved relative-preference Noul extension is allowed")
                proof["added"][role] = membership_identity(added)
    return proof


def check_extension(features, previous, current):
    parents = {row["id"]: row for rows in previous.values() for row in rows}
    old_ids = set(parents)
    budgets = {"fit": 4096, "development": 512, "calibration": 512}
    for role, budget in budgets.items():
        added = {row["id"] for row in current[role]} - old_ids
        counts = Counter()
        for row in features.iter_records(role, primitive="noul", include_metadata=True):
            if row["id"] not in added:
                continue
            metadata = json.loads(row["metadata"])
            parent = parents.get(metadata.get("parent_id"))
            if (not parent or parent["primitive"] != "choice" or parent["role"] != role
                    or parent["group_"] != row["group_"] or parent["source"] != row["source"]
                    or not metadata.get("has_baseline")):
                raise ValueError("Preference parent ownership or raw label-token control is missing")
            counts[row["source"]] += 1
        expected = {"mm-rlhf": budget // 2, "helpsteer2": budget // 4, "helpsteer3": budget // 4}
        if counts != expected or len(added) != budget:
            raise ValueError("Relative-preference membership must match the approved 5120 budgets")
    return verify_parent_preparations(features, previous, current)


def verify_parent_preparations(features, previous, current):
    from .noul_augmentation import ADAPTER_VERSION, SEED, augment_parent

    readiness = features.require_ready("calibrate")
    pinned = {Path(item["path"]).name: item for item in readiness["preparations"]}
    old = {row["id"]: row for rows in previous.values() for row in rows}
    added = {row["id"]: row for rows in current.values() for row in rows if row["id"] not in old}
    children, fingerprints = {}, {}

    def prepared(name):
        expected = pinned[name]
        path = project_path("data/prepared/" + name)
        if Path(expected["path"]).resolve() != path.resolve():
            raise ValueError("Parent preparation is not the canonical pinned path")
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as handle:
            for line in handle:
                if len(line) > 1 << 20:
                    raise ValueError("Parent preparation record exceeds its bound")
                digest.update(line)
                count += 1
                yield json.loads(line), hashlib.sha256(line).hexdigest()
        if digest.hexdigest() != expected["sha256"] or count != expected["records"]:
            raise ValueError("Pinned parent/child preparation changed")
        fingerprints[name] = {"sha256": digest.hexdigest(), "records": count}

    for name in ("noul-preference-text.jsonl", "noul-preference-images.jsonl"):
        for child, _ in prepared(name):
            indexed = added.get(child["id"])
            if (indexed is None or child["id"] in children or child["role"] not in ROLES
                    or child["role"] != indexed["role"] or child["group"] != indexed["group_"]):
                raise ValueError("Preference child is outside the exact FIT/DEV/CAL membership")
            children[child["id"]] = child
    if set(children) != set(added):
        raise ValueError("Preference children are not accounted for by the pinned preparation")
    by_parent = {child["parent_id"]: child for child in children.values()}
    if len(by_parent) != len(children):
        raise ValueError("Duplicate original preference parent")
    matched = set()
    for name in ("text.jsonl", "images.jsonl"):
        for parent, line_sha256 in prepared(name):
            child = by_parent.get(parent["id"])
            if child is None:
                continue
            indexed = old.get(parent["id"])
            if (indexed is None or indexed["primitive"] != "choice" or parent["role"] not in ROLES
                    or indexed["role"] != parent["role"] or indexed["group_"] != parent["group"]
                    or parent["id"] in matched):
                raise ValueError("Original parent ownership differs from the frozen corpus")
            # Replay the existing pure adapter in memory, not preparation/extraction.
            expected = augment_parent(parent, seed=SEED)
            expected.update(parent_path=str(project_path("data/prepared/" + name).resolve()),
                            parent_line_sha256=line_sha256, parent_preparation_sha256=pinned[name]["sha256"])
            if digest(expected) != digest(child):
                raise ValueError("Preference context/responses/targets/image or parent fingerprints differ")
            matched.add(parent["id"])
    if matched != set(by_parent):
        raise ValueError("Missing original canonical preference parents")
    return {"verified_parents": len(matched), "preparations": fingerprints,
            "adapter_version": ADAPTER_VERSION, "seed": SEED,
            "adapter_sha256": file_sha256(Path(__file__).with_name("noul_augmentation.py"))}


def bind_frozen(features, manifest, roles):
    if (manifest.get("artifact_kind") != "calibrated-reference-head"
            or manifest.get("stage") != "mixed" or manifest.get("precision") != "float32"
            or manifest.get("calibration_role") != "calibration"
            or manifest.get("feature_identity") != features.identity
            or manifest.get("preparation") != features.require_ready("calibrate")
            or manifest.get("feature_origins") != features.producer_origins()):
        raise ValueError("Frozen reference does not match the complete previous corpus")
    for role, field in FIELDS.items():
        if manifest.get(field) != membership_identity(roles[role]):
            raise ValueError(f"Frozen reference {role} membership mismatch")
    for field in ("training_code", "calibration_code"):
        code = manifest.get(field, {})
        if set(code) != set(training_code_identity()) or any(
                not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value) for value in code.values()):
            raise ValueError("Frozen code provenance is incomplete")


def assert_unchanged(reference, candidate, changed, temperatures, previous_temperatures):
    for primitive in PRIMITIVES:
        if primitive not in changed and (head_sha(reference, primitive) != head_sha(candidate, primitive)
                or struct.pack("!d", temperatures[primitive]) != struct.pack("!d", previous_temperatures[primitive])):
            raise ValueError(f"Unchanged {primitive} weights/temperature are not byte-identical")


def initialize_stage(frozen, primitive):
    profile = PROFILE[primitive]
    torch.manual_seed(profile["seed"])
    bundle = copy.deepcopy(frozen).float()
    if profile["fresh"]:
        old = bundle.heads[primitive]
        bundle.heads[primitive] = CandidateHead(old.hidden_size, old.projection_size)
    bundle.requires_grad_(False)
    bundle.heads[primitive].requires_grad_(True)
    return bundle


def choice_weights(rows):
    check_rows(rows, "fit", "choice")
    counts = Counter(track(row) for row in rows if row["source"] == "mm-rlhf")
    best, pair = ("mm-rlhf/choice/image/" + subtype for subtype in ("best_response", "pairwise_preference"))
    if set(counts) != {best, pair} or not counts[best] or not counts[pair]:
        raise ValueError("Choice weighting needs both MM FIT adaptations")
    total = counts[best] + counts[pair]
    return {"text": 1., best: total / (2 * counts[best]), pair: total / (2 * counts[pair])}


def train_epoch(bundle, features, rows, primitive, seed, device, optimizer, weights=None):
    check_rows(rows, "fit", primitive)
    bundle.train()
    visits, loss_sum = Counter(), 0.
    for active, batch_rows in epoch_batches(rows, seed, mixed=True, batch_size=256):
        if active != primitive:
            raise ValueError("Independent stage received another primitive")
        batch = collate(features, batch_rows, device)
        logits = bundle(active, batch["query"], batch["candidates"], batch["mask"])
        logp = torch.log_softmax(logits, dim=1)
        loss = -(batch["targets"] * logp.masked_fill(~batch["mask"], 0)).sum(dim=1)
        loss = loss / batch["mask"].sum(dim=1).float().log()
        if weights is not None:
            factors = torch.tensor([weights["text"] if row["modality"] == "text" else weights[track(row)]
                                    for row in batch_rows], dtype=torch.float32, device=device)
            loss = loss * factors
        loss = loss.mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite independent-head fitting loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.heads[primitive].parameters(), 1.)
        optimizer.step()
        visits.update(row["id"] for row in batch_rows)
        loss_sum += float(loss.detach()) * len(batch_rows)
    if set(visits) != {row["id"] for row in rows}:
        raise ValueError("Not every applicable FIT record was visited")
    bundle.eval()
    return visits, loss_sum / sum(visits.values())


@torch.inference_mode()
def dev_evidence(bundle, features, rows, temperatures, device, raw_control=False):
    check_rows(rows, "development")
    summary = validate(bundle, features, rows, temperatures=temperatures, device=device)
    evidence = {}
    if raw_control:
        eligible = {row["id"] for row in features.iter_records("development", include_metadata=True)
                    if json.loads(row["metadata"]).get("has_baseline")}
        if not {row["id"] for row in rows} <= eligible:
            raise ValueError("Missing untouched DEV label-token baseline; never substitute zeros")
    for primitive in PRIMITIVES:
        selected = [row for row in rows if row["primitive"] == primitive]
        for start in range(0, len(selected), 256):
            batch_rows = selected[start:start + 256]
            batch = collate(features, batch_rows, device)
            logits = bundle(primitive, batch["query"], batch["candidates"], batch["mask"])
            values = per_record(logits.cpu().numpy(), batch["targets"].cpu().numpy(),
                                batch["mask"].cpu().numpy(), temperature=temperatures[primitive])
            raw = per_record(batch["baseline"].cpu().numpy(), batch["targets"].cpu().numpy(),
                             batch["mask"].cpu().numpy()) if raw_control else None
            for index, row in enumerate(batch_rows):
                item = evidence.setdefault(track(row), {"ids": [], "groups": [], "values": defaultdict(list),
                                                        "label_token": defaultdict(list)})
                item["ids"].append(row["id"])
                item["groups"].append(row["group_"])
                for destination, metrics in ((item["values"], values), (item["label_token"], raw)):
                    if metrics is not None:
                        for name in ("accuracy", "tie_aware_accuracy", "nll"):
                            value = float(metrics[name][index])
                            destination[name].append(value if math.isfinite(value) else None)
    return {"summary": summary, "tracks": evidence, "input_role": "development",
            "membership": membership_identity(rows)}


def acceptance(primitive, candidate, frozen):
    if primitive not in PROFILE:
        raise ValueError("Only the predeclared Noul/Choice stages may select checkpoints")
    expected = NOUL_TRACKS if primitive == "noul" else CHOICE_TRACKS
    if (candidate.get("input_role") != "development" or frozen.get("input_role") != "development"
            or candidate.get("membership") != frozen.get("membership")
            or set(candidate["tracks"]) != expected or set(frozen["tracks"]) != expected):
        raise ValueError("Acceptance requires aligned, complete predeclared DEV tracks")
    checks, paired = [], {}

    def check(name, value, limit, direction="minimum"):
        finite = value is not None and math.isfinite(value)
        checks.append({"check": name, "value": value, "limit": limit,
                       "passed": bool(finite and (value >= limit if direction == "minimum" else value <= limit))})

    improvements = []
    for key in sorted(expected):
        new, old = candidate["tracks"][key], frozen["tracks"][key]
        if new["ids"] != old["ids"] or new["groups"] != old["groups"] or not new["ids"]:
            raise ValueError("Unpaired DEV records or groups")
        control = old["values"] if primitive == "choice" or key.startswith("boolq/") else old["label_token"]
        means = {}
        for metric in ("accuracy", "tie_aware_accuracy"):
            a, b = np.asarray(control.get(metric, []), float), np.asarray(new["values"].get(metric, []), float)
            if len(a) != len(new["ids"]) or a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise ValueError("Missing or unaligned DEV accuracy observations")
            valid = np.isfinite(a)
            # Tied gold is retained, but never invented as unique-gold accuracy.
            if not valid.any() and metric == "accuracy":
                continue
            delta = float((b[valid] - a[valid]).mean()) if valid.any() else None
            means[metric] = delta
            check(f"{key}/{metric}/control_delta", delta, 0. if primitive == "choice" else -.02)
        if primitive == "choice":
            metric = "accuracy" if "accuracy" in means else "tie_aware_accuracy"
            a, b = np.asarray(control[metric], float), np.asarray(new["values"][metric], float)
            valid = np.isfinite(a) & np.isfinite(b)
            if len(set(np.asarray(new["groups"])[valid])) < 2:
                paired[key] = {"status": "unavailable", "reason": "need at least two independent groups"}
                check(f"{key}/paired_group_accuracy_lower", None, -.02)
            else:
                paired[key] = paired_group_interval(a, b, new["groups"])
                check(f"{key}/paired_group_accuracy_lower", paired[key]["low"], -.02)
            new_nll = candidate["summary"]["tracks"][key]["nll"]
            old_nll = frozen["summary"]["tracks"][key]["nll"]
            check(f"{key}/nll", new_nll, old_nll + 1e-6, "maximum")
            if key.startswith("mm-rlhf/"):
                improvements.append(old_nll - new_nll)
    new_macro = sum(candidate["summary"]["tracks"][key]["nll"] for key in expected) / len(expected)
    old_macro = sum(frozen["summary"]["tracks"][key]["nll"] for key in expected) / len(expected)
    if primitive == "noul":
        check("noul/macro_nll_strict_improvement", old_macro - new_macro, 0.)
        checks[-1]["passed"] = bool(math.isfinite(old_macro - new_macro) and new_macro < old_macro)
    else:
        check("choice/at_least_one_mm_nll_improvement", max(improvements), 1e-4)
    return {"eligible": all(item["passed"] for item in checks), "selection_role": "development",
            "checks": checks, "paired_group_intervals": paired,
            "candidate_macro_nll": new_macro, "frozen_macro_nll": old_macro,
            "external_selection": False, "calibration_selection": False}


def select_checkpoint(epochs):
    eligible = [epoch for epoch in epochs if epoch["acceptance"]["eligible"]
                and math.isfinite(epoch["development"]["macro_normalized_nll"])]
    selected = min(eligible, key=lambda epoch: epoch["development"]["macro_normalized_nll"]) if eligible else None
    return {"status": "accepted" if selected else "rejected", "selected": selected,
            "selection_role": "development", "retain_previous_head": selected is None,
            "failure": None if selected else "No DEV-eligible checkpoint in the predeclared stage"}


def visitation(rows, counts, epochs):
    ordered = [{"id": row["id"], "visits": counts[row["id"]]} for row in rows]
    if any(item["visits"] < 1 for item in ordered) or set(counts) != {row["id"] for row in rows}:
        raise ValueError("Invalid actual fitting visitation")
    return {"epochs": epochs, "visits": sum(counts.values()), "unique_records": len(rows),
            "membership": membership_identity(rows), "counts_identity": membership_identity(ordered)}, ordered


def stage_trial(features, roles, frozen, temperatures, primitive, control, output, source, device="cuda:0"):
    profile = PROFILE[primitive]
    rows = [row for row in roles["fit"] if row["primitive"] == primitive]
    development = [row for row in roles["development"] if row["primitive"] == primitive]
    check_rows(rows, "fit", primitive)
    check_rows(development, "development", primitive)
    bundle = initialize_stage(frozen, primitive).to(device)
    stage_temperatures = {**temperatures, primitive: 1.}
    weights = choice_weights(rows) if primitive == "choice" else None
    optimizer = torch.optim.AdamW(bundle.heads[primitive].parameters(), lr=1e-4, weight_decay=.01)
    provenance = {"primitive": primitive, "seed": profile["seed"], "learning_rate": 1e-4,
                  "brier_weight": 0., "teacher_loss": False,
                  "initialization": "fresh" if profile["fresh"] else "frozen-reference",
                  "frozen_reference": source, "feature_identity": features.identity,
                  "fit_membership": membership_identity(rows),
                  "development_membership": membership_identity(development),
                  "training_code": training_code_identity(), "operator_sha256": file_sha256(__file__),
                  "preparation": features.require_ready("mixed"),
                  "fit_track_counts": dict(Counter(track(row) for row in rows)),
                  "execution": {"device": str(device), "precision": "float32", "torch": torch.__version__,
                                "threads": torch.get_num_threads()},
                  "fit_weights": weights, "schedule": {**profile, "batch_size": 256, "mixed": True,
                                                        "epoch_seed": "seed+epoch", "minimum_improvement": 1e-4},
                  "optimizer": {"name": "AdamW", "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8,
                                "clip_grad_norm": 1.}}
    if features.scoped:
        provenance["feature_origins"] = features.producer_origins()
    directory = output / primitive
    directory.mkdir(exist_ok=False)
    write_json(directory / "development-control.json", control)
    epochs, counts, stale, best = [], Counter(), 0, None
    for epoch in range(profile["max_epochs"]):
        visits, loss = train_epoch(bundle, features, rows, primitive, profile["seed"] + epoch,
                                   device, optimizer, weights)
        counts.update(visits)
        assert_unchanged(frozen, bundle, {primitive}, stage_temperatures, temperatures)
        evidence = dev_evidence(bundle, features, development, stage_temperatures, device)
        gate = acceptance(primitive, evidence, control)
        checkpoint = directory / f"epoch-{epoch + 1:03d}"
        checkpoint.mkdir(exist_ok=False)
        visited, record_counts = visitation(rows, counts, epoch + 1)
        write_json(checkpoint / "fit-visits.json", {"records": record_counts})
        visited["counts_file"] = str(checkpoint / "fit-visits.json")
        visited["counts_file_sha256"] = file_sha256(checkpoint / "fit-visits.json")
        metadata = {**provenance, "artifact_kind": "training-checkpoint", "epoch": epoch + 1,
                    "actual_fit_visitation": visited, "development_acceptance": gate}
        save_bundle(checkpoint, bundle, metadata, stage_temperatures)
        result = {"epoch": epoch + 1, "checkpoint": str(checkpoint), "fit_loss": loss,
                  "development": evidence["summary"], "acceptance": gate,
                  "actual_fit_visitation": visited, "checkpoint_sha256": artifact_sha(checkpoint),
                  "weights_sha256": file_sha256(checkpoint / "head.safetensors")}
        epochs.append(result)
        current = evidence["summary"]["macro_normalized_nll"]
        improved = best is None or current < best - 1e-4
        best, stale = (current, 0) if improved else (best, stale + 1)
        write_json(directory / "trial.json", {"provenance": provenance, "epochs": epochs,
                                              **select_checkpoint(epochs), "completed": False})
        if profile["patience"] is not None and stale >= profile["patience"]:
            break
    selection = {**select_checkpoint(epochs), "primitive": primitive, "provenance": provenance,
                 "epochs": epochs, "executed_fit_visitation": visitation(rows, counts, len(epochs))[0],
                 "completed": True}
    write_json(directory / "trial.json", selection)
    del optimizer, bundle
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return selection


def coverage_identity(previous, current, changed):
    proof = prove_memberships(previous, current)
    memberships, per_head = {}, {}
    for primitive in PRIMITIVES:
        origin = "current" if primitive in changed else "previous"
        per_head[primitive] = proof["per_head"][primitive][origin]
    for role, field in FIELDS.items():
        covered = {row["id"] for primitive in PRIMITIVES
                   for row in (current if primitive in changed else previous)[role]
                   if row["primitive"] == primitive}
        union = membership_identity(row for row in current[role] if row["id"] in covered)
        if union != membership_identity(current[role]):
            raise ValueError("Cannot attach augmented global membership to an untrained/rejected Noul head")
        memberships[field] = union
    return memberships, per_head, proof


def compose_heads(frozen, temperatures, candidates, selections, previous, current):
    changed = set(candidates)
    if not changed or not changed <= {"noul", "choice"}:
        raise ValueError("Composition needs DEV-accepted Noul/Choice heads only")
    for primitive in changed:
        selection = selections[primitive]
        if (selection.get("status") != "accepted" or selection.get("selection_role") != "development"
                or not selection.get("selected") or not selection["selected"]["acceptance"]["eligible"]):
            raise ValueError("Ineligible checkpoints cannot be composed")
        assert_unchanged(frozen, candidates[primitive], {primitive}, temperatures, temperatures)
    memberships, per_head, proof = coverage_identity(previous, current, changed)
    bundle = copy.deepcopy(frozen)
    for primitive in changed:
        bundle.heads[primitive] = copy.deepcopy(candidates[primitive].heads[primitive])
    assert_unchanged(frozen, bundle, changed, temperatures, temperatures)
    return bundle, memberships, per_head, proof


def calibrate_changed(bundle, features, roles, changed, previous_temperatures, device):
    check_roles(roles)
    features.require_ready("calibrate")
    if not set(changed) <= {"noul", "choice"}:
        raise ValueError("Only changed Noul/Choice primitives may be calibrated")
    temperatures, visits = dict(previous_temperatures), {}
    for primitive in sorted(changed):
        rows = [row for row in roles["calibration"] if row["primitive"] == primitive]
        check_rows(rows, "calibration", primitive)
        if membership_identity(rows) != membership_identity(features.iter_records("calibration", primitive=primitive)):
            raise ValueError("Changed primitive must use all and only its CAL records")
        logits, targets, masks = [], [], []
        with torch.inference_mode():
            for offset in range(0, len(rows), 256):
                batch = collate(features, rows[offset:offset + 256], device)
                logits.append(bundle(primitive, batch["query"], batch["candidates"], batch["mask"]).cpu())
                targets.append(batch["targets"].cpu())
                masks.append(batch["mask"].cpu())
        width = max(value.shape[1] for value in logits)
        # CAL can mix two-candidate pairs with wider MM best-response decisions.
        def padded(values):
            return torch.cat([torch.nn.functional.pad(value, (0, width - value.shape[1])) for value in values])
        with torch.inference_mode(False):
            temperatures[primitive] = fit_temperature(padded(logits).clone(), padded(targets).clone(), padded(masks).clone())
        if not math.isfinite(temperatures[primitive]) or temperatures[primitive] <= 0:
            raise ValueError("Nonfinite changed-head calibration")
        visits[primitive] = {"input_role": "calibration", "records_visited": len(rows),
                             "membership": membership_identity(rows)}
    for primitive in set(PRIMITIVES) - set(changed):
        if struct.pack("!d", temperatures[primitive]) != struct.pack("!d", previous_temperatures[primitive]):
            raise ValueError("Unchanged primitive temperature changed")
    return temperatures, visits


def composite_metadata(frozen_manifest, source, features, memberships, per_head, proof, selections,
                       bundle, calibration_visits, *, temperatures, lineage_snapshot, migration_binding,
                       parent_binding, corpus_binding, final_dev_audit):
    lineage = {}
    for primitive in PRIMITIVES:
        selection = selections.get(primitive)
        accepted = selection is not None and selection["status"] == "accepted"
        item = {"status": "retrained" if accepted else "retained-frozen",
                "memberships": per_head[primitive], "head_sha256": head_sha(bundle, primitive),
                "source": source, "new_fit_visits": 0,
                "training_code": frozen_manifest["training_code"],
                "calibration_code": frozen_manifest["calibration_code"],
                "calibration": {"status": "retained", "temperature_bits": temperature_bits(temperatures[primitive])}}
        if accepted:
            selected = selection["selected"]
            checkpoint = Path(selected["checkpoint"])
            item.update({"source": {"directory": str(checkpoint), "sha256": artifact_sha(checkpoint)},
                         "initialization": selection["provenance"]["initialization"],
                         "training_profile": selection["provenance"], "selected_epoch": selected["epoch"],
                         "development_acceptance": selected["acceptance"],
                         "actual_fit_visitation": selected["actual_fit_visitation"],
                         "new_fit_visits": selected["actual_fit_visitation"]["visits"],
                         "calibration": {**calibration_visits[primitive], "status": "refit",
                                         "temperature_bits": temperature_bits(temperatures[primitive])},
                         "training_code": training_code_identity(), "calibration_code": training_code_identity()})
        else:
            item["previous_visitation"] = "inherited from frozen artifact; not reconstructed or claimed as new fitting"
        lineage[primitive] = item
    return {"artifact_kind": "calibrated-reference-head", "stage": "mixed", "precision": "float32",
            "calibration_role": "calibration", "feature_identity": features.identity,
            "feature_identity_sha256": file_sha256(features.directory / "identity.json"),
            "preparation": features.require_ready("calibrate"), "feature_origins": features.producer_origins(),
            **memberships, "membership_semantics": "union of independent per-head role coverage, not joint fitting",
            "training_procedure": PROCEDURE, "per_head_provenance": lineage,
            "lineage_snapshot": lineage_snapshot, "migration_binding": migration_binding,
            "parent_binding": parent_binding, "corpus_binding": corpus_binding,
            "final_dev_audit": final_dev_audit,
            "unchanged_membership_proof": proof["unchanged"],
            "frozen_reference": {**source, "manifest": frozen_manifest},
            "training_code": training_code_identity(), "calibration_code": training_code_identity(),
            "operator_sha256": file_sha256(__file__), "operator": Path(__file__).name,
            "external_selection": False,
            "execution": {"device": "cuda:0", "precision": "float32", "tf32": False,
                          "threads": 4, "torch": torch.__version__}}


def audit_calibrated(bundle, features, roles, changed, temperatures, controls, selections, device):
    heads = {}
    for primitive in sorted(changed):
        rows = [row for row in roles["development"] if row["primitive"] == primitive]
        evidence = dev_evidence(bundle, features, rows, temperatures, device)
        heads[primitive] = {"membership": membership_identity(rows),
                            "acceptance": acceptance(primitive, evidence, controls[primitive]),
                            "checkpoint_sha256": selections[primitive]["selected"]["checkpoint_sha256"]}
    return {"status": "passed" if all(item["acceptance"]["eligible"] for item in heads.values()) else "failed",
            "input_role": "development", "membership": membership_identity(roles["development"]),
            "selection_locked": True, "candidate_search": False, "heads": heads,
            "head_sha256": {primitive: head_sha(bundle, primitive) for primitive in PRIMITIVES},
            "temperature_bits": {primitive: temperature_bits(value) for primitive, value in temperatures.items()}}


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Preference head repair requires CUDA; no CPU fallback")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def run(feature_directory, previous_directory, frozen_directory):
    configure_cuda()
    output = project_path(OUTPUT)
    output.mkdir(exist_ok=False)
    stores = []
    try:
        source = {"directory": str(Path(frozen_directory).resolve()), "sha256": artifact_sha(frozen_directory)}
        for directory in (previous_directory, feature_directory):
            directory = Path(directory).expanduser().resolve()
            store = FeatureStore(directory, json.loads((directory / "identity.json").read_text()), read_only=True)
            stores.append(store)
        old, features = stores
        if old.identity.get("dataset_scope") != contract() or features.identity.get("dataset_scope") != contract(augmentation=True):
            raise ValueError("Need separate complete v2 and opt-in preference feature corpora")
        previous, current = read_roles(old), read_roles(features)
        prove_memberships(previous, current)
        migration_binding = bind_transition(old, features)
        parent_binding = check_extension(features, previous, current)
        frozen, manifest, temperatures = load_bundle(frozen_directory)
        bind_frozen(old, manifest, previous)
        lineage_snapshot = {"verification": "publication-and-corpus-aware-evaluation",
                            "frozen": snapshot(frozen_directory, frozen), "selected": {}}
        controls = {}
        frozen.to("cuda:0")
        for primitive in ("noul", "choice"):
            rows = [row for row in current["development"] if row["primitive"] == primitive]
            controls[primitive] = dev_evidence(frozen, features, rows, temperatures, "cuda:0", raw_control=primitive == "noul")
        frozen.cpu()
        selections = {primitive: stage_trial(features, current, frozen, temperatures, primitive,
                                             controls[primitive], output, source)
                      for primitive in ("noul", "choice")}
        write_json(output / "selection.json", selections)
        accepted = {primitive for primitive, selection in selections.items() if selection["status"] == "accepted"}
        failures = {primitive: selection["failure"] for primitive, selection in selections.items() if selection["status"] == "rejected"}
        report = {"training_procedure": PROCEDURE, "stage_failures": failures,
                  "accepted_heads": sorted(accepted), "retained_heads": sorted(set(PRIMITIVES) - accepted),
                  "frozen_reference": source, "selection_role": "development",
                  "qualification": "not qualified; no external evaluation or E2E performed"}
        if "noul" not in accepted:
            if artifact_sha(frozen_directory) != source["sha256"]:
                raise ValueError("Frozen reference changed during repair")
            report.update({"status": "rejected", "reference": str(frozen_directory),
                           "composition_failure": "Augmented Noul coverage cannot be claimed by the retained frozen Noul head"})
            write_json(output / "run.json", report)
            return report
        candidates = {}
        for primitive in sorted(accepted):
            selected = selections[primitive]["selected"]
            if artifact_sha(selected["checkpoint"]) != selected["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint files changed after DEV selection")
            candidate, checkpoint_manifest, checkpoint_temperatures = load_bundle(selected["checkpoint"])
            if (checkpoint_manifest["weights_sha256"] != selected["weights_sha256"]
                    or checkpoint_manifest["operator_sha256"] != file_sha256(__file__)
                    or checkpoint_manifest["frozen_reference"] != source
                    or checkpoint_manifest["fit_membership"] != membership_identity(
                        row for row in current["fit"] if row["primitive"] == primitive)
                    or checkpoint_manifest["development_membership"] != membership_identity(
                        row for row in current["development"] if row["primitive"] == primitive)):
                raise ValueError("Selected independent checkpoint provenance changed")
            assert_unchanged(frozen, candidate, {primitive}, checkpoint_temperatures, temperatures)
            candidate.to("cuda:0")
            evidence = dev_evidence(candidate, features,
                                    [row for row in current["development"] if row["primitive"] == primitive],
                                    checkpoint_temperatures, "cuda:0")
            if acceptance(primitive, evidence, controls[primitive]) != selected["acceptance"]:
                raise ValueError("Selected checkpoint no longer reproduces its DEV acceptance")
            candidates[primitive] = candidate.cpu()
            lineage_snapshot["selected"][primitive] = snapshot(selected["checkpoint"], candidates[primitive])
        bundle, memberships, per_head, proof = compose_heads(frozen, temperatures, candidates, selections, previous, current)
        bundle.to("cuda:0").eval()
        calibrated, calibration_visits = calibrate_changed(bundle, features, current, accepted, temperatures, "cuda:0")
        assert_unchanged(frozen, bundle, accepted, calibrated, temperatures)
        final_dev_audit = audit_calibrated(bundle, features, current, accepted, calibrated, controls, selections, "cuda:0")
        write_json(output / "final-dev-audit.json", final_dev_audit)
        if final_dev_audit["status"] != "passed":
            report.update({"status": "rejected", "reference": str(frozen_directory),
                           "composition_failure": "The locked calibrated composite failed final DEV retention; no reselection",
                           "final_dev_audit": final_dev_audit})
            write_json(output / "run.json", report)
            return report
        corpus_binding = {label: {"directory": str(store.directory.resolve()),
                                  "identity_sha256": file_sha256(store.directory / "identity.json")}
                          for label, store in (("previous", old), ("current", features))}
        metadata = composite_metadata(manifest, source, features, memberships, per_head, proof, selections,
                                      bundle, calibration_visits, temperatures=calibrated,
                                      lineage_snapshot=lineage_snapshot, migration_binding=migration_binding,
                                      parent_binding=parent_binding, corpus_binding=corpus_binding, final_dev_audit=final_dev_audit)
        validate_composite(bundle, metadata, calibrated, features=features, previous_features=old, corpus=True)
        reference = output / "reference"
        reference.mkdir(exist_ok=False)
        save_bundle(reference, bundle, metadata, calibrated)
        restored, final_manifest, final_temperatures = load_bundle(reference)
        assert_unchanged(frozen, restored, accepted, final_temperatures, temperatures)
        if any(final_manifest[field] != membership_identity(features.iter_records(role)) for role, field in FIELDS.items()):
            raise ValueError("Composite global membership does not satisfy the runtime/evaluator contract")
        if artifact_sha(frozen_directory) != source["sha256"]:
            raise ValueError("Frozen reference changed during repair")
        report.update({"status": "calibrated_with_stage_rejection" if failures else "dev_accepted_and_calibrated",
                       "reference": str(reference), "reference_sha256": artifact_sha(reference)})
        write_json(output / "run.json", report)
        return report
    except Exception as error:
        write_json(output / "run.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                         "reference": str(frozen_directory), "training_procedure": PROCEDURE})
        raise
    finally:
        for store in stores:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=os.environ.get("FUZEV_FEATURE_DIRECTORY"),
                        help="Complete, independently migrated/extracted preference feature directory")
    parser.add_argument("--previous-features", default=str(project_path("data/features/native-4.05-scope-v2-image-dimensions")))
    parser.add_argument("--frozen", default=str(project_path("weights/cuda-refine-20261005/reference")))
    args = parser.parse_args()
    if not args.features:
        parser.error("--features or FUZEV_FEATURE_DIRECTORY is required; no active-corpus default")
    print(json.dumps(run(args.features, args.previous_features, args.frozen), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
