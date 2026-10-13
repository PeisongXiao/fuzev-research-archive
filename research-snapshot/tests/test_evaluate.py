from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from fuzev_research.evaluate import (HEAD_FILES, REQUIRED_CELLS, ShardAudit, aligned_test_rows,
                            checked_batch, evaluate, evaluate_cell, load_reference,
                            markdown_report, open_features, require_complete, validate_plan)
from fuzev_research.features import FeatureStore, file_sha256
from fuzev_research.head import HeadBundle, save_bundle
from fuzev_research.train import membership_identity, training_code_identity


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Every input, annotation and feature is synthetic and lives under pytest's
    # worktree-local temp directory. Never open the project's actual data root.
    monkeypatch.setattr("fuzev_research.features.project_path", lambda name: tmp_path / name)
    identity = {"engine": "synthetic-cpu", "model": "synthetic", "revision": "fixture",
                "encoding": {"fixture": True}, "datasets": {name: "fixture" for name in ("text", "qasper", "images")},
                "runtime_sources": {"fixture": True}, "readout_runtime": {"fixture": True},
                "cache_precision": "float16", "feature_tap": "final-mixer/post-collapse/before-lm-head",
                "hidden_size": 2560}
    directory = tmp_path / "features"
    writer = FeatureStore(directory, identity, shard_size=3)
    for source in ("text", "qasper", "images"):
        rows, counts = [], {}
        for role in ("fit", "development", "calibration", "test"):
            for primitive in (("choice", "ordinal") if source == "text" else ("noul",) if source == "qasper" else ("choice",)):
                for group in range(2):
                    ids = ["yes", "no"] if primitive == "noul" else ["a", "b"]
                    candidates = [{"id": key, "text": key, **({"ordinal_value": i} if primitive == "ordinal" else {})}
                                  for i, key in enumerate(ids)]
                    row = {"id": f"{source}-{role}-{primitive}-{group}", "group": f"{source}-{role}-{group}",
                           "role": role, "primitive": primitive, "modality": "image" if source == "images" else "text",
                           "source": source, "revision": "fixture", "candidates": candidates, "target": [1., 0.],
                           "adaptation": "evidence" if primitive == "noul" else None}
                    rows.append(row)
                    key = f"{source}/{primitive}/{role}"
                    counts[key] = counts.get(key, 0) + 1
        path = tmp_path / f"data/prepared/{source}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = file_sha256(path)
        manifest = tmp_path / f"data/manifests/{'image' if source == 'images' else source}-splits.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"path": str(path), "sha256": digest, "records": len(rows), "counts": counts}))
        writer.register_preparation(path, digest)
        for row in rows:
            writer.add(row, torch.ones(2560), torch.ones(2, 2560), torch.tensor([0., 1.]))
        writer.finish_preparation(path, len(rows))
    writer.close()
    identity_hash = file_sha256(directory / "identity.json")
    reader = open_features(directory, identity_hash)
    head = tmp_path / "head"
    bundle = HeadBundle().eval()
    with torch.no_grad():
        for parameter in bundle.parameters():
            parameter.zero_()
    provenance = {"artifact_kind": "calibrated-reference-head", "stage": "mixed", "calibration_role": "calibration",
                  "feature_identity": identity, "preparation": require_complete(reader),
                  "training_code": training_code_identity(), "calibration_code": training_code_identity(),
                  **{f"{role}_membership": membership_identity(reader.iter_records(role))
                     for role in ("fit", "development", "calibration")}}
    save_bundle(head, bundle, provenance, {name: 2. for name in bundle.heads})
    hashes = {name: file_sha256(head / name) for name in HEAD_FILES}
    plan = {"schema_version": 1, "reference": {"features": str(directory), "feature_identity_sha256": identity_hash,
                                                "head": str(head), "sha256": hashes},
            "cells": [{"id": "native-4.05", "status": "evaluate", "features": str(directory),
                       "feature_identity_sha256": identity_hash}]}
    yield SimpleNamespace(root=tmp_path, directory=directory, head=head, hashes=hashes, reader=reader, plan=plan)
    reader.close()


def test_synthetic_end_to_end_only_scores_test_roles_and_retains_all_cells(fixture, monkeypatch):
    original = FeatureStore.batch_tensors

    def checked(self, rows):
        assert all(row["role"] == "test" for row in rows)
        yield from original(self, rows)

    monkeypatch.setattr(FeatureStore, "batch_tensors", checked)
    fixture.plan["cells"].extend([
        {"id": "native-5.05", "status": "memory_infeasible", "reason": "Synthetic declared fixture", "evidence": "fixture-log"},
        {"id": "vllm-nvidia", "status": "unsupported", "reason": "Synthetic missing loader", "evidence": "fixture-log"},
        {"id": "optional-fixture", "status": "failed", "reason": "Synthetic failure", "evidence": "fixture-log"},
    ])
    report = evaluate(fixture.plan, batch_size=3, draws=20)
    cell = report["cells"]["native-4.05"]
    assert cell["status"] == "evaluated", cell
    assert cell["test_membership"]["records"] == 8
    assert cell["feature_files_sha256"]
    assert cell["gates"]["head_calibrated"]["checks"]
    assert set(REQUIRED_CELLS) <= report["cells"].keys()
    assert report["cells"]["native-3.05"]["status"] == "not_run"
    assert report["cells"]["native-5.05"]["status"] == "memory_infeasible"
    assert report["cells"]["vllm-nvidia"]["status"] == "unsupported"
    assert report["cells"]["optional-fixture"]["status"] == "failed"
    assert not report["coverage_complete"] and not report["all_required_quality_gates_pass"]
    assert json.loads(json.dumps(report, allow_nan=False))["reference"]["sha256"] == fixture.hashes
    markdown = markdown_report(report)
    assert "memory_infeasible" in markdown and "evidence_f1" in markdown and "0.02" in markdown
    assert fixture.hashes == {name: file_sha256(fixture.head / name) for name in HEAD_FILES}


def test_membership_and_results_are_independent_of_batch_partition(fixture):
    bundle, _, temperatures = load_reference(fixture.reader, fixture.head, fixture.hashes)
    small = evaluate_cell(fixture.reader, fixture.reader, bundle, temperatures, batch_size=1, draws=10)
    large = evaluate_cell(fixture.reader, fixture.reader, bundle, temperatures, batch_size=8, draws=10)
    assert small["test_membership"] == large["test_membership"]
    assert small["tracks"] == large["tracks"]
    assert small["macro_normalized_nll"] == large["macro_normalized_nll"]


def test_optional_progress_does_not_change_quality_or_membership(fixture):
    observed = []
    plain = evaluate(fixture.plan, batch_size=3, draws=10)
    reported = evaluate(fixture.plan, batch_size=3, draws=10,
                        progress=lambda cell, processed, passed, total: observed.append((cell, processed, passed, total)))
    assert reported == plain
    assert observed[0] == ("native-4.05", 0, 0, 8)
    assert observed[-1] == ("native-4.05", 8, 8, 8)
    assert all(0 <= passed <= processed <= total for _, processed, passed, total in observed)
    assert [value[1] for value in observed] == sorted(value[1] for value in observed)


@pytest.mark.parametrize("sql,match", [
    ("UPDATE preparations SET completed=0", "not completed"),
    ("INSERT INTO failures VALUES('fixture-failure','unfinished')", "Extraction failures"),
    ("UPDATE groups_ SET role='fit' WHERE role='test'", "ownership"),
])
def test_incomplete_and_leaky_corpus_rejected_before_loading_head(fixture, sql, match):
    import sqlite3
    with sqlite3.connect(fixture.directory / "index.sqlite3") as db:
        db.execute(sql)
    with pytest.raises(ValueError, match=match):
        load_reference(fixture.reader, fixture.head, fixture.hashes)


@pytest.mark.parametrize("name", HEAD_FILES)
def test_all_frozen_artifacts_are_checksum_pinned(fixture, name):
    expected = dict(fixture.hashes, **{name: "0" * 64})
    with pytest.raises(ValueError, match="checksum"):
        load_reference(fixture.reader, fixture.head, expected)


@pytest.mark.parametrize("key,value,match", [
    ("artifact_kind", "training-checkpoint", "calibrated reference"),
    ("fit_membership", {}, "fit membership"),
    ("calibration_membership", {}, "calibration membership"),
    ("calibration_code", {}, "code provenance"),
])
def test_matching_weights_alone_do_not_establish_frozen_provenance(fixture, key, value, match):
    path = fixture.head / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    expected = {name: file_sha256(fixture.head / name) for name in HEAD_FILES}
    with pytest.raises(ValueError, match=match):
        load_reference(fixture.reader, fixture.head, expected)


@pytest.mark.parametrize("mutation", ["order", "text", "group", "target", "missing", "extra", "baseline"])
def test_exact_id_group_candidate_alignment_rejects_semantic_changes(fixture, monkeypatch, mutation):
    from fuzev_research.evaluate import test_rows
    original = list(test_rows(fixture.reader))
    changed = deepcopy(original)
    meta = json.loads(changed[0]["metadata"])
    if mutation == "order":
        meta["candidates"].reverse()
    elif mutation == "text":
        meta["candidates"][0]["text"] = "changed meaning"
    elif mutation == "group":
        meta["group"] = changed[0]["group_"] = "changed-group"
    elif mutation == "target":
        meta["target"] = [0., 1.]
    elif mutation == "baseline":
        meta["has_baseline"] = False
    changed[0]["metadata"] = json.dumps(meta)
    if mutation == "missing":
        changed.pop()
    elif mutation == "extra":
        changed.append(changed[-1])
    candidate = object()
    monkeypatch.setattr("fuzev_research.evaluate.test_rows", lambda features: iter(changed if features is candidate else original))
    with pytest.raises(ValueError, match="alignment|membership|baseline"):
        list(aligned_test_rows(fixture.reader, candidate))


def test_sidecar_and_tensor_candidate_order_are_verified(fixture, monkeypatch):
    import fuzev_research.evaluate as module
    row, meta = next(aligned_test_rows(fixture.reader, fixture.reader))
    audit = ShardAudit(fixture.reader)
    original = module.collate

    def damaged(*args):
        batch = original(*args)
        batch["targets"] = batch["targets"].flip(1)
        return batch

    monkeypatch.setattr(module, "collate", damaged)
    with pytest.raises(ValueError, match="tensor alignment"):
        checked_batch(fixture.reader, [row], [meta], audit)
    monkeypatch.setattr(module, "collate", original)
    path = fixture.directory / f"shard-{row['shard']:06d}.json"
    sidecar = json.loads(path.read_text())
    sidecar[row["row_"]]["id"] = "wrong-id"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="Shard row"):
        checked_batch(fixture.reader, [row], [meta], audit)
    with pytest.raises(ValueError, match="changed"):
        audit.finish()


def test_no_per_cell_adaptation_or_silent_bounds_escape(fixture, monkeypatch):
    plan = deepcopy(fixture.plan)
    plan["cells"][0]["temperatures"] = {"choice": 42.}
    with pytest.raises(ValueError, match="adaptation"):
        validate_plan(plan)
    with pytest.raises(ValueError, match="resource"):
        evaluate(fixture.plan, batch_size=257)
    monkeypatch.setattr("fuzev_research.evaluate.MAX_RECORDS", 1)
    with pytest.raises(MemoryError, match="no subsampling"):
        require_complete(fixture.reader)


def test_bad_cell_does_not_hide_other_cells_or_claim_success(fixture):
    fixture.plan["cells"].insert(0, {"id": "native-3.05", "status": "evaluate", "features": str(fixture.directory),
                                    "feature_identity_sha256": "bad"})
    report = evaluate(fixture.plan, draws=5)
    assert report["cells"]["native-3.05"]["status"] == "failed"
    assert report["cells"]["native-4.05"]["status"] == "evaluated"
    assert not report["all_required_quality_gates_pass"]
