import hashlib
import json

import pytest

from fuzev_research.features import FeatureStore
from fuzev_research.scope import PREPARATIONS, contract, required_preparations, validate_row


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setattr("fuzev_research.features.project_path", lambda path: tmp_path / path)
    identity = {"dataset_scope": contract(), "datasets": {name: "fixture" for name in
                ("helpsteer2", "helpsteer3", "boolq", "mm-rlhf", "qasper", "vl-rewardbench")}}
    store = FeatureStore(tmp_path / "features", identity)
    for filename in required_preparations("mixed"):
        manifest_name, modality, _, sources = PREPARATIONS[filename]
        rows = []
        for source in sources:
            for role in ("fit", "development", "calibration", "test"):
                row = {"id": source + "-" + role, "group": source + "-" + role, "role": role,
                       "source": source, "revision": "fixture", "primitive": "noul" if source == "boolq" else "choice",
                       "modality": modality, "target": [1., 0.],
                       "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
                       "original_split": "validation" if role == "test" else "train"}
                if source == "boolq":
                    row["adaptation"] = "boolean_truth"
                rows.append(row)
        path = tmp_path / "data/prepared" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        counts = {f"{row['source']}/{row['primitive']}/{row['role']}": 1 for row in rows}
        manifest = tmp_path / "data/manifests" / manifest_name
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"path": str(path), "sha256": digest, "records": len(rows), "counts": counts,
                                        "dataset_scope_revision": contract()["revision"], "evaluation_scope": "training"}))
        store.register_preparation(path, digest)
        for index, row in enumerate(rows):
            store.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)", (
                row["id"], 0, index, row["group"], row["role"], row["primitive"], modality, row["source"], json.dumps(row)))
        store.db.commit()
        store.finish_preparation(path, len(rows))
    yield store, tmp_path
    store.close()


def test_active_training_does_not_require_external_preparations(scoped):
    store, _ = scoped
    assert store.require_ready("text")["records"] == 12
    assert store.require_ready("mixed")["records"] == 16
    assert store.require_ready("calibrate") == store.require_ready("mixed")
    with pytest.raises(ValueError, match="Missing required"):
        store.require_external_ready()


def test_boolq_is_required_and_qasper_is_not_a_training_substitute(scoped):
    store, _ = scoped
    store.db.execute("UPDATE preparations SET completed=0 WHERE path LIKE '%boolq.jsonl'")
    store.db.commit()
    with pytest.raises(ValueError, match="boolq.jsonl"):
        store.require_ready("text")
    with pytest.raises(ValueError, match="outside"):
        store.register_preparation("qasper.jsonl", "0" * 64)


@pytest.mark.parametrize("source,role", [("aokvqa", "fit"), ("qasper", "development"),
                                        ("vl-rewardbench", "calibration")])
def test_inactive_or_external_role_leakage_is_rejected(scoped, source, role):
    store, _ = scoped
    store.db.execute("INSERT INTO records VALUES('injected',1,0,'injected',?,'choice','text',?,'{}')", (role, source))
    store.db.commit()
    with pytest.raises(ValueError, match="Inactive dataset|external-role leakage"):
        store.require_ready("text")


def test_external_test_rows_do_not_change_training_readiness(scoped):
    store, _ = scoped
    before = store.require_ready("text")
    store.db.execute("INSERT INTO records VALUES('qasper-test',1,0,'paper:test','test','noul','text','qasper','{}')")
    store.db.commit()
    assert store.require_ready("text") == before


def test_original_boolq_semantics_are_verified_against_committed_metadata(scoped):
    store, _ = scoped
    row = store.db.execute("SELECT metadata FROM records WHERE id='boolq-fit'").fetchone()
    meta = json.loads(row[0])
    meta["adaptation"] = "answerability"
    store.db.execute("UPDATE records SET metadata=? WHERE id='boolq-fit'", (json.dumps(meta),))
    store.db.commit()
    with pytest.raises(ValueError, match="not accounted"):
        store.require_ready("text")


@pytest.mark.parametrize("update", [{"role": "fit"}, {"original_split": "train"},
                                    {"sampling_weight": 0}, {"inclusion_probability": 0},
                                    {"sampling_weight": 3}, {"source_test_example_id": None}])
def test_external_row_contract_fails_closed(update):
    row = {"source": "qasper", "role": "test", "original_split": "test", "evaluation_scope": "external",
           "primitive": "noul", "adaptation": "evidence", "inclusion_probability": .5,
           "sampling_weight": 2., "source_test_example_id": "question:paragraph"}
    row.update(update)
    with pytest.raises(ValueError):
        validate_row(row)


def test_external_rejection_precedes_any_tensor_processing(scoped):
    store, _ = scoped
    with pytest.raises(ValueError, match="outside"):
        store.add({"source": "aokvqa"}, None, None)
