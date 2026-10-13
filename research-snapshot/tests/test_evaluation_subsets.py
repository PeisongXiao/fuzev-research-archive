from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from test_evaluate import fixture
from fuzev_research.evaluate import (SUBSET_FILE, TEST_SCOPE, canonical_test_membership,
                            evaluate, require_comparison_complete, require_complete,
                            sha256, test_rows as reference_rows)
from fuzev_research.features import FeatureStore, file_sha256


def make_subset(fixture, mutation=None):
    rows = [json.loads(row["metadata"]) for row in reference_rows(fixture.reader)]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        extra = deepcopy(rows[0])
        extra["id"] = "extra-test-id"
        rows.append(extra)
    elif mutation == "target":
        rows[0]["target"] = [0., 1.]
    elif mutation == "candidate":
        rows[0]["candidates"].reverse()
    elif mutation == "group":
        rows[0]["group"] = "wrong-group"
    elif mutation == "non_test":
        extra = deepcopy(rows[0])
        extra.update(id="extra-calibration", group="extra-calibration", role="calibration")
        rows.append(extra)
    directory = fixture.root / "test-only"
    identity = {**fixture.reader.identity, "model": "synthetic-comparison", "corpus_scope": TEST_SCOPE}
    writer = FeatureStore(directory, identity, shard_size=3)
    # Reverse physical order: canonical identity must not depend on shard layout.
    for row in reversed(rows):
        writer.add(row, torch.ones(2560), torch.ones(2, 2560), torch.tensor([0., 1.]))
    if mutation == "failure":
        writer.fail("failed-test", "synthetic failed readout")
    elif mutation == "fake_preparation":
        path = fixture.root / "data/prepared/text.jsonl"
        writer.register_preparation(path, file_sha256(path))
        writer.finish_preparation(path, 16)
    writer.close()
    reader = FeatureStore(directory, identity, read_only=True)
    count = len(rows)
    ownership = []
    for shard, records in reader.db.execute("SELECT shard,COUNT(*) FROM records GROUP BY shard ORDER BY shard"):
        ownership.append({"shard": shard, "records": records,
                          "tensors_sha256": file_sha256(directory / f"shard-{shard:06d}.safetensors"),
                          "metadata_sha256": file_sha256(directory / f"shard-{shard:06d}.json")})
    manifest = {"schema_version": 1, "artifact_kind": TEST_SCOPE, "roles": ["test"],
                "reference": {"feature_identity_sha256": fixture.plan["reference"]["feature_identity_sha256"],
                              "preparation_sha256": sha256(require_complete(fixture.reader)),
                              "test_membership": canonical_test_membership(fixture.reader)},
                "extraction": {"completed": True, "examined": count, "committed": count, "failures": 0},
                "ownership": {"feature_identity_sha256": file_sha256(directory / "identity.json"),
                              "index_sha256": file_sha256(directory / "index.sqlite3"), "shards": ownership}}
    return SimpleNamespace(reader=reader, directory=directory, manifest=manifest)


def seal(subset):
    path = subset.directory / SUBSET_FILE
    path.write_text(json.dumps(subset.manifest))
    return file_sha256(path)


def test_complete_test_only_comparison_matches_full_results_without_training_readiness(fixture, monkeypatch):
    subset = make_subset(fixture)
    original = FeatureStore.require_ready

    def no_training_readiness(self, stage):
        if self.directory == subset.directory:
            pytest.fail("A test-only comparison must not require fitting/development/calibration features")
        return original(self, stage)

    pin = seal(subset)
    monkeypatch.setattr(FeatureStore, "require_ready", no_training_readiness)
    try:
        assert len(list(subset.reader.iter_records())) == 8  # full fixture has 32 all-role rows
        assert subset.reader.preparations() == []
        assert require_comparison_complete(fixture.reader, subset.reader, pin)["scope"] == TEST_SCOPE
        with pytest.raises(ValueError, match="full corpus"):
            require_complete(subset.reader)
        fixture.plan["cells"].append({"id": "native-3.05", "status": "evaluate", "features": str(subset.directory),
                                      "feature_identity_sha256": file_sha256(subset.directory / "identity.json"),
                                      "subset_manifest_sha256": pin})
        report = evaluate(fixture.plan, batch_size=3, draws=10)
        full, test_only = (report["cells"][name] for name in ("native-4.05", "native-3.05"))
        assert test_only["status"] == "evaluated", test_only
        assert test_only["corpus_scope"] == TEST_SCOPE
        assert test_only["test_membership"] == full["test_membership"]
        assert test_only["tracks"] == full["tracks"]
        assert test_only["macro_normalized_nll"] == full["macro_normalized_nll"]
        assert test_only["preparation"]["manifest"]["reference"]["test_membership"]["records"] == 8
    finally:
        subset.reader.close()


def test_test_only_is_independently_not_train_ready(fixture):
    subset = make_subset(fixture)
    try:
        for stage in ("text", "mixed", "calibrate"):
            with pytest.raises(ValueError, match="not completed"):
                subset.reader.require_ready(stage)
        with pytest.raises(ValueError, match="full corpus"):
            require_complete(subset.reader)
    finally:
        subset.reader.close()


@pytest.mark.parametrize("mutation,match", [
    ("missing", "missing or extra"), ("extra", "missing or extra"),
    ("target", "alignment"), ("candidate", "alignment"), ("group", "alignment"),
    ("non_test", "exclusively test"), ("failure", "Extraction failures"),
    ("fake_preparation", "fake full preparation"),
])
def test_subset_claims_cannot_replace_canonical_membership_or_ownership(fixture, mutation, match):
    subset = make_subset(fixture, mutation)
    try:
        with pytest.raises(ValueError, match=match):
            require_comparison_complete(fixture.reader, subset.reader, seal(subset))
    finally:
        subset.reader.close()


@pytest.mark.parametrize("mutation,match", [
    ("not_completed", "incomplete"), ("not_examined", "incomplete"),
    ("false_reference", "canonical reference"), ("false_membership", "canonical reference"),
    ("wrong_index", "index ownership"), ("wrong_shard", "shard checksum"),
    ("missing_shard_owner", "every committed"), ("adaptation_roles", "test-only extraction manifest"),
])
def test_explicit_subset_manifest_must_pin_complete_extraction_and_actual_owned_features(fixture, mutation, match):
    subset = make_subset(fixture)
    manifest = subset.manifest
    if mutation == "not_completed":
        manifest["extraction"]["completed"] = False
    elif mutation == "not_examined":
        manifest["extraction"]["examined"] -= 1
    elif mutation == "false_reference":
        manifest["reference"]["feature_identity_sha256"] = "0" * 64
    elif mutation == "false_membership":
        manifest["reference"]["test_membership"]["ordered_sha256"] = "0" * 64
    elif mutation == "wrong_index":
        manifest["ownership"]["index_sha256"] = "0" * 64
    elif mutation == "wrong_shard":
        manifest["ownership"]["shards"][0]["tensors_sha256"] = "0" * 64
    elif mutation == "missing_shard_owner":
        manifest["ownership"]["shards"].pop()
    else:
        manifest["roles"] = ["calibration", "test"]
    try:
        with pytest.raises(ValueError, match=match):
            require_comparison_complete(fixture.reader, subset.reader, seal(subset))
    finally:
        subset.reader.close()


def test_subset_requires_separately_pinned_manifest_and_actual_shard_files(fixture):
    subset = make_subset(fixture)
    pin = seal(subset)
    try:
        with pytest.raises(ValueError, match="missing pin"):
            require_comparison_complete(fixture.reader, subset.reader)
        with pytest.raises(ValueError, match="checksum"):
            require_comparison_complete(fixture.reader, subset.reader, "0" * 64)
        path = subset.directory / "shard-000000.safetensors"
        path.unlink()
        with pytest.raises(ValueError, match="Missing committed"):
            require_comparison_complete(fixture.reader, subset.reader, pin)
    finally:
        subset.reader.close()


def test_subset_sidecar_ownership_must_match_exact_indexed_rows(fixture):
    subset = make_subset(fixture)
    path = subset.directory / "shard-000000.json"
    rows = json.loads(path.read_text())
    rows[0]["id"] = "wrong-owned-id"
    path.write_text(json.dumps(rows))
    subset.manifest["ownership"]["shards"][0]["metadata_sha256"] = file_sha256(path)
    try:
        with pytest.raises(ValueError, match="row ownership"):
            require_comparison_complete(fixture.reader, subset.reader, seal(subset))
    finally:
        subset.reader.close()
