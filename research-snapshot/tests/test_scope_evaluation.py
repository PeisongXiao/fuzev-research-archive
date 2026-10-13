"""Synthetic CPU-only scope/design fixtures; never open project data or a model."""

from copy import deepcopy
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from fuzev_research import evaluate as evaluation
from fuzev_research import comparison_inputs, comparison_seal
from fuzev_research.evaluation_metrics import METHODS, compare_tracks
from fuzev_research.external_sampling import sample_evidence
from fuzev_research.metrics import per_record, probabilities
from fuzev_research.scope import contract


def qasper(subtype, question, paragraph=None):
    qid, paper = f"q-{question}", str(question % 8)
    result = {"id": f"qasper-{subtype}-{question}-{paragraph}", "source": "qasper", "revision": "fixture",
              "group": "paper:" + paper, "source_paper_id": paper, "role": "test", "original_split": "test",
              "modality": "text", "primitive": "noul", "adaptation": subtype, "evaluation_scope": "external",
              "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}], "target": [1., 0.],
              "source_question_id": qid, "source_question_ids": [qid], "source_test_example_id": qid,
              "sampling_weight": 1., "inclusion_probability": 1., "sampling_design": "census",
              "sampling_stratum": subtype, "sampling_seed": 20261002, "has_baseline": True,
              "dataset_scope_revision": "2026-10-02-v2", "dataset_purpose": "external"}
    if paragraph is not None:
        result.update(evidence_id=f"paragraph-{paragraph}", target=[float(paragraph % 2), float(1 - paragraph % 2)])
    return result


def vl_pair(index=0, *, image_index=None):
    image_index = index if image_index is None else image_index
    responses = [f"response zero {index}", f"response one {index}"]
    pair_digest = hashlib.sha256(("Synthetic query?\0" + json.dumps(responses, ensure_ascii=True)).encode()).hexdigest()
    raw_id = f"vl-{image_index}"
    example = hashlib.sha256((raw_id + "\0" + pair_digest).encode()).hexdigest()
    choice = {"id": f"vl-choice-{index}", "source": "vl-rewardbench", "revision": "fixture", "group": f"image:{image_index}",
              "role": "test", "original_split": "test", "modality": "image", "primitive": "choice",
              "adaptation": "official_heldout_preference", "evaluation_scope": "external",
              "source_test_example_id": example, "source_example_id": raw_id, "image_sha256": f"{image_index:064x}",
              "image_path": f"/synthetic/images/{image_index}.png", "source_image_id": raw_id,
              "candidates": [{"id": str(i), "text": text} for i, text in enumerate(responses)],
              "target": [1., 0.] if index % 2 == 0 else [0., 1.],
              "sampling_weight": 1., "inclusion_probability": 1., "sampling_design": "census",
              "sampling_stratum": "official_test", "label_semantics": "relative_pairwise_preference",
              "dataset_scope_revision": "2026-10-02-v2", "dataset_purpose": "external",
              "source_response_pair_sha256": pair_digest,
              "question": "Which response best answers the question about the image?\nSynthetic query?", "has_baseline": True}
    noul = {**choice, "id": f"vl-noul-{index}", "primitive": "noul", "adaptation": "pairwise_preference_yes_no",
            "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
            "question": f"Question about the image: Synthetic query?\nResponse A: {responses[0]}\n"
                        f"Response B: {responses[1]}\nIs response A preferred to response B for this question about the image?"}
    return choice, noul


def indexed(meta):
    return {**{key: meta[key] for key in ("id", "role", "primitive", "modality", "source")},
            "group_": meta["group"], "metadata": json.dumps(meta), "shard": 0, "row_": 0}


@pytest.fixture(scope="module")
def population():
    # Fractional expected stratum quotas make realized n/N demonstrably wrong.
    sample, design = sample_evidence(qasper("evidence", question, paragraph)
                                    for question in range(32) for paragraph in range(129))
    census = [qasper(subtype, question) for subtype, size in (("yes_no", 239), ("answerability", 1450))
              for question in range(size)]
    internal = {**qasper("boolean_truth", 5000), "id": "internal-boolq-test", "source": "boolq",
                "group": "boolq:heldout", "evaluation_scope": "training", "dataset_purpose": "training",
                "original_split": "validation"}
    return [*census, *sample, *vl_pair(), *vl_pair(1), internal], design


@pytest.fixture
def scoped_store(population, tmp_path, monkeypatch):
    rows, design = deepcopy(population)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE records(id TEXT PRIMARY KEY,shard INTEGER,row_ INTEGER,group_ TEXT,role TEXT,"
               "primitive TEXT,modality TEXT,source TEXT,metadata TEXT)")
    for meta in rows:
        row = indexed(meta)
        db.execute("INSERT INTO records VALUES(:id,:shard,:row_,:group_,:role,:primitive,:modality,:source,:metadata)", row)
    path = tmp_path / "data/manifests/qasper-external-splits.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"sampling": design}))
    monkeypatch.setattr(evaluation, "project_path", lambda name: tmp_path / name)
    preparations = [{"path": str(tmp_path / "data/prepared" / name), "records": count,
                     "sha256": "a" * 64, "manifest_sha256": evaluation.file_sha256(path)}
                    for name, count in (("qasper-external.jsonl", 3689), ("vl-rewardbench-choice.jsonl", 2),
                                        ("vl-rewardbench-noul.jsonl", 2))]
    readiness = {"preparations": preparations, "records": 3693,
                 "counts": {"text/qasper/noul/test": 3689, "image/vl-rewardbench/choice/test": 2,
                            "image/vl-rewardbench/noul/test": 2}}
    store = SimpleNamespace(db=db, identity={"dataset_scope": contract()}, read_only=True,
                            require_external_ready=lambda: deepcopy(readiness), readiness=readiness,
                            rows=rows, manifest=path)
    yield store
    db.close()


def replace(store, meta):
    row = indexed(meta)
    store.db.execute("UPDATE records SET group_=:group_,role=:role,primitive=:primitive,modality=:modality,"
                     "source=:source,metadata=:metadata WHERE id=:id", row)


def variants(base, head, targets):
    return {method: per_record(base if method == "label_token" else head, targets,
                               temperature=2. if method == "head_calibrated" else 1.) for method in METHODS}


def test_weighted_metrics_counts_ece_and_group_bootstrap_use_the_same_design():
    base, head = np.array([[0., 2.], [2., 0.], [1., 0.], [0., 1.]]), np.array([[2., 0.]] * 4)
    target = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    values = variants(base, head, target)
    weights = np.array([1., 9., 3., 2.])
    groups = np.array(["paper:a", "paper:a", "paper:b", "paper:b"])
    evidence = {method: (probabilities(base if method == "label_token" else head,
                                       2. if method == "head_calibrated" else 1.)[:, 0], target[:, 0])
                for method in METHODS}
    report = compare_tracks(["evidence"] * 4, groups, values, evidence, weights=weights,
                            inclusion_probabilities=1 / weights, sampling_strata=["a", "a", "b", "b"], draws=40)
    track = report["tracks"]["evidence"]
    for method in METHODS:
        assert track["methods"][method]["accuracy"] == pytest.approx(np.average(values[method]["accuracy"], weights=weights))
        assert track["methods"][method]["normalized_nll"] == pytest.approx(
            np.average(values[method]["normalized_nll"], weights=weights))
    assert track["methods"]["head_raw"]["accuracy"] != np.mean(values["head_raw"]["accuracy"])
    assert track["methods"]["head_raw"]["evidence_f1"] == pytest.approx(8 / 19)
    assert track["methods"]["head_raw"]["evidence_tp"] == 2
    assert track["methods"]["head_raw"]["evidence_tp_weight"] == 4
    assert track["eligible_records"]["accuracy"] == 4 and track["eligible_weights"]["accuracy"] == 15
    assert track["sampling"]["effective_records"] == pytest.approx(225 / 95)
    assert track["methods"]["head_raw"]["ece"] == pytest.approx(abs(probabilities(head)[0, 0] - 4 / 15))
    rng, draws = np.random.default_rng(17), []
    difference = values["head_raw"]["normalized_nll"] - values["label_token"]["normalized_nll"]
    for _ in range(40):
        selected = np.concatenate([np.flatnonzero(groups == ("paper:a", "paper:b")[i])
                                   for i in rng.integers(0, 2, 2)])
        draws.append(np.average(difference[selected], weights=weights[selected]))
    interval = track["paired"]["head_raw"]["normalized_nll"]
    assert [interval["low"], interval["high"]] == pytest.approx(np.quantile(draws, [.025, .975]))
    assert "accuracy" not in report and "accuracy" not in report["macro_normalized_nll"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("weights,probabilities_", [([0, 1], None), ([True, True], None),
                                                    ([float("nan"), 1], None), ([float("inf"), 1], None),
                                                    ([1, -1], None), ([1, 2], [1, 1]), ([1, 2], [1, 0]),
                                                    ([1, 1], [1, 1.1]), ([1], None)])
def test_invalid_metric_weights_fail_closed(weights, probabilities_):
    values = variants([[0, 1]] * 2, [[1, 0]] * 2, [[1, 0]] * 2)
    with pytest.raises(ValueError, match="weights|Weights"):
        compare_tracks(["track"] * 2, ["a", "b"], values, weights=weights, inclusion_probabilities=probabilities_, draws=2)


def test_exact_external_censuses_and_all_internal_holdouts_are_canonical(scoped_store):
    membership = evaluation.canonical_test_membership(scoped_store)
    assert membership["records"] == 3694
    assert membership["external"]["qasper_subtypes"] == {"yes_no": 239, "answerability": 1450, "evidence": 2000}
    assert membership["external"]["vl_paired_examples"] == 2
    readiness = evaluation.require_external_complete(scoped_store)
    assert readiness["sampling_design"]["population"] == 4128
    assert readiness["sampling_design"]["sample_size"] == 2000
    assert readiness["sampling_design"]["frame_sha256"]


@pytest.mark.parametrize("where", ["source='qasper' AND json_extract(metadata,'$.adaptation')='yes_no'",
                                  "source='qasper' AND json_extract(metadata,'$.adaptation')='answerability'",
                                  "source='qasper' AND json_extract(metadata,'$.adaptation')='evidence'",
                                  "id='vl-choice-0'", "id='vl-noul-0'", "source='vl-rewardbench'"])
def test_missing_membership_cannot_qualify(scoped_store, where):
    scoped_store.db.execute("DELETE FROM records WHERE id=(SELECT id FROM records WHERE " + where + " LIMIT 1)")
    with pytest.raises(ValueError, match="membership|same source test"):
        evaluation.canonical_test_membership(scoped_store)


def test_missing_both_members_of_a_pair_still_fails_pinned_census(scoped_store):
    scoped_store.db.execute("DELETE FROM records WHERE id IN ('vl-choice-0','vl-noul-0')")
    with pytest.raises(ValueError, match="exact external test membership"):
        evaluation.require_external_complete(scoped_store)


def test_qasper_subtype_lookup_accepts_different_canonical_aliases_in_the_same_paper(scoped_store):
    census = deepcopy(next(row for row in scoped_store.rows if row["id"] == "qasper-answerability-5-None"))
    census["source_question_ids"].append("q-alias")
    replace(scoped_store, census)
    for subtype in ("yes_no", "evidence"):
        row = deepcopy(next(row for row in scoped_store.rows if row["source"] == "qasper" and
                            row["adaptation"] == subtype and row["source_question_id"] == "q-5"))
        row.update(source_question_id="q-alias", source_question_ids=["q-alias"], source_test_example_id="q-alias")
        if subtype == "evidence":
            row["sampling_stratum"] = json.dumps([row["group"], "q-alias", row["sampling_label"]], separators=(",", ":"))
        replace(scoped_store, row)
    assert evaluation.external_test_membership(scoped_store)["qasper_subtypes"]["answerability"] == 1450


def test_qasper_subtype_lookup_does_not_accept_an_alias_from_another_paper(scoped_store):
    row = deepcopy(next(row for row in scoped_store.rows if row["id"] == "qasper-yes_no-5-None"))
    row.update(source_question_id="q-6", source_question_ids=["q-6"], source_test_example_id="q-6")
    replace(scoped_store, row)
    with pytest.raises(ValueError, match="paper/question alias"):
        evaluation.external_test_membership(scoped_store)


@pytest.mark.parametrize("field,value", [("sampling_weight", None), ("sampling_weight", 0), ("sampling_weight", -1),
                                        ("sampling_weight", True), ("sampling_weight", "1"), ("sampling_weight", float("nan")),
                                        ("inclusion_probability", 0), ("inclusion_probability", 1.1),
                                        ("evaluation_scope", "training"), ("original_split", "validation"),
                                        ("role", "calibration"), ("source_test_example_id", None),
                                        ("sampling_design", "invented"), ("sampling_seed", 7),
                                        ("sampling_stratum", "invented"), ("sampling_label", "invented"),
                                        ("dataset_scope_revision", "old"), ("dataset_purpose", "training"),
                                        ("inclusion_probability_numerator", True), ("inclusion_probability_denominator", 0),
                                        ("source_question_ids", []), ("source_paper_id", "another-paper")])
def test_malicious_or_missing_external_metadata_is_rejected(population, field, value):
    meta = deepcopy(next(row for row in population[0] if row.get("adaptation") == "evidence"))
    meta[field] = value
    with pytest.raises(ValueError):
        evaluation.metadata(indexed(meta), scoped=True)


@pytest.mark.parametrize("mutation", ["group", "image_sha256", "source_image_id", "image_path", "target", "question",
                                     "source_test_example_id", "source_response_pair_sha256", "candidate_order",
                                     "correctness", "missing_choice"])
def test_paired_vl_identity_direction_and_original_dispatch_are_not_substitutable(scoped_store, mutation):
    choice, noul = deepcopy(vl_pair())
    if mutation == "missing_choice":
        scoped_store.db.execute("DELETE FROM records WHERE primitive='choice'")
    elif mutation == "candidate_order":
        choice["candidates"].reverse()
        replace(scoped_store, choice)
    else:
        if mutation == "target":
            noul[mutation] = [0., 1.]
        elif mutation == "correctness":
            noul["label_semantics"] = "absolute_correctness"
        else:
            noul[mutation] = "f" * 64 if mutation.endswith("sha256") else "changed"
        replace(scoped_store, noul)
    with pytest.raises(ValueError):
        evaluation.external_test_membership(scoped_store)


def test_original_and_noul_dispatch_remain_separate_methods_and_shared_groups():
    pairs = [*vl_pair(), *vl_pair(1)]
    dispatch = list(evaluation.primitive_batches([(indexed(meta), meta) for meta in pairs]))
    assert [primitive for primitive, _, _ in dispatch] == ["choice", "noul"]
    tracks, groups, parts = [], [], {method: [] for method in METHODS}
    for primitive, _, metas in dispatch:
        assert all(meta["primitive"] == primitive for meta in metas)
        targets = [meta["target"] for meta in metas]
        # Distinct fixture head outputs expose accidental dispatch substitution.
        head = [[3., 0.], [0., 3.]] if primitive == "choice" else [[0., 3.], [3., 0.]]
        for method, values in variants([[0., 0.]] * 2, head, targets).items():
            parts[method].append(values)
        tracks.extend(evaluation.task_track(meta) for meta in metas)
        groups.extend(meta["group"] for meta in metas)
    metrics = {method: {key: np.concatenate([part[key] for part in values]) for key in values[0]}
               for method, values in parts.items()}
    report = compare_tracks(tracks, groups, metrics, draws=10)
    assert set(report["tracks"]) == {evaluation.task_track(meta) for meta in vl_pair()}
    assert report["tracks"][evaluation.task_track(pairs[0])]["methods"]["head_raw"]["accuracy"] == 1
    assert report["tracks"][evaluation.task_track(pairs[1])]["methods"]["head_raw"]["accuracy"] == 0
    assert report["macro_normalized_nll"]["groups"] == 2
    assert tuple(report["tracks"][tracks[0]]["methods"]) == METHODS


@pytest.mark.parametrize("field,value", [("sampling_cluster", "changed"), ("sampling_design", "changed"),
                                        ("sampling_stratum", "changed"), ("dataset_purpose", "changed"),
                                        ("source_question_ids", ["different"]), ("inclusion_probability_numerator", 77),
                                        ("question", "different"), ("image_sha256", "f" * 64)])
def test_membership_hash_binds_design_purpose_prompt_and_provenance(population, field, value):
    original = deepcopy(population[0][0])
    changed = {**original, field: value}
    left, right = hashlib.sha256(), hashlib.sha256()
    evaluation.update_membership(left, original)
    evaluation.update_membership(right, changed)
    assert left.hexdigest() != right.hexdigest()


def test_semantic_alignment_rejects_valid_but_different_sampling_metadata(scoped_store, monkeypatch):
    originals = list(evaluation.test_rows(scoped_store))
    changed = deepcopy(originals)
    selected = next(row for row in changed if json.loads(row["metadata"])["adaptation"] == "evidence")
    meta = json.loads(selected["metadata"])
    meta["sampling_cluster"] = "another-cluster"
    selected["metadata"] = json.dumps(meta)
    candidate = SimpleNamespace(identity=deepcopy(scoped_store.identity))
    monkeypatch.setattr(evaluation, "test_rows", lambda features: iter(changed if features is candidate else originals))
    with pytest.raises(ValueError, match="alignment"):
        list(evaluation.aligned_test_rows(scoped_store, candidate))


@pytest.mark.parametrize("mutation", ["census", "realized_fraction", "offset", "drop_frame", "missing_stratum", "member"])
def test_sampler_attestation_is_not_a_weight_one_or_realized_count_claim(population, mutation):
    rows, design = deepcopy(population)
    sample = [row for row in rows if row.get("adaptation") == "evidence"]
    if mutation == "census":
        for row in sample:
            row.update(sampling_weight=1., inclusion_probability=1., inclusion_probability_numerator=1,
                       inclusion_probability_denominator=1)
    elif mutation == "realized_fraction":
        entry = design["strata"][0]
        entry.update(inclusion_probability_numerator=entry["sampled"], inclusion_probability_denominator=entry["population"])
    elif mutation == "offset":
        design["systematic_offset"] += 1
    elif mutation == "drop_frame":
        design.pop("strata")
    elif mutation == "missing_stratum":
        design["strata"].pop()
    else:
        sample.pop()
    with pytest.raises(ValueError):
        evaluation.validate_sampling_design(design, sample)


def test_external_preparations_cannot_be_omitted_or_training_readiness_used_as_a_fallback(scoped_store, monkeypatch):
    training = {"preparations": [], "records": 100, "counts": {"text/boolq/noul/fit": 100}}
    monkeypatch.setattr(evaluation, "require_complete", lambda features: training)
    result = evaluation.require_final_ready(scoped_store)
    assert result["training"] == training and result["records"] == 3793
    assert len(result["preparations"]) == 3
    scoped_store.readiness["preparations"].pop()
    with pytest.raises(ValueError, match="all three"):
        evaluation.require_final_ready(scoped_store)
    scoped_store.require_external_ready = None
    with pytest.raises(ValueError, match="require_external_ready"):
        evaluation.require_final_ready(scoped_store)


def test_legacy_compatibility_exists_only_when_scope_field_is_absent(scoped_store):
    scoped_store.identity = {}
    assert evaluation.canonical_test_membership(scoped_store)["records"] == 3694
    for scope in (None, {}, {**contract(), "revision": "old"}):
        scoped_store.identity = {"dataset_scope": scope}
        with pytest.raises(ValueError, match="scope"):
            evaluation.canonical_test_membership(scoped_store)


def test_external_fit_or_disguised_training_source_is_not_hidden_by_test_filter(scoped_store):
    meta = deepcopy(vl_pair()[0])
    for changed in ({"role": "fit"}, {"source": "helpsteer3"}):
        replace(scoped_store, {**meta, **changed})
        with pytest.raises(ValueError):
            evaluation.external_test_membership(scoped_store)
        replace(scoped_store, meta)


def test_frozen_producer_origins_are_verified_separately_from_preparation_readiness(scoped_store):
    origins = {"current_runtime_sources": {"fixture": "current"},
               "reused": {"receipt_sha256": "a" * 64, "records": 42, "computation_proof": "synthetic"}}
    scoped_store.producer_origins = lambda: deepcopy(origins)
    evaluation.require_reference_origins(scoped_store, {"feature_origins": deepcopy(origins)})
    for manifest in ({}, {"feature_origins": {}}, {"feature_origins": {**origins, "reused": {}}}):
        with pytest.raises(ValueError, match="producer origins"):
            evaluation.require_reference_origins(scoped_store, manifest)
    scoped_store.producer_origins = None
    with pytest.raises(ValueError, match="producer origins"):
        evaluation.require_reference_origins(scoped_store, {"feature_origins": origins})
    scoped_store.identity = {}
    evaluation.require_reference_origins(scoped_store, {})


def test_full_vl_census_keeps_1247_pairs_with_1238_raw_image_ids(scoped_store):
    scoped_store.db.execute("DELETE FROM records WHERE source='vl-rewardbench'")
    for index in range(1247):
        for meta in vl_pair(index, image_index=index % 1238):
            scoped_store.db.execute("INSERT INTO records VALUES(:id,:shard,:row_,:group_,:role,:primitive,:modality,:source,:metadata)",
                                   indexed(meta))
    membership = evaluation.external_test_membership(scoped_store)
    assert membership["vl_paired_examples"] == 1247
    assert membership["vl_raw_example_ids"] == membership["vl_image_groups"] == 1238
    assert membership["records"] == 3689 + 2 * 1247
    for preparation in scoped_store.readiness["preparations"]:
        if "vl-rewardbench" in preparation["path"]:
            preparation["records"] = 1247
    scoped_store.readiness["records"] = membership["records"]
    scoped_store.readiness["counts"] = membership["counts"]
    assert evaluation.require_external_complete(scoped_store)["records"] == 6183
    # Dropping both representations still leaves every raw image ID present.
    scoped_store.db.execute("DELETE FROM records WHERE id IN ('vl-choice-1238','vl-noul-1238')")
    with pytest.raises(ValueError, match="exact external test membership"):
        evaluation.require_external_complete(scoped_store)


def test_shared_vl_images_cannot_acquire_independent_bootstrap_groups(scoped_store):
    for meta in vl_pair(2, image_index=0):
        meta["group"] = "independent-but-same-image"
        scoped_store.db.execute("INSERT INTO records VALUES(:id,:shard,:row_,:group_,:role,:primitive,:modality,:source,:metadata)",
                               indexed(meta))
    with pytest.raises(ValueError, match="shared image identity and bootstrap group"):
        evaluation.external_test_membership(scoped_store)


@pytest.fixture
def comparison_store(scoped_store, tmp_path, monkeypatch):
    """Real JSONL/SQLite metadata; stub only frozen-head loading/training tensors."""
    store = scoped_store
    helpsteer = {"id": "imported-helpsteer", "source": "helpsteer2", "revision": "fixture", "role": "test",
                 "group": "prompt:helpsteer", "primitive": "choice", "modality": "text",
                 "candidates": [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}],
                 "target": [1., 0.], "adaptation": None, "has_baseline": True}
    mm = {**helpsteer, "id": "mm-heldout", "source": "mm-rlhf", "group": "image:mm", "modality": "image",
          "evaluation_scope": "training", "dataset_purpose": "training", "dataset_scope_revision": "2026-10-02-v2"}
    for meta in (helpsteer, mm):
        store.db.execute("INSERT INTO records VALUES(:id,:shard,:row_,:group_,:role,:primitive,:modality,:source,:metadata)",
                         indexed(meta))
    store.directory = tmp_path / "reference"
    store.directory.mkdir()
    (store.directory / "identity.json").write_text(json.dumps(store.identity))
    store.close = lambda: None
    store.spec = {"features": str(store.directory), "feature_identity_sha256": evaluation.file_sha256(store.directory / "identity.json"),
                  "head": str(tmp_path / "synthetic-frozen-head"), "sha256": {name: "a" * 64 for name in evaluation.HEAD_FILES}}
    store.helpsteer_input = {**helpsteer, "question": "original immutable prompt", "original_split": "validation"}
    rows = [*store.rows, store.helpsteer_input, mm]
    training = {"preparations": [], "records": 0, "counts": {}}
    for name, selected in (
            ("text.jsonl", [store.helpsteer_input]), ("boolq.jsonl", [row for row in rows if row["source"] == "boolq"]),
            ("images.jsonl", [mm]), ("qasper-external.jsonl", [row for row in rows if row["source"] == "qasper"]),
            *((f"vl-rewardbench-{primitive}.jsonl", [row for row in rows if row["source"] == "vl-rewardbench" and
                                                     row["primitive"] == primitive]) for primitive in ("choice", "noul"))):
        path = tmp_path / "data/prepared" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps({key: value for key, value in row.items() if key != "has_baseline"}) + "\n"
                                for row in selected))
        prepared = {"path": str(path), "records": len(selected), "sha256": evaluation.file_sha256(path),
                    "manifest_sha256": "b" * 64}
        if name in {"text.jsonl", "boolq.jsonl", "images.jsonl"}:
            training["preparations"].append(prepared)
            training["records"] += len(selected)
            for row in selected:
                key = "/".join(row[field] for field in ("modality", "source", "primitive", "role"))
                training["counts"][key] = training["counts"].get(key, 0) + 1
        else:
            current = next(item for item in store.readiness["preparations"] if item["path"] == str(path))
            current.update({key: prepared[key] for key in ("path", "records", "sha256")})
    store.training = training
    monkeypatch.setattr(evaluation, "require_complete", lambda features: deepcopy(store.training))
    monkeypatch.setattr(comparison_inputs, "open_features", lambda *args: store)
    monkeypatch.setattr(comparison_inputs, "load_reference", lambda *args: (None, {}, {}))
    monkeypatch.setattr(comparison_inputs, "check_hashes", lambda *args: None)
    return store


def test_comparison_inputs_include_all_external_preparations_and_pin_final_readiness(comparison_store, tmp_path):
    store, destination = comparison_store, tmp_path / "inputs"
    readiness = evaluation.require_final_ready(store)
    receipt = comparison_inputs.prepare_inputs(store.spec, destination)
    assert receipt["reference"]["preparation_sha256"] == evaluation.sha256(readiness)
    assert receipt["reference"]["preparation_sha256"] != evaluation.sha256(store.training)
    expected = b"".join(open(item["path"], "rb").read() for item in readiness["preparations"])
    assert (destination / comparison_inputs.INPUTS).read_bytes() == expected
    pin = evaluation.file_sha256(destination / comparison_inputs.MANIFEST)
    selected = list(comparison_inputs.iter_inputs(destination, pin, store.spec))
    assert len(selected) == 3696
    assert {row["source"] for row in selected} == {"helpsteer2", "boolq", "mm-rlhf", "qasper", "vl-rewardbench"}
    # Enriched comparison metadata must align and hash like the imported reference.
    original = deepcopy(next(row for row in evaluation.test_rows(store) if row["id"] == "imported-helpsteer"))
    a, b = hashlib.sha256(), hashlib.sha256()
    evaluation.update_membership(a, json.loads(original["metadata"]))
    evaluation.update_membership(b, store.helpsteer_input)
    assert a.hexdigest() == b.hexdigest()


def test_comparison_consumer_rejects_training_only_preparation_pin(comparison_store, tmp_path):
    destination = tmp_path / "inputs"
    receipt = comparison_inputs.prepare_inputs(comparison_store.spec, destination)
    receipt["reference"]["preparation_sha256"] = evaluation.sha256(comparison_store.training)
    manifest = destination / comparison_inputs.MANIFEST
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="another frozen reference"):
        list(comparison_inputs.iter_inputs(destination, evaluation.file_sha256(manifest), comparison_store.spec))


@pytest.mark.parametrize("mutation", ["preparation", "membership", "sampling_frame"])
def test_comparison_inputs_fail_before_copying_without_complete_external_preparations(comparison_store, tmp_path, monkeypatch, mutation):
    if mutation == "preparation":
        comparison_store.readiness["preparations"].pop()
    elif mutation == "membership":
        comparison_store.db.execute("DELETE FROM records WHERE id='vl-choice-0'")
    else:
        comparison_store.manifest.write_text("{}")
    monkeypatch.setattr(comparison_inputs, "jsonl_rows", lambda _: pytest.fail("No prepared inputs may be copied before readiness"))
    destination = tmp_path / "inputs"
    with pytest.raises(ValueError):
        comparison_inputs.prepare_inputs(comparison_store.spec, destination)
    assert not destination.exists()


def test_frozen_reference_rejects_changed_final_preparation_identity(comparison_store):
    with pytest.raises(ValueError, match="final preparations"):
        with comparison_inputs.frozen_reference(comparison_store.spec):
            comparison_store.training["preparations"][0]["manifest_sha256"] = "c" * 64


@pytest.mark.parametrize("key,value", [("sampling_weight", 2.), ("dataset_purpose", "training"),
                                      ("question", "Is response A absolutely correct?"), ("source_example_id", "changed")])
def test_comparison_external_semantics_cannot_use_helpsteer_legacy_exception(comparison_store, key, value):
    row = {**vl_pair()[0], key: value}
    with pytest.raises(ValueError):
        comparison_inputs.require_semantics(comparison_store, row, {row["id"]})


def test_comparison_seal_pins_final_readiness_not_training_only(comparison_store, tmp_path, monkeypatch):
    store, directory = comparison_store, tmp_path / "comparison"
    directory.mkdir()
    identity = {**store.identity, "corpus_scope": evaluation.TEST_SCOPE}
    (directory / "identity.json").write_text(json.dumps(identity))
    store.db.execute("CREATE TABLE preparations(path TEXT, sha256 TEXT)")
    store.db.commit()
    with sqlite3.connect(directory / "index.sqlite3") as db:
        store.db.backup(db)
    # This fixture exercises receipt wiring/alignment, not tensor ownership.
    # The existing Torch integration tests independently cover actual shard seals.
    monkeypatch.setattr(comparison_seal, "require_feature_ownership", lambda features: [])
    verified = []

    def check(reference, features, pin):
        manifest = json.loads((directory / evaluation.SUBSET_FILE).read_text())
        assert pin == evaluation.file_sha256(directory / evaluation.SUBSET_FILE)
        assert manifest["reference"]["preparation_sha256"] == evaluation.sha256(evaluation.require_final_ready(reference))
        assert manifest["reference"]["test_membership"] == evaluation.canonical_test_membership(reference)
        verified.append(True)

    monkeypatch.setattr(comparison_seal, "require_comparison_complete", check)
    result = comparison_seal.seal_features(directory, store.spec, 3696)
    assert verified == [True]
    assert result["manifest"]["reference"]["preparation_sha256"] != evaluation.sha256(store.training)
