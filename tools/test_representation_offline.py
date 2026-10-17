"""Synthetic CPU goldens for serialized capture conversion and offline CLI glue."""

from copy import deepcopy
import json
import math

import pytest
from safetensors.torch import save_file
import torch

from fuzev.encoding import LIST_AWARE_VERSION
import representation_probe as probe
from test_representation_probe import GEOMETRY, row
from train_shared_head import initialize


def capture_fixture(roles=("fit", "development")):
    records, sources = [], []
    for role in roles:
        for unavailable in (False, True):
            group = f"{role}-{unavailable}"
            for order in ("ab", "ba"):
                source = row(f"{group}-{order}", 2, role)
                source.update(group=group, order=order, source_modality="image")
                source["baseline"] = torch.tensor([1.25, -2.5])
                if order == "ba":
                    source["candidate_ids"].reverse()
                    for name in ("baseline", "target"):
                        source[name] = source[name].flip(0)
                    for features in source["views"].values():
                        features["candidates"] = features["candidates"].flip(0)
                for name in ("baseline", "target"):
                    source[name] = source[name].tolist()
                source["views"] = {view: {name: value.tolist() for name, value in features.items()}
                                   for view, features in source["views"].items()}
                source["head"] = {"scores": [{"id": identifier, "logit": {"c0": 9., "c1": -4.}[identifier]}
                                             for identifier in reversed(source["candidate_ids"])]}
                records.append({**{name: source[name] for name in ("id", "group", "role", "order", "primitive", "target")},
                                "modality": "image", "positive_id": "c1",
                                "source_image_id": group, "source_example_id": group,
                                "question": "Which caption?", "context": [], "image_path": "synthetic.jpg",
                                "candidates": [{"id": identifier, "text": f"Caption {identifier}"}
                                               for identifier in source["candidate_ids"]]})
                sources.append(source)
    return {"head": "synthetic-v1", "representation_rows": sources[::-1]}, records


def inputs(tmp_path, *, image_support=True):
    capture, records = capture_fixture()
    feature = {"tap": "final-pre-lm", "pooling": "last-token"}
    config = dict(GEOMETRY, schema="fuzev-shared-head-package-v2",
                  architecture_version="fuzev-shared-decision-head-v1", encoding_version=LIST_AWARE_VERSION,
                  features={"query": feature, "candidate": feature},
                  baseline={"labelled": "query-label-logits", "label_free": "zero"},
                  capacities={"choice": {"text": {"implemented_max": 2, "validated_max": None}}},
                  residual_strength={"text": 1.})
    if image_support:
        config["capacities"]["choice"]["image_text"] = {"implemented_max": 2, "validated_max": None}
        config["residual_strength"]["image_text"] = 0.
    model = initialize(GEOMETRY, 43)
    with torch.no_grad():
        model.scorer.key.weight.zero_()
        model.scorer.key.bias.zero_()
        model.gate.bias.fill_(math.log(3.))
    head = tmp_path / "head"
    head.mkdir()
    save_file(model.state_dict(), head / "head.safetensors")
    (head / "config.json").write_text(json.dumps(config))
    (head / "calibration.json").write_text(json.dumps({"strata": [
        {"primitive": "choice", "modality": modality, "k_min": 2, "k_max": 2, "temperature": 25.}
        for modality in config["residual_strength"]]}))
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    for role in ("fit", "development"):
        (tmp_path / f"{role}.jsonl").write_text("".join(
            json.dumps(record) + "\n" for record in records if record["role"] == role))
    return ["--capture", str(tmp_path / "capture.json"), "--head", str(head),
            "--fit-records", str(tmp_path / "fit.jsonl"), "--dev-records", str(tmp_path / "development.jsonl"),
            "--device", "cpu", "--score-batch-size", "3"]


def test_tensorize_and_id_join_preserve_roles_and_orders():
    capture, records = capture_fixture()
    original = deepcopy(capture)
    rows = probe.tensorize_capture(capture, records, GEOMETRY["hidden_size"])
    assert capture == original
    assert len(rows) == len(records)
    assert all(r["baseline"].dtype == torch.float32 and r["target"].dtype == torch.float32 for r in rows)
    scores = [{"id": r["id"], "candidate_ids": ["c1", "c0"], "utilities": [2., -3.]} for r in rows[::-1]]
    for result, source in zip(probe.aligned_score_rows(rows, scores), rows):
        assert result["candidate_ids"] == source["candidate_ids"]
        assert result["role"] == source["role"] and result["order"] == source["order"]
        assert result["head"]["scores"] == [{"id": identifier, "logit": {"c1": 2., "c0": -3.}[identifier]}
                                            for identifier in source["candidate_ids"]]


@pytest.mark.parametrize("image_support", [True, False])
def test_rescore_cli_golden_ignores_image_strength_and_text_calibration(tmp_path, image_support):
    args = inputs(tmp_path, image_support=image_support)
    output = tmp_path / "rescore"
    probe.main(["rescore", *args, "--output", str(output)])
    for role in ("fit", "development"):
        v2 = [json.loads(line) for line in (output / f"v2-{role}-scores.jsonl").read_text().splitlines()]
        v1 = [json.loads(line) for line in (output / f"v1-{role}-scores.jsonl").read_text().splitlines()]
        assert len(v2) == len(v1) == 4
        for actual, diagnostic in zip(v2, v1):
            assert actual["head_source"] == str(tmp_path / "head")
            assert diagnostic["head_source"] == "synthetic-v1"
            assert actual["baseline"] == diagnostic["baseline"]
            base = torch.tensor([entry["logit"] for entry in actual["baseline"]["scores"]])
            expected = base + .5 * (base - base.mean())
            torch.testing.assert_close(torch.tensor([entry["logit"] for entry in actual["head"]["scores"]]), expected)
            assert diagnostic["head"]["scores"] == [
                {"id": identifier, "logit": {"c0": 9., "c1": -4.}[identifier]} for identifier in actual["candidate_ids"]]
    assert {path.name for path in output.iterdir()} == {
        f"{prefix}-{role}-scores.jsonl" for prefix in ("v1", "v2") for role in ("fit", "development")}


def test_frozen_eval_rescore_cli_preserves_raw_utilities_and_diagnostics(tmp_path):
    inputs(tmp_path, image_support=False)
    capture, records = capture_fixture(("test",))
    capture_path, records_path = tmp_path / "eval-capture.json", tmp_path / "eval.jsonl"
    capture_path.write_text(json.dumps(capture))
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    raw_output = tmp_path / "eval-raw"
    probe.main(["rescore", "--capture", str(capture_path), "--head", str(tmp_path / "head"),
                "--eval-records", str(records_path), "--device", "cpu", "--output", str(raw_output)])
    raw = [json.loads(line) for line in (raw_output / "v2-test-scores.jsonl").read_text().splitlines()]
    diagnostic = [json.loads(line) for line in (raw_output / "v1-test-scores.jsonl").read_text().splitlines()]
    assert len(raw) == len(diagnostic) == len(records)
    by_id = {row["id"]: row for row in capture["representation_rows"]}
    for actual, v1 in zip(raw, diagnostic):
        source = by_id[actual["id"]]
        assert actual["role"] == "test" and actual["order"] == source["order"]
        assert actual["group"] == source["group"] and actual["id"] == v1["id"]
        assert actual["candidate_ids"] == v1["candidate_ids"]
        assert actual["baseline"] == v1["baseline"]
        base = torch.tensor([entry["logit"] for entry in actual["baseline"]["scores"]])
        torch.testing.assert_close(torch.tensor([entry["logit"] for entry in actual["head"]["scores"]]),
                                   base + .5 * (base - base.mean()))
        assert v1["head"]["scores"] == [
            {"id": identifier, "logit": {"c0": 9., "c1": -4.}[identifier]} for identifier in source["candidate_ids"]]
