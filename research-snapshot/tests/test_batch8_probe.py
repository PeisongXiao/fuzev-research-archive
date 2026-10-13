import json

import pytest

torch = pytest.importorskip("torch")

from tools.probe_batch8 import NATIVE, compare, control_report, select, stable, verify_control_sources


def fixture(track, index):
    return {"track": track, "row": {"id": f"{track}-{index}", "role": "fit"}}


def test_probe_keeps_four_natural_ordinal_cases_and_other_fit_tracks(tmp_path):
    rows = [fixture("ordinal", index) for index in range(4)]
    rows += [fixture(track, 0) for track in ("text_choice", "qasper_noul", "aok_choice", "mm_choice")]
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps({"rows": rows}))
    assert select(path) == rows
    rows[-1]["row"]["role"] = "test"
    path.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="fitting"):
        select(path)


def test_physical4_control_does_not_accept_b4_chunks_on_eight_slots(tmp_path):
    selected = [fixture("ordinal", 0)]
    path = tmp_path / "report.json"
    report = {"status": "fit_stability_pass_quality_pending", "physical_batch_capacity": 8,
              "fixture_ids": [selected[0]["row"]["id"]],
              "initial_runtime": {key: 4 for key in ("batch_capacity", "model_batch_capacity", "max_hybrid_slots", "candidate_batch_size")}}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="physical-4"):
        control_report(path, selected)
    report["physical_batch_capacity"] = 4
    path.write_text(json.dumps(report))
    assert control_report(path, selected) == report
    report["initial_runtime"]["max_hybrid_slots"] = 8
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="actually declare four"):
        control_report(path, selected)


def test_centered_drift_is_reported_even_with_a_shared_candidate_offset():
    execution = {"base_token_sha256": "base", "query_token_sha256": "query", "prefix_tokens": 1,
                 "branch_prefix_tokens": 2, "expanded_image_tokens": 0}
    a = {"query": [1., 2.], "candidates": [[1., 2.], [2., 1.]], "baseline": [0., 1.], "execution": execution}
    b = {**a, "candidates": [[4., 5.], [5., 4.]]}
    metrics = compare(a, b)
    assert metrics["candidates"]["max"] == 3
    assert metrics["centered_candidates"]["max"] == 0
    assert stable(compare(a, a)) and not stable(metrics)
    b["execution"] = {**execution, "base_token_sha256": "different-image-alias"}
    with pytest.raises(ValueError, match="alias identity"):
        compare(a, b)


def test_control_sources_record_observed_and_control_revisions(monkeypatch):
    control = {"recipes": "before", "exllamav3-fuzev": NATIVE}
    candidate = {"recipes": "after", "exllamav3-fuzev": "capacity-pin"}
    result = verify_control_sources(control, candidate)
    assert result["control_sources"] == control and result["candidate_sources"] == candidate
    assert result["recipe_changes"] is None
    assert verify_control_sources({**control, "exllamav3-fuzev": None}, candidate)["candidate_sources"] == candidate
