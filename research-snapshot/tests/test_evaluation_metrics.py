import numpy as np
import pytest

from fuzev_research.evaluation_metrics import METHODS, compare_tracks, gate, interval
from fuzev_research.metrics import evidence_metrics, per_record, probabilities, summarize


def methods(baseline, head, targets, temperature=2., ordinal_values=None):
    return {name: per_record(baseline if name == "label_token" else head, targets,
                             temperature=temperature if name == "head_calibrated" else 1.,
                             ordinal_values=ordinal_values) for name in METHODS}


def test_all_existing_metrics_and_ece_match_direct_summary():
    target = np.array([[1, 0, 0], [.5, .5, 0], [0, 0, 1], [0, 1, 0]])
    baseline, head = np.array([[0, 1, 2]] * 4), np.array([[2, 1, 0]] * 4)
    variants = methods(baseline, head, target, ordinal_values=[[0, 1, 2]] * 4)
    report = compare_tracks(["ordinal"] * 4, ["a", "a", "b", "c"], variants, draws=30)
    for method in METHODS:
        expected = summarize(variants[method])
        actual = report["tracks"]["ordinal"]["methods"][method]
        for key, value in actual.items():
            assert value == pytest.approx(expected[key])
    assert report["tracks"]["ordinal"]["eligible_records"]["accuracy"] == 3


def test_evidence_f1_is_group_resampled_count_ratio_not_per_record_score():
    target = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [.5, .5]])
    base = np.array([[3, 0], [0, 3], [3, 0], [0, 3], [3, 0]])
    head = np.array([[3, 0], [3, 0], [0, 3], [3, 0], [3, 0]])
    variants = methods(base, head, target)
    groups = np.array(["paper-a", "paper-a", "paper-a", "paper-b", "paper-c"])
    evidence = {name: (probabilities(base if name == "label_token" else head)[:, 0], target[:, 0])
                for name in METHODS}
    report = compare_tracks(["evidence"] * 5, groups, variants, evidence, draws=60, seed=31)
    track = report["tracks"]["evidence"]
    for method in METHODS:
        expected = evidence_metrics(*evidence[method])
        assert track["methods"][method]["evidence_f1"] == pytest.approx(expected["f1"])
        assert track["methods"][method]["evidence_eligible_records"] == 4
    rng = np.random.default_rng(31)
    unique = np.unique(groups)
    differences = []
    for _ in range(60):
        selected = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([np.flatnonzero(groups == unique[i]) for i in selected])
        if not (target[rows, 0] != .5).any():
            continue
        differences.append(evidence_metrics(evidence["head_raw"][0][rows], target[rows, 0])["f1"] -
                           evidence_metrics(evidence["label_token"][0][rows], target[rows, 0])["f1"])
    result = track["paired"]["head_raw"]["evidence_f1"]
    assert [result["low"], result["high"]] == pytest.approx(np.quantile(differences, [.025, .975]))
    assert result["valid_draws"] == len(differences)
    assert result["groups"] == 2  # tied-only paper supplies no hard F1 evidence


def test_macro_equal_track_weight_and_shared_groups_match_manual_bootstrap():
    tracks = np.array(["short", "long", "long", "long", "short", "long", "long", "long"])
    groups = np.array(["a"] * 4 + ["b"] * 4)
    baseline = np.zeros((8, 2))
    head = np.array([[3, 0], [0, 1], [0, 1], [0, 1], [1, 0], [0, 2], [0, 2], [0, 2]])
    variants = methods(baseline, head, [[1, 0]] * 8)
    report = compare_tracks(tracks, groups, variants, draws=50)
    losses = variants["head_raw"]["normalized_nll"] - variants["label_token"]["normalized_nll"]
    expected_point = np.mean([losses[tracks == track].mean() for track in np.unique(tracks)])
    assert expected_point != pytest.approx(losses.mean())
    rng = np.random.default_rng(17)
    differences = []
    for _ in range(50):
        selected = rng.integers(0, 2, 2)
        rows = np.concatenate([np.flatnonzero(groups == ("a", "b")[i]) for i in selected])
        differences.append(np.mean([losses[rows][tracks[rows] == track].mean() for track in np.unique(tracks)]))
    interval_ = report["macro_normalized_nll"]["paired"]["head_raw"]
    assert interval_["difference"] == pytest.approx(expected_point)
    assert [interval_["low"], interval_["high"]] == pytest.approx(np.quantile(differences, [.025, .975]))
    assert report == compare_tracks(tracks, groups, variants, draws=50)


def test_ties_are_not_relabelled_as_wins_or_used_as_hard_accuracy():
    variants = methods([[0, 1]] * 3, [[1, 0]] * 3, [[.5, .5]] * 3)
    report = compare_tracks(["preference"] * 3, ["a", "b", "c"], variants, draws=20)
    track = report["tracks"]["preference"]
    for method in METHODS:
        assert track["methods"][method]["accuracy"] is None
        assert track["methods"][method]["expected_accuracy"] == .5
        assert track["methods"][method]["tie_aware_accuracy"] == 1.
    assert not track["paired"]["head_raw"]["accuracy"]["supported"]


def test_singleton_or_missing_track_draws_cannot_pass_gate():
    variants = methods([[0, 2]] * 4, [[2, 0]] * 4, [[1, 0]] * 4)
    report = compare_tracks(["rare", "common", "common", "common"], ["a", "b", "c", "d"], variants, draws=40)
    macro = report["macro_normalized_nll"]["paired"]["head_raw"]
    assert macro["valid_draws"] < 40 and not macro["supported"]
    assert gate(macro, 0., strict=True)["status"] == "inconclusive"


@pytest.mark.parametrize("low,high,margin,higher,strict,expected", [
    (-.02, .01, .02, True, False, "pass"),
    (-.021, .01, .02, True, False, "inconclusive"),
    (-.03, -.021, .02, True, False, "fail"),
    (.02, .1, .1, False, False, "pass"),
    (.09, .11, .1, False, False, "inconclusive"),
    (.101, .11, .1, False, False, "fail"),
    (-.01, 0., 0., False, True, "inconclusive"),
    (-.01, -.001, 0., False, True, "pass"),
    (0., .001, 0., False, True, "fail"),
])
def test_predeclared_gate_bounds(low, high, margin, higher, strict, expected):
    estimate = {"supported": True, "low": low, "high": high}
    assert gate(estimate, margin, higher, strict)["status"] == expected


def test_alignment_and_nonfinite_metrics_fail_closed():
    variants = methods([[0, 1]] * 2, [[1, 0]] * 2, [[1, 0]] * 2)
    with pytest.raises(ValueError, match="aligned"):
        compare_tracks(["one"], ["a", "b"], variants)
    variants["head_raw"]["nll"][0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        compare_tracks(["one"] * 2, ["a", "b"], variants)
    assert interval(0., [0., 0.], 1, 2, 17)["supported"] is False


def test_complete_134515_row_holdout_is_not_rejected_or_truncated():
    records = 134_515
    tracks = np.full(records, "complete-holdout")
    groups = np.arange(records) % 8
    variants = {
        method: {"normalized_nll": np.full(records, value),
                 "confidence": np.full(records, .6),
                 "expected_accuracy": np.full(records, .8)}
        for method, value in zip(METHODS, (.8, .7, .65))
    }
    report = compare_tracks(tracks, groups, variants, draws=2)
    assert report["tracks"]["complete-holdout"]["records"] == records
    assert report["macro_normalized_nll"]["methods"]["head_calibrated"] == pytest.approx(.65)
