import numpy as np
import pytest

from fuzev_research.metrics import evidence_metrics, paired_group_interval, per_record, probabilities, summarize


def test_masked_targets_and_ties_have_explicit_metric_semantics():
    logits = [[2, 0, 100], [1, 1, 100]]
    target = [[1, 0, 0], [.5, .5, 0]]
    values = per_record(logits, target, [[True, True, False]] * 2)
    assert np.isnan(values["accuracy"][1])
    assert values["expected_accuracy"].tolist() == [1, .5]
    assert values["tie_aware_accuracy"].all()
    assert np.isfinite(values["nll"]).all()
    assert summarize(values)["accuracy"] == 1


def test_calibration_preserves_order_and_ordinal_score_is_finite():
    logits = np.array([[1, 2, 4]])
    assert probabilities(logits, 3).argmax() == probabilities(logits).argmax()
    values = per_record(logits, [[0, 0, 1]], ordinal_values=[[0, 1, 2]])
    assert 0 < values["ordinal_mae"][0] < 2
    assert values["ranked_probability_score"][0] >= 0


def test_paired_group_uncertainty_retains_groups_and_is_reproducible():
    result = paired_group_interval([1, 1, 2], [2, 2, 3], ["a", "a", "b"], draws=50)
    assert result["groups"] == 2
    assert result["low"] == result["high"] == 1
    assert result == paired_group_interval([1, 1, 2], [2, 2, 3], ["a", "a", "b"], draws=50)


def test_evidence_excludes_tied_votes_not_all_negative_cases():
    scores = evidence_metrics([.9, .8, .1], [1, .5, 0])
    assert scores["eligible_records"] == 2
    assert scores["f1"] == 1


def test_nonfinite_or_all_masked_logits_fail_closed():
    with pytest.raises(ValueError):
        probabilities([[float("inf"), 0]])
    with pytest.raises(ValueError):
        probabilities([[float("-inf"), float("-inf")]])
