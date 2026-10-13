from collections import Counter
from fractions import Fraction
from itertools import product

import pytest

from fuzev_research.external_sampling import _equal_quotas, _systematic_counts, sample_evidence


def population():
    result = []
    for paper, questions in (("a", 3), ("b", 1), ("c", 2)):
        for question in range(questions):
            for label, size in ((0., 7), (.5, 2), (1., 1)):
                for item in range(size):
                    result.append({"id": f"{paper}-{question}-{label}-{item}", "group": f"paper:{paper}",
                                   "source_question_id": f"q{question}", "target": [label, 1 - label]})
    return result


def test_water_filling_caps_then_redistributes_exactly():
    assert _equal_quotas({"tiny": 1, "large": 20, "medium": 8}, 10) == {
        "tiny": Fraction(1), "large": Fraction(9, 2), "medium": Fraction(9, 2)}


def test_exact_systematic_expectations_and_fixed_size_for_every_offset():
    quotas = [Fraction(1, 3), Fraction(4, 3), Fraction(3, 2), Fraction(5, 6)]
    total = [0] * len(quotas)
    for offset in range(6):
        counts = _systematic_counts(quotas, offset, 6)
        assert sum(counts) == 4
        for index, (count, quota) in enumerate(zip(counts, quotas)):
            assert count in {quota.numerator // quota.denominator, -(-quota.numerator // quota.denominator)}
            total[index] += count
    assert [Fraction(value, 6) for value in total] == quotas


def test_sampling_determinism_coverage_and_inverse_probability_metadata():
    rows = population()
    selected, summary = sample_evidence(rows, size=12)
    assert sample_evidence(reversed(rows), size=12) == (selected, summary)
    assert len(selected) == len({row["id"] for row in selected}) == 12
    assert Counter(row["group"] for row in selected) == {"paper:a": 4, "paper:b": 4, "paper:c": 4}
    assert summary["population"] == len(rows)
    assert summary["seed"] == 20261002
    frame = {item["sampling_stratum"]: item for item in summary["strata"]}
    assert sum(Fraction(item["inclusion_probability_numerator"], item["inclusion_probability_denominator"])
               * item["population"] for item in frame.values()) == 12
    for row in selected:
        stratum = frame[row["sampling_stratum"]]
        probability = Fraction(row["inclusion_probability_numerator"], row["inclusion_probability_denominator"])
        expected_quota = Fraction(stratum["expected_quota_numerator"], stratum["expected_quota_denominator"])
        assert probability == expected_quota / stratum["population"]
        assert row["inclusion_probability"] == float(probability)
        assert row["sampling_weight"] * row["inclusion_probability"] == pytest.approx(1)
        assert row["sampling_cluster"] == row["group"]
        assert row["population_size"] == len(rows)
        assert row["sample_size"] == 12
    assert len({row["inclusion_probability"] for row in selected}) > 1
    assert any(Fraction(item["expected_quota_numerator"], item["expected_quota_denominator"]) != item["sampled"]
               for item in frame.values())
    assert all("sampling_weight" not in row for row in rows)


def test_exact_2000_without_model_inputs():
    rows = [{"id": f"{paper}:{question}:{paragraph}", "group": f"paper:{paper}",
             "source_question_id": str(question), "target": [float(paragraph == 0), float(paragraph != 0)]}
            for paper, question, paragraph in product(range(50), range(20), range(4))]
    selected, summary = sample_evidence(rows)
    assert len(selected) == 2000
    assert summary["sampled_papers"] == 50
    assert summary["sampled_questions"] == 1000
    assert summary["label_sample"] == {"no": 1000, "yes": 1000}


def test_census_and_very_small_samples_keep_positive_frame_probabilities():
    rows = population()
    census, summary = sample_evidence(rows, len(rows))
    assert len(census) == len(rows)
    assert all(row["sampling_weight"] == row["inclusion_probability"] == 1 for row in census)
    selected, summary = sample_evidence(rows, 1)
    assert len(selected) == 1
    assert all(item["inclusion_probability_numerator"] > 0 for item in summary["strata"])


def test_marginals_match_declared_probabilities_across_fixed_seeds():
    rows = population()
    observed, expected = Counter(), {}
    for seed in range(600):
        selected, _ = sample_evidence(rows, 12, seed=seed)
        for row in selected:
            observed[row["id"]] += 1
            expected[row["id"]] = row["inclusion_probability"]
    assert set(observed) == {row["id"] for row in rows}
    assert all(abs(observed[key] / 600 - probability) < .07 for key, probability in expected.items())


def test_bad_sampling_frames_fail_instead_of_silent_shortfalls():
    rows = population()
    with pytest.raises(ValueError, match="Deduplicate"):
        sample_evidence([rows[0], rows[0]], 1)
    with pytest.raises(ValueError, match="cannot sample exactly"):
        sample_evidence(rows, len(rows) + 1)
    for size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            sample_evidence(rows, size)
    with pytest.raises(ValueError, match="binary probability"):
        sample_evidence([{**rows[0], "target": [float("nan"), 0]}], 1)
