"""Annotation-only, fixed-size sampling with exact first-order probabilities."""

import json
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction


SEED = 20261002
DESIGN = "paper-question-label-systematic-srs-v1"


def _equal_quotas(capacities, budget):
    """Water-fill equally, redistributing the quota of exhausted children."""
    budget = Fraction(budget)
    if not capacities or budget <= 0 or budget > sum(capacities.values()):
        raise ValueError("Infeasible positive sampling budget")
    pending, quotas = dict(capacities), {}
    while pending:
        share = budget / len(pending)
        capped = [key for key, capacity in pending.items() if capacity <= share]
        if not capped:
            quotas.update({key: share for key in pending})
            break
        for key in capped:
            quota = Fraction(pending.pop(key))
            quotas[key] = quota
            budget -= quota
    return quotas


def _systematic_counts(quotas, offset, denominator):
    """Exact systematic rounding; offset is uniform on range(denominator)."""
    if not 0 <= offset < denominator:
        raise ValueError("Invalid systematic offset")
    cumulative = 0
    counts = []
    for quota in quotas:
        scaled = quota * denominator
        if scaled.denominator != 1:
            raise ValueError("Systematic denominator must represent every quota exactly")
        previous = cumulative
        cumulative += int(scaled)
        counts.append((cumulative + offset) // denominator - (previous + offset) // denominator)
    if cumulative % denominator:
        raise ValueError("Sampling quotas must sum to an integer")
    return counts


def evidence_label(target):
    if (len(target) != 2 or any(not math.isfinite(x) or x < 0 for x in target)
            or not math.isclose(sum(target), 1)):
        raise ValueError("Evidence target must be a binary probability distribution")
    return "no" if target[0] == 0 else "yes" if target[0] == 1 else "mixed"


def sample_evidence(population, size=2000, seed=SEED):
    """Return sampled copies and an auditable frame, without rendering prompts.

    Strata are (paper group, question ID, annotation-vote label). Hierarchical
    expected quotas, not realized stratum counts, determine marginal pi.
    """
    if type(size) is not int or size <= 0 or type(seed) is not int:
        raise ValueError("Sample size must be a positive integer and seed an integer")
    strata, seen = defaultdict(list), set()
    for row in population:
        if row["id"] in seen:
            raise ValueError("Deduplicate the evidence population before sampling")
        seen.add(row["id"])
        key = (row["group"], row["source_question_id"], evidence_label(row["target"]))
        strata[key].append(row)
    if size > len(seen):
        raise ValueError(f"Evidence pool has {len(seen)} rows; cannot sample exactly {size}")

    paper_sizes, question_sizes = Counter(), defaultdict(Counter)
    for (paper, question, _), rows in strata.items():
        paper_sizes[paper] += len(rows)
        question_sizes[paper][question] += len(rows)
    paper_quotas = _equal_quotas(paper_sizes, size)
    question_quotas = {paper: _equal_quotas(sizes, paper_quotas[paper])
                       for paper, sizes in question_sizes.items()}
    label_sizes = defaultdict(dict)
    for (paper, question, label), rows in strata.items():
        label_sizes[paper, question][label] = len(rows)
    quotas = {}
    for (paper, question), sizes in label_sizes.items():
        for label, quota in _equal_quotas(sizes, question_quotas[paper][question]).items():
            quotas[paper, question, label] = quota

    # Contiguous paper/question blocks also receive floor/ceil of their quotas.
    keys = sorted(strata)
    denominator = math.lcm(*(quota.denominator for quota in quotas.values()))
    rng = random.Random(seed)
    offset = rng.randrange(denominator)
    counts = _systematic_counts([quotas[key] for key in keys], offset, denominator)
    selected, frame = [], []
    for key, count in zip(keys, counts):
        rows = sorted(strata[key], key=lambda row: row["id"])
        quota = quotas[key]
        probability = quota / len(rows)
        stratum = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
        metadata = {"sampling_stratum": stratum, "sampling_label": key[2],
                    "inclusion_probability": float(probability),
                    "sampling_weight": float(1 / probability),
                    "inclusion_probability_numerator": probability.numerator,
                    "inclusion_probability_denominator": probability.denominator,
                    "sampling_seed": seed, "sampling_design": DESIGN,
                    "sampling_cluster": key[0], "population_size": len(seen), "sample_size": size}
        for row in sorted(rng.sample(rows, count), key=lambda row: row["id"]):
            selected.append({**row, **metadata})
        frame.append({"sampling_stratum": stratum, "population": len(rows), "sampled": count,
                      "expected_quota_numerator": quota.numerator,
                      "expected_quota_denominator": quota.denominator,
                      "inclusion_probability_numerator": probability.numerator,
                      "inclusion_probability_denominator": probability.denominator})
    if len(selected) != size or sum(quotas.values()) != size:
        raise AssertionError("Fixed-size evidence sampling invariant failed")
    summary = {"design": DESIGN, "seed": seed, "population": len(seen), "sample_size": size,
               "population_papers": len(paper_sizes),
               "population_questions": sum(map(len, question_sizes.values())),
               "sampled_papers": len({row["group"] for row in selected}),
               "sampled_questions": len({(row["group"], row["source_question_id"]) for row in selected}),
               "label_population": {label: sum(len(rows) for key, rows in strata.items() if key[2] == label)
                                    for label in ("no", "mixed", "yes")},
               "label_sample": dict(Counter(row["sampling_label"] for row in selected)),
               "systematic_offset": offset, "systematic_denominator": denominator,
               "strata": frame}
    return selected, summary
