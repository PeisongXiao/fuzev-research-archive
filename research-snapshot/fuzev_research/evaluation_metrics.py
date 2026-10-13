"""Paired final-test statistics, with groups kept intact across task tracks."""

import numpy as np


METHODS = ("label_token", "head_raw", "head_calibrated")
MAX_RECORDS = 200_000
MAX_TRACKS = 128
BINS = 15


def interval(difference, samples, groups, draws, seed):
    samples = np.asarray(samples, float)
    finite = samples[np.isfinite(samples)]
    low, high = np.quantile(finite, [.025, .975]) if len(finite) else (None, None)
    return {"difference": float(difference) if np.isfinite(difference) else None,
            "low": float(low) if low is not None else None,
            "high": float(high) if high is not None else None,
            "groups": int(groups), "draws": draws, "valid_draws": len(finite), "seed": seed,
            "supported": bool(groups >= 2 and len(finite) == draws and np.isfinite(difference)),
            "method": "paired group percentile bootstrap; candidate minus label-token"}


def gate(estimate, margin, higher_is_better=False, strict=False):
    """Noninferiority uses the adverse CI endpoint, never just a point estimate."""
    if not estimate["supported"]:
        status = "inconclusive"
    elif higher_is_better:
        status = ("pass" if estimate["low"] >= -margin else
                  "fail" if estimate["high"] < -margin else "inconclusive")
    else:
        passed = estimate["high"] < margin if strict else estimate["high"] <= margin
        failed = estimate["low"] >= margin if strict else estimate["low"] > margin
        status = "pass" if passed else "fail" if failed else "inconclusive"
    return {"status": status, "margin": margin, "higher_is_better": higher_is_better,
            "strict": strict, "interval": estimate}


def _aggregate(values, evidence, indices, inverse, groups, weights):
    """Group sufficient statistics avoid a draws-by-records allocation."""
    columns, names = [], []

    def add(name, value):
        names.append(name)
        columns.append(np.bincount(inverse, weights=value, minlength=groups))

    add("records", np.ones(len(indices)))
    add("weight", weights)
    for name, value in values.items():
        if name == "confidence":
            continue
        selected = value[indices]
        valid = np.isfinite(selected)
        add(name + "/sum", np.where(valid, selected, 0) * weights)
        add(name + "/count", valid.astype(float))
        add(name + "/weight", valid * weights)
    confidence = values["confidence"][indices]
    accuracy = values["expected_accuracy"][indices]
    bins = np.minimum((confidence * BINS).astype(int), BINS - 1)
    for index in range(BINS):
        selected = bins == index
        add(f"bin/{index}/count", selected * weights)
        add(f"bin/{index}/confidence", np.where(selected, confidence, 0) * weights)
        add(f"bin/{index}/accuracy", np.where(selected, accuracy, 0) * weights)
    if evidence is not None:
        p, gold = (value[indices] for value in evidence)
        valid = np.isfinite(gold) & (gold != .5)
        predicted, positive = p >= .5, gold > .5
        for name, selected in (("tp", valid & predicted & positive),
                               ("fp", valid & predicted & ~positive),
                               ("fn", valid & ~predicted & positive)):
            add("evidence_" + name, selected.astype(float))
            add("evidence_" + name + "_weight", selected * weights)
        add("evidence_eligible_records", valid.astype(float))
        add("evidence_eligible_weight", valid * weights)
    return names, np.stack(columns, axis=1)


def _summarize(names, totals):
    sums = dict(zip(names, totals))
    result = {name[:-4]: sums[name] / sums[name[:-4] + "/weight"]
              if sums[name[:-4] + "/weight"] else np.nan
              for name in names if name.endswith("/sum")}
    result["ece"] = (sum(abs(sums[f"bin/{i}/confidence"] - sums[f"bin/{i}/accuracy"])
                         for i in range(BINS)) / sums["weight"] if sums["weight"] else np.nan)
    if "evidence_tp" in sums:
        tp, fp, fn = (sums[f"evidence_{key}_weight"] for key in ("tp", "fp", "fn"))
        # Match metrics.evidence_metrics: zero denominator has score zero;
        # resampling recomputes the ratio from counts, not per-record F1.
        eligible = sums["evidence_eligible_records"]
        result.update({"evidence_precision": tp / (tp + fp) if tp + fp else 0.,
                       "evidence_recall": tp / (tp + fn) if tp + fn else 0.,
                       "evidence_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.})
        if not eligible:
            result.update({key: np.nan for key in result if key.startswith("evidence_")})
    return result


def sampling_summary(weights, probabilities=None):
    """Weight dispersion is not the number of independent sampling units."""
    total, squared = float(weights.sum()), float(np.square(weights).sum())
    result = {"sum_weights": total, "sum_squared_weights": squared,
              "effective_records": total * total / squared,
              "min_weight": float(weights.min()), "max_weight": float(weights.max())}
    if probabilities is not None:
        result.update({"min_inclusion_probability": float(probabilities.min()),
                       "max_inclusion_probability": float(probabilities.max())})
    return result


def compare_tracks(tracks, groups, variants, evidence=None, *, draws=2000, seed=17,
                   weights=None, inclusion_probabilities=None, sampling_strata=None):
    """Compare aligned test rows using one shared group resample for all tracks.

    ``variants`` contains the arrays from metrics.per_record for the three fixed
    methods. Missing ordinal metrics are NaN outside ordinal tracks. Optional
    evidence maps methods to (yes_probability, yes_target), NaN on other tracks.
    Optional weights are fixed inverse inclusion probabilities, not bootstrap
    selection probabilities. Whole groups are resampled jointly across tracks.
    These intervals are not exact multistage/finite-population survey intervals.
    This function does not load features, fit parameters or select configurations.
    """
    tracks, groups = np.asarray(tracks), np.asarray(groups)
    if not 1 <= len(tracks) <= MAX_RECORDS or tracks.shape != groups.shape or tracks.ndim != 1:
        raise ValueError("Expected bounded, aligned one-dimensional tracks and groups")
    weighted = weights is not None
    weights = np.ones(len(tracks)) if weights is None else np.asarray(weights)
    if (weights.shape != tracks.shape or weights.dtype.kind not in "iuf"
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Sampling weights must be aligned, finite positive numbers")
    weights = weights.astype(float)
    # Bound subsequent sufficient statistics and effective sample-size arithmetic.
    with np.errstate(over="ignore"):
        bounded = np.isfinite(weights.sum() ** 2) and np.isfinite(np.square(weights).sum())
    if not bounded:
        raise ValueError("Sampling weights exceed numeric resource bounds")
    if inclusion_probabilities is not None:
        inclusion_probabilities = np.asarray(inclusion_probabilities)
        if (not weighted or inclusion_probabilities.shape != tracks.shape
                or inclusion_probabilities.dtype.kind not in "iuf"
                or not np.isfinite(inclusion_probabilities).all()
                or (inclusion_probabilities <= 0).any() or (inclusion_probabilities > 1).any()
                or not np.allclose(weights * inclusion_probabilities, 1., rtol=1e-9, atol=1e-12)):
            raise ValueError("Weights must match positive inverse inclusion probabilities in (0, 1]")
    if sampling_strata is not None:
        sampling_strata = np.asarray(sampling_strata)
        if sampling_strata.shape != tracks.shape or sampling_strata.dtype.kind not in "US":
            raise ValueError("Sampling strata must be aligned strings")
        if inclusion_probabilities is not None and ((inclusion_probabilities < 1) & (sampling_strata == "")).any():
            raise ValueError("Sampled records require a sampling stratum")
    if set(variants) != set(METHODS) or not 1 <= draws <= 10_000:
        raise ValueError("Use the three frozen methods and 1..10000 bootstrap draws")
    keys = set(variants[METHODS[0]])
    if not {"normalized_nll", "confidence", "expected_accuracy"} <= keys:
        raise ValueError("Missing required per-record metrics")
    for values in variants.values():
        if set(values) != keys or any(np.asarray(v).shape != tracks.shape for v in values.values()):
            raise ValueError("Unaligned method metrics")
        if any(np.isinf(value).any() for value in values.values()):
            raise ValueError("Infinite method metrics")
        if any(not np.isfinite(values[key]).all() for key in
               ("normalized_nll", "confidence", "expected_accuracy")):
            raise ValueError("Nonfinite required metrics")
        if any(((values[key] < 0) | (values[key] > 1)).any() for key in ("confidence", "expected_accuracy")):
            raise ValueError("Invalid confidence/accuracy probabilities")
    if evidence is not None:
        if set(evidence) != set(METHODS):
            raise ValueError("Unaligned evidence methods")
        for p, gold in evidence.values():
            if np.asarray(p).shape != tracks.shape or np.asarray(gold).shape != tracks.shape:
                raise ValueError("Unaligned evidence arrays")
            valid = np.isfinite(gold)
            if (not np.isfinite(p[valid]).all() or (p[valid] < 0).any() or (p[valid] > 1).any()
                    or (gold[valid] < 0).any() or (gold[valid] > 1).any()
                    or not np.array_equal(gold, evidence[METHODS[0]][1], equal_nan=True)):
                raise ValueError("Invalid or unpaired evidence targets")
    unique, group_ids = np.unique(groups, return_inverse=True)
    labels = np.unique(tracks)
    if len(labels) > MAX_TRACKS:
        raise MemoryError("Final evaluation track bound exceeded")
    prepared, result = {}, {}
    for track in labels:
        indices = np.flatnonzero(tracks == track)
        local_groups, inverse = np.unique(group_ids[indices], return_inverse=True)
        aggregates, points, eligible = {}, {}, {}
        for method in METHODS:
            ev = evidence[method] if evidence is not None and np.isfinite(evidence[method][1][indices]).any() else None
            names, matrix = _aggregate(variants[method], ev, indices, inverse, len(local_groups), weights[indices])
            aggregates[method] = (names, matrix)
            points[method] = _summarize(names, matrix.sum(axis=0))
        base_names, base_matrix = aggregates[METHODS[0]]
        for name, column in zip(base_names, base_matrix.T):
            if name.endswith("/count") and not name.startswith("bin/"):
                eligible[name[:-6]] = int((column > 0).sum())
        if "evidence_eligible_records" in base_names:
            count = base_matrix[:, base_names.index("evidence_eligible_records")]
            eligible.update({f"evidence_{name}": int((count > 0).sum()) for name in ("precision", "recall", "f1")})
        eligible["ece"] = len(local_groups)
        prepared[track] = (local_groups, aggregates, points, eligible)
        result[track] = {"records": len(indices), "groups": len(local_groups),
                         "methods": {method: {key: float(value) if np.isfinite(value) else None
                                               for key, value in points[method].items()} for method in METHODS},
                         "eligible_records": {key[:-6]: int(base_matrix[:, i].sum()) for i, key in enumerate(base_names)
                                              if key.endswith("/count") and not key.startswith("bin/")},
                         "eligible_weights": {key[:-7]: float(base_matrix[:, i].sum()) for i, key in enumerate(base_names)
                                              if key.endswith("/weight")},
                         "sampling": sampling_summary(weights[indices], None if inclusion_probabilities is None else
                                                       inclusion_probabilities[indices]),
                         "paired": {method: {} for method in METHODS[1:]}}
        if sampling_strata is not None:
            strata = {}
            for stratum in np.unique(sampling_strata[indices]):
                selected = indices[sampling_strata[indices] == stratum]
                strata[stratum] = {"records": len(selected), "groups": len(np.unique(groups[selected])),
                                   **sampling_summary(weights[selected], None if inclusion_probabilities is None else
                                                      inclusion_probabilities[selected])}
            result[track]["sampling"]["strata"] = strata
        if "evidence_tp" in base_names:
            for method, (names, matrix) in aggregates.items():
                result[track]["methods"][method].update(
                    {key: (float if key.endswith("_weight") else int)(matrix[:, i].sum())
                     for i, key in enumerate(names) if key.startswith("evidence_")})
    samples = {track: {method: {key: np.empty(draws) for key in item[2][method]} for method in METHODS[1:]}
               for track, item in prepared.items()}
    macro_samples = {method: np.empty(draws) for method in METHODS[1:]}
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        multiplicities = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        macro = {method: [] for method in METHODS[1:]}
        for track, (local, aggregates, _, _) in prepared.items():
            scores = {method: _summarize(names, np.einsum("g,gk->k", multiplicities[local], matrix, optimize=False))
                      for method, (names, matrix) in aggregates.items()}
            for method in METHODS[1:]:
                for key in samples[track][method]:
                    samples[track][method][key][draw] = scores[method][key] - scores[METHODS[0]][key]
                macro[method].append(scores[method]["normalized_nll"] - scores[METHODS[0]]["normalized_nll"])
        for method in METHODS[1:]:
            # Never drop a missing track and accidentally change the macro estimand.
            macro_samples[method][draw] = np.mean(macro[method])
    macro_points = {method: float(np.mean([item[2][method]["normalized_nll"] for item in prepared.values()]))
                    for method in METHODS}
    for track, (_, _, points, eligible) in prepared.items():
        for method in METHODS[1:]:
            for key, values in samples[track][method].items():
                result[track]["paired"][method][key] = interval(
                    points[method][key] - points[METHODS[0]][key], values, eligible[key], draws, seed)
    # A singleton track cannot acquire independent evidence from unrelated groups.
    minimum_track_groups = min(len(item[0]) for item in prepared.values())
    macro = {"methods": macro_points, "tracks": len(labels), "groups": len(unique),
             "minimum_track_groups": minimum_track_groups,
             "weighting": ("equal task-track weight; inverse-inclusion-weighted within track" if weighted else
                           "equal task-track weight; record-weighted within track"),
             "paired": {method: interval(macro_points[method] - macro_points[METHODS[0]],
                                          macro_samples[method], len(unique), draws, seed)
                        for method in METHODS[1:]}}
    for value in macro["paired"].values():
        value["supported"] = value["supported"] and minimum_track_groups >= 2
    return {"tracks": result, "macro_normalized_nll": macro,
            "sampling_design": {"estimator": "self-normalized inverse-inclusion weighted ratios" if weighted else
                                "unweighted record means",
                                "resampling": "shared paired whole-group percentile bootstrap with fixed record weights",
                                "variance_scope": "group bootstrap; not exact stratified/multistage survey-design variance",
                                "effective_records": "Kish weight-dispersion ESS, not independent group count"}}
