"""Task metrics and paired, group-respecting uncertainty for frozen heads."""

import numpy as np


def probabilities(logits, temperature=1):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.asarray(logits, dtype=np.float64) / temperature
    if logits.ndim != 2 or np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("Expected finite or masked candidate logits")
    maximum = logits.max(axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("Each decision must have an unmasked candidate")
    exponent = np.exp(logits - maximum)
    return exponent / exponent.sum(axis=1, keepdims=True)


def per_record(logits, targets, mask=None, ordinal_values=None, temperature=1):
    logits, targets = np.asarray(logits, float), np.asarray(targets, float)
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("Logits and probability targets must have the same 2D shape")
    if not np.isfinite(targets).all() or (targets < 0).any() or not np.allclose(targets.sum(axis=1), 1):
        raise ValueError("Invalid candidate target distribution")
    mask = np.ones_like(targets, dtype=bool) if mask is None else np.asarray(mask, bool)
    if mask.shape != targets.shape or (mask.sum(axis=1) < 2).any() or (targets[~mask] != 0).any():
        raise ValueError("Invalid candidate mask")
    logits = np.where(mask, logits, -np.inf)
    p = probabilities(logits, temperature)
    selected = p.argmax(axis=1)
    rows = np.arange(len(p))
    logp = np.log(np.clip(p, 1e-300, 1))
    loss = -(targets * logp).sum(axis=1)
    max_target = targets.max(axis=1, keepdims=True)
    unique_gold = (targets == max_target).sum(axis=1) == 1
    result = {
        "nll": loss, "normalized_nll": loss / np.log(mask.sum(axis=1)),
        "brier": ((p - targets) ** 2).sum(axis=1),
        "expected_accuracy": targets[rows, selected],
        "accuracy": np.where(unique_gold, selected == targets.argmax(axis=1), np.nan),
        "tie_aware_accuracy": targets[rows, selected] == max_target[:, 0],
        "confidence": p[rows, selected],
    }
    if ordinal_values is not None:
        values = np.asarray(ordinal_values, float)
        if values.shape != targets.shape or not np.isfinite(values[mask]).all():
            raise ValueError("Invalid ordinal values")
        values = np.where(mask, values, 0)
        result["ordinal_mae"] = np.abs((p * values).sum(axis=1) - (targets * values).sum(axis=1))
        cumulative = np.cumsum(p - targets, axis=1)
        valid_intervals = np.arange(targets.shape[1])[None, :] < (mask.sum(axis=1) - 1)[:, None]
        result["ranked_probability_score"] = (cumulative ** 2 * valid_intervals).sum(axis=1) / (mask.sum(axis=1) - 1)
    return result


def summarize(values, bins=15):
    result = {name: float(np.nanmean(value)) for name, value in values.items()
              if name != "confidence" and np.isfinite(value).any()}
    confidence, accuracy = values["confidence"], values["expected_accuracy"]
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.
    for index in range(bins):
        members = indices == index
        if members.any():
            ece += members.mean() * abs(confidence[members].mean() - accuracy[members].mean())
    result["ece"] = float(ece)
    result["records"] = len(confidence)
    return result


def paired_group_interval(reference, candidate, groups, draws=2000, seed=17):
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    groups = np.asarray(groups)
    if reference.shape != candidate.shape or reference.shape != groups.shape:
        raise ValueError("Paired comparisons require aligned records and groups")
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if not valid.any():
        raise ValueError("No paired finite observations")
    delta, groups = (candidate - reference)[valid], groups[valid]
    unique, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        selected = rng.integers(0, len(unique), len(unique))
        estimates.append(totals[selected].sum() / counts[selected].sum())
    low, high = np.quantile(estimates, [.025, .975])
    return {"difference": float(delta.mean()), "low": float(low), "high": float(high),
            "groups": len(unique), "draws": draws, "seed": seed,
            "method": "paired group bootstrap; record-weighted mean"}


def evidence_metrics(probabilities_, targets, threshold=.5):
    p, gold = np.asarray(probabilities_, float), np.asarray(targets, float)
    # Exactly split annotator votes do not supply a hard binary gold label.
    eligible = gold != .5
    prediction, positive = p[eligible] >= threshold, gold[eligible] > .5
    tp = int((prediction & positive).sum())
    fp = int((prediction & ~positive).sum())
    fn = int((~prediction & positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "tp": tp, "fp": fp, "fn": fn, "eligible_records": int(eligible.sum())}
