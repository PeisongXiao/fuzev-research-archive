"""Offline FIT/DEV residual selection and frozen image-group reporting."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from fuzev_research.metrics import paired_group_interval, per_record, summarize


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align(records, scores, role):
    """Join native diagnostic logits to gold, retaining presented candidate order."""
    record_ids = [row["id"] for row in records]
    score_ids = [row["id"] for row in scores]
    if (not records or len(set(record_ids)) != len(records)
            or len(set(score_ids)) != len(scores) or set(record_ids) != set(score_ids)):
        raise ValueError("Expected exactly one score per prepared record, with matching IDs")
    by_id = {row["id"]: row for row in scores}
    grouped, images, joined = {}, {}, {}
    for row in records:
        group, image = row["group"], row["source_image_id"]
        if row["role"] != role or row["order"] not in ("ab", "ba"):
            raise ValueError(f"Unexpected role/order for {row['id']}")
        if not isinstance(group, str) or not group or not isinstance(image, str) or not image:
            raise ValueError("Expected nonempty image/group IDs")
        if images.setdefault(image, group) != group:
            raise ValueError("One source image occurs in multiple groups")
        grouped.setdefault(group, []).append(row)
        ids = [candidate["id"] for candidate in row["candidates"]]
        target = np.asarray(row["target"], float)
        if len(ids) != 2 or len(set(ids)) != 2 or target.shape != (2,) or sorted(target) != [0., 1.]:
            raise ValueError("Expected two distinct candidates and one-hot gold")
        score = by_id[row["id"]]
        if score["candidate_ids"] != ids or score.get("question", row["question"]) != row["question"]:
            raise ValueError(f"Request order/question differs for {row['id']}")
        logits = []
        for name in ("baseline", "head"):
            entries = score[name]["scores"]
            indexed = {entry["id"]: entry["logit"] for entry in entries}
            if len(entries) != 2 or set(indexed) != set(ids):
                raise ValueError(f"Candidate score IDs differ for {row['id']}: {name}")
            values = np.asarray([indexed[candidate_id] for candidate_id in ids], float)
            if not np.isfinite(values).all():
                raise ValueError("Expected finite raw logits")
            logits.append(values)
        joined[row["id"]] = (*logits, target, ids)

    ordered = []
    for group in sorted(grouped):
        pair = sorted(grouped[group], key=lambda row: row["order"])
        if len(pair) != 2 or [row["order"] for row in pair] != ["ab", "ba"]:
            raise ValueError(f"Expected one AB/BA pair per image: {group}")
        a, b = pair
        shared = ("source_image_id", "source_example_id", "question", "context", "image_path")
        if (any(a.get(key) != b.get(key) for key in shared)
                or a["candidates"] != b["candidates"][::-1] or a["target"] != b["target"][::-1]):
            raise ValueError(f"AB/BA image, captions or gold differ: {group}")
        ordered.extend(joined[row["id"]] for row in pair)
    baseline, v11_head, target, candidate_ids = zip(*ordered)
    return {"baseline": np.asarray(baseline), "v1.1_head": np.asarray(v11_head),
            "target": np.asarray(target), "candidate_ids": np.asarray(candidate_ids),
            "groups": sorted(grouped), "image_ids": sorted(images), "role": role}


def blend(data, strength):
    if strength == 0.:
        return data["baseline"]
    if strength == 1.:
        return data["v1.1_head"]
    return data["baseline"] + strength * (data["v1.1_head"] - data["baseline"])


def losses(logits, target, temperature=1.):
    centered = (logits - logits.max(axis=1, keepdims=True)) / temperature
    return (logsumexp(centered, axis=1) - (target * centered).sum(axis=1)) / np.log(2.)


def bounded_minimum(objective, low, high):
    result = minimize_scalar(objective, bounds=(low, high), method="bounded")
    if not result.success:
        raise ValueError(f"Scalar fit failed: {result.message}")
    return min((low, float(result.x), high), key=lambda value: (objective(value), value))


def fit_candidates(fit, temperature_bounds):
    low, high = temperature_bounds
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("Expected finite 0 < temperature-min < temperature-max")
    strength = bounded_minimum(lambda value: losses(blend(fit, value), fit["target"]).mean(), 0., 1.)
    candidates = []
    for name, value in (("baseline", 0.), ("v1.1_head", 1.), ("fit_residual", strength)):
        logits = blend(fit, value)
        # NLL is convex in inverse temperature; include both bounded endpoints.
        inverse = bounded_minimum(lambda beta: losses(logits, fit["target"], 1. / beta).mean(),
                                  1. / high, 1. / low)
        temperature = 1. / inverse
        candidates.append({"name": name, "lambda": value, "temperature": temperature,
                           "fit_normalized_nll": float(losses(logits, fit["target"], temperature).mean())})
    return candidates


def check_disjoint(left, right):
    if set(left["groups"]) & set(right["groups"]) or set(left["image_ids"]) & set(right["image_ids"]):
        raise ValueError("Image groups overlap between roles")


def interval(values, groups, draws, seed, reference=None):
    reference = np.zeros(len(values)) if reference is None else reference
    result = paired_group_interval(reference, values, groups, draws=draws, seed=seed)
    result["mean"] = result.pop("difference")
    result["method"] = "paired image-unit bootstrap; one observation per image"
    return result


def report(data, selected, draws, seed):
    strength, temperature = selected["lambda"], selected["temperature"]
    if (not np.isfinite([strength, temperature]).all() or not 0 <= strength <= 1
            or temperature <= 0 or draws < 1):
        raise ValueError("Expected lambda in [0,1], positive temperature and bootstrap draws")
    groups, target = data["groups"], data["target"]
    baseline_correct = data["baseline"].argmax(axis=1) == target.argmax(axis=1)
    image_mean = lambda values: np.asarray(values, float).reshape(-1, 2).mean(axis=1)
    result = {"role": data["role"], "records": len(target), "image_groups": len(groups)}
    for name, logits, temp in (("baseline", data["baseline"], 1.),
                               ("v1.1_head", data["v1.1_head"], 1.),
                               ("selected", blend(data, strength), temperature)):
        values = per_record(logits, target, temperature=temp)
        values["normalized_nll"] = losses(logits, target, temp)
        values["nll"] = values["normalized_nll"] * np.log(2.)
        correct = values["accuracy"].astype(bool)
        winners = data["candidate_ids"][np.arange(len(target)), logits.argmax(axis=1)]
        agreement = winners[::2] == winners[1::2]
        summary = {"temperature": temp, "metrics": summarize(values)}
        summary["image_intervals"] = {
            key: interval(image_mean(values[key]), groups, draws, seed)
            for key in ("accuracy", "normalized_nll", "brier")}
        summary["order_agreement"] = {"count": int(agreement.sum()),
                                       **interval(agreement, groups, draws, seed)}
        summary["vs_baseline"] = {
            "accuracy_delta": interval(image_mean(correct), groups, draws, seed,
                                       reference=image_mean(baseline_correct))}
        for event, flags in (("harmful_flips", baseline_correct & ~correct),
                             ("repairs", ~baseline_correct & correct)):
            summary["vs_baseline"][event] = {"count": int(flags.sum()),
                                            **interval(image_mean(flags), groups, draws, seed)}
        result[name] = summary
    result["selected"].update({"lambda": strength, "name": selected["name"]})
    return result


def select(fit, development, temperature_bounds, draws, seed):
    if fit["role"] != "fit" or development["role"] != "development":
        raise ValueError("Selection requires FIT and DEV roles")
    check_disjoint(fit, development)
    candidates = fit_candidates(fit, temperature_bounds)
    for candidate in candidates:
        candidate["dev_normalized_nll"] = float(losses(
            blend(development, candidate["lambda"]), development["target"], candidate["temperature"]).mean())
    chosen = min(candidates, key=lambda item: (item["dev_normalized_nll"], item["lambda"]))
    selected = {key: chosen[key] for key in ("name", "lambda", "temperature")}
    return {"selected": selected, "candidates": candidates,
            "selection_rule": "minimum DEV normalized NLL; exact ties prefer smaller lambda",
            "temperature_bounds": list(temperature_bounds),
            "used_images": {key: fit[key] + development[key] for key in ("groups", "image_ids")},
            "fit": report(fit, selected, draws, seed),
            "development": report(development, selected, draws, seed)}


def evaluate(data, selection, draws, seed):
    if data["role"] != "test":
        raise ValueError("Frozen reporting requires EVAL (test) role")
    check_disjoint(data, selection["used_images"])
    return report(data, selection["selected"], draws, seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select", help="Fit on FIT; select on DEV")
    for role in ("fit", "dev"):
        selection.add_argument(f"--{role}-records", type=Path, required=True)
        selection.add_argument(f"--{role}-scores", type=Path, required=True)
    selection.add_argument("--temperature-min", type=float, required=True)
    selection.add_argument("--temperature-max", type=float, required=True)
    evaluation = commands.add_parser("report", help="Apply frozen selection to EVAL scores")
    evaluation.add_argument("--eval-records", type=Path, required=True)
    evaluation.add_argument("--eval-scores", type=Path, required=True)
    evaluation.add_argument("--selection", type=Path, required=True)
    for command in (selection, evaluation):
        command.add_argument("--bootstrap-draws", type=int, required=True)
        command.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 1 or args.seed < 0:
        parser.error("bootstrap-draws must be positive and seed nonnegative")
    if args.command == "select":
        fit = align(read_jsonl(args.fit_records), read_jsonl(args.fit_scores), "fit")
        development = align(read_jsonl(args.dev_records), read_jsonl(args.dev_scores), "development")
        result = select(fit, development, (args.temperature_min, args.temperature_max),
                        args.bootstrap_draws, args.seed)
    else:
        with args.selection.open() as handle:
            frozen = json.load(handle)
        data = align(read_jsonl(args.eval_records), read_jsonl(args.eval_scores), "test")
        result = evaluate(data, frozen, args.bootstrap_draws, args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
