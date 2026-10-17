"""Train, calibrate or evaluate a bundle-shaped head from ragged frozen features."""

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from fuzev._scoring import SharedDecisionHead
from fuzev.encoding import LABEL_CAPACITY
from fuzev.head import calibration_temperature, read_artifacts, read_v2, V2_SCHEMA
from fuzev_research.metrics import per_record, summarize, paired_group_interval
from ragged_features import GEOMETRY, RaggedFeatures, batches, collate, json_rows
from train_shared_head import initialize, positive_int


DEFAULT_BUCKETS = ((2, 16), (17, 26), (27, 64), (65, 128), (129, 255))
TEMPERATURE_BOUNDS = (.03, 100.)
# Row block size for the checkpointed pairwise loss.
PAIR_BLOCK = 64


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def target_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != V2_SCHEMA:
        raise ValueError("Expected a product v2 target config")
    read_v2(config, {"strata": []})
    return config


def load_model(bundle, config, mode, seed):
    geometry = {key: config[key] for key in GEOMETRY}
    if mode == "scratch":
        return initialize(geometry, seed)
    previous, state, _ = read_artifacts(bundle)
    if any(previous[key] != value for key, value in geometry.items()):
        raise ValueError("Target geometry differs from the input bundle")
    if mode != "existing":
        raise ValueError("Initialization must be existing or scratch")
    model = SharedDecisionHead(**geometry).float()
    model.load_state_dict(state, strict=True)
    return model


def load_rows(paths, config, role, lane):
    result = []
    for path in paths:
        store = RaggedFeatures(path, config)
        if any(row["role"] != role for row in store.rows):
            raise ValueError(f"Supply a {role}-only feature directory: {path}")
        if lane == "replay" and any(row["modality"] != "text" for row in store.rows):
            raise ValueError("The replay lane contains compatible old text only")
        result.extend(dict(row, store=store, lane=lane) for row in store.rows)
    return result


def check_splits(*collections):
    ids, groups = set(), {}
    for rows in collections:
        for row in rows:
            if row["id"] in ids or groups.get(row["group"], row["role"]) != row["role"]:
                raise ValueError("Duplicate ID or related-group split leakage")
            ids.add(row["id"])
            groups[row["group"]] = row["role"]


def bucket_for(k, buckets):
    matching = [(low, high) for low, high in buckets if low <= k <= high]
    if len(matching) != 1:
        raise ValueError(f"K={k} needs exactly one declared bucket")
    return matching[0]


def validate_buckets(buckets):
    last = 1
    for low, high in sorted(buckets):
        if low <= last or high < low or low <= LABEL_CAPACITY < high:
            raise ValueError("K buckets overlap, are invalid or cross the encoding boundary")
        last = high


def row_masses(rows, buckets, replay_mass=.5):
    """Sum-one lane/track/K/group/view masses, independent of minibatch padding."""
    if not math.isfinite(replay_mass) or not 0 <= replay_mass <= 1 or not rows:
        raise ValueError("Invalid replay mass or empty role")
    lanes = {"replay": replay_mass, "new": 1 - replay_mass}
    if {r["lane"] for r in rows} != {key for key, mass in lanes.items() if mass > 0}:
        raise ValueError("Missing declared replay/new lane; weights are not reassigned")
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        track = (row["primitive"], row["modality"])
        cells[row["lane"]][track][bucket_for(row["k"], buckets)].append(row)
    masses = {}
    for lane, tracks in cells.items():
        for by_bucket in tracks.values():
            for members in by_bucket.values():
                views = Counter((r["source"], r["group"]) for r in members)
                mass = lanes[lane] / len(tracks) / len(by_bucket) / len(views)
                for row in members:
                    masses[row["id"]] = mass / views[row["source"], row["group"]]
    if not math.isclose(sum(masses.values()), 1., rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Row masses do not sum to one")
    return masses


def utilities(model, values, rows, config):
    scores = model(None, values["query"], values["candidates"],
                   mask=values["mask"], baseline=values["baseline"])
    scores = scores.masked_fill(~values["mask"], 0)
    strength = scores.new_tensor([config["residual_strength"][r["modality"]] for r in rows])[:, None]
    base = values["baseline"]
    mixed = torch.where(strength == 0, base,
                        torch.where(strength == 1, scores, base + strength * (scores - base)))
    return mixed.masked_fill(~values["mask"], -torch.inf)


def loss_components(scores, values, rows, pair_block=PAIR_BLOCK):
    logp = scores.log_softmax(-1).masked_fill(~values["mask"], 0)
    ce = -(values["targets"] * logp).sum(-1) / values["mask"].sum(-1).float().log()
    bt = []
    for index, row in enumerate(rows):
        z = scores[index, :len(row["target"])]
        total, denominator = z.sum() * 0, z.new_zeros(())
        if "grades" in row:
            grades = z.new_tensor(row["grades"])

            def block_loss(left, right, left_grades, right_grades):
                gap = (left_grades[:, None] - right_grades[None, :]).clamp_min(0)
                return (gap * torch.nn.functional.softplus(right[None, :] - left[:, None])).sum()

            for start in range(0, len(z), pair_block):
                args = (z[start:start + pair_block], z, grades[start:start + pair_block], grades)
                # Recompute pair matrices in backward instead of retaining every block.
                total = total + (checkpoint(block_loss, *args, use_reentrant=False, preserve_rng_state=False)
                                 if z.requires_grad else block_loss(*args))
                denominator = denominator + (args[2][:, None] - grades[None, :]).clamp_min(0).sum()
        bt.append(total / torch.where(denominator > 0, denominator, denominator.new_ones(())))
    return ce, torch.stack(bt)


def score_rows(model, rows, config, path, *, max_elements, batch_size, device, top_k=None):
    model.eval()
    with Path(path).open("x") as handle, torch.no_grad():
        for refs in batches(rows, max_elements=max_elements, max_rows=batch_size,
                            hidden_size=config["hidden_size"]):
            metadata, values = collate(refs, device)
            scores, baselines = torch.stack((utilities(model, values, metadata, config), values["baseline"])).cpu()
            for index, (ref, row) in enumerate(zip(refs, metadata)):
                result = {key: row[key] for key in ("id", "group", "source", "role", "primitive", "modality", "target")}
                result.update(lane=ref["lane"], candidate_ids=[c["id"] for c in row["request"]["candidates"]],
                              head={"scores": scores[index, :ref["k"]].tolist(), "score_type": "logit"},
                              baseline={"scores": baselines[index, :ref["k"]].tolist(),
                                        "score_type": "logit"},
                              baseline_kind="query-label-logits" if ref["k"] <= LABEL_CAPACITY else "zero")
                result.update({key: row[key] for key in ("grades", "ordinal_values", "acceptable_candidate_ids", "execution")
                               if key in row})
                if top_k is not None:
                    for method in ("head", "baseline"):
                        order = np.argsort(-np.array(result[method]["scores"]), kind="stable")[:top_k]
                        result[method]["top_ids"] = [result["candidate_ids"][i] for i in order]
                        result[method]["selected_id"] = result[method]["top_ids"][0]
                handle.write(json.dumps(result, allow_nan=False) + "\n")


def stratum_matches(row, stratum):
    return (row["primitive"] == stratum["primitive"] and row["modality"] == stratum["modality"]
            and stratum["k_min"] <= len(row["target"]) <= stratum["k_max"])


def calibration_rows(path):
    rows = []
    for row in json_rows(path):
        z, y = row["head"]["scores"], row["target"]
        if (len(z) != len(y) or len(z) < 2 or not all(math.isfinite(v) for v in z + y)
                or min(y) < 0 or not math.isclose(sum(y), 1., rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("Invalid calibration utility/target row")
        rows.append(row)
    return rows


def fit_temperature(path, stratum, *, batch_size=32, steps=50):
    source = calibration_rows(path) if isinstance(path, (str, Path)) else path
    rows = [row for row in source if stratum_matches(row, stratum)]
    groups, counts = Counter(), Counter()
    by_k = defaultdict(list)
    for row in rows:
        groups[row["group"]] += 1
        counts[len(row["target"])] += 1
        by_k[len(row["target"])].append(row)
    if not groups:
        raise ValueError(f"Missing CAL records for stratum {stratum}")
    tensors = [(torch.tensor([row["head"]["scores"] for row in members], dtype=torch.float64),
                torch.tensor([row["target"] for row in members], dtype=torch.float64),
                torch.tensor([1 / (len(groups) * groups[row["group"]]) for row in members], dtype=torch.float64))
               for members in by_k.values()]
    log_bounds = tuple(math.log(bound) for bound in TEMPERATURE_BOUNDS)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(*log_bounds).exp()
        total = sum((-(y * (z / temperature).log_softmax(-1)).sum(-1)
                     * weights / math.log(z.shape[-1])).sum() for z, y, weights in tensors)
        total.backward()
        return total

    optimizer.step(closure)
    log_value = log_temperature.detach().item()
    temperature = math.exp(max(log_bounds[0], min(log_value, log_bounds[1])))
    return temperature, {"records": sum(groups.values()), "groups": len(groups), "k_counts": dict(counts),
                         "bound": "lower" if log_value <= log_bounds[0] else
                                  "upper" if log_value >= log_bounds[1] else None}


def execution_summary(records):
    names = ("submitted_tokens", "forwarded_tokens", "cached_tokens", "completed_root_branches")
    sums, observed, count = Counter(), Counter(), 0
    for record in records:
        execution = record.get("execution")
        if execution is None:
            execution = {}
        if not isinstance(execution, dict):
            raise ValueError("Expected an observed execution object")
        count += 1
        for name in names:
            value = execution.get(name)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError(f"Expected nonnegative integer execution count: {name}")
                sums[name] += value
                observed[name] += 1
    return {name: {"observed_sum": sums[name] if observed[name] else None,
                   "observed_records": observed[name], "missing_records": count - observed[name]}
            for name in names}


def calibration_summary(records, requested):
    counts = Counter("raw" if row["temperature"] is None else "calibrated" for row in records)
    effective = "mixed" if counts["raw"] and counts["calibrated"] else (
        "calibrated" if counts["calibrated"] else "raw")
    return {"requested": requested, "effective": effective,
            "calibrated_records": counts["calibrated"], "raw_records": counts["raw"]}


def report_scores(path, buckets, *, masses=None, strata=(), calibration=None):
    calibration = calibration or ("calibrated" if strata else "raw")
    if calibration not in ("raw", "calibrated"):
        raise ValueError("Calibration must be raw or calibrated")
    cells, orders = defaultdict(list), defaultdict(list)
    objective = 0.
    for row in json_rows(path):
        k = len(row["target"])
        bucket = bucket_for(k, buckets)
        temperature = (calibration_temperature(strata, row["primitive"], modality=row["modality"], count=k)
                       if calibration == "calibrated" else None)
        record = {"group": row["group"], "k": k, "execution": row.get("execution"), "temperature": temperature}
        for method in ("head", "baseline"):
            z = np.array(row[method]["scores"])
            values = per_record(z[None, :], np.array(row["target"])[None, :],
                                temperature=temperature if method == "head" and temperature is not None else 1.,
                                ordinal_values=np.array(row["ordinal_values"])[None, :]
                                if "ordinal_values" in row else None)
            result = {key: float(value[0]) for key, value in values.items()}
            winner = int(z.argmax())
            if "acceptable_candidate_ids" in row:
                result["acceptable_accuracy"] = float(row["candidate_ids"][winner] in row["acceptable_candidate_ids"])
            if "grades" in row:
                grades = np.array(row["grades"])
                gain = np.exp2(grades - grades.max()) - np.exp2(-grades.max())
                discount = 1 / np.log2(np.arange(k) + 2)
                ideal = np.sort(gain)[::-1] @ discount
                result["ndcg"] = float(gain[np.argsort(-z, kind="stable")] @ discount / ideal) if ideal else float("nan")
                pairs, wins = 0, 0.
                for i in range(k):
                    valid = grades[i] > grades
                    pairs += int(valid.sum())
                    wins += float(((z[i] > z) * valid + .5 * (z[i] == z) * valid).sum())
                result["strict_pair_agreement"] = wins / pairs if pairs else float("nan")
            record[method] = result
        record["acceptable_size"] = len(row.get("acceptable_candidate_ids", []))
        record["grade_counts"] = dict(Counter(row.get("grades", [])))
        key = (row["lane"], row["primitive"], row["modality"], *bucket)
        cells[key].append(record)
        orders[(row["group"], tuple(sorted(row["candidate_ids"])))].append(
            row["candidate_ids"][int(np.argmax(row["head"]["scores"]))])
        if masses is not None:
            objective += masses[row["id"]] * record["head"]["normalized_nll"]
    output = []
    for key, records in sorted(cells.items()):
        result = dict(zip(("lane", "primitive", "modality", "k_min", "k_max"), key))
        result.update(records=len(records), groups=len({r["group"] for r in records}),
                      k_counts=dict(Counter(r["k"] for r in records)),
                      acceptable_set_sizes=dict(Counter(r["acceptable_size"] for r in records)))
        grade_counts = Counter()
        for record in records:
            grade_counts.update(record["grade_counts"])
        result["grade_counts"] = dict(grade_counts)
        result["execution"] = execution_summary(records)
        result["calibration_summary"] = calibration_summary(records, calibration)
        for method in ("head", "baseline"):
            names = set().union(*(r[method] for r in records))
            values = {name: np.array([r[method].get(name, np.nan) for r in records]) for name in names}
            result[method] = summarize(values)
            result[method].update({name: None for name in names if not np.isfinite(values[name]).any()})
        result["nll_difference"] = paired_group_interval(
            [r["baseline"]["normalized_nll"] for r in records],
            [r["head"]["normalized_nll"] for r in records], [r["group"] for r in records])
        output.append(result)
    repeated = [values for values in orders.values() if len(values) > 1]
    return {"weighted_normalized_nll": objective if masses is not None else None, "cells": output,
            "same_candidate_views": len(repeated),
            "order_agreement": sum(len(set(v)) == 1 for v in repeated) / len(repeated) if repeated else None,
            "execution": execution_summary(record for records in cells.values() for record in records),
            "calibration_summary": calibration_summary(
                (record for records in cells.values() for record in records), calibration),
            "execution_note": "Native extraction counters: forwarded_tokens are processed, cached_tokens are reused; "
                              "completed_root_branches counts completed native branches. Missing counts stay unavailable.",
            "baseline_note": "K above label capacity uses zero utilities, not a frozen-model label control"}


def export_bundle(output, model, config, strata, *, calibrated):
    config = copy.deepcopy(config)
    config["precision"] = "float32"
    for supported in config["capacities"].values():
        for capacity in supported.values():
            capacity["validated_max"] = None
    read_v2(config, {"strata": strata})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, output / "head.safetensors")
    write_json(output / "config.json", config)
    write_json(output / "calibration.json", {"strata": strata})
    return output


def train(args):
    config = target_config(args.config)
    fit = load_rows(args.fit, config, "fit", "new") + load_rows(args.replay_fit, config, "fit", "replay")
    dev = load_rows(args.development, config, "development", "new") + load_rows(
        args.replay_development, config, "development", "replay")
    check_splits(fit, dev)
    fit_mass = row_masses(fit, args.k_buckets, args.replay_mass)
    dev_mass = row_masses(dev, args.k_buckets, args.replay_mass)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"initialization": args.initialization, "seed": args.seed, "k_buckets": args.k_buckets,
              "replay_mass": args.replay_mass, "pairwise_weight": args.pairwise_weight,
              "fit_row_masses": fit_mass, "development_row_masses": dev_mass,
              "fit_ids": [r["id"] for r in fit], "development_ids": [r["id"] for r in dev],
              "fit_groups": sorted({r["group"] for r in fit}),
              "development_groups": sorted({r["group"] for r in dev}),
              "weight_normalization": "sum(p_i)=1; w_i=N*p_i; loss=sum(w_i*loss_i)/logical_batch_size",
              "selected_epoch": None, "epochs": []}
    write_json(args.output / "training.json", report)
    model = load_model(args.bundle, config, args.initialization, args.seed).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, selected = math.inf, None
    updates = 0
    for epoch in range(args.epochs):
        ordered = list(fit)
        random.Random(args.seed + epoch).shuffle(ordered)
        model.train()
        components = torch.zeros(2, dtype=torch.float64, device=args.device)
        for start in range(0, len(ordered), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logical = sorted(ordered[start:start + args.batch_size], key=lambda ref: ref["k"])
            for refs in batches(logical, max_elements=args.batch_elements, max_rows=args.batch_size,
                                hidden_size=config["hidden_size"]):
                rows, values = collate(refs, args.device)
                ce, bt = loss_components(utilities(model, values, rows, config), values, rows)
                mass = ce.new_tensor([fit_mass[r["id"]] for r in refs])
                objective = ((ce + args.pairwise_weight * bt) * (len(fit) * mass)).sum() / args.batch_size
                objective.backward()
                components += torch.stack(((ce.detach() * mass).sum(), (bt.detach() * mass).sum())).double()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
        totals = components.cpu().tolist()
        if not all(math.isfinite(value) for value in totals):
            raise ValueError("Nonfinite FIT objective at epoch validation")
        scores = args.output / f"development-{epoch + 1:03d}.jsonl"
        score_rows(model, dev, config, scores, max_elements=args.batch_elements,
                   batch_size=args.batch_size, device=args.device)
        metrics = report_scores(scores, args.k_buckets, masses=dev_mass)
        report["epochs"].append({"epoch": epoch + 1, "fit_ce": totals[0], "fit_bt": totals[1],
                                 "fit_visits": len(fit), "updates": updates, "development": metrics})
        if metrics["weighted_normalized_nll"] < best:
            best = metrics["weighted_normalized_nll"]
            selected = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            report["selected_epoch"] = epoch + 1
        write_json(args.output / "training.json", report)
    model.load_state_dict(selected, strict=True)
    model.eval().requires_grad_(False)
    export_bundle(args.output / "head", model, config, [], calibrated=False)
    return {"output": str(args.output), "selected_epoch": report["selected_epoch"], "updates": updates}


def calibrate(args):
    config, _, _ = read_artifacts(args.bundle)
    if config.get("schema") != V2_SCHEMA:
        raise ValueError("Calibration requires the selected v2 bundle")
    selection = json.loads(args.training_report.read_text())
    if selection.get("selected_epoch") is None:
        raise ValueError("Select the head on DEV before CAL")
    rows = load_rows(args.calibration, config, "calibration", "new") + load_rows(
        args.replay_calibration, config, "calibration", "replay")
    check_splits(rows)
    prior_ids = set(selection["fit_ids"] + selection["development_ids"])
    prior_groups = set(selection["fit_groups"] + selection["development_groups"])
    if any(row["id"] in prior_ids or row["group"] in prior_groups for row in rows):
        raise ValueError("CAL overlaps FIT or DEV")
    strata = json.loads(args.strata.read_text())["strata"]
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model(args.bundle, config, "existing", 0).to(args.device).eval().requires_grad_(False)
    path = args.output / "calibration-scores.jsonl"
    score_rows(model, rows, config, path, max_elements=args.batch_elements,
               batch_size=args.batch_size, device=args.device)
    scores = calibration_rows(path)
    fitted, support = [], []
    for stratum in strata:
        temperature, counts = fit_temperature(scores, stratum, steps=args.calibration_steps)
        fitted.append(dict(stratum, temperature=temperature))
        support.append(dict(stratum, **counts, temperature=temperature))
    export_bundle(args.output / "head", model, config, fitted, calibrated=True)
    write_json(args.output / "calibration-report.json", {"strata": support, "temperature_bounds": TEMPERATURE_BOUNDS,
        "selection_epoch": selection["selected_epoch"], "evaluation": "not_run"})
    return {"output": str(args.output), "strata": fitted}


def evaluate(args):
    calibrated = args.calibration == "calibrated"
    config, _, strata = read_artifacts(args.bundle)
    if config.get("schema") != V2_SCHEMA:
        raise ValueError("Evaluation requires a frozen v2 bundle")
    rows = load_rows(args.test, config, "test", "heldout")
    if not rows:
        raise ValueError("Evaluation requires nonempty TEST features")
    check_splits(rows)
    for ref in rows:
        bucket_for(ref["k"], args.k_buckets)
    model = load_model(args.bundle, config, "existing", 0).to(args.device).eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "evaluation-scores.jsonl"
    score_rows(model, rows, config, path, max_elements=args.batch_elements,
               batch_size=args.batch_size, device=args.device, top_k=args.top_k)
    applied = strata if calibrated else []
    report = report_scores(path, args.k_buckets, strata=applied, calibration=args.calibration)
    report.update(bundle=str(args.bundle), calibration=args.calibration, strata=applied,
                  k_buckets=args.k_buckets, top_k=args.top_k, records=len(rows),
                  groups=len({r["group"] for r in rows}), capacities=config["capacities"],
                  residual_strength=config["residual_strength"], scores=str(path))
    write_json(args.output / "evaluation.json", report)
    return {"output": str(args.output), "records": len(rows), "calibration": args.calibration}


def parse_bucket(value):
    try:
        low, high = map(int, value.split(":"))
        return low, high
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected inclusive low:high K interval") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    cal = commands.add_parser("calibrate")
    evaluation = commands.add_parser("evaluate")
    for command in (fit, cal, evaluation):
        command.add_argument("--bundle", type=Path, required=command is not fit)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--batch-size", type=positive_int, default=32)
        command.add_argument("--batch-elements", type=positive_int, required=True)
    fit.add_argument("--config", type=Path, required=True)
    for name in ("fit", "development", "replay-fit", "replay-development"):
        fit.add_argument("--" + name, type=Path, nargs="+", default=[])
    fit.add_argument("--initialization", choices=("existing", "scratch"), required=True)
    fit.add_argument("--epochs", type=positive_int, default=1)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--learning-rate", type=float, default=.001)
    fit.add_argument("--weight-decay", type=float, default=.01)
    fit.add_argument("--pairwise-weight", type=float, default=1.)
    fit.add_argument("--replay-mass", type=float)
    fit.add_argument("--k-buckets", nargs="+", type=parse_bucket, default=DEFAULT_BUCKETS)
    for name in ("calibration", "replay-calibration"):
        cal.add_argument("--" + name, type=Path, nargs="+", default=[])
    for name in ("training-report", "strata"):
        cal.add_argument("--" + name, type=Path, required=True)
    cal.add_argument("--calibration-steps", type=positive_int, default=50)
    evaluation.add_argument("--test", type=Path, nargs="+", required=True)
    evaluation.add_argument("--calibration", choices=("raw", "calibrated"), required=True)
    evaluation.add_argument("--k-buckets", nargs="+", type=parse_bucket, default=DEFAULT_BUCKETS)
    evaluation.add_argument("--top-k", type=positive_int, default=5)
    args = parser.parse_args(argv)
    if args.command in ("train", "evaluate"):
        validate_buckets(args.k_buckets)
    if args.command == "train":
        if args.initialization == "existing" and args.bundle is None:
            parser.error("--bundle is required for existing initialization")
        if args.replay_mass is None:
            args.replay_mass = .5 if args.replay_fit or args.replay_development else 0.
    if args.command == "train" and (not math.isfinite(args.pairwise_weight) or args.pairwise_weight < 0):
        parser.error("pairwise-weight must be finite and nonnegative")
    result = {"train": train, "calibrate": calibrate, "evaluate": evaluate}[args.command](args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
