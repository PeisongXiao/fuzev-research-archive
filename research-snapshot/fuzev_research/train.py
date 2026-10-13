"""Frozen-feature head fitting; development selects, calibration only scales.

Training never opens a backbone or a final-test feature. Every eligible fitting
record is visited each epoch; mixed-stage batches replay text alongside images.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .acquire import write_json
from .features import FeatureStore, collate, file_sha256
from .head import HeadBundle, load_bundle, save_bundle
from .metrics import per_record, summarize
from .paths import project_path
from .state import Store


def membership_identity(rows):
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode() + b"\n")
        count += 1
    return {"records": count, "ordered_sha256": digest.hexdigest()}


def training_code_identity():
    return {name: file_sha256(Path(__file__).with_name(name))
            for name in ("train.py", "features.py", "head.py", "metrics.py")}


def head_output_path(relative, output_directory=None):
    weights = project_path("weights")
    base = weights if output_directory is None else Path(output_directory).expanduser().resolve()
    if not base.is_relative_to(weights):
        raise ValueError("Head outputs must remain under the project weights directory")
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Head output path escapes its run directory")
    return path


def epoch_batches(rows, seed, mixed, batch_size=256):
    rng = random.Random(seed)
    streams = defaultdict(list)
    for row in rows:
        streams[(row["primitive"], row["modality"])].append(row)
    schedules = []
    for primitive in ("choice", "noul", "ordinal"):
        text = streams.get((primitive, "text"), [])
        images = streams.get((primitive, "image"), [])
        if mixed and text and images:
            width = batch_size // 2
            chunks = []
            for source in (text, images):
                parts = [source[index:index + width] for index in range(0, len(source), width)]
                rng.shuffle(parts)
                chunks.append(parts)
            count = max(map(len, chunks))
            schedules.extend((primitive, chunks[0][i % len(chunks[0])] + chunks[1][i % len(chunks[1])])
                             for i in range(count))
        else:
            source = text + images
            schedules.extend((primitive, source[index:index + batch_size])
                             for index in range(0, len(source), batch_size))
    rng.shuffle(schedules)
    for primitive, batch in schedules:
        rng.shuffle(batch)
        yield primitive, batch


@torch.inference_mode()
def validate(bundle, features, rows, temperatures=None, device="cpu"):
    features.require_ready("mixed" if any(row["modality"] == "image" for row in rows) else "text")
    if any(row["role"] != "development" for row in rows):
        raise ValueError("Development selection may only read development records")
    grouped = defaultdict(lambda: defaultdict(list))
    for primitive in ("choice", "noul", "ordinal"):
        selection = [row for row in rows if row["primitive"] == primitive]
        for offset in range(0, len(selection), 256):
            batch_rows = selection[offset:offset + 256]
            batch = collate(features, batch_rows, device)
            logits = bundle(primitive, batch["query"], batch["candidates"], batch["mask"])
            kwargs = {"ordinal_values": batch["ordinal_values"].cpu().numpy()} if primitive == "ordinal" else {}
            values = per_record(logits.cpu().numpy(), batch["targets"].cpu().numpy(),
                                batch["mask"].cpu().numpy(),
                                temperature=(temperatures or {}).get(primitive, 1), **kwargs)
            for index, row in enumerate(batch_rows):
                key = "/".join((row["source"], primitive, row["modality"],
                                row.get("attribute") or row.get("adaptation") or "main"))
                for name, value in values.items():
                    grouped[key][name].append(value[index])
    results = {key: summarize({name: np.asarray(value) for name, value in values.items()})
               for key, values in grouped.items()}
    if not results:
        raise ValueError("No separated development records for selection")
    return {"tracks": results,
            "macro_normalized_nll": sum(result["normalized_nll"] for result in results.values()) / len(results)}


def optimizer_checkpoint(directory, optimizer):
    tensors = {}
    for index, parameter in enumerate(optimizer.param_groups[0]["params"]):
        for name, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                tensors[f"{index}.{name}"] = value.detach().cpu().contiguous()
    if not tensors:
        raise ValueError("Cannot checkpoint a missing optimizer state")
    path = directory / "optimizer.safetensors"
    temporary = path.with_suffix(".tmp")
    save_file(tensors, str(temporary))
    temporary.replace(path)
    write_json(directory / "optimizer.json", {"sha256": file_sha256(path), "keys": sorted(tensors)})


def restore_optimizer(directory, optimizer, device):
    path = directory / "optimizer.safetensors"
    manifest_path = directory / "optimizer.json"
    if not path.exists() or not manifest_path.exists():
        raise ValueError("Missing optimizer checkpoint; refuse to resume with a fresh optimizer")
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(path) != manifest["sha256"]:
        raise ValueError("Optimizer checkpoint checksum mismatch")
    parameters = optimizer.param_groups[0]["params"]
    tensors = load_file(str(path))
    if not tensors or sorted(tensors) != manifest["keys"]:
        raise ValueError("Incomplete optimizer checkpoint")
    states = defaultdict(dict)
    for name, value in tensors.items():
        index, key = name.split(".", 1)
        index = int(index)
        if (not 0 <= index < len(parameters) or key not in {"step", "exp_avg", "exp_avg_sq"}
                or not torch.isfinite(value).all()
                or (key == "step" and (value.numel() != 1 or value.item() <= 0))
                or (key != "step" and value.shape != parameters[index].shape)):
            raise ValueError("Invalid optimizer checkpoint tensor")
        states[index][key] = value if key == "step" else value.to(device)
    if any(set(state) != {"step", "exp_avg", "exp_avg_sq"} for state in states.values()):
        raise ValueError("Incomplete optimizer checkpoint state")
    for index, state in states.items():
        optimizer.state[parameters[index]] = state


def train_trial(features, rows, development, stage, seed, learning_rate, brier_weight,
                initial=None, device="cpu", max_epochs=10, output_directory=None):
    readiness = features.require_ready(stage)
    modality = "text" if stage == "text" else None
    fit_identity, dev_identity = membership_identity(rows), membership_identity(development)
    if (fit_identity != membership_identity(features.iter_records("fit", modality))
            or dev_identity != membership_identity(features.iter_records("development", modality))):
        raise ValueError("Trials must use all applicable fitting and development records")
    if not rows or not development or max_epochs < 1:
        raise ValueError("Need nonempty fitting/development records and positive epochs")
    if {row["group_"] for row in rows} & {row["group_"] for row in development}:
        raise ValueError("Fitting/development group overlap")
    task = "training.text" if stage == "text" else "training.multimodal"
    key = f"seed-{seed}-lr-{learning_rate:g}-brier-{brier_weight:g}"
    directory = head_output_path(f"{stage}/trials/{key}", output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {"feature_identity": features.identity, "stage": stage, "seed": seed,
                  "learning_rate": learning_rate, "brier_weight": brier_weight,
                  "initial_sha256": file_sha256(Path(initial) / "head.safetensors") if initial else None,
                  "initial_metadata": {name: file_sha256(Path(initial) / name)
                                       for name in ("manifest.json", "calibration.json")} if initial else None,
                  "preparation": readiness, "fit_membership": fit_identity,
                  "development_membership": dev_identity,
                  "training_code": training_code_identity(),
                  "schedule": {"batch_size": 256, "mixed": stage == "mixed", "max_epochs": max_epochs,
                               "epoch_seed": "seed+epoch", "patience": 2, "minimum_improvement": 1e-4},
                  "optimizer": {"name": "AdamW", "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8},
                  "execution": {"torch": torch.__version__, "device": str(device), "threads": torch.get_num_threads()}}
    if features.scoped:
        provenance["feature_origins"] = features.producer_origins()
    trial_file = directory / "trial.json"
    if trial_file.exists():
        previous = json.loads(trial_file.read_text())
        if previous["provenance"] != provenance:
            raise ValueError("Trial identity changed; refuse to reuse a stale checkpoint")
        if previous.get("completed"):
            return previous
    else:
        previous = {"provenance": provenance, "epochs": [], "best_loss": None, "stale": 0}
    torch.manual_seed(seed)
    checkpoint = previous.get("latest_checkpoint")
    if previous["epochs"] and not checkpoint:
        raise ValueError("Missing latest training checkpoint for resumed trial")
    bundle = load_bundle(checkpoint)[0] if checkpoint else load_bundle(initial)[0] if initial else HeadBundle()
    bundle.to(device)
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=learning_rate, weight_decay=.01)
    if checkpoint:
        restore_optimizer(Path(checkpoint), optimizer, device)
    for epoch in range(len(previous["epochs"]), max_epochs):
        # A crash can follow the durable epoch commit but precede the stop check.
        if previous["stale"] >= 2:
            break
        bundle.train()
        loss_sum, count, started = 0., 0, time.monotonic()
        for primitive, batch_rows in epoch_batches(rows, seed + epoch, stage == "mixed"):
            batch = collate(features, batch_rows, device)
            logits = bundle(primitive, batch["query"], batch["candidates"], batch["mask"])
            logp = torch.log_softmax(logits, dim=1)
            nll = -(batch["targets"] * logp.masked_fill(~batch["mask"], 0)).sum(dim=1)
            loss = (nll / batch["mask"].sum(dim=1).float().log()).mean()
            loss = loss + brier_weight * ((logp.exp() - batch["targets"]) ** 2).sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite head-fitting loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item() * len(batch_rows)
            count += len(batch_rows)
        bundle.eval()
        validation = validate(bundle, features, development, device=device)
        current = validation["macro_normalized_nll"]
        result = {"stage": stage, "trial": key, "epoch": epoch + 1,
                  "fit_loss": loss_sum / count, "fit_visits": count,
                  "seconds": time.monotonic() - started, "development": validation}
        improved = previous["best_loss"] is None or current < previous["best_loss"] - 1e-4
        checkpoint = directory / "checkpoints" / f"epoch-{epoch + 1:03d}"
        save_bundle(checkpoint, bundle, {**provenance, "artifact_kind": "training-checkpoint"},
                    {name: 1. for name in bundle.heads})
        optimizer_checkpoint(checkpoint, optimizer)
        if improved:
            previous["best_loss"], previous["stale"] = current, 0
            previous["best_checkpoint"] = str(checkpoint)
        else:
            previous["stale"] += 1
        previous["epochs"].append(result)
        previous["latest_checkpoint"] = str(checkpoint)
        write_json(trial_file, previous)
        Store().record_result(task, result)
        print(json.dumps(result, allow_nan=False), flush=True)
        if previous["stale"] >= 2:
            break
    previous["completed"] = True
    previous["directory"] = previous["best_checkpoint"]
    write_json(trial_file, previous)
    return previous


def fit(features, stage, initial=None, device="cpu", output_directory=None):
    if stage not in {"text", "mixed"}:
        raise ValueError("Fitting requires a text or mixed stage")
    features.require_ready(stage)
    task = "training.text" if stage == "text" else "training.multimodal"
    rows = features.records("fit", "text" if stage == "text" else None)
    development = features.records("development", "text" if stage == "text" else None)
    if not rows or not development:
        raise ValueError("Need disjoint fitting and development features")
    if {row["primitive"] for row in rows} != {"choice", "noul", "ordinal"}:
        raise ValueError("Reference fitting requires eligible Choice, Noul and ordinal tracks")
    if {row["group_"] for row in rows} & {row["group_"] for row in development}:
        raise ValueError("Fitting/development group overlap")
    if stage == "mixed" and (not initial or not any(row["modality"] == "image" for row in rows)):
        raise ValueError("Mixed stage requires text-warmup initialization and real image features")
    store, trials = Store(), []
    for seed in (17, 29, 43):
        for learning_rate in (3e-4, 1e-3):
            for brier in (0., .1):
                store.update(task, "in_progress", f"Fitting {stage} heads: seed={seed}, lr={learning_rate}, Brier={brier}", len(trials) / 12)
                trials.append(train_trial(features, rows, development, stage, seed, learning_rate,
                                          brier, initial, device, output_directory=output_directory))
    best = min(trials, key=lambda trial: trial["best_loss"])
    features.require_ready(stage)
    selected = {"stage": stage, "selection_role": "development", "selected": best,
                "trials": [{"directory": value["directory"], "best_loss": value["best_loss"],
                            "provenance": value["provenance"]} for value in trials]}
    write_json(head_output_path(f"{stage}/selection.json", output_directory), selected)
    store.record_result(task, selected)
    store.update(task, "completed", "All 12 fitting profiles completed; selected by separated development macro normalized NLL.")
    return best["directory"]


def fit_temperature(logits, target, mask):
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")
    # Mask before dividing: differentiating -inf / temperature and masking
    # afterward can still produce 0 * inf = NaN for padded candidates.
    finite = logits.masked_fill(~mask, 0)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(.03), math.log(100)).exp()
        scaled = (finite / temperature).masked_fill(~mask, -1e9)
        loss = -(target * torch.log_softmax(scaled, dim=1)).sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite temperature-fitting loss")
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().clamp(math.log(.03), math.log(100)).exp().item()


def calibrate(features, directory, device="cpu", output_directory=None):
    readiness = features.require_ready("calibrate")
    bundle, manifest, _ = load_bundle(directory)
    if (manifest.get("feature_identity") != features.identity
            or manifest.get("preparation") != readiness
            or manifest.get("stage") != "mixed"
            or features.scoped and manifest.get("feature_origins") != features.producer_origins()):
        raise ValueError("Calibration requires a matching full-corpus mixed checkpoint")
    bundle.to(device)
    rows = features.records("calibration")
    if not rows:
        raise ValueError("Calibration-only features are required")
    store = Store()
    store.update("training.calibration", "in_progress", "Fitting positive temperatures on calibration groups only; rankings stay unchanged.")
    temperatures = {}
    for primitive in bundle.heads:
        selected = [row for row in rows if row["primitive"] == primitive]
        if not selected:
            raise ValueError(f"Missing calibration track: {primitive}")
        logits, targets = [], []
        with torch.inference_mode():
            for offset in range(0, len(selected), 256):
                batch = collate(features, selected[offset:offset + 256], device)
                value = bundle(primitive, batch["query"], batch["candidates"], batch["mask"]).cpu()
                for index in range(len(value)):
                    mask = batch["mask"][index].cpu()
                    logits.append(value[index, mask].clone())
                    targets.append(batch["targets"][index, mask].cpu().clone())
        # Pad after gathering: candidate count may differ between feature shards.
        width = max(map(len, logits))
        padded, target = torch.full((len(logits), width), -torch.inf), torch.zeros(len(logits), width)
        mask = torch.zeros(len(logits), width, dtype=torch.bool)
        for index, (value, truth) in enumerate(zip(logits, targets)):
            padded[index, :len(value)], target[index, :len(value)], mask[index, :len(value)] = value, truth, True
        temperatures[primitive] = fit_temperature(padded, target, mask)
    output = head_output_path("reference", output_directory)
    features.require_ready("calibrate")
    save_bundle(output, bundle, {**manifest, "artifact_kind": "calibrated-reference-head",
                                "calibration_membership": membership_identity(rows),
                                "calibration_code": training_code_identity(),
                                "calibration_role": "calibration"}, temperatures)
    store.record_result("training.calibration", {"temperatures": temperatures, "artifact": str(output)})
    store.update("training.calibration", "completed", "Frozen FP32 reference heads and positive temperatures saved locally; no test-based selection.")
    return str(output)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--stage", choices=["text", "mixed", "calibrate"], required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-directory", type=Path,
                        help="Isolated run directory under project weights; defaults to weights")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable; no CPU fallback")
    head_output_path("reference", args.output_directory)
    torch.set_num_threads(args.threads)
    features = FeatureStore(args.features, json.loads((args.features / "identity.json").read_text()), read_only=True)
    try:
        if args.stage == "calibrate":
            if not args.initial:
                parser.error("--initial is required for calibration")
            print(calibrate(features, args.initial, args.device, args.output_directory), flush=True)
        else:
            print(fit(features, args.stage, args.initial, args.device, args.output_directory), flush=True)
    finally:
        features.close()


if __name__ == "__main__":
    main()
