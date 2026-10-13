"""Train and calibrate a shared head from frozen feature JSONL and a backbone config."""

import argparse
from datetime import date
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from fuzev._scoring import SharedDecisionHead
from fuzev.head import ServingHead


def export_bundle(output, state, metadata, temperatures, encoding):
    config = dict(metadata, **encoding, schema="fuzev-shared-head-package-v1", precision="float32")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().cpu().contiguous() for name, value in state.items()},
              output / "head.safetensors")
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    (output / "calibration.json").write_text(
        json.dumps({"temperatures": temperatures}, indent=2, allow_nan=False) + "\n")
    return output


def read_rows(path, hidden_size, max_options):
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for name in ("query", "candidates", "baseline", "target"):
                row[name] = torch.tensor(row[name], dtype=torch.float32)
            count = row["baseline"].numel()
            if (row["query"].shape != (hidden_size,) or row["candidates"].shape != (count, hidden_size)
                    or row["baseline"].shape != (count,) or row["target"].shape != (count,)
                    or not 2 <= count <= max_options or (row["primitive"] == "noul" and count != 2)):
                raise ValueError(f"{path}: invalid feature geometry or option count for {row['id']}")
            if any(not tensor.isfinite().all() for tensor in
                   (row["query"], row["candidates"], row["baseline"], row["target"])):
                raise ValueError(f"{path}: nonfinite feature or target for {row['id']}")
            if (row["target"] < 0).any() or not torch.isclose(row["target"].sum(), torch.tensor(1.0)):
                raise ValueError(f"{path}: target must be a normalized distribution for {row['id']}")
            rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: expected nonempty rows with unique ids")
    return rows


def batch(rows, device):
    count = max(len(row["baseline"]) for row in rows)
    pad = torch.nn.functional.pad
    return {
        "query": torch.stack([row["query"] for row in rows]).to(device),
        "candidates": torch.stack([pad(row["candidates"], (0, 0, 0, count - len(row["baseline"])))
                                   for row in rows]).to(device),
        "baseline": torch.stack([pad(row["baseline"], (0, count - len(row["baseline"])))
                                 for row in rows]).to(device),
        "target": torch.stack([pad(row["target"], (0, count - len(row["baseline"])))
                               for row in rows]).to(device),
        "mask": torch.arange(count, device=device)[None, :] < torch.tensor(
            [len(row["baseline"]) for row in rows], device=device)[:, None],
    }


def logits(model, values):
    return model(None, values["query"], values["candidates"],
                 mask=values["mask"], baseline=values["baseline"])


def loss(scores, values):
    logp = scores.log_softmax(-1).masked_fill(~values["mask"], 0)
    return (-(values["target"] * logp).sum(-1) / values["mask"].sum(-1).float().log()).mean()


def initialize(geometry, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = SharedDecisionHead(**geometry).float()
        for layer in (model.scorer.query, model.adapter_down, model.key_adapter_down):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(model.scorer.key.weight, gain=1e-3)
        nn.init.zeros_(model.scorer.key.bias)
        for value in (model.adapter_up.weight, model.key_adapter_up.weight,
                      model.gate.weight, model.gate.bias, model.channel_a, model.channel_b, model.channel_c):
            nn.init.zeros_(value)
    return model


def calibrate(model, rows, device, steps):
    values = batch(rows, device)
    with torch.no_grad():
        frozen = logits(model, values).detach().masked_fill(~values["mask"], 0)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        scaled = (frozen / log_temperature.exp()).masked_fill(~values["mask"], -torch.inf)
        value = loss(scaled, values)
        value.backward()
        return value

    optimizer.step(closure)
    temperature = log_temperature.detach().exp().item()
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature fitting did not produce a finite positive scalar")
    return temperature


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.backbone_config.read_text())
    encoding = json.loads(args.backbone_config.with_name("encoding.json").read_text())
    geometry = dict(hidden_size=config.get("text_config", config)["hidden_size"],
                    projection_size=args.projection_size, query_adapter_size=args.query_adapter_size,
                    key_adapter_size=args.key_adapter_size)
    fit = read_rows(args.fit, geometry["hidden_size"], args.max_options)
    cal = read_rows(args.calibration, geometry["hidden_size"], args.max_options)
    dev = read_rows(args.development, geometry["hidden_size"], args.max_options) if args.development else []
    seen_ids, seen_groups = set(), set()
    for rows in (fit, dev, cal):
        ids = {row["id"] for row in rows}
        groups = {row.get("group", row["id"]) for row in rows}
        if ids & seen_ids or groups & seen_groups:
            raise ValueError("FIT, DEV and CAL must have disjoint ids and related-example groups")
        seen_ids.update(ids)
        seen_groups.update(groups)
    primitives = sorted({row["primitive"] for row in fit})
    if set(primitives) != {row["primitive"] for row in cal}:
        raise ValueError("FIT and CAL must cover the same primitives")
    if not {row["primitive"] for row in dev} <= set(primitives):
        raise ValueError("DEV contains primitives absent from FIT")
    model = initialize(geometry, args.seed).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    first_loss, last_loss, updates = None, None, 0
    with torch.autocast(device_type=torch.device(args.device).type, enabled=False):
        for _ in range(args.epochs):
            order = torch.randperm(len(fit), generator=generator).tolist()
            for start in range(0, len(fit), args.batch_size):
                values = batch([fit[i] for i in order[start:start + args.batch_size]], args.device)
                optimizer.zero_grad(set_to_none=True)
                objective = loss(logits(model, values), values)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_loss = objective.detach().item()
                if first_loss is None:
                    first_loss = last_loss
                updates += 1
        model.eval().requires_grad_(False)
        temperatures = {primitive: calibrate(model, [row for row in cal if row["primitive"] == primitive],
                                              args.device, args.calibration_steps)
                        for primitive in primitives}
    metadata = dict(geometry, primitives=primitives, max_options=args.max_options,
                    release_date=args.release_date)
    output = export_bundle(args.output, model.state_dict(), metadata, temperatures, encoding)
    result = dict(output=str(output), fit_rows=len(fit), development_rows=len(dev), calibration_rows=len(cal), updates=updates,
                  first_batch_loss=first_loss, last_batch_loss=last_loss, temperatures=temperatures)
    if args.inspect:
        with ServingHead(output, device=args.device, calibration="calibrated") as head:
            result["examples"] = {}
            inspection_rows = dev or cal
            result["inspection_role"] = "development" if dev else "calibration"
            for primitive in sorted({row["primitive"] for row in inspection_rows}):
                row = next(row for row in inspection_rows if row["primitive"] == primitive)
                scores, raw, calibrated = head.score(primitive, row["query"], row["candidates"], baseline=row["baseline"])
                result["examples"][primitive] = dict(logits=scores, raw=raw, calibrated=calibrated)
    return result


def fixture(output, hidden_size):
    output.mkdir(parents=True, exist_ok=False)
    (output / "backbone-config.json").write_text(json.dumps({"text_config": {"hidden_size": hidden_size}}) + "\n")
    (output / "encoding.json").write_text(json.dumps({"encoding_version": "synthetic-branches-v1"}) + "\n")
    for role, repeats, offset in (("fit", 2, 0), ("calibration", 1, 20)):
        with (output / f"{role}.jsonl").open("w") as handle:
            for repeat in range(repeats):
                for index, (primitive, count) in enumerate((("choice", 3), ("noul", 2), ("ordinal", 4))):
                    step = offset + 3 * repeat + index
                    query = ((torch.arange(hidden_size, dtype=torch.float32) + step) % 17 - 8) / 17
                    candidates = ((torch.arange(count * hidden_size, dtype=torch.float32)
                                   .reshape(count, hidden_size) * 3 + step) % 23 - 11) / 23
                    target = torch.full((count,), 0.2 / count)
                    target[step % count] += 0.8
                    row = dict(id=f"{role}-{primitive}-{repeat}", primitive=primitive,
                               query=query.tolist(), candidates=candidates.tolist(),
                               baseline=(0.3 * candidates[:, 0]).tolist(), target=target.tolist())
                    handle.write(json.dumps(row) + "\n")
    return {"fixture": str(output), "hidden_size": hidden_size, "synthetic": True}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tiny = commands.add_parser("fixture", help="Write synthetic CPU smoke inputs, not real backbone features")
    tiny.add_argument("--output", type=Path, required=True)
    tiny.add_argument("--hidden-size", type=positive_int, required=True)
    fit = commands.add_parser("train", help="Fit, calibrate and export from frozen feature rows")
    for name in ("backbone-config", "fit", "calibration", "output"):
        fit.add_argument("--" + name, type=Path, required=True)
    fit.add_argument("--development", type=Path, help="Separate held-out rows used only by --inspect, never for updates or calibration")
    for name in ("projection-size", "query-adapter-size", "key-adapter-size", "max-options"):
        fit.add_argument("--" + name, type=positive_int, required=True)
    fit.add_argument("--release-date", default=date.today().isoformat())
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--epochs", type=positive_int, default=1)
    fit.add_argument("--batch-size", type=positive_int, default=32)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--calibration-steps", type=positive_int, default=50)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--inspect", action="store_true", help="Reload the export with fuzev-runtime and print one score per primitive")
    args = parser.parse_args(argv)
    result = fixture(args.output, args.hidden_size) if args.command == "fixture" else train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
