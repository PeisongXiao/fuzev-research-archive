"""Rescore captured final-token features with raw shared-head utilities."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from train_shared_head import batch, logits, positive_int


def _validate(rows, hidden_size):
    for row in rows:
        count = len(row["candidate_ids"])
        for name in ("target", "baseline"):
            value = row[name]
            if value.shape != (count,) or not value.isfinite().all():
                raise ValueError("Expected finite full-slate target and baseline")
        features = row["views"]["control"]
        for name, shape in (("query", (hidden_size,)), ("candidates", (count, hidden_size))):
            if features[name].shape != shape or not features[name].isfinite().all():
                raise ValueError(f"Invalid final-token {name} features")


@torch.no_grad()
def score_probe(model, rows, view="control", *, batch_size=32):
    device = next(model.parameters()).device
    records = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        values = batch([{name: (row["views"][view][name] if name in ("query", "candidates")
                               else row[name]).detach().float()
                         for name in ("query", "candidates", "baseline", "target")} for row in selected], device)
        scores = logits(model, values).float().cpu()
        for row, padded in zip(selected, scores):
            ids = row["candidate_ids"]
            utilities = padded[:len(ids)]
            logp = utilities.log_softmax(0)
            probabilities = logp.exp()
            order = utilities.argsort(descending=True, stable=True).tolist()
            acceptable = set(row["acceptable_candidate_ids"])
            rank = next(i + 1 for i, index in enumerate(order) if ids[index] in acceptable)
            records.append({"id": row["id"], "group": row["group"], "role": row["role"],
                            "primitive": row["primitive"], "modality": row["modality"], "k": len(ids),
                            "candidate_ids": list(ids), "selected_id": ids[order[0]],
                            "utilities": utilities.tolist(), "probabilities": probabilities.tolist(),
                            "acceptable_accuracy": float(rank == 1), "mrr": 1.0 / rank,
                            "nll": -(row["target"] * logp).sum().item(),
                            "brier": ((probabilities - row["target"]) ** 2).sum().item()})
    return records


def tensorize_capture(capture, records, hidden_size):
    """Join complete prepared requests and their final-token captures by ID."""
    sources = capture["representation_rows"]
    expected = {row["id"]: row for row in records}
    if len(expected) != len(records) or {row["id"] for row in sources} != set(expected) or len(sources) != len(records):
        raise ValueError("Capture must contain every prepared request exactly once")
    rows, groups = [], defaultdict(list)
    for source in sources:
        record = expected[source["id"]]
        ids = [candidate["id"] for candidate in record["candidates"]]
        if (source.get("error") or source.get("failure")
                or any(source[name] != record[name] for name in ("group", "role", "order", "primitive", "target"))
                or source["candidate_ids"] != ids):
            raise ValueError(f"Failed capture or changed prepared request: {source['id']}")
        row = dict(source)
        for name in ("target", "baseline"):
            row[name] = torch.tensor(source[name], dtype=torch.float32)
        row["views"] = {"control": {name: torch.tensor(source["views"]["control"][name], dtype=torch.float32)
                                    for name in ("query", "candidates")}}
        rows.append(row)
        groups[row["group"]].append(row)
    _validate(rows, hidden_size)
    for members in groups.values():
        pair = sorted(members, key=lambda row: row["order"])
        if (len(pair) != 2 or [row["order"] for row in pair] != ["ab", "ba"]
                or pair[0]["candidate_ids"] != pair[1]["candidate_ids"][::-1]
                or not torch.equal(pair[0]["target"], pair[1]["target"].flip(0))):
            raise ValueError("Expected complete AB/BA image groups")
    return rows


def aligned_score_rows(rows, scores):
    """Join raw utilities by request/candidate IDs into residual_blend JSONL rows."""
    indexed = {record["id"]: record for record in scores}
    if len(indexed) != len(scores) or set(indexed) != {row["id"] for row in rows}:
        raise ValueError("Expected exactly one raw score per captured request")
    result = []
    for row in rows:
        record, ids = indexed[row["id"]], row["candidate_ids"]
        utilities = dict(zip(record["candidate_ids"], record["utilities"]))
        if (len(utilities) != len(record["candidate_ids"]) or set(utilities) != set(ids)
                or len(record["utilities"]) != len(ids)
                or not all(math.isfinite(value) for value in utilities.values())):
            raise ValueError(f"Expected finite, complete candidate utilities: {row['id']}")
        result.append({**{name: row[name] for name in
                         ("id", "group", "role", "order", "primitive", "modality", "candidate_ids")},
                       "baseline": {"scores": [{"id": identifier, "logit": value}
                                               for identifier, value in zip(ids, row["baseline"].tolist())]},
                       "head": {"scores": [{"id": identifier, "logit": utilities[identifier]}
                                           for identifier in ids]}})
    return result


def main(argv=None):
    from train_shared_head_v2 import load_model, target_config

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("rescore")
    for field in ("capture", "head", "output"):
        command.add_argument(f"--{field}", type=Path, required=True)
    for field in ("fit-records", "dev-records", "eval-records"):
        command.add_argument(f"--{field}", type=Path)
    command.add_argument("--device", default="cpu")
    command.add_argument("--score-batch-size", type=positive_int, default=32)
    args = parser.parse_args(argv)
    if (args.eval_records is not None and (args.fit_records is not None or args.dev_records is not None)
            or args.eval_records is None and (args.fit_records is None or args.dev_records is None)):
        parser.error("Rescore complete FIT/DEV inputs or --eval-records alone")
    config = target_config(args.head / "config.json")
    paths = (("test", args.eval_records),) if args.eval_records is not None else (
        ("fit", args.fit_records), ("development", args.dev_records))
    records = []
    for role, path in paths:
        with path.open() as handle:
            prepared = [json.loads(line) for line in handle if line.strip()]
        if not prepared or any(row["role"] != role for row in prepared):
            raise ValueError(f"Expected a complete {role}-only prepared input")
        records.extend(prepared)
    capture = json.loads(args.capture.read_text())
    rows = tensorize_capture(capture, records, config["hidden_size"])
    diagnostic = aligned_score_rows(rows, [
        {"id": row["id"], "candidate_ids": [entry["id"] for entry in row["head"]["scores"]],
         "utilities": [entry["logit"] for entry in row["head"]["scores"]]} for row in rows])
    # Score with raw head logits; residual strength and calibration are not applied.
    model = load_model(args.head, config, "existing", 0).float().to(args.device).eval().requires_grad_(False)
    rescored = aligned_score_rows(rows, score_probe(model, rows, batch_size=args.score_batch_size))
    args.output.mkdir(parents=True, exist_ok=False)
    for prefix, values in (("v2", rescored), ("v1", diagnostic)):
        for role, _ in paths:
            with (args.output / f"{prefix}-{role}-scores.jsonl").open("x") as handle:
                for row in values:
                    if row["role"] == role:
                        head_source = str(args.head) if prefix == "v2" else capture.get("head")
                        handle.write(json.dumps(dict(row, capture=str(args.capture), head_source=head_source),
                                                allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
