"""Run the frozen Kev partitions once and score them with the published scorer.

The append-only journal records a request before sending it. Resuming the same
output directory skips every attempted record, including transport failures.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from breadth_report import bootstrap_index, record_outcomes, score_system
from kev.benchmark import prediction_rows
from kev.predictors import RemotePredictor
from kev.suite import load_split, read_manifest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_journal(path):
    if not path.exists():
        return []
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def append_event(output, event):
    output.write(json.dumps(event, allow_nan=False) + "\n")
    output.flush()
    os.fsync(output.fileno())


def run_records(records, predictor, directory, *, system, split):
    """Keep one attempt per record across restarts; return saved rows and coverage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "requests.jsonl"
    events = read_journal(journal_path)
    attempted = {event["id"] for event in events if event["event"] == "attempt"}
    interrupted = False
    with journal_path.open("a") as output:
        for record in records:
            rid = record["_meta"]["id"]
            if rid in attempted:
                continue
            event = {"event": "attempt", "id": rid, "system": system,
                     "split": split, "source": record["_meta"]["source"],
                     "questions": list(record["questions"]), "started_unix": time.time()}
            append_event(output, event)
            events.append(event)
            attempted.add(rid)
            try:
                prediction = predictor(record)
                rows = prediction_rows(record, prediction)
            except Exception as error:
                event = {"event": "error", "id": rid,
                         "error_type": type(error).__name__, "error": str(error)}
                append_event(output, event)
                events.append(event)
                interrupted = True
                print(f"{system} {split}: {rid}: {type(error).__name__}: {error}", flush=True)
                break
            event = {"event": "result", "id": rid, "prediction": prediction, "rows": rows}
            append_event(output, event)
            events.append(event)
            if len(attempted) % 50 == 0:
                print(f"{system} {split}: attempted {len(attempted)}/{len(records)}", flush=True)
    results = [event for event in events if event["event"] == "result"]
    errors = [event for event in events if event["event"] == "error"]
    finished = {event["id"] for event in results + errors}
    rows = [row for result in results for row in result["rows"]]
    coverage = {
        "records": len(records), "questions": sum(len(r["questions"]) for r in records),
        "attempted_records": len(attempted), "successful_records": len(results),
        "answered_questions": len(rows), "errors": errors,
        "unresolved_records": sorted(attempted - finished),
        "unattempted_records": [r["_meta"]["id"] for r in records if r["_meta"]["id"] not in attempted],
    }
    write_json(directory / "rows.json", rows)
    write_json(directory / "coverage.json", coverage)
    return rows, coverage, interrupted


def score_results(records, systems, manifest):
    """Reuse the exact dataset metrics and record bootstrap behind v1.0."""
    scores = {}
    for name, rows in systems.items():
        report = score_system(records, rows, manifest)
        keyed = {(r["id"], r["question"]): r for r in rows if r.get("variant", "clean") == "clean"}
        outcomes = record_outcomes(records, keyed, "accuracy")
        correct, questions = int(outcomes[:, 0].sum()), int(outcomes[:, 1].sum())
        report["overall"].update(correct_questions=correct, question_accuracy=correct / questions)
        for source, data in report["datasets"].items():
            own = [r for r in records if r["_meta"]["source"] == source]
            outcomes = record_outcomes(own, keyed, "accuracy")
            n = int(outcomes[:, 1].sum())
            data.update(records=len(own), questions=n,
                        correct_questions=int(outcomes[:, 0].sum()),
                        question_accuracy=float(outcomes[:, 0].sum()) / n)
        scores[name] = report
    uncertainty = bootstrap_index(records, systems, manifest, samples=2000, seed=0)
    rng = np.random.default_rng(uncertainty["seed"])
    names = list(systems)
    keyed = {name: {(row["id"], row["question"]): row for row in rows
                    if row.get("variant", "clean") == "clean"}
             for name, rows in systems.items()}
    differences = {}
    # Match the published index bootstrap's dataset order and whole-record draws.
    for area in manifest["areas"].values():
        for dataset in area["datasets"]:
            own = [record for record in records if record["_meta"]["source"] == dataset]
            draw = rng.integers(0, len(own), size=(uncertainty["samples"], len(own)))
            draws = {}
            for name in names:
                outcomes = record_outcomes(own, keyed[name], manifest["datasets"][dataset]["metric"])[draw]
                draws[name] = outcomes[..., 0].sum(1) / outcomes[..., 1].sum(1)
            differences[dataset] = {name: [float(v) for v in np.quantile(
                100 * (draws[name] - draws[names[0]]), [0.025, 0.975])]
                for name in names[1:]}
    uncertainty["dataset_difference"] = differences
    uncertainty["dataset_difference_unit"] = "dataset score percentage points"
    return {"systems": scores,
            "uncertainty": uncertainty,
            "question_accuracy_policy": "all questions; missing responses count wrong"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="one HTTP attempt per frozen record, resumable")
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--model", required=True)
    run.add_argument("--system", required=True)
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--api-key", default="local")
    score = sub.add_parser("score", help="score saved rows without inference")
    score.add_argument("--result", action="append", required=True, help="NAME=DIRECTORY containing rows.json")
    for command in (run, score):
        command.add_argument("--suite", type=Path, required=True)
        command.add_argument("--split", choices=("development", "test"), required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = load_split(args.suite, args.split, allow_test=args.split == "test")
    manifest = read_manifest(args.suite)
    interrupted = False
    if args.command == "run":
        predictor = RemotePredictor(args.base_url, args.model, args.api_key, timeout=args.timeout, retries=1)
        rows, coverage, interrupted = run_records(records, predictor, args.out,
                                                  system=args.system, split=args.split)
        result = score_results(records, {args.system: rows}, manifest)
        result.update(coverage=coverage,
                      remote={"requested_model": args.model,
                              "served_model": predictor.served_model,
                              "attempts_per_record": 1, "concurrency": 1})
        out = args.out / "score.json"
    else:
        systems = {}
        for spec in args.result:
            name, directory = spec.split("=", 1)
            systems[name] = json.loads((Path(directory) / "rows.json").read_text())
        result = score_results(records, systems, manifest)
        out = args.out
    result.update(suite=manifest["version"], split=args.split,
                  records=len(records), questions=sum(len(r["questions"]) for r in records))
    write_json(out, result)
    for name, system in result["systems"].items():
        overall = system["overall"]
        print(f"{name} {args.split}: {overall['correct_questions']}/{overall['questions']} questions; "
              f"breadth-v1 {overall['index']:.6f}; "
              f"CI {result['uncertainty']['index'][name]}", flush=True)
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
