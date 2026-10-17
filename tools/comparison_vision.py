"""Measure image decisions over HTTP.

``run`` sends cases to a running FuZev server, ``retry`` re-sends HTTP 413
rows as lossless WebP, ``score`` writes summaries, and ``serve`` starts
fuzev-native with backbone-label capture.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import base64
import mimetypes


def image_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_cases(paths):
    """Keep existing membership, ordering, source records and track names."""
    cases = []
    for name, path in paths.items():
        if name == "eval-mix":
            rows = json.loads(Path(path).read_text())["rows"]
        else:
            rows = [{"row": row} for row in jsonl(path)]
        for position, item in enumerate(rows):
            row = item["row"]
            if name == "eval-mix" and row.get("modality") != "image":
                continue
            track = item.get("task_track") or "/".join((
                row["source"], row["primitive"], row["modality"],
                row.get("attribute") or row.get("adaptation") or "main"))
            cases.append({"set": name, "track": track, "record": deepcopy(row),
                          "source": {"set": name, "position": position},
                          "original_request": deepcopy(item.get("request"))})
    return cases


def request_for(case, model=None, image_root=Path(".")):
    """Use the published image transport and the preparer's empty-image context."""
    row = case["record"]
    request = {"primitive": row["primitive"], "question": row["question"],
               "candidates": deepcopy(row["candidates"]),
               "messages": deepcopy(row["context"]) or [
                   {"role": "user", "content": "Consider the supplied image."}],
               "image": image_url(image_root / row["image_path"])}
    if "additional_context" in row:
        request["additional_context"] = row["additional_context"]
    if model is not None:
        request["model"] = model
    return request


def inventory(cases, image_root=Path(".")):
    sets = defaultdict(lambda: {"requests": 0, "tracks": Counter(), "groups": set(),
                                "missing_images": [], "empty_contexts": 0})
    for case in cases:
        row, entry = case["record"], sets[case["set"]]
        entry["requests"] += 1
        entry["tracks"][case["track"]] += 1
        entry["groups"].add(row["group"])
        entry["empty_contexts"] += not bool(row["context"])
        if not (image_root / row["image_path"]).is_file():
            entry["missing_images"].append({"id": row["id"], "path": Path(row["image_path"]).name})
    return {name: {**entry, "tracks": dict(entry["tracks"]), "groups": len(entry["groups"])}
            for name, entry in sets.items()}


def send_request(row, client):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    response = client.post("/v1/decisions", json=row["request"])
    row["status_code"], row["response_text"] = response.status_code, response.text
    try:
        row["response"] = json.loads(response.text, parse_constant=invalid_constant)
    except ValueError as error:
        row["error"] = f"{type(error).__name__}: {error}"
    if response.status_code != 200:
        row["error"] = f"HTTP {response.status_code}: {response.text}"


def run_cases(cases, client, output, *, arm, model=None, image_root=Path(".")):
    """Write each attempt immediately; a failed request leaves the next one runnable."""
    counts = Counter()
    for index, case in enumerate(cases):
        row = {**case, "index": index, "arm": arm, "request": None,
               "status_code": None, "response": None, "response_text": None, "error": None}
        start = time.perf_counter()
        try:
            row["request"] = request_for(case, model, image_root)
            send_request(row, client)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["wall_seconds"] = time.perf_counter() - start
        output.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        output.flush()
        counts["requests"] += 1
        counts["errors"] += row["error"] is not None
        if index % 50 == 0 or index + 1 == len(cases):
            print(json.dumps({"arm": arm, "progress": index + 1, "total": len(cases),
                              "errors": counts["errors"]}), flush=True)
    return dict(counts)


def lossless_webp(image_path, cache):
    """Cache one encoding per canonical image for reuse by every arm."""
    from PIL import Image

    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (Path(image_path).name + ".webp")
    if not destination.exists():
        with Image.open(image_path) as image:
            # Match the native API decoder's RGB conversion.
            rgb = image.convert("RGB")
            rgb.info.clear()
            rgb.save(destination, format="WEBP", lossless=True, quality=80, method=4)
    return destination


def retry_cases(path, client, output, *, image_cache, image_root=Path(".")):
    """Replay each original HTTP 413 once, changing only its image transport."""
    counts = Counter(requests=0, errors=0)
    for original in jsonl(path):
        if original.get("status_code") != 413 or original.get("retry_of"):
            continue
        row = {**original, "request": None, "status_code": None, "response": None,
               "response_text": None, "error": None,
               "retry_of": {"path": Path(path).name, "index": original["index"],
                            "id": original["record"]["id"], "arm": original["arm"]},
               "transport": {"format": "WEBP", "lossless": True, "quality": 80, "method": 4}}
        start = time.perf_counter()
        try:
            cached = lossless_webp(image_root / original["record"]["image_path"], image_cache)
            row["transport"]["path"] = cached.name
            row["request"] = {**original["request"], "image": image_url(cached)}
            send_request(row, client)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["wall_seconds"] = time.perf_counter() - start
        output.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        output.flush()
        counts["requests"] += 1
        counts["errors"] += row["error"] is not None
        print(json.dumps({"arm": row["arm"], "retry_index": row["index"],
                          "status_code": row["status_code"], "errors": counts["errors"]}), flush=True)
    return dict(counts)


def scoring_row(row):
    """Discard image bodies and prompt context before retaining scoring inputs."""
    return {**{key: row.get(key) for key in (
        "arm", "set", "track", "index", "status_code", "response", "error", "retry_of")},
        "record": {key: row["record"][key] for key in ("id", "candidates", "target")}}


def score_row(row, method):
    """Align probabilities by supplied candidate ID before the existing scorer."""
    from fuzev_research.metrics import per_record

    if row.get("error") is not None or row["status_code"] != 200:
        raise ValueError(row.get("error") or f"HTTP {row['status_code']}")
    record = row["record"]
    response = row["response"] if method == "head" else row["response"]["backbone_control"]
    ids = [candidate["id"] for candidate in record["candidates"]]
    scores = response["scores"]
    by_id = {score["id"]: score for score in scores}
    if len(by_id) != len(scores) or set(by_id) != set(ids):
        raise ValueError("Response candidate IDs differ from the request")
    probability_field = ("calibrated_probability" if response.get("calibration_id") == "calibrated"
                         else "raw_probability")
    probabilities = [by_id[identifier][probability_field] for identifier in ids]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        raise ValueError("Response probabilities are invalid")
    if not any(probabilities):
        raise ValueError("Response has no positive probability")
    values = per_record([[math.log(max(value, 1e-300)) for value in probabilities]], [record["target"]])
    result = {name: float(value[0]) for name, value in values.items()}
    selected = ids.index(response["selected_id"])
    target, best = record["target"], max(record["target"])
    result["tie_aware_accuracy"] = float(target[selected] == best)
    result["accuracy"] = result["tie_aware_accuracy"] if target.count(best) == 1 else float("nan")
    result["expected_accuracy"] = target[selected]
    result["confidence"] = probabilities[selected] / sum(probabilities)
    return result


def summarize_rows(rows, methods=("head",)):
    import numpy as np
    from fuzev_research.metrics import summarize

    groups = defaultdict(lambda: {"requests": 0, "values": [], "errors": [], "order": Counter()})
    for row in rows:
        for method in methods:
            value, failure = None, None
            try:
                value = score_row(row, method)
            except Exception as error:
                failure = {"index": row["index"], "id": row["record"]["id"],
                           "error": f"{type(error).__name__}: {error}"}
            for track in (row["track"], "all"):
                group = groups[row["arm"], method, row["set"], track]
                group["requests"] += 1
                target = row["record"]["target"]
                if len(target) == 2:
                    order = group["order"]
                    order["binary_requests"] += 1
                    order["gold_first"] += target[0] > target[1]
                    order["gold_second"] += target[1] > target[0]
                    if failure is None:
                        response = row["response"] if method == "head" else row["response"]["backbone_control"]
                        first = response["selected_id"] == row["record"]["candidates"][0]["id"]
                        order["first_pick"] += first
                        order["gold_second_scored"] += target[1] > target[0]
                        order["gold_second_correct"] += target[1] > target[0] and not first
                if failure is None:
                    group["values"].append(value)
                else:
                    group["errors"].append(failure)
    result = []
    for (arm, method, name, track), group in groups.items():
        values, errors = group["values"], group["errors"]
        entry = {"arm": arm, "method": method, "set": name, "track": track,
                 "requests": group["requests"], "scored": len(values), "errors": errors}
        if group["order"]:
            entry.update(group["order"])
        if values:
            combined = {key: np.asarray([row[key] for row in values]) for key in values[0]}
            metrics = summarize(combined)
            metrics["unique_gold_records"] = int(np.isfinite(combined["accuracy"]).sum())
            metrics["unique_gold_accuracy"] = metrics.pop("accuracy", None)
            metrics["accuracy"] = metrics["tie_aware_accuracy"]
            metrics["correct"] = int(combined["tie_aware_accuracy"].sum())
            entry["metrics"] = metrics
        result.append(entry)
    return result


def score_files(inputs, methods=("head",), retries=()):
    replacements, attempts = {}, []
    for path in retries:
        for row in (scoring_row(raw) for raw in jsonl(path)):
            source = row["retry_of"]
            attempts.append({"path": Path(path).name, "retry_of": {**source, "path": Path(source["path"]).name},
                             "status_code": row["status_code"], "error": row["error"]})
            if row["status_code"] == 200 and row["error"] is None:
                key = (Path(source["path"]).name, source["index"], source["arm"], source["id"])
                replacements.setdefault(key, row)

    def rows(completed=False):
        for path in inputs:
            source = Path(path).name
            for row in (scoring_row(raw) for raw in jsonl(path)):
                key = (source, row["index"], row["arm"], row["record"]["id"])
                yield replacements.get(key, row) if completed and row["status_code"] == 413 else row

    initial = summarize_rows(rows(), methods)
    if not retries:
        return initial
    return {"initial_pass": initial, "retry_completed": summarize_rows(rows(True), methods),
            "retry_attempts": attempts}


def baseline_response(values, candidates):
    weights = [math.exp(value - max(values)) for value in values]
    probabilities = [weight / sum(weights) for weight in weights]
    return {"calibration_id": "raw", "selected_id": candidates[
        max(range(len(values)), key=values.__getitem__)].id,
        "scores": [{"id": candidate.id, "logit": logit, "raw_probability": probability,
                    "calibrated_probability": probability}
                   for candidate, logit, probability in zip(candidates, values, probabilities)]}


@contextmanager
def capture_backbone(service):
    """Observe the already computed baseline; serialize only this diagnostic server."""
    head = service.head.head
    score, decide = head.score, service.decide
    captured, lock = {}, asyncio.Lock()

    def capture(*args, baseline, **kwargs):
        result = score(*args, baseline=baseline, **kwargs)
        captured["baseline"] = baseline.detach().float().cpu().tolist()
        return result

    async def observed(request, *args, **kwargs):
        async with lock:
            captured.clear()
            try:
                response = await decide(request, *args, **kwargs)
                if "baseline" in captured:
                    response["backbone_control"] = baseline_response(captured["baseline"], request.candidates)
                else:
                    response["backbone_control_error"] = "The native scoring call produced no baseline"
                return response
            finally:
                captured.clear()

    with patch.object(head, "score", capture), patch.object(service, "decide", observed):
        yield service


def serve_with_capture(argv):
    from fuzev_exllamav3 import generator_service, server

    with ExitStack() as stack:
        original = generator_service.build_generator_service

        def build(*args, **kwargs):
            service = original(*args, **kwargs)
            return stack.enter_context(capture_backbone(service))

        stack.enter_context(patch.object(generator_service, "build_generator_service", build))
        asyncio.run(server.serve(server.parser().parse_args(argv)))


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("inventory", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--set", action="append", required=True, help="NAME=PATH of prepared inputs")
        sub.add_argument("--image-root", type=Path, required=True)
        if name == "run":
            sub.add_argument("--base-url", default="http://127.0.0.1:8000")
            sub.add_argument("--arm", required=True)
            sub.add_argument("--model")
            sub.add_argument("--timeout", type=float, default=3600.)
            sub.add_argument("--api-key")
            sub.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("inputs", type=Path, nargs="+")
    score.add_argument("--methods", choices=("head", "backbone"), nargs="+", default=["head"])
    score.add_argument("--retries", type=Path, nargs="+", default=[])
    retry = commands.add_parser("retry", help="Replay original HTTP 413 rows with lossless WebP images")
    retry.add_argument("input", type=Path)
    retry.add_argument("--base-url", default="http://127.0.0.1:8000")
    retry.add_argument("--timeout", type=float, default=3600.)
    retry.add_argument("--api-key")
    retry.add_argument("--image-root", type=Path, required=True)
    retry.add_argument("--image-cache", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)
    commands.add_parser("serve", help="Native server arguments, with isolated LM-label capture")
    return cli


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        serve_with_capture(argv[1:])
        return
    args = parser().parse_args(argv)
    headers = {"Authorization": f"Bearer {args.api_key}"} if getattr(args, "api_key", None) else None
    if args.command == "score":
        result = score_files(args.inputs, args.methods, args.retries)
    elif args.command == "retry":
        import httpx

        with args.output.open("x") as output, httpx.Client(base_url=args.base_url.rstrip("/"),
                timeout=args.timeout, headers=headers, trust_env=False) as client:
            result = retry_cases(args.input, client, output, image_cache=args.image_cache,
                                 image_root=args.image_root)
    else:
        paths = dict(spec.split("=", 1) for spec in args.set)
        cases = load_cases(paths)
        if args.command == "inventory":
            result = inventory(cases, args.image_root)
        else:
            import httpx

            with args.output.open("x") as output, httpx.Client(base_url=args.base_url.rstrip("/"),
                    timeout=args.timeout, headers=headers, trust_env=False) as client:
                result = run_cases(cases, client, output, arm=args.arm, model=args.model,
                                   image_root=args.image_root)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
