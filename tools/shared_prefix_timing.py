"""Time grouped and concurrent SystemOne questions over the same shared text.

Questions are a JSON object in the SystemOne ``questions`` format. Execution
counters come from separate, untimed decision probes with the same inputs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from smoke_head import Client


def state_text(value, indent=0):
    """Render JSON values with the SystemOne text encoding."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {state_text(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{state_text(item, indent + 1)}"
        if isinstance(item, (dict, list)) else f"{pad}{key}: {state_text(item)}"
        for key, item in value.items())


def decision_payload(model, context, question):
    """Render the text-only SystemOne question as a public decision request."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        options = [("yes", "Yes", criteria.get("true")),
                   ("no", "No", criteria.get("false"))]
    elif kind == "choice":
        options = [(key, key, value) for key, value in criteria.items()]
    else:
        options = [(str(index), value, None) for index, value in enumerate(criteria)]
    candidates = [{"id": key, "text": state_text(text) if description in (None, "")
                   else f"{text}: {state_text(description)}"} for key, text, description in options]
    return {"model": model, "primitive": "ordinal" if kind == "score" else kind,
            "messages": [{"role": "user", "content": context}],
            "question": state_text(question.get("instructions")), "candidates": candidates}


def timed_call(client, path, payload):
    started = time.perf_counter()
    response = client.call(path, payload)
    result = {"wall_seconds": time.perf_counter() - started}
    for key in ("usage", "execution", "latency_ms"):
        if key in response:
            result[key] = response[key]
    return result


def measure(client, context, questions, samples):
    records = []
    for sample in range(samples):
        record = {"sample": sample, "health": client.call("/health")}
        payload = {"model": client.model, "state": context, "questions": questions}
        record["grouped"] = timed_call(client, "/v1/systemone", payload)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(questions)) as pool:
            futures = {key: pool.submit(timed_call, client, "/v1/systemone",
                       {**payload, "questions": {key: question}})
                       for key, question in questions.items()}
            requests = {key: future.result() for key, future in futures.items()}
        record["concurrent"] = {"wall_seconds": time.perf_counter() - started,
                                "requests": requests}
        record["decision_probes"] = {key: timed_call(client, "/v1/decisions",
            decision_payload(client.model, context, question))
            for key, question in questions.items()}
        records.append(record)
    return records


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--model", required=True)
    cli.add_argument("--api-key")
    cli.add_argument("--context", type=Path, required=True)
    cli.add_argument("--questions", type=Path, required=True)
    cli.add_argument("--samples", type=int, default=1)
    cli.add_argument("--label", default="FuZev")
    cli.add_argument("--context-source", help="Public source description for the shared text")
    cli.add_argument("--environment", type=Path, help="Recorded device, software and server settings JSON")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    context = args.context.read_text()
    questions = json.loads(args.questions.read_text())
    result = {"label": args.label, "model": args.model,
              "context": args.context.name, "questions": args.questions.name,
              "question_count": len(questions), "context_characters": len(context),
              "protocol": {"grouped": "one SystemOne request with all questions",
                           "concurrent": "one concurrent SystemOne request per question",
                           "cache_order": "grouped first, concurrent next, then counter probes; no warmup or cache reset",
                           "decision_probes": "separate counter probes, excluded from SystemOne timings"},
              "samples": measure(Client(args.url, args.model, args.api_key),
                                 context, questions, args.samples)}
    if args.context_source:
        result["context_source"] = args.context_source
    if args.environment:
        result["environment"] = json.loads(args.environment.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
