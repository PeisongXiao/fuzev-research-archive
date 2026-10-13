"""Small generation and decision API smoke check."""

import argparse
import json
from pathlib import Path
import urllib.request
from uuid import uuid4


class Client:
    def __init__(self, url, model, api_key=None, timeout=600):
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout

    def call(self, path, data=None, method=None):
        headers = {"Content-Type": "application/json", "X-TypeSafe-Request-ID": "smoke_" + uuid4().hex}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url + path,
            data=None if data is None else json.dumps(data).encode(),
            method=method or ("GET" if data is None else "POST"), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def generate(self, *, max_tokens=32, retain=False, text="Name two primary colors."):
        return self.call("/v1/chat/completions", {
            "model": self.model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "retain_decision_prefix": retain})

    def decide(self, prefix_id=None):
        payload = {"model": self.model, "state": "The ticket requests a refund.",
                   "questions": {
                       "team": {"type": "choice", "criteria": {"billing": "Charges", "shipping": "Delivery"}},
                       "refund": {"type": "noul", "instructions": "Does the ticket request a refund?"},
                       "clarity": {"type": "score", "criteria": ["unclear", "clear"]}}}
        if prefix_id is not None:
            payload["prefix_id"] = prefix_id
        return self.call("/v1/systemone", payload)


def arguments(description):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--model", required=True)
    cli.add_argument("--api-key")
    cli.add_argument("--output", type=Path, required=True)
    return cli


def main():
    args = arguments(__doc__).parse_args()
    client = Client(args.url, args.model, args.api_key)
    result = {"model": args.model, "url": args.url, "health": client.call("/health")}
    prefix = None
    try:
        result["generation_before"] = client.generate(retain=True)
        prefix = result["generation_before"]["prefix_id"]
        result["standalone_decisions"] = client.decide()
        result["retained_prefix_decisions"] = client.decide(prefix)
        result["generation_after"] = client.generate()
        result["status"] = "passed"
    finally:
        if prefix is not None:
            result["release"] = client.call("/v1/prefixes/" + prefix, method="DELETE")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
