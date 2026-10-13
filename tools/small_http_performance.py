"""Measure end-to-end API latency for generation, decisions and their combination."""

from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time
from uuid import uuid4

from smoke_head import Client, arguments

CODE = ("Complete this Python file with a short main() function.\n\n"
        + "\n".join(f"def step_{index}(value):\n    return value + {index}\n" for index in range(24)))


def main():
    cli = arguments(__doc__)
    cli.add_argument("--streams", type=int, nargs="+", default=[1])
    cli.add_argument("--samples", type=int, default=3)
    cli.add_argument("--generation-tokens", type=int, default=32)
    args = cli.parse_args()
    client = Client(args.url, args.model, args.api_key)
    rows = []

    def pipeline(mode):
        prefix = None
        started = time.perf_counter()
        try:
            generation = None
            if mode != "decision":
                generation = client.generate(max_tokens=args.generation_tokens, retain=mode == "mixed",
                                             text=uuid4().hex + "\n" + CODE)
                prefix = generation.get("prefix_id")
            decision = client.decide(prefix) if mode != "generation" else None
            return {"seconds": time.perf_counter() - started,
                    "generation_usage": generation["usage"] if generation else None,
                    "decision_usage": decision["usage"] if decision else None}
        finally:
            if prefix is not None:
                client.call("/v1/prefixes/" + prefix, method="DELETE")

    for mode in ("generation", "decision", "mixed"):
        for streams in args.streams:
            with ThreadPoolExecutor(max_workers=streams) as pool:
                values = list(pool.map(lambda _: pipeline(mode), range(args.samples)))
            durations = [value["seconds"] for value in values]
            rows.append({"mode": mode, "streams": streams, "samples": values,
                         "median_seconds": statistics.median(durations),
                         "min_seconds": min(durations), "max_seconds": max(durations)})
    result = {"model": args.model, "url": args.url, "metric": "end-to-end HTTP latency",
              "requested_generation_tokens": args.generation_tokens, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
