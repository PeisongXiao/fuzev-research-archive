"""Submit decisions while a generation request remains in flight."""

from concurrent.futures import ThreadPoolExecutor
import json
import time

from smoke_head import Client, arguments


def main():
    cli = arguments(__doc__)
    cli.add_argument("--delay", type=float, default=0.1, help="Seconds before submitting decisions")
    cli.add_argument("--generation-tokens", type=int, default=256)
    args = cli.parse_args()
    client = Client(args.url, args.model, args.api_key)
    result = {"model": args.model, "delay_seconds": args.delay, "rows": []}

    def measured(name, function):
        started = time.monotonic()
        value = function()
        row = {"name": name, "started": started, "finished": time.monotonic(), "response": value}
        result["rows"].append(row)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(measured, "generation", lambda: client.generate(
            max_tokens=args.generation_tokens, text="Write a short story about a scientist."))
        time.sleep(args.delay)
        decision = pool.submit(measured, "decisions", client.decide)
        a, b = generation.result(), decision.result()
    result["overlap_seconds"] = max(0, min(a["finished"], b["finished"]) - max(a["started"], b["started"]))
    result["health_after"] = client.call("/health")
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
