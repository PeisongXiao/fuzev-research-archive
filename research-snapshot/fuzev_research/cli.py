import argparse
import json

from .state import Store


def main():
    parser = argparse.ArgumentParser(prog="fuzev-research")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    update = sub.add_parser("task")
    update.add_argument("id")
    update.add_argument("status")
    update.add_argument("--detail", default="")
    update.add_argument("--progress", type=float)
    result = sub.add_parser("result")
    result.add_argument("task")
    result.add_argument("file")
    serve = sub.add_parser("dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    store = Store()
    store.initialize()
    if args.command == "status":
        print(json.dumps(store.snapshot(), indent=2))
    elif args.command == "task":
        store.update(args.id, args.status, args.detail, args.progress)
    elif args.command == "result":
        from pathlib import Path
        store.record_result(args.task, json.loads(Path(args.file).read_text()))
    elif args.command == "dashboard":
        import uvicorn
        from .dashboard import create_app
        uvicorn.run(create_app(store), host=args.host, port=args.port,
                    timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
