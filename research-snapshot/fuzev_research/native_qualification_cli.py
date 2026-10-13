"""Explicit offline construction for the existing native correctness runner."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys


MODEL = "native-4.05"


def _backend_options(config):
    runtime = config.feature_identity.get("readout_runtime", {})
    for key in ("candidate_batch_size", "batch_capacity", "model_batch_capacity", "max_hybrid_slots"):
        if type(runtime.get(key)) is not int or runtime[key] != 8:
            raise ValueError("Offline native4 requires the frozen B8 feature geometry")
    cache_tokens = runtime.get("cache_tokens")
    fraction = runtime.get("memory_fraction")
    if type(cache_tokens) is not int or cache_tokens < 262144:
        raise ValueError("Frozen cache_tokens must cover the native 262144-token window")
    if fraction is not None and (type(fraction) not in (int, float) or not math.isfinite(fraction)
                                or not 0 < fraction <= 1):
        raise ValueError("memory_fraction must be in (0, 1]")
    if runtime.get("optimized_foundation") is not True:
        raise ValueError("Offline native4 requires the frozen optimized foundation")
    return {"candidate_batch_size": runtime["candidate_batch_size"],
            "batch_capacity": runtime["batch_capacity"], "cache_tokens": cache_tokens,
            "memory_fraction": fraction, "optimized": runtime["optimized_foundation"]}


def _run(args):
    from .acquire import MODELS
    from .native_qualification import QualificationConfig, run_qualification
    from .paths import project_path
    from .service_contracts import ServiceConfig

    config = ServiceConfig.model_validate_json(args.service_config.read_bytes())
    cases = QualificationConfig.model_validate_json(args.cases.read_bytes())
    options = _backend_options(config)
    model = json.loads(project_path("data/manifests/acquisition.json").read_text())["models"][MODEL]
    snapshot = model.get("local_snapshot")
    if not isinstance(snapshot, str) or not snapshot or not Path(snapshot).is_dir():
        raise ValueError("Pinned native-4.05 model acquisition is not complete")
    all_cases = cases.cases + ([cases.full_window.case] if cases.full_window else [])
    if (any(case.generation.model != MODEL or case.input_role != "operator-non-heldout"
            for case in all_cases)
            or any(control.request.model != MODEL or control.input_role != "operator-non-heldout"
                   for control in cases.image_controls)):
        raise ValueError("Offline native4 requires native-4.05 operator-non-heldout cases")

    # Reserve the operator's unique path before any backend can be constructed.
    with args.output.open("x", encoding="utf-8") as output:
        from .native import NativeBackend, runtime_sources
        from .native_service import build_service

        sources = runtime_sources()
        with ExitStack() as resources:
            backend = NativeBackend(snapshot, **options)
            resources.callback(backend.close)
            service = build_service(backend, MODEL, model, sources, config)
            resources.callback(service.close)
            report = dict(run_qualification(service, cases, cpu_fake=False, allow_inference=True))
            report["source_provenance"] = {"observed": sources, "recorded": config.service_sources}
            report["model_provenance"] = {"observed": {key: model.get(key) for key in ("repo", "revision")},
                                          "recorded": {"repo": MODELS[MODEL][0], "revision": MODELS[MODEL][1]}}
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            output.flush()

    if (report.get("execution_status") != "passed"
            or report.get("status") in {"failed", "cancelled", "memory_infeasible"}
            or report.get("cleanup_errors") or report.get("qualified") is not False):
        print(f"Native qualification execution did not pass; see {args.output}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--service-config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-inference", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("Native qualification interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Native qualification failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
