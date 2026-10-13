"""Small fit-only physical-4/physical-8 and grouping qualification.

Run --capacity 4 first, then --capacity 8 --control <first report.json>.
Each invocation loads one model. The operator owns exclusive GPU admission.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import torch

from fuzev_research.acquire import write_json
from fuzev_research.native import NativeBackend, hashlib_file, runtime_sources
from fuzev_research.paths import ROOT
from fuzev_research.telemetry import MemoryGuard, Sampler
from tools.probe_extraction_concurrency import EXTENSION, NATIVE, ObservedBackend, reset_prefixes


def select(path):
    saved = json.loads(path.read_text())["rows"]
    if any(item["row"]["role"] != "fit" for item in saved):
        raise ValueError("Only the saved fitting fixtures may enter this probe")
    selected = []
    for track, count in (("ordinal", 4), ("text_choice", 1), ("qasper_noul", 1), ("aok_choice", 1), ("mm_choice", 1)):
        items = [item for item in saved if item["track"] == track][:count]
        if len(items) != count:
            raise ValueError("Missing required saved fitting track: " + track)
        selected.extend(items)
    if len({item["row"]["id"] for item in selected}) != 8:
        raise ValueError("Fixture IDs are not unique")
    return selected


def vector_metrics(left, right):
    a, b = torch.as_tensor(left, dtype=torch.float64), torch.as_tensor(right, dtype=torch.float64)
    if a.shape != b.shape or not a.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Mismatched or nonfinite feature arrays")
    a, b = a.flatten(), b.flatten()
    difference = a - b
    na, nb = a.norm().item(), b.norm().item()
    return {"max": difference.abs().max().item(), "rms": difference.square().mean().sqrt().item(),
            "cosine": max(-1., min(1., (a @ b).item() / (na * nb))) if na and nb else None,
            "relative_l2": difference.norm().item() / na if na else None,
            "norm_ratio": nb / na if na else None}


def compare(a, b):
    for key in ("base_token_sha256", "query_token_sha256", "prefix_tokens", "branch_prefix_tokens", "expanded_image_tokens"):
        if a["execution"][key] != b["execution"][key]:
            raise ValueError("Exact input/visual alias identity changed: " + key)
    ca, cb = torch.tensor(a["candidates"], dtype=torch.float64), torch.tensor(b["candidates"], dtype=torch.float64)
    return {"query": vector_metrics(a["query"], b["query"]),
            "candidates": vector_metrics(a["candidates"], b["candidates"]),
            "baseline": vector_metrics(a["baseline"], b["baseline"]),
            "centered_candidates": vector_metrics(ca - ca.mean(0), cb - cb.mean(0))}


def stable(metrics):
    return all(metrics[key]["max"] <= .02 and metrics[key]["rms"] <= .005
               for key in ("query", "candidates", "baseline"))


def payload(result):
    query, candidates, baseline, execution = result
    return {"query": query.tolist(), "candidates": candidates.tolist(),
            "baseline": baseline.tolist(), "execution": execution}


def control_report(path, selected):
    report = json.loads(path.read_text())
    if report.get("status") != "fit_stability_pass_quality_pending" or report.get("physical_batch_capacity") != 4:
        raise ValueError("Control must be a completed physical-4 report")
    runtime = report.get("initial_runtime", {})
    if any(runtime.get(key) != 4 for key in ("batch_capacity", "model_batch_capacity", "max_hybrid_slots", "candidate_batch_size")):
        raise ValueError("Control runtime must actually declare four model/cache slots and B4 grouping")
    if report["fixture_ids"] != [item["row"]["id"] for item in selected]:
        raise ValueError("Control fixtures/order differ")
    return report


def verify_native(sources, expected):
    try:
        from exllamav3.fuzev import NativeLane
    except ImportError as error:
        raise ImportError("Requires the historical engine module exllamav3/fuzev.py "
                          "from research-snapshot/historical-native-source") from error
    import inspect
    repository = Path(inspect.getfile(NativeLane)).resolve().parents[1]
    return {"source_directory": str(repository), "observed_sources": sources, "recorded_native_commit": NATIVE,
            "operator_expected_commit": expected}


def verify_control_sources(control, sources):
    return {"control_sources": control, "candidate_sources": sources,
            "recorded_native_commit": NATIVE,
            "recipe_changes": None, "native_difference": "source revisions recorded; difference unmeasured"}


def run(args):
    if json.loads((ROOT / "data/manifests/acquisition.json").read_text()).get("dataset_scope"):
        raise ValueError("Historical QASPER/A-OKVQA probe is archived under active scope v2")
    if not args.allow_gpu:
        raise ValueError("Pass --allow-gpu after stopping competing GPU work")
    selected = select(args.selection)
    control = control_report(args.control, selected) if args.control else None
    if args.capacity == 4 and control is not None:
        raise ValueError("A physical-4 control invocation cannot consume another control")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs/native-real").resolve()):
        raise ValueError("Use a fresh isolated runs/native-real directory")
    sources = runtime_sources()
    verify_native(sources, args.expected_native_commit)
    selection_sha = hashlib_file(args.selection)
    if control is not None and selection_sha != control["selection_sha256"]:
        raise ValueError("Control selection file identity differs")
    source_comparison = verify_control_sources(control["sources"], sources) if control else None
    model = json.loads((ROOT / "data/manifests/acquisition.json").read_text())["models"]["native-4.05"]
    snapshot = Path(model["local_snapshot"])
    model_identity = {key: model[key] for key in ("repo", "revision", "local_snapshot")}
    model_identity["config_sha256"] = hashlib_file(snapshot / "config.json")
    if control is not None and control["model"] != model_identity:
        raise ValueError("Control model/configuration identity differs")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", {"rows": selected, "source_sha256": selection_sha})
    report = {"status": "running", "sources": sources, "model": model_identity, "cache_precision": "float16",
              "control_source_comparison": source_comparison,
              "selection_sha256": selection_sha,
              "physical_batch_capacity": args.capacity, "server_admission_default": 4,
              "fixture_ids": [item["row"]["id"] for item in selected], "records": [],
              "timing": "two warmups then two timed complete baseline=True readouts per grouping; no shard writes",
              "same_geometry_limits": {"max": .02, "rms": .005}, "cross_geometry_limit": None,
              "budget": {"limit": args.max_forward_tokens, "calls": 0, "forwarded_tokens": 0},
              "head_quality_tested": False, "physical4_control": str(args.control) if args.control else None}
    guard, sampler, backend = MemoryGuard(), Sampler(), None
    guard.start()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-batch-probe") as owner:
            try:
                backend = owner.submit(NativeBackend, snapshot, batch_capacity=args.capacity,
                                       candidate_batch_size=args.capacity).result()
                if backend.readout_runtime["native_extension_sha256"] != EXTENSION or not backend.readout_runtime["optimized_foundation"]:
                    raise ValueError("Original e139 extension and positive foundation are mandatory")
                report["initial_runtime"] = dict(backend.readout_runtime)
                report["telemetry_after_load"] = sampler.sample()
                if control is not None:
                    geometry = {"candidate_batch_size", "batch_capacity", "model_batch_capacity", "max_hybrid_slots",
                                "recurrent_state_storage_bytes", "extra_recurrent_state_bytes_vs_capacity4"}
                    current = {key: value for key, value in report["initial_runtime"].items() if key not in geometry}
                    reference = {key: value for key, value in control["initial_runtime"].items() if key not in geometry}
                    if current != reference or control["cache_precision"] != report["cache_precision"]:
                        raise ValueError("Control non-geometry runtime identity differs")
                images = {item["row"]["image_sha256"]: item["row"] for item in selected if item["row"].get("image_path")}
                for row in images.values():
                    owner.submit(backend.image_embedding, row["image_path"], row["image_sha256"]).result()
                observed = ObservedBackend(backend, {item["row"]["id"]: item for item in selected},
                                           report["budget"], guard, "geometry")
                timings = {"b4": [], "b8": []}
                for index, item in enumerate(selected):
                    row, outputs = item["row"], {}
                    order = [4] if args.capacity == 4 else ([4, 8] if index % 2 == 0 else [8, 4])
                    result = {"record": row["id"], "track": item["track"], "outputs": outputs, "grouping_order": order}
                    report["records"].append(result)
                    for size in order:
                        owner.submit(backend.set_candidate_batch_size, size).result()
                        for _ in range(2):
                            owner.submit(observed.extract, row, True).result()
                        results, durations = [], []
                        for _ in range(2):
                            started = time.perf_counter()
                            results.append(payload(owner.submit(observed.extract, row, True).result()))
                            durations.append(time.perf_counter() - started)
                        key = f"b{size}"
                        outputs[key] = {"result": results[0], "repeat_result": results[1],
                                        "seconds": durations, "grouping": results[0]["execution"]["candidate_forward_batches"]}
                        write_json(output / "report.json", report)
                        if any(value["execution"]["mode"] != "checkpoint" for value in results):
                            raise ValueError("Timed readout did not reuse its identical warm prefix")
                        repeat = compare(*results)
                        outputs[key]["repeat"] = repeat
                        if not stable(repeat):
                            raise ValueError(f"Same-geometry B{size} repeat is unstable")
                        timings[key].extend(durations)
                    owner.submit(backend.set_candidate_batch_size, args.capacity).result()
                    owner.submit(reset_prefixes, backend, backend).result()
                    fresh = payload(owner.submit(observed.extract, row, True).result())
                    result["fresh_result"] = fresh
                    write_json(output / "report.json", report)
                    if fresh["execution"]["mode"] != "fresh":
                        raise ValueError("Fresh-prefix check reused an existing prefix")
                    same = compare(outputs[f"b{args.capacity}"]["result"], fresh)
                    result["fresh_comparison"] = same
                    if not stable(same):
                        raise ValueError("Fresh-versus-warm stability failed")
                    if args.capacity == 8:
                        result["b4_on8_vs_b8"] = compare(outputs["b4"]["result"], outputs["b8"]["result"])
                        if control is not None:
                            reference = control["records"][index]["outputs"]["b4"]["result"]
                            result["physical4_vs_b4_on8"] = compare(reference, outputs["b4"]["result"])
                            result["physical4_vs_b8"] = compare(reference, outputs["b8"]["result"])
                    write_json(output / "report.json", report)
                    print(json.dumps({"completed": index + 1, "track": item["track"],
                                      "actual_groups": {key: value["grouping"] for key, value in outputs.items()}}), flush=True)
                report["warm_readout_rates"] = {key: {"calls": len(values), "seconds": sum(values),
                                                         "records_per_second": len(values) / sum(values)}
                                                for key, values in timings.items() if values}
                if args.capacity == 8:
                    report["b4_on8_vs_b8_speed_ratio"] = report["warm_readout_rates"]["b8"]["records_per_second"] / report["warm_readout_rates"]["b4"]["records_per_second"]
                    if control is not None:
                        report["physical4_vs_b8_speed_ratio"] = report["warm_readout_rates"]["b8"]["records_per_second"] / control["warm_readout_rates"]["b4"]["records_per_second"]
                report["actual_group_histogram"] = dict(Counter(size for item in report["records"]
                    for size in item["outputs"][f"b{args.capacity}"]["grouping"]))
                report["largest_observed_group"] = max(map(int, report["actual_group_histogram"]))
                report["full_eight_group_executed"] = report["largest_observed_group"] == 8
                report["telemetry_after"] = sampler.sample()
                guard.check()
                report["memory_guard"] = guard.snapshot()
                report["status"] = "fit_stability_pass_quality_pending"
                report["cross_geometry_review"] = "numeric drift reported; downstream quality gates remain required"
                if args.capacity == 8 and control is None:
                    report["physical4_comparison_missing"] = True
                if runtime_sources() != sources:
                    raise ValueError("Sources changed during the probe")
            finally:
                if backend is not None:
                    owner.submit(backend.close).result()
    except BaseException as error:
        report["status"] = "failed_or_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        guard.close()
        write_json(output / "report.json", report)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--capacity", type=int, choices=(4, 8), required=True)
    parser.add_argument("--expected-native-commit", required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--selection", type=Path, default=ROOT / "runs/native-real/stock-distribution-20261001/selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-forward-tokens", type=int, default=1000000)
    args = parser.parse_args()
    if not 0 < args.max_forward_tokens <= 1000000:
        parser.error("Positive token budget must be at most 1M")
    return args


if __name__ == "__main__":
    run(arguments())
