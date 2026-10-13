"""Fit-only matched extraction probe. Operator must stop other GPU work first.

One model load; original 6a133f2 serialized readout versus candidate pipeline,
written against unpublished lab engine commits requiring exllamav3/fuzev.py,
both stock B4 with baseline=True and identical complete rows and shard sizes.
No model promotion, head fitting, held-out reads, or service/container changes.
"""

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types

import torch

from fuzev_research.acquire import write_json
from fuzev_research.extraction import FeatureProxy, PIPELINE_POLICY, write_rows
from fuzev_research.features import FeatureStore
from fuzev_research.native import NativeBackend, hashlib_file, runtime_sources
from fuzev_research.paths import ROOT
from fuzev_research.preparation import ReadoutPreparer
from fuzev_research.telemetry import MemoryGuard, Sampler


CONTROL = "6a133f226f220e24a7b19af8beb3e3e182798b88"  # unpublished lab commit ID
NATIVE = "d61c4df089ac19d52edb4ce45450b222cefc5959"  # unpublished lab commit ID
EXTENSION = "e1398c20e1b1c4a0c91c8661790cd2543156c3556af2d5963c75dfbe4fc6c05b"
TRACKS = ("ordinal", "text_choice", "qasper_noul", "aok_choice", "mm_choice")


def select(path, count):
    selected = json.loads(path.read_text())["rows"]
    if any(item["row"]["role"] != "fit" for item in selected):
        raise ValueError("Fixture file is not fitting-only")
    quotas = {track: count // len(TRACKS) for track in TRACKS}
    # Every track participates; spare cases favor repeated long-paper encodings.
    for track in ("qasper_noul", "ordinal", "text_choice", "aok_choice")[:count % len(TRACKS)]:
        quotas[track] += 1
    groups = []
    for track in TRACKS:
        values = [item for item in selected if item["track"] == track][:quotas[track]]
        if len(values) != quotas[track] or not values:
            raise ValueError(f"Insufficient saved fit fixtures: {track}")
        for item in values:
            if item["bounds"]["branch_proxy"] > 26000 or item["bounds"]["tail_byte_bound"] > 8192:
                raise ValueError("Fixture exceeds the previously declared small-probe bounds")
        groups.append(values)
    if len({item["row"]["id"] for group in groups for item in group}) != count:
        raise ValueError("Fixture IDs are not unique")
    return groups


def original_backend(source_directory):
    path = Path(source_directory) / "fuzev_research/native.py"
    source = path.read_bytes()
    module = types.ModuleType("fuzev_research._probe_serialized_6a133f2")
    module.__package__ = "fuzev_research"
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.NativeBackend, hashlib.sha256(source).hexdigest()


class ObservedBackend:
    """Identical admission/accounting around either actual readout implementation."""
    def __init__(self, backend, fixtures, budget, guard, arm):
        self.backend, self.fixtures, self.budget = backend, fixtures, budget
        self.guard, self.arm, self.trace = guard, arm, []

    def _run(self, method, row, *args):
        self.guard.check()
        bounds = self.fixtures[row["id"]]["bounds"]
        reserve = bounds["branch_proxy"] + bounds["tail_byte_bound"]
        if self.budget["forwarded_tokens"] + reserve > self.budget["limit"]:
            raise ValueError("Probe forward-token admission budget exhausted")
        result = method(row, *args)
        self.guard.check()
        query, candidates, baseline, execution = result
        actual = execution["forwarded_tokens"]
        self.budget["forwarded_tokens"] += actual
        self.budget["calls"] += 1
        if actual > reserve or self.budget["forwarded_tokens"] > self.budget["limit"]:
            raise ValueError("Readout exceeded conservative forward-token admission")
        if execution["branch_prefix_tokens"] > 26000:
            raise ValueError("Readout exceeded the small-probe context bound")
        if baseline is None:
            raise ValueError("Full extraction probe requires untouched baseline logits")
        for value in (query, candidates, baseline):
            if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
                raise ValueError("Readout must return owned, gradient-free CPU features")
        self.trace.append({"record": row["id"], **execution})
        return result

    def extract(self, row, baseline=True):
        if not baseline:
            raise ValueError("Never omit the label-token baseline in this probe")
        return self._run(self.backend.extract, row, baseline)

    def extract_prepared(self, row, prepared, baseline=True):
        if not baseline:
            raise ValueError("Never omit the label-token baseline in this probe")
        return self._run(self.backend.extract_prepared, row, prepared, baseline)

    def make_preparer(self):
        return self.backend.make_preparer()


def compare_stores(left, right):
    a, b = left.records(include_metadata=True), right.records(include_metadata=True)
    if [row["id"] for row in a] != [row["id"] for row in b]:
        raise ValueError("Readout/writer changed record order or membership")
    differences = []
    fields = ("query", "candidates", "mask", "targets", "ordinal_values", "baseline")
    for x, y in zip(a, b):
        if json.loads(x["metadata"]) != json.loads(y["metadata"]):
            raise ValueError("Metadata, baseline presence, or execution geometry changed")
        tx, ty = left.tensors(x), right.tensors(y)
        for field in fields:
            if not torch.equal(tx[field], ty[field]):
                delta = (tx[field].float() - ty[field].float()).abs()
                differences.append({"record": x["id"], "tensor": field,
                                    "max_absolute": delta.max().item(),
                                    "rms": delta.square().mean().sqrt().item()})
    return differences


def reset_prefixes(backend, reference):
    try:
        from exllamav3.fuzev import NativeLane
    except ImportError as error:
        raise ImportError("Requires the historical engine module exllamav3/fuzev.py "
                          "from research-snapshot/historical-native-source") from error
    backend.lane.close()
    if len(backend.cache.free_list) != backend.cache.num_slots:
        raise ValueError("Probe reset found an unowned hybrid slot")
    backend.cache.reset_states()
    backend.lane = NativeLane(backend.model, backend.cache, context_limit=262144, chunk_size=8192)
    backend.prefixes = OrderedDict()
    reference.lane, reference.prefixes = backend.lane, backend.prefixes
    backend.preparer = ReadoutPreparer(backend.tokenizer)


def run_arm(name, groups, repeats, backend, owner, output, identity, guard, sampler, shard_size):
    stores = [FeatureStore(output / f"{name}-{repeat}", {**identity, "arm": name, "repeat": repeat},
                           shard_size=shard_size) for repeat in range(repeats)]
    proxy = FeatureProxy(backend, owner)
    timings = {"encode_calls": 0, "encode_seconds": 0., "render_calls": 0, "render_seconds": 0.}
    original_encode = backend.backend.encode
    tokenizer = backend.backend.tokenizer
    original_render = tokenizer.hf_render_chat_template

    def encode(*args, **kwargs):
        start = time.perf_counter()
        result = original_encode(*args, **kwargs)
        timings["encode_calls"] += 1
        timings["encode_seconds"] += time.perf_counter() - start
        return result

    def render(*args, **kwargs):
        start = time.perf_counter()
        result = original_render(*args, **kwargs)
        timings["render_calls"] += 1
        timings["render_seconds"] += time.perf_counter() - start
        return result

    if name == "serialized":
        backend.backend.encode = encode
        tokenizer.hf_render_chat_template = render
    stats, segments = None, []
    before = sampler.sample()
    initialization_seconds = 0.
    try:
        if name == "candidate":
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-prepare-init") as cpu:
                cpu.submit(proxy.make_preparer).result()
            initialization_seconds = time.perf_counter() - start
        started = time.perf_counter()
        for group in groups:
            for repeat, features in enumerate(stores):
                guard.check()
                start = time.perf_counter()
                rows = [item["row"] for item in group]
                if name == "serialized":
                    for row in rows:
                        guard.check()
                        result = proxy.extract(row, baseline=True)
                        guard.check()
                        features.add(row, *result)
                else:
                    stats = write_rows(rows, features, proxy, guard)
                    if stats["records"] != len(rows) or stats["skipped"] or stats["failures"]:
                        raise ValueError("Matched pipeline failed to process every full row")
                segments.append({"track": group[0]["track"], "repeat": repeat,
                                 "records": len(rows), "seconds": time.perf_counter() - start})
                print(json.dumps({"arm": name, **segments[-1]}), flush=True)
        for features in stores:
            features.flush()
        elapsed = time.perf_counter() - started
        after = sampler.sample()
        total = sum(len(group) for group in groups) * repeats
        cpu = timings
        if stats is not None:
            worker, gpu_owner = stats["cpu_preparation"], backend.backend.preparer.stats()
            cpu = {"worker": worker, "gpu_owner": gpu_owner,
                   "encode_calls": worker["tokenizer_calls"] + gpu_owner["tokenizer_calls"],
                   "encode_seconds": worker["tokenizer_seconds"] + gpu_owner["tokenizer_seconds"],
                   "render_calls": worker["template_calls"] + gpu_owner["template_calls"],
                   "render_seconds": worker["template_seconds"] + gpu_owner["template_seconds"]}
        value = {"seconds_including_shard_writes": elapsed, "records": total,
                 "records_per_second": total / elapsed, "segments": segments,
                 "private_tokenizer_initialization_seconds": initialization_seconds,
                 "cold_job_seconds_including_tokenizer_init": elapsed + initialization_seconds,
                 "cold_job_records_per_second": total / (elapsed + initialization_seconds),
                 "execution": backend.trace, "telemetry_before": before, "telemetry_after": after,
                 "memory_guard": guard.snapshot(),
                 "cpu_preparation": cpu}
        write_json(output / (name + ".json"), value)
        return value
    finally:
        if name == "serialized":
            backend.backend.encode = original_encode
            tokenizer.hf_render_chat_template = original_render
        for features in stores:
            features.close()


def run(args):
    if json.loads((ROOT / "data/manifests/acquisition.json").read_text()).get("dataset_scope"):
        raise ValueError("Historical QASPER/A-OKVQA probe is archived under active scope v2")
    if not args.allow_gpu:
        raise ValueError("Operator must pass --allow-gpu after stopping other GPU work")
    groups = select(args.selection, args.rows)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs/native-real").resolve()):
        raise ValueError("Probe outputs must be isolated under runs/native-real, never data/features")
    sources = runtime_sources()
    original, source_sha = original_backend(args.control_source)
    manifest = json.loads((ROOT / "data/manifests/acquisition.json").read_text())
    snapshot = manifest["models"]["native-4.05"].get("local_snapshot")
    if not snapshot:
        raise ValueError("Approved model snapshot is missing")
    output.mkdir(parents=True, exist_ok=False)
    selected = [item for group in groups for item in group]
    write_json(output / "selection.json", {"rows": selected, "source_sha256": hashlib_file(args.selection)})
    budget = {"limit": args.max_forward_tokens, "calls": 0, "forwarded_tokens": 0}
    report = {"status": "running", "sources": sources, "control_source": CONTROL,
              "control_native_py_sha256": source_sha, "pipeline_policy": PIPELINE_POLICY,
              "rows": args.rows, "repeats": args.repeats, "shard_size": args.shard_size,
              "baseline": True, "candidate_batch_size": 4, "quality_evaluation": False,
              "schedule": "stored fitting fixtures, track order; repeat each <=7-row track before next track",
              "budget": budget, "arms": {}, "differences": []}
    fixtures = {item["row"]["id"]: item for item in selected}
    guard, sampler, backend = MemoryGuard(), Sampler(), None
    guard.start()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-probe-gpu") as owner:
            try:
                backend = owner.submit(NativeBackend, snapshot).result()
                report["recorded_native_commit"] = NATIVE
                report["recorded_native_extension_sha256"] = EXTENSION
                if not backend.readout_runtime["optimized_foundation"]:
                    raise ValueError("Positive optimization foundation was not enabled")
                report["readout_runtime"] = backend.readout_runtime
                reference = object.__new__(original)
                reference.__dict__ = backend.__dict__.copy()
                warm = ObservedBackend(reference, fixtures, budget, guard, "warmup")
                # Warm one complete fitting readout per track. Image embeddings
                # remain instance-identical across arms (at most eight images).
                images = {item["row"]["image_sha256"]: item["row"] for item in selected if item["row"].get("image_path")}
                if len(images) > 8:
                    raise ValueError("Tighten --rows to keep image aliases identical within the eight-image LRU")
                warm_start = time.perf_counter()
                for row in images.values():
                    guard.check()
                    owner.submit(backend.image_embedding, row["image_path"], row["image_sha256"]).result()
                for group in groups:
                    owner.submit(warm.extract, group[0]["row"], True).result()
                report["warmup_seconds"] = time.perf_counter() - warm_start
                report["warmup_execution"] = warm.trace
                identity = {"probe_only": True, "full_corpus": False, "selection_sha256": hashlib_file(args.selection),
                            "sources": sources, "control_source": CONTROL, "readout_runtime": backend.readout_runtime}
                for name, actual in (("serialized", reference), ("candidate", backend)):
                    owner.submit(reset_prefixes, backend, reference).result()
                    observed = ObservedBackend(actual, fixtures, budget, guard, name)
                    report["arms"][name] = run_arm(name, groups, args.repeats, observed, owner,
                        output, identity, guard, sampler, args.shard_size)
                    write_json(output / "report.json", report)
                if report["arms"]["serialized"]["execution"] != report["arms"]["candidate"]["execution"]:
                    raise ValueError("Forward order, token hashes, counts, prefix reuse, or B4 grouping changed")
                for repeat in range(args.repeats):
                    stores = []
                    try:
                        for name in ("serialized", "candidate"):
                            path = output / f"{name}-{repeat}"
                            stores.append(FeatureStore(path, json.loads((path / "identity.json").read_text()), read_only=True))
                        report["differences"].extend(compare_stores(*stores))
                    finally:
                        for store in stores:
                            store.close()
                report["speedup"] = report["arms"]["candidate"]["records_per_second"] / report["arms"]["serialized"]["records_per_second"]
                if report["differences"]:
                    raise ValueError("Same-geometry features or baseline logits were not exactly identical")
                report["status"] = "exact_fit_probe_pass_not_task_quality_qualification"
                report["actual_batches"] = dict(Counter(size for item in report["arms"]["candidate"]["execution"]
                                                        for size in item["candidate_forward_batches"]))
                if runtime_sources() != sources:
                    raise ValueError("Sources changed during qualification")
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
    print(json.dumps({"status": report["status"], "speedup": report["speedup"], "output": str(output)}), flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--selection", type=Path, default=ROOT / "runs/native-real/stock-distribution-20261001/selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True, help="Unpacked historical control source directory")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument("--shard-size", type=int, choices=(16, 128), default=16)
    parser.add_argument("--max-forward-tokens", type=int, default=1000000)
    args = parser.parse_args()
    if not 10 <= args.rows <= 32 or not 0 < args.max_forward_tokens <= 1000000:
        parser.error("Use 10..32 complete fitting rows and a positive forward budget <=1M")
    return args


if __name__ == "__main__":
    run(arguments())
