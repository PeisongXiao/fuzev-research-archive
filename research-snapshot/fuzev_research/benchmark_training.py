"""Synthetic CPU-only microbenchmark: python -m fuzev_research.benchmark_training.

No production corpus, backbone, weights, selection, or calibration is touched.
Temporary role-sparse shards deliberately exceed the four-shard cache. Results
are warm-filesystem microbenchmarks, not measured full-training throughput.
"""

import argparse
import json
import random
import resource
import statistics
import tempfile
import time
from collections import OrderedDict
from pathlib import Path


class OriginalD260024Reader:
    """Original fixed-four-shard LRU; intentionally no byte-budget eviction."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.loaded = OrderedDict()
        self.full_shard_loads = self.loaded_bytes = 0

    def tensors(self, row):
        from safetensors.torch import load_file

        shard = row["shard"]
        if shard not in self.loaded:
            self.loaded[shard] = load_file(str(self.directory / f"shard-{shard:06d}.safetensors"))
            self.full_shard_loads += 1
        self.loaded.move_to_end(shard)
        while len(self.loaded) > 4:
            self.loaded.popitem(last=False)
        return {name: tensor[row["row_"]] for name, tensor in self.loaded[shard].items()}


def original_d260024_collate(store, rows, device="cpu"):
    """Verbatim d260024 collation, with its local Torch import added."""
    import torch

    samples = [store.tensors(row) for row in rows]
    width = max(len(sample["mask"]) for sample in samples)
    tensors = {}
    for name in samples[0]:
        if name == "query":
            tensors[name] = torch.stack([sample[name] for sample in samples]).to(device)
        else:
            expanded = []
            for sample in samples:
                value = sample[name]
                padded = torch.zeros((width, *value.shape[1:]), dtype=value.dtype)
                padded[:len(value)] = value
                expanded.append(padded)
            tensors[name] = torch.stack(expanded).to(device)
    return tensors


def head_loss(head, value):
    logits = head(value["query"], value["candidates"], value["mask"])
    logp = logits.log_softmax(dim=1)
    nll = -(value["targets"] * logp.masked_fill(~value["mask"], 0)).sum(dim=1)
    loss = (nll / value["mask"].sum(dim=1).float().log()).mean()
    return loss + .1 * ((logp.exp() - value["targets"]) ** 2).sum(dim=1).mean(), logits


def check_equivalence(original, current, head):
    import torch

    width = current["mask"].shape[1]
    assert original.keys() == current.keys()
    assert not original["mask"][:, width:].any()
    for name, value in current.items():
        reference = original[name] if name == "query" else original[name][:, :width]
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    old_loss, old_logits = head_loss(head, original)
    new_loss, new_logits = head_loss(head, current)
    old_valid, new_valid = old_logits[:, :width][current["mask"]], new_logits[current["mask"]]
    torch.testing.assert_close(old_valid, new_valid)
    torch.testing.assert_close(old_loss, new_loss)
    parameters = tuple(head.parameters())
    old_gradients = torch.autograd.grad(old_loss, parameters)
    new_gradients = torch.autograd.grad(new_loss, parameters)
    for old, new in zip(old_gradients, new_gradients):
        torch.testing.assert_close(old, new)
    return {"output_slice_exact": True, "valid_logits_and_loss_close": True,
            "parameter_gradients_close": True,
            "max_valid_logit_difference": (old_valid - new_valid).abs().max().item(),
            "max_parameter_gradient_difference": max((old - new).abs().max().item()
                                                     for old, new in zip(old_gradients, new_gradients))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.iterations <= 100:
        parser.error("Use positive threads and 1-100 iterations")
    import torch
    from .features import FeatureStore, collate
    from .head import CandidateHead

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)

    class CountingStore(FeatureStore):
        full_shard_loads = 0

        def shard_tensors(self, shard):
            self.full_shard_loads += int(shard not in self.loaded)
            return super().shard_tensors(shard)

    class BoundedRowReader:
        def __init__(self, store):
            self.tensors = store.tensors

    def measure(operation, store):
        operation()
        times, loads = [], []
        for _ in range(args.iterations):
            store.loaded.clear()
            store.loaded_bytes = store.full_shard_loads = 0
            started = time.perf_counter()
            operation()
            times.append(time.perf_counter() - started)
            loads.append(store.full_shard_loads)
        return {"median_seconds": statistics.median(times), "min_seconds": min(times),
                "median_full_shard_loads": statistics.median(loads)}

    results = {}
    for candidates in (2, 5):
        with tempfile.TemporaryDirectory(prefix="fuzev-cpu-benchmark-") as temporary:
            identity = {"synthetic_benchmark_only": True}
            writer = FeatureStore(Path(temporary), identity, shard_size=8)
            for index in range(512):
                selected = index % 8 < 4
                count = candidates if selected else 16
                row = {"id": str(index), "group": str(index), "role": "fit" if selected else "unused",
                       "primitive": "choice", "modality": "text", "source": "synthetic-benchmark",
                       "revision": "fixture", "candidates": [{"id": str(i)} for i in range(count)],
                       "target": [1.] + [0.] * (count - 1)}
                writer.add(row, torch.randn(2560), torch.randn(count, 2560))
            writer.close()
            store = CountingStore(temporary, identity, read_only=True)
            try:
                rows = store.records("fit")
                random.Random(17).shuffle(rows)
                bounded_reader = BoundedRowReader(store)
                original_reader = OriginalD260024Reader(temporary)
                reference = collate(bounded_reader, rows)
                current = collate(store, rows)
                for name in reference:
                    torch.testing.assert_close(current[name], reference[name], rtol=0, atol=0)
                del reference
                original_batch = original_d260024_collate(original_reader, rows)
                head = CandidateHead()
                equivalence = check_equivalence(original_batch, current, head)
                original_width = original_batch["mask"].shape[1]
                del original_batch, current
                original = measure(lambda: original_d260024_collate(original_reader, rows), original_reader)
                bounded = measure(lambda: collate(bounded_reader, rows), store)
                new = measure(lambda: collate(store, rows), store)
                batch = collate(store, rows)
                initial_state = {name: value.detach().clone() for name, value in head.state_dict().items()}

                def measure_fitting(batch_factory, reader):
                    head.load_state_dict(initial_state)
                    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01)

                    def step():
                        loss, _ = head_loss(head, batch_factory())
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                        optimizer.step()

                    return measure(step, reader)

                fit = measure_fitting(lambda: batch, store)
                combined = measure_fitting(lambda: collate(store, rows), store)
                original_combined = measure_fitting(lambda: original_d260024_collate(original_reader, rows),
                                                    original_reader)
                results[str(candidates)] = {"batch_records": len(rows), "original_d260024": original,
                    "bounded_row_reader": bounded, "original_candidate_width": original_width,
                    "cropped_candidate_width": batch["mask"].shape[1], "equivalence": equivalence,
                    "collate": new, "head_step": fit, "collate_and_step": combined,
                    "original_d260024_collate_and_step": original_combined,
                    "collate_speedup_vs_original_d260024": original["median_seconds"] / new["median_seconds"],
                    "collate_speedup_vs_bounded_row_reader": bounded["median_seconds"] / new["median_seconds"],
                    "collate_and_step_speedup_vs_original_d260024":
                        original_combined["median_seconds"] / combined["median_seconds"]}
                original_reader.loaded.clear()
                del batch, head, initial_state
            finally:
                store.close()
    seconds_two = results["2"]["collate_and_step"]["median_seconds"]
    seconds_five = results["5"]["collate_and_step"]["median_seconds"]
    from .paths import project_path
    from .scope import PREPARATIONS, required_preparations

    counts = {}
    for filename in required_preparations("text"):
        manifest = json.loads(project_path("data/manifests/" + PREPARATIONS[filename][0]).read_text())
        for track, value in manifest["counts"].items():
            parts = track.split("/")
            if len(parts) == 3 and parts[-1] == "fit":
                if parts[0] not in PREPARATIONS[filename][3]:
                    raise ValueError("Inactive source in benchmark training-count projection")
                counts[parts[1]] = counts.get(parts[1], 0) + value
    two_batches = sum((counts.get(primitive, 0) + 255) // 256 for primitive in ("choice", "noul"))
    five_batches = (counts.get("ordinal", 0) + 255) // 256
    epoch = two_batches * seconds_two + five_batches * seconds_five
    original_epoch = two_batches * results["2"]["original_d260024_collate_and_step"]["median_seconds"]
    original_epoch += five_batches * results["5"]["original_d260024_collate_and_step"]["median_seconds"]
    print(json.dumps({"kind": "synthetic-cpu-microbenchmark", "threads": args.threads,
        "baseline_definitions": {
            "original_d260024": "Original pad-to-shard-width then stack; fixed-four-shard LRU without byte bound",
            "bounded_row_reader": "aaa45ed-style two-pass row access through the current bounded cache, not d260024",
            "collate": "Current mask-only width scan and shard-grouped, byte-bounded collation"},
        "torch": torch.__version__, "iterations": args.iterations, "results": results,
        "text_fit_records": sum(counts.values()), "text_fit_counts": counts,
        "profiles": 12, "maximum_epochs": 10,
        "projected_text_fit_hours": epoch * 12 * 10 / 3600,
        "projected_original_d260024_text_fit_hours": original_epoch * 12 * 10 / 3600,
        "projected_first_epoch_overlap_ceiling_hours": epoch * 12 / 3600,
        "projection_excludes": ["development", "checkpoint_io", "readiness", "mixed_stage", "cold_storage"],
        "peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_rss_scope": "Whole process, including all baselines and equivalence checks"}, indent=2))


if __name__ == "__main__":
    main()
