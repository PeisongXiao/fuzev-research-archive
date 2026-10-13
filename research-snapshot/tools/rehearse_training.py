"""Isolated CPU synthetic lifecycle rehearsal, never a production training recipe."""

import contextlib
import importlib.util
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

WORKTREE = Path(__file__).resolve().parents[1]
PRIMITIVES = ("choice", "noul", "ordinal")
PROFILES = list(itertools.product((17, 29, 43), (3e-4, 1e-3), (0., .1)))


def cpu_setup():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES= before starting the rehearsal")
    root = Path(os.environ["FUZEV_ROOT"]).expanduser().resolve()
    if not root.is_relative_to(WORKTREE / ".pytest_cache"):
        raise RuntimeError("FUZEV_ROOT must be below this worktree's .pytest_cache")
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("The rehearsal must not have GPU access")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return torch


@contextlib.contextmanager
def isolated(root):
    root = Path(root).resolve()
    if not root.is_relative_to(WORKTREE / ".pytest_cache"):
        raise ValueError("Synthetic artifacts must remain inside this worktree")
    from fuzev_research import paths
    from fuzev_research.state import Store
    with patch.object(paths, "ROOT", root), patch.dict(os.environ, FUZEV_ROOT=str(root)):
        Store().initialize()
        yield


def build_corpus(root):
    """Write full synthetic manifests and real feature shards, without mocking readiness."""
    import torch
    from fuzev_research.acquire import write_json
    from fuzev_research.features import FeatureStore, file_sha256

    identity = {"synthetic_rehearsal_only": True,
                "datasets": dict.fromkeys(("text", "qasper", "images"), "synthetic-v1")}
    generator = torch.Generator().manual_seed(7103)
    with contextlib.closing(FeatureStore(root / "features", identity, shard_size=4)) as store:
        for source, primitives in (("text", ("choice", "ordinal")),
                                   ("qasper", ("noul",)), ("images", PRIMITIVES)):
            rows, counts = [], {}
            for primitive, role, variant in itertools.product(
                    primitives, ("fit", "development", "calibration", "test"), (0, 1)):
                count = (3 if primitive == "ordinal" else 2) + variant
                key = f"synthetic-{source}-{primitive}-{role}-{variant}"
                candidates = [{"id": str(i), "text": f"synthetic candidate {i}",
                               **({"ordinal_value": float(i)} if primitive == "ordinal" else {})}
                              for i in range(count)]
                row = {"id": key, "group": key, "role": role, "primitive": primitive,
                       "modality": "image" if source == "images" else "text",
                       "source": source, "revision": "synthetic-v1", "candidates": candidates,
                       "target": [1. if i == variant else 0. for i in range(count)]}
                rows.append(row)
                track = f"{source}/{primitive}/{role}"
                counts[track] = counts.get(track, 0) + 1
                store.add(row, torch.randn(2560, generator=generator),
                          torch.randn(count, 2560, generator=generator))
            path = root / "data" / "prepared" / f"{source}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            digest = file_sha256(path)
            name = "image" if source == "images" else source
            write_json(root / f"data/manifests/{name}-splits.json",
                       {"path": str(path), "sha256": digest, "records": len(rows), "counts": counts})
            store.register_preparation(path, digest)
            store.finish_preparation(path, len(rows))
        assert store.require_ready("text")["records"] == 24
        assert store.require_ready("mixed")["records"] == 48
    return identity


def module_runner():
    spec = importlib.util.spec_from_file_location("rehearsal_orchestrator", WORKTREE / "tools/train_when_ready.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class SyntheticGuard:
    """No live resource sampling: only synthetic correctness, not memory admission qualification."""
    def start(self):
        return self

    def check(self):
        pass

    def close(self):
        pass


def lifecycle(root):
    import torch
    from fuzev_research import train
    from fuzev_research.features import FeatureStore, collate, file_sha256
    from fuzev_research.head import load_bundle

    identity = build_corpus(root)
    feature_hashes = {path.name: file_sha256(path)
                      for path in (root / "features").glob("*.safetensors")}
    runner = module_runner()
    trial = train.train_trial
    original_collate = train.collate
    original_fit = runner.fit
    original_calibrate = runner.calibrate
    phase, reads, profiles = [None], [], []

    def audited_collate(features, rows, device="cpu"):
        roles = {row["role"] for row in rows}
        assert "test" not in roles, "Final-test rows must never be selected for fitting or calibration"
        allowed = {"calibration"} if phase[0] == "calibrate" else {"fit", "development"}
        assert roles <= allowed
        if phase[0] == "text":
            assert all(row["modality"] == "text" for row in rows)
        reads.extend((phase[0], row["role"], row["id"]) for row in rows)
        return original_collate(features, rows, device)

    def one_epoch(*args, **kwargs):
        bound = inspect.signature(trial).bind(*args, **kwargs)
        bound.apply_defaults()
        assert bound.arguments["max_epochs"] == 10, "Production caller schedule unexpectedly changed"
        profiles.append((bound.arguments["stage"], bound.arguments["seed"],
                         bound.arguments["learning_rate"], bound.arguments["brier_weight"]))
        kwargs["max_epochs"] = 1  # Explicit synthetic-only runtime override; never edit train.py.
        return trial(*args, **kwargs)

    def fitting(features, stage, initial=None):
        phase[0] = stage
        return original_fit(features, stage, initial)

    def calibrating(features, directory):
        phase[0] = "calibrate"
        return original_calibrate(features, directory)

    with (patch.object(train, "train_trial", one_epoch),
          patch.object(train, "collate", audited_collate),
          patch.object(runner, "fit", fitting), patch.object(runner, "calibrate", calibrating),
          patch.object(runner, "MemoryGuard", SyntheticGuard), patch.object(runner, "admit"),
          patch.object(runner.time, "sleep", side_effect=AssertionError("Synthetic corpus is already complete")),
          patch.object(sys, "argv", ["train_when_ready.py", "--features", str(root / "features"), "--threads", "1"]),
          patch.object(torch, "set_num_interop_threads") as interop):
        runner.main()
        interop.assert_called_once_with(1)
    assert profiles == [(stage, *profile) for stage in ("text", "mixed") for profile in PROFILES]
    progress = json.loads((root / "runs/head-training.json").read_text())
    assert progress["status"] == "completed"
    expected_roles = {("text", "fit"), ("text", "development"), ("mixed", "fit"),
                      ("mixed", "development"), ("calibrate", "calibration")}
    assert {(phase, role) for phase, role, _ in reads} == expected_roles
    selections = {}
    for stage in ("text", "mixed"):
        selection = json.loads((root / f"weights/{stage}/selection.json").read_text())
        assert len(selection["trials"]) == 12
        assert selection["selection_role"] == "development"
        assert selection["selected"]["best_loss"] == min(x["best_loss"] for x in selection["trials"])
        for entry in selection["trials"]:
            provenance = entry["provenance"]
            assert provenance["schedule"] == {"batch_size": 256, "mixed": stage == "mixed",
                "max_epochs": 1, "epoch_seed": "seed+epoch", "patience": 2, "minimum_improvement": 1e-4}
            assert provenance["fit_membership"]["records"] == (6 if stage == "text" else 12)
            assert provenance["development_membership"]["records"] == (6 if stage == "text" else 12)
            expected_initial = file_sha256(Path(progress["text_checkpoint"]) / "head.safetensors") if stage == "mixed" else None
            assert provenance["initial_sha256"] == expected_initial
        selections[stage] = selection["selected"]["provenance"]
    reference, manifest, temperatures = load_bundle(progress["reference"])
    mixed = load_bundle(progress["mixed_checkpoint"])[0]
    assert sum(p.numel() for p in reference.parameters()) == 3933696
    assert all(torch.equal(v, mixed.state_dict()[k]) for k, v in reference.state_dict().items())
    assert manifest["calibration_role"] == "calibration"
    assert manifest["calibration_membership"]["records"] == 12
    assert all(.029 < value <= 100.01 for value in temperatures.values())
    calibration_diagnostics = {}
    with contextlib.closing(FeatureStore(root / "features", identity, read_only=True)) as features:
        assert manifest["calibration_membership"] == train.membership_identity(features.records("calibration"))
        for primitive in PRIMITIVES:
            rows = features.records("calibration", primitive=primitive)
            batch = collate(features, rows)
            with torch.inference_mode():
                logits = reference(primitive, batch["query"], batch["candidates"], batch["mask"])
                scaled = logits / temperatures[primitive]
            assert torch.equal(logits.argmax(1), scaled.argmax(1))
            assert torch.isfinite(scaled[batch["mask"]]).all()
            def nll(value):
                logp = value.log_softmax(1).masked_fill(~batch["mask"], 0)
                return -(batch["targets"] * logp).sum(1).mean().item()
            calibration_diagnostics[primitive] = {"nll_before": nll(logits), "nll_after": nll(scaled)}
        for stage in selections:
            modality = "text" if stage == "text" else None
            for role, key in (("fit", "fit_membership"), ("development", "development_membership")):
                assert selections[stage][key] == train.membership_identity(features.records(role, modality))
    assert feature_hashes == {path.name: file_sha256(path)
                              for path in (root / "features").glob("*.safetensors")}
    return {"profiles_per_stage": 12, "synthetic_max_epochs": 1, "hidden_size": 2560,
            "head_parameters": 3933696, "rows": 48, "tensor_read_roles": sorted(expected_roles),
            "test_tensor_row_selections": 0, "temperatures": temperatures,
            "calibration_diagnostics_synthetic_only": calibration_diagnostics,
            "frozen_feature_shards_byte_identical": True,
            "calibration_preserved_head_tensors_and_argmax": True}


class AfterCheckpoint(RuntimeError):
    pass


def resume_comparison(root):
    import torch
    from safetensors.torch import load_file
    from fuzev_research import train
    from fuzev_research.features import FeatureStore
    from fuzev_research.state import Store

    identity = build_corpus(root)
    original_record = Store.record_result
    original_restore = train.restore_optimizer
    restored = []

    def crash_after_committed_epoch(self, task, result):
        original_record(self, task, result)
        if result.get("epoch") == 1:
            raise AfterCheckpoint("Synthetic interruption after durable epoch and optimizer commit")

    def audited_restore(directory, optimizer, device):
        original_restore(directory, optimizer, device)
        steps = [state["step"].item() for state in optimizer.state.values()]
        assert len(steps) == 12 and set(steps) == {1.}
        restored.append(steps)

    with contextlib.closing(FeatureStore(root / "features", identity, read_only=True)) as features:
        args = (features, features.records("fit", "text"), features.records("development", "text"),
                "text", 17, 3e-4, .1)
        with patch.object(train, "project_path", lambda name: root / "control" / name):
            control = train.train_trial(*args, max_epochs=2)
        with patch.object(train, "project_path", lambda name: root / "resumed" / name):
            try:
                with patch.object(Store, "record_result", crash_after_committed_epoch):
                    train.train_trial(*args, max_epochs=2)
            except AfterCheckpoint:
                pass
            else:
                raise AssertionError("Synthetic interruption was not reached")
            with patch.object(train, "restore_optimizer", audited_restore):
                resumed = train.train_trial(*args, max_epochs=2)
    assert len(restored) == 1
    assert control["best_loss"] == resumed["best_loss"]
    for expected, actual in zip(control["epochs"], resumed["epochs"], strict=True):
        assert {k: v for k, v in expected.items() if k != "seconds"} == {k: v for k, v in actual.items() if k != "seconds"}
    for name in ("head.safetensors", "optimizer.safetensors"):
        expected = load_file(str(Path(control["latest_checkpoint"]) / name))
        actual = load_file(str(Path(resumed["latest_checkpoint"]) / name))
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[k], actual[k]) for k in expected), name
    return {"synthetic_max_epochs": 2, "interrupted_after_epoch": 1,
            "resumed_epoch": 2, "restored_parameter_states": 12,
            "head_and_optimizer_tensors_exact": True, "loss_and_development_results_exact": True}


def rehearse(root):
    torch = cpu_setup()
    from fuzev_research import train
    assert Path(train.__file__).resolve() == WORKTREE / "fuzev_research/train.py"
    started = time.monotonic()
    source_identity = train.training_code_identity()
    with isolated(root / "lifecycle"), contextlib.redirect_stdout(io.StringIO()):
        lifecycle_result = lifecycle(root / "lifecycle")
    with isolated(root / "resume"), contextlib.redirect_stdout(io.StringIO()):
        resume_result = resume_comparison(root / "resume")
    assert source_identity == train.training_code_identity()
    assert not any(name == "fuzev_research.native" or name.startswith(("exllamav3", "vllm")) for name in sys.modules)
    artifact_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert artifact_bytes < 2 << 30
    return {"scope": "CPU synthetic rehearsal only; NOT trained project artifacts or quality evidence",
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "production_default_max_epochs": inspect.signature(train.train_trial).parameters["max_epochs"].default,
            "frozen_module_sha256": source_identity, "lifecycle": lifecycle_result,
            "resume": resume_result, "temporary_artifact_bytes": artifact_bytes,
            "elapsed_seconds": time.monotonic() - started}


def main():
    cpu_setup()
    (WORKTREE / ".pytest_cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="training-rehearsal-", dir=WORKTREE / ".pytest_cache") as temporary:
        result = rehearse(Path(temporary))
    result["temporary_artifacts_removed"] = not Path(temporary).exists()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
