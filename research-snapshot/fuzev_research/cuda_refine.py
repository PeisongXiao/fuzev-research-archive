"""One predeclared lower-LR CUDA repeat, selected on development only."""

import hashlib
import json
from pathlib import Path

import torch

from .acquire import write_json
from .features import FeatureStore
from .head import load_bundle
from .paths import project_path
from .train import calibrate, membership_identity, train_trial, validate


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA quality refinement requires a real GPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = project_path("weights/cuda-retrain-20261005")
    run = json.loads((original / "run.json").read_text())
    if run["status"] != "trained_and_calibrated":
        raise ValueError("The original CUDA run must complete before a serial repeat")
    text = json.loads((original / "text/selection.json").read_text())["selected"]
    mixed = json.loads((original / "mixed/selection.json").read_text())["selected"]
    output = project_path("weights/cuda-refine-20261005")
    output.mkdir(exist_ok=False)
    directory = project_path("data/features/native-4.05-scope-v2-image-dimensions")
    identity_bytes = (directory / "identity.json").read_bytes()
    if hashlib.sha256(identity_bytes).hexdigest() != run["feature_identity_sha256"]:
        raise ValueError("Original native4 feature identity changed")
    features = FeatureStore(directory, json.loads(identity_bytes), read_only=True)
    try:
        fitting = features.records("fit")
        development = features.records("development")
        bundle = load_bundle(text["directory"])[0].to("cuda:0")
        warmup_control = validate(bundle, features, development, device="cuda:0")
        del bundle
        torch.cuda.empty_cache()
        write_json(output / "development-control.json", {
            "input_role": "development", "selection_role": "development",
            "membership": membership_identity(development),
            "warmup_on_same_mixed_development": warmup_control,
            "predeclared_repeat": {"seed": 29, "learning_rate": .0001, "brier_weight": 0.,
                                   "max_epochs": 10, "initial": text["directory"]},
            "reason": "Per-track mixed-stage degradation observed on development, not heldout outputs",
        })
        repeat = train_trial(features, fitting, development, "mixed", 29, .0001, 0.,
                             initial=text["directory"], device="cuda:0",
                             output_directory=output)
        selected = min((mixed, repeat), key=lambda value: value["best_loss"])
        write_json(output / "selection.json", {
            "selection_role": "development", "selected": selected,
            "controls": [{"directory": value["directory"], "best_loss": value["best_loss"]}
                         for value in (mixed, repeat)],
            "external_selection": False,
        })
        reference = calibrate(features, selected["directory"], device="cuda:0", output_directory=output)
        print(json.dumps({"status": "refined_and_calibrated", "reference": reference,
                          "selected": selected["directory"], "best_development_nll": selected["best_loss"],
                          "validation": "pending native4 quality and real API E2E; not qualified"}), flush=True)
    finally:
        features.close()


if __name__ == "__main__":
    main()
