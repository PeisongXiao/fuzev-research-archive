"""Fresh CUDA fitting and calibration; never scores heldout/external records."""

import hashlib
import json
from pathlib import Path
import time

import torch

from .acquire import write_json
from .features import FeatureStore
from .paths import project_path
from .train import calibrate, fit, head_output_path


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh retraining requires CUDA; CPU fallback is forbidden")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = head_output_path("cuda-retrain-20261005")
    output.mkdir(exist_ok=False)
    directory = project_path("data/features/native-4.05-scope-v2-image-dimensions")
    identity_bytes = (directory / "identity.json").read_bytes()
    identity_sha = hashlib.sha256(identity_bytes).hexdigest()
    if identity_sha != "eeda1ef88e7c8e2b2a5fa27d9b2e74bca853d55e6d7e7bd050ac6addcd698167":
        raise ValueError("Frozen native4 feature identity changed")
    receipt = {
        "status": "in_progress", "started_at": time.time(),
        "device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "precision": "float32", "allow_tf32": False,
        "from_scratch": True, "old_reference_reused": False,
        "feature_identity_sha256": identity_sha,
        "output_directory": str(output),
        "selection_role": "development", "calibration_role": "calibration",
        "external_selection": False,
        "validation_status": "pending; no completion claim until validation passes",
    }
    write_json(output / "run.json", receipt)
    features = FeatureStore(directory, json.loads(identity_bytes), read_only=True)
    try:
        text = fit(features, "text", device="cuda:0", output_directory=output)
        mixed = fit(features, "mixed", initial=text, device="cuda:0", output_directory=output)
        reference = calibrate(features, mixed, device="cuda:0", output_directory=output)
        receipt.update(status="trained_and_calibrated", finished_at=time.time(),
                       text_checkpoint=text, mixed_checkpoint=mixed, reference=reference)
        write_json(output / "run.json", receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        features.close()


if __name__ == "__main__":
    main()
