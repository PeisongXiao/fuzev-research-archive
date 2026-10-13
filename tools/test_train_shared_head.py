"""One tiny CPU fit/calibrate/export/load run using the public training commands."""

import json
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file

from fuzev.head import ServingHead
from train_shared_head import initialize


def test_train_export_load_cpu(tmp_path):
    script = Path(__file__).with_name("train_shared_head.py")
    fixture = tmp_path / "features"
    output = tmp_path / "head"
    subprocess.run([sys.executable, str(script), "fixture", "--output", str(fixture),
                    "--hidden-size", "32"], check=True, capture_output=True, text=True)
    command = [sys.executable, str(script), "train",
               "--backbone-config", str(fixture / "backbone-config.json"),
               "--fit", str(fixture / "fit.jsonl"), "--calibration", str(fixture / "calibration.jsonl"),
               "--output", str(output), "--projection-size", "11", "--query-adapter-size", "5",
               "--key-adapter-size", "3", "--max-options", "6",
               "--epochs", "2", "--batch-size", "3", "--learning-rate", "0.01",
               "--calibration-steps", "5", "--release-date", "2026-10-12", "--device", "cpu", "--inspect"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert (result["fit_rows"], result["calibration_rows"], result["updates"]) == (6, 3, 4)
    assert set(result["examples"]) == {"choice", "noul", "ordinal"}
    state = load_file(output / "head.safetensors")
    initial = initialize(dict(hidden_size=32, projection_size=11, query_adapter_size=5, key_adapter_size=3), seed=0)
    assert any(not torch.equal(value, initial.state_dict()[name]) for name, value in state.items())
    with ServingHead(output, device="cpu", calibration="calibrated") as head:
        assert head.metadata["hidden_size"] == 32
        assert head.max_options == 6
        assert head.encoding_version == "synthetic-branches-v1"
        for line in (fixture / "calibration.jsonl").read_text().splitlines():
            row = json.loads(line)
            scores, raw, calibrated = head.score(
                row["primitive"], torch.tensor(row["query"]), torch.tensor(row["candidates"]),
                baseline=torch.tensor(row["baseline"]),
            )
            assert torch.isfinite(torch.tensor(scores)).all()
            for probabilities in (raw, calibrated):
                assert abs(sum(probabilities) - 1) < 1e-6
            temperature = result["temperatures"][row["primitive"]]
            torch.testing.assert_close(torch.tensor(calibrated), (torch.tensor(scores) / temperature).softmax(0))
            torch.testing.assert_close(torch.tensor(scores), torch.tensor(result["examples"][row["primitive"]]["logits"]))
    print(completed.stdout)
