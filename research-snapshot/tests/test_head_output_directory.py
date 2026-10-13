from pathlib import Path

import pytest

from fuzev_research import train


@pytest.fixture
def weights(tmp_path, monkeypatch):
    root = tmp_path / "weights"
    root.mkdir()
    monkeypatch.setattr(train, "project_path", lambda relative: tmp_path / relative)
    return root


def test_default_preserves_existing_output_location(weights):
    assert train.head_output_path("reference") == weights / "reference"


def test_fresh_cuda_run_does_not_reuse_old_trial_paths(weights):
    run = weights / "cuda-retrain"
    assert train.head_output_path("text/trials/seed-17", run) == run / "text/trials/seed-17"
    assert train.head_output_path("reference", run) != train.head_output_path("reference")


def test_outside_weights_is_rejected(weights):
    with pytest.raises(ValueError, match="project weights"):
        train.head_output_path("reference", weights.parent / "other")


def test_relative_escape_is_rejected(weights):
    with pytest.raises(ValueError, match="escapes"):
        train.head_output_path("../reference", weights / "cuda-retrain")


def test_symlink_escape_is_rejected(weights, tmp_path):
    (weights / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="project weights"):
        train.head_output_path("reference", weights / "escape")


def test_cli_accepts_explicit_cuda_output_directory():
    args = train.argument_parser().parse_args([
        "--features", "/features", "--stage", "text", "--device", "cuda",
        "--output-directory", "/project/weights/cuda-retrain",
    ])
    assert args.device == "cuda"
    assert args.initial is None
    assert args.output_directory == Path("/project/weights/cuda-retrain")
