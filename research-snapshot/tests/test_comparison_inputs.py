from copy import deepcopy
import json

import pytest
torch = pytest.importorskip("torch")

from test_evaluate import fixture
from fuzev_research.comparison_inputs import INPUTS, MANIFEST, iter_inputs, prepare_inputs
from fuzev_research.features import file_sha256


def expected_lines(fixture):
    selected = []
    for prepared in fixture.reader.require_ready("calibrate")["preparations"]:
        with open(prepared["path"], "rb") as handle:
            for raw in handle:
                if json.loads(raw)["role"] == "test":
                    selected.append(raw)
    return b"".join(selected)


def test_complete_inputs_preserve_bytes_order_and_every_canonical_test(fixture):
    directory = fixture.root / "comparison-inputs"
    spec = fixture.plan["reference"]
    receipt = prepare_inputs(spec, directory)
    assert (directory / INPUTS).read_bytes() == expected_lines(fixture)
    assert receipt["inputs"]["records"] == 8
    assert receipt["reference"]["test_membership"]["records"] == 8
    rows = list(iter_inputs(directory, file_sha256(directory / MANIFEST), spec))
    assert len(rows) == 8 and all(row["role"] == "test" for row in rows)
    assert {row["modality"] for row in rows} == {"text", "image"}
    assert len({row["id"] for row in rows}) == 8
    assert fixture.hashes == {name: file_sha256(fixture.head / name) for name in fixture.hashes}
    assert len(list(fixture.reader.iter_records())) == 32


def test_existing_inputs_are_never_overwritten(fixture):
    directory = fixture.root / "comparison-inputs"
    directory.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_inputs(fixture.plan["reference"], directory)


def test_frozen_head_is_required_before_prepared_tests_are_read(fixture):
    spec = deepcopy(fixture.plan["reference"])
    spec["sha256"]["head.safetensors"] = "0" * 64
    directory = fixture.root / "comparison-inputs"
    with pytest.raises(ValueError, match="checksum"):
        prepare_inputs(spec, directory)
    assert not directory.exists()


def test_calibration_not_a_training_checkpoint_is_required(fixture):
    path = fixture.head / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_kind"] = "training-checkpoint"
    path.write_text(json.dumps(manifest))
    spec = deepcopy(fixture.plan["reference"])
    spec["sha256"]["manifest.json"] = file_sha256(path)
    directory = fixture.root / "comparison-inputs"
    with pytest.raises(ValueError, match="calibrated reference"):
        prepare_inputs(spec, directory)
    assert not directory.exists()


def test_head_change_before_commit_does_not_expose_inputs(fixture, monkeypatch):
    import fuzev_research.comparison_inputs as module
    original = module.write_json

    def changed(path, value):
        original(path, value)
        with (fixture.head / "calibration.json").open("a") as handle:
            handle.write(" ")

    monkeypatch.setattr(module, "write_json", changed)
    directory = fixture.root / "comparison-inputs"
    with pytest.raises(ValueError, match="checksum"):
        prepare_inputs(fixture.plan["reference"], directory)
    assert not directory.exists()
    assert not list(fixture.root.glob(".comparison-inputs-*"))


@pytest.mark.parametrize("mutation", ["file", "manifest", "reference", "missing", "target", "duplicate"])
def test_consumer_rejects_modified_or_noncanonical_inputs(fixture, mutation):
    directory = fixture.root / "comparison-inputs"
    spec = fixture.plan["reference"]
    prepare_inputs(spec, directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    pin = file_sha256(directory / MANIFEST)
    if mutation == "file":
        with (directory / INPUTS).open("ab") as handle:
            handle.write(b"\n")
    elif mutation == "manifest":
        manifest["roles"] = ["fit", "test"]
        (directory / MANIFEST).write_text(json.dumps(manifest))
    elif mutation == "reference":
        manifest["reference"]["head_sha256"]["head.safetensors"] = "0" * 64
    else:
        rows = [json.loads(raw) for raw in (directory / INPUTS).read_bytes().splitlines()]
        if mutation == "missing":
            rows.pop()
        elif mutation == "target":
            rows[0]["target"] = [0., 1.]
        else:
            rows[-1] = rows[0]
        (directory / INPUTS).write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["inputs"].update(sha256=file_sha256(directory / INPUTS),
                                  bytes=(directory / INPUTS).stat().st_size)
    if mutation not in {"file", "manifest"}:
        (directory / MANIFEST).write_text(json.dumps(manifest))
        pin = file_sha256(directory / MANIFEST)
    with pytest.raises(ValueError, match="checksum|reference|incomplete|semantics|duplicate"):
        list(iter_inputs(directory, pin, spec))
