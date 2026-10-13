import pytest
torch = pytest.importorskip("torch")

from test_evaluate import fixture
from test_evaluation_subsets import make_subset
from fuzev_research.comparison_seal import seal_features
from fuzev_research.evaluate import SUBSET_FILE, require_comparison_complete
from fuzev_research.features import FeatureStore, file_sha256


def test_sealed_complete_features_pass_the_independent_evaluator(fixture):
    subset = make_subset(fixture)
    identity = subset.reader.identity
    subset.reader.close()
    result = seal_features(subset.directory, fixture.plan["reference"], 8)
    reader = FeatureStore(subset.directory, identity, read_only=True)
    try:
        assert result["manifest"] == subset.manifest
        assert result["subset_manifest_sha256"] == file_sha256(subset.directory / SUBSET_FILE)
        assert require_comparison_complete(fixture.reader, reader, result["subset_manifest_sha256"])["scope"] == "evaluation-test-only"
        assert reader.preparations() == []
        with pytest.raises(ValueError, match="not completed"):
            reader.require_ready("mixed")
    finally:
        reader.close()


@pytest.mark.parametrize("mutation", ["missing", "extra", "target", "candidate", "group",
                                     "non_test", "failure", "fake_preparation"])
def test_incomplete_or_false_ownership_never_gets_a_seal(fixture, mutation):
    subset = make_subset(fixture, mutation)
    subset.reader.close()
    with pytest.raises(ValueError):
        seal_features(subset.directory, fixture.plan["reference"], 8)
    assert not (subset.directory / SUBSET_FILE).exists()


def test_wrong_examined_count_and_reference_directory_are_rejected(fixture):
    subset = make_subset(fixture)
    subset.reader.close()
    with pytest.raises(ValueError, match="Examined"):
        seal_features(subset.directory, fixture.plan["reference"], 7)
    with pytest.raises(ValueError, match="full training reference"):
        seal_features(fixture.directory, fixture.plan["reference"], 8)
    assert not (subset.directory / SUBSET_FILE).exists()
    assert not (fixture.directory / SUBSET_FILE).exists()


def test_existing_receipt_is_preserved(fixture):
    subset = make_subset(fixture)
    subset.reader.close()
    path = subset.directory / SUBSET_FILE
    path.write_text("existing operator receipt")
    with pytest.raises(FileExistsError):
        seal_features(subset.directory, fixture.plan["reference"], 8)
    assert path.read_text() == "existing operator receipt"


def test_concurrent_receipt_is_not_overwritten_or_removed(fixture, monkeypatch):
    import fuzev_research.comparison_seal as module

    subset = make_subset(fixture)
    subset.reader.close()
    original = module.os.link

    def concurrent(source, destination):
        destination.write_text("concurrent operator receipt")
        original(source, destination)

    monkeypatch.setattr(module.os, "link", concurrent)
    with pytest.raises(FileExistsError):
        seal_features(subset.directory, fixture.plan["reference"], 8)
    assert (subset.directory / SUBSET_FILE).read_text() == "concurrent operator receipt"
    assert not list(subset.directory.glob("." + SUBSET_FILE + "-*.tmp"))


def test_staging_io_error_exposes_no_partial_receipt(fixture, monkeypatch):
    import fuzev_research.comparison_seal as module

    subset = make_subset(fixture)
    subset.reader.close()

    def failed(_):
        raise OSError("synthetic staging IO failure")

    monkeypatch.setattr(module.os, "fsync", failed)
    with pytest.raises(OSError, match="synthetic staging"):
        seal_features(subset.directory, fixture.plan["reference"], 8)
    assert not (subset.directory / SUBSET_FILE).exists()
    assert not list(subset.directory.glob("." + SUBSET_FILE + "-*.tmp"))


def test_invalid_tensor_rows_remove_only_the_new_receipt(fixture):
    from safetensors.torch import load_file, save_file

    subset = make_subset(fixture)
    subset.reader.close()
    path = subset.directory / "shard-000000.safetensors"
    tensors = load_file(str(path))
    tensors["query"] = tensors["query"][:1]
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="tensor rows/shapes"):
        seal_features(subset.directory, fixture.plan["reference"], 8)
    assert not (subset.directory / SUBSET_FILE).exists()
    assert path.exists()


def test_head_mutation_during_final_checks_removes_the_unqualified_receipt(fixture, monkeypatch):
    import fuzev_research.comparison_seal as module

    subset = make_subset(fixture)
    subset.reader.close()
    original = module.require_comparison_complete

    def changed(*args):
        result = original(*args)
        with (fixture.head / "calibration.json").open("a") as handle:
            handle.write(" ")
        return result

    monkeypatch.setattr(module, "require_comparison_complete", changed)
    with pytest.raises(ValueError, match="checksum"):
        seal_features(subset.directory, fixture.plan["reference"], 8)
    assert not (subset.directory / SUBSET_FILE).exists()
