import json

import pytest
torch = pytest.importorskip("torch")

from fuzev_research.features import FeatureStore, collate
from fuzev_research.head import HeadBundle, load_bundle, save_bundle
from fuzev_research.train import (argument_parser, epoch_batches, fit_temperature, train_trial,
                         optimizer_checkpoint, restore_optimizer, membership_identity)
from fuzev_research.native import NativeBackend
from fuzev_research.preparation import ReadoutPreparer


def record(record_id="one", role="fit", count=2):
    return {"id": record_id, "group": "group", "role": role, "primitive": "choice",
            "modality": "text", "source": "synthetic-unit-fixture", "revision": "fixture",
            "candidates": [{"id": str(i), "text": str(i)} for i in range(count)],
            "target": [1.] + [0.] * (count - 1)}


def test_feature_shards_resume_without_split_leakage(tmp_path):
    store = FeatureStore(tmp_path, {"revision": "fixture"}, shard_size=1)
    store.add(record(), torch.ones(2560), torch.ones(2, 2560), torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match="leakage"):
        store.add(record("two", "test"), torch.ones(2560), torch.ones(2, 2560))
    store.close()
    reopened = FeatureStore(tmp_path, {"revision": "fixture"})
    assert reopened.contains("one")
    tensors = collate(reopened, reopened.records("fit"))
    assert tensors["query"].shape == (1, 2560)
    assert tensors["mask"].tolist() == [[True, True]]
    reopened.close()
    with pytest.raises(ValueError, match="provenance"):
        FeatureStore(tmp_path, {"revision": "other"})


def test_registered_preparation_does_not_imply_completed_extraction(tmp_path):
    path = tmp_path / "input.jsonl"
    store = FeatureStore(tmp_path / "features", {"fixture": True})
    store.register_preparation(path, "fixture-digest")
    assert not store.preparations()[0]["completed"]
    store.add(record(), torch.ones(2560), torch.ones(2, 2560))
    store.close()
    store = FeatureStore(tmp_path / "features", {"fixture": True})
    assert store.contains("one")
    assert not store.preparations()[0]["completed"]
    store.finish_preparation(path, 1)
    assert store.preparations()[0]["completed"]
    assert store.preparations()[0]["examined"] == 1
    with pytest.raises(ValueError, match="changed"):
        store.register_preparation(path, "different-digest")
    with pytest.raises(ValueError, match="unregistered"):
        store.finish_preparation(tmp_path / "other.jsonl", 1)
    store.close()


def test_qwen4_image_positions_follow_advertised_capabilities():
    backend = object.__new__(NativeBackend)
    backend.model = type("Model", (), {"caps": {}})()
    embedding = object()
    assert backend.parameters((1, 2), [embedding]) == {"indexed_embeddings": [embedding]}
    assert backend.parameters((1, 2), []) == {}


def test_native_image_alias_does_not_duplicate_vision_delimiters():
    backend = object.__new__(NativeBackend)
    embedding = type("Embedding", (), {"text_alias": "image-alias"})()
    backend.image_embedding = lambda *_: embedding
    backend.tokenizer = type("Tokenizer", (), {"hf_render_chat_template": lambda *args, **kwargs:
        "user\n<|vision_start|><|image_pad|><|vision_end|>\nQuestion"})()
    backend.preparer = ReadoutPreparer(backend.tokenizer)
    backend.encode = lambda text, embeddings: text
    rendered, embeddings = backend.base_tokens([], "image.png", "digest")
    assert rendered == "user\nimage-alias\nQuestion"
    assert embeddings == [embedding]


def test_portable_head_roundtrip_and_checksum_gate(tmp_path):
    torch.manual_seed(17)
    bundle = HeadBundle().eval()
    query, candidates = torch.ones(1, 2560), torch.ones(1, 2, 2560)
    expected = bundle("choice", query, candidates)
    save_bundle(tmp_path, bundle, {"test_fixture": True}, {key: 1. for key in bundle.heads})
    loaded, manifest, calibration = load_bundle(tmp_path)
    assert torch.equal(expected, loaded("choice", query, candidates))
    assert manifest["precision"] == "float32"
    assert calibration["choice"] == 1
    metadata = json.loads((tmp_path / "manifest.json").read_text())
    metadata["weights_sha256"] = "bad"
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_temperature_fit_with_variable_candidate_padding_is_finite():
    logits = torch.tensor([[2., 0., -torch.inf], [1., 2., 3.]])
    targets = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    temperature = fit_temperature(logits, targets, mask)
    assert .029 <= temperature <= 100.01
    assert torch.isfinite(torch.tensor(temperature))


def test_mixed_batches_cover_all_fit_records_and_replay_text():
    rows = [{"id": str(i), "primitive": "choice", "modality": "text"} for i in range(6)]
    rows += [{"id": f"image-{i}", "primitive": "choice", "modality": "image"} for i in range(10)]
    batches = list(epoch_batches(rows, 17, mixed=True, batch_size=4))
    assert {row["id"] for _, batch in batches for row in batch} == {row["id"] for row in rows}
    assert all(sum(row["modality"] == "text" for row in batch) == 2 for _, batch in batches)


def test_head_trial_checkpoint_resumes_completed_fit_without_touching_test(tmp_path, monkeypatch):
    monkeypatch.setattr(FeatureStore, "require_ready", lambda *_: {"unit_fixture": True})
    monkeypatch.setattr("fuzev_research.train.project_path", lambda path: tmp_path / path)
    monkeypatch.setattr("fuzev_research.train.Store", lambda: type("Store", (), {
        "record_result": lambda *args: None,
    })())
    torch.set_num_threads(2)
    features = FeatureStore(tmp_path / "features", {"fixture": True}, shard_size=2)
    for i, role in enumerate(["fit", "fit", "development", "test"]):
        row = record(str(i), role)
        row["group"] = f"group-{i}"
        features.add(row, torch.ones(2560), torch.stack([torch.ones(2560), -torch.ones(2560)]))
    features.flush()
    result = train_trial(features, features.records("fit"), features.records("development"),
                         "text", 17, 3e-4, 0, max_epochs=1)
    assert result["completed"]
    assert result["epochs"][0]["fit_visits"] == 2
    assert (tmp_path / result["directory"] / "head.safetensors").exists()
    assert result == train_trial(features, features.records("fit"), features.records("development"),
                                 "text", 17, 3e-4, 0, max_epochs=1)
    with pytest.raises(ValueError, match="identity changed"):
        train_trial(features, features.records("fit"), features.records("development"),
                    "text", 17, 3e-4, 0, max_epochs=2)
    trial_file = next((tmp_path / "weights/text/trials").glob("*/trial.json"))
    previous = json.loads(trial_file.read_text())
    previous["completed"] = False
    trial_file.write_text(json.dumps(previous))
    (tmp_path / result["directory"] / "optimizer.safetensors").unlink()
    with pytest.raises(ValueError, match="Missing optimizer"):
        train_trial(features, features.records("fit"), features.records("development"),
                    "text", 17, 3e-4, 0, max_epochs=1)
    features.close()


def test_collate_crops_actual_masks_and_bounds_cache(tmp_path):
    store = FeatureStore(tmp_path, {"fixture": True}, shard_size=2, cache_bytes=1)
    for i, count in enumerate((2, 16, 3, 15)):
        row = record(str(i), count=count)
        row["group"] = str(i)
        store.add(row, torch.full((2560,), float(i)), torch.full((count, 2560), float(i)))
    rows = store.records()
    batch = collate(store, [rows[0], rows[2]])
    assert batch["candidates"].shape == (2, 3, 2560)
    assert batch["mask"].tolist() == [[True, True, False], [True, True, True]]
    assert torch.equal(batch["candidates"][1], torch.full((3, 2560), 2.))
    assert store.loaded_bytes <= store.cache_bytes
    assert not store.loaded
    store.close()


def test_collate_preserves_nonprefix_masks_and_numerics():
    from fuzev_research.head import CandidateHead

    query = torch.randn(2560)
    candidates = torch.randn(8, 2560)
    mask = torch.tensor([True, False, True, False, False, False, False, False])
    sample = {"query": query, "candidates": candidates, "mask": mask,
              "targets": torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])}
    store = type("Store", (), {"tensors": lambda self, row: sample})()
    batch = collate(store, [{}])
    assert batch["mask"].tolist() == [[True, False, True]]
    head = CandidateHead()
    old = head(query[None], candidates[None], mask[None])
    new = head(batch["query"], batch["candidates"], batch["mask"])
    torch.testing.assert_close(old[:, :3], new)
    old_loss, new_loss = -old.log_softmax(1)[0, 0], -new.log_softmax(1)[0, 0]
    old_grads = torch.autograd.grad(old_loss, tuple(head.parameters()))
    new_grads = torch.autograd.grad(new_loss, tuple(head.parameters()))
    for expected, actual in zip(old_grads, new_grads):
        torch.testing.assert_close(expected, actual)


def test_optimizer_resume_requires_complete_verified_state(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    with pytest.raises(ValueError, match="Missing optimizer"):
        restore_optimizer(tmp_path, optimizer, "cpu")
    parameter.sum().backward()
    optimizer.step()
    optimizer_checkpoint(tmp_path, optimizer)
    other = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
    restore_optimizer(tmp_path, other, "cpu")
    assert next(iter(other.state.values()))["step"].item() == 1
    (tmp_path / "optimizer.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_optimizer(tmp_path, other, "cpu")


def test_membership_is_ordered_and_cli_threads_default_four():
    assert membership_identity([{"id": "a"}, {"id": "b"}]) != membership_identity([{"id": "b"}, {"id": "a"}])
    parser = argument_parser()
    assert parser.parse_args(["--features", "/tmp/fixture", "--stage", "text"]).threads == 4
    assert parser.parse_args(["--features", "/tmp/fixture", "--stage", "text", "--threads", "2"]).threads == 2


def test_sparse_collate_loads_each_feature_shard_once_without_pinning_views(tmp_path, monkeypatch):
    import weakref
    import safetensors.torch

    store = FeatureStore(tmp_path, {"fixture": True}, shard_size=3, cache_bytes=0)
    for i in range(9):
        row = record(str(i), count=16 if i % 3 == 2 else 2)
        row["group"] = str(i)
        store.add(row, torch.full((2560,), float(i)), torch.full((len(row["candidates"]), 2560), float(i)))
    all_rows = store.records()
    rows = [all_rows[i] for i in (0, 3, 6, 1, 4, 7)]
    loads, references = [], []
    original = safetensors.torch.load_file
    def load(path):
        assert all(reference() is None for reference in references)
        tensors = original(path)
        references.extend(weakref.ref(value) for value in tensors.values())
        loads.append(path)
        return tensors
    monkeypatch.setattr(safetensors.torch, "load_file", load)
    result = collate(store, rows)
    assert len(loads) == 3
    assert all(reference() is None for reference in references)
    assert result["candidates"].shape == (6, 2, 2560)
    assert result["query"][:, 0].tolist() == [0., 3., 6., 1., 4., 7.]
    assert store.loaded_bytes == 0
    store.close()


def test_original_benchmark_preserves_padding_cache_and_gradient_baseline(tmp_path):
    from fuzev_research.benchmark_training import OriginalD260024Reader, check_equivalence, original_d260024_collate
    from fuzev_research.head import CandidateHead

    torch.manual_seed(17)
    store = FeatureStore(tmp_path, {"fixture": True}, shard_size=2, cache_bytes=0)
    for index in range(12):
        count = 16 if index % 2 else 2 + index % 3
        row = record(str(index), count=count)
        row["group"] = str(index)
        store.add(row, torch.randn(2560), torch.randn(count, 2560))
    rows = store.records()[::2]
    original_reader = OriginalD260024Reader(tmp_path)
    original = original_d260024_collate(original_reader, rows)
    current = collate(store, rows)
    assert original["candidates"].shape == (6, 16, 2560)
    assert current["candidates"].shape == (6, 4, 2560)
    assert original_reader.full_shard_loads == 6
    assert len(original_reader.loaded) == 4
    assert store.loaded_bytes == 0
    result = check_equivalence(original, current, CandidateHead())
    assert result["output_slice_exact"] and result["parameter_gradients_close"]
    store.close()
