"""Synthetic CPU-only scaffold checks; no real tensors, corpus, native or CUDA."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fuzev_research import anchor_head as ah
from fuzev_research.head import CandidateHead


def metadata():
    codebook = {"mcg": True, "mul1": False}
    return {"schema_version": 1, "architecture": ah.ANCHOR_PROFILE, "feature_tap": ah.TAP,
            "hidden_size": 2560, "vocab_size": 248320, "bits": 6,
            "input_dtype": "float16", "output_dtype": "float16",
            "label_text": [" " + chr(ord("A") + i) for i in range(16)], "label_ids": list(range(16)),
            "codebook": codebook, "tensors": ah._anchor_spec(codebook),
            "native_commit": "1" * 40, "extension_sha256": "2" * 64,
            "native_environment": {key: None for key in ah.ENV_KEYS},
            "source_readout_sha256": "3" * 64, "tokenizer_sha256": "4" * 64,
            "encoding_sha256": "5" * 64, "operator_sha256": "6" * 64,
            "model_repo": "synthetic/reference", "model_revision": "synthetic-pinned"}


@pytest.fixture(autouse=True)
def no_native_or_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = ah.importlib.import_module

    def guarded(name, *args, **kwargs):
        if name.startswith("exllamav3"):
            raise AssertionError("CPU tests must never import the native/model package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ah.importlib, "import_module", guarded)
    for key in ah.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def reference(tmp_path):
    return ah.FrozenNativeAnchor(tmp_path, metadata(), "a" * 64, "b" * 64)


@pytest.fixture
def head(reference):
    return ah.AnchorResidualHead(reference, primitive="noul", residual_bound=.25, seed=43,
                                 cache_feature_identity_sha256="c" * 64, hidden_size=4, projection_size=3)


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(7)
    return {"query": torch.randn(2, 4, generator=generator),
            "candidates": torch.randn(2, 3, 4, generator=generator),
            "mask": torch.tensor([[True, True, False], [True, True, True]]),
            "baseline": torch.tensor([[1.25, -.5, 0.], [.75, 1.5, -.25]]),
            "rows": [{"id": f"fit-{i}", "group_": f"group-{i}", "role": "fit",
                      "primitive": "noul", "source": "helpsteer2", "evaluation_scope": "training",
                      "dataset_purpose": "training"} for i in range(2)]}


def cached(head, batch, **kwargs):
    inputs = {key: batch[key] for key in ("query", "candidates", "mask", "baseline", "rows")}
    inputs.update(anchor_sha256=head.anchor.weights_sha256,
                  producer_identity_sha256=head.anchor.producer_identity_sha256,
                  feature_identity_sha256="c" * 64)
    inputs.update(kwargs)
    return head.forward_cached(**inputs)


def supervised_step(head, batch):
    optimizer = torch.optim.SGD(head.residual.parameters(), lr=.03)
    optimizer.zero_grad(set_to_none=True)
    targets = torch.tensor([[1., 0., 0.], [.5, .5, 0.]])
    logp = torch.log_softmax(cached(head, batch), dim=1).masked_fill(~batch["mask"], 0)
    loss = -(targets * logp).sum(dim=1).mean()
    loss.backward()
    optimizer.step()
    return loss


def test_initialization_exactly_anchor_and_rng_unchanged(head, batch):
    state = torch.random.get_rng_state().clone()
    same = ah.fresh_residual(43, 4, 3)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(value, same.state_dict()[key]) for key, value in head.residual.state_dict().items())
    assert head.residual.query.weight.count_nonzero() > 0
    assert head.residual.key.weight.count_nonzero() == head.residual.key.bias.count_nonzero() == 0
    logits = cached(head, batch)
    assert torch.equal(logits[batch["mask"]], batch["baseline"][batch["mask"]])
    assert torch.isneginf(logits[~batch["mask"]]).all()
    assert head.margin_diagnostics(batch["query"], batch["candidates"], batch["mask"])["centered_rms"] == 0
    assert head.anchor._runtime is None
    assert not any("anchor" in name or "bound" in name for name, _ in head.named_parameters())


def test_supervised_key_gradient_first_then_query_gradient(head, batch):
    initial = head.residual_sha256()
    assert torch.isfinite(supervised_step(head, batch))
    gradients = head.gradient_diagnostics()
    assert gradients["key.weight"]["nonzero"] and gradients["key.weight"]["finite"]
    assert not gradients["query.weight"]["nonzero"]
    assert torch.isfinite(supervised_step(head, batch))
    assert head.gradient_diagnostics()["query.weight"]["nonzero"]
    proof = head.require_learned_residual(initial, optimizer_steps=2, supervised_gradient_steps=2,
                                         query=batch["query"], candidates=batch["candidates"],
                                         mask=batch["mask"], rows=batch["rows"])
    assert proof["margins"]["max_candidate_margin"] > 0
    assert proof["dev_anchor_ablation_required"] and not proof["quality_pass"]
    assert head.anchor._runtime is None


def test_zero_start_gradient_has_unit_slope_independent_of_bound(head, batch):
    other = ah.AnchorResidualHead(head.anchor, primitive="noul", residual_bound=2., seed=43,
                                  cache_feature_identity_sha256="c" * 64, hidden_size=4, projection_size=3)
    gradients = []
    for candidate in (head, other):
        logp = torch.log_softmax(cached(candidate, batch), dim=1)
        loss = -logp[:, 0].mean()
        loss.backward()
        gradients.append(candidate.residual.key.weight.grad.clone())
    assert torch.equal(*gradients)
    assert gradients[0].count_nonzero() > 0


def test_fixed_bound_centering_and_mask_after(head, batch):
    with torch.no_grad():
        head.residual.key.weight.fill_(10.)
    raw = head.residual(batch["query"], batch["candidates"])
    mask = batch["mask"]
    centered = raw - raw.masked_fill(~mask, 0).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    expected = (.25 * torch.tanh(centered / .25)).masked_fill(~mask, -torch.inf)
    assert torch.equal(head.residual_logits(batch["query"], batch["candidates"], mask), expected)
    info = head.margin_diagnostics(batch["query"], batch["candidates"], mask)
    assert info["max_abs_residual"] <= .25 and info["max_candidate_margin"] <= .5
    assert head.residual_bound == .25
    changed = batch["candidates"].clone()
    changed[0, 2] = 1e4
    assert torch.equal(head.residual_logits(batch["query"], changed, mask)[mask], expected[mask])


def test_common_shift_is_not_learned_candidate_evidence(head, batch):
    batch["mask"][:, 2] = False
    initial = head.residual_sha256()
    with torch.no_grad():
        head.residual.key.bias.fill_(2.)
    assert head.residual_sha256() != initial
    with pytest.raises(ValueError, match="control-only"):
        head.require_learned_residual(initial, optimizer_steps=1, supervised_gradient_steps=1,
                                     query=batch["query"], candidates=batch["candidates"],
                                     mask=batch["mask"], rows=batch["rows"])


@pytest.mark.parametrize("steps,gradient_steps", [(0, 0), (1, 0), (1, 2), (True, 1)])
def test_step_zero_or_no_supervision_denied(head, batch, steps, gradient_steps):
    with pytest.raises(ValueError):
        head.require_learned_residual(head.residual_sha256(), optimizer_steps=steps,
                                     supervised_gradient_steps=gradient_steps, query=batch["query"],
                                     candidates=batch["candidates"], mask=batch["mask"], rows=batch["rows"])


@pytest.mark.parametrize("field,value", [("role", "test"), ("role", "calibration"), ("source", "qasper"),
                                         ("source", "vl-rewardbench"), ("dataset_purpose", "external"),
                                         ("evaluation_scope", "external"), ("primitive", "choice"), ("group_", "")])
def test_cached_external_and_wrong_rows_denied(head, batch, field, value):
    batch["rows"][0][field] = value
    with pytest.raises(ValueError):
        cached(head, batch)


def test_cached_dev_is_no_gradient_eval_only(head, batch):
    for row in batch["rows"]:
        row["role"] = "development"
    with torch.no_grad(), pytest.raises(ValueError):
        cached(head, batch)
    head.eval()
    with pytest.raises(ValueError):
        cached(head, batch)
    with torch.no_grad():
        assert torch.equal(cached(head, batch)[batch["mask"]], batch["baseline"][batch["mask"]])


@pytest.mark.parametrize("field", ["anchor_sha256", "producer_identity_sha256", "feature_identity_sha256"])
def test_cached_identity_pins_denied(head, batch, field):
    with pytest.raises(ValueError):
        cached(head, batch, **{field: "d" * 64})


def test_cached_missing_baseline_duplicate_rows_and_missing_feature_pin(head, batch):
    with pytest.raises(ValueError):
        cached(head, batch, baseline=None)
    batch["rows"][1] = copy.deepcopy(batch["rows"][0])
    with pytest.raises(ValueError):
        cached(head, batch)
    head.cache_feature_identity_sha256 = None
    with pytest.raises(ValueError):
        cached(head, batch)


@pytest.mark.parametrize("change", ["query-nan", "candidate-inf", "candidate-shape", "query-shape", "query-grad",
                                   "candidate-grad", "mask-float", "mask-hole", "mask-one", "mask-zero",
                                   "mask-shape", "baseline-nan", "baseline-shape", "baseline-grad", "baseline-double"])
def test_invalid_features_masks_or_controls_denied(head, batch, change):
    if change == "query-nan":
        batch["query"][0, 0] = float("nan")
    elif change == "candidate-inf":
        batch["candidates"][0, 2, 0] = float("inf")
    elif change == "candidate-shape":
        batch["candidates"] = batch["candidates"][:, :, :3]
    elif change == "query-shape":
        batch["query"] = batch["query"][:, :3]
    elif change in ("query-grad", "candidate-grad", "baseline-grad"):
        batch[{"query-grad": "query", "candidate-grad": "candidates", "baseline-grad": "baseline"}[change]].requires_grad_(True)
    elif change == "mask-float":
        batch["mask"] = batch["mask"].float()
    elif change == "mask-hole":
        batch["mask"][0] = torch.tensor([True, False, True])
    elif change in ("mask-one", "mask-zero"):
        batch["mask"][0] = torch.tensor([change == "mask-one", False, False])
    elif change == "mask-shape":
        batch["mask"] = batch["mask"][:1]
    elif change == "baseline-nan":
        batch["baseline"][0, 0] = float("nan")
    elif change == "baseline-shape":
        batch["baseline"] = batch["baseline"][:, :2]
    else:
        batch["baseline"] = batch["baseline"].double()
    with pytest.raises(ValueError):
        cached(head, batch)


@pytest.mark.parametrize("bound", [0., -1., float("nan"), float("inf"), 1, True, 1e-50])
def test_invalid_fixed_bound_denied(reference, bound):
    with pytest.raises(ValueError):
        ah.AnchorResidualHead(reference, primitive="choice", residual_bound=bound, seed=29)


def test_nonfinite_raw_parameters_and_half_residual_denied(head, batch):
    with torch.no_grad():
        head.residual.key.weight[0, 0] = float("nan")
    with pytest.raises(ValueError):
        head.residual_logits(batch["query"], batch["candidates"], batch["mask"])
    head.half()
    with pytest.raises(ValueError):
        head.residual_logits(batch["query"], batch["candidates"], batch["mask"])


def test_input_resource_dimensions_bounded(head):
    for batch, width in ((257, 2), (1, 17), (0, 2)):
        with pytest.raises(ValueError):
            head.residual_logits(torch.zeros(batch, 4), torch.zeros(batch, width, 4))


def test_ordinal_retained_not_anchored(reference):
    original = CandidateHead()
    before = {key: value.clone() for key, value in original.state_dict().items()}
    retained = ah.retain_ordinal(original)
    assert all(torch.equal(value, retained.state_dict()[key]) for key, value in before.items())
    assert all(not parameter.requires_grad for parameter in retained.parameters())
    assert retained is not original
    with pytest.raises(ValueError):
        ah.AnchorResidualHead(reference, primitive="ordinal", residual_bound=.25, seed=43)


def test_runtime_is_lazy_cpu_rejected_without_import_and_env_unchanged(reference, head, batch, monkeypatch):
    with pytest.raises(ValueError, match="Synthetic"):
        head(batch["query"], batch["candidates"], batch["mask"])
    query, mask = torch.zeros(1, 2560), torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        reference.logits(query, mask)
    monkeypatch.setenv("EXL3_INT8_GEMV", "0")
    with pytest.raises(ValueError, match="environment"):
        reference.logits(query, mask)
    assert ah.os.environ["EXL3_INT8_GEMV"] == "0"
    assert reference._runtime is None


def test_exact_production_schema_without_allocation():
    spec = ah._anchor_spec({"mcg": True, "mul1": False})
    assert spec["trellis"] == {"shape": [160, 15520, 96], "dtype": "I16"}
    assert spec["suh"]["shape"] == [2560] and spec["svh"]["shape"] == [248320]
    assert spec["mcg"] == {"shape": [], "dtype": "I32"}


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    # Private schema substitution exercises I/O without allocating a 455 MiB fixture.
    def tiny_spec(codebook):
        result = {"trellis": {"shape": [2, 2, 96], "dtype": "I16"},
                  "suh": {"shape": [4], "dtype": "F16"}, "svh": {"shape": [8], "dtype": "F16"}}
        for name, enabled in codebook.items():
            if enabled:
                result[name] = {"shape": [], "dtype": "I32"}
        return result

    monkeypatch.setattr(ah, "_anchor_spec", tiny_spec)
    values = {"trellis": torch.arange(384, dtype=torch.int16).reshape(2, 2, 96),
              "suh": torch.ones(4, dtype=torch.float16), "svh": torch.ones(8, dtype=torch.float16),
              "mcg": torch.tensor(ah.CODEBOOKS["mcg"], dtype=torch.int32)}
    reference = ah.save_anchor(tmp_path / "reference", values, metadata())
    return reference, values


def rewrite_json(path, value):
    data = json.dumps(value, allow_nan=False).encode()
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_typed_self_contained_anchor_roundtrip_is_lazy(artifact):
    reference, values = artifact
    loaded = ah.load_anchor(reference.directory, manifest_sha256=reference.manifest_sha256)
    assert loaded.weights_sha256 == reference.weights_sha256 and loaded._runtime is None
    with safe_open(str(reference.directory / "anchor.safetensors"), framework="pt") as handle:
        assert all(torch.equal(value, handle.get_tensor(key)) for key, value in values.items())
    altered = loaded.metadata
    altered["label_ids"][0] = 100
    assert loaded.metadata["label_ids"][0] == 0
    assert loaded.producer_identity_sha256 == reference.producer_identity_sha256


@pytest.mark.parametrize("field,value", [("hidden_size", 4), ("vocab_size", 512), ("bits", 4), ("schema_version", True),
                                         ("output_dtype", "float32"), ("label_ids", [0] * 16),
                                         ("label_ids", list(range(15)) + [248320]), ("label_ids", [[0]] * 16),
                                         ("native_environment", {}), ("extension_sha256", ""), ("exec", "payload")])
def test_malformed_metadata_denied(field, value):
    description = metadata()
    description[field] = value
    with pytest.raises(ValueError):
        ah._anchor_metadata(description)


@pytest.mark.parametrize("change", ["float-cast", "wrong-shape", "unknown-tensor", "bad-codebook", "nonfinite", "extra-metadata"])
def test_corrupt_packed_state_rejected_after_checksum_repin(artifact, change):
    reference, values = artifact
    values = {key: value.clone() for key, value in values.items()}
    extra = None
    if change == "float-cast":
        values = {key: value.float() for key, value in values.items()}
    elif change == "wrong-shape":
        values["trellis"] = values["trellis"].reshape(4, 96)
    elif change == "unknown-tensor":
        values["cell_head"] = torch.zeros(1)
    elif change == "bad-codebook":
        values["mcg"].fill_(0)
    elif change == "nonfinite":
        values["suh"][0] = float("nan")
    else:
        extra = {"executable": "forbidden"}
    path = reference.directory / "anchor.safetensors"
    save_file(values, str(path), metadata=extra)
    description = json.loads((reference.directory / "anchor.json").read_text())
    description["weights_sha256"] = ah._sha(path, ah.ANCHOR_LIMIT)
    pin = rewrite_json(reference.directory / "anchor.json", description)
    with pytest.raises(ValueError):
        ah.load_anchor(reference.directory, manifest_sha256=pin)


def test_checksum_oversize_symlink_and_exclusive_outputs(artifact, tmp_path, monkeypatch):
    reference, values = artifact
    with pytest.raises(ValueError):
        ah.load_anchor(reference.directory, manifest_sha256="f" * 64)
    with pytest.raises(FileExistsError):
        ah.save_anchor(reference.directory, values, metadata())
    path = reference.directory / "anchor.safetensors"
    monkeypatch.setattr(ah, "ANCHOR_LIMIT", path.stat().st_size)
    with pytest.raises(ValueError, match="oversized"):
        ah.load_anchor(reference.directory, manifest_sha256=reference.manifest_sha256)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        ah._sha(link, 1 << 20)


def test_head_roundtrip_preserves_fixed_bound_learned_state_and_reference(artifact, tmp_path):
    reference, _ = artifact
    head = ah.AnchorResidualHead(reference, primitive="choice", residual_bound=.25, seed=29,
                                 cache_feature_identity_sha256="c" * 64)
    with torch.no_grad():
        head.residual.key.weight[0, 0] = .015
    destination = tmp_path / "bundle"
    pin = ah.save_head(destination, head)
    loaded = ah.load_head(destination, manifest_sha256=pin)
    assert loaded.residual_sha256() == head.residual_sha256()
    assert loaded.residual_bound == .25 and not loaded.training
    assert loaded.anchor.weights_sha256 == head.anchor.weights_sha256
    assert loaded.anchor.directory == destination and loaded.anchor._runtime is None
    assert json.loads((destination / "head.json").read_text())["artifact_kind"] == "unqualified-head-scaffold"
    with pytest.raises(FileExistsError):
        ah.save_head(destination, head)
    description = json.loads((destination / "head.json").read_text())
    description["anchor_coefficient"] = .5
    repin = rewrite_json(destination / "head.json", description)
    with pytest.raises(ValueError):
        ah.load_head(destination, manifest_sha256=repin)


def test_tiny_head_may_not_be_published(head, tmp_path):
    with pytest.raises(ValueError, match="Synthetic"):
        ah.save_head(tmp_path / "bundle", head)
