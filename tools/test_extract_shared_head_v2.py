"""Synthetic CPU extraction boundaries; no native engine or prepared corpus run."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from fuzev.encoding import VERSION, candidate_suffix, query_suffix, list_aware_query_suffix
from fuzev.head import ServingHead
from fuzev.prefix import PrefixRegistry
from fuzev_exllamav3.factory import LMConditionedHead
from fuzev_exllamav3.generator_service import GeneratorService
from fuzev_exllamav3.preparation import ReadoutPreparer
from extract_shared_head_v2 import extraction_head, extract_rows
from ragged_features import ROLES, RaggedFeatures, RaggedWriter
from train_shared_head import initialize, export_bundle


@pytest.fixture
def bundle(tmp_path):
    source, output = tmp_path / "historical", tmp_path / "features"
    geometry = dict(hidden_size=4, projection_size=3, query_adapter_size=2, key_adapter_size=2)
    export_bundle(source, initialize(geometry, 7).state_dict(),
                  dict(geometry, primitives=["choice"], max_options=16), {"choice": 1.7},
                  {"encoding_version": VERSION})
    config = extraction_head(source, output / "extraction-head")
    return source, output, config


def example(identifier, k=2, role="fit"):
    candidates = [{"id": f"product-{i}", "text": f"Complete product {i}.\nDescription {i}; no clipping."}
                  for i in reversed(range(k))]
    return {"id": identifier, "group": "group-" + identifier, "role": role, "source": "synthetic",
            "source_question_id": identifier, "primitive": "choice", "modality": "text",
            "context": [{"role": "user", "content": "Full context\nwith original whitespace."}],
            "question": "Which complete product?", "additional_context": "Extra evidence.",
            "candidates": candidates, "target": [1.] + [0.] * (k - 1),
            "grades": [2] + [0] * (k - 1), "acceptable_candidate_ids": [candidates[0]["id"]]}


def writers(output, config):
    return {role: RaggedWriter(output / role, config, shard_bytes=1 << 20) for role in ROLES}


class Tokenizer:
    def encode(self, text, **kwargs):
        if len(text) == 2 and text[0] == " " and text[1].isupper():
            return torch.tensor([[ord(text[1])]])
        return torch.tensor([list(text.encode())])

    def hf_render_chat_template(self, messages, **kwargs):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def service(output):
    target = object.__new__(GeneratorService)
    owner = SimpleNamespace(error=None, _closing=False, on_owner=False)

    async def run_on_owner(function, *args, **kwargs):
        owner.on_owner = True
        try:
            return function(*args, **kwargs)
        finally:
            owner.on_owner = False

    owner.run_on_owner = run_on_owner
    runtime = ServingHead(output / "extraction-head", device="cpu", calibration="raw")
    target.preparer = ReadoutPreparer(Tokenizer(), runtime.encoding_version)
    lm_calls = []

    def project(*args):
        assert owner.on_owner
        lm_calls.append(1)
        return torch.arange(256, dtype=torch.float32).reshape(1, 1, 256)

    lm = SimpleNamespace(caps={"logits_output": True}, forward=project,
                         lora_a_tensors={}, lora_b_tensors={}, select_outputs=lambda labels: None)
    model = SimpleNamespace(fwd_modules=[(lm, 0, 0)],
                            config=SimpleNamespace(hidden_size=4, infer_params=SimpleNamespace(prefill_optimizations=True)))
    target.head = LMConditionedHead(model, target.preparer.encode, runtime)
    target.async_generator = owner
    target.generator = SimpleNamespace(max_batch_size=2, max_chunk_size=256, num_draft_tokens=3,
                                       cache=SimpleNamespace(max_num_tokens=131072), model=model)
    target.context_tokens, target.model_name, target.closed = 65536, "synthetic", False
    target.prefixes, target.jobs = PrefixRegistry(), set()
    target.readouts, target.roots, target.lm_calls = [], [], lm_calls

    def features(tokens):
        return torch.tensor([len(tokens), sum(tokens) % 11, tokens[-1], 1.], dtype=torch.float32) / 100

    async def readout(ids, embeddings, *, checkpoint=True, readout_root_tokens=None, readout_root=None):
        tokens = ids.view(-1).tolist()
        target.readouts.append((tokens, readout_root))
        root = None
        if readout_root_tokens is not None:
            root = SimpleNamespace(position=len(tokens), closed=False)

            def close():
                assert owner.on_owner
                root.closed = True

            root.close = close
            target.roots.append(root)
        cached = readout_root.position if readout_root is not None else 0
        return {"features": features(tokens), "readout_root": root,
                "cached_tokens": cached, "forwarded_tokens": len(tokens) - cached,
                "job": SimpleNamespace(readout_root=readout_root)}

    target._readout = readout
    return target


def test_v1_copy_keeps_weights_and_v2_head_is_used_directly(bundle):
    source, output, config = bundle
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    assert (output / "extraction-head/head.safetensors").read_bytes() == before["head.safetensors"]
    assert config["capacities"] == {"choice": {"text": {"implemented_max": 255, "validated_max": None}}}
    assert config["extraction"]["purpose"] == "feature-extraction" and config["extraction"]["weights"] == "unchanged"
    assert "max_options" not in config and "primitives" not in config
    assert json.loads((output / "extraction-head/calibration.json").read_text()) == {"strata": []}
    assert extraction_head(source, output / "extraction-head") == config
    loaded = ServingHead(output / "extraction-head", device="cpu", calibration="raw")
    assert loaded.max_options == 255 and not loaded.calibrated
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert extraction_head(output / "extraction-head", output / "unused") == config
    assert not (output / "unused").exists()


@pytest.mark.parametrize("k,role", [(26, "fit"), (27, "development"), (255, "calibration")])
def test_actual_decide_order_baseline_rendering_owned_features_and_execution(bundle, k, role):
    _, output, config = bundle
    native, item = service(output), example("row", k, role)
    before = deepcopy(item)
    head, observed = native.head.head, []
    score = head.score

    def observe(primitive, query, candidates, *, baseline, modality="text"):
        assert native.async_generator.on_owner
        observed.append(tuple(t.clone() for t in (query, candidates, baseline)))
        result = score(primitive, query, candidates, baseline=baseline, modality=modality)
        for value in (query, candidates, baseline):
            value.fill_(-123)
        return result

    head.score = observe
    decide, returned = native.decide, []

    async def additive(*args, **kwargs):
        result = await decide(*args, **kwargs)
        result["execution"]["future_native_count"] = 17
        returned.append(deepcopy(result))
        return result

    native.decide = additive
    stores = writers(output, config)
    counts = asyncio.run(extract_rows(native, [item], stores, {"engine": "CPU fixture"}))
    assert counts[role] == {"written": 1, "skipped": 0}
    row, values = RaggedFeatures(output / role, config).read(stores[role].completed.by_id[item["id"]])
    for name, value in zip(("query", "candidates", "baseline"), observed[0]):
        torch.testing.assert_close(values[name], value, rtol=0, atol=0)
        assert not (values[name] == -123).any()
    baseline = torch.arange(ord("A"), ord("A") + k).float() if k <= 26 else torch.zeros(k)
    torch.testing.assert_close(values["baseline"], baseline, rtol=0, atol=0)
    suffix = query_suffix if k <= 26 else list_aware_query_suffix
    assert row["rendering"]["query_suffix"] == suffix(item["question"], item["candidates"], item["additional_context"])
    assert row["rendering"]["candidate_suffixes"] == [candidate_suffix(c) for c in item["candidates"]]
    assert row["request"]["candidates"] == item["candidates"]
    assert not ({"target", "grades", "acceptable_candidate_ids"} & row["request"].keys())
    assert row["execution"] == returned[0]["execution"]
    assert row["execution"]["future_native_count"] == 17
    assert not ({"scores", "selected_id", "probabilities"} & row.keys())
    assert item == before
    for key in item:
        assert row[key] == item[key]
    assert all(not store.completed.rows for other, store in stores.items() if other != role)


def test_resume_keeps_completed_feature_bytes(bundle):
    _, output, config = bundle
    native, item = service(output), example("same")
    stores = writers(output, config)
    asyncio.run(extract_rows(native, [item], stores, {}))
    saved = {path.name: path.read_bytes() for path in (output / "fit").iterdir()}
    native.readouts.clear()
    stores = writers(output, config)
    counts = asyncio.run(extract_rows(native, [item], stores, {}))
    assert counts["fit"] == {"written": 0, "skipped": 1} and not native.readouts
    assert {path.name: path.read_bytes() for path in (output / "fit").iterdir()} == saved


@pytest.mark.parametrize("error", [RuntimeError("score failure"), asyncio.CancelledError()])
def test_failure_or_cancellation_restores_hook_and_preserves_completed_row(bundle, error):
    _, output, config = bundle
    native, stores = service(output), writers(output, config)
    asyncio.run(extract_rows(native, [example("done")], stores, {}))
    head = native.head.head

    def fail(*args, **kwargs):
        raise error

    head.score = fail
    with pytest.raises(type(error)):
        asyncio.run(extract_rows(native, [example("unfinished")], stores, {}))
    assert list(RaggedFeatures(output / "fit", config).by_id) == ["done"]


def test_shard_budget_buffers_more_than_one_extracted_row(bundle):
    _, output, config = bundle
    stores = writers(output, config)
    counts = asyncio.run(extract_rows(service(output), [example("one"), example("two")], stores, {}))
    dataset = RaggedFeatures(output / "fit", config)
    assert counts["fit"] == {"written": 2, "skipped": 0}
    assert len(list((output / "fit").glob("*.safetensors"))) == 1
    assert list(dataset.by_id) == ["one", "two"]
