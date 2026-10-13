"""CPU encoding/export contracts with a stub backbone and no training."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

import extract_hf_features as extractor
import train_shared_head as trainer
from fuzev._scoring import SharedDecisionHead
from fuzev.head import ServingHead


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "auto"])
def test_extraction_writes_its_encoding_and_fp32_features(tmp_path, monkeypatch, dtype):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text('{"hidden_size": 8}')
    source = tmp_path / "fit.jsonl"
    source.write_text(json.dumps({"id": "fit-1", "primitive": "choice", "question": "Which?",
                                 "context": "Example", "candidates": [{"text": "Yes"}, {"text": "No"}],
                                 "target": [1., 0.]}) + "\n")
    output = tmp_path / "features"
    loaded = {}

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [0 if text == "A" else 1] if text in {"A", "B"} else [2, 3]

    class Backbone:
        def to(self, device):
            loaded["device"] = device
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def __call__(self, *, input_ids, **kwargs):
            assert input_ids.device.type == "cpu"
            hidden = torch.arange(input_ids.numel() * 8, dtype=self.dtype).reshape(1, -1, 8)
            return SimpleNamespace(hidden_states=[hidden], last_hidden_state=hidden,
                                   logits=torch.tensor([[[.25, -.75]]], dtype=self.dtype))

        @property
        def base_model(self):
            return self

    def load_model(directory, **kwargs):
        loaded.update(kwargs)
        model = Backbone()
        model.dtype = torch.float16 if kwargs["dtype"] == "auto" else kwargs["dtype"]
        return model

    config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=8, max_position_embeddings=32))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    result = extractor.extract(SimpleNamespace(model_directory=model_directory, input=[source], output=output,
                                               device="cpu", dtype=dtype))
    encoding = json.loads((output / "encoding.json").read_text())
    assert encoding == {"encoding_version": "fuzev-hf-text-branches-v1"}
    assert result["encoding_version"] == encoding["encoding_version"]
    assert loaded["dtype"] == ("auto" if dtype == "auto" else getattr(torch, dtype))
    assert loaded["device"] == "cpu" and result["cuda_initialized"] is False
    row = json.loads((output / "fit.jsonl").read_text())
    assert row["query"] == list(range(8, 16)) and row["baseline"] == [.25, -.75]
    assert row["candidates"] == [list(range(24, 32)), list(range(24, 32))]

    geometry = dict(hidden_size=8, projection_size=4, query_adapter_size=2, key_adapter_size=1)
    head = SharedDecisionHead(**geometry)
    bundle = trainer.export_bundle(tmp_path / "head", head.state_dict(),
                                   dict(geometry, primitives=["choice"], max_options=2, release_date="2026-10-12"),
                                   {"choice": 1.}, encoding)
    with ServingHead(bundle, device="cpu") as scorer:
        assert scorer.encoding_version == "fuzev-hf-text-branches-v1"
        scores, raw, calibrated = scorer.score("choice", torch.tensor(row["query"], dtype=torch.float32),
                                              torch.tensor(row["candidates"], dtype=torch.float32),
                                              baseline=torch.tensor(row["baseline"]))
        assert torch.isfinite(torch.tensor(scores)).all()
        assert sum(raw) == pytest.approx(1.) and sum(calibrated) == pytest.approx(1.)
