from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.features import FeatureStore
from fuzev_research.native import NativeBackend
from fuzev_research.preparation import ReadoutPreparer


def probe():
    path = Path(__file__).resolve().parents[1] / "tools/probe_extraction_concurrency.py"
    spec = importlib.util.spec_from_file_location("extraction_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_selects_complete_stratified_fit_rows_only(tmp_path):
    module = probe()
    items = [{"track": track, "row": {"id": f"{track}-{i}", "role": "fit"},
              "bounds": {"branch_proxy": 128, "tail_byte_bound": 32}}
             for track in module.TRACKS for i in range(8)]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"rows": items}))
    groups = module.select(path, 16)
    assert sum(map(len, groups)) == 16
    assert len(groups[2]) == 4
    assert all(item in items for group in groups for item in group)
    items[-1]["row"]["role"] = "test"
    path.write_text(json.dumps({"rows": items}))
    with pytest.raises(ValueError, match="fitting-only"):
        module.select(path, 16)


class Tokenizer:
    def hf_render_chat_template(self, messages, **kwargs):
        return json.dumps(messages)
    def encode(self, text, **kwargs):
        return torch.tensor([[ord(text[1])]] if len(text) == 2 and text[0] == " " else [list(map(ord, text))])


class Lane:
    def __init__(self):
        self.pool = SimpleNamespace(free=range(100000))
    def create(self):
        return SimpleNamespace(tokens=(), closed=False)
    def freeze(self, branch):
        return branch
    def fork(self, branch):
        return SimpleNamespace(tokens=branch.tokens, closed=False)
    def release(self, branch):
        branch.closed = True
    def append(self, branch, tokens, **kwargs):
        branch.tokens += tuple(tokens)
        return torch.full((1, 1, 2560), float(sum(branch.tokens) % 251))
    def append_features_many(self, branches, tails, **kwargs):
        return torch.cat([self.append(branch, tail) for branch, tail in zip(branches, tails)])


def fake_backend(kind):
    backend = object.__new__(kind)
    backend.tokenizer = Tokenizer()
    backend.preparer = ReadoutPreparer(backend.tokenizer)
    backend.lane, backend.prefixes = Lane(), OrderedDict()
    head = SimpleNamespace(caps={"logits_output": True}, forward=lambda feature, _: torch.arange(128).reshape(1, 1, -1) + feature[..., :1])
    backend.model = SimpleNamespace(caps={}, fwd_modules=[(head, None, None)])
    backend.candidate_batch_size = 4
    return backend


def test_probe_reads_an_unpacked_control_source(tmp_path):
    module = probe()
    source = tmp_path / "fuzev_research/native.py"
    source.parent.mkdir()
    source.write_text("class NativeBackend:\n    control_name = 'synthetic-control'\n")
    old_type, digest = module.original_backend(tmp_path)
    assert len(digest) == 64
    assert old_type.control_name == "synthetic-control"


def test_non_mrope_parameters_do_not_concatenate_prefix_and_suffix():
    backend = fake_backend(NativeBackend)
    class Tokens:
        def __add__(self, _):
            raise AssertionError("Unnecessary full token-tuple copy")
    assert backend.parameters(Tokens(), ["embedding"], (1, 2)) == {"indexed_embeddings": ["embedding"]}


def test_probe_compares_full_baseline_logits_and_metadata(tmp_path):
    module = probe()
    row = {"id": "one", "group": "one", "role": "fit", "primitive": "choice", "modality": "text",
           "source": "fixture", "revision": "fixture", "target": [1., 0.],
           "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]}
    stores = [FeatureStore(tmp_path / arm, {"fixture": True}) for arm in ("a", "b")]
    try:
        for index, store in enumerate(stores):
            store.add(row, torch.zeros(2560), torch.zeros(2, 2560), torch.tensor([1., float(index)]), {"batches": [2]})
            store.flush()
        result = module.compare_stores(*stores)
        assert len(result) == 1 and result[0]["tensor"] == "baseline"
        assert result[0]["max_absolute"] == 1
    finally:
        for store in stores:
            store.close()
