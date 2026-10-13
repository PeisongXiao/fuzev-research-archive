from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")

from fuzev_research import native


def engine(monkeypatch, tmp_path):
    calls = {"load": [], "unload": 0}
    config = SimpleNamespace(directory="fixture", infer_params=SimpleNamespace())
    class Model:
        @staticmethod
        def from_config(_):
            return Model()
        def load(self, **kwargs):
            calls["load"].append(kwargs)
        def unload(self):
            calls["unload"] += 1
    class Cache:
        def __init__(self, model, max_num_tokens, max_batch_size, max_history):
            self.num_slots = max_batch_size
            self.max_num_tokens = max_num_tokens
            self.recurrent_layers = {0: SimpleNamespace(storage_size=lambda: max_batch_size * 128)}
            calls["cache"] = (max_num_tokens, max_batch_size, max_history)
    class Lane:
        def __init__(self, model, cache, context_limit, chunk_size):
            self.context_limit, self.chunk_size = context_limit, chunk_size
            self.batch_capacity = min(model.fuzev_batch_capacity, cache.num_slots)
    module = SimpleNamespace(Cache=Cache, Config=SimpleNamespace(from_directory=lambda _: config),
                             Model=Model, Tokenizer=SimpleNamespace(from_config=lambda _: SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "exllamav3", module)
    monkeypatch.setitem(sys.modules, "exllamav3.fuzev", SimpleNamespace(NativeLane=Lane))
    path = tmp_path / "fixture.so"
    path.write_bytes(b"CPU test fixture")
    monkeypatch.setitem(sys.modules, "exllamav3.ext", SimpleNamespace(exllamav3_ext=SimpleNamespace(__file__=str(path))))
    monkeypatch.setattr(native.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    return calls


@pytest.mark.parametrize("capacity", [4, 8])
def test_backend_capacity_reaches_model_cache_lane_and_identity(monkeypatch, tmp_path, capacity):
    calls = engine(monkeypatch, tmp_path)
    backend = native.NativeBackend("fixture", optimized=False, batch_capacity=capacity, candidate_batch_size=capacity)
    assert calls["cache"] == (294912, capacity, 3)
    assert calls["load"][0]["max_batch_size"] == capacity
    assert backend.lane.batch_capacity == capacity
    assert backend.readout_runtime["batch_capacity"] == capacity
    assert backend.readout_runtime["max_hybrid_slots"] == capacity
    assert backend.readout_runtime["model_batch_capacity"] == capacity
    assert backend.readout_runtime["context_tokens"] == 262144
    assert backend.readout_runtime["cache_tokens"] == 294912
    assert backend.readout_runtime["recurrent_state_storage_bytes"] == capacity * 128
    expected_extra = (capacity - 4) * 128
    assert backend.readout_runtime["extra_recurrent_state_bytes_vs_capacity4"] == expected_extra
    backend.set_candidate_batch_size(1)
    assert backend.candidate_batch_size == backend.readout_runtime["candidate_batch_size"] == 1
    with pytest.raises(ValueError, match="declared native capacity"):
        backend.set_candidate_batch_size(capacity + 1)


@pytest.mark.parametrize("capacity,candidates", [(4, 5), (8, 9), (9, 4), (0, 4), (True, 1), (8, True)])
def test_invalid_capacity_never_loads_model(monkeypatch, tmp_path, capacity, candidates):
    calls = engine(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        native.NativeBackend("fixture", optimized=False, batch_capacity=capacity, candidate_batch_size=candidates)
    assert calls["load"] == []
