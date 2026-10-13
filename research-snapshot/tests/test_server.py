import json
from threading import Event, current_thread

import pytest

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient

from fuzev_research import server
from fuzev_research.features import file_sha256


def test_writer_yields_gpu_queue_to_requests_and_cancels_at_a_record_boundary(tmp_path, monkeypatch):
    manifest = tmp_path / "data/manifests/acquisition.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"models": {"native-4.05": {"local_snapshot": "unit-model"}}}))
    prepared = tmp_path / "data/prepared/text.jsonl"
    prepared.parent.mkdir(parents=True)
    monkeypatch.setattr(server, "project_path", lambda path: tmp_path / path)
    monkeypatch.setattr(server, "runtime_sources", lambda: {"recipes": "unit"})
    ready, release, finished = Event(), Event(), Event()
    workers, owner_threads = [], []
    row = {"primitive": "choice", "context": [], "question": "queued decision",
           "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]}
    prepared.write_text(json.dumps({**row, "id": "synthetic-job-row", "group": "synthetic-group",
                                   "source": "helpsteer3", "revision": "fixture", "modality": "text",
                                   "role": "fit", "target": [1., 0.]}) + "\n")
    (manifest.parent / "text-splits.json").write_text(json.dumps({
        "path": str(prepared), "sha256": file_sha256(prepared), "records": 1,
        "counts": {"helpsteer3/choice/fit": 1}}))

    class Backend:
        candidate_batch_size = 1
        readout_runtime = {"candidate_batch_size": 1, "native_extension_sha256": "fixture", "memory_fraction": None}
        def __init__(self, _directory, **_kwargs):
            owner_threads.append(current_thread().name)
        def extract(self, row, baseline=True):
            owner_threads.append(current_thread().name)
            return torch.zeros(2560), torch.zeros(2, 2560), None, {}
        def close(self):
            owner_threads.append(current_thread().name)

    def extraction(model, paths, proxy, cancel):
        workers.append(current_thread().name)
        proxy.extract(row)
        ready.set()
        try:
            assert release.wait(timeout=5)
            if cancel.is_set():
                raise InterruptedError("Cancelled at committed boundary")
        finally:
            finished.set()

    monkeypatch.setattr(server, "NativeBackend", Backend)
    monkeypatch.setattr(server, "extract_features", extraction)
    app = server.create_app()
    with TestClient(app) as client:
        try:
            health = client.get("/health").json()
            assert health["status"] == "ok"
            assert health["candidate_batch_size"] == 1
            assert health["readout_runtime"] == Backend.readout_runtime
            response = client.post("/v1/jobs/extract", json={"prepared": ["text.jsonl"]})
            assert response.status_code == 202, response.text
            job = response.json()["id"]
            assert ready.wait(timeout=2)
            # The writer is still active, but it does not hold the GPU executor.
            response = client.post("/v1/features", json={"row": row})
            assert response.status_code == 200
            assert not release.is_set()
            assert len(response.json()["query"]) == 2560
            assert client.post(f"/v1/jobs/{job}/cancel").status_code == 202
            release.set()
            assert finished.wait(timeout=2)
            assert client.get(f"/v1/jobs/{job}").json()["status"] == "cancelled"
        finally:
            release.set()
    assert workers == ["fuzev-feature-writer_0"]
    assert len(owner_threads) == 4
    assert set(owner_threads) == {"fuzev-gpu-owner_0"}


def test_writer_proxy_copies_the_readout_contract():
    backend = type("Backend", (), {"readout_runtime": {"candidate_batch_size": 4}})()
    proxy = server.FeatureProxy(backend, None)
    observed = proxy.readout_runtime
    observed["candidate_batch_size"] = 1
    assert backend.readout_runtime["candidate_batch_size"] == 4


def test_proxy_retains_one_private_preparer_across_drained_files():
    calls = []
    def make():
        calls.append(1)
        return object()
    backend = type("Backend", (), {"make_preparer": staticmethod(make)})()
    proxy = server.FeatureProxy(backend, None)
    assert proxy.make_preparer() is proxy.make_preparer()
    assert calls == [1]


def test_provenance_follows_imported_worktrees_not_root_main(monkeypatch, tmp_path):
    import importlib.util
    import subprocess
    from fuzev_research import native
    recipes, engine = tmp_path / "recipes-worktree", tmp_path / "native-worktree"
    monkeypatch.setattr(native, "__file__", str(recipes / "fuzev_research/native.py"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: type("Spec", (), {
        "origin": str(engine / "exllamav3/__init__.py")})())
    seen = []
    def output(command, cwd, text=False, **kwargs):
        seen.append(cwd)
        if command[1] == "rev-parse":
            return cwd.name + "-commit\n"
        return "" if text else b""
    monkeypatch.setattr(subprocess, "check_output", output)
    assert native.runtime_sources() == {"recipes": "recipes-worktree-commit",
                                        "exllamav3-fuzev": "native-worktree-commit"}
    assert set(seen) == {recipes, engine}
