import json
from threading import Event, current_thread

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.comparison_extract import RUN_FAILURE, extract_native, main
from fuzev_research.evaluate import TEST_SCOPE
from fuzev_research.features import FeatureStore


def row(index, role="test"):
    return {"id": str(index), "group": str(index), "role": role, "primitive": "choice",
            "modality": "text", "source": "synthetic", "revision": "fixture",
            "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
            "target": [1., 0.]}


class Guard:
    def start(self):
        self.closed = False
        return self

    def check(self):
        pass

    def close(self):
        self.closed = True


class Backend:
    def __init__(self, failure=None):
        self.executed, self.closed, self.failure = [], False, failure
        self.readout_runtime = {"unit": True}

    def make_preparer(self):
        class Preparer:
            def prepare(self, item, baseline):
                assert baseline and current_thread().name.startswith("fuzev-prepare")
                return item["id"]

            def stats(self):
                return {"unit": True}
        return Preparer()

    def extract_prepared(self, item, prepared, baseline):
        assert current_thread().name.startswith("fuzev-comparison-owner")
        assert prepared == item["id"] and baseline
        if item["id"] == self.failure:
            raise ValueError("unit readout failure")
        self.executed.append(item["id"])
        return torch.full((2560,), float(item["id"])), torch.ones(2, 2560), torch.tensor([1., 0.]), {}

    def close(self):
        assert current_thread().name.startswith("fuzev-comparison-owner")
        self.closed = True


def run(tmp_path, rows, backend, **options):
    def inputs(*args):
        yield from rows
    return extract_native("native-3.05", tmp_path / "inputs", "a" * 64,
                          {"features": str(tmp_path / "reference")}, tmp_path / "comparison",
                          backend_factory=lambda _: backend,
                          identity_factory=lambda *_: {"corpus_scope": TEST_SCOPE, "unit": True},
                          input_factory=inputs, guard_factory=Guard, **options)


def test_extract_test_only_then_resume_without_repeating_gpu_work(tmp_path):
    backend = Backend()
    result = run(tmp_path, [row(0), row(1), row(2)], backend)
    assert result["committed"] == result["counts"]["examined"] == 3
    assert result["requires_independent_seal"] and not result["runtime_qualified"]
    assert result["status"] == "extracted_pending_independent_seal"
    assert backend.closed and backend.executed == ["0", "1", "2"]
    store = FeatureStore(tmp_path / "comparison", {"corpus_scope": TEST_SCOPE, "unit": True}, read_only=True)
    try:
        assert not store.preparations()
        assert {r["role"] for r in store.records()} == {"test"}
    finally:
        store.close()
    resumed = Backend()
    result = run(tmp_path, [row(0), row(1), row(2)], resumed, resume=True)
    assert result["counts"]["skipped"] == 3 and result["counts"]["records"] == 0
    assert not resumed.executed and resumed.closed


@pytest.mark.parametrize("items", [[], [row(0, "fit")]])
def test_invalid_first_input_never_loads_model_or_writes_output(tmp_path, items):
    calls = []
    with pytest.raises(ValueError):
        extract_native("native-3.05", tmp_path / "inputs", "a" * 64,
                       {"features": str(tmp_path / "reference")}, tmp_path / "output",
                       backend_factory=lambda _: calls.append(True),
                       input_factory=lambda *_: iter(items))
    assert not calls and not (tmp_path / "output").exists()


def test_freeze_error_occurs_before_model_load(tmp_path):
    calls = []
    def inputs(*args):
        raise ValueError("unfrozen reference")
        yield
    with pytest.raises(ValueError, match="unfrozen"):
        extract_native("native-3.05", "inputs", "a" * 64,
                       {"features": str(tmp_path / "reference")}, tmp_path / "output",
                       backend_factory=lambda _: calls.append(True), input_factory=inputs)
    assert not calls and not (tmp_path / "output").exists()


def test_complete_preflight_catches_late_invalid_role_before_model_load(tmp_path):
    backend = Backend()
    with pytest.raises(ValueError, match="held-out"):
        run(tmp_path, [row(0), row(1, "fit")], backend)
    assert not backend.closed and not backend.executed
    assert not (tmp_path / "comparison").exists()


def test_cleanup_failure_keeps_complete_rows_unsealable(tmp_path):
    class BrokenClose(Backend):
        def close(self):
            super().close()
            raise RuntimeError("cleanup failed")
    backend = BrokenClose()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        run(tmp_path, [row(0), row(1)], backend)
    store = FeatureStore(tmp_path / "comparison", {"corpus_scope": TEST_SCOPE, "unit": True}, read_only=True)
    try:
        assert len(store.records()) == 2
        assert store.db.execute("SELECT 1 FROM failures WHERE id=?", (RUN_FAILURE,)).fetchone()
    finally:
        store.close()
    receipt = json.loads((tmp_path / "comparison/comparison-extraction.json").read_text())
    assert receipt["status"] == "failed" and "cleanup_error" in receipt
    assert not receipt["resume_allowed"]
    with pytest.raises(ValueError, match="Resume requires"):
        run(tmp_path, [row(0), row(1)], Backend(), resume=True)


def test_error_drains_and_closes_owner_preserving_partial_commits(tmp_path):
    backend = Backend(failure="2")
    with pytest.raises(ValueError, match="readout failure"):
        run(tmp_path, [row(i) for i in range(5)], backend)
    assert backend.closed
    receipt = json.loads((tmp_path / "comparison/comparison-extraction.json").read_text())
    assert receipt["status"] == "failed"
    resumed = Backend()
    result = run(tmp_path, [row(i) for i in range(5)], resumed, resume=True)
    assert result["committed"] == 5 and result["counts"]["skipped"] >= 1


def test_cancel_keeps_partial_output_unsealed(tmp_path):
    cancel, backend = Event(), Backend()
    cancel.set()
    with pytest.raises(InterruptedError):
        run(tmp_path, [row(0)], backend, cancel=cancel)
    assert backend.closed
    receipt = json.loads((tmp_path / "comparison/comparison-extraction.json").read_text())
    assert receipt["status"] == "cancelled" and receipt["requires_independent_seal"]


@pytest.mark.parametrize("mutation", ["binding", "sealed", "existing"])
def test_reject_invalid_resume_before_gpu_work(tmp_path, mutation):
    run(tmp_path, [row(0)], Backend())
    if mutation == "binding":
        path = tmp_path / "comparison/comparison-extraction.json"
        value = json.loads(path.read_text())
        value["binding"]["inputs_manifest_sha256"] = "b" * 64
        path.write_text(json.dumps(value))
    elif mutation == "sealed":
        (tmp_path / "comparison/subset-extraction.json").write_text("{}")
    backend = Backend()
    with pytest.raises((ValueError, FileExistsError)):
        run(tmp_path, [row(0)], backend, resume=mutation != "existing")
    assert not backend.executed and not backend.closed


def test_cannot_write_into_reference(tmp_path):
    with pytest.raises(ValueError, match="training reference"):
        extract_native("native-3.05", "inputs", "a" * 64,
                       {"features": str(tmp_path)}, tmp_path)


def test_cli_requires_permission_before_reading_missing_plan(capsys):
    with pytest.raises(SystemExit):
        main(["--model", "native-3.05", "--plan", "/absent-plan.json", "--inputs", "/absent-inputs",
              "--inputs-sha256", "a" * 64, "--output", "/absent-output"])
    assert "--allow-inference required" in capsys.readouterr().err
