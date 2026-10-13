import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

SPEC = importlib.util.spec_from_file_location(
    "train_when_ready", Path(__file__).resolve().parents[1] / "tools/train_when_ready.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Guard:
    def check(self):
        pass


def test_text_waits_five_minutes_and_requires_authoritative_readiness(monkeypatch):
    sleeps, verified = [], []

    class Reader:
        guard = Guard()
        calls = 0

        def preparations(self):
            self.calls += 1
            return [{"path": "text.jsonl", "completed": 1},
                    {"path": "qasper.jsonl", "completed": self.calls > 1}]

        def require_ready(self, stage):
            verified.append(stage)
            return {"records": 379706}

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert runner.wait_for_corpus(Reader(), "text") == {"records": 379706}
    assert sleeps == [300]
    assert verified == ["text"]


def test_mixed_waits_for_images_and_does_not_accept_completion_flags_alone(monkeypatch):
    sleeps = []

    class Reader:
        guard = Guard()
        calls = 0

        def preparations(self):
            self.calls += 1
            return [{"path": name, "completed": name != "images.jsonl" or self.calls > 1}
                    for name in ("text.jsonl", "qasper.jsonl", "images.jsonl")]

        def require_ready(self, stage):
            assert stage == "mixed"
            raise ValueError("Missing accounted-for inputs")

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(ValueError, match="accounted-for"):
        runner.wait_for_corpus(Reader(), "mixed")
    assert sleeps == [300]


@pytest.mark.parametrize("stage", ["text", "mixed", "calibrate"])
def test_exact_readiness_race_retries_after_five_minutes_with_guards(monkeypatch, stage):
    sleeps, checks, attempts = [], [], []
    result = object()

    class Reader:
        class guard:
            @staticmethod
            def check():
                checks.append(len(attempts))

    def operation():
        attempts.append(stage)
        if len(attempts) == 1:
            raise ValueError("Preparation corpus changed during readiness verification; retry")
        return result

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert runner.retry_readiness_change(Reader(), stage, operation) is result
    assert attempts == [stage, stage] and checks == [0, 1] and sleeps == [300]


@pytest.mark.parametrize("error", [
    ValueError("Feature identity changed on disk"),
    ValueError("Extraction failures remain; no silently omitted training inputs are permitted"),
    ValueError("Preparation corpus changed during readiness verification; retry: unsafe"),
    RuntimeError("Preparation corpus changed during readiness verification; retry"),
    MemoryError("insufficient headroom"),
    KeyboardInterrupt(),
])
def test_retry_does_not_swallow_safety_identity_or_other_failures(monkeypatch, error):
    sleeps = []

    class Reader:
        guard = Guard()

    def operation():
        raise error

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(type(error)) as raised:
        runner.retry_readiness_change(Reader(), "mixed", operation)
    assert raised.value is error and sleeps == []


def test_guard_failure_on_retry_stops_without_rerunning_operation(monkeypatch):
    sleeps, attempts = [], []

    class Reader:
        class guard:
            @staticmethod
            def check():
                if attempts:
                    raise MemoryError("swap guard")

    def operation():
        attempts.append(1)
        raise ValueError("Preparation corpus changed during readiness verification; retry")

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(MemoryError, match="swap guard"):
        runner.retry_readiness_change(Reader(), "mixed", operation)
    assert attempts == [1] and sleeps == [300]


def test_complete_scoped_corpus_still_requires_authoritative_readiness(monkeypatch):
    sleeps, calls = [], []

    class Reader:
        scoped = True
        guard = Guard()

        def preparations(self):
            return [{"path": name, "completed": 1}
                    for name in runner.required_preparations("mixed")]

        def require_ready(self, stage):
            calls.append(stage)
            if len(calls) == 1:
                raise ValueError("Preparation corpus changed during readiness verification; retry")
            return {"records": 242092}

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert runner.wait_for_corpus(Reader(), "mixed") == {"records": 242092}
    assert calls == ["mixed", "mixed"] and sleeps == [300]


def test_supervisor_retries_fit_and_calibration_without_changing_text_warmstart(monkeypatch, tmp_path):
    import json
    import sys

    (tmp_path / "identity.json").write_text(json.dumps({"schema_version": 1}))
    sleeps, fit_calls, calibration_calls, closed = [], [], [], []

    class RuntimeGuard(Guard):
        def start(self):
            return self

        def close(self):
            closed.append("guard")

    class Reader:
        scoped = True

        def __init__(self, directory, identity, *, read_only, guard):
            assert directory == tmp_path and read_only
            self.guard = guard

        def preparations(self):
            return [{"path": name, "completed": 1}
                    for name in runner.required_preparations("mixed")]

        def require_ready(self, stage):
            return {"stage": stage, "records": 242092}

        def close(self):
            closed.append("reader")

    def fit(_features, stage, initial):
        fit_calls.append((stage, initial))
        if stage == "text":
            return "selected-text-checkpoint"
        if len(fit_calls) == 2:
            raise ValueError("Preparation corpus changed during readiness verification; retry")
        return "selected-mixed-checkpoint"

    def calibrate(_features, initial):
        calibration_calls.append(initial)
        if len(calibration_calls) == 1:
            raise ValueError("Preparation corpus changed during readiness verification; retry")
        return "frozen-reference"

    monkeypatch.setattr(sys, "argv", ["train_when_ready.py", "--features", str(tmp_path)])
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda _threads: None)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda _threads: None)
    monkeypatch.setattr(runner, "admit", lambda _bytes: None)
    monkeypatch.setattr(runner, "MemoryGuard", RuntimeGuard)
    monkeypatch.setattr(runner, "GuardedReader", Reader)
    monkeypatch.setattr(runner, "Store", lambda: object())
    monkeypatch.setattr(runner, "project_path", lambda _path: tmp_path / "progress.json")
    monkeypatch.setattr(runner, "fit", fit)
    monkeypatch.setattr(runner, "calibrate", calibrate)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner.main()
    assert fit_calls == [("text", None), ("mixed", "selected-text-checkpoint"),
                         ("mixed", "selected-text-checkpoint")]
    assert calibration_calls == ["selected-mixed-checkpoint"] * 2
    assert sleeps == [300, 300] and closed == ["reader", "guard"]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["reference"] == "frozen-reference"


@pytest.mark.parametrize("stage", ["mixed", "calibrate"])
def test_real_external_append_readiness_race_preserves_training_membership(monkeypatch, tmp_path, stage):
    import json

    import fuzev_research.features as features_module
    from fuzev_research.scope import validate_row
    from test_scope_readiness import scoped

    fixture = scoped.__wrapped__(tmp_path, monkeypatch)
    writer, _ = next(fixture)
    reader = features_module.FeatureStore(writer.directory, writer.identity, read_only=True)
    reader.guard = Guard()
    sleeps, inserted = [], []
    before = reader.require_ready(stage)
    # Force a real uncached verification on the independent reader.
    reader._readiness.clear()
    original = features_module.verify_preparation

    def verify(*args, **kwargs):
        original(*args, **kwargs)
        if not inserted:
            row = {"id": "external-append", "group": "paper:external-append", "role": "test",
                   "primitive": "noul", "modality": "text", "source": "qasper", "revision": "fixture",
                   "evaluation_scope": "external", "original_split": "test", "adaptation": "evidence",
                   "inclusion_probability": .5, "sampling_weight": 2.,
                   "source_test_example_id": "external-question:paragraph"}
            validate_row(row, "qasper-external.jsonl")
            with writer.db:
                writer.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)", (
                    row["id"], 1, 0, row["group"], "test", "noul", "text", "qasper", json.dumps(row)))
            inserted.append(row["id"])

    monkeypatch.setattr(features_module, "verify_preparation", verify)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    try:
        after = runner.retry_readiness_change(reader, stage, lambda: reader.require_ready(stage))
        assert after == before and inserted == ["external-append"] and sleeps == [300]
        assert dict(reader.db.execute(
            "SELECT role,COUNT(*) FROM records WHERE role!='test' GROUP BY role")) == {
                "fit": 4, "development": 4, "calibration": 4}
    finally:
        reader.close()
        next(fixture, None)
