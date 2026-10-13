"""Bounded CPU mocks only: no engine, inference, datasets, or native evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from threading import get_ident
from types import ModuleType, SimpleNamespace

import pytest

from fuzev_research import native_qualification_cli as cli


def operator_case(name="operator"):
    return {"name": name, "input_origin": "synthetic CPU CLI fixture; no dataset",
            "input_role": "operator-non-heldout",
            "generation": {"model": cli.MODEL, "prompt": "A", "max_tokens": 3},
            "decisions": [{"primitive": "choice", "question": f"C{index}?",
                           "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]}
                          for index in range(4)]}


@pytest.fixture
def operator(tmp_path, monkeypatch):
    for name in ("torch", "transformers", "exllamav3", "fuzev_research.server", "fuzev_research.owner_queue"):
        monkeypatch.setitem(sys.modules, name, None)
    from fuzev_research.acquire import MODELS
    from fuzev_research import native_qualification, paths

    snapshot = tmp_path / "empty-model-fixture"
    snapshot.mkdir()
    sources = {"recipes": "mock-clean-commit", "exllamav3-fuzev": "mock-native-commit"}
    model = {"repo": MODELS[cli.MODEL][0], "revision": MODELS[cli.MODEL][1],
             "local_snapshot": str(snapshot)}
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({"models": {cli.MODEL: model}}))

    def project_path(relative):
        assert relative == "data/manifests/acquisition.json"
        return acquisition

    monkeypatch.setattr(paths, "project_path", project_path)
    config = {"head_directory": str(tmp_path / "unopened-head-fixture"),
              "manifest_sha256": "a" * 64, "calibration_sha256": "b" * 64,
              "feature_identity": {"model": model["repo"], "revision": model["revision"],
                                   "readout_runtime": {
                                       "candidate_batch_size": 8, "batch_capacity": 8,
                                       "model_batch_capacity": 8, "max_hybrid_slots": 8,
                                       "cache_tokens": 294912, "memory_fraction": .85,
                                       "optimized_foundation": True}},
              "service_sources": sources}
    cases = {"cases": [operator_case()], "observe_native_state": True}
    state = SimpleNamespace(
        config=config, cases=cases, model=model, acquisition=acquisition,
        config_path=tmp_path / "service.json", cases_path=tmp_path / "cases.json",
        output=tmp_path / "unique-report.json", events=[], threads=[], sources=sources,
        backend_error=None, build_error=None, run_error=None,
        service_close_error=None, backend_close_error=None,
        report={"status": "incomplete", "execution_status": "passed", "qualified": False,
                "evidence_kind": "native", "cleanup_errors": [],
                "metadata": {"synthetic_cli_mock": True},
                "gates": {"complete_hybrid_state": {"status": "incomplete"},
                          "accepted_boundaries": {"status": "incomplete"}}})

    def event(name):
        state.events.append(name)
        state.threads.append(get_ident())

    def backend_close():
        event("backend.close")
        if state.backend_close_error:
            raise state.backend_close_error

    def service_close():
        event("service.close")
        if state.service_close_error:
            raise state.service_close_error

    state.backend = SimpleNamespace(close=backend_close)
    state.service = SimpleNamespace(close=service_close)

    def backend(directory, **options):
        event("backend")
        state.directory, state.options = directory, options
        if state.backend_error:
            raise state.backend_error
        return state.backend

    def runtime_sources():
        event("sources")
        return state.sources

    def build_service(backend, name, model, sources, config):
        event("service")
        assert backend is state.backend and name == cli.MODEL
        assert model == state.model and sources == state.sources
        state.service_config = config
        if state.build_error:
            raise state.build_error
        return state.service

    def run_qualification(service, config, **flags):
        event("qualification")
        assert service is state.service
        state.qualification_config, state.flags = config, flags
        if state.run_error:
            raise state.run_error
        return state.report

    native = ModuleType("fuzev_research.native")
    native.NativeBackend, native.runtime_sources = backend, runtime_sources
    service = ModuleType("fuzev_research.native_service")
    service.build_service = build_service
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, service.__name__, service)
    monkeypatch.setattr(native_qualification, "run_qualification", run_qualification)

    def invoke():
        state.config_path.write_text(json.dumps(state.config))
        state.cases_path.write_text(json.dumps(state.cases))
        return cli.main(["--service-config", str(state.config_path),
                         "--cases", str(state.cases_path), "--output", str(state.output),
                         "--allow-inference"])

    state.invoke = invoke
    return state


@pytest.mark.parametrize("arguments,expected", [
    ([], 2), (["--help"], 0),
    (["--service-config", "service", "--cases", "cases", "--output", "output"], 2),
    (["--allow-inference", "--cases", "cases", "--output", "output"], 2),
    (["--allow-inference", "--service-config", "service", "--output", "output"], 2),
    (["--allow-inference", "--service-config", "service", "--cases", "cases"], 2),
    (["--service-config", "service", "--cases", "cases", "--output", "output", "--allow-i"], 2),
    (["--service-config", "service", "--cases", "cases", "--output", "output",
      "--allow-inference", "--cpu-fake"], 2),
    (["--service-config", "service", "--cases", "cases", "--output", "output",
      "--allow-inference", "--model", "native-3.05"], 2),
    (["--service-config", "service", "--cases", "cases", "--output", "output",
      "--allow-inference", "--atol", ".1"], 2),
])
def test_arguments_and_help_need_no_consumer_or_engine_imports(arguments, expected):
    result = subprocess.run([sys.executable, "-c", """
import sys
for name in ('torch', 'transformers', 'exllamav3', 'fuzev_research.native',
             'fuzev_research.native_service', 'fuzev_research.native_qualification', 'fuzev_research.acquire'):
    sys.modules[name] = None
from fuzev_research.native_qualification_cli import main
raise SystemExit(main())
""", *arguments], text=True, capture_output=True, timeout=10)
    assert result.returncode == expected, result.stderr
    assert "ImportError" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_one_offline_owner_uses_frozen_b8_and_preserves_report(operator):
    from fuzev_research import telemetry
    from fuzev_research.native_qualification import QualificationConfig
    from fuzev_research.service_contracts import ServiceConfig

    guard, admit = telemetry.MemoryGuard, telemetry.admit
    original = deepcopy(operator.report)
    assert operator.invoke() == 0
    assert operator.events == ["sources", "backend", "service", "qualification",
                               "service.close", "backend.close"]
    assert set(operator.threads) == {get_ident()}
    assert operator.directory == operator.model["local_snapshot"]
    assert operator.options == {"candidate_batch_size": 8, "batch_capacity": 8,
                                "cache_tokens": 294912, "memory_fraction": .85, "optimized": True}
    assert operator.flags == {"cpu_fake": False, "allow_inference": True}
    assert operator.service_config == ServiceConfig.model_validate(operator.config)
    assert operator.qualification_config == QualificationConfig.model_validate(operator.cases)
    saved = json.loads(operator.output.read_text())
    assert {key: saved[key] for key in original} == original == operator.report
    assert saved["source_provenance"]["observed"] == operator.sources
    assert telemetry.MemoryGuard is guard and telemetry.admit is admit


def test_full_window_image_and_state_controls_are_forwarded_unchanged(operator):
    from fuzev_research.native_qualification import QualificationConfig

    operator.cases.update(
        full_window={"case": operator_case("full"), "additional_peak_bytes": 1024,
                     "estimate_basis": "CPU fixture only; no allocation"}, run_full_window=True,
        image_controls=[{
            "name": "image", "input_origin": "synthetic CPU fixture", "input_role": "operator-non-heldout",
            "request": {"model": cli.MODEL, "primitive": "choice", "question": "C?",
                        "messages": [{"role": "user", "content": "A"}],
                        "image": {"path": "/unopened-image.png", "sha256": "c" * 64},
                        "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]},
            "mismatched_image": {"path": "/unopened-other-image.png", "sha256": "d" * 64},
            "expected_id": "a"}], atol=1e-6, rtol=1e-5)
    assert operator.invoke() == 0
    assert operator.qualification_config == QualificationConfig.model_validate(operator.cases)


@pytest.mark.parametrize("symlink", [False, True])
def test_output_is_never_overwritten_or_followed(operator, symlink):
    target = operator.output
    if symlink:
        target = operator.output.with_name("existing-target.json")
        operator.output.symlink_to(target)
    target.write_text("original")
    assert operator.invoke() == 1
    assert target.read_text() == "original"
    assert operator.events == []


def test_missing_output_parent_refuses_before_backend(operator):
    operator.output = operator.output.parent / "missing" / "report.json"
    assert operator.invoke() == 1
    assert operator.events == []


def test_actual_source_mismatch_is_recorded(operator):
    operator.sources = {**operator.sources, "recipes": "different-actual-commit"}
    assert operator.invoke() == 0
    report = json.loads(operator.output.read_text())
    assert report["source_provenance"] == {"observed": operator.sources,
                                           "recorded": operator.config["service_sources"]}


@pytest.mark.parametrize("key,value", [
    ("batch_capacity", 4), ("candidate_batch_size", 4), ("model_batch_capacity", 4),
    ("max_hybrid_slots", 4), ("batch_capacity", True), ("cache_tokens", 0),
    ("cache_tokens", True), ("memory_fraction", True),
    ("memory_fraction", float("nan")), ("optimized_foundation", False),
])
def test_unsupported_frozen_geometry_refuses_before_backend(operator, key, value):
    operator.config["feature_identity"]["readout_runtime"][key] = value
    assert operator.invoke() == 1
    assert operator.events == []


@pytest.mark.parametrize("key,value", [("local_snapshot", None), ("local_snapshot", "/missing-model")])
def test_missing_acquisition_refuses_before_backend(operator, key, value):
    operator.model[key] = value
    operator.acquisition.write_text(json.dumps({"models": {cli.MODEL: operator.model}}))
    assert operator.invoke() == 1
    assert operator.events == []


@pytest.mark.parametrize("change", ["model", "role", "tolerance", "unknown"])
def test_cases_reject_other_models_fake_inputs_loose_tolerance_and_unknown_fields(operator, change):
    if change == "model":
        operator.cases["cases"][0]["generation"]["model"] = "native-3.05"
    elif change == "role":
        operator.cases["cases"][0]["input_role"] = "cpu-fake"
    elif change == "tolerance":
        operator.cases["atol"] = .1
    else:
        operator.cases["skip_exact_state"] = True
    assert operator.invoke() == 1
    assert operator.events == []


@pytest.mark.parametrize("status", ["failed", "cancelled", "memory_infeasible", "incomplete"])
def test_failed_execution_writes_the_returned_report_and_closes(operator, status):
    operator.report.update(status=status, execution_status="failed", error="synthetic refusal")
    assert operator.invoke() == 1
    saved = json.loads(operator.output.read_text())
    assert {key: saved[key] for key in operator.report} == operator.report
    assert operator.events[-2:] == ["service.close", "backend.close"]


@pytest.mark.parametrize("phase", ["backend", "build", "run"])
def test_constructor_and_runner_exceptions_close_only_owned_resources(operator, phase, capsys):
    setattr(operator, phase + "_error", ValueError("synthetic pin/runtime refusal"))
    assert operator.invoke() == 1
    closes = [event for event in operator.events if event.endswith(".close")]
    assert closes == {"backend": [], "build": ["backend.close"],
                      "run": ["service.close", "backend.close"]}[phase]
    assert "synthetic pin/runtime refusal" in capsys.readouterr().err


@pytest.mark.parametrize("phase", ["service", "backend"])
def test_close_failure_is_nonzero_without_suppressing_other_close_or_rewriting_report(operator, phase):
    setattr(operator, phase + "_close_error", RuntimeError("synthetic close failure"))
    assert operator.invoke() == 1
    assert operator.events[-2:] == ["service.close", "backend.close"]
    saved = json.loads(operator.output.read_text())
    assert {key: saved[key] for key in operator.report} == operator.report


def test_keyboard_interrupt_still_closes_both_resources(operator):
    operator.run_error = KeyboardInterrupt()
    assert operator.invoke() == 130
    assert operator.events[-2:] == ["service.close", "backend.close"]


def test_nonfinite_report_serialization_is_nonzero_and_still_closes(operator):
    operator.report["bad_mock_value"] = float("nan")
    assert operator.invoke() == 1
    assert operator.events[-2:] == ["service.close", "backend.close"]
    assert operator.output.read_text() == ""


def test_output_write_failure_is_nonzero_and_still_closes(operator, monkeypatch, capsys):
    original_open = Path.open

    class FailedOutput:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *_exception):
            self.handle.close()

        def write(self, _value):
            raise OSError("synthetic full output device")

    def open_output(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return FailedOutput(handle) if path == operator.output else handle

    monkeypatch.setattr(Path, "open", open_output)
    assert operator.invoke() == 1
    assert operator.events[-2:] == ["service.close", "backend.close"]
    assert "synthetic full output device" in capsys.readouterr().err
