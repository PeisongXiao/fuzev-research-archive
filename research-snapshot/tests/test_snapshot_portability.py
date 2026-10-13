"""Environment and endpoint behavior only; no research inputs or engines."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SNAPSHOT = Path(__file__).resolve().parents[1]


def test_project_root_uses_operator_environment(tmp_path):
    env = dict(os.environ)
    command = [sys.executable, "-c", "from fuzev_research.paths import project_path; print(project_path('reports/result.json'))"]
    env["HOME"] = str(tmp_path)
    env["FUZEV_ROOT"] = "~/operator-artifacts"
    configured = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert configured.stdout.strip() == str(tmp_path / "operator-artifacts/reports/result.json")
    assert not (tmp_path / "operator-artifacts").exists()


@pytest.mark.parametrize("relative", [
    "scripts/audit_sampling_exposure.py",
    "scripts/prepare_kev_e2e_panel.py",
    "scripts/compare_workload_robust.py",
    "scripts/probe_mature_direction.py",
    "scripts/run_workload_robust.py",
    "scripts/run_direction_queue.py",
    "scripts/benchmark_kev_text.py",
    "scripts/kev_e2e_client.py",
    "tools/check_dashboard.py",
])
def test_script_paths_use_operator_environment(tmp_path, monkeypatch, relative):
    root, source = tmp_path / "artifacts", tmp_path / "workload-source"
    monkeypatch.setenv("FUZEV_ROOT", str(root))
    monkeypatch.setenv("FUZEV_WORKLOAD_SOURCE", str(source))
    monkeypatch.delenv("FUZEV_KEV_SOURCE", raising=False)
    monkeypatch.syspath_prepend(str(SNAPSHOT / "scripts"))
    path = SNAPSHOT / relative
    name = "_portability_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.ROOT == root
    if path.stem in {"compare_workload_robust", "run_workload_robust", "run_direction_queue"}:
        assert module.TREE == source
    if path.stem == "probe_mature_direction":
        assert module.OLD_WORKER == source / "scripts/run_workload_robust.py"
    if path.stem == "benchmark_kev_text":
        assert module.KEV == root / "kev"
    assert not root.exists() and not source.exists()


@pytest.mark.parametrize("options,host,port", [
    ([], "127.0.0.1", 8000),
    (["--host", "0.0.0.0", "--port", "8765"], "0.0.0.0", 8765),
])
def test_dashboard_cli_uses_operator_endpoint(monkeypatch, options, host, port):
    import uvicorn
    from fuzev_research import cli

    calls = []
    monkeypatch.setattr(cli, "Store", lambda: SimpleNamespace(initialize=lambda: None))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    monkeypatch.setattr(sys, "argv", ["fuzev-research", "dashboard", *options])
    cli.main()
    assert len(calls) == 1
    app, kwargs = calls[0]
    assert app.title == "FuZev observability"
    assert kwargs["host"] == host and kwargs["port"] == port
