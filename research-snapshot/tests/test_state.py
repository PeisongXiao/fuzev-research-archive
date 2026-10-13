import pytest
from fastapi.testclient import TestClient

from fuzev_research.dashboard import create_app
from fuzev_research.state import Store


def test_state_survives_reopen_and_parent_guard(tmp_path):
    path = tmp_path / "state.sqlite"
    store = Store(path)
    store.initialize()
    with pytest.raises(ValueError, match="unfinished"):
        store.update("workspace", "completed")
    store.update("sources", "running", "Importing pinned sources", .5)
    assert Store(path).snapshot()["tasks"][1]["progress"] == .5
    store.update("sources", "completed")
    store.update("dashboard", "completed")
    store.update("workspace", "completed")


def test_dashboard_read_only_and_telemetry(tmp_path):
    store = Store(tmp_path / "state.sqlite")
    with TestClient(create_app(store)) as client:
        assert client.get("/health").json()["read_only"]
        assert client.post("/api/state", json={}).status_code == 405
        assert client.get("/").status_code == 200
        assert client.get("/api/state").json()["tasks"]
        assert client.get("/api/metrics").status_code == 200


def test_in_progress_propagates_and_clears_on_every_ancestor(tmp_path):
    path = tmp_path / "state.sqlite"
    store = Store(path)
    store.initialize()
    with store.connect() as db:
        db.execute(
            "INSERT INTO tasks(id,parent,title,updated) VALUES(?,?,?,?)",
            ("native.tap.check", "native.tap", "Validate feature capture", 0),
        )
    store.update("native.tap.check", "in_progress", "Checking features", .4)
    tasks = {task["id"]: task for task in Store(path).snapshot()["tasks"]}
    for parent in ("native.tap", "native"):
        assert tasks[parent]["status"] == "in_progress"
        assert tasks[parent]["active_descendants"] == ["native.tap.check"]
        assert tasks[parent]["own_status"] == "pending"
    store.update("native.tap.check", "completed")
    tasks = {task["id"]: task for task in store.snapshot()["tasks"]}
    assert tasks["native.tap"]["status"] == "completed"
    assert tasks["native"]["status"] == "pending"
    assert not tasks["native"]["active_descendants"]


def test_active_siblings_keep_parent_in_progress(tmp_path):
    store = Store(tmp_path / "state.sqlite")
    store.initialize()
    store.update("native.tap", "in_progress", "Capturing final mixer features", .4)
    store.update("native.prefix", "running", "Checking branch isolation", .2)
    store.update("native.tap", "completed")
    tasks = {task["id"]: task for task in store.snapshot()["tasks"]}
    assert tasks["native"]["status"] == "in_progress"
    assert tasks["native"]["active_descendants"] == ["native.prefix"]
    assert tasks["native.prefix"]["status"] == "in_progress"
    store.update("native.prefix", "blocked", "Waiting for a checkpoint")
    tasks = {task["id"]: task for task in store.snapshot()["tasks"]}
    assert tasks["native"]["status"] == "blocked"
    assert not tasks["native"]["active_descendants"]


def test_dashboard_api_returns_inherited_in_progress(tmp_path):
    store = Store(tmp_path / "state.sqlite")
    store.initialize()
    store.update("data.features", "in_progress", "Extracting image features", .1)
    with TestClient(create_app(store)) as client:
        tasks = {task["id"]: task for task in client.get("/api/state").json()["tasks"]}
        assert tasks["data"]["status"] == "in_progress"
        assert tasks["data"]["active_descendants"] == ["data.features"]
