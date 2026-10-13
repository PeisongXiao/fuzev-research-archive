import json
import sqlite3
import time
from contextlib import contextmanager

from .paths import prepare, project_path


STATUSES = {"pending", "in_progress", "completed", "blocked", "failed", "cancelled"}
TASKS = [
    ("workspace", None, "Workspace and provenance"),
    ("sources", "workspace", "Independent source forks and optimized foundation"),
    ("dashboard", "workspace", "Dashboard and durable telemetry"),
    ("contracts", None, "Portable decision contracts and head lifecycle"),
    ("native", None, "Native fused runtime"),
    ("native.tap", "native", "Post-collapse feature capture without vocabulary projection"),
    ("native.prefix", "native", "Exact-prefix continuation and branch isolation"),
    ("native.vision", "native", "Image processing and multimodal prefix identity"),
    ("vllm", None, "vLLM fused runtime"),
    ("vllm.runner", "vllm", "Task-aware runner and scheduler integration"),
    ("vllm.exl3", "vllm", "Packed EXL3 loading and NVMe PLE adapter"),
    ("data", None, "Dataset acquisition and separated manifests"),
    ("data.models", "data", "Pinned reference and comparison weight acquisition"),
    ("data.models.reference", "data.models", "Native 4.05-bpw training reference"),
    ("data.models.3", "data.models", "Native 3.05-bpw comparison"),
    ("data.models.5", "data.models", "Native 5.05-bpw comparison"),
    ("data.models.nvidia", "data.models", "NVIDIA NVFP4 comparison"),
    ("data.models.radixark", "data.models", "RadixArk NVFP4 comparison"),
    ("data.images", "data", "Still-image dataset acquisition"),
    ("data.audit", "data", "Pinned data provenance and training eligibility"),
    ("data.splits", "data", "Cross-corpus deduplication and grouped splits"),
    ("data.splits.text", "data.splits", "HelpSteer conversation-group splits"),
    ("data.splits.boolq", "data.splits", "BoolQ truth-label passage-group splits"),
    ("data.splits.qasper", "data.splits", "External official-test QASPER sample"),
    ("data.splits.external", "data.splits", "External QASPER and paired VL-RewardBench evaluation sets"),
    ("data.scope", "data", "Active dataset scope transition and compatible feature reuse"),
    ("data.splits.images", "data.splits", "Image clusters and exclusive visual heldouts"),
    ("data.features", "data", "Frozen-backbone text and image feature extraction"),
    ("training", None, "Reference head training"),
    ("training.text", "training", "Text-only warmup"),
    ("training.multimodal", "training", "Mixed text/image training with text replay"),
    ("training.calibration", "training", "Calibration-only split and frozen bundle"),
    ("validation", None, "Correctness and quality qualification"),
    ("validation.correctness", "validation", "Tokens, hybrid state, branches, images and failures"),
    ("validation.transfer", "validation", "Frozen-head transfer across six required cells"),
    ("validation.performance", "validation", "Matched generation, decision and mixed workloads"),
    ("release", None, "Reports and reviewable local release"),
    ("release.cards", "release", "Model cards, notices and final report"),
    ("release.artifacts", "release", "Hugging Face-ready trained head bundles"),
    ("release.cleanup", "release", "Stop inference and retain observability"),
]


class Store:
    def __init__(self, path=None):
        prepare()
        self.path = path or project_path("state/project.sqlite3")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                  id TEXT PRIMARY KEY, parent TEXT REFERENCES tasks(id), title TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending', progress REAL NOT NULL DEFAULT 0,
                  detail TEXT NOT NULL DEFAULT '', updated REAL NOT NULL, started REAL,
                  finished REAL, attempt INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                  sequence INTEGER PRIMARY KEY, timestamp REAL NOT NULL, task TEXT,
                  kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metrics (
                  timestamp REAL PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS results (
                  id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, task TEXT NOT NULL,
                  payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_time ON events(timestamp);
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, unit TEXT NOT NULL, task TEXT NOT NULL,
                  validator TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                  observed REAL, detail TEXT NOT NULL DEFAULT '');
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        with self.connect() as db:
            db.executemany("INSERT OR IGNORE INTO tasks(id,parent,title,updated) VALUES(?,?,?,?)",
                           [(key, parent, title, time.time()) for key, parent, title in TASKS])

    def update(self, task, status, detail="", progress=None):
        if status == "running":
            status = "in_progress"
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status}")
        if progress is not None and not 0 <= progress <= 1:
            raise ValueError("Progress must be between zero and one")
        now = time.time()
        with self.connect() as db:
            old = db.execute("SELECT * FROM tasks WHERE id=?", (task,)).fetchone()
            if old is None:
                raise ValueError(f"Unknown task: {task}")
            children = db.execute("SELECT status FROM tasks WHERE parent=?", (task,)).fetchall()
            if status == "completed" and any(c["status"] != "completed" for c in children):
                raise ValueError("A parent cannot complete with unfinished children")
            value = 1 if status == "completed" else (old["progress"] if progress is None else progress)
            started = now if status == "in_progress" and old["status"] not in {"running", "in_progress"} else old["started"]
            finished = now if status in {"completed", "failed", "cancelled"} else None
            attempt = old["attempt"] + int(status == "in_progress" and old["status"] not in {"running", "in_progress"})
            db.execute("UPDATE tasks SET status=?,detail=?,progress=?,updated=?,started=?,finished=?,attempt=? WHERE id=?",
                       (status, detail, value, now, started, finished, attempt, task))
            db.execute("INSERT INTO events(timestamp,task,kind,payload) VALUES(?,?,?,?)",
                       (now, task, "task", json.dumps({"status": status, "detail": detail, "progress": value})))

    def record_metric(self, payload):
        now = payload["timestamp"]
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO metrics VALUES(?,?)", (now, json.dumps(payload)))
            db.execute("DELETE FROM metrics WHERE timestamp<?", (now - 86400,))

    def record_result(self, task, payload):
        with self.connect() as db:
            if db.execute("SELECT 1 FROM tasks WHERE id=?", (task,)).fetchone() is None:
                raise ValueError(f"Unknown task: {task}")
            db.execute("INSERT INTO results(timestamp,task,payload) VALUES(?,?,?)",
                       (time.time(), task, json.dumps(payload, allow_nan=False)))

    def register_job(self, job, unit, task, validator):
        with self.connect() as db:
            if db.execute("SELECT 1 FROM tasks WHERE id=?", (task,)).fetchone() is None:
                raise ValueError(f"Unknown task: {task}")
            db.execute("INSERT OR IGNORE INTO jobs(id,unit,task,validator) VALUES(?,?,?,?)",
                       (job, unit, task, validator))

    def jobs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY rowid")]

    def observe_job(self, job, status, detail):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,observed=?,detail=? WHERE id=?",
                       (status, time.time(), detail, job))

    def snapshot(self):
        with self.connect() as db:
            tasks = [dict(row) for row in db.execute("SELECT * FROM tasks ORDER BY rowid")]
            children = {}
            for task in tasks:
                task["own_status"] = "in_progress" if task["status"] == "running" else task["status"]
                task["status"] = task["own_status"]
                children.setdefault(task["parent"], []).append(task)

            def aggregate(task):
                descendants = children.get(task["id"], [])
                for child in descendants:
                    aggregate(child)
                if not descendants:
                    task["active_descendants"] = []
                    return
                statuses = [child["status"] for child in descendants]
                if task["own_status"] == "in_progress" or "in_progress" in statuses:
                    task["status"] = "in_progress"
                elif "failed" in statuses:
                    task["status"] = "failed"
                elif "blocked" in statuses:
                    task["status"] = "blocked"
                elif all(status == "completed" for status in statuses):
                    task["status"] = "completed"
                elif task["own_status"] == "completed":
                    task["status"] = "pending"
                task["progress"] = sum(child["progress"] for child in descendants) / len(descendants)
                active = []
                for child in descendants:
                    if child["own_status"] == "in_progress":
                        active.append(child["id"])
                    active.extend(child["active_descendants"])
                task["active_descendants"] = active

            for task in children.get(None, []):
                aggregate(task)
            results = [{**dict(row), "payload": json.loads(row["payload"])}
                       for row in db.execute("SELECT * FROM results ORDER BY id DESC LIMIT 500")]
            row = db.execute("SELECT payload FROM metrics ORDER BY timestamp DESC LIMIT 1").fetchone()
            events = [dict(row) for row in db.execute("SELECT sequence,timestamp,task,kind,payload FROM events ORDER BY sequence DESC LIMIT 30")]
            return {"tasks": tasks, "results": results, "metrics": json.loads(row[0]) if row else None,
                    "events": events, "timestamp": time.time()}

    def metrics(self, since=0, limit=900):
        with self.connect() as db:
            return [json.loads(row[0]) for row in reversed(db.execute(
                "SELECT payload FROM metrics WHERE timestamp>=? ORDER BY timestamp DESC LIMIT ?",
                (since, min(limit, 86400))).fetchall())]
