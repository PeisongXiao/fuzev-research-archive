from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.extraction import FeatureProxy, write_rows
from fuzev_research.features import FeatureStore
from fuzev_research.owner_queue import OwnerQueue


def row(index):
    return {"id": str(index), "group": str(index), "role": "fit", "primitive": "choice",
            "modality": "text", "source": "synthetic", "revision": "fixture",
            "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
            "target": [1., 0.]}


class Guard:
    def check(self):
        pass


class Backend:
    def __init__(self, prepare_hook=None, gpu_hook=None):
        self.prepared, self.executed, self.threads = [], [], []
        self.prepare_hook, self.gpu_hook = prepare_hook, gpu_hook
        self.buffer = torch.zeros(2560)
    def make_preparer(self):
        outer = self
        class Preparer:
            def prepare(self, item, baseline):
                assert baseline
                assert current_thread().name.startswith("fuzev-prepare")
                outer.prepared.append(item["id"])
                if outer.prepare_hook:
                    outer.prepare_hook(item)
                return item["id"]
            def stats(self):
                return {"rows": len(outer.prepared)}
        return Preparer()
    def extract_prepared(self, item, prepared, baseline):
        assert prepared == item["id"] and baseline
        self.threads.append(current_thread().name)
        self.executed.append(item["id"])
        if self.gpu_hook:
            self.gpu_hook(item)
        self.buffer.fill_(int(item["id"]))
        return self.buffer.clone(), self.buffer.expand(2, -1).clone(), torch.tensor([1., 0.]), {"batch": [2]}


@pytest.mark.parametrize("shared_queue", [False, True])
def test_cpu_and_writer_overlap_gpu_with_one_owner_and_owned_results(tmp_path, shared_queue):
    gpu0, prepared1, gpu1, wrote0 = (Event() for _ in range(4))
    def prepare(item):
        if item["id"] == "1":
            assert gpu0.wait(5)
            prepared1.set()
    def gpu(item):
        if item["id"] == "0":
            gpu0.set()
            assert prepared1.wait(5)
        if item["id"] == "1":
            gpu1.set()
            assert wrote0.wait(5)
    backend = Backend(prepare, gpu)
    features = FeatureStore(tmp_path, {"unit": True}, shard_size=1)
    add = features.add
    def write(item, *tensors):
        assert current_thread().name == "MainThread"
        if item["id"] == "0":
            assert gpu1.wait(5), "Writer failed to overlap the next GPU readout"
            wrote0.set()
        add(item, *tensors)
    features.add = write
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="unit-gpu") as owner:
        queue = OwnerQueue(owner) if shared_queue else None
        result = write_rows(map(row, range(3)), features, FeatureProxy(backend, owner, queue), Guard())
    assert backend.executed == ["0", "1", "2"]
    assert set(backend.threads) == {"unit-gpu_0"}
    assert result["records"] == result["examined"] == 3
    for item in features.records():
        assert torch.equal(features.tensors(item)["query"], torch.full((2560,), float(item["id"])))
    features.close()


def test_ordered_default_128_row_shards_and_resume_duplicates(tmp_path):
    backend, features = Backend(), FeatureStore(tmp_path, {"unit": True})
    with ThreadPoolExecutor(max_workers=1) as owner:
        proxy = FeatureProxy(backend, owner)
        items = [row(0), row(0), *map(row, range(1, 130))]
        result = write_rows(items, features, proxy, Guard())
        features.flush()
        assert result["records"] == 130 and result["skipped"] == 1
        assert backend.executed == list(map(str, range(130)))
        assert features.db.execute("SELECT COUNT(*) FROM records WHERE shard=0").fetchone()[0] == 128
        assert [item["id"] for item in features.records()] == list(map(str, range(130)))
        result = write_rows(items, features, proxy, Guard())
        assert result["records"] == 0 and result["skipped"] == 131
        assert len(backend.executed) == 130
    features.close()


def test_cancel_drains_and_commits_current_without_queueing_next(tmp_path):
    started, cancel, finished = Event(), Event(), Event()
    def prepare(item):
        if item["id"] == "1":
            assert started.wait(5)
            cancel.set()
    def gpu(item):
        started.set()
        assert cancel.wait(5)
        finished.set()
    backend, features = Backend(prepare, gpu), FeatureStore(tmp_path, {"unit": True})
    with ThreadPoolExecutor(max_workers=1) as owner:
        with pytest.raises(InterruptedError, match="committed"):
            write_rows(map(row, range(3)), features, FeatureProxy(backend, owner), Guard(), cancel)
        assert finished.is_set()
        assert backend.executed == ["0"]
    features.close()
    reopened = FeatureStore(tmp_path, {"unit": True})
    assert [item["id"] for item in reopened.records()] == ["0"]
    reopened.close()


def test_writer_error_drains_inflight_result_without_out_of_order_commit(tmp_path):
    started, release, finished = Event(), Event(), Event()
    def gpu(item):
        if item["id"] == "1":
            started.set()
            assert release.wait(5)
            finished.set()
    backend, features = Backend(gpu_hook=gpu), FeatureStore(tmp_path, {"unit": True})
    def fail(*_):
        assert started.wait(5)
        release.set()
        raise OSError("writer failure")
    features.add = fail
    with ThreadPoolExecutor(max_workers=1) as owner:
        with pytest.raises(OSError, match="writer failure"):
            write_rows(map(row, range(3)), features, FeatureProxy(backend, owner), Guard())
        assert finished.is_set()
    assert features.records() == []
    features.close()


@pytest.mark.parametrize("failure", ["prepare", "gpu", "guard"])
def test_failure_never_leaves_a_gpu_owner_running(tmp_path, failure):
    finished = Event()
    def prepare(item):
        if item["id"] == "1" and failure == "prepare":
            raise RuntimeError("prepare failure")
    def gpu(item):
        finished.set()
        if item["id"] == "1" and failure == "gpu":
            raise RuntimeError("gpu failure")
    class FailGuard(Guard):
        def check(self):
            if failure == "guard" and finished.is_set():
                raise MemoryError("headroom guard")
    backend, features = Backend(prepare, gpu), FeatureStore(tmp_path, {"unit": True})
    with ThreadPoolExecutor(max_workers=1) as owner:
        with pytest.raises((RuntimeError, MemoryError)):
            write_rows(map(row, range(3)), features, FeatureProxy(backend, owner), FailGuard())
        # FIFO sentinel proves previously submitted work was settled on return.
        before = list(backend.executed)
        owner.submit(lambda: None).result(timeout=5)
        assert backend.executed == before
    features.close()


def test_out_of_window_is_recorded_not_silently_dropped(tmp_path):
    def gpu(item):
        if item["id"] == "1":
            raise ValueError("Record exceeds the native 262144-token window")
    backend, features = Backend(gpu_hook=gpu), FeatureStore(tmp_path, {"unit": True})
    with ThreadPoolExecutor(max_workers=1) as owner:
        result = write_rows(map(row, range(3)), features, FeatureProxy(backend, owner), Guard())
    features.flush()
    assert result["failures"] == 1 and result["records"] == 2
    assert [item["id"] for item in features.records()] == ["0", "2"]
    assert features.db.execute("SELECT id FROM failures").fetchone()[0] == "1"
    features.close()
