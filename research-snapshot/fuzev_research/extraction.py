"""Bounded CPU preparation and ordered writes around one GPU-owning queue."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from functools import partial


PIPELINE_POLICY = {
    "version": "ordered-extraction-lookahead-v1",
    "cpu_workers": 1,
    "prepared_lookahead_rows": 1,
    "gpu_owner_threads": 1,
    "gpu_inflight_rows": 1,
    "writer": "caller owns SQLite; next readout overlaps previous owned CPU result write",
    "record_order": "unchanged; preparation boundaries drain; no padding or regrouping",
    "shard_rows": 128,
}


class FeatureProxy:
    def __init__(self, backend, executor, queue=None):
        self.backend, self.executor = backend, executor
        self.queue = queue
        self.preparer = None

    def _submit(self, function, *args):
        if self.queue is not None:
            return self.queue.submit(partial(function, *args))
        return self.executor.submit(function, *args)

    def extract(self, row, baseline=True):
        return self._submit(self.backend.extract, row, baseline).result()

    def submit_prepared(self, row, prepared, baseline=True):
        return self._submit(self.backend.extract_prepared, row, prepared, baseline)

    def make_preparer(self):
        # One extraction job owns this proxy. File boundaries drain its worker,
        # so its CPU-only tokenizer can be reused without concurrent mutation.
        if self.preparer is None:
            self.preparer = self.backend.make_preparer()
        return self.preparer

    @property
    def readout_runtime(self):
        return dict(self.backend.readout_runtime)


class PreparedRows:
    """At most one preparing/ready row beyond the row held by the consumer."""

    def __init__(self, rows, factory, check):
        self.rows, self.factory, self.check = iter(rows), factory, check
        self.executor = self.future = self.preparer = None

    def _next(self):
        self.check()
        if self.preparer is None:
            self.preparer = self.factory()
        row = next(self.rows, None)
        if row is None:
            return None
        self.check()
        encoded = self.preparer.prepare(row, baseline=True)
        self.check()
        return row, encoded

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-prepare")
        self.future = self.executor.submit(self._next)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        value = self.future.result()
        if value is None:
            raise StopIteration
        self.future = self.executor.submit(self._next)
        return value

    def __exit__(self, *exc):
        self.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        # Close JSONL handles even on cancellation; no detached CPU producer.
        if hasattr(self.rows, "close"):
            self.rows.close()


def write_rows(rows, features, backend, guard, cancel=None, on_record=None):
    """Commit in input order, draining all workers before returning or raising.

    CPU preparation may be one row ahead of the GPU. The next GPU readout starts
    only once the previous result is owned CPU storage, then overlaps its write.
    Cancellation commits the current readout; other failures retain prior commits
    and discard/drain any lookahead result for a reproducible resume.
    """
    counts = {"examined": 0, "records": 0, "skipped": 0, "failures": 0}
    pending = None

    def check():
        guard.check()
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Feature job cancelled at a committed record boundary")

    def resolve(item):
        row, future = item
        try:
            return row, future.result(), None
        except ValueError as error:
            if "exceeds the native" not in str(error):
                raise
            return row, None, str(error)

    def commit(result):
        row, tensors, error = result
        guard.check()
        if error is not None:
            features.fail(row["id"], error)
            counts["failures"] += 1
            return
        features.add(row, *tensors)
        counts["records"] += 1
        if on_record is not None:
            on_record(counts)

    prepared = PreparedRows(rows, backend.make_preparer, check)
    try:
        with prepared:
            for row, encoded in prepared:
                check()
                counts["examined"] += 1
                if (features.contains(row["id"]) or
                        pending is not None and pending[0]["id"] == row["id"]):
                    counts["skipped"] += 1
                    continue
                previous = resolve(pending) if pending is not None else None
                pending = None
                # A cancellation received during the previous GPU forward must
                # commit it, but must not enqueue another row.
                try:
                    check()
                except InterruptedError:
                    if previous is not None:
                        commit(previous)
                    raise
                pending = row, backend.submit_prepared(row, encoded, baseline=True)
                if previous is not None:
                    commit(previous)
            if pending is not None:
                result = resolve(pending)
                pending = None
                commit(result)
            check()
    except InterruptedError:
        if pending is not None:
            result = resolve(pending)
            pending = None
            commit(result)
        raise
    finally:
        if pending is not None:
            _, future = pending
            future.cancel()
            try:
                future.result()
            except (Exception, CancelledError):
                pass
    counts["cpu_preparation"] = prepared.preparer.stats() if prepared.preparer else {}
    return counts
