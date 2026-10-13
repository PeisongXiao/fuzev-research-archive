"""Extract only sealed held-out inputs with one native comparison backbone."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import json
from pathlib import Path
import time

from .acquire import write_json
from .comparison_inputs import iter_inputs
from .evaluate import TEST_SCOPE
from .extraction import FeatureProxy, write_rows
from .features import FeatureStore, file_sha256
from .telemetry import MemoryGuard


MODELS = ("native-3.05", "native-5.05")
RECEIPT = "comparison-extraction.json"
RUN_FAILURE = "__native_comparison_extraction__"


def native_factory(model_name):
    from .native import NativeBackend
    from .paths import project_path
    model = json.loads(project_path("data/manifests/acquisition.json").read_text())["models"][model_name]
    snapshot = model.get("local_snapshot")
    if not snapshot or not Path(snapshot).is_dir():
        raise ValueError("Pinned comparison acquisition must be complete before inference")
    return NativeBackend(snapshot, batch_capacity=8, candidate_batch_size=8)


def native_identity(model_name, backend):
    from .native import scoped_feature_identity
    return {**scoped_feature_identity(model_name, backend.readout_runtime), "corpus_scope": TEST_SCOPE}


def extract_native(model_name, inputs, inputs_sha256, reference, output, *, resume=False,
                   backend_factory=native_factory, identity_factory=native_identity,
                   input_factory=iter_inputs, guard_factory=MemoryGuard, cancel=None):
    """Leave drained, test-only shards for the independent comparison sealer."""
    if model_name not in MODELS:
        raise ValueError("Only the two required native bitrate comparisons are supported")
    reference = json.loads(json.dumps(reference, allow_nan=False))
    output = Path(output).expanduser().resolve()
    if output == Path(reference["features"]).expanduser().resolve():
        raise ValueError("Never write comparison features into the training reference")
    binding = {"model": model_name, "inputs": str(Path(inputs).expanduser().resolve()),
               "inputs_manifest_sha256": inputs_sha256, "reference": reference}
    receipt_path = output / RECEIPT
    if output.exists():
        if not resume:
            raise FileExistsError("Existing comparison output requires explicit resume")
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if previous.get("binding") != binding or previous.get("resume_allowed") is not True:
            raise ValueError("Resume requires the exact frozen reference and input manifest")
        if (output / "subset-extraction.json").exists():
            raise ValueError("Never resume or mutate already sealed comparison features")
    elif resume:
        raise FileNotFoundError("Resume requires an existing extraction")

    def canonical_rows():
        stream = iter(input_factory(inputs, inputs_sha256, reference))
        try:
            for row in stream:
                if row.get("role") != "test":
                    raise ValueError("Only held-out test roles may enter comparison extraction")
                yield row
        finally:
            if hasattr(stream, "close"):
                stream.close()

    # Finish every freeze, checksum and membership check before loading a model.
    preflight = sum(1 for _ in canonical_rows())
    if not preflight:
        raise ValueError("Comparison inputs cannot be empty")
    rows = canonical_rows()
    backend = features = guard = None
    started = time.time()
    progress = None
    postflight = False
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-comparison-owner") as owner:
        try:
            first = next(rows, None)
            if first is None:
                raise ValueError("Comparison inputs cannot be empty")
            if first.get("role") != "test":
                raise ValueError("Only held-out test roles may enter comparison extraction")
            backend = owner.submit(backend_factory, model_name).result()
            identity = identity_factory(model_name, backend)
            if identity.get("corpus_scope") != TEST_SCOPE:
                raise ValueError("Require explicitly identified test-only comparison features")
            features = FeatureStore(output, identity)
            if features.preparations() or features.db.execute(
                    "SELECT 1 FROM records WHERE role!='test' OR role IS NULL LIMIT 1").fetchone():
                raise ValueError("Comparison corpus must not contain training preparations or roles")
            progress = {"schema_version": 1, "binding": binding,
                        "feature_identity_sha256": file_sha256(output / "identity.json"),
                        "started": started, "status": "extracting", "preflight_records": preflight,
                        "runtime_qualified": False, "requires_independent_seal": True,
                        "resume_allowed": False}
            write_json(receipt_path, progress)
            features.fail(RUN_FAILURE, "Readouts, postflight verification and cleanup pending")
            guard = guard_factory().start()

            def on_record(counts):
                if counts["records"] % 128 == 0:
                    progress.update(counts=counts, observed=time.time())
                    write_json(receipt_path, progress)
                    print(json.dumps({"model": model_name, **counts}), flush=True)

            counts = write_rows(chain((first,), rows), features, FeatureProxy(backend, owner),
                                guard, cancel, on_record)
            postflight = True
            if identity_factory(model_name, backend) != identity:
                raise ValueError("Comparison runtime/source identity changed during extraction")
            features.flush()
            committed = features.db.execute("SELECT count(*) FROM records").fetchone()[0]
            failures = features.db.execute("SELECT count(*) FROM failures WHERE id!=?", (RUN_FAILURE,)).fetchone()[0]
            if failures or counts["examined"] != committed or committed != preflight:
                raise ValueError("Comparison inputs have unresolved failures, missing or extra features")
            progress.update(status="draining", counts=counts, committed=committed)
            write_json(receipt_path, progress)
            return progress
        except BaseException as error:
            if progress is not None:
                progress.update(status="cancelled" if isinstance(error, (InterruptedError, KeyboardInterrupt))
                                else "failed", error=str(error), finished=time.time(),
                                resume_allowed=not postflight)
                write_json(receipt_path, progress)
            raise
        finally:
            cleanup_error = None
            cleanup = [rows.close]
            if features is not None:
                cleanup.append(features.close)
            if guard is not None:
                cleanup.append(guard.close)
            if backend is not None:
                cleanup.append(lambda: owner.submit(backend.close).result())
            for close in cleanup:
                try:
                    close()
                except BaseException as error:
                    cleanup_error = cleanup_error or error
            if cleanup_error is not None:
                if progress is not None:
                    progress.update(status="failed", cleanup_error=str(cleanup_error), finished=time.time(),
                                    resume_allowed=False)
                    write_json(receipt_path, progress)
                raise cleanup_error
            if progress is not None and progress["status"] == "draining":
                completed = FeatureStore(output, identity)
                try:
                    with completed.db:
                        completed.db.execute("DELETE FROM failures WHERE id=?", (RUN_FAILURE,))
                finally:
                    completed.close()
                progress.update(status="extracted_pending_independent_seal", finished=time.time(),
                                resume_allowed=True)
                write_json(receipt_path, progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_inference:
        parser.error("Explicit --allow-inference required; never run alongside the live extractor")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported frozen evaluation plan")
    print(json.dumps(extract_native(args.model, args.inputs, args.inputs_sha256,
                                   plan["reference"], args.output, resume=args.resume), allow_nan=False))


if __name__ == "__main__":
    main()
