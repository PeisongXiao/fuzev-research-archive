"""Exact complete test-only inputs, prepared only after the reference is frozen."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile

from .acquire import write_json
from .evaluate import (canonical_test_membership, check_hashes, dataset_scope,
                       load_reference, open_features, read_json, require_final_ready,
                       semantic_fields, sha256, test_rows)
from .features import file_sha256
from .scope import validate_row


MANIFEST = "comparison-inputs.json"
INPUTS = "test.jsonl"
MAX_ROW_BYTES = 16 << 20


@contextmanager
def frozen_reference(spec):
    reference = open_features(spec["features"], spec["feature_identity_sha256"])
    try:
        # Loading verifies calibrated mixed-stage provenance, not predictions.
        bundle, _, _ = load_reference(reference, spec["head"], spec["sha256"])
        del bundle
        version = reference.db.execute("PRAGMA data_version").fetchone()[0]
        readiness = require_final_ready(reference)
        yield reference
        check_hashes(spec["head"], spec["sha256"])
        if file_sha256(reference.directory / "identity.json") != spec["feature_identity_sha256"]:
            raise ValueError("Reference feature identity changed during input preparation")
        if (require_final_ready(reference) != readiness or
                reference.db.execute("PRAGMA data_version").fetchone()[0] != version):
            raise ValueError("Reference final preparations or membership changed during comparison preparation")
    finally:
        reference.close()


def jsonl_rows(path):
    with Path(path).open("rb") as handle:
        while raw := handle.readline(MAX_ROW_BYTES + 1):
            if len(raw) > MAX_ROW_BYTES:
                raise MemoryError("Prepared input exceeds the 16 MiB per-record bound")
            yield raw, json.loads(raw)


def require_semantics(reference, row, remaining):
    if row.get("role") != "test" or row.get("id") not in remaining:
        raise ValueError("Missing, extra, duplicate, or non-test comparison record")
    actual = reference.db.execute("SELECT metadata FROM records WHERE id=?", (row["id"],)).fetchone()
    expected = json.loads(actual[0])
    if dataset_scope(reference) is not None:
        validate_row(row)
    if any(row.get(key) != expected.get(key) for key in semantic_fields(expected)):
        raise ValueError("Comparison input changes canonical ID/group/candidate/target semantics")
    remaining.remove(row["id"])


def prepare_inputs(spec, destination):
    """Copy exact prepared test lines; never sample or render/tokenize inputs."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("Refuse to overwrite existing comparison inputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-inputs-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with frozen_reference(spec) as reference:
            readiness = require_final_ready(reference)
            membership = canonical_test_membership(reference)
            remaining = {row["id"] for row in test_rows(reference)}
            selected_digest, count, size = hashlib.sha256(), 0, 0
            with (staging / INPUTS).open("xb") as output:
                for prepared in readiness["preparations"]:
                    digest, examined = hashlib.sha256(), 0
                    for raw, row in jsonl_rows(prepared["path"]):
                        digest.update(raw)
                        examined += 1
                        if row.get("role") != "test":
                            continue
                        require_semantics(reference, row, remaining)
                        output.write(raw)
                        selected_digest.update(raw)
                        count += 1
                        size += len(raw)
                    if digest.hexdigest() != prepared["sha256"] or examined != prepared["records"]:
                        raise ValueError("Prepared root changed while copying comparison inputs")
            if remaining or count != membership["records"]:
                raise ValueError("Prepared inputs omit canonical test records")
            receipt = {
                "schema_version": 1, "artifact_kind": "complete-canonical-test-inputs",
                "roles": ["test"],
                "reference": {
                    "feature_identity_sha256": spec["feature_identity_sha256"],
                    "preparation_sha256": sha256(readiness),
                    "test_membership": membership, "head_sha256": spec["sha256"],
                },
                "inputs": {"file": INPUTS, "sha256": selected_digest.hexdigest(),
                           "records": count, "bytes": size,
                           "order": "original preparation order; unchanged raw JSONL bytes"},
                "producer": {"module": "fuzev_research.comparison_inputs",
                             "source_sha256": file_sha256(Path(__file__))},
                "scope": "inputs only; no feature extraction, predictions, or quality result",
            }
            write_json(staging / MANIFEST, receipt)
            check_hashes(spec["head"], spec["sha256"])
            if require_final_ready(reference) != readiness:
                raise ValueError("Reference final preparations changed while copying comparison inputs")
        # The freeze's post-checks also finish before exposing a complete set.
        if destination.exists():
            raise FileExistsError("Comparison destination appeared during preparation")
        staging.rename(destination)
    return receipt


def iter_inputs(directory, manifest_sha256, spec):
    """Verify the sealed inputs against a frozen reference before model use."""
    directory = Path(directory)
    path = directory / MANIFEST
    if file_sha256(path) != manifest_sha256:
        raise ValueError("Comparison input manifest checksum mismatch")
    manifest = read_json(path)
    if (manifest.get("schema_version") != 1
            or manifest.get("artifact_kind") != "complete-canonical-test-inputs"
            or manifest.get("roles") != ["test"]
            or manifest.get("inputs", {}).get("file") != INPUTS):
        raise ValueError("Invalid explicit comparison input manifest")
    inputs = directory / INPUTS
    before = (inputs.stat().st_size, inputs.stat().st_mtime_ns, inputs.stat().st_ctime_ns)
    if before[0] != manifest["inputs"]["bytes"] or file_sha256(inputs) != manifest["inputs"]["sha256"]:
        raise ValueError("Comparison input file checksum/size mismatch")
    with frozen_reference(spec) as reference:
        canonical = canonical_test_membership(reference)
        expected = {"feature_identity_sha256": spec["feature_identity_sha256"],
                    "preparation_sha256": sha256(require_final_ready(reference)),
                    "test_membership": canonical, "head_sha256": spec["sha256"]}
        if manifest["reference"] != expected or manifest["inputs"]["records"] != canonical["records"]:
            raise ValueError("Comparison inputs belong to another frozen reference or incomplete test set")
        remaining = {row["id"] for row in test_rows(reference)}
        digest = hashlib.sha256()
        for raw, row in jsonl_rows(inputs):
            digest.update(raw)
            require_semantics(reference, row, remaining)
            yield row
        if remaining or digest.hexdigest() != manifest["inputs"]["sha256"]:
            raise ValueError("Comparison inputs changed or are incomplete")
        after = (inputs.stat().st_size, inputs.stat().st_mtime_ns, inputs.stat().st_ctime_ns)
        if before != after or file_sha256(path) != manifest_sha256:
            raise ValueError("Comparison inputs changed during extraction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Previously frozen evaluation plan")
    parser.add_argument("--output", type=Path, required=True, help="New local input directory")
    args = parser.parse_args()
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Run input preparation without GPU access")
    torch.set_num_threads(1)
    plan = read_json(args.plan)
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported frozen evaluation plan")
    receipt = prepare_inputs(plan["reference"], args.output)
    print(json.dumps({"output": str(args.output.resolve()), "receipt": receipt,
                      "manifest_sha256": file_sha256(args.output / MANIFEST)}, allow_nan=False))


if __name__ == "__main__":
    main()
