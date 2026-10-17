"""Bounded, resumable frozen-feature shards using safetensors and row JSONL."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import warnings

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fuzev.contracts import Primitive
from fuzev.encoding import (LABEL_CAPACITY, LIST_AWARE_VERSION, VERSION,
                            query_suffix, list_aware_query_suffix, candidate_suffix)


GEOMETRY = ("hidden_size", "projection_size", "query_adapter_size", "key_adapter_size")
ROLES = ("fit", "development", "calibration", "test")


def json_rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def indexed_rows(path):
    with Path(path).open("rb") as handle:
        while True:
            position, line = handle.tell(), handle.readline()
            if not line:
                return
            if line.strip():
                yield position, json.loads(line)


def contract(config):
    return {key: config[key] for key in ("hidden_size", "encoding_version", "features", "baseline")}


def validate_row(row, config):
    for key in ("id", "group", "source"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Missing nonempty {key}")
    if row.get("role") not in ROLES or row.get("primitive") not in tuple(Primitive):
        raise ValueError("Unknown role or primitive")
    request, rendering = row.get("request"), row.get("rendering")
    if not isinstance(request, dict) or not isinstance(rendering, dict):
        raise ValueError("Rows require original request and rendering fields for safe resume")
    if (not isinstance(request.get("messages"), list) or not request["messages"]
            or any(not isinstance(message, dict) or not isinstance(message.get("content"), str)
                   or message.get("role") not in ("system", "user", "assistant")
                   for message in request["messages"])
            or request.get("primitive") != row["primitive"]
            or not all(isinstance(request.get(key), str) for key in ("question", "additional_context"))):
        raise ValueError("Store resolved prefix messages and exact decision request fields")
    candidates = request.get("candidates")
    if (not isinstance(candidates, list) or len(candidates) < 2
            or any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"]
                   or not isinstance(c.get("text"), str) for c in candidates)):
        raise ValueError("Expected ordered candidate id/text objects")
    ids, count = [c["id"] for c in candidates], len(candidates)
    if len(set(ids)) != count:
        raise ValueError("Duplicate candidate IDs")
    if "candidates" in row and [{"id": c["id"], "text": c["text"]} for c in row["candidates"]] != candidates:
        raise ValueError("Prepared candidates differ from the model-visible request")
    if (not isinstance(rendering.get("prefix"), str)
            or not isinstance(rendering.get("query_suffix"), str)
            or not isinstance(rendering.get("candidate_suffixes"), list)
            or len(rendering["candidate_suffixes"]) != count
            or not all(isinstance(tail, str) for tail in rendering["candidate_suffixes"])
            or rendering.get("encoding_version") not in (VERSION, LIST_AWARE_VERSION)
            or count > LABEL_CAPACITY and rendering["encoding_version"] != LIST_AWARE_VERSION):
        raise ValueError("Store exact rendered prefix, query suffix and ordered candidate suffixes")
    suffix = query_suffix if count <= LABEL_CAPACITY else list_aware_query_suffix
    if (rendering["query_suffix"] != suffix(request["question"], candidates, request["additional_context"])
            or rendering["candidate_suffixes"] != [candidate_suffix(c) for c in candidates]):
        raise ValueError("Rendered decision suffixes differ from the frozen product encoding")
    capacity = config["capacities"].get(row["primitive"], {}).get(row.get("modality"))
    if capacity is None or count > capacity["implemented_max"]:
        raise ValueError("Feature row exceeds declared primitive/modality capacity")
    if row["primitive"] == "noul" and ids != ["yes", "no"]:
        raise ValueError("Noul requires ordered yes/no candidates")
    target = torch.as_tensor(row.get("target"), dtype=torch.float32)
    if (target.shape != (count,) or not target.isfinite().all() or (target < 0).any()
            or not torch.isclose(target.sum(), target.new_tensor(1.), atol=1e-6, rtol=1e-6)):
        raise ValueError("Expected normalized finite target in candidate order")
    for name in ("grades", "ordinal_values"):
        if name in row:
            value = torch.as_tensor(row[name], dtype=torch.float32)
            if value.shape != (count,) or not value.isfinite().all():
                raise ValueError(f"Invalid aligned {name}")
            if name == "grades" and (value < 0).any():
                raise ValueError("Ranking grades must be nonnegative")
    if row["primitive"] == "ordinal" and (
            "ordinal_values" not in row or not (torch.tensor(row["ordinal_values"]).diff() > 0).all()):
        raise ValueError("Ordinal levels must be strictly increasing")
    if "acceptable_candidate_ids" in row:
        acceptable = row["acceptable_candidate_ids"]
        if (not isinstance(acceptable, list) or not acceptable or len(set(acceptable)) != len(acceptable)
                or not set(acceptable) <= set(ids)):
            raise ValueError("Invalid acceptable candidate set")
    if row["role"] != "test" and row.get("evaluation_scope") in ("external", "heldout", "evaluation"):
        raise ValueError("External evaluation records cannot become FIT, DEV or CAL")
    json.dumps(row, allow_nan=False)
    return count


def validate_values(row, values, config):
    count, width = len(row["request"]["candidates"]), config["hidden_size"]
    for name, shape in (("query", (width,)), ("candidates", (count, width)),
                        ("baseline", (count,)), ("targets", (count,))):
        value = values[name]
        if value.shape != shape or value.dtype != torch.float32 or not value.isfinite().all():
            raise ValueError(f"Invalid finite FP32 {name} geometry for {row['id']}")
    if not torch.equal(values["targets"], torch.tensor(row["target"], dtype=torch.float32)):
        raise ValueError("Stored target differs from row metadata")
    if count > LABEL_CAPACITY and values["baseline"].count_nonzero():
        raise ValueError("The label-free route requires a zero baseline")


class RaggedFeatures:
    """Read only row locators at startup; materialize one selected row at a time."""

    def __init__(self, directory, config):
        self.handles = ExitStack()
        self.metadata_handles, self.tensor_handles = {}, {}
        self.directory, self.config = Path(directory), config
        self.rows, self.by_id, self.groups = [], {}, {}
        files = sorted(set(self.directory.glob("shard-*.safetensors")) |
                       {p.with_suffix(".safetensors") for p in self.directory.glob("shard-*.jsonl")})
        self.next_shard = 0
        for path in files:
            number = int(path.stem.removeprefix("shard-"))
            if not path.exists() or not path.with_suffix(".jsonl").exists():
                if path != files[-1]:
                    raise ValueError(f"Incomplete nonfinal shard: {path}")
                warnings.warn(f"Incomplete tail shard will be retried: {path}", stacklevel=2)
                self.next_shard = number
                break
            with safe_open(path, framework="pt", device="cpu") as handle:
                if json.loads((handle.metadata() or {}).get("features", "null")) != contract(config):
                    raise ValueError(f"Feature contract differs: {path}")
                offset_tensor = handle.get_tensor("offsets")
                offsets = offset_tensor.tolist()
                if (offset_tensor.dtype != torch.int64 or offset_tensor.ndim != 1
                        or len(offsets) < 2 or offsets[0] != 0
                        or any(b - a < 2 for a, b in zip(offsets, offsets[1:]))):
                    raise ValueError("Invalid ragged offsets")
                if (handle.get_slice("query").get_shape() != [len(offsets) - 1, config["hidden_size"]]
                        or handle.get_slice("candidates").get_shape() != [offsets[-1], config["hidden_size"]]
                        or any(handle.get_slice(key).get_shape() != [offsets[-1]]
                               for key in ("baseline", "targets"))):
                    raise ValueError("Invalid ragged shard geometry")
            count = 0
            for index, (position, row) in enumerate(indexed_rows(path.with_suffix(".jsonl"))):
                k = validate_row(row, config)
                if index + 1 >= len(offsets) or offsets[index + 1] - offsets[index] != k:
                    raise ValueError("Shard row metadata and offsets differ")
                self.remember(row, number, index, offsets[index], offsets[index + 1], position)
                count += 1
            if count != len(offsets) - 1:
                raise ValueError("Missing shard row metadata")
            self.next_shard = number + 1

    def remember(self, row, number, index, start, stop, metadata_offset):
        if row["id"] in self.by_id or self.groups.get(row["group"], row["role"]) != row["role"]:
            raise ValueError("Duplicate record or group role leakage")
        ref = {key: row[key] for key in ("id", "group", "role", "primitive", "modality", "source")}
        ref.update(shard=number, row=index, start=start, stop=stop, k=stop - start, metadata_offset=metadata_offset)
        self.rows.append(ref)
        self.by_id[row["id"]] = ref
        self.groups[row["group"]] = row["role"]

    def metadata(self, ref):
        number = ref["shard"]
        if number not in self.metadata_handles:
            path = self.directory / f"shard-{number:06d}.jsonl"
            self.metadata_handles[number] = self.handles.enter_context(path.open("rb"))
        handle = self.metadata_handles[number]
        handle.seek(ref["metadata_offset"])
        return json.loads(handle.readline())

    def read(self, ref):
        row = self.metadata(ref)
        number = ref["shard"]
        if number not in self.tensor_handles:
            path = self.directory / f"shard-{number:06d}.safetensors"
            self.tensor_handles[number] = self.handles.enter_context(safe_open(path, framework="pt", device="cpu"))
        handle = self.tensor_handles[number]
        values = {"query": handle.get_slice("query")[ref["row"]:ref["row"] + 1][0]}
        for key in ("candidates", "baseline", "targets", "grades", "ordinal_values"):
            if key in ("candidates", "baseline", "targets") or key in row:
                values[key] = handle.get_slice(key)[ref["start"]:ref["stop"]]
        return row, values

    def close(self):
        self.handles.close()
        self.metadata_handles.clear()
        self.tensor_handles.clear()

    def __del__(self):
        self.close()


class RaggedWriter:
    def __init__(self, directory, config, *, shard_bytes):
        if shard_bytes < 1:
            raise ValueError("shard_bytes must be positive")
        self.directory, self.config = Path(directory), config
        self.directory.mkdir(parents=True, exist_ok=True)
        self.completed = RaggedFeatures(directory, config)
        self.pending, self.bytes, self.shard_bytes = {}, 0, shard_bytes

    def contains(self, row):
        validate_row(row, self.config)
        if row["id"] in self.pending:
            previous = self.pending[row["id"]][0]
        elif row["id"] in self.completed.by_id:
            previous = self.completed.metadata(self.completed.by_id[row["id"]])
        else:
            return False
        # Execution accounting is observed after extraction, not model input.
        if {k: v for k, v in previous.items() if k != "execution"} != {
                k: v for k, v in row.items() if k != "execution"}:
            raise ValueError(f"Existing row ID has changed metadata/request/rendering: {row['id']}")
        return True

    def add(self, row, query, candidates, baseline):
        if self.contains(row):
            return False
        row = json.loads(json.dumps(row, allow_nan=False))
        values = {key: torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()
                  for key, value in (("query", query), ("candidates", candidates), ("baseline", baseline),
                                     ("targets", row["target"]))}
        validate_values(row, values, self.config)
        for key in ("grades", "ordinal_values"):
            if key in row:
                values[key] = torch.tensor(row[key], dtype=torch.float32)
        # Reserve both optional aligned fields even when this particular row omits them.
        size = sum(t.numel() * t.element_size() for t in values.values())
        size += 8 * len(row["target"]) + 16 + len(json.dumps(row, allow_nan=False).encode())
        if size > self.shard_bytes:
            raise ValueError(f"Row {row['id']} needs {size} bytes; shard budget is {self.shard_bytes}")
        if self.bytes + size > self.shard_bytes:
            self.flush()
        roles = {r[0]["role"] for r in self.pending.values() if r[0]["group"] == row["group"]}
        previous = self.completed.groups.get(row["group"], row["role"])
        if roles - {row["role"]} or previous != row["role"]:
            raise ValueError("Group role leakage")
        self.pending[row["id"]] = (row, values)
        self.bytes += size
        return True

    def flush(self):
        if not self.pending:
            return
        entries = list(self.pending.values())
        offsets = [0]
        for row, _ in entries:
            offsets.append(offsets[-1] + len(row["target"]))
        tensors = {"offsets": torch.tensor(offsets, dtype=torch.int64),
                   "query": torch.stack([v["query"] for _, v in entries])}
        for key in ("candidates", "baseline", "targets", "grades", "ordinal_values"):
            if any(key in v for _, v in entries):
                tensors[key] = torch.cat([v.get(key, torch.zeros(len(row["target"]))) for row, v in entries])
        path = self.directory / f"shard-{self.completed.next_shard:06d}.safetensors"
        temporary, metadata = path.with_suffix(".safetensors.tmp"), path.with_suffix(".jsonl.tmp")
        save_file(tensors, temporary, metadata={"features": json.dumps(contract(self.config), allow_nan=False)})
        positions = []
        with metadata.open("wb") as handle:
            for row, _ in entries:
                positions.append(handle.tell())
                handle.write((json.dumps(row, allow_nan=False) + "\n").encode())
        temporary.replace(path)
        metadata.replace(path.with_suffix(".jsonl"))
        for index, (row, _) in enumerate(entries):
            self.completed.remember(row, self.completed.next_shard, index, offsets[index], offsets[index + 1], positions[index])
        self.completed.next_shard += 1
        self.pending.clear()
        self.bytes = 0


def batches(rows, *, max_elements, max_rows, hidden_size):
    if min(max_elements, max_rows, hidden_size) < 1:
        raise ValueError("Batch dimensions and budgets must be positive")
    pending, width = [], 0
    for ref in rows:
        k = ref["k"]
        if k * hidden_size > max_elements:
            raise ValueError(f"Row {ref['id']} exceeds padded-element budget")
        if pending and (len(pending) == max_rows or (len(pending) + 1) * max(width, k) * hidden_size > max_elements):
            yield pending
            pending, width = [], 0
        pending.append(ref)
        width = max(width, k)
    if pending:
        yield pending


def collate(rows, device="cpu"):
    count = max(ref["k"] for ref in rows)
    metadata, tensors = [], {}
    for index, ref in enumerate(rows):
        row, values = ref["store"].read(ref)
        metadata.append(row)
        if not tensors:
            width = values["query"].numel()
            tensors = {"query": torch.zeros(len(rows), width), "candidates": torch.zeros(len(rows), count, width),
                       "baseline": torch.zeros(len(rows), count), "targets": torch.zeros(len(rows), count),
                       "mask": torch.zeros(len(rows), count, dtype=torch.bool)}
        tensors["query"][index] = values["query"]
        for name in ("candidates", "baseline", "targets"):
            tensors[name][index, :ref["k"]] = values[name]
        tensors["mask"][index, :ref["k"]] = True
    return metadata, {key: value.to(device) for key, value in tensors.items()}


def legacy_index(directory):
    path = Path(directory) / "index.sqlite3"
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Replay import needs a completed, checkpointed original cache")
    # A plain mode=ro connection may still create SQLite WAL/SHM sidecars.
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def import_legacy(directory, rows, writer, *, max_new_rows=None, lineage=None):
    """Import joined original request metadata after checking the legacy feature semantics."""
    from fuzev_research import encoding as old_encoding
    from fuzev_research.anchor_head import TAP
    from fuzev.encoding import TEMPLATE

    directory = Path(directory)
    identity = json.loads((directory / "identity.json").read_text())
    old_identity = old_encoding.identity()
    if (identity.get("hidden_size") != writer.config["hidden_size"]
            or any(identity.get("encoding", {}).get(key) != old_identity[key]
                   for key in ("version", "thinking", "readout"))
            or identity.get("feature_tap") != TAP
            or old_encoding.TEMPLATE.read_text() != TEMPLATE.read_text()):
        raise ValueError("Legacy encoding/tap semantics are not established compatible")
    # Use the existing SQLite/shard layout without importing acquisition/root globals.
    source = legacy_index(directory)
    source.row_factory = sqlite3.Row
    added, skipped, limited = 0, 0, False
    try:
        for row in rows:
            if row["role"] not in ROLES[:3] or row["modality"] != "text":
                raise ValueError("Legacy replay import accepts original text FIT/DEV/CAL only")
            stored = source.execute("SELECT * FROM records WHERE id=?", (row["id"],)).fetchone()
            if stored is None:
                raise ValueError(f"Missing legacy row: {row['id']}")
            original, request = json.loads(stored["metadata"]), row["request"]
            if (any(original.get(key) != row.get(key) for key in
                    ("id", "group", "role", "primitive", "modality", "source", "target"))
                    or any(key in row and row[key] != value for key, value in original.items() if key != "execution")
                    or original.get("has_baseline") is not True
                    or [{"id": c["id"], "text": c["text"]} for c in original["candidates"]] != request["candidates"]):
                raise ValueError("Legacy ownership, candidates, target or baseline differs")
            row = dict(original, **row)
            row["producer"] = {key: identity[key] for key in
                               ("engine", "model", "revision", "cache_precision", "feature_tap") if key in identity}
            if lineage is not None:
                row["producer"]["published_cache_lineage"] = lineage
            if row["primitive"] == "ordinal":
                row["ordinal_values"] = [candidate["ordinal_value"] for candidate in original["candidates"]]
            args = (request["question"], request["candidates"], request["additional_context"])
            if (old_encoding.query_suffix(*args) != query_suffix(*args)
                    or row["rendering"]["query_suffix"] != query_suffix(*args)
                    or any(old_encoding.candidate_suffix(c) != candidate_suffix(c) for c in request["candidates"])):
                raise ValueError("Legacy and current model-visible encoding differ")
            if writer.contains(row):
                skipped += 1
                continue
            if max_new_rows is not None and added >= max_new_rows:
                limited = True
                break
            with safe_open(directory / f"shard-{stored['shard']:06d}.safetensors", framework="pt", device="cpu") as handle:
                values = {key: handle.get_slice(key)[stored["row_"]:stored["row_"] + 1][0]
                          for key in ("query", "candidates", "baseline", "targets", "mask")}
            k = len(row["target"])
            if (values["mask"].dtype != torch.bool or not values["mask"][:k].all() or values["mask"][k:].any()
                    or not torch.equal(values["targets"][:k], torch.tensor(row["target"], dtype=torch.float32))):
                raise ValueError("Legacy feature mask or target differs")
            validate_values(row, {key: value if key == "query" else value[:k]
                                  for key, value in values.items() if key != "mask"}, writer.config)
            writer.add(row, values["query"], values["candidates"][:k], values["baseline"][:k])
            added += 1
        writer.flush()
    finally:
        source.close()
    return {"added": added, "resumed": skipped, "limited": limited}


def prepared_legacy_rows(paths, role):
    """Join model-visible inputs from original preparations, never from feature guesses."""
    from transformers.utils.chat_template_utils import render_jinja_template
    from fuzev_research.encoding import TEMPLATE

    template, seen = TEMPLATE.read_text(), set()
    for path in paths:
        for row in json_rows(path):
            if row["role"] != role or row["modality"] != "text":
                continue
            if row["id"] in seen:
                raise ValueError(f"Duplicate prepared replay ID: {row['id']}")
            seen.add(row["id"])
            if row.get("image_path"):
                raise ValueError("Text replay cannot contain an image input")
            messages = row["context"] or [{"role": "user", "content": "Context for the decision."}]
            candidates = [{"id": c["id"], "text": c["text"]} for c in row["candidates"]]
            rendered, _ = render_jinja_template(
                [messages], chat_template=template, add_generation_prompt=False,
                enable_thinking=False, reasoning_effort="none", preserve_thinking=True)
            # The original producer passed question/candidates only, not extra evidence.
            request = {"primitive": row["primitive"], "messages": messages, "question": row["question"],
                       "candidates": candidates, "additional_context": ""}
            yield dict(row, request=request, rendering={
                "prefix": rendered[0], "query_suffix": query_suffix(row["question"], candidates),
                "candidate_suffixes": [candidate_suffix(c) for c in candidates], "encoding_version": VERSION})


def replay_config(bundle_config, directory):
    """Storage contract plus observed old K support, not a serving or quality declaration."""
    geometry = json.loads(Path(bundle_config).read_text())
    source = legacy_index(directory)
    try:
        capacities = {primitive: {"text": {"implemented_max": k, "validated_max": None}}
                      for primitive, k in source.execute(
                          "SELECT primitive,MAX(json_array_length(metadata,'$.target')) FROM records "
                          "WHERE modality='text' AND role IN ('fit','development','calibration') GROUP BY primitive")}
        counts = dict(source.execute("SELECT role,COUNT(*) FROM records WHERE modality='text' "
                                     "AND role IN ('fit','development','calibration') GROUP BY role"))
    finally:
        source.close()
    feature = {"tap": "final-pre-lm", "pooling": "last-token"}
    return dict({key: geometry[key] for key in GEOMETRY}, encoding_version=LIST_AWARE_VERSION,
                features={"query": feature, "candidate": feature},
                baseline={"labelled": "query-label-logits", "label_free": "zero"}, capacities=capacities), counts


def main(argv=None):
    from train_shared_head import positive_int

    parser = argparse.ArgumentParser(description="Import original text FIT/DEV/CAL features; no extraction or fitting.")
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, nargs="+", required=True)
    parser.add_argument("--bundle-config", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-bytes", type=positive_int, required=True)
    parser.add_argument("--max-new-rows", type=positive_int, help="New rows per role this invocation; omit to finish")
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.legacy.resolve()):
        raise ValueError("Replay output must be separate from original features")
    config, expected = replay_config(args.bundle_config, args.legacy)
    origins = json.loads(args.provenance.read_text())["feature_origins"]
    lineage = {"scope": "published cache lineage, not per-row producer attribution",
               "current_runtime_sources": origins["current_runtime_sources"],
               "reused": {key: origins["reused"][key] for key in ("runtime_sources", "records")}}
    for role in ROLES[:3]:
        writer = RaggedWriter(args.output / role, config, shard_bytes=args.shard_bytes)
        if any(ref["role"] != role or ref["modality"] != "text" for ref in writer.completed.rows):
            raise ValueError("Replay output must contain original text rows in role-only directories")
        rows = prepared_legacy_rows(args.prepared, role)
        try:
            result = import_legacy(args.legacy, rows, writer, max_new_rows=args.max_new_rows, lineage=lineage)
        finally:
            rows.close()
        completed = len(writer.completed.rows)
        if not result["limited"] and result["added"] + result["resumed"] != expected.get(role, 0):
            raise ValueError(f"Original {role} features are not fully joined by the supplied preparations")
        if completed != result["added"] + result["resumed"]:
            raise ValueError("Replay output contains rows outside the supplied original preparations")
        print(json.dumps(dict(result, role=role, records=completed, expected=expected.get(role, 0),
                              complete=not result["limited"], output=str(writer.directory)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
