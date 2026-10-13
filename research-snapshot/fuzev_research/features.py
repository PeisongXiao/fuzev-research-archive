"""Resumable, bounded frozen-feature shards with durable split ownership."""

import hashlib
import json
import sqlite3
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from .acquire import write_json
from .paths import project_path
from .scope import (ALL_PREPARATIONS as PREPARATIONS, DIMENSION_METADATA,
                    EXTERNAL_SOURCES, TRAINING_SOURCES, required_preparations,
                    validate_contract, validate_manifest, validate_row)


SCOPE_METADATA = ("original_split", "evaluation_scope", "source_test_example_id", "source_question_id",
                  "evidence_id", "sampling_weight", "inclusion_probability", "sampling_stratum",
                  "sampling_cluster", "sampling_design", "population_size", "sample_size",
                  "sampling_label", "inclusion_probability_numerator", "inclusion_probability_denominator",
                  "sampling_seed", "dataset_scope_revision", "dataset_purpose", "source_example_id", "original_answer")
SCOPE_METADATA += ("source_original_split", "source_response_pair_sha256", "label_semantics",
                   "image_sha256", "source_image_id", "query_source", "annotation_id",
                   "source_question_ids", "source_paper_id", "image_path", "official_holdout_overlap", "question")
SCOPE_METADATA += ("adapter_version", "parent_id", "parent_role", "parent_source", "parent_primitive",
                   "parent_candidate_ids", "parent_target", "candidate_permutation", "orientation_seed",
                   "parent_record_sha256", "parent_input_sha256", "parent_adaptation")


_MASK_CACHE_SHARDS = 8192


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_preparation(db, path, identity, modality, datasets, scoped=False):
    digest, count, pending = hashlib.sha256(), 0, []
    fields = ("id", "group", "role", "primitive", "modality", "source", "revision",
              "candidates", "target", "adaptation", "attribute")

    def check():
        placeholders = ",".join("?" for _ in pending)
        committed = {row["id"]: row for row in db.execute(
            "SELECT * FROM records WHERE id IN (" + placeholders + ")",
            [row["id"] for row in pending])}
        for expected in pending:
            actual = committed.get(expected["id"])
            metadata = json.loads(actual["metadata"]) if actual else {}
            extra = SCOPE_METADATA if scoped and (expected["source"] in EXTERNAL_SOURCES + ("boolq", "mm-rlhf")
                                                  or expected.get("adaptation") == "pairwise_preference_yes_no") else ()
            if scoped and expected.get("adapter_version") == "human-dimension-choice-v1":
                extra = SCOPE_METADATA + DIMENSION_METADATA
            if (not actual or any(metadata.get(key) != expected[key] for key in fields + extra)
                    or any(actual[key] != expected[key] for key in ("id", "role", "primitive", "modality", "source"))
                    or actual["group_"] != expected["group"]
                    or expected["modality"] != modality
                    or datasets.get(expected["source"]) != expected["revision"]):
                raise ValueError(f"Prepared input is not accounted for by matching features: {expected['id']}")
        pending.clear()

    # Parse one root record at a time, retaining only bounded feature metadata.
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            if scoped:
                validate_row(row, path.name)
            pending.append({key: row.get(key) for key in fields + SCOPE_METADATA + DIMENSION_METADATA})
            count += 1
            if len(pending) == 128:
                check()
    if pending:
        check()
    if digest.hexdigest() != identity["sha256"] or count != identity["records"]:
        raise ValueError(f"Prepared root file hash/count mismatch: {path.name}")


def _shard_stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class FeatureStore:
    def __init__(self, directory, identity, shard_size=128, *, read_only=False,
                 cache_bytes=64 << 20, mask_cache_bytes=1 << 20):
        self.directory = Path(directory)
        self.read_only = read_only
        if not read_only:
            self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "identity.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Feature provenance changed; use a separate feature directory")
        if not manifest.exists():
            if read_only:
                raise ValueError("Missing feature identity for read-only corpus")
            write_json(manifest, identity)
        self.identity, self.shard_size = identity, shard_size
        self.scoped = "dataset_scope" in identity
        if self.scoped:
            validate_contract(identity["dataset_scope"])
        database = self.directory / "index.sqlite3"
        self.db = (sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
                   if read_only else sqlite3.connect(database))
        self.db.row_factory = sqlite3.Row
        self.buffer, self.pending = [], set()
        self.loaded = OrderedDict()
        self.cache_bytes, self.loaded_bytes = max(0, cache_bytes), 0
        self._mask_descriptors = OrderedDict()
        self.mask_cache_bytes, self._mask_descriptor_bytes = max(0, mask_cache_bytes), 0
        self._readiness = {}
        self._record_versions = {}
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
            return
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS records (
              id TEXT PRIMARY KEY, shard INTEGER, row_ INTEGER, group_ TEXT,
              role TEXT, primitive TEXT, modality TEXT, source TEXT, metadata TEXT);
            CREATE TABLE IF NOT EXISTS groups_ (id TEXT PRIMARY KEY, role TEXT);
            CREATE INDEX IF NOT EXISTS role_index ON records(role,modality,primitive);
            CREATE TABLE IF NOT EXISTS failures (id TEXT PRIMARY KEY, reason TEXT);
            CREATE TABLE IF NOT EXISTS preparations (path TEXT PRIMARY KEY, sha256 TEXT);
            CREATE TABLE IF NOT EXISTS record_versions (modality TEXT PRIMARY KEY, version INTEGER NOT NULL);
            CREATE TRIGGER IF NOT EXISTS records_version_insert BEFORE INSERT ON records BEGIN
              INSERT INTO record_versions SELECT modality,1 FROM records WHERE id=NEW.id
              ON CONFLICT(modality) DO UPDATE SET version=version+1;
              INSERT INTO record_versions VALUES(NEW.modality,1)
              ON CONFLICT(modality) DO UPDATE SET version=version+1;
            END;
            CREATE TRIGGER IF NOT EXISTS records_version_delete AFTER DELETE ON records BEGIN
              INSERT INTO record_versions VALUES(OLD.modality,1)
              ON CONFLICT(modality) DO UPDATE SET version=version+1;
            END;
            CREATE TRIGGER IF NOT EXISTS records_version_update AFTER UPDATE ON records BEGIN
              INSERT INTO record_versions VALUES(OLD.modality,1)
              ON CONFLICT(modality) DO UPDATE SET version=version+1;
              INSERT INTO record_versions VALUES(NEW.modality,1)
              ON CONFLICT(modality) DO UPDATE SET version=version+1;
            END;
        """)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(preparations)")}
        with self.db:
            if "completed" not in columns:
                self.db.execute("ALTER TABLE preparations ADD COLUMN completed INTEGER NOT NULL DEFAULT 0")
                self.db.execute("ALTER TABLE preparations ADD COLUMN examined INTEGER")

    def _record_version(self, stage):
        schema = self.db.execute("PRAGMA schema_version").fetchone()[0]
        triggers = {row[0] for row in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'records_version_%'")}
        where = " WHERE modality='text'" if stage == "text" else ""
        if triggers == {"records_version_insert", "records_version_delete", "records_version_update"}:
            return (schema, tuple(tuple(row) for row in self.db.execute(
                "SELECT modality,version FROM record_versions" + where + " ORDER BY modality")))
        # Legacy writers need no live migration: hash only SQLite feature metadata
        # after database changes, never reparse unchanged multi-GB prepared roots.
        version = (schema, self.db.execute("PRAGMA data_version").fetchone()[0], self.db.total_changes)
        cached = self._record_versions.get(stage)
        if cached and cached[0] == version:
            return cached[1]
        digest = hashlib.sha256()
        for row in self.db.execute("SELECT * FROM records" + where + " ORDER BY id"):
            digest.update(json.dumps(tuple(row), separators=(",", ":")).encode() + b"\n")
        result = (schema, digest.hexdigest())
        self._record_versions[stage] = (version, result)
        return result

    def require_ready(self, stage):
        """Gate use of committed features against the full prepared corpus."""
        if stage not in {"text", "mixed", "calibrate"}:
            raise ValueError("Unknown training stage")
        return self._require_preparations(stage)

    def require_external_ready(self):
        if not self.scoped:
            raise ValueError("External evaluation requires an explicit dataset-scope contract")
        result = self._require_preparations("external")
        subtypes = dict(self.db.execute("SELECT json_extract(metadata,'$.adaptation'),count(*) FROM records "
                                       "WHERE source='qasper' GROUP BY json_extract(metadata,'$.adaptation')"))
        if subtypes != {"yes_no": 239, "answerability": 1450, "evidence": 2000}:
            raise ValueError("External QASPER requires exactly the declared official-test census and evidence sample")
        examples = {}
        for primitive in ("choice", "noul"):
            examples[primitive] = {row[0] for row in self.db.execute(
                "SELECT json_extract(metadata,'$.source_test_example_id') FROM records "
                "WHERE source='vl-rewardbench' AND primitive=?", (primitive,))}
            if None in examples[primitive] or not examples[primitive]:
                raise ValueError("Missing VL-RewardBench original held-out example IDs")
            count = self.db.execute("SELECT count(*) FROM records WHERE source='vl-rewardbench' AND primitive=?", (primitive,)).fetchone()[0]
            if count != len(examples[primitive]):
                raise ValueError("Duplicate paired VL-RewardBench examples")
        if examples["choice"] != examples["noul"]:
            raise ValueError("Choice and Noul must evaluate the same VL-RewardBench held-out examples")
        return result

    def _require_preparations(self, stage):
        if self.pending or self.buffer:
            raise ValueError("Training requires committed features, not pending records")
        if self.scoped:
            names = required_preparations(stage, self.identity["dataset_scope"])
            files = [(project_path("data/prepared/" + name),
                      project_path("data/manifests/" + PREPARATIONS[name][0]), PREPARATIONS[name][1])
                     for name in names]
        else:
            names = [("text", "text"), ("qasper", "text")]
            if stage != "text":
                names.append(("images", "image"))
            files = [(project_path(f"data/prepared/{name}.jsonl"),
                      project_path(f"data/manifests/{'image' if name == 'images' else name}-splits.json"), modality)
                     for name, modality in names]
        paths = [self.directory / "identity.json", *(path for pair in files for path in pair[:2])]
        try:
            stamps = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns,
                            path.stat().st_ctime_ns) for path in paths)
        except FileNotFoundError as error:
            raise ValueError(f"Missing required preparation identity: {error.filename}") from error
        if self.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]:
            raise ValueError("Extraction failures remain; no silently omitted training inputs are permitted")
        preparations = {row["path"]: row for row in self.preparations()}
        preparation_state = tuple(tuple(sorted(preparations.get(str(path.resolve()), {}).items()))
                                  for path, _, _ in files)
        version = self._record_version(stage)
        key = (stamps, version, preparation_state)
        cached = self._readiness.get(stage)
        if cached and cached[0] == key:
            return cached[1]
        if json.loads(paths[0].read_text()) != self.identity:
            raise ValueError("Feature identity changed on disk")
        expected, identities = Counter(), []
        for path, manifest_path, modality in files:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if self.scoped:
                validate_manifest(path.name, manifest)
            registered = preparations.get(str(path.resolve()))
            if not registered or registered.get("completed") != 1:
                raise ValueError(f"Required preparation is not completed: {path.name}")
            digest = manifest.get("sha256", "")
            if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                    or Path(manifest.get("path", "")).resolve() != path.resolve()
                    or registered["sha256"] != digest):
                raise ValueError(f"Preparation identity/hash mismatch: {path.name}")
            count = manifest.get("records")
            if not isinstance(count, int) or count <= 0 or registered.get("examined") != count:
                raise ValueError(f"Preparation inputs not fully examined: {path.name}")
            tracks = Counter()
            for track, value in manifest["counts"].items():
                if track.startswith("excluded_"):
                    continue
                parts = track.split("/")
                if (len(parts) != 3 or parts[1] not in {"choice", "noul", "ordinal"}
                        or parts[2] not in {"fit", "development", "calibration", "test"}
                        or not isinstance(value, int) or value < 0):
                    raise ValueError(f"Invalid preparation count: {track}")
                tracks[(modality, *parts)] += value
            if sum(tracks.values()) != count:
                raise ValueError(f"Preparation counts do not account for all inputs: {path.name}")
            expected.update(tracks)
            identities.append({"path": str(path.resolve()), "sha256": digest, "records": count,
                               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
        where = " WHERE modality='text'" if stage == "text" else ""
        if self.scoped:
            allowed = EXTERNAL_SOURCES if stage == "external" else TRAINING_SOURCES
            clause = "source IN (" + ",".join("?" for _ in allowed) + ")"
            where = where + " AND " + clause if where else " WHERE " + clause
            invalid = self.db.execute("SELECT source,role FROM records WHERE source IS NULL OR role IS NULL OR source NOT IN (" +
                                      ",".join("?" for _ in TRAINING_SOURCES + EXTERNAL_SOURCES) +
                                      ") OR (source IN (?,?) AND role!='test') LIMIT 1",
                                      (*TRAINING_SOURCES, *EXTERNAL_SOURCES, *EXTERNAL_SOURCES)).fetchone()
            if invalid:
                raise ValueError("Inactive dataset or external-role leakage in the active feature corpus")
        else:
            allowed = ()
        actual = Counter({tuple(row[:4]): row[4] for row in self.db.execute(
            "SELECT modality,source,primitive,role,COUNT(*) FROM records" + where +
            " GROUP BY modality,source,primitive,role", allowed)})
        if actual != expected:
            raise ValueError("Committed feature counts do not account for the full preparation corpus")
        for (path, _, modality), identity in zip(files, identities):
            verify_preparation(self.db, path, identity, modality, self.identity.get("datasets", {}), self.scoped)
        after = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns,
                       path.stat().st_ctime_ns) for path in paths)
        current = {row["path"]: row for row in self.preparations()}
        current_state = tuple(tuple(sorted(current.get(str(path.resolve()), {}).items()))
                              for path, _, _ in files)
        if (stamps != after or version != self._record_version(stage) or preparation_state != current_state
                or self.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]):
            raise ValueError("Preparation corpus changed during readiness verification; retry")
        result = {"preparations": identities, "records": sum(expected.values()),
                  "counts": {"/".join(track): count for track, count in sorted(expected.items())}}
        self._readiness[stage] = (key, result)
        return result

    def register_preparation(self, path, sha256):
        if self.scoped and Path(path).name not in PREPARATIONS:
            raise ValueError("Preparation is outside the revised dataset scope")
        if self.scoped and Path(path).name not in self.identity["dataset_scope"]["preparations"]:
            raise ValueError("Preparation is not enabled in this feature identity")
        if self.scoped:
            name = Path(path).name
            manifest = validate_manifest(name, json.loads(project_path(
                "data/manifests/" + PREPARATIONS[name][0]).read_text()))
            if (Path(path).resolve() != project_path("data/prepared/" + name).resolve()
                    or Path(manifest["path"]).resolve() != Path(path).resolve() or manifest["sha256"] != sha256):
                raise ValueError("Active preparation path/hash does not match its manifest")
        path = str(Path(path).resolve())
        previous = self.db.execute("SELECT sha256 FROM preparations WHERE path=?", (path,)).fetchone()
        if previous and previous[0] != sha256:
            raise ValueError("Prepared data changed; use a separately identified feature corpus")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO preparations(path,sha256) VALUES(?,?)", (path, sha256))

    def finish_preparation(self, path, examined):
        if examined <= 0:
            raise ValueError("An empty preparation cannot establish completed extraction")
        self.flush()
        with self.db:
            result = self.db.execute("UPDATE preparations SET completed=1,examined=? WHERE path=?",
                                     (examined, str(Path(path).resolve())))
            if result.rowcount != 1:
                raise ValueError("Cannot complete an unregistered preparation")

    def preparations(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM preparations ORDER BY path")]

    def producer_origins(self):
        """Bind reused rows to their original producer without relabelling them."""
        result = {"current_runtime_sources": self.identity.get("runtime_sources"), "reused": None}
        exists = self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_origins'").fetchone()
        if not exists:
            return result
        path = self.directory / "scope-transition.json"
        receipt = json.loads(path.read_text())
        count = self.db.execute("SELECT count(*) FROM records r JOIN feature_origins o ON r.shard=o.shard").fetchone()[0]
        if (receipt.get("records_reused") != count or receipt.get("destination_identity_sha256") !=
                file_sha256(self.directory / "identity.json") or not receipt.get("computation_proof")):
            raise ValueError("Reused-feature origin receipt changed or is incomplete")
        result["reused"] = {"receipt_sha256": file_sha256(path), "records": count,
                            "source_identity_sha256": receipt["source_identity_sha256"],
                            "runtime_sources": receipt["source_runtime"], "computation_proof": receipt["computation_proof"]}
        return result

    def contains(self, record_id):
        return record_id in self.pending or self.db.execute(
            "SELECT 1 FROM records WHERE id=?", (record_id,)).fetchone() is not None

    def add(self, row, query, candidates, baseline=None, execution=None):
        if self.read_only:
            raise ValueError("Cannot add to a read-only feature corpus")
        if self.scoped:
            validate_row(row)
            if (row.get("source") in TRAINING_SOURCES and row.get("adaptation") == "pairwise_preference_yes_no"
                    and "noul_preference" not in self.identity["dataset_scope"]):
                raise ValueError("Preference augmentation is not enabled in this feature identity")
            if (row.get("adapter_version") == "human-dimension-choice-v1"
                    and "dimension_choice" not in self.identity["dataset_scope"]):
                raise ValueError("Dimension Choice is not enabled in this feature identity")
        import torch
        if self.contains(row["id"]):
            return
        count = len(row["candidates"])
        if query.shape != (2560,) or candidates.shape != (count, 2560) or not 2 <= count <= 16:
            raise ValueError("Feature shapes do not match the declared candidate-aware tap")
        query, candidates = query.detach().float().cpu().clone(), candidates.detach().float().cpu().clone()
        if not torch.isfinite(query).all() or not torch.isfinite(candidates).all():
            raise ValueError("Nonfinite frozen features")
        if baseline is not None:
            baseline = baseline.detach().float().cpu().clone()
            if baseline.shape != (count,) or not torch.isfinite(baseline).all():
                raise ValueError("Invalid untouched label-token baseline")
        previous = self.db.execute("SELECT role FROM groups_ WHERE id=?", (row["group"],)).fetchone()
        pending_roles = {entry[0]["role"] for entry in self.buffer if entry[0]["group"] == row["group"]}
        if (previous and previous[0] != row["role"]) or pending_roles - {row["role"]}:
            raise ValueError("Training/development/calibration/test group leakage")
        metadata = {key: row[key] for key in ("id", "group", "role", "primitive", "modality", "source", "revision")}
        metadata.update({"candidates": row["candidates"], "target": row["target"],
                         "adaptation": row.get("adaptation"), "attribute": row.get("attribute"),
                         "execution": execution, "has_baseline": baseline is not None})
        if self.scoped:
            metadata.update({key: row[key] for key in SCOPE_METADATA if key in row})
            if row.get("adapter_version") == "human-dimension-choice-v1":
                metadata.update({key: row[key] for key in DIMENSION_METADATA if key in row})
        self.buffer.append((metadata, query, candidates, baseline))
        self.pending.add(row["id"])
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        import torch
        from safetensors.torch import save_file
        shard = self.db.execute("SELECT COALESCE(MAX(shard),-1)+1 FROM records").fetchone()[0]
        rows, width = len(self.buffer), max(len(entry[0]["candidates"]) for entry in self.buffer)
        tensors = {"query": torch.empty(rows, 2560), "candidates": torch.zeros(rows, width, 2560),
                   "mask": torch.zeros(rows, width, dtype=torch.bool), "targets": torch.zeros(rows, width),
                   "ordinal_values": torch.zeros(rows, width), "baseline": torch.zeros(rows, width)}
        metadata = []
        for index, (row, query, candidates, baseline) in enumerate(self.buffer):
            count = len(row["candidates"])
            tensors["query"][index] = query
            tensors["candidates"][index, :count] = candidates
            tensors["mask"][index, :count] = True
            tensors["targets"][index, :count] = torch.tensor(row["target"])
            if row["primitive"] == "ordinal":
                tensors["ordinal_values"][index, :count] = torch.tensor([value["ordinal_value"] for value in row["candidates"]])
            if baseline is not None:
                tensors["baseline"][index, :count] = baseline
            metadata.append(row)
        filename = self.directory / f"shard-{shard:06d}.safetensors"
        temporary = filename.with_suffix(".tmp")
        save_file(tensors, str(temporary))
        temporary.replace(filename)
        write_json(filename.with_suffix(".json"), metadata)
        # Shard files precede the index transaction. A crash before commit leaves
        # an unindexed shard that a resumed writer safely replaces, not partial rows.
        with self.db:
            for index, row in enumerate(metadata):
                self.db.execute("INSERT OR IGNORE INTO groups_ VALUES(?,?)", (row["group"], row["role"]))
                self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)", (
                    row["id"], shard, index, row["group"], row["role"], row["primitive"],
                    row["modality"], row["source"], json.dumps(row, allow_nan=False)))
                self.db.execute("DELETE FROM failures WHERE id=?", (row["id"],))
        self.buffer.clear()
        self.pending.clear()

    def fail(self, record_id, reason):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO failures VALUES(?,?)", (record_id, reason))

    def iter_records(self, role=None, modality=None, primitive=None, include_metadata=False):
        clauses, arguments = [], []
        for key, value in (("role", role), ("modality", modality), ("primitive", primitive)):
            if value is not None:
                clauses.append(f"{key}=?")
                arguments.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        fields = ("*" if include_metadata else
                  "id,shard,row_,group_,role,primitive,modality,source,"
                  "json_extract(metadata,'$.adaptation') AS adaptation,"
                  "json_extract(metadata,'$.attribute') AS attribute")
        for row in self.db.execute("SELECT " + fields + " FROM records" + where + " ORDER BY shard,row_", arguments):
            yield dict(row)

    def records(self, role=None, modality=None, primitive=None, include_metadata=False):
        return list(self.iter_records(role, modality, primitive, include_metadata))

    def shard_tensors(self, shard):
        from safetensors.torch import load_file

        if shard in self.loaded:
            tensors = self.loaded[shard]
            self.loaded.move_to_end(shard)
        else:
            tensors = load_file(str(self.directory / f"shard-{shard:06d}.safetensors"))
            size = sum(value.numel() * value.element_size() for value in tensors.values())
            while self.loaded and (len(self.loaded) >= 4 or self.loaded_bytes + size > self.cache_bytes):
                _, removed = self.loaded.popitem(last=False)
                self.loaded_bytes -= sum(value.numel() * value.element_size() for value in removed.values())
                del removed
            if size <= self.cache_bytes:
                self.loaded[shard] = tensors
                self.loaded_bytes += size
        return tensors

    def tensors(self, row):
        tensors = self.shard_tensors(row["shard"])
        return {name: tensor[row["row_"]] for name, tensor in tensors.items()}

    def _mask_descriptor(self, shard):
        from safetensors import safe_open

        path = self.directory / f"shard-{shard:06d}.safetensors"
        stamp = _shard_stamp(path)
        cached = self._mask_descriptors.get(shard)
        if cached and cached[0] == stamp:
            self._mask_descriptors.move_to_end(shard)
            return cached[1]
        if cached:
            del self._mask_descriptors[shard]
            self._mask_descriptor_bytes -= len(cached[1])
        with safe_open(str(path), framework="pt") as handle:
            # Shards have at most 16 columns. Zero marks an invalid row, but
            # only selecting that row should fail. Bytes retain no tensor views.
            widths = bytes(0 if sum(mask) < 2 else
                           max(index + 1 for index, valid in enumerate(mask) if valid)
                           for mask in handle.get_tensor("mask").tolist())
        if stamp != _shard_stamp(path):
            raise ValueError("Feature shard changed during mask verification; retry")
        if self.mask_cache_bytes and len(widths) <= self.mask_cache_bytes:
            while self._mask_descriptors and (len(self._mask_descriptors) >= _MASK_CACHE_SHARDS
                    or self._mask_descriptor_bytes + len(widths) > self.mask_cache_bytes):
                _, removed = self._mask_descriptors.popitem(last=False)
                self._mask_descriptor_bytes -= len(removed[1])
            self._mask_descriptors[shard] = (stamp, widths)
            self._mask_descriptor_bytes += len(widths)
        return widths

    def mask_width(self, rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["shard"]].append(row["row_"])
        width = 0
        for shard, indices in groups.items():
            if shard not in self.loaded:
                descriptor = self._mask_descriptor(shard)
                selected = [descriptor[index] for index in indices]
                if 0 in selected:
                    raise ValueError("Feature mask must contain at least two candidates")
                width = max(width, max(selected))
                continue
            masks = self.loaded[shard]["mask"][indices]
            if (masks.sum(dim=1) < 2).any():
                raise ValueError("Feature mask must contain at least two candidates")
            width = max(width, int(masks.any(dim=0).nonzero()[-1]) + 1)
            del masks
        return width

    def batch_tensors(self, rows):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["shard"]].append((index, row["row_"]))
        for shard, entries in groups.items():
            tensors = self.shard_tensors(shard)
            for index, row in entries:
                yield index, {name: value[row] for name, value in tensors.items()}
            del tensors

    def close(self):
        if not self.read_only:
            self.flush()
        self.db.close()
        self.loaded.clear()
        self.loaded_bytes = 0
        self._mask_descriptors.clear()
        self._mask_descriptor_bytes = 0


def collate(store, rows, device="cpu"):
    import torch

    if not rows:
        raise ValueError("Cannot collate an empty batch")
    if isinstance(store, FeatureStore):
        # Read tiny mask tensors first, then visit each feature shard once even
        # when the shuffled role-specific batch exceeds the cache byte budget.
        width = store.mask_width(rows)
        samples = store.batch_tensors(rows)
    else:
        width = 0
        for row in rows:
            sample = store.tensors(row)
            valid = sample["mask"].nonzero().flatten()
            if len(valid) < 2:
                raise ValueError("Feature mask must contain at least two candidates")
            width = max(width, int(valid[-1]) + 1)
            del sample
        samples = ((index, store.tensors(row)) for index, row in enumerate(rows))
    tensors = {}
    for index, sample in samples:
        if not tensors:
            tensors = {name: torch.zeros((len(rows), *value.shape) if name == "query" else
                                         (len(rows), width, *value.shape[1:]), dtype=value.dtype)
                       for name, value in sample.items()}
        for name, value in sample.items():
            if name == "query":
                tensors[name][index].copy_(value)
            else:
                count = min(width, len(value))
                tensors[name][index, :count].copy_(value[:count])
        del sample, value
    return {name: value.to(device) for name, value in tensors.items()}
