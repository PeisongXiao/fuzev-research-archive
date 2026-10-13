"""Opt-in, local, test-only comparison extraction on one fused AsyncLLM."""

import argparse
import asyncio
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, closing, contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import torch

from .acquire import MODELS as ACQUIRED_MODELS, write_json
from .comparison_inputs import INPUTS, MANIFEST, frozen_reference, iter_inputs
from .encoding import identity as encoding_identity, token_hash
from .evaluate import (SUBSET_FILE, TEST_SCOPE, ShardAudit, checked_batch,
                       read_json, semantic_fields, validate_plan)
from .features import FeatureStore, file_sha256
from .preparation import ReadoutPreparer
from .telemetry import MemoryGuard, admit
from .vllm_ple_storage import validate_ple_storage_config, validate_ple_storage_options
from .vllm_receipts import aggregate_forward_receipts, forward_receipt
from .vllm_service import (CONTEXT, VllmService,
                           VllmServiceConfig, _source_revision, _text, _tokens,
                           describe_runtime, drain_tasks, validate_engine_config)


MODELS = ("vllm-nvidia", "vllm-radixark", "vllm-exl3-4.05")
RECEIPT = "vllm-comparison-extraction.json"
RUN_FAILURE = "__vllm_comparison_extraction__"
SOURCE_MODULES = {"vllm-fuzev": "vllm", "vllm-exl3-fuzev": "vllm_exl3",
                  "exllamav3-fuzev": "exllamav3"}
TAP = "final-mixer/post-collapse/before-lm-head"
PROGRESS_INTERVAL = 128


class OfflineAPI:
    @staticmethod
    def pooling(labels=None):
        from vllm.pooling_params import PoolingParams
        return PoolingParams(task="fuzev_features", extra_kwargs=(
            None if labels is None else {"baseline_token_ids": list(labels)}))


class PreparedTokenizer:
    """Adapt the existing HF tokenizer to ReadoutPreparer, not a second model."""

    def __init__(self, client):
        self.client = client

    def encode(self, text, *, encode_special_tokens, embeddings=None):
        if not encode_special_tokens or embeddings is not None:
            raise ValueError("Native indexed image aliases are not vLLM inputs")
        return torch.tensor(self.client._encode(text), dtype=torch.long)

    def hf_render_chat_template(self, messages, **options):
        return self.client.tokenizer.apply_chat_template(messages, tokenize=False, **options)


class ComparisonReadout(VllmService):
    """Reuse exact-token submission, image ownership, guards and abort/drain.

    No ServingHead is constructed. The only new wire payload is the explicit
    offline query control appended by GPUModelRunner._fuzev_readout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preparer = ReadoutPreparer(PreparedTokenizer(self))

    async def _read(self, iterator, tokens, request_id, features):
        if features is not True:
            raise ValueError("Comparison extraction never generates tokens")
        result = None
        async for output in iterator:
            if result is not None:
                raise ValueError("Engine produced output after a final response")
            if not output.finished:
                continue
            receipt = self._receipt(output, tokens, request_id)
            value = output.outputs.data
            if (not isinstance(value, torch.Tensor) or value.ndim != 1
                    or value.numel() not in (2560, *range(2562, 2577))):
                raise ValueError("Expected 2560 features with optional 2..16 offline controls")
            value = value.detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite feature/control readout")
            receipt.update(forward_receipt(output, len(tokens), features=True))
            result = value, receipt
        if result is None:
            raise ValueError("Engine stream ended without a final response")
        return result

    async def readout(self, tokens, labels=None, multimodal=None):
        value, receipt = await self._engine_call(
            _tokens(tokens), self.api.pooling(labels), features=True, multimodal=multimodal)
        if value.shape != (2560 + (len(labels) if labels is not None else 0),):
            raise ValueError("Feature/control width does not match this request")
        return value, receipt

    async def extract(self, row):
        # One row at a time; only its independent candidate wave is concurrent.
        if self.operations:
            raise RuntimeError("Comparison rows must be serialized")
        validate_input(row)
        async with self._operation("comparison_" + uuid4().hex, self.config.model_name):
            prepared = self.preparer.prepare(row, baseline=True)
            labels = prepared.label_ids
            if len(set(labels)) != len(labels):
                raise ValueError("Offline labels must be distinct one-token vocabulary IDs")
            payload = None
            if prepared.base is not None:
                base = _tokens(prepared.base[0])
            else:
                raw = self._encode(prepared.image_template, image=True)
                base, payload = await self._prepare_image(
                    {"path": row["image_path"], "sha256": row["image_sha256"]}, raw)
            query = _tokens(base + prepared.suffix)
            tails = tuple(_tokens(tail) for tail in prepared.tails)
            if len(query) + max(map(len, tails)) > CONTEXT:
                raise ValueError("Record exceeds the 262144-token window; no truncation")
            query_value, query_receipt = await self.readout(query, labels, payload)
            branches, receipts = [], [query_receipt]
            tasks = [asyncio.create_task(self.readout(query + tail, multimodal=payload)) for tail in tails]
            try:
                results = await asyncio.gather(*tasks)
                for value, receipt in results:
                    branches.append(value)
                    receipts.append(receipt)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await drain_tasks(tasks)
            execution = {
                "mode": "fresh", "branch_policy": "independent full-token query+candidate replay",
                "prefix_tokens": len(base), "branch_prefix_tokens": len(query),
                "base_token_sha256": token_hash(base), "query_token_sha256": token_hash(query),
                "baseline_token_ids": list(labels), "requests": receipts,
                "submitted_tokens": sum(item["submitted_tokens"] for item in receipts),
                "reused_tokens": None, "replayed_tokens": None, "checkpoint_retained": False,
                "images": [] if payload is None else payload.receipt(),
                **aggregate_forward_receipts(receipts),
            }
            return query_value[:2560].clone(), torch.stack(branches), query_value[2560:].clone(), execution


def validate_input(row):
    if row.get("role") != "test" or not 2 <= len(row["candidates"]) <= 16:
        raise ValueError("Only canonical test rows with 2..16 candidates are accepted")
    _text(row["question"])
    for message in row["context"]:
        _text(message["content"])
    for candidate in row["candidates"]:
        _text(candidate["text"])
    image = bool(row.get("image_path"))
    if row["modality"] not in {"text", "image"} or (row["modality"] == "image") != image:
        raise ValueError("Canonical image rows require their original acquired image")
    if image and not re.fullmatch(r"[0-9a-f]{64}", row.get("image_sha256", "")):
        raise ValueError("Canonical image RGB checksum is required")


def credential_key(key):
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    parts = name.split("_")
    return (bool(set(parts) & {"password", "passwd", "pwd", "secret", "secrets",
                              "credential", "credentials", "auth", "authentication",
                              "authorization", "bearer"})
            or parts[-1] == "token"
            or bool(re.search(r"(?:^|_)(?:api_key|apikey|private_key|access_key|"
                              r"hf_token|hub_token|auth_token|access_token|refresh_token|"
                              r"signing_key|encryption_key)(?:_|$)", name)))


def reject_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if credential_key(key):
                raise ValueError(f"Credential-named key is forbidden for offline extraction: {key}")
            reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            reject_credentials(item)


def runtime_environment():
    names = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES"}
    keys = sorted(os.environ)
    for key in keys:
        if (key.startswith(("VLLM_", "EXL3_", "FUZEV_", "HF_", "HUGGING_FACE_", "HUGGINGFACE_"))
                and credential_key(key)):
            raise ValueError(f"Credential-named environment key is forbidden: {key}")
    return {key: os.environ[key] for key in keys
            if key in names or key.startswith(("VLLM_", "EXL3_", "FUZEV_"))}


def verify_sources(pins):
    expected = pins["runtime_sources"]
    required = {"recipes", "vllm-fuzev"}
    if pins["runtime_cell"]["quantization"] == "exl3":
        required.update(("vllm-exl3-fuzev", "exllamav3-fuzev"))
    if not required <= set(expected) or set(expected) - {"recipes", *SOURCE_MODULES}:
        raise ValueError("Pin actual recipes/core and, for EXL3, both plugin sources")
    observed = {}
    for name in expected:
        if name == "recipes":
            directory = Path(__file__).resolve().parents[1]
        else:
            spec = importlib.util.find_spec(SOURCE_MODULES[name])
            if spec is None or not spec.origin:
                raise ValueError(f"Pinned runtime source is not importable: {name}")
            directory = Path(spec.origin).resolve().parent
        observed[name] = _source_revision(directory)
    return observed


def artifact_stamp(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def artifact_files(snapshot):
    paths = list(snapshot.rglob("*"))
    if any(path.is_symlink() and path.is_dir() for path in paths):
        raise ValueError("Snapshot directories cannot be symlink overlays")
    if any(not path.is_file() and not path.is_dir() for path in paths):
        raise ValueError("Snapshot/view cannot contain broken links or nonregular files")
    return {path.relative_to(snapshot).as_posix(): path for path in paths if path.is_file()}


def checked_file(path, digest):
    before = artifact_stamp(path)
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or file_sha256(path) != digest or artifact_stamp(path) != before):
        raise ValueError(f"Local artifact checksum changed: {path.name}")
    return before


def artifact_state(pins):
    paths = list(artifact_files(Path(pins["artifact"]["local_snapshot"])).values())
    if "serving_view" in pins:
        view = pins["serving_view"]
        paths.extend(artifact_files(Path(view["directory"])).values())
        paths.append(Path(view["canonical_pack_scan"]["path"]))
    return {str(path): artifact_stamp(path) for path in paths}


def verify_serving_view(view, artifact, files):
    """Verify reviewed loader metadata against canonical bytes, not a repack.

    Pack-scan/header receipts are source evidence, never replacements for the
    caller's complete canonical SHA256 verification. No tensor scan runs here.
    """
    if set(view) != {"directory", "files_sha256", "canonical_pack_scan"}:
        raise ValueError("Pin the EXL3 view directory, all metadata hashes and canonical pack scan")
    directory, snapshot = Path(view["directory"]), Path(artifact["local_snapshot"])
    if not directory.is_absolute() or not directory.is_dir() or directory.resolve() != directory:
        raise ValueError("Require an existing absolute resolved EXL3 serving view")
    separate_output(directory, [snapshot])
    derived = artifact_files(directory)
    tensors = {name for name in files if name.endswith(".safetensors")}
    required = {"config.json", "config.json.native", "quantization_config.json",
                "pack_scan.json", "serving-view.json"}
    index, backup = "model.safetensors.index.json", "model.safetensors.index.json.native"
    overrides = required | {index, backup}
    if (not (set(files) | required) <= set(derived) or set(derived) - set(files) - overrides
            or {name for name in derived if name.endswith(".safetensors")} != tensors
            or set(view["files_sha256"]) != set(derived) - tensors):
        raise ValueError("EXL3 view must preserve the complete canonical payload and pin all metadata")
    stamps = {}
    for name, path in sorted(derived.items()):
        if name not in overrides and not path.samefile(files[name]):
            raise ValueError(f"EXL3 view file is not the canonical same-file payload/metadata: {name}")
        stamps[str(path)] = (artifact_stamp(path) if name in tensors
                             else checked_file(path, view["files_sha256"][name]))
    hashes = artifact["files_sha256"]
    if view["files_sha256"]["config.json.native"] != hashes["config.json"]:
        raise ValueError("EXL3 view native config must preserve the canonical config bytes")
    if backup in derived and view["files_sha256"][backup] != hashes.get(index):
        raise ValueError("EXL3 view native index must preserve the canonical index bytes")

    receipt = read_json(directory / "serving-view.json")
    if (receipt.get("schema_version") != 1 or receipt.get("source") != str(snapshot)
            or receipt.get("source_config_sha256") != hashes["config.json"]
            or receipt.get("tensor_conversion") is not False
            or receipt.get("configuration_only") is not True or receipt.get("gpu_qualified") is not False
            or set(receipt.get("payload", {})) != tensors):
        raise ValueError("EXL3 serving-view receipt does not prove the pinned canonical source")
    total = 0
    for name in sorted(tensors):
        entry, path = receipt["payload"][name], files[name]
        if (entry.get("source") != str(path.resolve()) or entry.get("same_file") is not True
                or entry.get("bytes") != path.stat().st_size
                or not re.fullmatch(r"[0-9a-f]{64}", entry.get("header_sha256", ""))):
            raise ValueError(f"EXL3 serving-view payload receipt differs from canonical file: {name}")
        total += path.stat().st_size
    if receipt.get("payload_bytes") != total:
        raise ValueError("EXL3 serving-view payload byte count differs from canonical files")

    proof = view["canonical_pack_scan"]
    if set(proof) != {"path", "sha256"}:
        raise ValueError("Pin the existing canonical pack-scan receipt path and checksum")
    scan_path = Path(proof["path"])
    if not scan_path.is_absolute() or not scan_path.is_file() or scan_path.resolve() != scan_path:
        raise ValueError("Canonical pack-scan proof must be an existing absolute resolved file")
    stamps[str(scan_path)] = checked_file(scan_path, proof["sha256"])
    canonical_scan, scan = read_json(scan_path), read_json(directory / "pack_scan.json")
    if (canonical_scan.pop("pack", None) != str(snapshot) or scan.pop("pack", None) != str(directory)
            or scan != canonical_scan or not isinstance(scan.get("tensors"), int) or scan["tensors"] <= 0
            or scan.get("ngram_problems") != []):
        raise ValueError("EXL3 view pack scan differs from the pinned canonical scan")

    native, config = read_json(directory / "config.json.native"), read_json(directory / "config.json")
    original = (native.get("text_config", native).get("quantization_config")
                or native.get("quantization_config") or {})
    quant = read_json(directory / "quantization_config.json")
    if (quant.get("quant_method") != "exl3" or quant.get("native_quantization_config") != original
            or quant.get("derived_from_headers") != "pack_scan.json"
            or config.get("quantization_config") != quant):
        raise ValueError("EXL3 derived quantization config lacks the canonical native config proof")
    expected = json.loads(json.dumps(native))
    expected["quantization_config"] = quant
    if isinstance(expected.get("text_config"), dict) and "quantization_config" in expected["text_config"]:
        expected["text_config"]["quantization_config"] = quant
    if config != expected:
        raise ValueError("EXL3 view changed config outside the reviewed quantization adaptation")
    if index in derived and view["files_sha256"][index] != hashes.get(index):
        weight_map = read_json(derived[index]).get("weight_map", {})
        if (not isinstance(weight_map, dict) or len(weight_map) != scan["tensors"]
                or set(weight_map.values()) != tensors):
            raise ValueError("EXL3 derived index must cover the scanned canonical tensor payload only")
    return stamps


def verify_artifact(cell, pins, options, reference_identity):
    reject_credentials(options)
    reject_credentials(pins)
    environment = runtime_environment()
    required = {"schema_version", "artifact", "runtime_cell", "runtime_sources", "runtime_environment"}
    if (cell not in MODELS or not required <= set(pins) or set(pins) - required - {"serving_view"}
            or ("serving_view" in pins and cell != "vllm-exl3-4.05")):
        raise ValueError("Require explicit supported vLLM comparison pins")
    if pins["schema_version"] != 1:
        raise ValueError("Unsupported vLLM comparison pin schema")
    artifact = pins["artifact"]
    if set(artifact) != {"repo", "revision", "local_snapshot", "files_sha256"}:
        raise ValueError("Pin the complete local artifact, not only a model alias")
    model_key = "native-4.05" if cell == "vllm-exl3-4.05" else cell
    if (artifact["repo"], artifact["revision"]) != ACQUIRED_MODELS[model_key]:
        raise ValueError("Comparison artifact is not the pinned canonical model revision")
    if (reference_identity["model"], reference_identity["revision"]) != ACQUIRED_MODELS["native-4.05"]:
        raise ValueError("The immutable reference must be the canonical native-4 artifact")
    if cell == "vllm-exl3-4.05" and (
            artifact["repo"] != reference_identity["model"]
            or artifact["revision"] != reference_identity["revision"]):
        raise ValueError("vLLM EXL3 must use the same canonical native-4 reference artifact")
    snapshot = Path(artifact["local_snapshot"])
    if not snapshot.is_absolute() or not snapshot.is_dir() or snapshot.resolve() != snapshot:
        raise ValueError("Require an existing absolute local snapshot, never a remote model")
    model = pins.get("serving_view", {}).get("directory", str(snapshot))
    if (options.get("model") != model or options.get("tokenizer") not in {model, str(snapshot)}
            or any(pins["runtime_cell"].get(key) != options.get(key) for key in ("model", "tokenizer"))):
        raise ValueError("Actual model/tokenizer must use the pinned snapshot or explicit EXL3 serving view")
    for key in ("revision", "tokenizer_revision"):
        if options.get(key) not in (None, artifact["revision"]):
            raise ValueError("Engine revision differs from the local artifact")
    if (options.get("trust_remote_code", False)
            or options.get("load_format", "auto") not in {"auto", "safetensors"}):
        raise ValueError("Only local standard weight loading without remote code is supported")
    quantization = pins["runtime_cell"]["quantization"]
    if (cell == "vllm-exl3-4.05") != (quantization == "exl3"):
        raise ValueError("Actual quantization does not match the requested comparison cell")
    if quantization == "modelopt_mixed" and cell != "vllm-nvidia":
        raise ValueError("modelopt_mixed is supported only for the canonical NVIDIA comparison")
    if cell != "vllm-exl3-4.05" and quantization not in {
            "modelopt", "modelopt_fp4", "modelopt_mixed", "compressed-tensors"}:
        raise ValueError("NVFP4 comparisons require the actual pinned NVFP4 quantization path")
    if pins["runtime_environment"] != environment:
        raise ValueError("Runtime environment differs from the pinned extraction environment")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Offline Hugging Face and Transformers mode is required")
    validate_ple_storage_options(options, quantization=quantization)
    files, hashes = artifact_files(snapshot), artifact["files_sha256"]
    if (set(files) != set(hashes) or not {"config.json", "tokenizer.json"} <= set(files)
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("Checksums must cover every file of the complete model/tokenizer snapshot")
    stamps = {}
    if quantization == "modelopt_mixed":
        # Check the small canonical config before hashing tensor payloads.
        stamps[str(files["config.json"])] = checked_file(files["config.json"], hashes["config.json"])
        config = read_json(files["config.json"])
        text = config.get("text_config", config) if isinstance(config, dict) else {}
        quant = (text.get("quantization_config") or config.get("quantization_config", {})
                 if isinstance(text, dict) and isinstance(config, dict) else {})
        scheme = quant.get("quantization", quant) if isinstance(quant, dict) else {}
        layers = scheme.get("quantized_layers", {}) if isinstance(scheme, dict) else {}
        ple = (layers.get("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", {})
               if isinstance(layers, dict) else {})
        if (not isinstance(quant, dict) or quant.get("quant_method") != "modelopt"
                or not isinstance(scheme, dict) or scheme.get("quant_algo") != "MIXED_PRECISION"
                or not isinstance(ple, dict) or ple.get("quant_algo") != "FP8"):
            raise ValueError("Canonical NVIDIA modelopt_mixed requires its MIXED_PRECISION config and FP8 PLE entry")
    stamps.update({str(path): checked_file(path, hashes[name]) for name, path in sorted(files.items())
                   if str(path) not in stamps})
    if "serving_view" in pins:
        stamps.update(verify_serving_view(pins["serving_view"], artifact, files))
    if artifact_state(pins) != stamps:
        raise ValueError("Pinned model snapshot/view changed during verification")
    return stamps


def feature_identity(cell, pins, options, reference, reference_identity):
    runtime = pins["runtime_cell"]
    return {
        "schema_version": 1, "engine": "vllm-fuzev", "cell": cell,
        "model": pins["artifact"]["repo"], "revision": pins["artifact"]["revision"],
        "artifact": pins["artifact"], "runtime_sources": pins["runtime_sources"],
        **({"serving_view": pins["serving_view"]} if "serving_view" in pins else {}),
        "runtime_environment": pins["runtime_environment"], "readout_runtime": runtime,
        "engine_options": options, "cache_precision": runtime["cache_dtype_config"],
        "encoding": encoding_identity(), "feature_tap": TAP, "hidden_size": 2560,
        "datasets": reference_identity["datasets"], "corpus_scope": TEST_SCOPE,
        **({"dataset_scope": reference_identity["dataset_scope"]}
           if "dataset_scope" in reference_identity else {}),
        "reference_head": reference,
        "extraction_pipeline": {"module": "fuzev_research.vllm_comparison_extract",
                                "source_sha256": file_sha256(Path(__file__)),
                                "baseline": "offline label-token controls",
                                "rows": "serial; no generation or fitting"},
    }


@asynccontextmanager
async def engine_context(options):
    from vllm.engine.arg_utils import AsyncEngineArgs
    config = AsyncEngineArgs(**options).create_engine_config()
    validate_engine_config(config)
    validate_ple_storage_options(options, quantization=config.model_config.quantization)
    validate_ple_storage_config(config)
    admit(128 << 20)
    from vllm.v1.engine.async_llm import AsyncLLM
    engine = AsyncLLM.from_vllm_config(config)
    try:
        yield engine
    finally:
        engine.shutdown()


def separate_output(output, protected):
    for directory in protected:
        directory = Path(directory).expanduser().resolve()
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("Comparison output must not overlap frozen inputs, reference, or model")


def storage_pins(output):
    wal = output / "index.sqlite3-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Writer must be drained before recording resume storage pins")
    paths = [output / "identity.json", output / "index.sqlite3",
             *sorted(output.glob("shard-*.safetensors")), *sorted(output.glob("shard-*.json"))]
    if any(path.is_symlink() or path.stat().st_nlink != 1 for path in paths):
        raise ValueError("Comparison storage must not alias another corpus")
    return {path.name: file_sha256(path) for path in paths}


def finish_storage(output, identity, error=None):
    # Keep the run failure marker until even the outer freeze and engine close
    # have succeeded. A complete row count alone must never make a seal possible.
    with closing(FeatureStore(output, identity)) as store:
        if error is not None:
            store.fail(RUN_FAILURE, error)
        else:
            with store.db:
                store.db.execute("DELETE FROM failures WHERE id=?", (RUN_FAILURE,))
        busy, _, _ = store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise RuntimeError("Comparison index is still busy")
    return storage_pins(output)


@contextmanager
def output_lock(output, resume):
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    fd = os.open(output / ".vllm-comparison.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_corpus(features):
    if features.preparations() or features.db.execute(
            "SELECT 1 FROM records WHERE role!='test' OR role IS NULL LIMIT 1").fetchone():
        raise ValueError("Test-only comparison cannot contain full preparations or training roles")


def audit_resume(features, rows):
    check_corpus(features)
    audit, found = ShardAudit(features), 0
    for row in rows:
        validate_input(row)
        actual = features.db.execute("SELECT * FROM records WHERE id=?", (row["id"],)).fetchone()
        if actual is None:
            continue
        meta = json.loads(actual["metadata"])
        if (any(meta.get(key) != row.get(key) for key in semantic_fields(meta))
                or meta.get("has_baseline") is not True):
            raise ValueError("Resume record differs from canonical test semantics or lacks controls")
        checked_batch(features, [dict(actual)], [meta], audit)
        found += 1
    if found != features.db.execute("SELECT count(*) FROM records").fetchone()[0]:
        raise ValueError("Resume has extra noncanonical features")
    audit.finish()


def write_progress(path, progress, counts, started, *, force=False):
    processed = counts["records"] + counts["skipped"]
    if not force and (not processed or processed % PROGRESS_INTERVAL):
        return
    elapsed = max(0., time.monotonic() - started)
    progress.update(counts=dict(counts), observed=time.time(), elapsed_seconds=elapsed,
                    records_per_second=counts["records"] / elapsed if elapsed else 0.)
    write_json(path, progress)


async def extract_vllm(cell, inputs, inputs_sha256, reference, output, options, pins, *,
                       resume=False, timeout=3600, engine_factory=engine_context,
                       inspector=describe_runtime, source_checker=verify_sources,
                       guard_factory=MemoryGuard, admission=admit, api=None):
    """No seal or quality claim: return only after writer/client/engine drainage."""
    if cell not in MODELS or not 1 <= timeout <= 86400:
        raise ValueError("Unsupported comparison cell or request timeout")
    # Reject before making a durable binding, identity, failure or progress receipt.
    reject_credentials(options)
    reject_credentials(pins)
    runtime_environment()
    # Own operator metadata before asynchronous code can observe caller mutation.
    reference, options, pins = json.loads(json.dumps([reference, options, pins], allow_nan=False))
    validate_ple_storage_options(options, quantization=pins["runtime_cell"]["quantization"])
    output, inputs = Path(output).expanduser().resolve(), Path(inputs).expanduser().resolve()
    separate_output(output, [reference["features"], reference["head"], inputs,
                             pins["artifact"]["local_snapshot"], Path(__file__).resolve().parents[1]])
    if "serving_view" in pins:
        separate_output(output, [pins["serving_view"]["directory"],
                                 pins["serving_view"]["canonical_pack_scan"]["path"]])
    binding = {"cell": cell, "inputs": str(inputs), "inputs_manifest_sha256": inputs_sha256,
               "reference": reference, "options": options, "pins": pins, "timeout": timeout}
    receipt_path, progress, features = output / RECEIPT, None, None
    counts = {"examined": 0, "records": 0, "skipped": 0}
    ownership = ExitStack()
    postflight = False
    try:
        async with AsyncExitStack() as stack:
            frozen = stack.enter_context(frozen_reference(reference))
            # Exhaust the complete validator, including its final checksum/freeze
            # checks, BEFORE importing/constructing an engine or writing features.
            preflight = 0
            with closing(iter_inputs(inputs, inputs_sha256, reference)) as rows:
                for row in rows:
                    validate_input(row)
                    preflight += 1
            if not preflight:
                raise ValueError("Canonical test inputs cannot be empty")
            stamps = verify_artifact(cell, pins, options, frozen.identity)
            observed_sources = source_checker(pins)
            protected = [inputs / INPUTS, inputs / MANIFEST,
                         Path(reference["features"]) / "identity.json",
                         Path(reference["features"]) / "index.sqlite3",
                         *(Path(reference["head"]) / name for name in reference["sha256"])]
            input_stamps = {path: artifact_stamp(path) for path in protected}

            def unchanged():
                source_checker(pins)
                if artifact_state(pins) != stamps:
                    raise ValueError("Pinned model snapshot/view or canonical scan changed during extraction")
                if {path: artifact_stamp(path) for path in protected} != input_stamps:
                    raise ValueError("Frozen input/reference files changed during extraction")
                if runtime_environment() != pins["runtime_environment"]:
                    raise ValueError("Runtime environment changed during extraction")

            identity = feature_identity(cell, pins, options, reference, frozen.identity)
            identity["comparison_inputs"] = {"directory": str(inputs), "manifest_sha256": inputs_sha256}
            if output.exists() != resume:
                raise ValueError("Existing output requires explicit resume; resume requires existing output")
            ownership.enter_context(output_lock(output, resume))
            if (output / SUBSET_FILE).exists():
                raise ValueError("Never resume or mutate already sealed comparison features")
            if resume:
                previous = read_json(receipt_path)
                if (previous.get("resume_allowed") is not True or previous.get("binding") != binding
                        or previous.get("storage") != storage_pins(output)):
                    raise ValueError("Resume requires exact pins and unchanged drained storage")
                with closing(FeatureStore(output, identity, read_only=True)) as reader:
                    with closing(iter_inputs(inputs, inputs_sha256, reference)) as rows:
                        audit_resume(reader, rows)
            started = time.monotonic()
            progress = {"schema_version": 1, "binding": binding, "started": time.time(),
                        "status": "starting", "preflight_records": preflight,
                        "observed_runtime_sources": observed_sources,
                        "runtime_qualified": False, "requires_independent_seal": True,
                        "resume_allowed": False,
                        "storage": previous["storage"] if resume else None}
            write_progress(receipt_path, progress, counts, started, force=True)
            features = FeatureStore(output, identity)

            def close_writer():
                try:
                    features.flush()
                    busy, _, _ = features.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if busy:
                        raise RuntimeError("Comparison writer could not drain its WAL")
                    features.close()
                finally:
                    features.db.close()

            stack.callback(close_writer)
            check_corpus(features)
            features.fail(RUN_FAILURE, "Extraction, postflight verification and drainage pending")
            guard = guard_factory()
            stack.callback(guard.close)
            guard.start()
            guard.check()
            admission(128 << 20)
            engine = await stack.enter_async_context(engine_factory(options))
            actual = inspector(engine)
            if actual != {key: pins[key] for key in ("runtime_cell", "runtime_sources")}:
                raise ValueError("Actual AsyncLLM runtime does not match the pinned cell/sources")
            if not {"generate", "fuzev_features"} <= set(await engine.get_supported_tasks()):
                raise ValueError("One fused generation/readout engine is required")
            config = VllmServiceConfig(
                model_name=cell, head_directory=reference["head"],
                manifest_sha256=reference["sha256"]["manifest.json"],
                calibration_sha256=reference["sha256"]["calibration.json"],
                feature_identity=frozen.identity, request_timeout_seconds=timeout, **actual)
            client = ComparisonReadout(engine, engine.get_tokenizer(), None, config, actual,
                                       guard, api=api or OfflineAPI(), admission=admission)
            stack.push_async_callback(client.close)
            progress.update(status="extracting", feature_identity_sha256=file_sha256(output / "identity.json"))
            write_progress(receipt_path, progress, counts, started, force=True)
            rows = stack.enter_context(closing(iter_inputs(inputs, inputs_sha256, reference)))
            for row in rows:
                guard.check()
                counts["examined"] += 1
                if features.contains(row["id"]):
                    counts["skipped"] += 1
                else:
                    try:
                        features.add(row, *(await client.extract(row)))
                        counts["records"] += 1
                    except BaseException as error:
                        features.fail(row["id"], type(error).__name__ + ": " + str(error))
                        raise
                write_progress(receipt_path, progress, counts, started)
            postflight = True
            features.flush()
            committed = features.db.execute("SELECT count(*) FROM records").fetchone()[0]
            if (counts["examined"] != preflight or committed != preflight
                    or features.db.execute("SELECT count(*) FROM failures WHERE id!=?", (RUN_FAILURE,)).fetchone()[0]):
                raise ValueError("Comparison has missing/extra features or unresolved failures")
            unchanged()
            if inspector(engine) != actual:
                raise ValueError("Runtime identity changed during extraction")
            progress.update(committed=committed)
        progress.update(status="extracted_pending_independent_seal",
                        finished=time.time(), storage=finish_storage(output, identity), resume_allowed=True)
        write_progress(receipt_path, progress, counts, started, force=True)
        return progress
    except BaseException as error:
        if progress is not None:
            status = ("memory_infeasible" if isinstance(error, (MemoryError, torch.OutOfMemoryError)) else "cancelled"
                      if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, InterruptedError))
                      else "failed")
            progress.update(status=status, error=type(error).__name__ + ": " + str(error),
                            finished=time.time(), storage=None, resume_allowed=False)
            if not postflight:
                try:
                    unchanged()
                    if counts["records"] and inspector(engine) != actual:
                        raise ValueError("Runtime identity changed before failure")
                    progress["resume_allowed"] = True
                except Exception as changed:
                    progress["resume_blocker"] = str(changed)
            else:
                progress["resume_blocker"] = "Postflight/cleanup failed; committed rows must not be adopted"
            if features is not None:
                try:
                    progress["storage"] = finish_storage(output, identity, progress["error"])
                except Exception as storage_error:
                    progress["storage_error"] = str(storage_error)
            write_progress(receipt_path, progress, counts, started, force=True)
        raise
    finally:
        ownership.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_inference:
        parser.error("Explicit --allow-inference required; never run alongside the live extractor")
    plan = read_json(args.plan)
    validate_plan(plan)
    result = asyncio.run(extract_vllm(
        args.model, args.inputs, args.inputs_sha256, plan["reference"], args.output,
        read_json(args.engine_config), read_json(args.pins), resume=args.resume,
        timeout=args.request_timeout))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
