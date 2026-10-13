"""Native frozen-feature extraction on the retained optimized EXL3 foundation.

Written against unpublished lab engine commits; research-snapshot/historical-native-source
holds their exllamav3/fuzev.py.
"""

import argparse
import json
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from .acquire import write_json
from .data import acquisition, json_lines
from .encoding import identity as encoding_identity, token_hash
from .extraction import FeatureProxy, PIPELINE_POLICY, write_rows
from .features import FeatureStore
from .paths import ROOT, project_path, source_revision
from .preparation import PREPARATION_POLICY, ReadoutPreparer
from .state import Store
from .telemetry import MemoryGuard, Sampler


class NativeBackend:
    def __init__(self, directory, cache_tokens=294912, memory_fraction=None, optimized=True,
                 candidate_batch_size=4, batch_capacity=4):
        try:
            from exllamav3.fuzev import NativeLane
        except ImportError as error:
            raise ImportError("Requires the historical engine module exllamav3/fuzev.py "
                              "from research-snapshot/historical-native-source") from error
        from exllamav3 import Cache, Config, Model, Tokenizer

        if type(batch_capacity) is not int or not 1 <= batch_capacity <= 8:
            raise ValueError("Native batch capacity must be between one and eight")
        if type(candidate_batch_size) is not int or not 1 <= candidate_batch_size <= batch_capacity:
            raise ValueError("Candidate batch size must fit the declared native capacity")
        self.candidate_batch_size, self.batch_capacity = candidate_batch_size, batch_capacity
        self.memory_fraction = memory_fraction
        if optimized:
            from exllamav3.optimized import enable_prefill_optimizations
            enable_prefill_optimizations()
        self.config = Config.from_directory(str(directory))
        self.config.infer_params.ngram_stream_from_disk = True
        self.model = Model.from_config(self.config)
        self.model.fuzev_batch_capacity = batch_capacity
        self.cache = Cache(self.model, max_num_tokens=cache_tokens, max_batch_size=batch_capacity, max_history=3)
        recurrent_bytes = sum(layer.storage_size() for layer in self.cache.recurrent_layers.values())
        try:
            extra_slots_bytes = recurrent_bytes * max(0, batch_capacity - 4) // batch_capacity
            self.model.load(device="cuda:0", max_chunk_size=16384, max_output_size=1,
                            max_batch_size=batch_capacity, progressbar=False)
        except BaseException:
            self.model.unload()
            raise
        self.tokenizer = Tokenizer.from_config(self.config)
        self.tokenizer.hf_tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.preparer = ReadoutPreparer(self.tokenizer)
        self.lane = NativeLane(self.model, self.cache, context_limit=262144, chunk_size=8192)
        self.prefixes, self.vision, self.images = OrderedDict(), None, OrderedDict()
        try:
            if getattr(self.lane, "batch_capacity", 4) != batch_capacity:
                raise ValueError("Imported native lane does not support the declared physical batch capacity")
        except BaseException:
            self.close()
            raise
        from exllamav3.ext import exllamav3_ext
        self.readout_runtime = {
            "candidate_batch_size": candidate_batch_size,
            "batch_capacity": batch_capacity,
            "model_batch_capacity": self.model.fuzev_batch_capacity,
            "candidate_policy": "equal-length unpadded tails; first-seen length groups; original-order chunks",
            "feature_chunk_tokens": self.lane.chunk_size,
            "context_tokens": self.lane.context_limit,
            "cache_tokens": cache_tokens,
            "max_hybrid_slots": self.cache.num_slots,
            "recurrent_state_storage_bytes": recurrent_bytes,
            "extra_recurrent_state_bytes_vs_capacity4": extra_slots_bytes,
            "memory_fraction": memory_fraction,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "optimized_foundation": optimized,
            "native_extension_sha256": hashlib_file(Path(exllamav3_ext.__file__)),
            "cpu_preparation": dict(PREPARATION_POLICY),
        }

    def set_candidate_batch_size(self, size):
        if type(size) is not int or not 1 <= size <= self.batch_capacity:
            raise ValueError("Candidate batch size must fit the declared native capacity")
        self.candidate_batch_size = size
        self.readout_runtime["candidate_batch_size"] = size

    def encode(self, text, embeddings=None):
        return self.preparer.encode(text, embeddings)

    def make_preparer(self):
        """Construct a private, CPU-only tokenizer on the preparation worker."""
        from exllamav3 import Config, Tokenizer
        config = Config.from_directory(self.config.directory)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.hf_tokenizer = AutoTokenizer.from_pretrained(config.directory, local_files_only=True)
        return ReadoutPreparer(tokenizer)

    def image_embedding(self, path, digest):
        if digest not in self.images:
            if self.vision is None:
                from exllamav3 import Model
                self.vision = Model.from_config(self.config, component="vision")
                self.vision.load(device="cuda:0", progressbar=False)
            with Image.open(path) as image:
                self.images[digest] = self.vision.get_image_embeddings(self.tokenizer, image.convert("RGB"))
        self.images.move_to_end(digest)
        while len(self.images) > 8:
            self.images.popitem(last=False)
        return self.images[digest]

    def base_tokens(self, messages, image_path=None, image_sha256=None, rendered=None):
        if not image_path:
            return self.preparer.base(messages)[0], []
        embeddings = [self.image_embedding(image_path, image_sha256)]
        if rendered is None:
            rendered = self.preparer.render(messages, image=True)
        for embedding in embeddings:
            placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            if rendered.count(placeholder) != len(embeddings):
                raise ValueError("Fixed template did not render the expected image placeholder")
            # Native aliases already encode the start/end vision delimiters.
            rendered = rendered.replace(placeholder, embedding.text_alias, 1)
        return self.encode(rendered, embeddings), embeddings

    def parameters(self, tokens, embeddings, suffix=()):
        if not embeddings:
            return {}
        parameters = {"indexed_embeddings": embeddings}
        # Qwen4Exp uses ordinary linear positions even for images. Only towers
        # whose language model advertises mRoPE use three-axis frequencies.
        if self.model.caps.get("mrope"):
            tokens = tokens + suffix
            ids = torch.tensor([tokens], dtype=torch.long)
            frequencies, _ = self.model.g_rope.get_mrope_freqs(ids, embeddings, len(tokens))
            parameters["inv_freq"] = frequencies
        return parameters

    def evict(self):
        if not self.prefixes:
            return False
        _, (branch, _) = self.prefixes.popitem(last=False)
        self.lane.release(branch)
        return True

    def reserve(self, extra_tokens):
        pages = (extra_tokens + 255) // 256 + 2
        while len(self.lane.pool.free) < pages:
            if not self.evict():
                raise MemoryError("Native decision suffix cannot fit the available KV pool")

    @torch.inference_mode()
    def extract(self, row, baseline=True):
        return self.extract_prepared(row, self.preparer.prepare(row, baseline), baseline)

    @torch.inference_mode()
    def extract_prepared(self, row, prepared, baseline=True):
        if prepared.base is not None:
            (tokens, key), embeddings = prepared.base, []
        else:
            tokens, embeddings = self.base_tokens(row["context"], row.get("image_path"),
                row.get("image_sha256"), rendered=prepared.image_template)
            key = token_hash(tokens)
        suffix, tails = prepared.suffix, prepared.tails
        if len(tokens) + len(suffix) + max(map(len, tails)) > 262144:
            raise ValueError("Record exceeds the native 262144-token window; truncation is disabled")
        reused = key in self.prefixes
        if not reused:
            self.reserve(len(tokens) + len(suffix) + max(map(len, tails)))
            parameters = self.parameters(tokens, embeddings)
            base = self.lane.create()
            try:
                self.lane.append(base, tokens, features=True, params=parameters)
                self.prefixes[key] = (self.lane.freeze(base), embeddings)
            except BaseException:
                if not base.closed:
                    self.lane.release(base)
                raise
        self.prefixes.move_to_end(key)
        while len(self.prefixes) > 8:
            self.evict()
        base, embeddings = self.prefixes[key]
        # Remove the needed prefix from the eviction set while sizing its tails.
        self.prefixes.pop(key)
        try:
            self.reserve(len(suffix) + max(map(len, tails)))
        finally:
            self.prefixes[key] = (base, embeddings)
        query = self.lane.fork(base)
        try:
            feature = self.lane.append(query, suffix, features=True,
                                       params=self.parameters(query.tokens, embeddings, suffix))
            self.lane.freeze(query)
            keys = [None] * len(tails)
            groups = defaultdict(list)
            for index, tail in enumerate(tails):
                groups[len(tail)].append(index)
            forward_batches = []
            for indices in groups.values():
                for offset in range(0, len(indices), self.candidate_batch_size):
                    batch = indices[offset:offset + self.candidate_batch_size]
                    self.reserve(sum(len(tails[index]) + 512 for index in batch))
                    branches = []
                    try:
                        for _ in batch:
                            branches.append(self.lane.fork(query))
                        arguments = self.parameters(query.tokens, embeddings, tails[batch[0]])
                        if len(batch) == 1:
                            features = self.lane.append(branches[0], tails[batch[0]], features=True, params=arguments)
                        else:
                            features = self.lane.append_features_many(branches, [tails[index] for index in batch], params=arguments)
                        owned = features.reshape(len(batch), -1).cpu()
                        del features
                        for index, feature_ in zip(batch, owned):
                            keys[index] = feature_.clone()
                        forward_batches.append(len(batch))
                    finally:
                        for candidate in branches:
                            if not candidate.closed:
                                self.lane.release(candidate)
            logits = None
            if baseline:
                # This is an explicitly separate label-token control. Production
                # decisions never request this projection or sample an answer.
                head = next(module for module, _, _ in self.model.fwd_modules if module.caps.get("logits_output"))
                vocabulary = head.forward(feature.half(), {})[0, -1]
                if prepared.label_ids is None:
                    raise ValueError("Prepared readout omitted label-token baseline IDs")
                logits = vocabulary[list(prepared.label_ids)].float().cpu()
            execution = {"mode": "checkpoint" if reused else "fresh", "prefix_tokens": len(tokens),
                         "reused_tokens": len(tokens) if reused else 0,
                         "forwarded_tokens": (0 if reused else len(tokens)) + len(suffix) + sum(map(len, tails)),
                         "branch_prefix_tokens": len(query.tokens), "expanded_image_tokens": sum(item.mm_length for item in embeddings),
                         "candidate_forward_batches": forward_batches,
                         "base_token_sha256": key,
                         "query_token_sha256": prepared.query_hash or token_hash(query.tokens)}
            return feature.reshape(-1).cpu(), torch.stack(keys), logits, execution
        finally:
            if not query.closed:
                self.lane.release(query)

    def close(self):
        self.lane.close()
        if self.vision:
            self.vision.unload()
        self.model.unload()


def scoped_feature_identity(model_name, readout_runtime):
    import os
    from .scope import EXTERNAL_SOURCES, TRAINING_SOURCES, contract
    model = json.loads(project_path("data/manifests/acquisition.json").read_text())["models"][model_name]
    datasets = acquisition()
    return {"schema_version": 1, "engine": "optimized-native-exllamav3-fuzev", "model": model["repo"],
            "revision": model["revision"], "encoding": encoding_identity(), "cache_precision": "float16",
            "feature_tap": "final-mixer/post-collapse/before-lm-head", "hidden_size": 2560,
            "datasets": {name: datasets[name]["revision"] for name in TRAINING_SOURCES + EXTERNAL_SOURCES},
            "runtime_sources": runtime_sources(), "extraction_pipeline": dict(PIPELINE_POLICY),
            "readout_runtime": dict(readout_runtime),
            "dataset_scope": contract(augmentation=os.environ.get("FUZEV_NOUL_PREFERENCE") == "1",
                                      dimensions=os.environ.get("FUZEV_DIMENSION_CHOICE") == "1")}


def extract_features(model_name, prepared, backend=None, cancel=None):
    model = json.loads(project_path("data/manifests/acquisition.json").read_text())["models"][model_name]
    if not model.get("local_snapshot"):
        raise ValueError("Reference acquisition has not completed; do not load partial weights")
    from .scope import ALL_PREPARATIONS as PREPARATIONS, required_preparations
    if any(path.name not in PREPARATIONS for path in prepared):
        raise ValueError("Only the revised training and external preparations may be extracted")
    import os
    out = project_path(os.environ.get("FUZEV_FEATURE_DIRECTORY", f"data/features/{model_name}-scope-v2"))
    if not out.is_relative_to(project_path("data/features")) or out == project_path("data/features"):
        raise ValueError("Feature destination must be a dedicated directory under data/features")
    store, features = Store(), None
    store.update("data.features", "in_progress", f"Extracting frozen text/image features on {model_name}; resumable shards, no context truncation.")
    owned_backend = backend is None
    sampler, count, skipped, started = Sampler(), 0, 0, time.monotonic()
    guard = MemoryGuard()
    owner = owned_native = None
    try:
        if backend is None:
            owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-gpu-owner")
            owned_native = owner.submit(NativeBackend, model["local_snapshot"]).result()
            backend = FeatureProxy(owned_native, owner)
        identity = scoped_feature_identity(model_name, backend.readout_runtime)
        features = FeatureStore(out, identity)
        for path in prepared:
            features.register_preparation(path, hashlib_file(path))
        guard.start()
        for path in prepared:
            prior_count, prior_skipped = count, skipped

            def on_record(progress):
                nonlocal count, skipped
                count, skipped = prior_count + progress["records"], prior_skipped + progress["skipped"]
                if count % 128 == 0:
                    metric = sampler.sample()
                    guard.check()
                    store.record_result("data.features", {"model": model_name, "records": count,
                        "resumed_records": skipped, "elapsed_seconds": time.monotonic() - started,
                        "records_per_second": count / (time.monotonic() - started), "ram_available": metric["ram_available"],
                        "memory_guard": guard.snapshot()})
                    print(json.dumps({"records": count, "resumed": skipped}), flush=True)
            progress = write_rows(json_lines(path), features, backend, guard, cancel, on_record)
            count, skipped = prior_count + progress["records"], prior_skipped + progress["skipped"]
            features.finish_preparation(path, progress["examined"])
            store.record_result("data.features", {"model": model_name, "preparation": str(path),
                "pipeline": dict(PIPELINE_POLICY), **progress})
        features.flush()
        failures = features.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        summary = {"model": model_name, "records": len(features.records()), "failures": failures,
                   "identity": identity, "elapsed_seconds": time.monotonic() - started}
        write_json(out / "summary.json", summary)
        store.record_result("data.features", summary)
        completed = {Path(row["path"]).name for row in features.preparations() if row["completed"]}
        if model_name == "native-4.05" and set(required_preparations("mixed", identity["dataset_scope"])) <= completed:
            store.update("data.features", "completed", "Full eligible reference feature extraction finished; out-of-window records are explicitly recorded.")
    except BaseException as error:
        store.update("data.features", "cancelled" if isinstance(error, InterruptedError) else "failed",
                     f"Feature extraction stopped: {type(error).__name__}: {error}")
        raise
    finally:
        guard.close()
        if features is not None:
            features.close()
        if owned_backend and owner is not None:
            try:
                if owned_native is not None:
                    owner.submit(owned_native.close).result()
            finally:
                owner.shutdown(wait=True, cancel_futures=True)


def runtime_sources():
    import importlib.util
    native = importlib.util.find_spec("exllamav3")
    directories = {"recipes": Path(__file__).resolve().parents[1],
                   "exllamav3-fuzev": Path(native.origin).resolve().parents[1] if native and native.origin else None}
    return {name: source_revision(directory, "fuzev-research" if name == "recipes" else "exllamav3")
            for name, directory in directories.items()}


def hashlib_file(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="native-4.05")
    parser.add_argument("--prepared", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    extract_features(args.model, args.prepared)


if __name__ == "__main__":
    main()
