"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks.

All methods run on the existing single GPU owner. Retained state is immutable;
the accepted sequence can extend one token past its consumed checkpoint.
"""

import math
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from threading import get_ident
from uuid import uuid4
from weakref import WeakValueDictionary

import torch

from .contracts import DecisionResponse, Execution, Score
from .encoding import TEMPLATE, candidate_suffix, identity as encoding_identity
from .encoding import query_suffix, token_hash
from .prefix import identity_digest
from .paths import project_path
from .service_contracts import TextMessage
from .service_images import (IMAGE_ADMISSION_BYTES, ImageBinding, checked_image,
                             embeddings_for, validate_image_tokens)
from .serving_head import TAP, ServingHead
from .telemetry import MemoryGuard, admit


IMAGE_MARKERS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                 "<|video_pad|>", "<|image|>", "<$EMB_")


def text_only(text):
    if not isinstance(text, str) or any(marker in text for marker in IMAGE_MARKERS):
        raise ValueError("Use the explicit trusted image parameter, not image markers or structured content")
    return text


def text_messages(messages):
    result = [TextMessage.model_validate(message).model_dump() for message in messages]
    for message in result:
        text_only(message["content"])
    return result


def generation_timing(prompt_tokens, completion_tokens, started, first, last):
    """CPU-observed boundaries; sampling already owns each token on CPU."""
    result = {
        "schema_version": 1, "source": "native-owner-cpu-observed",
        "status": "unavailable", "reason": "invalid_monotonic_boundaries",
        "scope": "prompt forward through first CPU sample; first to last CPU sample; not GPU-kernel-only",
        "prefill_seconds": None, "decode_seconds": None, "queue_seconds": None,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "decode_tokens": completion_tokens - 1, "reported_cached_prompt_tokens": 0,
        "num_preemptions": None,
    }
    boundaries = (started, first, last)
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in boundaries) or not started < first <= last
            or (completion_tokens == 1 and last != first)
            or (completion_tokens > 1 and last <= first)):
        return result
    result.update(status="available", reason=None, prefill_seconds=first - started,
                  decode_seconds=last - first)
    return result


@dataclass(frozen=True)
class RetainedPrefix:
    id: str
    accepted: tuple[int, ...]
    branch: object
    identity: str
    images: tuple[ImageBinding, ...] = ()

    @property
    def image_hashes(self):
        return tuple(image.sha256 for image in self.images)


def build_service(backend, model_name, model, sources, config):
    """Pin actual serving separately from the immutable reference when opted in."""
    identity = config.feature_identity
    actual = {
        "schema_version": 1, "engine": "optimized-native-exllamav3-fuzev",
        "model": model["repo"], "revision": model["revision"],
        "encoding": encoding_identity(), "cache_precision": "float16",
        "feature_tap": TAP, "hidden_size": 2560,
        "readout_runtime": dict(backend.readout_runtime),
    }
    serving_identity = {"cell": model_name, **actual}
    if config.serving_identity is None:
        if any(identity.get(key) != value for key, value in actual.items()):
            raise ValueError("Serving runtime is incompatible with the head's feature identity")
    else:
        if model_name not in {"native-3.05", "native-4.05", "native-5.05"}:
            raise ValueError("Explicit native transfer requires a required native bitrate cell")
        if config.serving_identity != serving_identity:
            raise ValueError("Actual native serving identity does not match the operator's pins")
        if any(identity.get(key) != actual[key] for key in (
                "schema_version", "engine", "encoding", "feature_tap", "hidden_size")):
            raise ValueError("Reference head has incompatible native encoding/features")
    origin = identity.get("runtime_sources", {})
    if not identity.get("datasets"):
        raise ValueError("Missing or incompatible extraction provenance")
    # CPU FP32 heads are tiny; never load a second backbone or fit in a request.
    admit(128 << 20, backend.readout_runtime["memory_fraction"])
    head = ServingHead(config)
    guard = MemoryGuard(fraction=backend.readout_runtime["memory_fraction"])
    try:
        guard.start()
        return NativeService(
            backend, model_name, head,
            {"feature_identity": deepcopy(identity), "serving_identity": deepcopy(serving_identity),
             "service_sources": deepcopy(sources),
             "recorded_service_sources": deepcopy(config.service_sources),
             "recorded_extraction_sources": deepcopy(origin),
             "transfer_qualification": "unqualified; actual cell quality and runtime gates required",
             "head_manifest_sha256": head.manifest_sha256,
             "calibration_sha256": config.calibration_sha256}, guard,
            capacity=config.prefix_capacity, ttl=config.prefix_ttl_seconds)
    except BaseException:
        guard.close()
        raise


class NativeService:
    def __init__(self, backend, model_name, head, identity, guard, *,
                 capacity=8, ttl=900, clock=time.monotonic, admission=admit,
                 phase_clock=time.perf_counter):
        size, physical = backend.candidate_batch_size, backend.batch_capacity
        runtime = backend.readout_runtime
        if (type(size) is not int or type(physical) is not int
                or not 1 <= size <= physical <= 8 or capacity < 1 or ttl <= 0
                or backend.lane.batch_capacity != physical
                or runtime.get("candidate_batch_size") != size
                or any(runtime.get(key) != physical for key in
                       ("batch_capacity", "model_batch_capacity", "max_hybrid_slots"))):
            raise ValueError("Invalid native service resource bounds")
        self.backend, self.lane = backend, backend.lane
        self.model_name, self.head, self.guard = model_name, head, guard
        self.provenance = identity
        self.identity = identity_digest(identity)
        self.readout_runtime = deepcopy(runtime)
        self.owner = get_ident()
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.phase_clock = phase_clock
        self.admission = admission
        self.prefixes = OrderedDict()
        self.verified_images = WeakValueDictionary()
        self.closed = False

    def _owner(self):
        if get_ident() != self.owner or self.closed:
            raise RuntimeError("Native service requires its live, single GPU owner")

    def _check(self, cancel):
        self._owner()
        if cancel.is_set():
            raise InterruptedError("Native request cancelled at a target-forward boundary")
        if (self.backend.readout_runtime != self.readout_runtime
                or self.backend.candidate_batch_size != self.readout_runtime["candidate_batch_size"]
                or self.backend.batch_capacity != self.readout_runtime["batch_capacity"]
                or self.lane.batch_capacity != self.readout_runtime["batch_capacity"]):
            raise ValueError("Native readout geometry changed after calibrated-head admission")
        self.guard.check()
        self.admission(0, self.backend.readout_runtime["memory_fraction"])

    def _release(self, branch):
        if branch is not None and not branch.closed:
            self.lane.release(branch)

    def release_prefix(self, prefix_id):
        self._owner()
        value = self.prefixes.pop(prefix_id, None)
        if value is not None:
            try:
                self._release(value[0].branch)
            finally:
                self._release_images(value[0].images)
        return value is not None

    def expire(self):
        self._owner()
        now = self.clock()
        for key, (_, used) in list(self.prefixes.items()):
            if now - used >= self.ttl:
                self.release_prefix(key)

    def _reserve(self, tokens, protected=None):
        # Feature-pilot cached prefixes and service prefixes use the SAME pool.
        while True:
            try:
                self.backend.reserve(tokens)
                return
            except MemoryError:
                key = next((key for key in self.prefixes if key != protected), None)
                if key is None:
                    raise
                self.release_prefix(key)

    def _get_prefix(self, key):
        self.expire()
        prefix, _ = self.prefixes[key]
        if prefix.identity != self.identity:
            raise ValueError("Incompatible prefix identity")
        try:
            for image in prefix.images:
                image.validate()
            validate_image_tokens(prefix.accepted, prefix.images)
        except ValueError:
            self.release_prefix(key)
            raise
        consumed = tuple(prefix.branch.tokens)
        if (prefix.branch.closed or not prefix.branch.immutable
                or consumed != prefix.accepted[:len(consumed)]
                or not 0 <= len(prefix.accepted) - len(consumed) <= 1):
            raise ValueError("Retained native checkpoint does not cover the accepted tokens")
        self.prefixes.move_to_end(key)
        self.prefixes[key] = (prefix, self.clock())
        return prefix

    def _retain(self, branch, accepted, images=()):
        if (tuple(branch.tokens) != accepted[:len(branch.tokens)]
                or not 0 <= len(accepted) - len(branch.tokens) <= 1):
            raise ValueError("Cannot retain a mismatched accepted/consumed boundary")
        validate_image_tokens(accepted, images)
        self.lane.freeze(branch)
        key = "prefix_" + uuid4().hex
        self.prefixes[key] = (RetainedPrefix(key, accepted, branch, self.identity, tuple(images)), self.clock())
        while len(self.prefixes) > self.capacity:
            self.release_prefix(next(iter(self.prefixes)))
        return key

    @contextmanager
    def _acquire_image(self, source, cancel, images):
        self._check(cancel)
        self.admission(IMAGE_ADMISSION_BYTES, self.readout_runtime["memory_fraction"])
        with checked_image(source, project_path("data/images")) as snapshot:
            self._check(cancel)
            cached = self.backend.images.get(source.sha256)
            if cached is not None and self.verified_images.get(source.sha256) is not cached:
                # Feature requests historically trusted a supplied digest. Do
                # not inherit an unverified cache association into this API.
                self.backend.images.pop(source.sha256)
            embedding = self.backend.image_embedding(snapshot, source.sha256)
            try:
                binding = ImageBinding.capture(source.sha256, embedding)
            except BaseException:
                if self.backend.images.get(source.sha256) is embedding:
                    self.backend.images.pop(source.sha256)
                raise
            images.append(binding)
            self.verified_images[source.sha256] = embedding
            self._check(cancel)
            yield binding, snapshot

    def _release_images(self, images):
        # Removing our lookup entry never mutates an embedding or another
        # prefix. Live feature-cache and service prefixes retain their objects.
        live = {id(image.embedding) for prefix, _ in self.prefixes.values() for image in prefix.images}
        live.update(id(embedding) for _, embeddings in self.backend.prefixes.values() for embedding in embeddings)
        for image in images:
            if (id(image.embedding) not in live
                    and self.backend.images.get(image.sha256) is image.embedding):
                self.backend.images.pop(image.sha256)

    def _append(self, branch, tokens, cancel, *, features, protected=None, images=()):
        """Check cancellation/headroom between the existing native chunk sizes."""
        if not tokens:
            raise ValueError("An exact nonempty token sequence is required")
        self._check(cancel)
        validate_image_tokens(tokens, images)
        # Supply the whole planned suffix, including any image that straddles a
        # chunk. Native mRoPE needs its full grid; linear Qwen4Exp stays linear.
        parameters = self.backend.parameters(branch.tokens, embeddings_for(images), tokens)
        result = None
        for offset in range(0, len(tokens), self.lane.chunk_size):
            self._check(cancel)
            chunk = tokens[offset:offset + self.lane.chunk_size]
            self._reserve(len(chunk), protected)
            last = offset + len(chunk) == len(tokens)
            result = self.lane.append(branch, chunk, features=features or not last, params=parameters)
            self._check(cancel)
        return result

    def _model(self, model):
        if model != self.model_name:
            raise ValueError("Request model does not match the loaded native backbone")

    def _generation_prompt(self, request, cancel, images):
        prompt = text_only(request.prompt) if request.prompt is not None else None
        messages = (text_messages([message.model_dump() for message in request.messages])
                    if request.messages is not None else None)
        source = (self._acquire_image(request.image, cancel, images)
                  if request.image else nullcontext((None, None)))
        with source as (image, snapshot):
            if prompt is not None:
                if image:
                    return self.backend.encode(image.embedding.text_alias + prompt, [image.embedding])
                return self.backend.encode(prompt)
            if image:
                # Same placement as ReadoutPreparer: image first in the first
                # user turn. Only generation adds the assistant-start suffix.
                user = next((message for message in messages if message["role"] == "user"), None)
                if user is None:
                    user = {"role": "user", "content": "Consider the supplied image."}
                    messages.insert(0, user)
                user["content"] = [{"type": "image"}, {"type": "text", "text": user["content"]}]
            rendered = self.backend.tokenizer.hf_render_chat_template(
                messages, add_generation_prompt=True, chat_template=TEMPLATE.read_text(),
                enable_thinking=False, reasoning_effort="none", preserve_thinking=True)
            if image:
                tokens, embeddings = self.backend.base_tokens(
                    messages, snapshot, image.sha256, rendered=rendered)
                if len(embeddings) != 1 or embeddings[0] is not image.embedding:
                    raise ValueError("Native generation image instance changed during encoding")
                return tokens
            return self.backend.encode(rendered)

    @staticmethod
    def _sample(logits, request, generator):
        values = logits[0, -1].detach().float()
        if values.ndim != 1:
            raise ValueError("Invalid native generation logits")
        finite = torch.isfinite(values)
        if not bool((finite | torch.isneginf(values)).all() & finite.any()):
            raise ValueError("Invalid native generation logits")
        if request.temperature == 0:
            return int(values.cpu().argmax())
        ordered = None
        if request.temperature == 1 and 0 < request.top_k <= 64 and 4 * request.top_k < len(values):
            selected = torch.topk(values, request.top_k + 1)
            head = selected.values.cpu()
            # Ties depend on the original full CPU sort, including ties at the cutoff.
            if torch.isfinite(head[:-1]).all() and (head[:-1] > head[1:]).all():
                ordered = torch.full((len(values),), -torch.inf, dtype=torch.float32, device="cpu")
                ordered[:request.top_k] = head[:-1]
                indices = selected.indices[:request.top_k].cpu()
        if ordered is None:
            values = values.cpu() / request.temperature
            if request.top_k:
                cutoff = torch.topk(values, min(request.top_k, len(values))).values[-1]
                values = values.masked_fill(values < cutoff, -torch.inf)
            ordered, indices = torch.sort(values, descending=True)
        cumulative = torch.softmax(ordered, dim=0).cumsum(0)
        remove = cumulative > request.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        probabilities = torch.softmax(ordered.masked_fill(remove, -torch.inf), dim=0)
        # Keep the vocabulary-length population: compact multinomial changes CPU RNG draws.
        return int(indices[torch.multinomial(probabilities, 1, generator=generator)])

    @torch.inference_mode()
    def generate(self, request, cancel, request_id):
        self._check(cancel)
        self.expire()
        self._model(request.model)
        eos = set(self.backend.config.eos_token_id_list)
        if not eos or any(type(token) is not int or token < 0 for token in eos):
            raise ValueError("Native EOS configuration is unavailable")
        rng = torch.Generator(device="cpu").manual_seed(request.seed)
        branch, prefix_id = None, None
        images = []
        generated, visible, matched_stop = [], "", None
        finish = "length"
        try:
            prompt = self._generation_prompt(request, cancel, images)
            if not prompt or len(prompt) + request.max_tokens > self.lane.context_limit:
                raise ValueError("Prompt plus output budget exceeds the native context window; no truncation")
            branch = self.lane.create()
            prefill_started = self.phase_clock()
            first_sample = last_sample = None
            logits = self._append(branch, prompt, cancel, features=False, images=images)
            for index in range(request.max_tokens):
                self._check(cancel)
                token = self._sample(logits, request, rng)
                last_sample = self.phase_clock()
                if first_sample is None:
                    first_sample = last_sample
                generated.append(token)
                # EOS and stop text are accepted, even when hidden from output.
                # The final sampled token has not been forwarded to the target.
                decoded = generated[:-1] if token in eos else generated
                visible = self.backend.tokenizer.decode(torch.tensor(decoded, dtype=torch.long))
                hits = [(visible.find(stop), stop) for stop in request.stop if stop in visible]
                if hits:
                    position, matched_stop = min(hits, key=lambda pair: pair[0])
                    visible, finish = visible[:position], "stop"
                    break
                if token in eos:
                    finish = "stop"
                    break
                if index + 1 < request.max_tokens:
                    logits = self._append(branch, (token,), cancel, features=False, images=images)
            self._check(cancel)
            accepted = prompt + tuple(generated)
            prefix_id = self._retain(branch, accepted, images)
            self._check(cancel)
            return {
                "id": request_id, "model": self.model_name,
                "object": "chat.completion" if request.messages is not None else "text_completion",
                "choices": [{"index": 0, "finish_reason": finish,
                             **({"message": {"role": "assistant", "content": visible}}
                                if request.messages is not None else {"text": visible})}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(generated),
                          "total_tokens": len(accepted)},
                "phase_timing": generation_timing(len(prompt), len(generated),
                                                  prefill_started, first_sample, last_sample),
                "prefix_id": prefix_id, "prefix_identity": self.identity,
                "images": [image.receipt() for image in images],
                "accepted_token_sha256": token_hash(accepted),
                "completion_token_ids": generated, "consumed_tokens": len(branch.tokens),
                "unconsumed_token_ids": list(accepted[len(branch.tokens):]),
                "matched_stop": matched_stop,
                "stop_policy": "presentation-only trimming; all accepted tokens retained",
            }
        except BaseException:
            if prefix_id is not None:
                self.release_prefix(prefix_id)
            raise
        finally:
            try:
                if prefix_id is None:
                    self._release(branch)
            finally:
                self._release_images(images)

    @torch.inference_mode()
    def decide(self, request, cancel, request_id):
        self._check(cancel)
        self.expire()
        self._model(request.model)
        if request.head_id != self.head.head_id:
            raise ValueError("Request head does not match the loaded calibrated bundle")
        text_only(request.question)
        text_only(request.additional_context)
        candidates = [candidate.model_dump() for candidate in request.candidates]
        for candidate in candidates:
            text_only(candidate["text"])
        tails = [self.backend.encode(candidate_suffix(candidate)) for candidate in candidates]
        prefix = self._get_prefix(request.prefix_id) if request.prefix_id is not None else None
        images = list(prefix.images) if prefix else []
        query = None
        try:
            image = getattr(request, "image", None)
            additional = request.additional_context
            if prefix is None:
                messages = text_messages(request.messages)
                if image:
                    with self._acquire_image(image, cancel, images) as (binding, snapshot):
                        accepted, embeddings = self.backend.base_tokens(messages, snapshot, binding.sha256)
                        if len(embeddings) != 1 or embeddings[0] is not binding.embedding:
                            raise ValueError("Native decision image instance changed during encoding")
                else:
                    accepted, _ = self.backend.base_tokens(messages)
                reused = 0
            else:
                accepted, reused = prefix.accepted, len(prefix.branch.tokens)
                if image:
                    with self._acquire_image(image, cancel, images) as (binding, _):
                        additional = binding.embedding.text_alias + "\n" + additional
            suffix = self.backend.encode(query_suffix(request.question, candidates, additional),
                embeddings_for(images) if prefix and image else None)
            if not suffix or any(not tail for tail in tails):
                raise ValueError("Decision encoding produced an empty suffix")
            if len(accepted) + len(suffix) + max(map(len, tails)) > self.lane.context_limit:
                raise ValueError("ABC plus candidate tail exceeds the native context window; no truncation")
            self._check(cancel)
            query = self.lane.fork(prefix.branch) if prefix else self.lane.create()
            missing = accepted[reused:]
            if missing:
                self._append(query, missing, cancel, features=True, protected=request.prefix_id,
                             images=prefix.images if prefix else images)
            if tuple(query.tokens) != accepted:
                raise RuntimeError("Decision failed to resolve the exact accepted AB boundary")
            feature = self._append(query, suffix, cancel, features=True, protected=request.prefix_id, images=images)
            # Own the query before another forward can overwrite engine scratch.
            owned_query = feature.reshape(-1).detach().float().cpu().clone()
            self.lane.freeze(query)
            groups, keys, batches = defaultdict(list), [None] * len(tails), []
            for index, tail in enumerate(tails):
                groups[len(tail)].append(index)
            for indices in groups.values():
                for offset in range(0, len(indices), self.backend.candidate_batch_size):
                    batch = indices[offset:offset + self.backend.candidate_batch_size]
                    self._check(cancel)
                    self._reserve(sum(len(tails[index]) + 512 for index in batch), request.prefix_id)
                    branches = []
                    try:
                        for _ in batch:
                            branches.append(self.lane.fork(query))
                        parameters = self.backend.parameters(query.tokens, embeddings_for(images), tails[batch[0]])
                        width = len(tails[batch[0]])
                        for start in range(0, width, self.lane.chunk_size):
                            self._check(cancel)
                            chunks = [tails[index][start:start + self.lane.chunk_size] for index in batch]
                            if len(batch) == 1:
                                features = self.lane.append(branches[0], chunks[0], features=True, params=parameters)
                            else:
                                features = self.lane.append_features_many(branches, chunks, params=parameters)
                            self._check(cancel)
                        owned = features.reshape(len(batch), -1).detach().float().cpu().clone()
                        for index, value in zip(batch, owned):
                            keys[index] = value
                        batches.append(len(batch))
                    finally:
                        for branch in branches:
                            self._release(branch)
            self._check(cancel)
            logits, raw, calibrated = self.head.score(request.primitive, owned_query, torch.stack(keys))
            self._check(cancel)
            result = DecisionResponse(
                id=request_id, model=self.model_name, primitive=request.primitive,
                head_id=self.head.head_id, calibration_id=self.head.calibration_id,
                selected_id=request.candidates[max(range(len(logits)), key=logits.__getitem__)].id,
                scores=[Score(id=candidate.id, logit=logit, raw_probability=p,
                              calibrated_probability=q)
                        for candidate, logit, p, q in zip(request.candidates, logits, raw, calibrated)],
                execution=Execution(
                    mode="checkpoint" if prefix else "fresh", logical_prefix_tokens=len(accepted),
                    reused_tokens=reused, replayed_tokens=0,
                    forwarded_tokens=len(missing) + len(suffix) + sum(map(len, tails)),
                    consumed_tokens=len(query.tokens)))
            return {**result.model_dump(mode="json"), "prefix_id": request.prefix_id,
                    "prefix_identity": self.identity, "accepted_token_sha256": token_hash(accepted),
                    "query_token_sha256": token_hash(query.tokens),
                    "resolved_unconsumed_tokens": len(missing) if prefix else 0,
                    "candidate_forward_batches": batches,
                    "images": [image.receipt() for image in images],
                    "head_manifest_sha256": self.head.manifest_sha256,
                    "layout": "accepted-ab-append-c" if prefix else "standalone-training-encoding",
                    "qualification": "unqualified; CPU lifecycle tests only"}
        finally:
            try:
                self._release(query)
            finally:
                self._release_images(images)

    def close(self):
        self._owner()
        try:
            for key in list(self.prefixes):
                self.release_prefix(key)
        finally:
            self.verified_images.clear()
            self.guard.close()
            self.closed = True
