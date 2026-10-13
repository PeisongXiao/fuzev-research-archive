"""Exact, bounded CPU encodings. Each instance owns its tokenizer and caches."""

import copy
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

from .encoding import TEMPLATE, candidate_suffix, query_suffix, token_hash


PREPARATION_POLICY = {
    "version": "exact-cpu-preparation-v1",
    "context_cache_bytes": 32 << 20,
    "fragment_cache_bytes": 8 << 20,
    "render_cache_bytes": 4 << 20,
    "image_token_cache": "disabled; aliases resolved on GPU owner only",
    "tokenizer": "one private instance per preparation worker; no truncation",
}


def retained_bytes(value):
    # Count Python token integers too, not just eight-byte tuple references.
    size = sys.getsizeof(value)
    if isinstance(value, tuple):
        size += sum(retained_bytes(item) for item in value)
    return size


class ExactCache:
    def __init__(self, max_bytes, max_items):
        self.max_bytes, self.max_items = max_bytes, max_items
        self.items, self.bytes = OrderedDict(), 0
        self.hits, self.misses = 0, 0

    def get(self, key, build):
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        self.misses += 1
        value = build()
        # Allow for the mapping node and stored (value, size) pair as well.
        size = retained_bytes(key) + retained_bytes(value) + 256
        if size <= self.max_bytes and self.max_items > 0:
            while self.items and (len(self.items) >= self.max_items or
                                  self.bytes + size > self.max_bytes):
                _, (_, old_size) = self.items.popitem(last=False)
                self.bytes -= old_size
            self.items[key] = value, size
            self.bytes += size
        return value

    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "items": len(self.items), "retained_bytes": self.bytes}


@dataclass(frozen=True)
class PreparedReadout:
    # Image aliases are allocated by the GPU owner, never by the CPU worker.
    base: tuple | None
    image_template: str | None
    suffix: tuple
    tails: tuple
    label_ids: tuple | None
    query_hash: str | None


class ReadoutPreparer:
    def __init__(self, tokenizer, *, cache=True):
        self.tokenizer, self.template = tokenizer, TEMPLATE.read_text()
        self.contexts = ExactCache(PREPARATION_POLICY["context_cache_bytes"] if cache else 0, 256)
        self.fragments = ExactCache(PREPARATION_POLICY["fragment_cache_bytes"] if cache else 0, 1024)
        self.rendered = ExactCache(PREPARATION_POLICY["render_cache_bytes"] if cache else 0, 256)
        self.rows, self.seconds, self.encodes, self.encode_seconds = 0, 0., 0, 0.
        self.renders, self.render_seconds = 0, 0.

    def encode(self, text, embeddings=None):
        def build():
            started = time.perf_counter()
            value = tuple(self.tokenizer.encode(
                text, encode_special_tokens=True, embeddings=embeddings).reshape(-1).tolist())
            self.encodes += 1
            self.encode_seconds += time.perf_counter() - started
            return value
        # Image aliases are not ordinary text tokens. Retaining them would need
        # embedding-instance invalidation, not an image-content hash alone.
        return build() if embeddings else self.fragments.get(text, build)

    def render(self, messages, image=False):
        key = (image, json.dumps(messages, ensure_ascii=False, separators=(",", ":")))

        def build():
            context = copy.deepcopy(messages) or [{"role": "user", "content": "Context for the decision."}]
            if image:
                user = next((row for row in context if row["role"] == "user"), None)
                if user is None:
                    context.insert(0, {"role": "user", "content": "Consider the supplied image."})
                    user = context[0]
                if not isinstance(user["content"], str):
                    raise ValueError("Prepared image contexts must use text plus one separately indexed image")
                user["content"] = [{"type": "image"}, {"type": "text", "text": user["content"]}]
            started = time.perf_counter()
            value = self.tokenizer.hf_render_chat_template(
                context, add_generation_prompt=False, chat_template=self.template,
                enable_thinking=False, reasoning_effort="none", preserve_thinking=True)
            self.renders += 1
            self.render_seconds += time.perf_counter() - started
            return value
        return self.rendered.get(key, build)

    def base(self, messages):
        key = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))

        def build():
            tokens = self.encode(self.render(messages))
            return tokens, token_hash(tokens)
        return self.contexts.get(key, build)

    def prepare(self, row, baseline=True):
        started = time.perf_counter()
        image = bool(row.get("image_path"))
        base = None if image else self.base(row["context"])
        rendered = self.render(row["context"], image=True) if image else None
        suffix = self.encode(query_suffix(row["question"], row["candidates"]))
        tails = tuple(self.encode(candidate_suffix(candidate)) for candidate in row["candidates"])
        labels = None
        if baseline:
            labels = tuple(self.encode(" " + chr(ord("A") + index)) for index in range(len(tails)))
            if any(len(ids) != 1 for ids in labels):
                raise ValueError("Label-token control requires one-token candidate labels")
            labels = tuple(ids[0] for ids in labels)
        query_hash = token_hash(base[0] + suffix) if base is not None else None
        self.rows += 1
        self.seconds += time.perf_counter() - started
        return PreparedReadout(base, rendered, suffix, tails, labels, query_hash)

    def stats(self):
        return {"rows": self.rows, "seconds": self.seconds,
                "tokenizer_calls": self.encodes, "tokenizer_seconds": self.encode_seconds,
                "template_calls": self.renders, "template_seconds": self.render_seconds,
                "contexts": self.contexts.stats(), "fragments": self.fragments.stats(),
                "rendered": self.rendered.stats()}
