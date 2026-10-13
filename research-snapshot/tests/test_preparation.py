import copy
import json

import pytest

from fuzev_research.encoding import candidate_suffix, query_suffix, token_hash
from fuzev_research.preparation import ExactCache, ReadoutPreparer


class Tokens:
    def __init__(self, values):
        self.values = values
    def reshape(self, *_):
        return self
    def tolist(self):
        return list(self.values)


class Tokenizer:
    def __init__(self):
        self.calls, self.templates = [], []
    def encode(self, text, encode_special_tokens, embeddings):
        assert encode_special_tokens
        self.calls.append(text)
        if len(text) == 2 and text[0] == " ":
            return Tokens([ord(text[1])])
        if embeddings:
            return Tokens([*map(ord, text), embeddings[0].token])
        return Tokens(map(ord, text))
    def hf_render_chat_template(self, messages, **kwargs):
        self.templates.append((copy.deepcopy(messages), kwargs))
        if any(isinstance(row["content"], list) for row in messages):
            return "<|vision_start|><|image_pad|><|vision_end|>text"
        return json.dumps(messages, ensure_ascii=False)


def row(context="Exact context", question="Which?", texts=("Yes", "No")):
    return {"context": [{"role": "user", "content": context}], "question": question,
            "candidates": [{"id": str(i), "text": text} for i, text in enumerate(texts)]}


def test_memoized_full_tokens_and_hashes_equal_uncached_encoding():
    cached, uncached = ReadoutPreparer(Tokenizer()), ReadoutPreparer(Tokenizer(), cache=False)
    rows = [row(), row(question="Other?"), row(context="different"), row()]
    for item in rows:
        result = cached.prepare(item)
        assert result == uncached.prepare(item)
        assert result.base[1] == token_hash(result.base[0])
        assert result.query_hash == token_hash(result.base[0] + result.suffix)
        assert result.label_ids == (ord("A"), ord("B"))
        assert len(result.tails) == 2
    assert cached.stats()["contexts"]["hits"] == 2
    assert cached.stats()["tokenizer_calls"] < uncached.stats()["tokenizer_calls"]
    # Complete BPE boundaries and special strings, never subword-spliced fragments.
    assert query_suffix(rows[0]["question"], rows[0]["candidates"]) in cached.tokenizer.calls
    assert candidate_suffix(rows[0]["candidates"][0]) in cached.tokenizer.calls
    assert all(values[1]["enable_thinking"] is False for values in cached.tokenizer.templates)


def test_exact_context_keys_follow_mutated_inputs_and_private_tokenizers():
    first, second = ReadoutPreparer(Tokenizer()), ReadoutPreparer(Tokenizer())
    item = row()
    original = first.prepare(item)
    item["context"][0]["content"] = "mutated"
    changed = first.prepare(item)
    assert changed.base != original.base
    assert second.prepare(row()) == original
    assert second.stats()["contexts"]["hits"] == 0
    with pytest.raises(TypeError):
        original.base[0][0] = 0


def test_image_rendering_never_memoizes_dynamic_embedding_tokens():
    preparer = ReadoutPreparer(Tokenizer())
    item = {**row(), "image_path": "image.png", "image_sha256": "same-image"}
    context = copy.deepcopy(item["context"])
    prepared = preparer.prepare(item)
    assert prepared.base is None and prepared.query_hash is None
    assert item["context"] == context
    assert prepared.image_template.count("<|vision_start|><|image_pad|><|vision_end|>") == 1
    first = type("Embedding", (), {"token": 100})()
    second = type("Embedding", (), {"token": 200})()
    assert preparer.encode("same-alias", [first])[-1] == 100
    assert preparer.encode("same-alias", [second])[-1] == 200
    assert preparer.prepare(item) == prepared
    assert preparer.stats()["rendered"]["hits"] == 1


def test_cache_byte_bound_eviction_and_oversized_passthrough():
    cache = ExactCache(1200, 2)
    for index in range(40):
        cache.get(str(index), lambda: tuple(range(10)))
        assert cache.bytes <= 1200
        assert len(cache.items) <= 2
    before = cache.bytes
    value = tuple(range(1000))
    assert cache.get("oversized", lambda: value) is value
    assert "oversized" not in cache.items and cache.bytes == before
    before = cache.misses
    cache.get("0", lambda: (1,))
    assert cache.misses == before + 1


def test_bad_label_is_not_silently_split_or_truncated():
    tokenizer = Tokenizer()
    encode = tokenizer.encode
    tokenizer.encode = lambda text, **kwargs: Tokens([1, 2]) if text == " A" else encode(text, **kwargs)
    with pytest.raises(ValueError, match="one-token"):
        ReadoutPreparer(tokenizer).prepare(row())
