"""Synthetic still images and mocked lanes only; no native model is loaded."""

import gc
import hashlib
import re
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

torch = pytest.importorskip("torch")

from fuzev_research import native_service, service_images
from fuzev_research.encoding import candidate_suffix, query_suffix, token_hash
from fuzev_research.native import NativeBackend
from fuzev_research.native_service import NativeService
from fuzev_research.preparation import ReadoutPreparer
from fuzev_research.service_contracts import GenerationRequest, TrustedImage
from fuzev_research.service_images import ImageBinding, checked_image, embeddings_for
from test_native_service import Backend, Guard, Head, decision, generation, http_app


@dataclass
class Embedding:
    first_index: int
    mm_length: int = 7
    grid_thw: tuple = (1, 2, 2)
    mrope_merge_size: int = 1

    def __post_init__(self):
        self.last_index = self.first_index + self.mm_length
        self.text_alias = f"<$EMB_{self.first_index}$>"
        self.token_list = [100, *range(self.first_index, self.last_index), 101]
        self.full_length = len(self.token_list)


class ImageBackend(Backend):
    base_tokens = NativeBackend.base_tokens
    parameters = NativeBackend.parameters
    image_embedding = NativeBackend.image_embedding
    encode = NativeBackend.encode

    def __init__(self, mrope=False):
        super().__init__(8, 8)
        self.images, self.embedding_count, self.positions = OrderedDict(), 0, []
        self.vision = SimpleNamespace(get_image_embeddings=self.vision_encode)
        self.model = SimpleNamespace(caps={"mrope": mrope}, g_rope=SimpleNamespace(get_mrope_freqs=self.frequencies))
        self.tokenizer.encode = self.tokenize
        self.preparer = ReadoutPreparer(self.tokenizer)
        self.lane.chunk_size = 6  # Force an image to straddle native append chunks.

    def vision_encode(self, _tokenizer, image):
        assert image.mode == "RGB"
        self.embedding_count += 1
        return Embedding(1_000_000_000 + 100 * self.embedding_count)

    def render(self, messages, **kwargs):
        self.renders.append((messages, kwargs))
        template = kwargs.pop("chat_template")
        return ImmutableSandboxedEnvironment().from_string(template).render(messages=messages, **kwargs)

    @staticmethod
    def tokenize(text, encode_special_tokens, embeddings=None):
        assert encode_special_tokens
        aliases = {e.text_alias: e.token_list for e in embeddings or ()}
        tokens = []
        for part in re.split(r"(<\$EMB_\d+\$>)", text):
            tokens.extend(aliases.get(part, tuple(ord(char) + 1 for char in part)))
        return torch.tensor([tokens], dtype=torch.long)

    def frequencies(self, ids, embeddings, length):
        assert length == ids.shape[-1]
        # Inspect only IDs and object identities, never retain tensor storage.
        entry = (tuple(ids.reshape(-1).tolist()), tuple(id(e) for e in embeddings))
        self.positions.append(entry)
        return entry, length


def acquired(root, color="red"):
    root.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (4, 3), color) as image:
        digest = hashlib.sha256(b"4,3:" + image.tobytes()).hexdigest()
        path = root / (digest + ".png")
        image.save(path)
    return TrustedImage(path=str(path), sha256=digest)


@pytest.fixture(params=[False, True], ids=["linear", "native-mrope"])
def image_service(request, tmp_path, monkeypatch):
    root = tmp_path / "data/images"
    source = acquired(root)
    monkeypatch.setattr(native_service, "project_path", lambda name: tmp_path / name)
    backend = ImageBackend(mrope=request.param)
    service = NativeService(backend, "unit", Head(), {}, Guard(), admission=lambda *_: None)
    yield service, source, root
    if not service.closed:
        service.close()
    assert all(branch.closed for branch in backend.lane.branches)
    assert not backend.images


@pytest.mark.parametrize("chat", [False, True])
def test_image_generation_retains_exact_instances_after_lru_eviction(image_service, chat):
    service, source, _ = image_service
    request = (GenerationRequest(model="unit", messages=[{"role": "user", "content": "Describe."}],
                                 image=source, max_tokens=3, temperature=0)
               if chat else generation(image=source))
    result = service.generate(request, Event(), "image-a")
    prefix = service._get_prefix(result["prefix_id"])
    image = prefix.images[0]
    assert prefix.image_hashes == (source.sha256,)
    assert result["images"] == [image.receipt()]
    assert image.embedding is service.backend.images[source.sha256]
    assert all(args["indexed_embeddings"][0] is image.embedding for args in service.lane.params)
    assert all(("inv_freq" in args) == service.backend.model.caps["mrope"] for args in service.lane.params)
    prompt = prefix.accepted[:-3]
    if chat:
        assert service.backend.renders[0][1]["add_generation_prompt"]
    if service.backend.positions:
        # The first image straddles chunks, but every chunk receives the full
        # original prompt's native mRoPE layout, never a partial image grid.
        assert service.backend.positions[0][0] == prompt
        assert service.lane.params[0]["inv_freq"][0] == prompt
    original = prefix.accepted
    for index in range(9):
        # Exercise the actual bounded native image LRU, with synthetic images.
        service.backend.image_embedding(source.path, str(index))
    assert source.sha256 not in service.backend.images
    service.backend.image_embedding(source.path, source.sha256)
    replacement = service.backend.images[source.sha256]
    assert replacement is not image.embedding
    Path(source.path).unlink()  # Continuation must not reopen the original file.
    before = len(service.lane.params)
    output = service.decide(decision(messages=None, prefix_id=prefix.id), Event(), "after-lru")
    assert output["accepted_token_sha256"] == token_hash(original)
    assert all(args["indexed_embeddings"][0] is image.embedding for args in service.lane.params[before:])
    assert service._get_prefix(prefix.id).accepted == original
    service.release_prefix(prefix.id)
    # Cleanup must not remove a newer embedding with the same pixel digest.
    assert service.backend.images[source.sha256] is replacement
    service.backend.images.clear()  # Release only this test's synthetic LRU entries.


def test_standalone_image_decision_uses_canonical_encoding_and_b8(image_service):
    service, source, _ = image_service
    candidates = [{"id": str(i), "text": "option", "ordinal_value": i} for i in range(5)]
    request = decision(primitive="ordinal", image=source, candidates=candidates)
    # CPU-only native encoding method with a fake vision tensor, not a model.
    with service._acquire_image(source, Event(), []) as (_, snapshot):
        expected, embeds = service.backend.base_tokens(request.messages, snapshot, source.sha256)
    suffix = service.backend.encode(query_suffix(request.question, candidates))
    response = service.decide(request, Event(), "canonical")
    assert response["query_token_sha256"] == token_hash(expected + suffix)
    assert response["candidate_forward_batches"] == [5]
    assert response["layout"] == "standalone-training-encoding"
    assert service.lane.calls[0][0] == ()
    assert all(args["indexed_embeddings"][0] is embeds[0] for args in service.lane.params)
    assert not service.backend.images and not service.prefixes


def test_unverified_shared_cache_cannot_supply_another_images_tokens(image_service):
    service, source, root = image_service
    different = acquired(root, "blue")
    incorrect = service.backend.image_embedding(different.path, source.sha256)
    # A feature prefix may still own the older object. Neither that prefix nor
    # its embedding can be mutated while the new service verifies its source.
    owner = SimpleNamespace()
    service.backend.prefixes["feature"] = (owner, [incorrect])
    response = service.generate(generation(image=source), Event(), "verified")
    retained = service._get_prefix(response["prefix_id"])
    assert retained.images[0].embedding is not incorrect
    assert service.backend.prefixes["feature"] == (owner, [incorrect])
    assert service.backend.embedding_count == 2
    service.backend.prefixes.clear()


@pytest.mark.parametrize("prior_image", [False, True])
def test_new_image_appends_to_c_without_rerendering_ab(image_service, prior_image):
    service, source, root = image_service
    first = service.generate(generation(image=source if prior_image else None), Event(), "ab")
    prefix = service._get_prefix(first["prefix_id"])
    accepted, checkpoint = prefix.accepted, prefix.branch.tokens
    new = acquired(root, "blue")
    before, render_count = len(service.lane.params), len(service.backend.renders)
    response = service.decide(decision(messages=None, prefix_id=prefix.id, image=new,
                                      additional_context="Compare the new evidence."), Event(), "abc")
    assert response["accepted_token_sha256"] == token_hash(accepted)
    assert response["resolved_unconsumed_tokens"] == 1
    assert len(service.backend.renders) == render_count
    assert service._get_prefix(prefix.id).branch.tokens == checkpoint
    calls = service.lane.params[before:]
    assert len(calls[0].get("indexed_embeddings", [])) == int(prior_image)
    # New-image modality must not retroactively affect the pending accepted B.
    assert all(len(args["indexed_embeddings"]) == 1 + int(prior_image) for args in calls[1:])
    assert [value["rgb_sha256"] for value in response["images"]] == (
        [source.sha256, new.sha256] if prior_image else [new.sha256])
    assert new.sha256 not in service.backend.images
    if prior_image:
        assert service.backend.images[source.sha256] is prefix.images[0].embedding


@pytest.mark.parametrize("failure", ["cancel", "forward", "memory", "head", "context"])
def test_image_decision_failure_releases_new_state_but_keeps_ab(image_service, failure):
    service, source, root = image_service
    first = service.generate(generation(image=source), Event(), "ab")
    prefix = service._get_prefix(first["prefix_id"])
    new = acquired(root, "blue")
    cancel = Event()
    if failure == "cancel":
        service.lane.after_append = cancel.set
        error = InterruptedError
    elif failure == "forward":
        def fail():
            raise RuntimeError("mock forward")
        service.lane.after_append = fail
        error = RuntimeError
    elif failure == "memory":
        reserve = service._reserve
        def fail(*args):
            if new.sha256 in service.backend.images:
                raise MemoryError("mock pressure")
            reserve(*args)
        service._reserve = fail
        error = MemoryError
    elif failure == "head":
        def fail(*args):
            raise ValueError("mock head failure")
        service.head.score = fail
        error = ValueError
    else:
        service.lane.context_limit = len(prefix.accepted)
        error = ValueError
    with pytest.raises(error):
        service.decide(decision(messages=None, prefix_id=prefix.id, image=new), cancel, "fail")
    assert new.sha256 not in service.backend.images
    assert service.backend.images[source.sha256] is prefix.images[0].embedding
    assert all(branch.closed or branch is prefix.branch for branch in service.lane.branches)


@pytest.mark.parametrize("release", ["ttl", "delete", "close", "cancel"])
def test_image_references_are_bounded_and_released(image_service, release):
    service, source, _ = image_service
    weak = []
    encode = service.backend.vision.get_image_embeddings
    def observed(*args):
        result = encode(*args)
        weak.append(weakref.ref(result))
        return result
    service.backend.vision.get_image_embeddings = observed
    if release == "cancel":
        cancel = Event()
        service.lane.after_append = cancel.set
        with pytest.raises(InterruptedError):
            service.generate(generation(image=source), cancel, "cancel")
    else:
        response = service.generate(generation(image=source), Event(), "ab")
        if release == "ttl":
            service.clock = lambda: float("inf")
            service.expire()
        elif release == "delete":
            service.release_prefix(response["prefix_id"])
        else:
            service.close()
    assert not service.prefixes and not service.backend.images
    service.lane.params.clear()  # Synthetic call recorder is not runtime storage.
    gc.collect()
    assert weak and all(ref() is None for ref in weak)


@pytest.mark.parametrize("field", ["tokens", "length", "layout"])
def test_retained_image_mutation_is_not_retokenized_or_reused(image_service, field):
    service, source, _ = image_service
    response = service.generate(generation(image=source), Event(), "ab")
    prefix = service._get_prefix(response["prefix_id"])
    embedding = prefix.images[0].embedding
    if field == "tokens":
        embedding.token_list[1] += 1
    elif field == "length":
        embedding.mm_length += 1
    else:
        embedding.grid_thw = (1, 3, 2)
    before = len(service.lane.calls)
    with pytest.raises(ValueError, match="image embedding identity changed"):
        service.decide(decision(messages=None, prefix_id=prefix.id), Event(), "bad")
    assert len(service.lane.calls) == before
    assert prefix.branch.closed and not service.prefixes and not service.backend.images


def test_distinct_embedding_instances_cannot_alias_the_same_dynamic_tokens():
    first = ImageBinding.capture("0" * 64, Embedding(1_000_000_000))
    other = ImageBinding.capture("1" * 64, Embedding(1_000_000_003))
    assert embeddings_for([first, first]) == [first.embedding]
    with pytest.raises(ValueError, match="Conflicting"):
        embeddings_for([first, other])


def test_image_prefix_capacity_and_shared_reference_cleanup(image_service):
    service, source, root = image_service
    service.capacity = 2
    first = service.generate(generation(image=source), Event(), "one")["prefix_id"]
    first_branch = service._get_prefix(first).branch
    service.lane.output = [91, 92, 99]
    second = service.generate(generation(image=source), Event(), "two")["prefix_id"]
    embedding = service._get_prefix(second).images[0].embedding
    assert service.backend.embedding_count == 1
    service.lane.output = [91, 92, 99]
    service.generate(generation(image=acquired(root, "blue")), Event(), "three")
    assert first_branch.closed and first not in service.prefixes
    assert service.backend.images[source.sha256] is embedding
    service.release_prefix(second)
    assert source.sha256 not in service.backend.images


def test_cancel_at_vision_boundary_cleans_embedding_without_disabling_tower(image_service):
    service, source, _ = image_service
    cancel = Event()
    tower, encode = service.backend.vision, service.backend.vision.get_image_embeddings
    def cancel_after_encoding(*args):
        embedding = encode(*args)
        cancel.set()
        return embedding
    tower.get_image_embeddings = cancel_after_encoding
    with pytest.raises(InterruptedError):
        service.generate(generation(image=source), cancel, "cancel-vision")
    assert not service.prefixes and not service.backend.images and not service.lane.calls
    assert service.backend.vision is tower


def test_http_image_contract_and_feature_boundary_are_local_mocked_only(http_app, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fuzev_research import server

    app, _, _ = http_app
    backend = ImageBackend()
    backend.close = lambda: None
    feature_inputs = []
    def features(row, baseline):
        feature_inputs.append(row["image_path"])
        assert not isinstance(row["image_path"], str)  # Verified bytes, not a reopened pathname.
        with Image.open(row["image_path"]) as image:
            assert image.getpixel((0, 0)) == (255, 0, 0)
        return torch.zeros(2560), torch.zeros(2, 2560), None, {}
    backend.extract = features
    monkeypatch.setattr(server, "NativeBackend", lambda *_, **__: backend)
    monkeypatch.setattr(server, "admit", lambda *_: None)
    monkeypatch.setattr(native_service, "project_path", lambda name: tmp_path / name)
    source = acquired(tmp_path / "data/images")
    with TestClient(app) as client:
        response = client.post("/v1/completions", json=generation(image=source).model_dump())
        assert response.status_code == 200, response.text
        prefix = response.json()["prefix_id"]
        response = client.post("/v1/decisions", json=decision(
            messages=None, prefix_id=prefix, image=source).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.json()["images"][0] == response.json()["images"][1]
        row = {"primitive": "choice", "question": "Which?", "context": [],
               "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
               "image_path": source.path, "image_sha256": source.sha256}
        response = client.post("/v1/features", json={"row": row})
        assert response.status_code == 200, response.text
        assert feature_inputs[0].closed
        assert client.post("/v1/features", json={"row": {**row, "image_sha256": "0" * 64}}).status_code == 400
        assert len(feature_inputs) == 1
        assert client.post("/v1/completions", json={**generation().model_dump(), "image": {
            "path": "https://example.com/test.png", "sha256": source.sha256}}).status_code == 400
        assert client.post("/v1/completions", json={**generation().model_dump(), "image": {
            "path": source.path, "sha256": source.sha256, "base64": "not accepted"}}).status_code == 422


def test_bounded_png_validation_and_immutable_snapshot(tmp_path):
    root = tmp_path / "images"
    source = acquired(root)
    with checked_image(source, root) as snapshot:
        assert not snapshot.closed
        # A changed pathname after validation must not change the decoded bytes.
        with Image.new("RGB", (4, 3), "blue") as changed:
            changed.save(source.path)
        with Image.open(snapshot) as verified:
            assert verified.getpixel((0, 0)) == (255, 0, 0)
    assert snapshot.closed
    with pytest.raises(ValueError, match="checksum"):
        with checked_image(source, root):
            pytest.fail("changed pixels were accepted")


@pytest.mark.parametrize("kind", ["outside", "relative", "url", "traversal", "symlink", "fifo",
                                 "bad-hash", "large-bytes", "large-pixels", "animated", "not-png"])
def test_untrusted_or_unbounded_images_fail_before_native_encoding(tmp_path, monkeypatch, kind):
    root = tmp_path / "images"
    source = acquired(root)
    import os
    path = Path(source.path)
    if kind == "outside":
        source = acquired(tmp_path / "outside")
    elif kind == "relative":
        source = source.model_copy(update={"path": path.name})
    elif kind == "url":
        source = source.model_copy(update={"path": "https://example.com/" + path.name})
    elif kind == "traversal":
        source = source.model_copy(update={"path": str(root / ".." / "images" / path.name)})
    elif kind in {"symlink", "fifo"}:
        path.unlink()
        if kind == "symlink":
            path.symlink_to(acquired(tmp_path / "outside").path)
        else:
            os.mkfifo(path)
    elif kind == "bad-hash":
        source = source.model_copy(update={"sha256": "0" * 64})
    elif kind == "large-bytes":
        monkeypatch.setattr(service_images, "MAX_IMAGE_BYTES", 1)
    elif kind == "large-pixels":
        monkeypatch.setattr(service_images, "MAX_IMAGE_PIXELS", 1)
    elif kind == "animated":
        with Image.new("RGB", (4, 3), "red") as a, Image.new("RGB", (4, 3), "blue") as b:
            a.save(path, save_all=True, append_images=[b])
    else:
        path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        with checked_image(source, root):
            pytest.fail("invalid image was accepted")
