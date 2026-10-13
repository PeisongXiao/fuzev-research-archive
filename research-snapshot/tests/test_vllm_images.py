"""Synthetic CPU-only image contracts; no models, dataset images, or predictions."""

import asyncio
import gc
import hashlib
import threading
import weakref
from collections import UserDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from fuzev_research.encoding import candidate_suffix, query_suffix, token_hash
from fuzev_research.service_contracts import ServiceDecisionRequest, TrustedImage
from fuzev_research.vllm_images import PLACEHOLDER, _bytes, _structure
from fuzev_research.vllm_server import create_app
from fuzev_research.vllm_service import CONTEXT, VllmService
from test_vllm_service import API, Engine, Guard, Head, Tokenizer, config, decision, generate, run, until


@dataclass(frozen=True)
class Position:
    offset: int
    length: int
    is_embed: object = None


@dataclass(frozen=True)
class Field:
    dim: int = 0
    keep_on_cpu: bool = True
    slices: tuple = (slice(0, 4),)


@dataclass
class Elem:
    data: object
    field: Field = Field()


class Item(UserDict):
    pass


def test_tensor_slice_endpoints_are_owned_hashed_and_accounted():
    start, stop = torch.tensor(0), torch.tensor(4)
    original = Field(slices=(slice(start, stop),))
    owned = _structure(original, clone=True)
    signature = _structure(owned)
    digest = hashlib.sha256()
    _structure(owned, digest=digest)
    assert owned.slices[0].start.data_ptr() != start.data_ptr()
    assert owned.slices[0].stop.dtype == stop.dtype
    assert _bytes(original.slices[0]) > _bytes(slice(0, 4))
    stop.fill_(2)
    assert _structure(owned) == signature
    owned.slices[0].stop.fill_(3)
    changed = hashlib.sha256()
    _structure(owned, digest=changed)
    assert _structure(owned) != signature and changed.digest() != digest.digest()


def test_tensor_slice_endpoint_cannot_bypass_cpu_payload_boundary():
    with pytest.raises(ValueError, match="CPU processor payloads"):
        _structure(slice(torch.tensor(0, device="meta"), 4), clone=True)
    with pytest.raises(ValueError, match="CPU processor payloads"):
        _bytes(slice(0, torch.tensor(4, device="meta")))


class ImageTokenizer(Tokenizer):
    markers = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
    marker_ids = (150001, 150002, 150003)

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        self.calls.append(text)
        tokens = []
        while text:
            for marker, token in zip(self.markers, self.marker_ids):
                if text.startswith(marker):
                    tokens.append(token)
                    text = text[len(marker):]
                    break
            else:
                tokens.append(ord(text[0]) + 1)
                text = text[1:]
        return tokens

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False and kwargs["enable_thinking"] is False
        assert kwargs["reasoning_effort"] == "none" and kwargs["preserve_thinking"] is True
        self.renders.append((messages, kwargs))

        def content(value):
            if isinstance(value, str):
                return value
            return "".join(PLACEHOLDER if part["type"] == "image" else part["text"] for part in value)

        return "[context]" + "|".join(content(x["content"]) for x in messages) + (
            "[assistant]" if kwargs["add_generation_prompt"] else "")


class Renderer:
    """API-shaped fake. Hashes/grid/expansion below are NOT real HF preprocessing."""

    def __init__(self, limit=2, width=4):
        self.mm_processor = NS(info=NS(allowed_mm_limits={"image": limit}))
        self.calls, self.outputs, self.pil_images = [], [], []
        self.width = width
        self.patch = lambda _output: None
        self.wait = None
        self.entered = asyncio.Event()

    async def process_for_engine_async(self, prompt, arrival_time, *, skip_mm_cache=False):
        assert skip_mm_cache is True and isinstance(arrival_time, float)
        assert prompt["mm_processor_kwargs"] == {"truncation": False}
        assert set(prompt["multi_modal_data"]) == {"image"}
        image, = prompt["multi_modal_data"]["image"]
        assert image.mode == "RGB"
        self.pil_images.append(image)
        raw = tuple(prompt["prompt_token_ids"])
        self.calls.append((raw, prompt["multi_modal_uuids"]["image"][0], image.getpixel((0, 0))))
        self.entered.set()
        if self.wait is not None:
            await self.wait.wait()
        pad = raw.index(150002)
        digest = hashlib.sha256(self.calls[-1][1].encode()).hexdigest()
        # CPU synthetic data mimics the inspected MultiModalFieldElem layout.
        result = {"type": "multimodal", "prompt_token_ids": list(
            raw[:pad] + (150002,) * self.width + raw[pad + 1:]),
            "mm_kwargs": {"image": [Item({
                "pixel_values": Elem(torch.full((4, 3), float(image.getpixel((0, 0))[0]))),
                "image_grid_thw": Elem(torch.tensor([1, 4, 4], dtype=torch.int64)),
            })]},
            "mm_hashes": {"image": [digest]},
            "mm_placeholders": {"image": [Position(pad, self.width,
                torch.ones(self.width, dtype=torch.bool))]}}
        self.patch(result)
        self.outputs.append(result)
        return result


def acquired(root, color=(10, 20, 30), size=(8, 8)):
    root.mkdir(exist_ok=True)
    with Image.new("RGB", size, color) as image:
        digest = hashlib.sha256(f"{size[0]},{size[1]}:".encode() + image.tobytes()).hexdigest()
        path = root / (digest + ".png")
        image.save(path)
    return TrustedImage(path=str(path), sha256=digest)


def image_service(root, *, limit=2, width=4, **kwargs):
    engine = Engine()
    engine.tokenizer, engine.renderer = ImageTokenizer(), Renderer(limit, width)
    cfg = config(**kwargs)
    admissions = []
    svc = VllmService(engine, engine.tokenizer, Head(), cfg, {"cell": "synthetic-image-cell"},
                       Guard(), api=API(), admission=lambda *args: admissions.append(args))
    svc.images.root = root
    svc.admissions = admissions
    return svc


def image_decision(source=None, **kwargs):
    return ServiceDecisionRequest(**(decision(**kwargs).model_dump() | {"image": source}))


def assert_closed(image):
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))


def test_fresh_generation_owns_expanded_ids_fields_cache_identity_and_eos(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        source = acquired(tmp_path)
        result = await svc.generate(generate(image=source), "fresh")
        raw, uuid, color = svc.engine.renderer.calls[0]
        pad = raw.index(150002)
        expected = raw[:pad] + (150002,) * 4 + raw[pad + 1:]
        assert tuple(result["prompt_token_ids"]) == expected
        assert tuple(result["completion_token_ids"]) == (700, 701, 0)
        prefix = svc.prefixes[result["prefix_id"]][0]
        assert prefix.accepted == expected + (700, 701, 0)
        assert result["accepted_token_sha256"] == token_hash(prefix.accepted)
        assert uuid == "fuzev-rgb:" + svc.identity + ":" + source.sha256
        assert color == (10, 20, 30)
        assert len(result["images"]) == 1
        receipt = result["images"][0]
        assert receipt["offset"] == pad and receipt["length"] == 4
        assert receipt["rgb_sha256"] == source.sha256
        assert receipt["mm_hash"] == svc.engine.inputs[0]["mm_hashes"]["image"][0]
        assert receipt["is_embed"]["length"] == 4
        assert svc.retained_multimodal_bytes == prefix.multimodal.nbytes > 0
        original = svc.engine.renderer.outputs[0]["mm_kwargs"]["image"][0]
        owned = prefix.multimodal.images[0].item
        sent = svc.engine.inputs[0]["mm_kwargs"]["image"][0]
        assert isinstance(owned, Item) and isinstance(sent, Item)
        assert original["pixel_values"].data.data_ptr() != owned["pixel_values"].data.data_ptr()
        assert sent["pixel_values"].field == owned["pixel_values"].field
        assert any(size >= 512 << 20 for size, *_ in svc.admissions)
        assert_closed(svc.engine.renderer.pil_images[0])
        await svc.close()
    run(case())


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "System"}],
    [{"role": "system", "content": "System"}, {"role": "user", "content": "First"},
     {"role": "user", "content": "Second"}]])
@pytest.mark.parametrize("generation", [True, False])
def test_image_context_placement_matches_native_portable_policy(tmp_path, messages, generation):
    async def case():
        svc = image_service(tmp_path)
        source = acquired(tmp_path)
        if generation and not messages:  # Generation contract requires a message.
            chosen = [{"role": "user", "content": "Context for the decision."}]
        else:
            chosen = messages
        if generation:
            await svc.generate(generate(prompt=None, messages=chosen, image=source), "chat")
        else:
            await svc.decide(image_decision(source, messages=chosen), "decision")
        rendered, flags = svc.tokenizer.renders[0]
        users = [value for value in rendered if value["role"] == "user"]
        first = users[0]["content"]
        assert first[0] == {"type": "image"}
        text = ("Context for the decision." if not messages else
                "Consider the supplied image." if not any(x["role"] == "user" for x in messages)
                else "First")
        assert first[1] == {"type": "text", "text": text}
        assert all(isinstance(value["content"], str) for value in users[1:])
        assert flags["add_generation_prompt"] is generation
        await svc.close()
    run(case())


@pytest.mark.parametrize("primitive", ["choice", "noul", "ordinal"])
def test_standalone_images_use_same_engine_typed_head_and_independent_branches(tmp_path, primitive):
    async def case():
        svc = image_service(tmp_path)
        svc.engine.mutate_after_yield = True
        candidates = [{"id": "yes", "text": "Yes" if primitive == "noul" else "First", "ordinal_value": 0},
                      {"id": "no", "text": "No" if primitive == "noul" else "Second", "ordinal_value": 1}]
        request = image_decision(acquired(tmp_path), primitive=primitive, candidates=candidates)
        result = await svc.decide(request, "standalone")
        assert result["primitive"] == primitive and result["execution"]["mode"] == "fresh"
        assert result["qualification"].startswith("unqualified")
        assert all(call[0] == "encode" for call in svc.engine.calls)
        query = svc.engine.calls[0][1]
        for call, candidate in zip(svc.engine.calls[1:], candidates):
            assert call[1] == query + tuple(svc.tokenizer.encode(candidate_suffix(candidate),
                add_special_tokens=False, truncation=False))
        assert all(torch.all(input["mm_kwargs"]["image"][0]["pixel_values"].data == 10)
                   for input in svc.engine.inputs)
        assert not svc.prefixes and svc.retained_multimodal_bytes == 0
        assert len(svc.engine.renderer.calls) == 1
        await svc.close()
    run(case())


def test_retained_ab_survives_source_delete_processor_eviction_and_driver_mutation(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        source = acquired(tmp_path)
        svc.engine.mutate_after_yield = True
        result = await svc.generate(generate(image=source), "A")
        prefix = svc.prefixes[result["prefix_id"]][0]
        (tmp_path / (source.sha256 + ".png")).unlink()
        svc.engine.renderer.outputs[0]["mm_kwargs"]["image"][0]["pixel_values"].data.fill_(77)
        svc.engine.renderer.outputs.clear()
        svc.tokenizer.renders.clear()
        request = image_decision(messages=None, prefix_id=prefix.id, additional_context="C")
        scored = await svc.decide(request, "C")
        candidates = [x.model_dump() for x in request.candidates]
        expected = prefix.accepted + tuple(svc.tokenizer.encode(
            query_suffix(request.question, candidates, "C"), add_special_tokens=False, truncation=False))
        assert svc.engine.calls[1][1] == expected
        assert expected[:len(prefix.accepted)][-1] == 0
        assert not svc.tokenizer.renders and len(svc.engine.renderer.calls) == 1
        assert scored["images"] == result["images"]
        assert all(torch.all(input["mm_kwargs"]["image"][0]["pixel_values"].data == 10)
                   for input in svc.engine.inputs)
        assert all(call[1][:len(prefix.accepted)] == prefix.accepted for call in svc.engine.calls[1:])
        assert all(value["images"] == scored["images"] for value in scored["execution"]["requests"])
        await svc.close()
    run(case())


@pytest.mark.parametrize("old_image", [False, True])
@pytest.mark.parametrize("same_image", [False, True])
def test_new_image_c_expands_only_suffix_and_shifts_ranges_without_rerendering_ab(tmp_path, old_image, same_image):
    async def case():
        svc = image_service(tmp_path)
        a = acquired(tmp_path)
        c = a if same_image else acquired(tmp_path, (40, 50, 60))
        generated = await svc.generate(generate(image=a if old_image else None), "A")
        prefix = svc.prefixes[generated["prefix_id"]][0]
        calls_before, renders_before = len(svc.engine.renderer.calls), len(svc.tokenizer.renders)
        request = image_decision(c, messages=None, prefix_id=prefix.id, additional_context="look")
        result = await svc.decide(request, "C")
        assert len(svc.engine.renderer.calls) == calls_before + 1
        assert len(svc.tokenizer.renders) == renders_before
        raw = svc.engine.renderer.calls[-1][0]
        expected_raw = tuple(svc.tokenizer.encode(query_suffix(request.question,
            [x.model_dump() for x in request.candidates], PLACEHOLDER + "\nlook"),
            add_special_tokens=False, truncation=False))
        assert raw == expected_raw
        pad = raw.index(150002)
        suffix = raw[:pad] + (150002,) * 4 + raw[pad + 1:]
        assert svc.engine.calls[1][1] == prefix.accepted + suffix
        assert result["images"][-1]["offset"] == len(prefix.accepted) + pad
        assert len(result["images"]) == 1 + old_image
        assert result["images"][-1]["rgb_sha256"] == c.sha256
        if old_image:
            assert result["images"][0] == generated["images"][0]
            if same_image:
                assert result["images"][0]["mm_hash"] == result["images"][1]["mm_hash"]
        assert all(input["prompt_token_ids"][:len(prefix.accepted)] == list(prefix.accepted)
                   for input in svc.engine.inputs[1:])
        assert len(prefix.multimodal.images) == 1 if old_image else prefix.multimodal is None
        await svc.close()
    run(case())


def test_new_c_respects_actual_image_limit_before_loading_source(tmp_path):
    async def case():
        svc = image_service(tmp_path, limit=1)
        source = acquired(tmp_path)
        result = await svc.generate(generate(image=source), "A")
        (tmp_path / (source.sha256 + ".png")).unlink()
        with pytest.raises(ValueError, match="image limit"):
            await svc.decide(image_decision(source, messages=None, prefix_id=result["prefix_id"]), "C")
        assert len(svc.engine.renderer.calls) == 1 and len(svc.engine.calls) == 1
        await svc.close()
    run(case())


@pytest.mark.parametrize("fault", ["cache-reference", "missing-hash", "extra-modality", "wrong-offset",
    "wrong-mask", "wrong-grid", "missing-pixels", "changed-text", "native-alias", "over-context"])
def test_invalid_processed_inputs_fail_explicitly_before_engine_admission(tmp_path, fault):
    def patch(output):
        item = output["mm_kwargs"]["image"][0]
        position = output["mm_placeholders"]["image"][0]
        if fault == "cache-reference":
            output["mm_kwargs"]["image"] = [None]
        elif fault == "missing-hash":
            output.pop("mm_hashes")
        elif fault == "extra-modality":
            output["mm_kwargs"]["video"] = []
        elif fault == "wrong-offset":
            output["mm_placeholders"]["image"][0] = replace(position, offset=0)
        elif fault == "wrong-mask":
            output["mm_placeholders"]["image"][0] = replace(position, is_embed=torch.ones(2))
        elif fault == "wrong-grid":
            item["image_grid_thw"].data[0] = 2
        elif fault == "missing-pixels":
            item.pop("pixel_values")
        elif fault == "changed-text":
            output["prompt_token_ids"][-1] = 999
        elif fault == "native-alias":
            output["prompt_token_ids"][position.offset] = 1_000_000_000
        elif fault == "over-context":
            output["prompt_token_ids"] = [150002] * (CONTEXT + 1)

    async def case():
        svc = image_service(tmp_path)
        svc.engine.renderer.patch = patch
        with pytest.raises(ValueError):
            await svc.generate(generate(image=acquired(tmp_path)), "bad")
        assert not svc.engine.calls and not svc.operations and not svc.prefixes
        assert_closed(svc.engine.renderer.pil_images[0])
        await svc.close()
    run(case())


def test_suffix_processor_rewrite_is_explicit_unsupported_without_touching_ab(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        generated = await svc.generate(generate(), "A")
        prefix = svc.prefixes[generated["prefix_id"]][0]
        svc.engine.renderer.patch = lambda output: output["prompt_token_ids"].append(999)
        with pytest.raises(ValueError, match="exact append is unsupported"):
            await svc.decide(image_decision(acquired(tmp_path), messages=None, prefix_id=prefix.id), "C")
        assert len(svc.engine.calls) == 1 and svc.prefixes[prefix.id][0] is prefix
        await svc.close()
    run(case())


@pytest.mark.parametrize("mutation", ["pixels", "grid", "mask", "field", "tokens"])
def test_retained_payload_mutation_is_detected(tmp_path, mutation):
    async def case():
        svc = image_service(tmp_path)
        result = await svc.generate(generate(image=acquired(tmp_path)), "A")
        prefix = svc.prefixes[result["prefix_id"]][0]
        binding = prefix.multimodal.images[0]
        if mutation == "pixels":
            binding.item["pixel_values"].data.fill_(0)
        elif mutation == "grid":
            binding.item["image_grid_thw"].data[1] = 6
        elif mutation == "mask":
            binding.placeholder.is_embed[0] = False
        elif mutation == "field":
            binding.item["pixel_values"].field = Field(dim=1)
        else:
            changed = list(prefix.prompt)
            changed[binding.placeholder.offset] = 99
            replacement = replace(prefix, prompt=tuple(changed), accepted=tuple(changed) + prefix.completion)
            svc.prefixes[prefix.id] = (replacement, svc.clock())
        with pytest.raises(ValueError, match="changed|range"):
            await svc.decide(image_decision(messages=None, prefix_id=prefix.id), "C")
        assert len(svc.engine.calls) == 1
        await svc.close()
    run(case())


@pytest.mark.parametrize("bad", ["digest", "symlink", "subdirectory", "relative", "url", "base64", "jpeg"])
def test_no_arbitrary_image_sources_are_loaded(tmp_path, bad):
    async def case():
        svc = image_service(tmp_path)
        source = acquired(tmp_path)
        path = tmp_path / (source.sha256 + ".png")
        if bad == "digest":
            with Image.new("RGB", (8, 8), (90, 80, 70)) as image:
                image.save(path)
        elif bad == "symlink":
            other = tmp_path / "other.png"
            path.rename(other)
            path.symlink_to(other)
        elif bad == "subdirectory":
            nested = tmp_path / "nested"
            nested.mkdir()
            path.rename(nested / path.name)
            source = source.model_copy(update={"path": str(nested / path.name)})
        elif bad in ("relative", "url", "base64"):
            value = {"relative": path.name, "url": "https://example.invalid/image.png",
                     "base64": "data:image/png;base64,eA=="}[bad]
            source = TrustedImage(path=value, sha256=source.sha256)
        elif bad == "jpeg":
            with Image.new("RGB", (8, 8)) as image:
                image.save(path, format="JPEG")
        with pytest.raises(ValueError):
            await svc.generate(generate(image=source), "bad")
        assert not svc.engine.calls and not svc.engine.renderer.calls
        assert not svc.operations
        await svc.close()
    run(case())


@pytest.mark.parametrize("release", ["delete", "ttl", "count", "bytes", "close"])
def test_retained_image_lifetime_is_resource_bounded(tmp_path, release):
    async def case():
        svc = image_service(tmp_path, prefix_capacity=1 if release == "count" else 8,
                            retained_multimodal_bytes=1 << 20)
        if release == "bytes":
            svc.engine.renderer.patch = lambda output: setattr(
                output["mm_kwargs"]["image"][0]["pixel_values"], "data", torch.ones(100_000))
        result = await svc.generate(generate(image=acquired(tmp_path)), "A")
        prefix = svc.prefixes[result["prefix_id"]][0]
        ref = weakref.ref(prefix.multimodal)
        size = prefix.multimodal.nbytes
        del prefix
        if release == "delete":
            assert svc.release_prefix(result["prefix_id"])
        elif release == "ttl":
            svc.clock = lambda: 1e15
            svc.expire()
        elif release in ("count", "bytes"):
            for i in range(2 if release == "bytes" else 1):
                await svc.generate(generate(image=acquired(tmp_path)), f"other{i}")
        else:
            await svc.close()
        assert result["prefix_id"] not in svc.prefixes
        assert svc.retained_multimodal_bytes == sum(x[0].multimodal.nbytes for x in svc.prefixes.values())
        assert svc.retained_multimodal_bytes <= 1 << 20
        gc.collect()
        assert ref() is None
        assert size > 0
        await svc.close()
    run(case())


def test_payload_and_retained_byte_limits_are_independent(tmp_path):
    async def case():
        svc = image_service(tmp_path, retained_multimodal_bytes=1 << 20)
        svc.images.max_payload_bytes = 100
        with pytest.raises(MemoryError, match="owned byte bound"):
            await svc.generate(generate(image=acquired(tmp_path)), "small")
        assert not svc.engine.calls
        svc.images.max_payload_bytes = 512 << 20
        svc.engine.renderer.patch = lambda output: setattr(
            output["mm_kwargs"]["image"][0]["pixel_values"], "data", torch.ones(300_000))
        with pytest.raises(MemoryError, match="byte pool"):
            await svc.generate(generate(image=acquired(tmp_path)), "pool")
        assert len(svc.engine.calls) == 1 and not svc.prefixes and svc.retained_multimodal_bytes == 0
        await svc.close()
    run(case())


@pytest.mark.parametrize("failure", ["cancel", "engine", "headroom"])
def test_image_engine_failure_or_cancel_does_not_leak_retained_storage(tmp_path, failure):
    async def case():
        svc = image_service(tmp_path)
        svc.engine.wait = asyncio.Event()
        task = asyncio.create_task(svc.generate(generate(image=acquired(tmp_path)), "A"))
        await svc.engine.entered.wait()
        if failure == "cancel":
            assert svc.cancel("A")
            expected = asyncio.CancelledError
        elif failure == "engine":
            svc.engine.fail_call = 1
            expected = RuntimeError
        else:
            svc.guard.error = MemoryError("synthetic reserve loss")
            expected = MemoryError
        svc.engine.wait.set()
        with pytest.raises(expected):
            await task
        assert not svc.prefixes and svc.retained_multimodal_bytes == 0
        assert not svc.operations and not svc.engine_requests and svc.engine.active == 0
        assert len(svc.engine.aborted) == 1
        assert_closed(svc.engine.renderer.pil_images[0])
        await svc.close()
    run(case())


def test_preprocessing_cancel_is_drained_and_holds_admission_until_pil_closed(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        svc.engine.renderer.wait = asyncio.Event()
        source = acquired(tmp_path)
        tasks = [asyncio.create_task(svc.generate(generate(image=source), f"g{i}")) for i in range(4)]
        await svc.engine.renderer.entered.wait()
        assert len(svc.operations) == 4
        svc.cancel("g0")
        tasks[0].cancel()
        await asyncio.sleep(.01)
        assert not tasks[0].done() and len(svc.operations) == 4 and not svc.engine.calls
        svc.engine.renderer.wait.set()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        await asyncio.gather(*tasks[1:])
        assert len(svc.engine.renderer.calls) == 4 and len(svc.engine.calls) == 3
        assert all(image for image in svc.engine.renderer.pil_images)
        for image in svc.engine.renderer.pil_images:
            assert_closed(image)
        assert not svc.operations
        await svc.close()
    run(case())


def test_cancel_during_cpu_decode_cannot_detach_thread_or_pil_snapshot(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        source = acquired(tmp_path)
        entered, release = threading.Event(), threading.Event()
        load = svc.images._load

        def slow_load(value):
            entered.set()
            assert release.wait(3)
            return load(value)

        svc.images._load = slow_load
        task = asyncio.create_task(svc.generate(generate(image=source), "thread"))
        await until(entered.is_set)
        task.cancel()
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done() and "thread" in svc.operations
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not svc.operations and not svc.engine.calls
        assert_closed(svc.engine.renderer.pil_images[0])
        await svc.close()
    run(case())


def test_guard_failure_drains_preprocessing_worker_before_returning(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        svc.engine.renderer.wait = asyncio.Event()
        task = asyncio.create_task(svc.generate(generate(image=acquired(tmp_path)), "guard"))
        await svc.engine.renderer.entered.wait()
        svc.guard.error = MemoryError("reserve")
        await asyncio.sleep(.01)
        assert not task.done() and "guard" in svc.operations
        svc.engine.renderer.wait.set()
        with pytest.raises(MemoryError):
            await task
        assert_closed(svc.engine.renderer.pil_images[0])
        assert not svc.operations and not svc.engine.calls
        await svc.close()
    run(case())


@pytest.mark.parametrize("kind", ["generate", "decision"])
def test_expanded_image_context_is_checked_not_unexpanded_placeholder_size(tmp_path, kind):
    async def case():
        svc = image_service(tmp_path, width=CONTEXT - 20)
        source = acquired(tmp_path)
        with pytest.raises(ValueError, match="context|tokens"):
            if kind == "generate":
                await svc.generate(generate(image=source, max_tokens=30), "long")
            else:
                await svc.decide(image_decision(source), "long")
        assert not svc.engine.calls
        await svc.close()
    run(case())


def test_http_trusted_images_and_retained_decisions_are_explicit_only(tmp_path):
    from fastapi.testclient import TestClient

    source = acquired(tmp_path)
    cfg = config()

    @asynccontextmanager
    async def context():
        yield NS()  # Builder below injects only a CPU fake, no AsyncLLM.

    async def builder(_engine, _config):
        return image_service(tmp_path)

    with TestClient(create_app(context, cfg, builder=builder)) as client:
        assert client.get("/health").json()["capabilities"]["trusted_still_images"] is True
        response = client.post("/v1/completions", json=generate(image=source).model_dump())
        assert response.status_code == 200
        result = response.json()
        decision_response = client.post("/v1/decisions", json=image_decision(source,
            messages=None, prefix_id=result["prefix_id"]).model_dump())
        assert decision_response.status_code == 200
        assert len(decision_response.json()["images"]) == 2
        assert client.get("/health").json()["retained_multimodal_bytes"] > 0
        assert client.delete("/v1/prefixes/" + result["prefix_id"]).json()["released"]
        assert client.get("/health").json()["retained_multimodal_bytes"] == 0
        unsafe = generate(prompt=None, messages=[{"role": "user", "content": "text"}]).model_dump()
        unsafe["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "https://bad"}}]
        assert client.post("/v1/chat/completions", json=unsafe).status_code == 422
        assert client.post("/v1/completions", json=generate(prompt="<$EMB_1000000000$>").model_dump()).status_code == 400


def test_source_replacement_after_snapshot_cannot_change_loaded_pixels(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        source = acquired(tmp_path)
        svc.engine.renderer.wait = asyncio.Event()
        task = asyncio.create_task(svc.generate(generate(image=source), "snapshot"))
        await svc.engine.renderer.entered.wait()
        with Image.new("RGB", (8, 8), (99, 98, 97)) as replacement:
            replacement.save(source.path)
        svc.engine.renderer.wait.set()
        result = await task
        assert result["images"][0]["rgb_sha256"] == source.sha256
        assert torch.all(svc.engine.inputs[0]["mm_kwargs"]["image"][0]["pixel_values"].data == 10)
        with pytest.raises(ValueError, match="checksum"):
            await svc.generate(generate(image=source), "now-invalid")
        await svc.close()
    run(case())


@pytest.mark.parametrize("failure", ["query", "candidate", "cancel"])
def test_image_readout_failures_preserve_ab_but_clean_all_active_branches(tmp_path, failure):
    async def case():
        svc = image_service(tmp_path)
        result = await svc.generate(generate(image=acquired(tmp_path)), "A")
        prefix = svc.prefixes[result["prefix_id"]][0]
        request = image_decision(acquired(tmp_path, (40, 50, 60)), messages=None, prefix_id=prefix.id,
            candidates=[{"id": str(i), "text": "candidate " + str(i)} for i in range(7)])
        if failure in ("query", "candidate"):
            svc.engine.fail_call = 2 if failure == "query" else 4
            with pytest.raises(RuntimeError, match="Synthetic"):
                await svc.decide(request, "decision")
        else:
            svc.engine.wait = asyncio.Event()
            task = asyncio.create_task(svc.decide(request, "decision"))
            await until(lambda: svc.engine.active == 1)
            svc.cancel("decision")
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not svc.operations and not svc.engine_requests and svc.engine.active == 0
        assert svc.engine.aborted
        assert svc.prefixes[prefix.id][0] is prefix
        assert len(prefix.multimodal.images) == 1
        assert svc.retained_multimodal_bytes == prefix.multimodal.nbytes
        assert len(svc.engine.renderer.calls) == 2
        for image in svc.engine.renderer.pil_images:
            assert_closed(image)
        await svc.close()
    run(case())


def test_deleted_prefix_image_payload_stays_alive_only_for_active_borrower(tmp_path):
    async def case():
        svc = image_service(tmp_path)
        result = await svc.generate(generate(image=acquired(tmp_path)), "A")
        ref = weakref.ref(svc.prefixes[result["prefix_id"]][0].multimodal)
        svc.engine.wait = asyncio.Event()
        task = asyncio.create_task(svc.decide(image_decision(messages=None,
            prefix_id=result["prefix_id"]), "active"))
        await until(lambda: svc.engine.active == 1)
        assert svc.release_prefix(result["prefix_id"])
        assert svc.retained_multimodal_bytes == 0 and ref() is not None
        svc.engine.wait.set()
        await task
        gc.collect()
        assert ref() is None
        await svc.close()
    run(case())


def test_image_limit_one_allows_new_c_after_text_ab(tmp_path):
    async def case():
        svc = image_service(tmp_path, limit=1)
        generated = await svc.generate(generate(), "text-A")
        result = await svc.decide(image_decision(acquired(tmp_path), messages=None,
            prefix_id=generated["prefix_id"], candidates=[
                {"id": str(i), "text": "option " + str(i)} for i in range(7)]), "new-image-C")
        assert len(result["images"]) == 1
        assert not svc.engine_requests
        await svc.close()
    run(case())
