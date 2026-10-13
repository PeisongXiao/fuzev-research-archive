"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks.

Only acquired, RGB-digest-checked still PNGs cross this boundary. Expanded
placeholder IDs and model-specific field metadata come from vLLM itself.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace

import torch
from PIL import Image

from .encoding import token_hash
from .service_contracts import TrustedImage
from .service_images import IMAGE_ADMISSION_BYTES, checked_image


PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_PAYLOAD_BYTES = 512 << 20
POLICY = "qwen-image-pad-expansion-owned-cpu-v1"


def _structure(value, *, clone=False, digest=None, depth=0):
    """Bounded data/field traversal; tensors stay CPU and keep their dtype."""
    if depth > 16:
        raise ValueError("Multimodal payload nesting exceeds its resource bound")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Only dense CPU processor payloads can be retained")
        if clone:
            with torch.inference_mode(False):
                return value.detach().clone(memory_format=torch.contiguous_format)
        if digest is not None:
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
        return ("tensor", str(value.dtype), tuple(value.shape), value.data_ptr(), value._version)
    if isinstance(value, Mapping):
        if len(value) > 64 or any(not isinstance(key, str) or len(key) > 4096 for key in value):
            raise ValueError("Invalid or oversized multimodal field mapping")
        if digest is not None:
            digest.update(json.dumps(["mapping", type(value).__name__, list(value)]).encode())
        items = [(key, _structure(item, clone=clone, digest=digest, depth=depth + 1))
                 for key, item in value.items()]
        if clone:
            return type(value)(dict(items))
        return (type(value).__module__, type(value).__name__, tuple(items))
    if is_dataclass(value) and not isinstance(value, type):
        if digest is not None:
            digest.update(json.dumps(["dataclass", type(value).__name__,
                [field.name for field in fields(value) if field.init]]).encode())
        values = {field.name: _structure(getattr(value, field.name), clone=clone,
                      digest=digest, depth=depth + 1) for field in fields(value) if field.init}
        if clone:
            return replace(value, **values)
        return (type(value).__module__, type(value).__name__, tuple(values.items()))
    if isinstance(value, (list, tuple)):
        if len(value) > 262144:
            raise ValueError("Multimodal metadata exceeds its resource bound")
        if digest is not None:
            digest.update(json.dumps(["sequence", type(value).__name__, len(value)]).encode())
        items = [_structure(item, clone=clone, digest=digest, depth=depth + 1) for item in value]
        return type(value)(items) if clone else (type(value).__name__, tuple(items))
    if isinstance(value, slice):
        if digest is not None:
            digest.update(b'["slice"]')
        parts = tuple(_structure(part, clone=clone, digest=digest, depth=depth + 1)
                      for part in (value.start, value.stop, value.step))
        return slice(*parts) if clone else ("slice", *parts)
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError("Multimodal metadata string exceeds its bound")
        if digest is not None:
            digest.update(json.dumps(["scalar", type(value).__name__, value], allow_nan=False).encode())
        return value
    raise ValueError(f"Unsupported multimodal metadata type: {type(value).__name__}")


def _bytes(value, depth=0):
    if depth > 16:
        raise ValueError("Multimodal payload nesting exceeds its resource bound")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Only dense CPU processor payloads can be retained")
        return 256 + value.numel() * value.element_size()
    if isinstance(value, Mapping):
        if len(value) > 64:
            raise ValueError("Multimodal fields exceed their resource bound")
        return 256 + sum(128 + len(key) + _bytes(item, depth + 1)
                         for key, item in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return 256 + sum(_bytes(getattr(value, field.name), depth + 1)
                   for field in fields(value) if field.init)
    if isinstance(value, (list, tuple)):
        if len(value) > 262144:
            raise ValueError("Multimodal metadata exceeds its resource bound")
        return 64 + sum(8 + _bytes(item, depth + 1) for item in value)
    if isinstance(value, slice):
        return 64 + sum(_bytes(part, depth + 1) for part in (value.start, value.stop, value.step))
    if value is None or type(value) in (str, int, float, bool):
        return 64 + len(value) if isinstance(value, str) else 64
    raise ValueError("Unknown multimodal payload storage")


@dataclass(frozen=True)
class ImagePayload:
    rgb_sha256: str
    mm_hash: str
    item: object
    placeholder: object
    placeholder_tokens: tuple[int, ...]
    payload_sha256: str
    nbytes: int
    signature: tuple

    @classmethod
    def capture(cls, source, processed, raw_tokens, image_pad_id, pad_offset, limit):
        if processed.get("type") != "multimodal":
            raise ValueError("Renderer did not return a processed MultiModalInput")
        tokens = processed.get("prompt_token_ids")
        if (not isinstance(tokens, (list, tuple)) or not tokens or len(tokens) > 262144
                or any(type(token) is not int or not 0 <= token < 1_000_000_000 for token in tokens)):
            raise ValueError("Invalid expanded vLLM image tokens; native aliases are not accepted")
        groups = [processed.get(name) for name in ("mm_kwargs", "mm_hashes", "mm_placeholders")]
        if any(not isinstance(group, Mapping) or set(group) != {"image"}
               or not isinstance(group["image"], (list, tuple)) or len(group["image"]) != 1
               for group in groups):
            raise ValueError("Renderer must return one complete still-image payload")
        item, mm_hash, position = (group["image"][0] for group in groups)
        if item is None or not isinstance(item, Mapping) or not item:
            raise ValueError("Cache-only multimodal references cannot own retained image state")
        if not isinstance(mm_hash, str) or not 1 <= len(mm_hash) <= 256:
            raise ValueError("Invalid processor image hash")
        if (not is_dataclass(position) or type(position.offset) is not int
                or type(position.length) is not int or position.offset != pad_offset
                or not 1 <= position.length <= 262144):
            raise ValueError("Unsupported image placeholder placement")
        end = position.offset + position.length
        tokens = tuple(tokens)
        if (end > len(tokens) or tokens[:pad_offset] != raw_tokens[:pad_offset]
                or tokens[end:] != raw_tokens[pad_offset + 1:]
                or tokens[pad_offset:end] != (image_pad_id,) * position.length):
            raise ValueError("Processor changed tokens outside the image pad; exact append is unsupported")
        mask = position.is_embed
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
                or mask.dtype != torch.bool or mask.shape != (position.length,)):
            raise ValueError("Invalid processor placeholder embedding mask")
        for key, elem in item.items():
            if not is_dataclass(elem) or not hasattr(elem, "data") or elem.data is None:
                raise ValueError(f"Processor field {key} lacks owned data/field metadata")
        if "image_grid_thw" not in item or "pixel_values" not in item:
            raise ValueError("The Qwen still-image path requires pixels and image_grid_thw")
        grid = item["image_grid_thw"].data
        if (not isinstance(grid, torch.Tensor) or grid.device.type != "cpu" or grid.shape != (3,)
                or grid.dtype not in (torch.int32, torch.int64) or grid[0].item() != 1
                or (grid <= 0).any()):
            raise ValueError("Invalid still-image grid metadata")
        pixels = item["pixel_values"].data
        if not isinstance(pixels, torch.Tensor) or pixels.numel() == 0:
            raise ValueError("Missing processed image pixels")
        nbytes = _bytes(item) + _bytes(position)
        if not 0 < nbytes <= limit:
            raise MemoryError("Processed image payload exceeds the owned byte bound")
        owned_item, owned_position = _structure(item, clone=True), _structure(position, clone=True)
        digest = hashlib.sha256()
        _structure(owned_item, digest=digest)
        signature = (_structure(owned_item), _structure(owned_position))
        return tokens, cls(source.sha256, mm_hash, owned_item, owned_position,
            tokens[pad_offset:end], digest.hexdigest(), nbytes, signature)

    def validate(self, tokens):
        if (_structure(self.item), _structure(self.placeholder)) != self.signature:
            raise ValueError("Retained processor payload or placeholder metadata changed")
        start, length = self.placeholder.offset, self.placeholder.length
        if tuple(tokens[start:start + length]) != self.placeholder_tokens:
            raise ValueError("Retained image range does not match exact prompt tokens")

    def shifted(self, offset):
        position = replace(self.placeholder, offset=self.placeholder.offset + offset)
        return replace(self, placeholder=position,
                       signature=(_structure(self.item), _structure(position)))

    def receipt(self):
        mask = self.placeholder.is_embed
        mask_info = None if mask is None else {
            "length": mask.numel(), "sha256": hashlib.sha256(
                memoryview(mask.numpy()).cast("B")).hexdigest()}
        return {"rgb_sha256": self.rgb_sha256, "mm_hash": self.mm_hash,
                "payload_sha256": self.payload_sha256, "owned_bytes": self.nbytes,
                "offset": self.placeholder.offset, "length": self.placeholder.length,
                "is_embed": mask_info, "placeholder_token_sha256": token_hash(self.placeholder_tokens)}


@dataclass(frozen=True)
class MultimodalPayload:
    images: tuple[ImagePayload, ...]

    @property
    def nbytes(self):
        return sum(image.nbytes for image in self.images)

    def validate(self, tokens):
        if not 1 <= len(self.images) <= 2:
            raise ValueError("At most two still-image bindings are supported in AB+C")
        previous_end = 0
        for image in self.images:
            image.validate(tokens)
            if image.placeholder.offset < previous_end:
                raise ValueError("Overlapping or reordered image placeholder ranges")
            previous_end = image.placeholder.offset + image.placeholder.length

    def append(self, suffix, offset):
        images = self.images + tuple(image.shifted(offset) for image in suffix.images)
        if len(images) > 2:
            raise ValueError("AB+C may contain at most two still images")
        return MultimodalPayload(images)

    def engine_input(self, tokens, cache_salt):
        self.validate(tokens)
        # Copy mutable CPU fields for each engine request; IPC/driver mutation
        # cannot change the retained canonical payload or another candidate.
        return {"type": "multimodal", "prompt_token_ids": list(tokens),
                "mm_kwargs": {"image": [_structure(image.item, clone=True) for image in self.images]},
                "mm_hashes": {"image": [image.mm_hash for image in self.images]},
                "mm_placeholders": {"image": [_structure(image.placeholder, clone=True)
                                                  for image in self.images]},
                "cache_salt": cache_salt}

    def receipt(self):
        return [image.receipt() for image in self.images]


class VllmImages:
    def __init__(self, renderer, tokenizer, root, identity, *, check, admission,
                 max_payload_bytes=MAX_PAYLOAD_BYTES):
        if type(max_payload_bytes) is not int or not 0 < max_payload_bytes <= MAX_PAYLOAD_BYTES:
            raise ValueError("Owned image payload budget must be positive and at most 512 MiB")
        self.renderer, self.tokenizer, self.root, self.identity = renderer, tokenizer, root, identity
        self.check, self.admission = check, admission
        self.max_payload_bytes = max_payload_bytes
        self.lock = asyncio.Lock()
        processor = renderer.mm_processor
        if processor is None:
            raise ValueError("This existing vLLM engine has no multimodal processor")
        self.image_limit = processor.info.allowed_mm_limits.get("image", 0)
        if self.image_limit < 1:
            raise ValueError("This vLLM cell does not admit still images")
        ids = tokenizer.encode(PLACEHOLDER, add_special_tokens=False, truncation=False)
        if (len(ids) != 3 or len(set(ids)) != 3
                or any(type(token) is not int or not 0 <= token < 1_000_000_000 for token in ids)):
            raise ValueError("The image adapter requires Qwen's three special image marker IDs")
        self.marker_ids = tuple(ids)

    def check_count(self, payload):
        if payload is not None and len(payload.images) > min(2, self.image_limit):
            raise ValueError("AB plus new image C exceeds this cell's image limit; configure image limit >=2")

    def _load(self, source):
        with checked_image(source, self.root) as snapshot:
            with Image.open(snapshot) as opened:
                return opened.convert("RGB")

    async def _prepare(self, source, raw_tokens, pad_offset):
        image = None
        try:
            image = await asyncio.to_thread(self._load, source)
            self.check()
            prompt = {"prompt_token_ids": list(raw_tokens),
                      "multi_modal_data": {"image": [image]},
                      "multi_modal_uuids": {"image": ["fuzev-rgb:" + self.identity + ":" + source.sha256]},
                      "mm_processor_kwargs": {"truncation": False}}
            processed = await self.renderer.process_for_engine_async(
                prompt, time.time(), skip_mm_cache=True)
            self.check()
            tokens, binding = ImagePayload.capture(source, processed, raw_tokens,
                self.marker_ids[1], pad_offset, self.max_payload_bytes)
            return tokens, MultimodalPayload((binding,))
        finally:
            if image is not None:
                image.close()

    async def prepare(self, source, raw_tokens):
        source = TrustedImage.model_validate(source)
        raw_tokens = tuple(raw_tokens)
        if (not 1 <= len(raw_tokens) <= 262144
                or any(type(token) is not int or not 0 <= token < 1_000_000_000 for token in raw_tokens)):
            raise ValueError("Fresh image tokens exceed the native window or contain indexed native aliases")
        starts = [index for index in range(len(raw_tokens) - 2)
                  if raw_tokens[index:index + 3] == self.marker_ids]
        if len(starts) != 1 or raw_tokens.count(self.marker_ids[1]) != 1:
            raise ValueError("Fresh image preprocessing requires one controlled image placeholder")
        async with self.lock:
            self.check()
            self.admission(IMAGE_ADMISSION_BYTES + 2 * self.max_payload_bytes)
            task = asyncio.create_task(self._prepare(source, raw_tokens, starts[0] + 1))
            interrupted, error = False, None
            # CPU executors cannot be preempted safely. Hold the request/lock
            # through completion and close PIL storage before reporting cancel.
            while not task.done():
                try:
                    await asyncio.wait({task}, timeout=1.)
                except asyncio.CancelledError:
                    interrupted = True
                if error is None:
                    try:
                        self.check()
                    except Exception as failure:
                        error = failure
            try:
                result = task.result()
            except BaseException:
                if interrupted:
                    raise asyncio.CancelledError from None
                raise
            if interrupted:
                raise asyncio.CancelledError
            if error is not None:
                raise error
            return result
