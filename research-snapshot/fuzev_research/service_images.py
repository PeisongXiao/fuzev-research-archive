"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks.

No URLs, uploads, generated file paths, or image-token cache live here. The
service passes the verified byte snapshot to the backend's existing vision path.
"""

import hashlib
import io
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .encoding import token_hash


MAX_IMAGE_BYTES = 128 << 20
MAX_IMAGE_PIXELS = 32_000_000
IMAGE_ADMISSION_BYTES = 512 << 20
FIRST_IMAGE_TOKEN = 1_000_000_000  # Native MMTokenAllocator's indexed range.


@contextmanager
def checked_image(source, root):
    """Open only an acquired digest.png, without following a client symlink.

The bounded immutable byte snapshot prevents path replacement between checksum
validation and native decoding. ImageIndex hashes RGB pixels, not PNG bytes.
The acquired cache is flat; no client-controlled subdirectory is needed.
"""
    path, root = Path(source.path), root.resolve()
    if (not path.is_absolute() or path.parent != root
            or path.name != source.sha256 + ".png"):
        raise ValueError("Image must be an acquired digest.png directly under data/images")
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_IMAGE_BYTES:
                raise ValueError("Image is not a bounded regular file")
            raw = handle.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds its encoded byte bound")
    except OSError as error:
        raise ValueError("Acquired image cannot be opened safely") from error
    with io.BytesIO(raw) as snapshot:
        del raw
        try:
            with Image.open(snapshot) as opened:
                if (opened.format != "PNG" or getattr(opened, "n_frames", 1) != 1
                        or opened.width * opened.height > MAX_IMAGE_PIXELS):
                    raise ValueError("Only acquired still PNG images up to 32 million pixels are accepted")
                with opened.convert("RGB") as rgb:
                    digest = hashlib.sha256(f"{rgb.width},{rgb.height}:".encode())
                    digest.update(rgb.tobytes())
                if digest.hexdigest() != source.sha256:
                    raise ValueError("Acquired image RGB checksum mismatch")
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("Invalid acquired image") from error
        snapshot.seek(0)
        yield snapshot


def embedding_identity(embedding):
    return (embedding.first_index, embedding.last_index, embedding.text_alias,
            tuple(embedding.token_list), tuple(embedding.grid_thw or ()),
            embedding.mrope_merge_size)


@dataclass(frozen=True)
class ImageBinding:
    sha256: str
    embedding: object
    identity: tuple

    @classmethod
    def capture(cls, digest, embedding):
        identity = embedding_identity(embedding)
        first, last, _, tokens, _, _ = identity
        if (not FIRST_IMAGE_TOKEN <= first < last
                or last - first != embedding.mm_length or len(tokens) != embedding.full_length):
            raise ValueError("Invalid native indexed image identity")
        return cls(digest, embedding, identity)

    def validate(self):
        if (embedding_identity(self.embedding) != self.identity
                or self.embedding.mm_length != self.identity[1] - self.identity[0]
                or self.embedding.full_length != len(self.identity[3])):
            raise ValueError("Retained image embedding identity changed")

    def receipt(self):
        return {"rgb_sha256": self.sha256, "token_sha256": token_hash(self.identity[3]),
                "first_index": self.identity[0], "last_index": self.identity[1]}


def embeddings_for(bindings):
    # A repeated image can share one native instance, even if it appears twice.
    result = list({id(binding.embedding): binding.embedding for binding in bindings}.values())
    for index, embedding in enumerate(result):
        if any(embedding.first_index < prior.last_index and prior.first_index < embedding.last_index
               for prior in result[:index]):
            raise ValueError("Conflicting retained image token ranges")
    return result


def validate_image_tokens(tokens, bindings):
    ranges = [(binding.identity[0], binding.identity[1]) for binding in bindings]
    if any(token >= FIRST_IMAGE_TOKEN and not any(a <= token < b for a, b in ranges)
           for token in tokens):
        raise ValueError("Indexed image tokens have no retained embedding instance")
