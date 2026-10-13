import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Prefix:
    id: str
    token_ids: tuple[int, ...]
    consumed: int
    identity: str
    image_hashes: tuple[str, ...]


def identity_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class PrefixRegistry:
    def __init__(self, identity, capacity=32, ttl=900, clock=time.monotonic):
        self.identity = identity
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def expire(self):
        now = self.clock()
        for key, (_, used) in list(self.entries.items()):
            if now - used >= self.ttl:
                del self.entries[key]

    def retain(self, token_ids, consumed, image_hashes=()):
        tokens = tuple(token_ids)
        if not 0 <= consumed <= len(tokens):
            raise ValueError("Consumed boundary must lie within the accepted token sequence")
        if any(not isinstance(t, int) or t < 0 for t in tokens):
            raise ValueError("Invalid token sequence")
        with self.lock:
            self.expire()
            prefix = Prefix("prefix_" + secrets.token_hex(16), tokens, consumed,
                            self.identity, tuple(image_hashes))
            self.entries[prefix.id] = (prefix, self.clock())
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return prefix

    def get(self, key, identity):
        with self.lock:
            self.expire()
            prefix, _ = self.entries[key]
            if prefix.identity != identity:
                raise ValueError("Prefix belongs to an incompatible execution identity")
            self.entries.move_to_end(key)
            self.entries[key] = (prefix, self.clock())
            return prefix

    def release(self, key):
        with self.lock:
            return self.entries.pop(key, None) is not None

