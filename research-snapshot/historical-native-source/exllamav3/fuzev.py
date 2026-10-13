"""Exact-token native lanes with KV/QSA copy-on-write and hybrid checkpoints.

FuZev additions, 2026-10-01. Generation and decisions use the same model and
cache; decisions bypass the vocabulary head. No physical state crosses engines.
"""

from collections import deque
from dataclasses import dataclass, field
from threading import RLock

import torch

from .constants import PAGE_SIZE


class PagePool:
    def __init__(self, cache):
        self.cache = cache
        self.free = deque(range(cache.max_num_tokens // PAGE_SIZE))
        self.references = [0] * len(self.free)

    def allocate(self):
        if not self.free:
            raise MemoryError("Native KV page pool exhausted")
        page = self.free.popleft()
        self.references[page] = 1
        return page

    def retain(self, pages):
        for page in pages:
            if self.references[page] == 0:
                raise RuntimeError("Cannot retain a released cache page")
            self.references[page] += 1

    def release(self, pages):
        for page in pages:
            if self.references[page] <= 0:
                raise RuntimeError("Double cache-page release")
            self.references[page] -= 1
            if self.references[page] == 0:
                self.free.append(page)

    def prepare_write(self, pages, position, count):
        required = (position + count + PAGE_SIZE - 1) // PAGE_SIZE
        partial = position % PAGE_SIZE
        shared_tail = bool(partial and self.references[pages[-1]] > 1)
        needed = max(0, required - len(pages)) + int(shared_tail)
        if needed > len(self.free):
            raise MemoryError("Insufficient native KV pages for independent continuation")
        if shared_tail:
            old, new = pages[-1], self.allocate()
            try:
                # Cache.copy_page also copies raw and pooled QSA indexer planes.
                self.cache.copy_page(self.cache, old, new, partial)
            except BaseException:
                self.release([new])
                raise
            pages[-1] = new
            self.release([old])
        while len(pages) < required:
            pages.append(self.allocate())


@dataclass(eq=False)
class Branch:
    owner: object
    recurrent: object
    pages: list = field(default_factory=list)
    tokens: tuple = ()
    closed: bool = False
    immutable: bool = False
    checkpoint: object = None


class NativeLane:
    def __init__(self, model, cache, context_limit=262144, chunk_size=8192):
        if model.loaded_tp:
            raise NotImplementedError("FuZev native lanes currently require a single GPU")
        if not cache.initialized:
            raise ValueError("Load the model and attached cache before creating a lane")
        self.model, self.cache = model, cache
        self.context_limit, self.chunk_size = context_limit, chunk_size
        declared = getattr(model, "fuzev_batch_capacity", 4)
        slots = getattr(cache, "num_slots", len(cache.free_list))
        if type(declared) is not int or not 1 <= declared <= 8 or type(slots) is not int or slots < 1:
            raise ValueError("Native lane needs positive cache slots and a declared batch capacity of one to eight")
        self.batch_capacity = min(declared, slots)
        self.pool, self.lock = PagePool(cache), RLock()
        self.branches = set()

    def _check(self, branch):
        if branch.owner is not self or branch.closed or branch not in self.branches:
            raise ValueError("Released or foreign native branch")

    def create(self):
        with self.lock:
            if self.cache.recurrent_layers and not self.cache.free_list:
                raise MemoryError("No free hybrid-state slot")
            state = self.cache.get_new_state() if self.cache.recurrent_layers else None
            branch = Branch(self, state)
            self.branches.add(branch)
            return branch

    def freeze(self, branch):
        with self.lock:
            self._check(branch)
            if branch.recurrent is not None:
                branch.checkpoint = branch.recurrent.stash()
                self.cache.release_state(branch.recurrent)
                branch.recurrent = None
            branch.immutable = True
            return branch

    def fork(self, branch):
        with self.lock:
            self._check(branch)
            if branch.recurrent is not None or branch.checkpoint is not None:
                if not self.cache.free_list:
                    raise MemoryError("No free hybrid-state slot for continuation")
                stashed = branch.checkpoint if branch.checkpoint is not None else branch.recurrent.stash()
                state = self.cache.new_from_stashed(stashed, len(branch.tokens))
            else:
                state = None
            child = Branch(self, state, list(branch.pages), branch.tokens)
            self.pool.retain(child.pages)
            self.branches.add(child)
            return child

    def release(self, branch):
        with self.lock:
            self._check(branch)
            self.pool.release(branch.pages)
            if branch.recurrent is not None:
                self.cache.release_state(branch.recurrent)
            branch.closed = True
            branch.checkpoint = None
            self.branches.remove(branch)

    @torch.inference_mode()
    def append(self, branch, tokens, *, features=False, params=None):
        with self.lock:
            self._check(branch)
            if branch.immutable:
                raise ValueError("Fork an immutable prefix before continuing it")
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.reshape(-1).tolist()
            tokens = tuple(tokens)
            if not tokens or any(not isinstance(token, int) or token < 0 for token in tokens):
                raise ValueError("Append requires nonempty exact token IDs")
            if len(branch.tokens) + len(tokens) > self.context_limit:
                raise ValueError("Expanded input exceeds the native context window")
            if params and set(params) & {"cache", "block_table", "cache_seqlens", "recurrent_states", "batch_shape"}:
                raise ValueError("Cache ownership parameters cannot be overridden")
            output = None
            try:
                for offset in range(0, len(tokens), self.chunk_size):
                    chunk = tokens[offset:offset + self.chunk_size]
                    position = len(branch.tokens)
                    self.pool.prepare_write(branch.pages, position, len(chunk))
                    arguments = dict(params or {})
                    arguments.update({
                        "attn_mode": "flash_attn", "cache": self.cache,
                        "block_table": torch.tensor([branch.pages], dtype=torch.int32),
                        "cache_seqlens": torch.tensor([position], dtype=torch.int32),
                        "last_tokens_only": 1,
                    })
                    if branch.recurrent is not None:
                        arguments["recurrent_states"] = [branch.recurrent]
                    input_ids = torch.tensor([chunk], dtype=torch.long)
                    last = offset + len(chunk) == len(tokens)
                    if last:
                        if features:
                            output = self.model.forward_features(input_ids, arguments)
                        else:
                            output = self.model.forward(input_ids, arguments)
                    else:
                        self.model.prefill(input_ids, arguments)
                    branch.tokens += chunk
                    if branch.recurrent is not None and branch.recurrent.position != len(branch.tokens):
                        raise RuntimeError("Hybrid checkpoint and accepted tokens have diverged")
            except BaseException:
                # Failed forwards may have mutated every hybrid component. Never
                # return such a branch to clients as a reusable committed prefix.
                self.release(branch)
                raise
            return output

    def close(self):
        with self.lock:
            for branch in list(self.branches):
                self.release(branch)

    @torch.inference_mode()
    def append_features_many(self, branches, tails, *, params=None):
        """Batch exact equal-length tails; do not pad or share mutable state."""
        with self.lock:
            branches = tuple(branches)
            tails = tuple(tuple(tail) for tail in tails)
            if not 1 <= len(branches) <= self.batch_capacity or len(branches) != len(tails):
                raise ValueError(f"Feature batching requires one to {self.batch_capacity} matching branches")
            if len(set(branches)) != len(branches):
                raise ValueError("Feature branches must be independent")
            for branch in branches:
                self._check(branch)
                if branch.immutable:
                    raise ValueError("Fork an immutable prefix before continuing it")
            lengths = {len(tail) for tail in tails}
            positions = {len(branch.tokens) for branch in branches}
            if len(lengths) != 1 or not next(iter(lengths)) or len(positions) != 1:
                raise ValueError("Batch exact equal-length tails at a common prefix position")
            if any(type(token) is not int or token < 0 for tail in tails for token in tail):
                raise ValueError("Append requires exact token IDs")
            count = len(tails[0])
            if next(iter(positions)) + count > self.context_limit:
                raise ValueError("Expanded input exceeds the native context window")
            if params and set(params) & {"cache", "block_table", "cache_seqlens", "recurrent_states", "batch_shape"}:
                raise ValueError("Cache ownership parameters cannot be overridden")
            try:
                for offset in range(0, count, self.chunk_size):
                    chunks = [tail[offset:offset + self.chunk_size] for tail in tails]
                    position = len(branches[0].tokens)
                    for branch, chunk in zip(branches, chunks):
                        self.pool.prepare_write(branch.pages, position, len(chunk))
                    width = max(len(branch.pages) for branch in branches)
                    arguments = dict(params or {})
                    arguments.update({
                        "attn_mode": "flash_attn", "cache": self.cache,
                        "block_table": torch.tensor([
                            branch.pages + [-1] * (width - len(branch.pages))
                            for branch in branches], dtype=torch.int32),
                        "cache_seqlens": torch.tensor([position] * len(branches), dtype=torch.int32),
                        "last_tokens_only": 1,
                    })
                    if branches[0].recurrent is not None:
                        arguments["recurrent_states"] = [branch.recurrent for branch in branches]
                    input_ids = torch.tensor(chunks, dtype=torch.long)
                    if offset + len(chunks[0]) == count:
                        output = self.model.forward_features(input_ids, arguments)
                    else:
                        self.model.prefill(input_ids, arguments)
                    for branch, chunk in zip(branches, chunks):
                        branch.tokens += chunk
                        if branch.recurrent is not None and branch.recurrent.position != len(branch.tokens):
                            raise RuntimeError("Hybrid checkpoint and accepted tokens have diverged")
                if output.shape[0] != len(branches):
                    raise RuntimeError("Feature batch lost its independent branch axis")
                return output
            except BaseException:
                for branch in branches:
                    if not branch.closed:
                        self.release(branch)
                raise
