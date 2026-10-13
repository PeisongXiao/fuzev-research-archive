from collections import deque

import pytest
import torch

from exllamav3.fuzev import NativeLane


class State:
    def __init__(self, slot, position=0, checksum=0):
        self.slot, self.position, self.checksum = slot, position, checksum

    def stash(self):
        return {"position": self.position, "checksum": self.checksum}


class Cache:
    def __init__(self, capacity=4):
        self.max_num_tokens, self.initialized = 2048, True
        if capacity > 4:
            self.max_num_tokens = 4096
        self.num_slots = capacity
        self.recurrent_layers = {0: object()}
        self.free_list = deque(range(capacity))
        self.pages, self.qsa = {}, {}

    def get_new_state(self):
        return State(self.free_list.popleft())

    def new_from_stashed(self, stashed, position):
        assert position == stashed["position"]
        return State(self.free_list.popleft(), position, stashed["checksum"])

    def release_state(self, state):
        self.free_list.append(state.slot)

    def copy_page(self, target, old, new, count):
        self.pages[new] = list(self.pages[old][:count])
        self.qsa[new] = list(self.qsa[old][:count])


class Model:
    loaded_tp = False

    def forward(self, ids, params):
        cache = params["cache"]
        outputs = []
        for row, state in enumerate(params["recurrent_states"]):
            start = params["cache_seqlens"][row].item()
            for index, token in enumerate(ids[row].tolist(), start):
                page = params["block_table"][row, index // 256].item()
                local = index % 256
                for storage, value in ((cache.pages, token), (cache.qsa, token * 2)):
                    data = storage.setdefault(page, [])
                    if len(data) > local:
                        data[local] = value
                    else:
                        data.append(value)
                state.checksum += token
            state.position += ids.shape[1]
            outputs.append([[state.checksum]])
        return torch.tensor(outputs, dtype=torch.float32)

    prefill = forward
    forward_features = forward


def test_partial_page_cow_and_hybrid_checkpoint_are_isolated():
    lane = NativeLane(Model(), Cache(), chunk_size=64)
    base = lane.create()
    lane.append(base, list(range(260)), features=True)
    lane.freeze(base)
    a, b = lane.fork(base), lane.fork(base)
    assert a.pages[0] == b.pages[0] == base.pages[0]
    initial = sum(range(260))
    assert lane.append(a, [11], features=True).item() == initial + 11
    assert lane.append(b, [29], features=True).item() == initial + 29
    assert len({base.pages[-1], a.pages[-1], b.pages[-1]}) == 3
    assert lane.cache.pages[base.pages[-1]] == [256, 257, 258, 259]
    assert lane.cache.qsa[a.pages[-1]][-1] == 22
    assert base.checkpoint["checksum"] == initial
    assert base.recurrent is None
    with pytest.raises(ValueError, match="immutable"):
        lane.append(base, [7])
    lane.close()
    assert len(lane.pool.free) == 8
    assert len(lane.cache.free_list) == 4


def test_frozen_prefixes_do_not_consume_active_hybrid_slots():
    lane = NativeLane(Model(), Cache())
    bases = []
    for _ in range(7):
        base = lane.create()
        lane.append(base, [3])
        bases.append(lane.freeze(base))
    assert len(lane.cache.free_list) == 4
    child = lane.fork(bases[0])
    assert lane.append(child, [5], features=True).item() == 8
    lane.close()
    assert len(lane.cache.free_list) == 4


def test_failure_poisoned_branch_is_released_and_prefix_survives():
    model, cache = Model(), Cache()
    lane = NativeLane(model, cache)
    base = lane.create()
    lane.append(base, [3])
    child = lane.fork(lane.freeze(base))
    model.forward_features = lambda *args: (_ for _ in ()).throw(RuntimeError("failed forward"))
    with pytest.raises(RuntimeError, match="failed forward"):
        lane.append(child, [5], features=True)
    assert child.closed
    assert not base.closed
    assert base.tokens == (3,)
    lane.close()


def test_context_and_foreign_branch_checks_do_not_mutate_state():
    lane = NativeLane(Model(), Cache(), context_limit=2)
    branch = lane.create()
    with pytest.raises(ValueError, match="context"):
        lane.append(branch, [1, 2, 3])
    assert branch.tokens == ()
    other = NativeLane(Model(), Cache())
    with pytest.raises(ValueError, match="foreign"):
        other.fork(branch)
    lane.close()


def test_exact_tail_batch_matches_serial_independent_continuations():
    lane = NativeLane(Model(), Cache(), chunk_size=2)
    base = lane.create()
    lane.append(base, [3, 7], features=True)
    lane.freeze(base)
    tails = [[11, 13, 17], [19, 23, 29], [31, 37, 41], [43, 47, 53]]
    expected = []
    for tail in tails:
        child = lane.fork(base)
        expected.append(lane.append(child, tail, features=True))
        lane.release(child)
    children = [lane.fork(base) for _ in tails]
    actual = lane.append_features_many(children, tails)
    assert torch.equal(actual, torch.cat(expected))
    assert len({child.pages[0] for child in children}) == 4
    assert all(child.tokens == base.tokens + tuple(tail) for child, tail in zip(children, tails))
    assert base.checkpoint["checksum"] == 10
    assert lane.cache.qsa[base.pages[0]] == [6, 14]
    lane.close()
    assert len(lane.pool.free) == 8
    assert len(lane.cache.free_list) == 4


def test_batch_validation_never_pads_or_mutates_accepted_tokens():
    lane = NativeLane(Model(), Cache())
    base = lane.create()
    lane.append(base, [3])
    lane.freeze(base)
    children = [lane.fork(base) for _ in range(2)]
    with pytest.raises(ValueError, match="equal-length"):
        lane.append_features_many(children, [[1], [2, 3]])
    assert all(child.tokens == (3,) for child in children)
    with pytest.raises(ValueError, match="independent"):
        lane.append_features_many([children[0]] * 2, [[1], [1]])
    lane.close()


def test_batch_failure_releases_all_mutable_branches_not_the_source():
    model, cache = Model(), Cache()
    lane = NativeLane(model, cache)
    base = lane.create()
    lane.append(base, [3])
    lane.freeze(base)
    children = [lane.fork(base) for _ in range(4)]
    model.forward_features = lambda *args: (_ for _ in ()).throw(RuntimeError("failed batch"))
    with pytest.raises(RuntimeError, match="failed batch"):
        lane.append_features_many(children, [[5]] * 4)
    assert all(child.closed for child in children)
    assert not base.closed
    assert len(lane.cache.free_list) == 4
    lane.close()


@pytest.mark.parametrize("count", range(1, 9))
def test_declared_eight_lane_capacity_keeps_every_branch_and_page_independent(count):
    model = Model()
    model.fuzev_batch_capacity = 8
    lane = NativeLane(model, Cache(8), chunk_size=2)
    base = lane.create()
    lane.append(base, [3, 7], features=True)
    lane.freeze(base)
    tails = [[11 + index, 13 + index, 17 + index] for index in range(count)]
    expected = []
    for tail in tails:
        child = lane.fork(base)
        expected.append(lane.append(child, tail, features=True))
        lane.release(child)
    children = [lane.fork(base) for _ in tails]
    actual = lane.append_features_many(children, tails)
    assert lane.batch_capacity == 8
    assert torch.equal(actual, torch.cat(expected))
    assert len({child.recurrent.slot for child in children}) == count
    assert len({base.pages[0], *(child.pages[0] for child in children)}) == count + 1
    assert all(child.tokens == base.tokens + tuple(tail) for child, tail in zip(children, tails))
    assert base.checkpoint["checksum"] == 10
    lane.close()
    assert len(lane.cache.free_list) == 8 and len(lane.pool.free) == 16


@pytest.mark.parametrize("declared,slots,expected", [(4, 8, 4), (8, 4, 4), (8, 8, 8), (8, 5, 5)])
def test_batch_capacity_is_bounded_by_both_model_declaration_and_actual_slots(declared, slots, expected):
    model = Model()
    model.fuzev_batch_capacity = declared
    lane = NativeLane(model, Cache(slots))
    assert lane.batch_capacity == expected
    with pytest.raises(ValueError, match="matching branches"):
        lane.append_features_many([None] * (expected + 1), [[1]] * (expected + 1))
    lane.close()


def test_eight_branch_error_cleanup_and_unpadded_validation():
    model, cache = Model(), Cache(8)
    model.fuzev_batch_capacity = 8
    lane = NativeLane(model, cache)
    base = lane.create()
    lane.append(base, [3])
    lane.freeze(base)
    children = [lane.fork(base) for _ in range(8)]
    with pytest.raises(ValueError, match="equal-length"):
        lane.append_features_many(children, [[5]] * 7 + [[5, 6]])
    assert all(child.tokens == (3,) for child in children)
    with pytest.raises(ValueError, match="independent"):
        lane.append_features_many([children[0]] * 8, [[5]] * 8)
    model.forward_features = lambda *args: (_ for _ in ()).throw(RuntimeError("failed eight batch"))
    with pytest.raises(RuntimeError, match="failed eight batch"):
        lane.append_features_many(children, [[5]] * 8)
    assert all(child.closed for child in children)
    assert not base.closed and base.checkpoint["checksum"] == 3
    assert len(cache.free_list) == 8
    lane.close()


@pytest.mark.parametrize("value", [0, 9, True, 4.0])
def test_invalid_model_capacity_fails_closed(value):
    model = Model()
    model.fuzev_batch_capacity = value
    with pytest.raises(ValueError, match="declared batch capacity"):
        NativeLane(model, Cache(8))
