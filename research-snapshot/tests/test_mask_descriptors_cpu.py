"""Real CPU tensors only; no train/native/exllamav3/vllm bootstrap imports."""

from contextlib import closing
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from fuzev_research.features import FeatureStore, collate
from fuzev_research.head import CandidateHead


def loss_and_gradients(head, batch):
    logits = head(batch["query"], batch["candidates"], batch["mask"])
    logp = torch.log_softmax(logits, dim=1)
    # Match train_trial's normalized NLL plus its nonzero Brier profile.
    nll = -(batch["targets"] * logp.masked_fill(~batch["mask"], 0)).sum(dim=1)
    loss = (nll / batch["mask"].sum(dim=1).float().log()).mean()
    loss = loss + .1 * ((logp.exp() - batch["targets"]) ** 2).sum(dim=1).mean()
    assert torch.isfinite(loss)
    return (logits.detach(), loss.detach(),
            *torch.autograd.grad(loss, tuple(head.parameters())))


def test_warm_descriptors_match_uncached_batches_head_gradients_and_rng(tmp_path):
    previous_threads = torch.get_num_threads()
    previous_rng, previous_python_rng = torch.get_rng_state(), random.getstate()
    try:
        torch.set_num_threads(1)
        # Read-only fixture: no preparation, state store, or acquisition writes.
        (tmp_path / "identity.json").write_text("{}")
        (tmp_path / "index.sqlite3").touch()
        masks = ([[True, False, True, False, False], [True] * 5],
                 [[True, True, False, False, False], [True, False, True, False, False]])
        for shard, mask in enumerate(masks):
            query = torch.arange(2 * 2560, dtype=torch.float32, device="cpu").reshape(2, 2560) / 2560
            query = query + 2 * shard
            offsets = torch.arange(5, dtype=torch.float32, device="cpu")
            targets = torch.zeros(2, 5, dtype=torch.float32, device="cpu")
            targets[:, 0] = 1
            save_file({"query": query,
                       "candidates": query[:, None, :].repeat(1, 5, 1) + offsets[None, :, None] / 10,
                       "mask": torch.tensor(mask, dtype=torch.bool, device="cpu"),
                       "targets": targets, "ordinal_values": offsets.repeat(2, 1),
                       "baseline": query[:, :5].contiguous()},
                      str(tmp_path / f"shard-{shard:06d}.safetensors"))
        with torch.device("cpu"):
            head = CandidateHead().float()
        rows = [{"shard": shard, "row_": row} for shard, row in ((1, 1), (0, 0), (1, 0), (0, 0))]
        original_rows = [dict(row) for row in rows]
        cpu_rng, python_rng = torch.get_rng_state(), random.getstate()
        with closing(FeatureStore(tmp_path, {}, read_only=True, cache_bytes=0)) as store:
            assert store.mask_width(rows) == 3
            descriptors = dict(store._mask_descriptors)
            actual = collate(store, rows, device="cpu")
            # The existing row-reader fallback predates the descriptor cache.
            # Unlike mask_cache_bytes=0, it never constructs mask descriptors.
            expected = collate(SimpleNamespace(tensors=store.tensors), rows, device="cpu")
            assert actual.keys() == expected.keys()
            assert actual["candidates"].shape == (4, 3, 2560)
            assert actual["query"][:, 0].tolist() == [3., 0., 2., 0.]
            assert actual["mask"].tolist() == [[True, False, True], [True, False, True],
                                               [True, True, False], [True, False, True]]
            for name in actual:
                assert actual[name].device.type == expected[name].device.type == "cpu"
                assert actual[name].dtype == expected[name].dtype
                assert torch.equal(actual[name], expected[name])
            assert all(store._mask_descriptors[shard] is descriptor
                       for shard, descriptor in descriptors.items())
            assert not store.loaded and store.loaded_bytes == 0
            reference = loss_and_gradients(head, expected)
            cached = loss_and_gradients(head, actual)
            for left, right in zip(reference, cached):
                assert left.device.type == right.device.type == "cpu"
                assert left.dtype == right.dtype == torch.float32
                assert torch.equal(left, right)
            assert any(torch.count_nonzero(value).item() for value in reference[2:])
            assert all(parameter.grad is None for parameter in head.parameters())
        assert rows == original_rows
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert random.getstate() == python_rng
    finally:
        torch.set_rng_state(previous_rng)
        random.setstate(previous_python_rng)
        torch.set_num_threads(previous_threads)
