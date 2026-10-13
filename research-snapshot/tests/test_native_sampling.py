"""Synthetic CPU sampling contracts, not CUDA or native-engine qualification."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fuzev_research.native_service import NativeService


def reference_sample(logits, request, generator):
    """Unchanged fea0428 full-population algorithm, independent of the fast path."""
    values = logits[0, -1].detach().float().cpu()
    if (values.ndim != 1 or torch.isnan(values).any() or torch.isposinf(values).any()
            or not torch.isfinite(values).any()):
        raise ValueError("Invalid native generation logits")
    if request.temperature == 0:
        return int(values.argmax())
    values = values / request.temperature
    if request.top_k:
        cutoff = torch.topk(values, min(request.top_k, len(values))).values[-1]
        values = values.masked_fill(values < cutoff, -torch.inf)
    ordered, indices = torch.sort(values, descending=True)
    cumulative = torch.softmax(ordered, dim=0).cumsum(0)
    remove = cumulative > request.top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    probabilities = torch.softmax(ordered.masked_fill(remove, -torch.inf), dim=0)
    return int(indices[torch.multinomial(probabilities, 1, generator=generator)])


def parameters(temperature=1., top_p=.95, top_k=20):
    return SimpleNamespace(temperature=temperature, top_p=top_p, top_k=top_k)


def unique_logits(width=8192, dtype=torch.float32):
    values = torch.full((width,), -30.)
    values[torch.arange(65) * 7 % width] = torch.linspace(5., 1., 65)
    return values.to(dtype).reshape(1, 1, width)


def compare_reference(monkeypatch, logits, request, seeds=(0, 1, 17, 123, 2**63 - 1)):
    draws = []
    original = torch.multinomial

    def record(probabilities, count, *, generator):
        draws.append(probabilities.clone())
        return original(probabilities, count, generator=generator)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "multinomial", record)
        for seed in seeds:
            before = logits.clone()
            expected_rng = torch.Generator().manual_seed(seed)
            actual_rng = torch.Generator().manual_seed(seed)
            expected = reference_sample(logits, request, expected_rng)
            actual = NativeService._sample(logits, request, actual_rng)
            assert actual == expected and 0 <= actual < logits.shape[-1]
            assert torch.equal(actual_rng.get_state(), expected_rng.get_state())
            assert torch.equal(logits, before)
            assert draws[-1].numel() == logits.shape[-1]
            torch.testing.assert_close(draws[-1], draws[-2], rtol=0, atol=0)
    return draws[-1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("width", [128, 8192])
@pytest.mark.parametrize("top_k", [1, 20, 64])
def test_unique_head_keeps_full_population_probabilities_samples_and_rng(
        monkeypatch, dtype, width, top_k):
    compare_reference(monkeypatch, unique_logits(width, dtype), parameters(top_k=top_k))


def test_fast_path_transfers_only_bounded_head_and_never_sorts_full_vocabulary(monkeypatch):
    logits = unique_logits()
    transferred = []
    cpu = torch.Tensor.cpu
    full = torch.full

    def record_cpu(tensor, *args, **kwargs):
        transferred.append(tensor.numel())
        return cpu(tensor, *args, **kwargs)

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("The untied bounded head must not sort the full vocabulary")

    def cpu_population(shape, *args, **kwargs):
        assert shape == (logits.shape[-1],) and kwargs["device"] == "cpu"
        return full(shape, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    monkeypatch.setattr(torch, "sort", forbidden_sort)
    monkeypatch.setattr(torch, "full", cpu_population)
    NativeService._sample(logits, parameters(), torch.Generator().manual_seed(17))
    assert transferred == [21, 20]


@pytest.mark.parametrize("top_p", [1e-8, .5, .95, 1.])
def test_top_p_extremes_and_renormalization(monkeypatch, top_p):
    probabilities = compare_reference(monkeypatch, unique_logits(), parameters(top_p=top_p))
    assert float(probabilities.sum()) == pytest.approx(1., abs=2e-7)
    assert 1 <= int(probabilities.count_nonzero()) <= 20
    if top_p == 1e-8:
        assert probabilities[0] == 1 and probabilities.count_nonzero() == 1


@pytest.mark.parametrize("rank", [0, 9, 18])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_top_p_exact_float32_boundaries(monkeypatch, rank, direction):
    logits = unique_logits()
    ordered = torch.full((logits.shape[-1],), -torch.inf)
    ordered[:20] = torch.topk(logits.flatten(), 20).values
    boundary = torch.softmax(ordered, 0).cumsum(0)[rank]
    if direction:
        boundary = torch.nextafter(boundary, torch.tensor(float("inf") * direction))
    compare_reference(monkeypatch, logits, parameters(top_p=float(boundary)))


@pytest.mark.parametrize("case", ["head_tie", "conversion_tie", "cutoff_tie", "all_equal", "signed_zero",
                                 "short_finite_head", "disabled", "large", "whole",
                                 "overwhole", "small", "temperature"])
def test_unproven_edges_keep_original_sort_and_tie_population(monkeypatch, case):
    logits, request = unique_logits(), parameters(top_p=1.)
    if case == "head_tie":
        logits[0, 0, 7] = logits[0, 0, 0]
    elif case == "conversion_tie":
        logits = unique_logits(dtype=torch.float64)
        logits[0, 0, 7] = logits[0, 0, 0] - 1e-10
    elif case == "cutoff_tie":
        logits[0, 0, 20 * 7] = logits[0, 0, 19 * 7]
    elif case == "all_equal":
        logits.fill_(0.)
    elif case == "signed_zero":
        logits.fill_(-torch.inf)
        logits[0, 0, :21] = 0.
        logits[0, 0, 0] = -0.
    elif case == "short_finite_head":
        logits.fill_(-torch.inf)
        logits[0, 0, 17] = 1.
    elif case == "disabled":
        request.top_k = 0
    elif case == "large":
        request.top_k = 65
    elif case == "whole":
        request.top_k = logits.shape[-1]
    elif case == "overwhole":
        request.top_k = logits.shape[-1] + 10
    elif case == "small":
        logits = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)
    else:
        request.temperature = .7
    sorts = []
    original = torch.sort

    def record_sort(values, *args, **kwargs):
        sorts.append(values.numel())
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch, "sort", record_sort)
    probabilities = compare_reference(monkeypatch, logits, request)
    assert sorts == [logits.shape[-1]] * 10
    if case == "cutoff_tie":
        assert probabilities.count_nonzero() == 21


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("invalid", ["nan", "positive_inf", "all_negative_inf"])
def test_invalid_logits_reject_without_mutation_or_rng_draw(monkeypatch, dtype, invalid):
    logits = unique_logits(dtype=dtype)
    if invalid == "all_negative_inf":
        logits.fill_(-torch.inf)
    else:
        logits[0, 0, -1] = float("nan" if invalid == "nan" else "inf")
    before = logits.clone()
    for temperature, top_k in ((0., 20), (1., 20), (.7, 20), (1., 0)):
        rng = torch.Generator().manual_seed(123)
        state = rng.get_state()
        with pytest.raises(ValueError, match="Invalid native generation logits"):
            NativeService._sample(logits, parameters(temperature=temperature, top_k=top_k), rng)
        assert torch.equal(rng.get_state(), state)
        torch.testing.assert_close(logits, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(1, 1, 0), (1, 1, 2, 3)])
def test_invalid_sample_shapes_remain_rejected(shape):
    with pytest.raises(ValueError, match="Invalid native generation logits"):
        NativeService._sample(torch.zeros(shape), parameters(), torch.Generator())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_greedy_first_index_ties_do_not_draw_or_mutate(dtype):
    logits = unique_logits(dtype=dtype)
    logits[0, 0, 3] = logits[0, 0, 0]
    before = logits.clone()
    rng = torch.Generator().manual_seed(17)
    state = rng.get_state()
    assert NativeService._sample(logits, parameters(temperature=0.), rng) == 0
    assert torch.equal(state, rng.get_state()) and torch.equal(logits, before)


def test_repeated_samples_keep_seeded_generator_draw_sequence():
    expected_rng, actual_rng = (torch.Generator().manual_seed(17) for _ in range(2))
    for step in range(12):
        logits = unique_logits().roll(step * 31, -1)
        request = parameters()
        assert NativeService._sample(logits, request, actual_rng) == reference_sample(
            logits, request, expected_rng)
        assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


def test_large_strided_logits_keep_reference_population(monkeypatch):
    logits = unique_logits(262144).repeat_interleave(2, -1)[..., ::2].requires_grad_()
    assert not logits.is_contiguous()
    compare_reference(monkeypatch, logits, parameters(), seeds=(0, 17, 2**63 - 1))
    assert logits.requires_grad and logits.grad is None


def test_temperature_scaling_overflow_keeps_original_failure_without_draw():
    logits, request = unique_logits(), parameters(temperature=1e-40)
    before = logits.clone()
    for sample in (reference_sample, NativeService._sample):
        rng = torch.Generator().manual_seed(17)
        state = rng.get_state()
        with pytest.raises(RuntimeError, match="probability tensor"):
            sample(logits, request, rng)
        assert torch.equal(rng.get_state(), state) and torch.equal(logits, before)


def test_float64_to_float32_overflow_remains_invalid():
    logits = unique_logits(dtype=torch.float64)
    logits[0, 0, -1] = 1e300
    for sample in (reference_sample, NativeService._sample):
        with pytest.raises(ValueError, match="Invalid native generation logits"):
            sample(logits, parameters(), torch.Generator())


@pytest.mark.parametrize("case", ["extreme_positive", "subnormal", "narrow_gaps"])
def test_extreme_finite_heads_keep_reference_probabilities_and_draws(monkeypatch, case):
    logits = torch.full((1, 1, 128), -torch.inf)
    info = torch.finfo(torch.float32)
    if case == "extreme_positive":
        head = torch.linspace(.5, 1., 21) * info.max
    elif case == "subnormal":
        head = torch.arange(21, dtype=torch.float32) * (info.tiny * info.eps)
    else:
        head = 1. + torch.arange(21, dtype=torch.float32) * info.eps
    logits[0, 0, :21] = head
    compare_reference(monkeypatch, logits, parameters())


def test_compacting_categorical_population_is_not_a_seeded_equivalence():
    head = torch.linspace(1., .1, 20)
    dense = torch.cat((head, torch.zeros(8192 - len(head))))
    seed = 17
    compact_rng, dense_rng = (torch.Generator().manual_seed(seed) for _ in range(2))
    compact = [int(torch.multinomial(head, 1, generator=compact_rng)) for _ in range(12)]
    full = [int(torch.multinomial(dense, 1, generator=dense_rng)) for _ in range(12)]
    assert compact != full or not torch.equal(compact_rng.get_state(), dense_rng.get_state())
