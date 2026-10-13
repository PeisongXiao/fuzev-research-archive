"""CUDA-only curriculum overlay; retained model/loss libraries stay unchanged."""

import math


def original_gold_risk(torch, logits, targets, mask):
    """Normalized gold KL on CUDA; only this scalar metric renormalizes roundoff."""
    if (any(not value.is_cuda for value in (logits, targets, mask))
            or logits.ndim != 2 or logits.shape != targets.shape or mask.shape != logits.shape
            or mask.dtype != torch.bool or targets.requires_grad or not bool((mask.sum(1) >= 2).all())):
        raise ValueError('Require frozen original targets and at least two valid CUDA options')
    with torch.no_grad():
        y = targets.double().masked_fill(~mask, 0)
        if not bool(torch.isfinite(y).all() & (y >= 0).all() & (y.sum(1) > 0).all()):
            raise ValueError('Invalid original gold probabilities')
        y = y / y.sum(1, keepdim=True)
        logp = logits.detach().double().masked_fill(~mask, -torch.inf).log_softmax(1).masked_fill(~mask, 0)
        risk = (y * (y.clamp_min(torch.finfo(torch.float64).tiny).log() - logp)).sum(1) / mask.sum(1).double().log()
        if not bool(torch.isfinite(risk).all() & (risk >= -2e-12).all()):
            raise ValueError('Nonfinite or materially negative original-gold KL')
        return risk.clamp_min(0)


def update(torch, prior, current, risk, classification_mask, eta=1., lower=.25, upper=4.):
    """Mirror step plus a prior-relative box KL projection; ordinal mass stays fixed."""
    vectors = (prior, current, risk)
    if (any(not value.is_cuda or value.dtype != torch.float64 or value.shape != (15,)
            or value.requires_grad for value in vectors)
            or not classification_mask.is_cuda or classification_mask.dtype != torch.bool
            or classification_mask.shape != (15,) or classification_mask.sum() != 10
            or len({value.device for value in (*vectors, classification_mask)}) != 1
            or not all(bool(torch.isfinite(value).all()) for value in vectors)
            or not bool((prior > 0).all() & (current > 0).all())):
        raise ValueError('Require fifteen finite, frozen CUDA FP64 masses/risks and ten classification tracks')
    if (type(eta) is not float or not math.isfinite(eta) or eta < 0
            or type(lower) is not float or type(upper) is not float
            or not math.isfinite(lower + upper) or not 0 < lower <= 1 <= upper):
        raise ValueError('Invalid mirror rate or prior-relative mass bounds')
    mask = classification_mask
    budget = prior[mask].sum()
    if (not torch.isclose(prior.sum(), prior.new_tensor(1.), atol=2e-12, rtol=0)
            or not torch.isclose(current[mask].sum(), budget, atol=2e-12, rtol=0)
            or not torch.equal(prior[~mask], current[~mask])
            or not bool((current >= prior * lower - 2e-12).all()
                        & (current <= prior * upper + 2e-12).all())):
        raise ValueError('Current mass must obey the declared budget, ordinal preservation and box')
    with torch.no_grad():
        if eta == 0 or bool((risk[mask] == risk[mask][0]).all()):
            return current.clone()
        score = current[mask].log() + eta * risk[mask]
        score = score - score.max()
        lo, hi = (prior[mask] * lower).log(), (prior[mask] * upper).log()
        left, right = (score - hi).min(), (score - lo).max()
        if not bool(torch.isfinite(left) & torch.isfinite(right)):
            raise ValueError('Mirror projection bracket exceeds finite FP64 arithmetic')
        for _ in range(80):
            midpoint = left + (right - left) / 2
            # Clipping before exp avoids overflow even for strongly separated risks.
            projected = torch.clamp(score - midpoint, min=lo, max=hi).exp()
            too_much = projected.sum() > budget
            left, right = torch.where(too_much, midpoint, left), torch.where(too_much, right, midpoint)
        raw_log_mass = score - (left + (right - left) / 2)
        at_lower, at_upper = raw_log_mass <= lo, raw_log_mass >= hi
        free = ~(at_lower | at_upper)
        masses = torch.where(at_lower, prior[mask] * lower, prior[mask] * upper)
        if bool(free.any()):
            # Polish the free face without subtracting a large shared log normalizer.
            free_risk = risk[mask][free]
            relative = current[mask][free] * (eta * (free_risk - free_risk.max())).exp()
            masses[free] = (budget - masses[~free].sum()) * relative / relative.sum()
        result = current.clone()
        result[mask] = masses
        if (not bool(torch.isfinite(result).all())
                or not torch.isclose(result[mask].sum(), budget, atol=2e-12, rtol=0)
                or not torch.equal(result[~mask], prior[~mask])
                or not bool((result >= prior * lower - 2e-12).all() & (result <= prior * upper + 2e-12).all())):
            raise ValueError('CUDA mass projection failed its exact-budget/box/ordinal contract')
        return result
