import pytest
import torch

from fuzev_research.shared_train import PROFILE, correction_penalty, decision_loss


def test_penalty_counts_only_supplied_candidates_and_preserves_gradient():
    baseline = torch.tensor([[2., -1., 9000.], [1., 1., 1.]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    logits = torch.tensor([[2.2, -1.2, -1e9], [1.1, 1., .9]], requires_grad=True)
    penalty = correction_penalty(logits, {'baseline': baseline, 'mask': mask})
    assert torch.allclose(penalty, torch.tensor([.04, .02 / 3]), atol=1e-7)
    penalty.sum().backward()
    assert logits.grad[0, 2].item() == 0
    assert logits.grad[0, 0].item() != 0
    assert baseline.grad is None


def test_penalty_rejects_trainable_lm_baseline():
    baseline = torch.ones(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match='frozen LM'):
        correction_penalty(baseline, {'baseline': baseline, 'mask': torch.ones(2, 3, dtype=torch.bool)})


def test_combined_loss_uses_labels_and_soft_regularizer_not_teacher_targets():
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[1., -1., -1e9]], requires_grad=True)
    batch = {'baseline': torch.tensor([[.5, -.5, 0.]]), 'mask': mask,
             'targets': torch.tensor([[1., 0., 0.]])}
    ce = decision_loss(logits, batch, ['noul'])
    combined = ce + PROFILE['correction_regularization'] * correction_penalty(logits, batch)
    assert torch.allclose(combined - ce, torch.tensor([.00025]), atol=1e-7)
    combined.sum().backward()
    assert logits.grad[0, :2].abs().sum().item() > 0
    assert logits.grad[0, 2].item() == 0
