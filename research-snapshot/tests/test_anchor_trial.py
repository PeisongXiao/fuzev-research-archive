"""Synthetic checks of the locked FIT/DEV-only trial protocol."""

from types import SimpleNamespace

import pytest
import torch

from fuzev_research.anchor_trial import COMMON, PROFILE, development, fitting_weights, track_masses
from fuzev_research.preference_head_retrain import track


def rows(primitive, role):
    result = []
    for key in sorted(track_masses(primitive)):
        source, _, modality, adaptation = key.split('/')
        for index in range(8):
            row = {'id': f'{role}-{key}-{index}', 'group_': f'{key}-{index}', 'role': role,
                   'primitive': primitive, 'source': source, 'modality': modality}
            if adaptation != 'main':
                row['adaptation'] = adaptation
            result.append(row)
    return result


@pytest.mark.parametrize('primitive', ['choice', 'noul'])
def test_track_and_group_loss_masses(primitive):
    fitting = rows(primitive, 'fit')
    duplicate = {**fitting[0], 'id': fitting[0]['id'] + '-expanded'}
    fitting.append(duplicate)
    weights = fitting_weights(fitting, primitive)
    for key, mass in track_masses(primitive).items():
        assert sum(weights[row['id']] for row in fitting if track(row) == key) == pytest.approx(len(fitting) * mass)
    assert weights[fitting[0]['id']] == weights[duplicate['id']]
    assert weights[fitting[0]['id']] * 2 == pytest.approx(weights[fitting[1]['id']])


def test_missing_track_and_test_training_rejected():
    fitting = rows('noul', 'fit')
    with pytest.raises(ValueError, match='All predeclared'):
        fitting_weights(fitting[8:], 'noul')
    with pytest.raises(ValueError, match='Invalid role'):
        fitting_weights(rows('noul', 'test'), 'noul')


class Cache:
    def __init__(self):
        self.role = 'development'
        self.rows = rows('noul', self.role)
        self.membership = {'records': len(self.rows), 'ordered_sha256': 'c' * 64}
        self.identity_sha256 = 'c' * 64

    def batch(self, selected):
        count = len(selected)
        return {'query': torch.ones(count, 4), 'candidates': torch.ones(count, 2, 4),
                'mask': torch.ones(count, 2, dtype=torch.bool), 'baseline': torch.zeros(count, 2),
                'targets': torch.tensor([[1., 0.]] * count)}


class Head:
    primitive = 'noul'
    residual_bound = .25
    anchor = SimpleNamespace(weights_sha256='a' * 64, producer_identity_sha256='b' * 64)

    def __init__(self, delta):
        self.delta = delta

    def eval(self):
        return self

    def forward_cached(self, query, candidates, mask, *, baseline, **kwargs):
        return baseline + torch.tensor([self.delta, -self.delta])

    def residual(self, query, candidates):
        return torch.tensor([self.delta, -self.delta])[None].expand(len(query), -1)

    def margin_diagnostics(self, query, candidates, mask):
        return {'max_candidate_margin': 2 * abs(self.delta),
                'nonzero_centered_candidates': int(mask.sum()) if self.delta else 0}


def test_control_only_not_eligible():
    result = development(Head(0.), Cache())
    assert result['eligible'] is False
    assert result['gain_vs_anchor_only'] == 0


def test_actual_residual_gain_with_retention_is_eligible():
    result = development(Head(.1), Cache())
    assert result['eligible'] is True
    assert result['gain_vs_anchor_only'] > COMMON['minimum_development_gain']
    assert result['learned_margin']['max_candidate_margin'] > 0
    assert all(check['passed'] for check in result['checks'])


def test_adverse_accuracy_and_nll_rejected():
    result = development(Head(-.1), Cache())
    assert result['eligible'] is False
    assert result['learned_margin']['argmax_changes_vs_anchor'] == 32


def test_selection_cannot_use_other_role():
    cache = Cache()
    cache.role = 'calibration'
    with pytest.raises(ValueError, match='DEV only'):
        development(Head(.1), cache)


def test_single_profile_is_locked():
    assert PROFILE == {'choice': {'seed': 29, 'max_epochs': 3}, 'noul': {'seed': 43, 'max_epochs': 5}}
    assert COMMON['residual_bound'] == .25 and COMMON['bootstrap_draws'] == 2000
