"""Synthetic late-CAL lifecycle checks; no real records, LLM or GPU."""

import json
from types import SimpleNamespace

import pytest
import torch

from fuzev_research import shared_qualification, shared_train as training


def setup_flow(tmp_path, monkeypatch, *, accepted=True, cache_failure=False):
    pin = 'a' * 64
    monkeypatch.setattr(training, 'FEATURE_SHA', pin)
    monkeypatch.setattr(training, 'file_sha256', lambda path: pin)
    memberships = {role: {'records': 1, 'role': role} for role in ('fit', 'development', 'calibration')}
    roles = {role: [{'role': role}] for role in memberships}
    monkeypatch.setattr(training, 'membership_identity', lambda rows: memberships[rows[0]['role']])
    features = SimpleNamespace(directory=tmp_path, identity={
        'native': '4.05', 'dataset_scope': training.contract(augmentation=True, dimensions=True)},
                               require_ready=lambda mode: {'ready': mode},
                               producer_origins=lambda: {'producer': 'fixed'})
    source = training.source_identity()
    protocol = {'source': source, 'profile': dict(training.PROFILE),
                'optimization': training.optimization_protocol(),
                'feature_identity_sha256': pin, 'roles': memberships}
    selected = {'directory': str(tmp_path / 'checkpoint'), 'sha256': pin,
                'learning_rate': 1e-5, 'group_learning_rates': {'scorer_and_gate': 1e-5, 'query_adapter': 1e-4, 'channel_gate': 1e-4, 'key_adapter': 1e-4},
                'epoch': 1, 'development': {'eligible': True}}
    locked = {'directory': selected['directory'], 'sha256': pin, 'manifest_sha256': pin,
              'selection_role': 'development'}
    trial = {'completed': True, 'status': 'dev_accepted', 'calibration': 'not_started', 'selected': selected}
    for name, data in (('protocol.json', protocol), ('selection-lock.json', locked)):
        (tmp_path / name).write_text(json.dumps(data))
    manifest = {'weights_sha256': pin, 'training_code': source, 'profile': dict(training.PROFILE),
                'optimization': training.optimization_protocol(), 'learning_rate': selected['learning_rate'],
                'group_learning_rates': dict(selected['group_learning_rates']),
                'feature_identity': features.identity, 'preparation': features.require_ready('calibrate'),
                'feature_origins': features.producer_origins(), 'fit_membership': memberships['fit'],
                'development_membership': memberships['development'], 'development': selected['development'],
                'epoch': 1, 'stage': 'mixed', 'shared_training_code': source,
                'operator_sha256': source['shared_train.py'], 'learned_head_count': 1, 'frozen_lm_head_trained': False}
    monkeypatch.setattr(training, 'load_shared_bundle', lambda *a, **kw: (torch.nn.Linear(1, 1), manifest, {}))
    monkeypatch.setattr(shared_qualification, '_gate', lambda *a: None)
    events, cache_calls = [], []
    monkeypatch.setattr(training, 'Store', lambda: SimpleNamespace(update=lambda *a, **kw: events.append(a)))

    class Cache:
        def __init__(self, corpus, rows, role, device):
            cache_calls.append(role)
            assert (tmp_path / 'calibration-attempt.json').exists()
            assert role == 'calibration'
            if cache_failure:
                raise RuntimeError('Synthetic CAL cache failure')
            self.rows = [{'id': p, 'primitive': p} for p in training.PRIMITIVES]
            self.membership = memberships[role]

        def batch(self, rows):
            return {'targets': torch.tensor([[.75, .25]] * len(rows)),
                    'mask': torch.ones(len(rows), 2, dtype=torch.bool)}

        def provenance(self):
            return {'role': 'calibration', 'membership': self.membership}

    monkeypatch.setattr(training, 'SharedFeatureCache', Cache)
    monkeypatch.setattr(training, 'role_weights', lambda rows: {row['id']: 1. for row in rows})
    monkeypatch.setattr(training, 'shared_logits', lambda bundle, batch, primitive: batch['targets'].log())
    monkeypatch.setattr(training, 'fit_shared_temperature', lambda *a: 1.4)
    monkeypatch.setattr(training, 'development', lambda *a: {'eligible': accepted})

    def save(directory, bundle, metadata, temperatures):
        assert metadata['optimization'] == training.optimization_protocol()
        assert metadata['group_learning_rates'] == selected['group_learning_rates']
        assert metadata['learning_rate'] == selected['learning_rate']
        directory.mkdir()
        for name in ('head.safetensors', 'manifest.json', 'calibration.json'):
            (directory / name).write_bytes(b'fixed synthetic artifact')

    monkeypatch.setattr(training, 'save_shared_bundle', save)
    return features, roles, trial, events, cache_calls, manifest


@pytest.mark.parametrize('accepted', [True, False])
def test_completed_and_rejected_calibration_return_and_deny_second_fit(tmp_path, monkeypatch, accepted):
    features, roles, trial, events, calls, _ = setup_flow(tmp_path, monkeypatch, accepted=accepted)
    result = training.calibrate_trial(features, roles, tmp_path, trial, device='cpu', dev=object())
    expected = 'completed' if accepted else 'completed_but_dev_rejected'
    assert result['calibration'] == expected
    assert calls == ['calibration']
    assert events[-1][1] == ('completed' if accepted else 'failed')
    assert (tmp_path / 'reference').exists() is accepted
    outcome = json.loads((tmp_path / 'calibration-outcome.json').read_text())
    assert outcome['status'] == ('completed' if accepted else 'dev_rejected')
    assert outcome['temperatures'] == {p: 1.4 for p in training.PRIMITIVES}
    result['calibration'] = 'not_started'
    with pytest.raises(ValueError, match='unchanged DEV-selected'):
        training.calibrate_trial(features, roles, tmp_path, result, device='cpu', dev=object())
    assert calls == ['calibration']


def test_failure_after_claim_keeps_marker_and_blocks_retry(tmp_path, monkeypatch):
    features, roles, trial, _, calls, _ = setup_flow(tmp_path, monkeypatch, cache_failure=True)
    with pytest.raises(RuntimeError, match='CAL cache failure'):
        training.calibrate_trial(features, roles, tmp_path, trial, device='cpu', dev=object())
    assert calls == ['calibration']
    assert (tmp_path / 'calibration-attempt.json').exists()
    assert not (tmp_path / 'calibration-outcome.json').exists()
    with pytest.raises(ValueError, match='unchanged DEV-selected'):
        training.calibrate_trial(features, roles, tmp_path, trial, device='cpu', dev=object())
    assert calls == ['calibration']


@pytest.mark.parametrize('field', ['feature_identity', 'preparation', 'feature_origins',
                                  'fit_membership', 'development_membership', 'development', 'epoch'])
def test_checkpoint_binding_mutation_blocks_cal_tensors(tmp_path, monkeypatch, field):
    features, roles, trial, _, calls, manifest = setup_flow(tmp_path, monkeypatch)
    manifest[field] = 'changed'
    with pytest.raises(ValueError, match='not bound'):
        training.calibrate_trial(features, roles, tmp_path, trial, device='cpu', dev=object())
    assert calls == []
    assert not (tmp_path / 'calibration-attempt.json').exists()
