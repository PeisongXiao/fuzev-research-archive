"""R17A group controls retained for the fused AdamW implementation."""

import json

import pytest

from fuzev_research import shared_head, shared_train as training
from test_shared_calibration_flow import setup_flow


PROFILE = 'query-channel-rank64-light-consistency-medium'
BASE_NAMES = ('scorer.query.weight', 'scorer.query.bias', 'scorer.key.weight',
              'scorer.key.bias', 'gate.weight', 'gate.bias')
ADAPTER_NAMES = ('adapter_down.weight', 'adapter_down.bias', 'adapter_up.weight')
CHANNEL_NAMES = ('channel_a', 'channel_b', 'channel_c')
PARENT_SOURCE = {
    'shared_training_data.py': '4f9364c5d0a421e75ff3cb22f636ac4cf52ac01fdae3a5c107701893234e82c8',
    'features.py': 'c6583d37f865befeedc55151a9a1b37a9f945563bf9299fa65cb909d986d7579',
    'metrics.py': '78ded3a8847ff3d10ced0fe5452bce72c2000730e48ce52ae73a1ed826380d68',
    'scope.py': '69befb3e286726123c1f56802e9783b28e867a2a9fa612f815c463d9fc79c317',
    'preference_head_retrain.py': '2fcaa6c2c919527cd1164058bb36852fae732202845dfe95453b0ccd579e519b',
}


def small():
    return shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)


@pytest.mark.parametrize('value', [None, False, True, 10, 0., 1., -10., 100., '10', float('nan'), float('inf')])
@pytest.mark.parametrize('field', ['adapter_learning_rate_multiplier', 'channel_gate_learning_rate_multiplier'])
def test_invalid_multiplier_rejected_before_feature_reads_or_output(tmp_path, monkeypatch, value, field):
    monkeypatch.setitem(training.PROFILE, field, value)
    monkeypatch.setattr(training, 'require_feature_pin', lambda *args: pytest.fail('Unexpected feature access'))
    monkeypatch.setattr(training, 'SharedFeatureCache', lambda *args: pytest.fail('Unexpected tensor access'))
    with pytest.raises(ValueError, match='learning-rate multiplier'):
        training.run_trial(None, {}, tmp_path / 'denied', device='cpu', development_only=True)
    assert not (tmp_path / 'denied').exists()


@pytest.mark.parametrize('case', ['missing_group', 'extra_group', 'duplicate_group', 'names', 'multiplier',
                                 'bool_multiplier', 'missing_parameter', 'duplicate_parameter', 'swapped', 'weight_decay'])
@pytest.mark.parametrize('group_index', [1, 2])
def test_schedule_rejects_group_tampering_before_changing_any_lr(case, group_index):
    head = small()
    optimizer = training.make_optimizer(head)
    base, adapter = optimizer.param_groups[0], optimizer.param_groups[group_index]
    if case == 'missing_group':
        optimizer.param_groups.pop()
    elif case == 'extra_group':
        optimizer.param_groups.append(dict(adapter))
    elif case == 'duplicate_group':
        adapter['name'] = base['name']
    elif case == 'names':
        adapter['parameter_names'] = list(BASE_NAMES[:3])
    elif case == 'multiplier':
        adapter['lr_multiplier'] = 1.
    elif case == 'bool_multiplier':
        base['lr_multiplier'] = True
    elif case == 'missing_parameter':
        adapter['params'] = adapter['params'][:-1]
    elif case == 'duplicate_parameter':
        adapter['params'][1] = adapter['params'][0]
    elif case == 'swapped':
        adapter['params'][0], base['params'][0] = base['params'][0], adapter['params'][0]
    else:
        adapter['weight_decay'] = 0.
    before = [group['lr'] for group in optimizer.param_groups]
    with pytest.raises(ValueError, match='optimizer parameter groups|Optimizer groups'):
        training.set_optimizer_learning_rates(optimizer, head, 5)
    assert [group['lr'] for group in optimizer.param_groups] == before


@pytest.mark.parametrize('epoch', [0, -1, True, 1., None, '5'])
def test_invalid_epoch_does_not_touch_groups(epoch):
    head = small()
    optimizer = training.make_optimizer(head)
    before = [group['lr'] for group in optimizer.param_groups]
    with pytest.raises(ValueError, match='one-based epoch'):
        training.set_optimizer_learning_rates(optimizer, head, epoch)
    assert [group['lr'] for group in optimizer.param_groups] == before


@pytest.mark.parametrize('field', ['optimization', 'group_learning_rates', 'learning_rate'])
def test_checkpoint_optimization_provenance_mismatch_denied_before_any_score_access(tmp_path, monkeypatch, field):
    features, roles, trial, _, calls, manifest = setup_flow(tmp_path, monkeypatch)
    manifest[field] = None
    with pytest.raises(ValueError, match='training provenance changed'):
        training.calibrate_trial(features, roles, tmp_path, trial, device='cpu', dev=object())
    assert calls == [] and not (tmp_path / 'calibration-attempt.json').exists()


@pytest.mark.parametrize('location', ['protocol', 'selection'])
def test_changed_group_schedule_receipt_is_rejected_before_any_score_access(tmp_path, monkeypatch, location):
    features, roles, trial, _, calls, _ = setup_flow(tmp_path, monkeypatch)
    if location == 'protocol':
        path = tmp_path / 'protocol.json'
        value = json.loads(path.read_text())
        value['optimization']['parameter_groups'][1]['lr_multiplier'] = 1.
        path.write_text(json.dumps(value))
    else:
        trial['selected']['group_learning_rates']['query_adapter'] = 1e-5
    with pytest.raises(ValueError, match='unchanged DEV-selected|training provenance changed'):
        training.calibrate_trial(features, roles, tmp_path, trial, device='cpu', dev=object())
    assert calls == [] and not (tmp_path / 'calibration-attempt.json').exists()
