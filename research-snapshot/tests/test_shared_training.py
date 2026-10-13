"""Synthetic split, weighting, and joint-loss checks; no real data or GPU."""

from collections import defaultdict
import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fuzev_research.preference_head_retrain import track
from fuzev_research.shared_train import (PRIMITIVES, decision_loss, development, fitting_weights,
                                fit_shared_temperature, role_weights, track_masses)
from fuzev_research.shared_training_data import SharedFeatureCache


def fixture(tmp_path, primitive='ordinal', role='fit'):
    (tmp_path / 'identity.json').write_text('{}')
    count = 5 if primitive == 'ordinal' else 2
    row = {'id': role + '-a', 'role': role, 'primitive': primitive, 'modality': 'text',
           'source': 'helpsteer2' if primitive == 'ordinal' else 'boolq',
           'group_': role + '-group-a', 'shard': 0, 'row_': 0}
    candidates = [{'id': str(i), 'text': str(i), 'ordinal_value': i} for i in range(count)]
    meta = {**row, 'group': row['group_'], 'has_baseline': True,
            'candidates': candidates, 'target': [1.] + [0.] * (count - 1)}
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE records(id,role,primitive,source,group_,shard,row_,metadata)')
    db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',
               tuple(row[key] for key in ('id', 'role', 'primitive', 'source', 'group_', 'shard', 'row_')) + (json.dumps(meta),))
    tensors = {'query': torch.ones(2, 2560), 'candidates': torch.ones(2, count, 2560),
               'mask': torch.ones(2, count, dtype=torch.bool),
               'targets': torch.tensor([meta['target']] * 2), 'baseline': torch.zeros(2, count),
               'ordinal_values': torch.arange(count).float().repeat(2, 1)}
    tensors['query'][1] = torch.nan
    save_file(tensors, str(tmp_path / 'shard-000000.safetensors'))
    return SimpleNamespace(directory=tmp_path, read_only=True, db=db), row, tensors, meta


def test_owned_ordinal_slices_and_zero_level(tmp_path):
    features, row, _, _ = fixture(tmp_path)
    cache = SharedFeatureCache(features, [row], 'fit', 'cpu')
    assert cache.batch([row])['ordinal_values'].tolist() == [[0., 1., 2., 3., 4.]]
    assert torch.isfinite(cache.data['query']).all()
    assert cache.provenance()['tensor_access'] == 'owned row slices only'
    row['row_'] = 1
    with pytest.raises(ValueError, match='pinned role cache'):
        cache.batch([row])


@pytest.mark.parametrize('failure', ['targets', 'ordinal', 'mask', 'baseline'])
def test_metadata_tensor_binding(tmp_path, failure):
    features, row, tensors, _ = fixture(tmp_path)
    if failure == 'targets':
        tensors['targets'][0] = torch.tensor([0., 1., 0., 0., 0.])
    elif failure == 'ordinal':
        tensors['ordinal_values'][0, 0] = -1
    elif failure == 'mask':
        tensors['mask'][0, 0] = False
    else:
        tensors['baseline'][0, 0] = torch.nan
    save_file(tensors, str(tmp_path / 'shard-000000.safetensors'))
    with pytest.raises(ValueError):
        SharedFeatureCache(features, [row], 'fit', 'cpu')


@pytest.mark.parametrize('role,source', [('test', 'boolq'), ('fit', 'qasper'), ('fit', 'vl-rewardbench')])
def test_external_test_denied_before_read(tmp_path, role, source):
    features, row, _, _ = fixture(tmp_path, primitive='noul')
    with pytest.raises(ValueError, match='training-source|Invalid role'):
        SharedFeatureCache(features, [{**row, 'role': role, 'source': source}], role, 'cpu')


def rows(role):
    result = []
    for key in sorted(track_masses()):
        source, primitive, modality, subtype = key.split('/')
        for index in range(4):
            row = {'id': f'{role}-{key}-{index}', 'group_': f'{role}-{key}-{index}', 'role': role,
                   'primitive': primitive, 'source': source, 'modality': modality}
            if primitive == 'ordinal':
                row['attribute'] = subtype
            elif subtype != 'main':
                row['adaptation'] = subtype
            result.append(row)
    return result


def test_fit_macro_primitive_budgets_and_group_expansion_invariance():
    fitting = rows('fit')
    child = {**fitting[0], 'id': fitting[0]['id'] + '-view'}
    fitting.append(child)
    weights = fitting_weights(fitting)
    masses = defaultdict(float)
    for row in fitting:
        masses[row['primitive']] += weights[row['id']]
    assert set(masses) == set(PRIMITIVES)
    expected = {'choice': 6 / 15, 'noul': 4 / 15, 'ordinal': 5 / 15}
    assert all(value == pytest.approx(len(fitting) * expected[primitive]) for primitive, value in masses.items())
    for key, fraction in track_masses().items():
        assert sum(weights[row['id']] for row in fitting if track(row) == key) == pytest.approx(len(fitting) * fraction)
    assert weights[fitting[0]['id']] == weights[child['id']]
    assert 2 * weights[fitting[0]['id']] == pytest.approx(weights[fitting[1]['id']])
    assert sum(role_weights(rows('calibration')).values()) == pytest.approx(len(rows('calibration')))


def test_missing_track_or_test_fit_rejected():
    with pytest.raises(ValueError):
        fitting_weights(rows('fit')[4:])
    with pytest.raises(ValueError):
        fitting_weights(rows('test'))


def test_all_tasks_use_same_normalized_soft_target_loss_and_gradients():
    logits = torch.tensor([[.5, -.5, -torch.inf]] * 3, requires_grad=True)
    batch = {'mask': torch.tensor([[True, True, False]] * 3),
             'targets': torch.tensor([[.25, .75, 0.]] * 3)}
    loss = decision_loss(logits, batch, list(PRIMITIVES))
    assert torch.equal(loss, loss[0].expand(3))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad[:, :2] != 0).all()
    assert (logits.grad[:, 2] == 0).all()


def test_calibration_mask_and_positive_scalar():
    logits = torch.tensor([[2., -2., -torch.inf], [-2., 2., -torch.inf]])
    target = torch.tensor([[.8, .2, 0.], [.2, .8, 0.]])
    mask = torch.tensor([[True, True, False]] * 2)
    t = fit_shared_temperature(logits, target, mask, torch.ones(2))
    assert .03 <= t <= 100 and torch.isfinite(torch.tensor(t))
    with pytest.raises(ValueError):
        fit_shared_temperature(logits, target, mask, torch.zeros(2))


def test_selection_cannot_read_cal_or_test():
    for role in ('calibration', 'test'):
        with pytest.raises(ValueError, match='DEV only'):
            development(None, SimpleNamespace(role=role))
