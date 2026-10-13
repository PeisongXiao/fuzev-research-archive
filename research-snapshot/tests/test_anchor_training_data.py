"""Synthetic guards for selected training-role caching; no actual corpus."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fuzev_research.anchor_training_data import OwnedFeatureCache


def fixture(tmp_path):
    (tmp_path / 'identity.json').write_text('{}')
    row = {'id': 'fit-a', 'role': 'fit', 'primitive': 'noul', 'modality': 'text',
           'source': 'boolq', 'group_': 'group-a', 'shard': 0, 'row_': 0}
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE records(id,role,primitive,source,group_,shard,row_,metadata)')
    db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',
               (row['id'], row['role'], row['primitive'], row['source'], row['group_'], 0, 0,
                json.dumps({'has_baseline': True, 'candidates': ['yes', 'no']})))
    tensors = {'query': torch.ones(2, 2560), 'candidates': torch.ones(2, 2, 2560),
               'mask': torch.ones(2, 2, dtype=torch.bool), 'targets': torch.tensor([[1., 0.], [0., 1.]]),
               'baseline': torch.tensor([[.5, -.5], [.8, -.8]])}
    # This unselected heldout-like row must not be read or used to reject the FIT cache.
    tensors['query'][1] = torch.nan
    save_file(tensors, str(tmp_path / 'shard-000000.safetensors'))
    return SimpleNamespace(directory=tmp_path, read_only=True, db=db), row, tensors


def test_owned_slices_only_and_exact_batch(tmp_path):
    features, row, _ = fixture(tmp_path)
    cache = OwnedFeatureCache(features, [row], 'fit', 'noul', device='cpu')
    batch = cache.batch([row])
    assert torch.isfinite(batch['query']).all()
    assert batch['baseline'].tolist() == [[.5, -.5]]
    assert cache.provenance()['membership']['records'] == 1
    with pytest.raises(ValueError, match='Batch does not belong'):
        cache.batch([{**row, 'id': 'other'}])
    with pytest.raises(ValueError, match='Batch does not belong'):
        cache.batch([{**row, 'row_': 1}])


@pytest.mark.parametrize('role,source', [('test', 'boolq'), ('fit', 'qasper'), ('fit', 'vl-rewardbench')])
def test_external_or_test_rejected_before_reads(tmp_path, role, source):
    features, row, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='training-source|Invalid role'):
        OwnedFeatureCache(features, [{**row, 'role': role, 'source': source}], role, 'noul', device='cpu')


def test_native_control_required(tmp_path):
    features, row, _ = fixture(tmp_path)
    features.db.execute('UPDATE records SET metadata=?', (json.dumps({'has_baseline': False, 'candidates': ['yes', 'no']}),))
    with pytest.raises(ValueError, match='Missing native control'):
        OwnedFeatureCache(features, [row], 'fit', 'noul', device='cpu')


def test_caller_mutation_cannot_relabel_cached_tensors(tmp_path):
    features, row, _ = fixture(tmp_path)
    cache = OwnedFeatureCache(features, [row], 'fit', 'noul', device='cpu')
    row['row_'] = 1
    with pytest.raises(ValueError, match='Batch does not belong'):
        cache.batch([row])
    assert cache.rows[0]['row_'] == 0


def test_fit_and_dev_diagnostic_samples_cannot_be_swapped(tmp_path):
    features, fitting, tensors = fixture(tmp_path)
    development = {**fitting, 'id': 'dev-b', 'role': 'development', 'group_': 'group-b', 'row_': 1}
    features.db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',
                       (development['id'], 'development', 'noul', 'boolq', 'group-b', 0, 1,
                        json.dumps({'has_baseline': True, 'candidates': ['yes', 'no']})))
    tensors['query'][1] = 1
    save_file(tensors, str(tmp_path / 'shard-000000.safetensors'))
    fit_cache = OwnedFeatureCache(features, [fitting], 'fit', 'noul', device='cpu')
    dev_cache = OwnedFeatureCache(features, [development], 'development', 'noul', device='cpu')
    with pytest.raises(ValueError, match='Invalid role'):
        fit_cache.batch(dev_cache.rows)
    with pytest.raises(ValueError, match='Invalid role'):
        dev_cache.batch(fit_cache.rows)
    assert fit_cache.batch(fit_cache.rows)['targets'].tolist() == [[1., 0.]]


@pytest.mark.parametrize('failure', ['mask', 'targets', 'nonfinite', 'shape'])
def test_invalid_selected_tensor_rejected(tmp_path, failure):
    features, row, tensors = fixture(tmp_path)
    if failure == 'mask':
        tensors['mask'][0, 0] = False
    elif failure == 'targets':
        tensors['targets'][0] = 0
    elif failure == 'nonfinite':
        tensors['baseline'][0, 0] = torch.nan
    else:
        tensors['candidates'] = torch.ones(2, 2, 2559)
    save_file(tensors, str(tmp_path / 'shard-000000.safetensors'))
    with pytest.raises(ValueError, match='Invalid owned|Nonfinite owned'):
        OwnedFeatureCache(features, [row], 'fit', 'noul', device='cpu')
