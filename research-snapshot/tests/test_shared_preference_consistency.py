"""Synthetic FIT-only equivalence and loss-normalization checks."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from fuzev_research import shared_head, shared_train as training


def fixtures(permutation=(1, 0), target=(1., 0.)):
    parent = dict(id='parent', role='fit', primitive='choice', source='helpsteer2',
                  group_='group', modality='text', adaptation=None, attribute=None)
    child = dict(parent, id='child', primitive='noul', adaptation='pairwise_preference_yes_no')
    original = {**parent, 'group': 'group', 'candidates': [{'id': 'a'}, {'id': 'b'}], 'target': list(target)}
    metadata = {**child, 'group': 'group', 'candidates': [{'id': 'yes'}, {'id': 'no'}],
                'target': [target[index] for index in permutation], 'parent_id': 'parent',
                'parent_role': 'fit', 'parent_source': 'helpsteer2', 'parent_primitive': 'choice',
                'candidate_permutation': list(permutation), 'parent_candidate_ids': ['a', 'b'],
                'parent_target': list(target)}
    return [parent, child], {'parent': original, 'child': metadata}


class Features:
    read_only = True

    def __init__(self, metadata):
        self.metadata, self.queries, self.db = metadata, [], self

    def execute(self, sql, parameters):
        assert sql == 'SELECT metadata FROM records WHERE id=? AND role=?'
        assert parameters[1] == 'fit'
        self.queries.append(parameters)
        value = self.metadata.get(parameters[0])
        return SimpleNamespace(fetchone=lambda: {'metadata': json.dumps(value)} if value else None)


@pytest.mark.parametrize('permutation', [(0, 1), (1, 0)])
@pytest.mark.parametrize('target', [(1., 0.), (0., 1.), (.5, .5)])
def test_exact_parent_child_orientation_and_ties(permutation, target):
    rows, metadata = fixtures(permutation, target)
    features = Features(metadata)
    pairs = training.fit_preference_pairs(features, rows)
    assert pairs == {'child': {'parent': rows[0], 'child': rows[1], 'permutation': list(permutation)}}
    assert features.queries == [('child', 'fit'), ('parent', 'fit')]


def test_one_contribution_per_group_is_stable_under_input_order():
    rows, metadata = fixtures()
    second = dict(rows[1], id='z-child')
    metadata['z-child'] = {**metadata['child'], 'id': 'z-child'}
    for ordered in (rows + [second], [second, *reversed(rows)]):
        pairs = training.fit_preference_pairs(Features(metadata), ordered)
        assert list(pairs) == ['child']


@pytest.mark.parametrize('bad', ['role', 'readonly', 'parent_missing', 'parent_role', 'parent_source',
                                 'parent_primitive', 'permutation', 'bool_index', 'group', 'target',
                                 'parent_target', 'candidate_ids', 'dimension_parent', 'external'])
def test_malformed_or_nonequivalent_views_are_denied(bad):
    rows, metadata = fixtures()
    features = Features(metadata)
    if bad == 'role':
        rows[1]['role'] = 'development'
    elif bad == 'readonly':
        features.read_only = False
    elif bad == 'parent_missing':
        del metadata['parent']
    elif bad in {'parent_role', 'parent_source', 'parent_primitive'}:
        metadata['child'][bad] = 'wrong'
    elif bad == 'permutation':
        metadata['child']['candidate_permutation'] = [0, 0]
    elif bad == 'bool_index':
        metadata['child']['candidate_permutation'] = [True, False]
    elif bad == 'group':
        metadata['child']['group'] = 'foreign'
    elif bad == 'target':
        metadata['child']['target'] = [1., 0.]
    elif bad == 'parent_target':
        metadata['child']['parent_target'] = [.5, .5]
    elif bad == 'candidate_ids':
        metadata['child']['parent_candidate_ids'] = ['b', 'a']
    elif bad == 'dimension_parent':
        rows[0]['adaptation'] = 'correctness_pairwise_choice'
    else:
        rows[1]['evaluation_scope'] = 'external'
    with pytest.raises(ValueError):
        training.fit_preference_pairs(features, rows)


def test_boolq_dimension_and_ordinal_rows_do_not_become_pairs():
    rows, metadata = fixtures()
    ignored = [dict(rows[0], id='dimension', adaptation='correctness_pairwise_choice'),
               dict(rows[0], id='ordinal', primitive='ordinal', attribute='correctness'),
               dict(rows[1], id='boolq', source='boolq', adaptation='boolean_truth')]
    features = Features(metadata)
    assert list(training.fit_preference_pairs(features, rows + ignored)) == ['child']
    assert len(features.queries) == 2


def test_js_exact_formula_permutation_symmetry_and_gradients():
    parent = torch.tensor([[.7, -.2], [-.4, .8]], requires_grad=True)
    child = torch.tensor([[.4, -.6], [.3, -.5]], requires_grad=True)
    order = torch.tensor([[1, 0], [0, 1]])
    actual = training.preference_js(parent, child, order, role='fit')
    p, q = parent.gather(1, order).double().softmax(1), child.double().softmax(1)
    m = (p + q) / 2
    expected = ((p * (p.log() - m.log())).sum(1) + (q * (q.log() - m.log())).sum(1)) / (2 * math.log(2))
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-6)
    reverse = training.preference_js(child, parent, order, role='fit')
    torch.testing.assert_close(actual, reverse)
    actual.sum().backward()
    for value in (parent, child):
        assert torch.isfinite(value.grad).all() and (value.grad != 0).any()


@pytest.mark.parametrize('scores', [[[0., 0.]], [[1e4, -1e4]], [[.2, -.7]]])
def test_identical_or_permuted_scores_have_zero_js(scores):
    parent = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    child = parent.detach().flip(1).requires_grad_(True)
    loss = training.preference_js(parent, child, torch.tensor([[1, 0]]), role='fit')
    torch.testing.assert_close(loss, torch.zeros(1), atol=2e-7, rtol=0)
    loss.sum().backward()
    assert torch.isfinite(parent.grad).all() and torch.isfinite(child.grad).all()


@pytest.mark.parametrize('role', ['development', 'calibration', 'test', 'external', None])
def test_nonfit_denied_before_scores_or_cache_access(role):
    with pytest.raises(ValueError, match='FIT only'):
        training.preference_js(None, None, None, role=role)
    with pytest.raises(ValueError, match='FIT only'):
        training.batch_preference_consistency(None, SimpleNamespace(role=role), [], {})


@pytest.mark.parametrize('bad', ['nan', 'inf', 'dtype', 'shape', 'permutation'])
def test_invalid_js_scores_or_permutation(bad):
    parent, child = torch.zeros(2, 2), torch.ones(2, 2)
    permutation = torch.tensor([[0, 1], [1, 0]])
    if bad in {'nan', 'inf'}:
        parent[0, 0] = float(bad)
    elif bad == 'dtype':
        child = child.double()
    elif bad == 'shape':
        child = child[:, :1]
    else:
        permutation[0] = 0
    with pytest.raises(ValueError):
        training.preference_js(parent, child, permutation, role='fit')


def test_batch_regularizer_reaches_same_shared_parameters_without_gold_replay():
    rows, metadata = fixtures()
    pairs = training.fit_preference_pairs(Features(metadata), rows)
    tensors = {row['id']: {'query': torch.randn(1, 4), 'candidates': torch.randn(1, 2, 4),
                          'mask': torch.ones(1, 2, dtype=torch.bool),
                          'targets': torch.tensor([metadata[row['id']]['target']]),
                          'baseline': torch.zeros(1, 2)} for row in rows}
    cache = SimpleNamespace(role='fit', data={'query': tensors['parent']['query']},
                            batch=lambda batch: {key: torch.cat([tensors[row['id']][key] for row in batch])
                                                 for key in tensors['parent']})
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    loss, visited = training.batch_preference_consistency(head, cache, [rows[1]], pairs)
    assert visited == ['child']
    loss.backward()
    assert head.scorer.query.weight.grad.abs().sum() > 0
    assert head.scorer.key.weight.grad.abs().sum() > 0
    empty, visited = training.batch_preference_consistency(head, cache, [rows[0]], pairs)
    assert visited == [] and empty.item() == 0


def test_epoch_scaling_is_exact_mean_js_independent_of_batch_placement():
    beta, records, batch_size = .05, 7, 3
    values = torch.tensor([.2, .6])
    batches = [values[:1], values.new_empty(0), values[1:]]
    contribution = sum(beta * records / (len(values) * batch_size) * batch.sum() for batch in batches)
    torch.testing.assert_close(contribution * batch_size / records, beta * values.mean())
    profile = training.BASE_PROFILE | training.SEARCH_PROFILES['key-adapter-ordinal-rps015-light-consistency-medium']
    assert profile['teacher_distribution_regularization'] == .03 and profile['preference_consistency_beta'] == .05
    assert profile['learning_rate'] == 1e-5 and profile['seed'] == 43
    assert shared_head.SCHEMA_VERSION == 25 and len(list(shared_head.SharedDecisionHead(seed=43).parameters())) == 15
