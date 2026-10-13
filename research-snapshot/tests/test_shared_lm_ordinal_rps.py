"""R41 ordinal auxiliary: random CPU tensors and exclusively owned fixtures."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fuzev_research import shared_train as training
from fuzev_research.shared_training_data import SharedFeatureCache
from test_shared_training import fixture as owned_fixture


def batch_for(count=5, width=8, dtype=torch.float32):
    generator = torch.Generator().manual_seed(43)
    logits = torch.randn(3, width, generator=generator, dtype=dtype).requires_grad_()
    mask = torch.arange(width)[None, :].expand(3, -1) < count
    target = torch.zeros_like(logits)
    target[:, :count] = 1 / count
    target[0, :count] = 0
    target[0, 0] = 1
    return logits, {'mask': mask, 'targets': target, 'baseline': torch.zeros_like(logits)}


@pytest.mark.parametrize('count', [2, 5, 16])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_formula_valid_thresholds_padding_soft_ties_and_detached_gold(count, dtype):
    logits, batch = batch_for(count, count + 3, dtype)
    with torch.no_grad():
        logits[~batch['mask']] = torch.nan
        batch['targets'][~batch['mask']] = torch.nan
    batch['targets'].requires_grad_()
    p = logits[:, :count].softmax(dim=1)
    q = batch['targets'][:, :count].detach()
    expected = (p - q).cumsum(dim=1)[:, :-1].square().mean(dim=1)
    actual = training.ordinal_cdf_penalty(logits, batch, ['ordinal'] * 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2 * torch.finfo(dtype).eps)
    expected_gradient, = torch.autograd.grad(expected.sum(), logits, retain_graph=True)
    gradient, target_gradient = torch.autograd.grad(actual.sum(), (logits, batch['targets']), allow_unused=True)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert target_gradient is None and torch.isfinite(gradient).all() and not gradient[~batch['mask']].any()
    assert ((actual >= 0) & (actual <= 1)).all()


@pytest.mark.parametrize('gold', [[.5, .5, 0., 0., 0.], [.2] * 5, [0., 0., 1., 0., 0.]])
def test_zero_when_distribution_matches_original_soft_gold(gold):
    q = torch.tensor([gold])
    logits = q.log().requires_grad_()
    batch = {'mask': torch.ones_like(q, dtype=torch.bool), 'targets': q.requires_grad_()}
    loss = training.ordinal_cdf_penalty(logits, batch, ['ordinal'])
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=1e-14)
    gradient, = torch.autograd.grad(loss.sum(), logits)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-7)


def test_far_wrong_level_costs_more_than_near_wrong_level_without_changing_gold():
    q = torch.tensor([[0., 0., 1., 0., 0.]])
    batch = {'mask': torch.ones_like(q, dtype=torch.bool), 'targets': q}
    near, far = torch.tensor([[-100., -100., -100., 100., -100.]]), torch.tensor([[100., -100., -100., -100., -100.]])
    assert training.ordinal_cdf_penalty(near, batch, ['ordinal']).item() == .25
    assert training.ordinal_cdf_penalty(far, batch, ['ordinal']).item() == .5
    assert torch.equal(q, batch['targets'])


def test_nonordinal_zero_row_permutation_and_extreme_finite_gradients():
    logits, batch = batch_for()
    with torch.no_grad():
        logits[:, :5] = torch.tensor([[1e30, -1e30, 0., 0., 0.]] * 3)
    primitives = ['choice', 'ordinal', 'noul']
    loss = training.ordinal_cdf_penalty(logits, batch, primitives)
    assert loss[0] == loss[2] == 0 and torch.isfinite(loss).all()
    gradient, = torch.autograd.grad(loss.sum(), logits)
    assert torch.isfinite(gradient).all() and not gradient[[0, 2]].any()
    permutation = torch.tensor([2, 0, 1])
    permuted = training.ordinal_cdf_penalty(logits[permutation], {key: value[permutation] for key, value in batch.items()},
                                         [primitives[i] for i in permutation])
    assert torch.equal(permuted, loss[permutation])


def test_loss_decomposition_unit_gold_and_rps_logging_components_use_same_vector():
    logits, batch = batch_for()
    primitives = list(training.PRIMITIVES)
    gold = training.decision_loss(logits.masked_fill(~batch['mask'], -torch.inf), batch, primitives)
    masked_logits = logits.masked_fill(~batch['mask'], -torch.inf)
    teacher = training.teacher_distribution_penalty(masked_logits, batch)
    correction = training.correction_penalty(masked_logits, batch)
    rps = training.ordinal_cdf_penalty(masked_logits, batch, primitives)
    loss, component = training._fitting_loss_components(masked_logits, batch, primitives, role='fit')
    expected = gold + .001 * correction + .03 * teacher + .15 * rps
    assert torch.equal(component, rps) and torch.equal(loss, expected)
    assert torch.equal(training.fitting_loss(masked_logits, batch, primitives, role='fit'), loss)
    expected_gradient, = torch.autograd.grad(expected.sum(), logits, retain_graph=True)
    actual_gradient, = torch.autograd.grad(loss.sum(), logits)
    assert torch.equal(expected_gradient, actual_gradient)


@pytest.mark.parametrize('role', ['development', 'calibration', 'test', 'external', None, True])
def test_loss_role_guard_before_arithmetic(role):
    with pytest.raises(ValueError, match='FIT-only'):
        training.fitting_loss(None, {}, [], role=role)


def test_loss_role_is_required():
    with pytest.raises(TypeError):
        training.fitting_loss(None, {}, [])


@pytest.mark.parametrize('coefficient', [0., .01, .05, .1, True, False, None, '0.15', float('nan')])
def test_fixed_coefficient_guard_before_output_state_cache_or_loss(monkeypatch, tmp_path, coefficient):
    monkeypatch.setitem(training.PROFILE, 'ordinal_rps_coefficient', coefficient)
    monkeypatch.setattr(training, 'Store', lambda: pytest.fail('Invalid profile reached state'))
    monkeypatch.setattr(training, 'require_feature_pin', lambda *a: pytest.fail('Invalid profile reached features'))
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='RPS coefficient'):
        training.run_trial(None, {}, output, device='cpu', development_only=True)
    with pytest.raises(ValueError, match='RPS coefficient'):
        training.fitting_loss(None, {}, [], role='fit')
    assert not output.exists()


@pytest.mark.parametrize('alpha', [0., .05, None])
def test_smoothing_not_silently_combined(monkeypatch, alpha):
    monkeypatch.setitem(training.PROFILE, 'gold_target_smoothing', alpha)
    with pytest.raises(ValueError, match='unsmoothed'):
        training.optimization_protocol()


def test_real_owned_cpu_cache_validates_canonical_levels_without_mutation(tmp_path):
    features, row, _, _ = owned_fixture(tmp_path)
    cache = SharedFeatureCache(features, [row], 'fit', 'cpu')
    before = {name: value.clone() for name, value in cache.data.items()}
    provenance = copy.deepcopy(cache.provenance())
    training.validate_fit_ordinal_cache(cache)
    cache.data = {name: value.to('cpu') for name, value in cache.data.items()}
    assert cache.provenance() == provenance
    assert all(torch.equal(cache.data[name], value) for name, value in before.items())
    features.db.close()


@pytest.mark.parametrize('bad', ['role', 'four', 'six', 'hole', 'descending', 'shifted', 'gap', 'nan', 'dtype', 'missing', 'device'])
def test_canonical_level_cpu_admission_refuses_wrong_order_or_mask(bad):
    cache = SimpleNamespace(role='fit', rows=[{'primitive': 'ordinal'}], data={
        'mask': torch.tensor([[True] * 5 + [False]]), 'ordinal_values': torch.tensor([[0., 1., 2., 3., 4., 0.]])})
    if bad == 'role':
        cache.role = 'development'
    elif bad == 'four':
        cache.data['mask'][0, 4] = False
    elif bad == 'six':
        cache.data['mask'][0, 5] = True
    elif bad == 'hole':
        cache.data['mask'][0, 1] = False
    elif bad == 'descending':
        cache.data['ordinal_values'][0, :5] = torch.arange(4, -1, -1)
    elif bad == 'shifted':
        cache.data['ordinal_values'][0, :5] += 1
    elif bad == 'gap':
        cache.data['ordinal_values'][0, 4] = 5
    elif bad == 'nan':
        cache.data['ordinal_values'][0, 0] = torch.nan
    elif bad == 'dtype':
        cache.data['ordinal_values'] = cache.data['ordinal_values'].double()
    elif bad == 'device':
        cache.data['ordinal_values'] = cache.data['ordinal_values'].to('meta')
    else:
        cache.rows[0]['primitive'] = 'choice'
    with pytest.raises(ValueError, match='Canonical'):
        training.validate_fit_ordinal_cache(cache)


def test_canonical_guard_adds_exact_scale_check_after_all_original_owned_cache_guards(tmp_path):
    import json
    features, row, tensors, metadata = owned_fixture(tmp_path)
    for candidate in metadata['candidates']:
        candidate['ordinal_value'] += 1
    features.db.execute('UPDATE records SET metadata=? WHERE id=?', (json.dumps(metadata), row['id']))
    tensors['ordinal_values'][0] = torch.arange(1, 6)
    save_file(tensors, str(tmp_path / 'shard-000000.safetensors'))
    cache = SharedFeatureCache(features, [row], 'fit', 'cpu')
    with pytest.raises(ValueError, match='exactly'):
        training.validate_fit_ordinal_cache(cache)
    features.db.close()


def test_cpu_admission_occurs_before_any_upload_and_hot_auxiliary_has_no_scalar_guards():
    source = Path(training.__file__).read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'run_trial')
    statements = [ast.unparse(node) for node in function.body]
    constructor = next(i for i, value in enumerate(statements) if value.startswith('fitting = SharedFeatureCache'))
    assert "'cpu'" in statements[constructor]
    assert statements[constructor + 1] == 'validate_fit_ordinal_cache(fitting)'
    assert statements[constructor + 2].startswith('fitting.data =') and '.to(device)' in statements[constructor + 2]
    auxiliary = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'ordinal_cdf_penalty')
    assert not any(isinstance(node, ast.Attribute) and node.attr in ('item', 'all', 'any', 'cpu') for node in ast.walk(auxiliary))
    components = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == '_fitting_loss_components')
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ordinal_cdf_penalty'
               for node in ast.walk(components)) == 1
