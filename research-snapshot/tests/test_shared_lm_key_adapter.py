"""F43 CPU-only key residual, exact A41 controls and synthetic artifact guards."""

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from fuzev_research import evaluate, head as head_module, shared_head, shared_qualification, shared_train as training
from fuzev_research.head import CandidateHead, load_bundle
from test_shared_lm_query_adapter import inputs
from test_shared_lm_query_channel_rank64 import production, artifact
from test_shared_lm_rank512 import repin, sha
from test_shared_qualification import manifest as qualified_manifest
from fuzev_research.train import membership_identity


PARENT = 'da1a527d18d1e053e3a72da31fc0cbd72258fcda'  # unpublished lab commit ID
REFERENCE = '6721adaf03b0598839c2d59c6059632912bc883a'  # unpublished lab commit ID
KEY_NAMES = ('key_adapter_down.weight', 'key_adapter_down.bias', 'key_adapter_up.weight')
GROUPS = ('scorer_and_gate', 'query_adapter', 'channel_gate', 'key_adapter')




def test_key_draws_append_after_all_parent_draws_and_exact_one_head_geometry():
    head = shared_head.SharedDecisionHead(seed=43)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(43)
        scorer = CandidateHead(2560, 512).float()
        nn.init.xavier_uniform_(scorer.query.weight, gain=1.)
        nn.init.xavier_uniform_(scorer.key.weight, gain=.001)
        nn.init.zeros_(scorer.query.bias)
        nn.init.zeros_(scorer.key.bias)
        nn.Linear(2560, 1).float()
        query_down = nn.Linear(2560, 64).float()
        nn.Linear(64, 512, bias=False).float()
        nn.init.xavier_uniform_(query_down.weight, gain=1.)
        down = nn.Linear(2560, 16).float()
        nn.Linear(16, 512, bias=False).float()
        nn.init.xavier_uniform_(down.weight, gain=1.)
    assert torch.equal(head.key_adapter_down.weight, down.weight)
    assert head.key_adapter_down.weight.abs().max() <= math.sqrt(6 / (2560 + 16))
    assert not head.key_adapter_down.bias.any() and not head.key_adapter_up.weight.any()
    assert head.key_adapter_up.bias is None
    assert head.key_adapter_down.weight.shape == (16, 2560)
    assert head.key_adapter_up.weight.shape == (512, 16)
    assert sum(head.state_dict()[name].numel() for name in KEY_NAMES) == 49168
    assert len(tuple(head.parameters())) == shared_head.STATE_TENSOR_COUNT == 15
    assert sum(parameter.numel() for parameter in head.parameters()) == 2823233 + 49168 == 2872401
    assert shared_head.KEY_ADAPTER_SIZE == 16 and shared_head.SCHEMA_VERSION == 25
    assert shared_head.ARCHITECTURE == 'shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1'
    assert set(dict(head.named_children())) == {
        'scorer', 'gate', 'adapter_down', 'adapter_up', 'key_adapter_down', 'key_adapter_up'}
    assert head.requires_lm_scores and head.lm_assistance['extra_lm_copy'] is False
    assert head.key_adapter == {
        'rank': 16, 'activation': 'SiLU', 'input': 'candidate-only', 'output': 'additive-key-projection',
        'down_bias': True, 'up_bias': False, 'down_gain': 1., 'initial_down_bias': 0.,
        'initial_up_weight': 0., 'construction': 'after-all-A41-parent-RNG-draws',
        'parent_commit': PARENT, 'source_only_reference_commit': REFERENCE}


@pytest.fixture
def active_key():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    with torch.no_grad():
        head.adapter_up.weight.fill_(.013)
        head.key_adapter_up.weight.fill_(.021)
        head.key_adapter_down.bias.fill_(.1)
        head.gate.weight.fill_(.04)
        head.gate.bias.fill_(-.3)
        head.channel_a.copy_(torch.tensor([.1, -.2, .3]))
        head.channel_b.copy_(torch.tensor([-.2, .3, .1]))
        head.channel_c.copy_(torch.tensor([.3, -.1, -.2]))
    return head


@pytest.mark.parametrize('count', [2, 3, 5, 16])
@pytest.mark.parametrize('primitive', ['choice', 'noul', 'ordinal'])
def test_live_key_formula_public_training_masks_permutations_and_soft_ties(active_key, count, primitive):
    query, candidates, baseline, mask = inputs(4, count)
    if count == 2:
        mask.fill_(True)
    candidates.requires_grad_(True)
    u = active_key.scorer.query(query) + active_key.adapter_up(nn.functional.silu(active_key.adapter_down(query)))
    v = active_key.scorer.key(candidates) + active_key.key_adapter_up(nn.functional.silu(active_key.key_adapter_down(candidates)))
    channel = active_key.channel_a * u[:, None] + active_key.channel_b * v + active_key.channel_c
    raw = torch.einsum('bd,bkd->bk', u, v) / math.sqrt(3)
    raw = raw + (u[:, None] * v * (2 * channel.sigmoid() - 1)).sum(-1) / math.sqrt(3)
    def center(values):
        values = values.masked_fill(~mask, 0)
        return (values - values.sum(1, keepdim=True) / mask.sum(1, keepdim=True)).masked_fill(~mask, 0)
    expected = (baseline.masked_fill(~mask, 0) + center(raw)
                + (2 * active_key.gate(query).sigmoid() - 1) * center(baseline)).masked_fill(~mask, -torch.inf)
    actual = active_key(primitive, query, candidates, mask, baseline=baseline)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    batch = {'query': query, 'candidates': candidates, 'mask': mask, 'baseline': baseline}
    assert torch.equal(training.shared_logits(active_key, batch, primitive), actual)
    order = torch.arange(count - 1, -1, -1)
    permuted = active_key(primitive, query, candidates[:, order], mask[:, order], baseline=baseline[:, order])
    torch.testing.assert_close(permuted, actual[:, order], rtol=1e-6, atol=1e-7)
    altered, altered_baseline = candidates.detach().clone(), baseline.clone()
    altered[~mask] = 1000.
    altered_baseline[~mask] = torch.finfo(torch.float32).max
    assert torch.equal(active_key(primitive, query, altered, mask, baseline=altered_baseline), actual)
    gold = mask.to(torch.float32) / mask.sum(1, keepdim=True)
    batch['targets'] = gold
    before = gold.clone()
    loss = training.fitting_loss(actual, batch, [primitive] * len(query), role='fit')
    loss.sum().backward()
    assert torch.equal(gold, before) and baseline.grad is None
    assert torch.isfinite(loss).all() and not candidates.grad[~mask].any()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in active_key.parameters())


def test_key_up_live_initially_then_down_live_after_one_global_clipped_step():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    query, candidates, baseline, mask = inputs()
    optimizer = training.make_optimizer(head)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = head('choice', query, candidates, mask, baseline=baseline)
        nn.functional.cross_entropy(logits, torch.tensor([2, 1])).backward()
        for name, parameter in head.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert head.key_adapter_up.weight.grad.abs().sum() > 0
        for parameter in head.key_adapter_down.parameters():
            assert bool(parameter.grad.abs().sum() > 0) == bool(step)
        parameters = list(head.parameters())
        assert len(parameters) == 15
        if step == 0:
            parent_norm = torch.stack([p.grad.norm() for name, p in head.named_parameters()
                                       if name not in KEY_NAMES]).norm()
            full_norm = torch.stack([p.grad.norm() for p in parameters]).norm()
            assert full_norm > parent_norm
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(optimizer.state[p]['step'].item() == step + 1 for p in parameters)


@torch.no_grad()
def compare_cpu_updates(*, hidden_size, projection_size):
    """Keep the inherited 24-step oracle, extending it only to the fourth group."""
    candidate = shared_head.SharedDecisionHead(seed=43, hidden_size=hidden_size, projection_size=projection_size)
    reference = copy.deepcopy(candidate)
    actual = training.make_optimizer(candidate)
    named, ref_named = dict(candidate.named_parameters()), dict(reference.named_parameters())
    declared = training.optimization_protocol()
    specs = declared['parameter_groups']
    assert [g['name'] for g in specs] == list(GROUPS)
    assert [len(g['parameter_names']) for g in specs] == [6, 3, 3, 3]
    expected = torch.optim.AdamW([
        {'params': [ref_named[name] for name in g['parameter_names']], 'lr': 1e-5 * g['lr_multiplier']}
        for g in specs], weight_decay=.01, fused=False, foreach=False)
    generator = torch.Generator().manual_seed(119)
    templates = {name: torch.randn(p.shape, generator=generator) for name, p in named.items()}
    maxima = {name: {'parameters': 0., 'moments': 0.} for name in named}
    zero_steps = [1, 6, 11, 16, 21]
    clipped, rates = [], []
    tolerance = declared['numerical_admission']['tolerances']
    def check(left, right, kind):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, **tolerance[kind])
        return float((left - right).abs().max())
    for step in range(1, 25):
        epoch = (step - 1) // 4 + 1
        before_moments = {p: {key: actual.state[p][key].clone() for key in ('step', 'exp_avg', 'exp_avg_sq')}
                          for p in candidate.parameters() if p in actual.state}
        actual_rates = training.set_optimizer_learning_rates(actual, candidate, epoch)
        for p, state in before_moments.items():
            for key, value in state.items():
                assert torch.equal(actual.state[p][key], value)
        for group, spec in zip(expected.param_groups, specs):
            group['lr'] = actual_rates[spec['name']]
        rates.append({'step': step, 'epoch': epoch, **actual_rates})
        actual.zero_grad(set_to_none=True)
        expected.zero_grad(set_to_none=True)
        for name, parameter in named.items():
            scale = (1e-4 if step % 2 else 10.) * (-1 if step % 3 else 1)
            gradient = torch.zeros_like(parameter) if step in zero_steps else (templates[name] + (step % 5 - 2) * .01) * scale
            parameter.grad, ref_named[name].grad = gradient.clone(), gradient.clone()
        norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        if float(norm) > 1.:
            clipped.append(step)
        gradients = {name: p.grad.clone() for name, p in named.items()}
        decay_only = {name: p.clone() for name, p in named.items()} if step == 1 else {}
        for optimizer in (actual, expected) if step % 2 else (expected, actual):
            optimizer.step()
        for name, parameter in named.items():
            maxima[name]['parameters'] = max(maxima[name]['parameters'], check(parameter, ref_named[name], 'parameters'))
            assert torch.equal(parameter.grad, gradients[name]) and torch.equal(ref_named[name].grad, gradients[name])
            if step == 1:
                group = next(g for g in specs if name in g['parameter_names'])
                check(parameter, decay_only[name] * (1 - actual_rates[group['name']] * .01), 'parameters')
            assert actual.state[parameter]['step'].item() == expected.state[ref_named[name]]['step'].item() == step
            for key in ('exp_avg', 'exp_avg_sq'):
                maxima[name]['moments'] = max(maxima[name]['moments'],
                    check(actual.state[parameter][key], expected.state[ref_named[name]][key], 'moments'))
    assert clipped and len(clipped) < 24 - len(zero_steps)
    return {'passed': True, 'steps': 24, 'parameters': list(named), 'maximum_absolute_errors': maxima,
            'parameter_count': sum(p.numel() for p in candidate.parameters()), 'zero_gradient_steps': zero_steps,
            'global_clips_per_head_per_step': 1, 'clipped_steps': clipped,
            'source': training.source_identity(), 'group_learning_rates': rates}


@pytest.mark.parametrize('case', ['rank8', 'down_bias', 'up_bias', 'in_features', 'half', 'nan', 'hidden_overflow', 'key_overflow'])
def test_new_key_geometry_and_nonfinite_fail_closed(active_key, case):
    head = active_key
    with torch.no_grad():
        if case == 'rank8':
            head.key_adapter_down, head.key_adapter_up = nn.Linear(4, 8), nn.Linear(8, 3, bias=False)
        elif case == 'down_bias':
            head.key_adapter_down = nn.Linear(4, 16, bias=False)
        elif case == 'up_bias':
            head.key_adapter_up = nn.Linear(16, 3)
        elif case == 'in_features':
            head.key_adapter_down.in_features = 5
        elif case == 'half':
            head.key_adapter_up.half()
        elif case == 'nan':
            head.key_adapter_up.weight[0, 0] = torch.nan
        elif case == 'hidden_overflow':
            head.key_adapter_down.weight.fill_(torch.finfo(torch.float32).max)
            head.key_adapter_up.weight.zero_()
        else:
            head.key_adapter_up.weight.fill_(torch.finfo(torch.float32).max)
            head.key_adapter_down.weight.fill_(1.)
    query, candidates, baseline, mask = inputs()
    if 'overflow' in case:
        candidates.fill_(2.)
    with pytest.raises(ValueError):
        head('choice', query, candidates, mask, baseline=baseline)
    if case in ('rank8', 'down_bias', 'up_bias', 'in_features', 'half'):
        with pytest.raises(ValueError):
            training.make_optimizer(head)


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'extra', 'wrong_multiplier', 'wrong_names', 'wrong_decay', 'foreach'])
def test_four_group_reconfiguration_refused_atomically(case):
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    optimizer = training.make_optimizer(head)
    group = optimizer.param_groups[-1]
    if case == 'duplicate':
        group['params'][0] = optimizer.param_groups[0]['params'][0]
    elif case == 'missing':
        optimizer.param_groups.pop()
    elif case == 'extra':
        optimizer.param_groups.append(copy.copy(group))
    elif case == 'wrong_multiplier':
        group['lr_multiplier'] = 1.
    elif case == 'wrong_names':
        group['parameter_names'] = list(reversed(group['parameter_names']))
    elif case == 'wrong_decay':
        group['weight_decay'] = 0.
    else:
        group['foreach'] = True
    before = [g['lr'] for g in optimizer.param_groups]
    with pytest.raises(ValueError):
        training.set_optimizer_learning_rates(optimizer, head, 5)
    assert [g['lr'] for g in optimizer.param_groups] == before


@pytest.mark.parametrize('value', [None, True, 10, 1., 100., float('nan')])
def test_key_rate_profile_denied_before_outputs_or_feature_access(tmp_path, monkeypatch, value):
    monkeypatch.setitem(training.PROFILE, 'key_adapter_learning_rate_multiplier', value)
    monkeypatch.setattr(training, 'require_feature_pin', lambda *a: pytest.fail('Invalid profile reached features'))
    output = tmp_path / 'forbidden'
    with pytest.raises(ValueError, match='key-adapter learning-rate'):
        training.run_trial(None, {}, output, device='cpu', development_only=True)
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('rank', 16.), ('down_gain', 1), ('up_bias', 0),
    ('parent_commit', REFERENCE), ('source_only_reference_commit', PARENT), ('construction', 'before-parent-draws')])
def test_key_metadata_strict_before_weight_decode(artifact, monkeypatch, field, value):
    path = artifact / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['key_adapter'][field] = value
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(shared_head, 'load', lambda *a: pytest.fail('Invalid key metadata reached decode'))
    with pytest.raises(ValueError):
        shared_head.load_shared_bundle(artifact)


@pytest.mark.parametrize('location', ['manifest', 'initialization', 'lm_assistance', 'calibration'])
@pytest.mark.parametrize('size', [64, 16., True, None])
def test_key_size_strict_in_every_saved_contract(artifact, location, size):
    path = artifact / ('calibration.json' if location == 'calibration' else 'manifest.json')
    value = json.loads(path.read_text())
    target = value if location in ('manifest', 'calibration') else value[location]
    target['key_adapter_size'] = size
    path.write_text(json.dumps(value))
    if location == 'calibration':
        repin(artifact, calibration_sha256=sha(path))
    with pytest.raises(ValueError):
        shared_head.load_shared_bundle(artifact)


@pytest.mark.parametrize('change', [{'schema_version': 18}, {'schema_version': 24}, {'schema_version': 25.},
    {'state_tensor_count': 12}, {'parameter_count': 2823233},
    {'architecture': 'shared-lm-assisted-query-channel-gate-adapter64-candidate-dot-product-rank512-v1'}])
def test_parent_or_feature_geometry_cannot_be_relabelled(artifact, monkeypatch, change):
    repin(artifact, **change)
    monkeypatch.setattr(shared_head, 'load', lambda *a: pytest.fail('Invalid geometry reached decode'))
    with pytest.raises(ValueError):
        load_bundle(artifact)


@pytest.mark.parametrize('case', ['missing', 'shape', 'extra_head', 'half', 'nonfinite'])
def test_new_state_guards_even_with_recomputed_weight_digest(artifact, case):
    path = artifact / 'head.safetensors'
    state = load_file(str(path))
    if case == 'missing':
        state.pop(KEY_NAMES[0])
    elif case == 'shape':
        state[KEY_NAMES[0]] = state[KEY_NAMES[0]][:8].contiguous()
    elif case == 'extra_head':
        state['second_head.weight'] = torch.zeros(1, 1)
    elif case == 'half':
        state[KEY_NAMES[0]] = state[KEY_NAMES[0]].half()
    else:
        state[KEY_NAMES[0]][0, 0] = torch.inf
    save_file(state, str(path))
    repin(artifact, weights_sha256=sha(path))
    with pytest.raises(ValueError):
        load_bundle(artifact)




@pytest.mark.parametrize('case', ['baseline_gradient', 'baseline_nan', 'empty_mask', 'mask_dtype', 'query_nan', 'candidate_nan'])
def test_original_frozen_baseline_and_feature_guards_unchanged(active_key, case):
    query, candidates, baseline, mask = inputs()
    if case == 'baseline_gradient':
        baseline.requires_grad_(True)
    elif case == 'baseline_nan':
        baseline[0, 0] = torch.nan
    elif case == 'empty_mask':
        mask[0] = False
    elif case == 'mask_dtype':
        mask = mask.float()
    elif case == 'query_nan':
        query[0, 0] = torch.nan
    else:
        candidates[0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        active_key('choice', query, candidates, mask, baseline=baseline)


@pytest.mark.parametrize('missing_field', [None, 'training_code', 'calibration_code'])
def test_evaluator_exact_shared_reference_dispatch_and_source8_refusal(monkeypatch, missing_field):
    bundle = shared_head.SharedDecisionHead(seed=43)
    value, temperatures = qualified_manifest()
    rows = {role: [{'id': f'synthetic-{role}', 'group_': f'synthetic-{role}-group', 'role': role}]
            for role in ('fit', 'development', 'calibration')}
    features = SimpleNamespace(identity=value['feature_identity'], iter_records=lambda role: iter(rows[role]))
    for role, field in (('fit', 'fit_membership'), ('development', 'development_membership'),
                        ('calibration', 'calibration_membership')):
        value[field] = membership_identity(rows[role])
    if missing_field is not None:
        value[missing_field] = {k: v for k, v in value[missing_field].items() if k != 'shared_head.py'}
    calls = []
    monkeypatch.setattr(evaluate, 'check_hashes', lambda *a: calls.append('hashes'))
    monkeypatch.setattr(evaluate, 'require_complete', lambda actual: value['preparation'])
    monkeypatch.setattr(head_module, 'load_bundle', lambda *a: (bundle, value, temperatures))
    def validate(actual_bundle, actual_manifest, actual_temperatures, *, features, corpus):
        assert actual_bundle is bundle and actual_manifest is value and actual_temperatures is temperatures
        assert features.identity == value['feature_identity'] and corpus is True
        calls.append('shared-full-corpus-validation')
    monkeypatch.setattr(shared_qualification, 'validate_shared_reference', validate)
    if missing_field is None:
        actual, manifest, actual_temperatures = evaluate.load_reference(features, 'synthetic-no-files', {})
        assert actual is bundle and manifest is value and actual_temperatures is temperatures
        assert calls == ['hashes', 'shared-full-corpus-validation', 'hashes']
        assert not any(parameter.requires_grad for parameter in bundle.parameters())
    else:
        with pytest.raises(ValueError, match='training/calibration code provenance'):
            evaluate.load_reference(features, 'synthetic-no-files', {})
        assert calls == ['hashes'] and all(parameter.requires_grad for parameter in bundle.parameters())
