"""R28 capacity control: random CPU inputs and synthetic artifacts only."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from fuzev_research import shared_head, shared_qualification, shared_train as training
from fuzev_research.head import CandidateHead, load_bundle
from test_shared_lm_consistency import MetadataFeatures, SyntheticCache, rows_for
from test_shared_lm_query_adapter import inputs
from test_shared_lm_rank512 import metadata, repin, sha
from test_shared_lm_rank512 import test_evaluation_report_records_exact_shared_source_hashes as assert_evaluation_hashes
from test_shared_qualification import manifest as qualified_metadata


PARENT = 'e2b88f1d3bd09a17e3651046cd3663574ae68707'  # unpublished lab commit ID
PROFILE = 'key-adapter-ordinal-rps015-light-consistency-medium'
COMMON_NAMES = ('scorer.query.weight', 'scorer.query.bias', 'scorer.key.weight', 'scorer.key.bias',
                'gate.weight', 'gate.bias', 'channel_a', 'channel_b', 'channel_c')












def test_parameter_count_is_derived_from_actual_fifteen_tensor_geometry():
    head = shared_head.SharedDecisionHead(seed=43)
    assert len(head.state_dict()) == len(tuple(head.parameters())) == 15
    sizes = {name: value.numel() for name, value in head.state_dict().items()}
    assert sizes['adapter_down.weight'] == 64 * 2560 and sizes['adapter_down.bias'] == 64
    assert sizes['adapter_up.weight'] == 512 * 64
    assert sizes['key_adapter_down.weight'] == 16 * 2560 and sizes['key_adapter_down.bias'] == 16
    assert sizes['key_adapter_up.weight'] == 512 * 16
    assert sum(sizes.values()) == shared_head.PARAMETER_COUNT == 2872401
    assert head.initialization['query_adapter_size'] == head.lm_assistance['query_adapter_size'] == 64
    assert head.requires_lm_scores and head.lm_assistance['extra_lm_copy'] is False


@pytest.fixture
def active():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    with torch.no_grad():
        head.adapter_up.weight.fill_(.03)
        head.adapter_down.bias.fill_(.1)
        head.scorer.key.weight.mul_(1000.)
        head.gate.weight.fill_(.1)
        head.gate.bias.fill_(-.3)
        head.channel_a.copy_(torch.tensor([.1, -.2, .3]))
        head.channel_b.copy_(torch.tensor([.4, -.5, .6]))
        head.channel_c.copy_(torch.tensor([-.2, .3, -.4]))
    return head


def test_nonzero_formula_permutation_padding_and_detached_teacher(active):
    query, candidates, baseline, mask = inputs()
    candidates.requires_grad_(True)
    u = active.scorer.query(query) + active.adapter_up(torch.nn.functional.silu(active.adapter_down(query)))
    v = active.scorer.key(candidates)
    gates = 2 * torch.sigmoid(active.channel_a * u[:, None] + active.channel_b * v + active.channel_c)
    raw = (u[:, None] * v * gates).sum(-1) / math.sqrt(3)
    def center(value):
        value = value.masked_fill(~mask, 0)
        return (value - value.sum(1, keepdim=True) / mask.sum(1, keepdim=True)).masked_fill(~mask, 0)
    expected = (baseline + center(raw) + (2 * active.gate(query).sigmoid() - 1) * center(baseline)).masked_fill(~mask, -torch.inf)
    actual = active('choice', query, candidates, mask, baseline=baseline)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    order = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(active('ordinal', query, candidates[:, order], mask[:, order], baseline=baseline[:, order]),
                               actual[:, order], rtol=1e-6, atol=1e-6)
    padded, padded_base = candidates.detach().clone(), baseline.clone()
    padded[~mask], padded_base[~mask] = 1000., torch.finfo(torch.float32).max
    assert torch.equal(active('noul', query, padded, mask, baseline=padded_base), actual)
    torch.nn.functional.cross_entropy(actual, torch.tensor([2, 1])).backward()
    assert not candidates.grad[~mask].any() and baseline.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.parameters())


def test_up_and_channel_gradients_first_step_down_gradients_after_actual_step():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    optimizer = training.make_optimizer(head)
    query, candidates, baseline, mask = inputs()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(head('choice', query, candidates, mask, baseline=baseline), torch.tensor([2, 1])).backward()
        for name, parameter in head.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if name.startswith('adapter_down.'):
                assert bool(parameter.grad.abs().sum() > 0) == (step > 0)
            elif name == 'adapter_up.weight' or name.startswith('channel_'):
                assert parameter.grad.abs().sum() > 0, name
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


@pytest.mark.parametrize('case', ['rank16', 'no_down_bias', 'up_bias', 'half', 'nan', 'overflow'])
def test_adapter_geometry_finiteness_and_optimizer_guards(active, case):
    with torch.no_grad():
        if case == 'rank16':
            active.adapter_down, active.adapter_up = nn.Linear(4, 16), nn.Linear(16, 3, bias=False)
        elif case == 'no_down_bias':
            active.adapter_down = nn.Linear(4, 64, bias=False)
        elif case == 'up_bias':
            active.adapter_up = nn.Linear(64, 3)
        elif case == 'half':
            active.adapter_down.half()
        elif case == 'nan':
            active.adapter_up.weight[0, 0] = torch.nan
        else:
            active.adapter_down.weight.fill_(torch.finfo(torch.float32).max)
    query, candidates, baseline, mask = inputs()
    if case == 'overflow':
        query.fill_(2.)
    with pytest.raises(ValueError):
        active('choice', query, candidates, mask, baseline=baseline)
    if case == 'rank16':
        with pytest.raises(ValueError, match='rank-64'):
            training.make_optimizer(active)


@pytest.fixture(scope='module')
def production():
    head = shared_head.SharedDecisionHead(seed=43)
    with torch.no_grad():
        head.adapter_up.weight.fill_(.003)
        head.gate.bias.fill_(-.3)
        head.channel_a.fill_(.1)
        head.channel_b.fill_(.2)
        head.channel_c.fill_(-.3)
    return head


@pytest.fixture
def artifact(tmp_path, production):
    directory = tmp_path / 'synthetic-rank64'
    shared_head.save_shared_bundle(directory, production, metadata(), {name: 1. for name in training.PRIMITIVES})
    return directory


def test_schema18_strict_direct_roundtrip_and_synthetic_qualification(artifact, production, tmp_path):
    loaded, manifest, _ = shared_head.load_shared_bundle(artifact)
    query, candidates, baseline, mask = inputs(2560)
    assert torch.equal(loaded('choice', query, candidates, mask, baseline=baseline),
                       production('choice', query, candidates, mask, baseline=baseline))
    assert manifest['schema_version'] == 25 and manifest['query_adapter_size'] == 64
    assert manifest['key_adapter_size'] == 16 and manifest['key_adapter'] == production.key_adapter
    assert manifest['parameter_count'] == 2872401 and manifest['initialization'] == production.initialization
    value, temperatures = qualified_metadata()
    value.update(schema_version=25, query_adapter_size=64, key_adapter_size=16, parameter_count=2872401)
    value['weights_sha256'] = sha(artifact / 'head.safetensors')
    value['selected_checkpoint'].update(directory=str(artifact), sha256=value['weights_sha256'],
                                         manifest_sha256=sha(artifact / 'manifest.json'))
    del value['calibration_sha256']
    directory = tmp_path / 'synthetic-reference'
    shared_head.save_shared_bundle(directory, production, value, temperatures)
    head, manifest, temperatures = shared_head.load_shared_bundle(directory)
    shared_qualification.validate_shared_reference(head, manifest, temperatures)
    assert manifest['training_code'] == manifest['shared_training_code'] == manifest['calibration_code'] == training.source_identity()
    with pytest.raises(ValueError, match='full read-only corpus'):
        shared_qualification.validate_shared_reference(head, manifest, temperatures, corpus=True)


def test_generic_loader_dispatches_new_architecture(artifact):
    head, manifest, _ = load_bundle(artifact)
    assert isinstance(head, shared_head.SharedDecisionHead) and manifest['schema_version'] == 25


@pytest.mark.parametrize('change', [{'schema_version': 17}, {'schema_version': 18.}, {'query_adapter_size': 16},
    {'query_adapter_size': 64.}, {'parameter_count': 2675729},
    {'architecture': 'shared-lm-assisted-query-channel-gate-candidate-dot-product-rank512-v1'}])
def test_old_schema_and_geometry_rejected_before_decode(artifact, monkeypatch, change):
    repin(artifact, **change)
    monkeypatch.setattr(shared_head, 'load', lambda *args: pytest.fail('Invalid rank reached weight decoding'))
    with pytest.raises(ValueError, match='Incompatible'):
        shared_head.load_shared_bundle(artifact)


@pytest.mark.parametrize('case', ['down', 'down_bias', 'up', 'extra_head', 'nan'])
def test_rank16_state_cannot_be_relabelled_with_new_checksums(artifact, case):
    path = artifact / 'head.safetensors'
    state = load_file(str(path))
    if case == 'down':
        state['adapter_down.weight'] = state['adapter_down.weight'][:16].contiguous()
    elif case == 'down_bias':
        state['adapter_down.bias'] = state['adapter_down.bias'][:16].contiguous()
    elif case == 'up':
        state['adapter_up.weight'] = state['adapter_up.weight'][:, :16].contiguous()
    elif case == 'extra_head':
        state['extra_head.weight'] = torch.zeros(1, 1)
    else:
        state['adapter_down.weight'][0, 0] = torch.nan
    save_file(state, str(path))
    repin(artifact, weights_sha256=sha(path))
    with pytest.raises(ValueError, match='state shape|parameters'):
        shared_head.load_shared_bundle(artifact)


@pytest.mark.parametrize('location', ['initialization', 'lm_assistance', 'calibration'])
@pytest.mark.parametrize('size', [16, 64., True, None])
def test_adapter_size_metadata_strict_in_all_profiles(artifact, location, size):
    if location == 'calibration':
        path = artifact / 'calibration.json'
        value = json.loads(path.read_text())
        value['query_adapter_size'] = size
        path.write_text(json.dumps(value))
        repin(artifact, calibration_sha256=sha(path))
    else:
        path = artifact / 'manifest.json'
        value = json.loads(path.read_text())
        value[location]['query_adapter_size'] = size
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        shared_head.load_shared_bundle(artifact)




def test_evaluation_records_new_architecture_source_hashes_using_only_synthetic_empty_plan(tmp_path, monkeypatch):
    assert_evaluation_hashes(tmp_path, monkeypatch, shared_head.ARCHITECTURE, True)


def test_real_synthetic_loop_one_clip_step_four_groups_visits_loss_and_checkpoint(tmp_path, monkeypatch):
    roles = {role: rows_for(role) for role in ('fit', 'development', 'calibration')}
    features = MetadataFeatures(tmp_path / 'features', roles['fit'])
    monkeypatch.setattr(training, 'FEATURE_SHA', training.file_sha256(features.directory / 'identity.json'))
    monkeypatch.setattr(training, 'PROFILE', training.PROFILE | {'max_epochs': 1, 'batch_size': 7})
    monkeypatch.setattr(training, 'SharedFeatureCache', SyntheticCache)
    monkeypatch.setattr(training, 'Store', lambda: SimpleNamespace(update=lambda *args, **kwargs: None))
    monkeypatch.setattr(training, 'development', lambda *args: {'eligible': True,
        'macro_normalized_nll': {'control': 2., 'head': 1.}, 'gain_vs_label_token': 1., 'checks': []})
    clips, steps = [], []
    native_clip, native_step = torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step
    def observed_clip(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        assert len(parameters) == len({id(p) for p in parameters}) == 15
        assert max_norm == 1. and kwargs == {'error_if_nonfinite': True}
        assert len(clips) == len(steps)
        clips.append(1)
        return native_clip(parameters, max_norm, **kwargs)
    def observed_step(optimizer, *args, **kwargs):
        assert len(clips) == len(steps) + 1
        assert [(g['name'], g['lr'], len(g['params'])) for g in optimizer.param_groups] == [
            ('scorer_and_gate', 1e-5, 6), ('query_adapter', 1e-4, 3), ('channel_gate', 1e-4, 3),
            ('key_adapter', 1e-4, 3)]
        steps.append(1)
        return native_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', observed_clip)
    monkeypatch.setattr(torch.optim.AdamW, 'step', observed_step)
    result = training.run_trial(features, roles, tmp_path / 'trial', device='cpu', development_only=True)
    protocol = json.loads((tmp_path / 'trial/protocol.json').read_text())
    assert protocol['primitive_masses'] == {'choice': 6 / 15, 'noul': 4 / 15, 'ordinal': 5 / 15}
    assert protocol['track_masses'] == dict.fromkeys(training.TRACKS, 1 / 15)
    assert protocol['profile']['fit_track_weighting'] == 'equal-track-macro'
    assert protocol['profile']['balanced_noul'] is False
    assert protocol['optimization']['fit_dev_hypothesis']['source_parent_commit'] == training.SOURCE_PARENT_COMMIT
    assert protocol['optimization']['fit_dev_hypothesis']['control_commit'] == training.RETAINED_CONTROL_COMMIT
    assert len(clips) == len(steps) == 5
    epoch = result['epochs'][0]
    assert epoch['fit_visits'] == 30 and epoch['fit_consistency_pairs'] == 6
    assert epoch['fit_loss'] == pytest.approx(epoch['fit_gold_ce'] + .03 * epoch['fit_teacher_kl']
        + .001 * epoch['fit_correction_penalty'] + .05 * epoch['fit_consistency_js']
        + .15 * epoch['fit_ordinal_rps'], rel=2e-7, abs=2e-7)
    head, manifest, _ = shared_head.load_shared_bundle(epoch['directory'])
    assert manifest['query_adapter_size'] == 64 and manifest['schema_version'] == 25
    assert manifest['key_adapter_size'] == 16
    assert manifest['profile'] == training.PROFILE and sum(p.numel() for p in head.parameters()) == 2872401
    assert manifest['training_code'] == training.source_identity() and result['calibration'] == 'not_started'


@pytest.mark.parametrize('value', [0., .1, False, True, float('nan'), None])
def test_only_light_kl_configuration_before_data_or_output(tmp_path, monkeypatch, value):
    monkeypatch.setitem(training.PROFILE, 'teacher_distribution_regularization', value)
    monkeypatch.setattr(training, 'require_feature_pin', lambda *args: pytest.fail('Unexpected feature access'))
    with pytest.raises(ValueError, match='teacher KL weight 0.03'):
        training.run_trial(None, {}, tmp_path / 'denied', device='cpu', development_only=True)
    assert not (tmp_path / 'denied').exists()


@pytest.mark.parametrize('explicit', [False, True])
def test_cli_default_and_explicit_rank64_profile_without_real_cuda(tmp_path, monkeypatch, explicit):
    directory = tmp_path / 'features'
    directory.mkdir()
    identity = {'dataset_scope': training.contract(augmentation=True, dimensions=True), 'synthetic_only': True}
    (directory / 'identity.json').write_text(json.dumps(identity))
    monkeypatch.setattr(training, 'FEATURE_SHA', sha(directory / 'identity.json'))
    monkeypatch.setattr(training, 'PROFILE', {})
    paths, calls = [], []
    def project_path(path):
        paths.append(path)
        return directory if path == training.FEATURES else tmp_path / 'output'
    features = SimpleNamespace(close=lambda: calls.append('closed'))
    roles = {'fit': [], 'development': []}
    def run_trial(actual_features, actual_roles, output, *, development_only):
        assert actual_features is features and actual_roles is roles
        assert output == tmp_path / 'output' and development_only
        calls.append('fit-dev')
        return {'status': 'dev_rejected'}
    monkeypatch.setattr(training, 'project_path', project_path)
    monkeypatch.setattr(training, 'FeatureStore', lambda *args, **kwargs: features)
    monkeypatch.setattr(training, 'read_roles', lambda actual: roles)
    monkeypatch.setattr(training, 'run_trial', run_trial)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('Unexpected CUDA execution'))
    monkeypatch.setattr(torch, 'set_num_threads', lambda *args: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *args: None)
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', lambda *args: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr('sys.argv', ['shared-train', '--development-only'] + (['--profile', PROFILE] if explicit else []))
    training.main()
    assert training.PROFILE == training.BASE_PROFILE | training.SEARCH_PROFILES[PROFILE]
    assert paths == [training.FEATURES, f'weights/cuda-shared-lm-ordinal-rps-20261006/{PROFILE}']
    assert calls == ['fit-dev', 'closed']


@pytest.mark.parametrize('profile', ['query-channel-gold-consistency-medium', 'query-channel-light-consistency-medium',
                                   'query-channel-gate-consistency-medium', 'query-channel-rank64-light-consistency-medium',
                                   'balanced-query-channel-gold-consistency-medium',
                                   'balanced-query-channel-rank64-light-consistency-medium',
                                   'ordinal-rps015-light-consistency-medium'])
def test_old_profiles_rejected_before_feature_or_cuda_access(monkeypatch, profile):
    monkeypatch.setattr('sys.argv', ['shared-train', '--profile', profile, '--development-only'])
    monkeypatch.setattr(training, 'require_feature_pin', lambda *args: pytest.fail('Unexpected feature access'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('Unexpected CUDA access'))
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2


@pytest.mark.parametrize('hidden,rank', [(4, 3), (2560, 512)])
def test_all_fifteen_fused_updates_moments_decay_clipping_and_epoch5_schedule(hidden, rank, monkeypatch):
    for name in ('FeatureStore', 'SharedFeatureCache', 'Store'):
        monkeypatch.setattr(training, name, lambda *args: pytest.fail('Unexpected data or state access'))
    from test_shared_lm_key_adapter import compare_cpu_updates
    result = compare_cpu_updates(hidden_size=hidden, projection_size=rank)
    assert result['passed'] and result['steps'] == 24
    assert len(result['parameters']) == len(result['maximum_absolute_errors']) == 15
    assert result['global_clips_per_head_per_step'] == 1
    assert result['zero_gradient_steps'] == [1, 6, 11, 16, 21]
    assert result['clipped_steps'] and result['source'] == training.source_identity()
    if hidden == 2560:
        assert result['parameter_count'] == shared_head.PARAMETER_COUNT == 2872401
    for step in result['group_learning_rates']:
        rate = 1e-5 if step['epoch'] < 5 else 3e-6
        assert step['scorer_and_gate'] == rate
        assert step['query_adapter'] == step['channel_gate'] == step['key_adapter'] == rate * 10
