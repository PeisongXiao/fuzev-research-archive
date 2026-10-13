"""Private CPU replay fixtures; no kernels, loaded model, CUDA or dataset."""

from types import ModuleType, SimpleNamespace as NS
import sys

import pytest
import torch

from fuzev_research.kev_vllm_state_trace import ReplayIntegrityError, private_moe_replay, snapshot


@pytest.fixture
def rig(monkeypatch):
    tuner = NS(choose_one=lambda *args, **kwargs: (0, kwargs['gemm_idx'] + 10))
    module = ModuleType('flashinfer.autotuner')
    module.AutoTuner = NS(get=lambda: tuner)
    monkeypatch.setitem(sys.modules, 'flashinfer.autotuner', module)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    arguments = {name: torch.ones(4, 8, device='cpu') for name in
                 ('input', 'fc1_expert_weights', 'fc2_expert_weights', 'output')}
    arguments.update(quant_scales=[torch.tensor([1.], device='cpu')],
                     token_selected_experts=torch.ones(4, 2, dtype=torch.int32, device='cpu'),
                     token_final_scales=torch.ones(4, 2, device='cpu'), output_dtype=torch.float32)
    calls = []

    def original(**kwargs):
        assert kwargs['output'].data_ptr() != arguments['output'].data_ptr()
        profiles = kwargs.get('profile_ids')
        if profiles is None:
            for index in (1, 2):
                tuner.choose_one('gemm', [], {}, [], gemm_idx=index)
        else:
            assert profiles in ([11, 12], [-1, -1], [0, 0])
        calls.append(kwargs)
        kwargs['output'].copy_(kwargs['input'] + (len(calls) / 100 if kwargs['use_fused_finalize'] else 0))
        return kwargs['output']

    return NS(tuner=tuner, arguments=arguments, original=original, calls=calls)


def test_private_replay_freezes_inputs_and_preserves_original_output(rig):
    before = {name: value.clone() for name, value in rig.arguments.items() if isinstance(value, torch.Tensor)}
    choose = rig.tuner.choose_one
    report = private_moe_replay(rig.original, rig.arguments)
    assert len(rig.calls) == 10 and report['inputs_unchanged']
    assert report['status'] == 'complete'
    assert rig.tuner.choose_one == choose
    assert report['modes'][0]['max_abs_deltas'][0] > 0
    assert report['modes'][1]['max_abs_deltas'] == [0., 0., 0.]
    assert all(torch.equal(value, rig.arguments[name]) for name, value in before.items())
    assert all(call['input'].data_ptr() != rig.arguments['input'].data_ptr() for call in rig.calls)
    assert all(call['quant_scales'][0].data_ptr() != rig.arguments['quant_scales'][0].data_ptr() for call in rig.calls)
    assert report['weights'] == {name: snapshot(rig.arguments[name]) for name in
                                 ('fc1_expert_weights', 'fc2_expert_weights')}
    for mode in report['modes']:
        assert mode['status'] == 'complete' and mode['tactic_mode'] == 'explicit'
        assert mode['profile_ids'] == mode['observed_profile_ids'] == [11, 12]
        assert len(mode['output_hashes']) == 4


def test_private_replay_refuses_missing_tactics_and_restores_tuner(rig):
    choose = rig.tuner.choose_one
    report = private_moe_replay(lambda **kwargs: kwargs['output'].zero_(), rig.arguments)
    assert report['status'] == 'failed' and report['inputs_unchanged']
    assert len(report['modes']) == 1
    for mode in report['modes']:
        assert mode['status'] == 'failed' and 'GEMM tactics' in mode['error']
        assert mode['observed_profile_ids'] == [None, None] and mode['output_hashes'] == []
        assert 'tactic_mode' not in mode and 'profile_ids' not in mode
    assert rig.tuner.choose_one == choose


@pytest.mark.parametrize('profiles', [[-1, -1], [11, -1], [-1, 12], [0, 0]])
def test_private_replay_labels_default_policy_without_claiming_frozen_tactics(rig, profiles):
    selections = []

    def choose_tactic(*args, **kwargs):
        selections.append(kwargs['gemm_idx'])
        return 0, profiles[kwargs['gemm_idx'] - 1]

    rig.tuner.choose_one = choose_tactic
    choose = rig.tuner.choose_one
    report = private_moe_replay(rig.original, rig.arguments)
    assert len(rig.calls) == 10 and rig.tuner.choose_one == choose
    assert report['status'] == 'complete' and report['inputs_unchanged']
    assert selections == [1, 2, 1, 2]
    assert 'not actual selected tactic IDs' in report['tactic_scope']
    for mode in report['modes']:
        assert mode['observed_profile_ids'] == profiles
        assert mode['tactic_mode'] == ('default_dispatch' if -1 in profiles else 'explicit')
        assert mode['profile_ids'] == ([-1, -1] if -1 in profiles else profiles)
        assert len(mode['output_hashes']) == 4 and len(mode['max_abs_deltas']) == 3
    assert report['modes'][0]['max_abs_deltas'][0] > 0
    assert report['modes'][1]['max_abs_deltas'] == [0., 0., 0.]


@pytest.mark.parametrize('profiles', [[-2, 12], [True, 12], [11, '12']])
def test_private_replay_does_not_invent_default_dispatch_for_invalid_tactics(rig, profiles):
    rig.tuner.choose_one = lambda *args, **kwargs: (0, profiles[kwargs['gemm_idx'] - 1])
    choose = rig.tuner.choose_one
    report = private_moe_replay(rig.original, rig.arguments)
    assert report['status'] == 'failed' and len(rig.calls) == 1
    assert len(report['modes']) == 1
    assert rig.tuner.choose_one == choose
    for mode in report['modes']:
        assert mode['observed_profile_ids'] == profiles and mode['output_hashes'] == []
        assert 'profile_ids' not in mode and 'tactic_mode' not in mode


@pytest.mark.parametrize('default', [False, True])
@pytest.mark.parametrize('failed_call', [1, 4])
def test_private_replay_keeps_fused_evidence_when_ordered_control_fails(rig, default, failed_call):
    if default:
        rig.tuner.choose_one = lambda *args, **kwargs: (0, -1)
    choose = rig.tuner.choose_one
    ordered_calls = 0

    def fail_ordered(**kwargs):
        nonlocal ordered_calls
        result = rig.original(**kwargs)
        if not kwargs['use_fused_finalize']:
            ordered_calls += 1
            if ordered_calls == failed_call:
                raise RuntimeError('ordered control failed')
        return result

    report = private_moe_replay(fail_ordered, rig.arguments)
    fused, ordered = report['modes']
    assert report['status'] == 'failed' and report['inputs_unchanged']
    assert rig.tuner.choose_one == choose
    assert fused['status'] == 'complete' and len(fused['output_hashes']) == 4
    assert len(fused['max_abs_deltas']) == 3 and fused['max_abs_deltas'][0] > 0
    assert ordered['status'] == 'failed' and ordered['error_type'] == 'RuntimeError'
    assert ordered['error'] == 'ordered control failed'
    assert ordered['observed_profile_ids'] == ([-1, -1] if default else [11, 12])
    assert len(ordered['output_hashes']) == (0 if failed_call == 1 else 2)
    assert len(ordered['max_abs_deltas']) == (0 if failed_call == 1 else 1)
    if failed_call == 4:
        assert ordered['tactic_mode'] == ('default_dispatch' if default else 'explicit')


def test_private_replay_restores_tuner_on_interrupt(rig):
    choose = rig.tuner.choose_one

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        private_moe_replay(interrupt, rig.arguments)
    assert rig.tuner.choose_one == choose


@pytest.mark.parametrize('name', ['input', 'fc1_expert_weights', 'fc2_expert_weights', 'quant_scales'])
@pytest.mark.parametrize('failed', [False, True])
def test_private_replay_detects_mutation_even_when_control_fails(rig, name, failed):
    choose = rig.tuner.choose_one

    def mutate(**kwargs):
        result = rig.original(**kwargs)
        if not kwargs['use_fused_finalize']:
            value = kwargs[name][0] if name == 'quant_scales' else kwargs[name]
            value.add_(1)
            if failed:
                raise RuntimeError('ordered control failed after mutation')
        return result
    with pytest.raises(ReplayIntegrityError, match='mutated'):
        private_moe_replay(mutate, rig.arguments)
    assert rig.tuner.choose_one == choose
