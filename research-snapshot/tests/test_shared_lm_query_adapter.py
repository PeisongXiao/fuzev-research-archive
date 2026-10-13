"""R16A nested query adapter; all inputs, artifacts and admissions are synthetic."""

import copy
import json
import math

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from fuzev_research import shared_head, shared_qualification, shared_train as training
from fuzev_research.head import CandidateHead, load_bundle
from test_shared_lm_rank512 import metadata, repin, sha
from test_shared_qualification import manifest as qualified_metadata


PARENT_ARCHITECTURE = 'shared-lm-assisted-context-trust-candidate-dot-product-rank512-v1'


def inputs(hidden=4, count=5):
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(2, hidden, generator=generator)
    candidates = torch.randn(2, count, hidden, generator=generator)
    baseline = torch.randn(2, count, generator=generator)
    mask = torch.ones(2, count, dtype=torch.bool)
    mask[0, 1::2], mask[1, 2::2] = False, False
    return query, candidates, baseline, mask


def centered(value, mask):
    value = value.masked_fill(~mask, 0)
    return (value - value.sum(1, keepdim=True) / mask.sum(1, keepdim=True)).masked_fill(~mask, 0)


@pytest.fixture
def small():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    with torch.no_grad():
        head.adapter_up.weight.copy_(torch.randn(3, 64, generator=torch.Generator().manual_seed(6)) * .3)
        head.adapter_down.bias.fill_(.15)
        head.gate.weight.copy_(torch.tensor([[.2, -.3, .4, -.1]]))
        head.gate.bias.fill_(.17)
    return head


@pytest.fixture(scope='module')
def production():
    head = shared_head.SharedDecisionHead(seed=43)
    with torch.no_grad():
        head.adapter_up.weight.copy_(torch.randn(512, 64, generator=torch.Generator().manual_seed(6)) * .03)
        head.adapter_down.bias.fill_(.15)
        head.gate.weight[0, 0] = .25
        head.gate.bias.fill_(-.3)
    return head


@pytest.fixture
def artifact(tmp_path, production):
    directory = tmp_path / 'query-adapter'
    shared_head.save_shared_bundle(directory, production, metadata(), {name: 1. for name in training.PRIMITIVES})
    return directory


@pytest.mark.parametrize('hidden,rank', [(4, 3), (2560, 512)])
@pytest.mark.parametrize('learned_gate', [False, True])
def test_zero_up_exact_parent_function_draws_and_seeded_adapter(hidden, rank, learned_gate):
    rng = torch.random.get_rng_state().clone()
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=hidden, projection_size=rank)
    assert torch.equal(rng, torch.random.get_rng_state())
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(43)
        parent = CandidateHead(hidden, rank).float()
        nn.init.xavier_uniform_(parent.query.weight, gain=1.)
        nn.init.xavier_uniform_(parent.key.weight, gain=.001)
        nn.init.zeros_(parent.query.bias)
        nn.init.zeros_(parent.key.bias)
        gate = nn.Linear(hidden, 1).float()
        nn.init.zeros_(gate.weight)
        nn.init.zeros_(gate.bias)
        down = nn.Linear(hidden, 64).float()
        nn.Linear(64, rank, bias=False).float()
        nn.init.xavier_uniform_(down.weight, gain=1.)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in parent.state_dict().items():
        assert torch.equal(value, head.scorer.state_dict()[key])
    for key, value in gate.state_dict().items():
        assert torch.equal(value, head.gate.state_dict()[key])
    assert torch.equal(down.weight, head.adapter_down.weight)
    assert head.adapter_down.weight.abs().max() <= math.sqrt(6 / (hidden + 64))
    assert torch.count_nonzero(head.adapter_down.bias) == torch.count_nonzero(head.adapter_up.weight) == 0
    assert head.adapter_up.bias is None
    if learned_gate:
        with torch.no_grad():
            gate.weight.fill_(.02)
            gate.bias.fill_(-.3)
            head.gate.load_state_dict(gate.state_dict())
    query, candidates, baseline, mask = inputs(hidden)
    expected = (baseline + centered(parent(query, candidates), mask)
                + (2 * gate(query).sigmoid() - 1) * centered(baseline, mask)).masked_fill(~mask, -torch.inf)
    for primitive in training.PRIMITIVES:
        assert torch.equal(head(primitive, query, candidates, mask, baseline=baseline), expected)


def test_up_learns_first_and_down_weight_bias_get_signal_after_first_adamw_step():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    query, candidates, baseline, mask = inputs()
    target = torch.tensor([2, 1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-5, weight_decay=.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head('choice', query, candidates, mask, baseline=baseline), target).backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
        for parameter in (head.scorer.query.weight, head.scorer.key.weight, head.adapter_up.weight,
                          head.gate.weight, head.gate.bias):
            assert parameter.grad.abs().sum() > 0
        for parameter in head.adapter_down.parameters():
            assert bool(parameter.grad.abs().sum() > 0) == (step > 0)
        assert baseline.grad is None
        nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert head.adapter_up.weight.abs().sum() > 0
    assert head.adapter_down.bias.abs().sum() > 0


def test_nonlinear_query_changes_residual_direction_and_candidate_ranking():
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=2, projection_size=2)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.scorer.query.bias[1] = 1.
        head.scorer.key.weight.copy_(torch.eye(2))
        head.adapter_down.weight[0, 0] = head.adapter_up.weight[0, 0] = 1.
    query = torch.tensor([[-2., 0.], [0., 0.], [2., 0.]])
    candidates = torch.eye(2).expand(3, -1, -1)
    baseline = torch.zeros(3, 2)
    output = head('choice', query, candidates, baseline=baseline)
    assert head.scorer(query, candidates).argmax(1).tolist() == [1, 1, 1]
    assert output.argmax(1).tolist() == [1, 1, 0]
    assert not torch.allclose(output[0] + output[2], 2 * output[1])


def test_learned_adapter_permutation_padding_shift_and_masked_gradients(small):
    query, candidates, baseline, mask = inputs()
    candidates.requires_grad_(True)
    original = small('choice', query, candidates, mask, baseline=baseline)
    order = torch.tensor([3, 0, 4, 1, 2])
    actual = small('noul', query, candidates[:, order], mask[:, order], baseline=baseline[:, order])
    torch.testing.assert_close(actual, original[:, order], rtol=1e-6, atol=1e-7)
    padded_candidates = torch.cat((candidates.detach(), torch.full((2, 3, 4), 1e6)), dim=1)
    padded_baseline = torch.cat((baseline, torch.full((2, 3), torch.finfo(torch.float32).max)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), dim=1)
    padded = small('ordinal', query, padded_candidates, padded_mask, baseline=padded_baseline)
    torch.testing.assert_close(padded[:, :5], original, rtol=1e-6, atol=1e-7)
    assert torch.isneginf(padded[:, 5:]).all()
    shifted = small('choice', query, candidates, mask, baseline=baseline + 7.)
    torch.testing.assert_close(shifted.softmax(1), original.softmax(1), rtol=1e-6, atol=1e-7)
    F.cross_entropy(original, torch.tensor([2, 1])).backward()
    assert torch.count_nonzero(candidates.grad[~mask]) == 0 and candidates.grad[mask].abs().sum() > 0
    assert baseline.grad is None


@pytest.mark.parametrize('case', ['missing_down', 'missing_up', 'down_shape', 'down_no_bias', 'up_shape', 'up_bias',
                                 'dtype', 'device', 'nan_down', 'inf_up', 'nan_bias', 'overflow', 'identity'])
def test_invalid_adapter_geometry_or_nonfinite_arithmetic_fails_closed(small, case):
    query, candidates, baseline, mask = inputs()
    with torch.no_grad():
        if case.startswith('missing_'):
            delattr(small, 'adapter_' + case.removeprefix('missing_'))
        elif case == 'down_shape':
            small.adapter_down = nn.Linear(4, 8)
        elif case == 'down_no_bias':
            small.adapter_down = nn.Linear(4, 64, bias=False)
        elif case == 'up_shape':
            small.adapter_up = nn.Linear(64, 4, bias=False)
        elif case == 'up_bias':
            small.adapter_up = nn.Linear(64, 3)
        elif case == 'dtype':
            small.adapter_down.half()
        elif case == 'device':
            small.adapter_up.to('meta')
        elif case == 'nan_down':
            small.adapter_down.weight[0, 0] = torch.nan
        elif case == 'inf_up':
            small.adapter_up.weight[0, 0] = torch.inf
        elif case == 'nan_bias':
            small.adapter_down.bias[0] = torch.nan
        elif case == 'overflow':
            query.fill_(2.)
            small.adapter_down.weight.fill_(torch.finfo(torch.float32).max)
            small.adapter_up.weight.zero_()
        else:
            small.adapter_down = nn.Identity()
    with pytest.raises(ValueError):
        small('choice', query, candidates, mask, baseline=baseline)


@pytest.mark.parametrize('loader', [load_bundle, shared_head.load_shared_bundle])
@pytest.mark.parametrize('change', [{'schema_version': 10}, {'schema_version': 12.},
                                   {'architecture': PARENT_ARCHITECTURE, 'schema_version': 10},
                                   {'architecture': PARENT_ARCHITECTURE, 'schema_version': 12},
                                   {'query_adapter_size': 8}, {'query_adapter_size': 16.}])
def test_schema10_or_wrong_adapter_geometry_rejected_before_weights(artifact, monkeypatch, loader, change):
    repin(artifact, **change)
    monkeypatch.setattr(shared_head, 'load', lambda *args: pytest.fail('Invalid adapter reached weight decoding'))
    with pytest.raises(ValueError, match='Incompatible'):
        loader(artifact)


@pytest.mark.parametrize('field,value', [('adapter_down_gain', .001), ('adapter_down_gain', True),
                                       ('adapter_down_bias', .01), ('adapter_down_bias', False),
                                       ('adapter_up_weight', .001), ('adapter_up_weight', False),
                                       ('adapter_up_bias', 0), ('adapter_up_bias', True),
                                       ('scheme', 'xavier-query-small-key-zero-context-gate-v1')])
def test_adapter_initialization_cannot_be_relabelled_on_save_or_load(artifact, production, tmp_path, field, value):
    initialization = dict(production.initialization, **{field: value})
    with pytest.raises(ValueError, match='initialization'):
        shared_head.save_shared_bundle(tmp_path / 'denied', production, metadata() | {'initialization': initialization},
                                       {name: 1. for name in training.PRIMITIVES})
    assert not (tmp_path / 'denied').exists()
    repin(artifact, initialization=initialization)
    with pytest.raises(ValueError, match='initialization'):
        load_bundle(artifact)


@pytest.mark.parametrize('case', ['schema10_six_tensors', 'missing_down_bias', 'extra_up_bias', 'wrong_down_shape',
                                 'wrong_up_shape', 'half', 'nan', 'extra_head', 'lm_copy'])
def test_adapter_state_tampering_rejected_with_recomputed_hash(artifact, case):
    path = artifact / 'head.safetensors'
    state = load_file(str(path))
    if case == 'schema10_six_tensors':
        state = {key: value for key, value in state.items() if not key.startswith('adapter_')}
    elif case == 'missing_down_bias':
        del state['adapter_down.bias']
    elif case == 'extra_up_bias':
        state['adapter_up.bias'] = torch.zeros(512)
    elif case == 'wrong_down_shape':
        state['adapter_down.weight'] = state['adapter_down.weight'][:8].contiguous()
    elif case == 'wrong_up_shape':
        state['adapter_up.weight'] = state['adapter_up.weight'].t().contiguous()
    elif case == 'half':
        state['adapter_down.weight'] = state['adapter_down.weight'].half()
    elif case == 'nan':
        state['adapter_up.weight'][0, 0] = torch.nan
    else:
        state['output.weight' if case == 'extra_head' else 'lm_head.weight'] = torch.zeros(1, 1)
    save_file(state, str(path))
    repin(artifact, weights_sha256=sha(path))
    for loader in (load_bundle, shared_head.load_shared_bundle):
        with pytest.raises(ValueError, match='state|parameters'):
            loader(artifact)


@pytest.mark.parametrize('case', ['missing_up', 'up_bias', 'down_geometry', 'nan_up', 'old_schema', 'old_formula',
                                 'wrong_adapter_size', 'up_init_bool', 'extra_head'])
def test_qualification_rechecks_adapter_state_and_provenance(production, case):
    head = copy.deepcopy(production)
    manifest, temperatures = qualified_metadata()
    if case == 'missing_up':
        del head.adapter_up
    elif case == 'up_bias':
        head.adapter_up = nn.Linear(64, 512)
    elif case == 'down_geometry':
        head.adapter_down.out_features = 8
    elif case == 'nan_up':
        with torch.no_grad():
            head.adapter_up.weight[0, 0] = torch.nan
    elif case == 'old_schema':
        manifest['schema_version'] = 10
    elif case == 'old_formula':
        manifest['lm_assistance'].pop('raw_formula')
    elif case == 'wrong_adapter_size':
        manifest['query_adapter_size'] = 8
    elif case == 'up_init_bool':
        manifest['initialization']['adapter_up_weight'] = False
    else:
        head.extra = nn.Linear(4, 1)
    with pytest.raises(ValueError, match='Shared reference:'):
        shared_qualification.validate_shared_reference(head, manifest, temperatures)


@pytest.mark.parametrize('filename', sorted(shared_head.CODE_FILES))
def test_all_eight_source_files_mandatory_even_for_self_consistent_manifest(artifact, filename):
    manifest = json.loads((artifact / 'manifest.json').read_text())
    for field in ('training_code', 'shared_training_code'):
        manifest[field].pop(filename)
    repin(artifact, **manifest)
    with pytest.raises(ValueError, match='code provenance'):
        load_bundle(artifact)
