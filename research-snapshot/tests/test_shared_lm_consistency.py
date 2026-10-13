"""LM-assisted consistency integration, frozen baseline and epoch accounting."""

from collections import Counter
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fuzev_research import shared_head, shared_train as training
from fuzev_research.head import load_bundle
from fuzev_research.preference_head_retrain import track


def rows_for(role):
    rows = []
    for key in sorted(training.TRACKS):
        source, primitive, modality, subtype = key.split('/')
        for index in range(2):
            row = {'id': f'{role}/{key}/{index}', 'group_': f'{role}/{source}/group-{index}',
                   'role': role, 'source': source, 'primitive': primitive, 'modality': modality}
            if primitive == 'ordinal':
                row['attribute'] = subtype
            elif subtype != 'main':
                row['adaptation'] = subtype
            rows.append(row)
    return rows


@pytest.mark.parametrize('key', sorted(training.TRACKS))
@pytest.mark.parametrize('teacher_kl', [0., .03])
def test_all_fifteen_tracks_keep_full_gold_and_declared_teacher_kl(key, teacher_kl, monkeypatch):
    monkeypatch.setitem(training.PROFILE, 'teacher_distribution_regularization', teacher_kl)
    row = next(row for row in rows_for('fit') if track(row) == key)
    logits = torch.tensor([[.2, -.4]], requires_grad=True)
    batch = {'baseline': torch.tensor([[1.7, -.8]]), 'mask': torch.ones(1, 2, dtype=torch.bool),
             'targets': torch.tensor([[0., 1.]])}
    primitives = [row['primitive']]
    gold = training.decision_loss(logits, batch, primitives)
    expected = gold + .001 * training.correction_penalty(logits, batch) + teacher_kl * training.teacher_distribution_penalty(logits, batch)
    expected = expected + .15 * training.ordinal_cdf_penalty(logits, batch, primitives)
    actual = training.fitting_loss(logits, batch, primitives, role='fit')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tied = {**batch, 'targets': torch.tensor([[.5, .5]])}
    torch.testing.assert_close(actual - training.fitting_loss(logits, tied, primitives, role='fit'),
                               gold - training.decision_loss(logits, tied, primitives)
                               + .15 * (training.ordinal_cdf_penalty(logits, batch, primitives)
                                        - training.ordinal_cdf_penalty(logits, tied, primitives)), rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    assert batch['baseline'].grad is None and logits.grad.abs().sum() > 0


def test_gold_default_profiles_and_rank512_schema_controls():
    assert set(training.SEARCH_PROFILES) == {'key-adapter-ordinal-rps015-light-consistency-medium'}
    assert training.PROFILE == training.BASE_PROFILE | training.SEARCH_PROFILES['key-adapter-ordinal-rps015-light-consistency-medium']
    for key, value in {'seed': 43, 'learning_rate': 1e-5, 'batch_size': 256, 'max_epochs': 12,
                       'patience': 3, 'weight_decay': .01, 'clip_grad_norm': 1., 'key_gain': 1e-3,
                       'balanced_noul': False, 'fit_track_weighting': 'equal-track-macro',
                       'teacher_distribution_regularization': .03, 'ordinal_rps_coefficient': .15,
                       'correction_regularization': .001, 'preference_consistency_beta': .05}.items():
        assert training.PROFILE[key] == value
    assert 'teacher_kl_zero_tracks' not in training.PROFILE
    assert shared_head.SCHEMA_VERSION == 25
    assert shared_head.ARCHITECTURE == 'shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1'
    head = shared_head.SharedDecisionHead(seed=43)
    assert head.scorer.projection_size == 512
    assert len(tuple(head.parameters())) == 15 and head.requires_lm_scores
    assert head.lm_assistance['formula'] == shared_head.ASSISTANCE_FORMULA
    masses = training.track_masses()
    assert all(masses[key] == 1 / 15 for key in training.CHOICE_TRACKS)
    assert all(masses[key] == 1 / 15 for key in training.NOUL_TRACKS)
    for role in ('development', 'calibration'):
        old_masses = training.track_masses(role)
        assert old_masses['boolq/noul/text/boolean_truth'] == 1 / 6
        assert all(old_masses[key] == 1 / 18 for key in training.NOUL_TRACKS if not key.startswith('boolq/'))
    assert all(masses[key] == 1 / 15 for key in training.ORDINAL_TRACKS)


@pytest.mark.parametrize('permutation', [[[0, 1]], [[1, 0]]])
def test_js_ignores_independent_view_bias_shifts(permutation):
    parent, child = torch.tensor([[.7, -.2]]), torch.tensor([[-.4, .8]])
    order = torch.tensor(permutation)
    expected = training.preference_js(parent, child, order, role='fit')
    actual = training.preference_js(parent + 10., child - 7., order, role='fit')
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert 0 <= actual.item() <= 1


def test_both_views_use_their_original_frozen_lm_baselines():
    parent = {'id': 'parent', 'group_': 'group', 'role': 'fit', 'primitive': 'choice',
              'source': 'helpsteer2', 'modality': 'text'}
    child = {**parent, 'id': 'child', 'primitive': 'noul', 'adaptation': 'pairwise_preference_yes_no'}
    data = {'parent': {'query': torch.ones(1, 4), 'candidates': torch.ones(1, 2, 4),
                       'baseline': torch.tensor([[2., -1.]]), 'mask': torch.ones(1, 2, dtype=torch.bool),
                       'targets': torch.tensor([[1., 0.]])},
            'child': {'query': torch.ones(1, 4), 'candidates': torch.ones(1, 2, 4),
                      'baseline': torch.tensor([[.4, -.6]]), 'mask': torch.ones(1, 2, dtype=torch.bool),
                      'targets': torch.tensor([[0., 1.]])}}
    cache = SimpleNamespace(role='fit', data={'query': data['parent']['query']},
                            batch=lambda rows: {key: torch.cat([data[row['id']][key] for row in rows])
                                                 for key in data['parent']})
    head = shared_head.SharedDecisionHead(seed=43, hidden_size=4, projection_size=3)
    with torch.no_grad():
        for value in head.parameters():
            value.zero_()
    pair = {'child': {'parent': parent, 'child': child, 'permutation': [1, 0]}}
    actual, visited = training.batch_preference_consistency(head, cache, [child], pair)
    expected = training.preference_js(data['parent']['baseline'], data['child']['baseline'],
                                       torch.tensor([[1, 0]]), role='fit').sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert visited == ['child'] and actual > 0
    actual.backward()
    assert all(data[name]['baseline'].grad is None for name in data)
    data['child']['baseline'].requires_grad_(True)
    with pytest.raises(ValueError, match='baseline'):
        training.batch_preference_consistency(head, cache, [child], pair)


class MetadataFeatures:
    read_only = True

    def __init__(self, directory, rows):
        self.directory, self.db = directory, self
        self.identity = {'synthetic_only': True,
                         'dataset_scope': training.contract(augmentation=True, dimensions=True)}
        directory.mkdir()
        (directory / 'identity.json').write_text(json.dumps(self.identity))
        self.metadata = {}
        parents = {'helpsteer2': 'helpsteer2/choice/text/main',
                   'helpsteer3': 'helpsteer3/choice/text/main',
                   'mm-rlhf': 'mm-rlhf/choice/image/pairwise_preference'}
        for row in rows:
            count = 5 if row['primitive'] == 'ordinal' else 2
            self.metadata[row['id']] = {**row, 'group': row['group_'],
                                        'candidates': [{'id': str(index)} for index in range(count)],
                                        'target': [1.] + [0.] * (count - 1)}
        for row in rows:
            if row.get('adaptation') != 'pairwise_preference_yes_no':
                continue
            parent = next(item for item in rows if track(item) == parents[row['source']]
                          and item['group_'] == row['group_'])
            self.metadata[row['id']].update(
                parent_id=parent['id'], parent_role='fit', parent_source=parent['source'], parent_primitive='choice',
                candidate_permutation=[1, 0], parent_candidate_ids=['0', '1'], parent_target=[1., 0.],
                target=[0., 1.])

    def execute(self, sql, parameters):
        assert sql == 'SELECT metadata FROM records WHERE id=? AND role=?' and parameters[1] == 'fit'
        metadata = self.metadata[parameters[0]]
        return SimpleNamespace(fetchone=lambda: {'metadata': json.dumps(metadata)})

    def require_ready(self, mode):
        return {'synthetic_only': True}

    def producer_origins(self):
        return {'synthetic_only': True}


class SyntheticCache:
    def __init__(self, features, rows, role, device):
        assert role in {'fit', 'development'} and device == 'cpu'
        self.rows, self.role = copy.deepcopy(rows), role
        self.positions = {row['id']: index for index, row in enumerate(rows)}
        self.membership = training.membership_identity(rows)
        generator = torch.Generator().manual_seed(43)
        count = len(rows)
        self.data = {'query': torch.randn(count, 2560, generator=generator),
                     'candidates': torch.randn(count, 5, 2560, generator=generator),
                     'mask': torch.zeros(count, 5, dtype=torch.bool),
                     'targets': torch.zeros(count, 5), 'baseline': torch.randn(count, 5, generator=generator),
                     'ordinal_values': torch.zeros(count, 5)}
        for index, row in enumerate(rows):
            width = 5 if row['primitive'] == 'ordinal' else 2
            self.data['mask'][index, :width] = True
            target = features.metadata[row['id']]['target'] if role == 'fit' else [1.] + [0.] * (width - 1)
            self.data['targets'][index, :width] = torch.tensor(target)
            if row['primitive'] == 'ordinal':
                self.data['ordinal_values'][index] = torch.arange(5)

    def batch(self, rows):
        indices = [self.positions[row['id']] for row in rows]
        return {key: value[indices] for key, value in self.data.items()}

    def provenance(self):
        return {'role': self.role, 'membership': self.membership, 'synthetic_only': True}


@pytest.mark.parametrize('beta', [0., .05])
@pytest.mark.parametrize('teacher_kl', [.03])
def test_actual_trainer_preserves_gold_visits_steps_pair_counts_and_loss_logs(tmp_path, monkeypatch, beta, teacher_kl):
    roles = {role: rows_for(role) for role in ('fit', 'development', 'calibration')}
    child = next(row for row in roles['fit'] if row.get('adaptation') == 'pairwise_preference_yes_no')
    roles['fit'].append({**child, 'id': child['id'] + '/extra-view'})
    features = MetadataFeatures(tmp_path / 'features', roles['fit'])
    monkeypatch.setattr(training, 'FEATURE_SHA', training.file_sha256(features.directory / 'identity.json'))
    monkeypatch.setattr(training, 'PROFILE', {**training.PROFILE, 'max_epochs': 1,
                                           'batch_size': 7, 'preference_consistency_beta': beta,
                                           'teacher_distribution_regularization': teacher_kl})
    monkeypatch.setattr(training, 'SharedFeatureCache', SyntheticCache)
    monkeypatch.setattr(training, 'Store', lambda: SimpleNamespace(update=lambda *args, **kwargs: None))
    monkeypatch.setattr(training, 'development', lambda *args: {
        'eligible': True, 'macro_normalized_nll': {'control': 2., 'head': 1.},
        'gain_vs_label_token': 1., 'checks': []})
    original = training.batch_preference_consistency
    visits = Counter()

    def regularizer(*args):
        value, ids = original(*args)
        visits.update(ids)
        return value, ids

    monkeypatch.setattr(training, 'batch_preference_consistency', regularizer)
    output = tmp_path / 'trial'
    result = training.run_trial(features, roles, output, device='cpu', development_only=True)
    epoch = result['epochs'][0]
    assert epoch['fit_visits'] == len(roles['fit']) == 31
    assert epoch['optimizer_steps'] == math.ceil(31 / 7) == 5
    assert epoch['fit_consistency_pairs'] == (6 if beta else 0)
    assert len(visits) == (6 if beta else 0) and all(count == 1 for count in visits.values())
    expected = (epoch['fit_gold_ce'] + teacher_kl * epoch['fit_teacher_kl']
                + .001 * epoch['fit_correction_penalty'] + beta * epoch['fit_consistency_js']
                + .15 * epoch['fit_ordinal_rps'])
    assert epoch['fit_loss'] == pytest.approx(expected, rel=2e-7, abs=2e-7)
    provenance = json.loads((output / 'cache-provenance.json').read_text())['consistency']
    assert provenance['pairs'] == (6 if beta else 0) and provenance['extra_supervised_visits'] == 0
    assert provenance['role'] == 'fit' and len(provenance['pairing_sha256']) == 64
    head, manifest, _ = load_bundle(epoch['directory'])
    assert manifest['schema_version'] == 25 and head.requires_lm_scores
    assert manifest['profile'] == training.PROFILE and manifest['training_code'] == training.source_identity()
    assert result['calibration'] == 'not_started' and not (output / 'calibration-attempt.json').exists()


def test_decay_updates_every_actual_optimizer_group_before_fit_and_logs_each_epoch(tmp_path, monkeypatch, capsys):
    roles = {role: rows_for(role) for role in ('fit', 'development', 'calibration')}
    child = next(row for row in roles['fit'] if row.get('adaptation') == 'pairwise_preference_yes_no')
    roles['fit'].append({**child, 'id': child['id'] + '/extra-view'})
    features = MetadataFeatures(tmp_path / 'features', roles['fit'])
    monkeypatch.setattr(training, 'FEATURE_SHA', training.file_sha256(features.directory / 'identity.json'))
    monkeypatch.setattr(training, 'PROFILE', {**training.PROFILE, 'max_epochs': 6, 'batch_size': 7})
    monkeypatch.setattr(training, 'Store', lambda: SimpleNamespace(update=lambda *args, **kwargs: None))
    adamw = torch.optim.AdamW
    optimizers = []
    ticks = iter([epoch * 10 + phase for epoch in range(6) for phase in (0., 1., 3., 6.)])
    monkeypatch.setattr(training, 'phase_timestamp', lambda device: next(ticks))

    class ObservedAdamW(adamw):
        def step(self, *args, **kwargs):
            self.observed_rates.append([group['lr'] for group in self.param_groups])
            return super().step(*args, **kwargs)

    def make_optimizer(parameters, **kwargs):
        parameters = list(parameters)
        assert len(parameters) == 4 and [len(group['params']) for group in parameters] == [6, 3, 3, 3]
        optimizer = ObservedAdamW(parameters, **kwargs)
        for group in optimizer.param_groups:
            group['lr'] = .123
        optimizer.observed_rates = []
        optimizers.append(optimizer)
        return optimizer

    class CheckedCache(SyntheticCache):
        def batch(self, rows):
            if self.role == 'fit':
                optimizer = optimizers[0]
                epoch = len(optimizer.observed_rates) // 5 + 1
                rate = training.learning_rate_for_epoch(epoch)
                assert [group['lr'] for group in optimizer.param_groups] == [rate, rate * 10, rate * 10, rate * 10]
            return super().batch(rows)

    monkeypatch.setattr(training.torch.optim, 'AdamW', make_optimizer)
    monkeypatch.setattr(training, 'SharedFeatureCache', CheckedCache)
    calls = 0

    def development(*args):
        nonlocal calls
        calls += 1
        nll = 2. - calls * .05
        return {'eligible': True, 'macro_normalized_nll': {'control': 3., 'head': nll},
                'gain_vs_label_token': 3. - nll, 'checks': []}

    monkeypatch.setattr(training, 'development', development)
    output = tmp_path / 'trial'
    result = training.run_trial(features, roles, output, device='cpu', development_only=True)
    rates = [1e-5] * 4 + [3e-6] * 2
    assert len(optimizers) == 1 and len(result['epochs']) == 6
    assert optimizers[0].observed_rates == [[rate, rate * 10, rate * 10, rate * 10] for rate in rates for _ in range(5)]
    assert [epoch['learning_rate'] for epoch in result['epochs']] == rates
    stdout = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert next(ticks, None) is None
    for receipt in stdout + result['epochs']:
        assert {key: receipt[key] for key in ('fit_seconds', 'development_seconds', 'checkpoint_seconds', 'seconds')} == {
            'fit_seconds': 1., 'development_seconds': 2., 'checkpoint_seconds': 3., 'seconds': 6.}
    assert [line['learning_rate'] for line in stdout] == rates
    expected_groups = [{'scorer_and_gate': rate, 'query_adapter': rate * 10, 'channel_gate': rate * 10,
                        'key_adapter': rate * 10} for rate in rates]
    assert [line['group_learning_rates'] for line in stdout] == expected_groups
    assert [epoch['group_learning_rates'] for epoch in result['epochs']] == expected_groups
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['learning_rate_schedule'] == protocol['profile']['learning_rate_schedule'] == training.PROFILE['learning_rate_schedule']
    assert len(protocol['source']) == 8 and protocol['source'] == training.source_identity()
    assert protocol['optimization'] == training.optimization_protocol()
    assert training.PROFILE['learning_rate'] == 1e-5
    for index, epoch in enumerate(result['epochs']):
        assert epoch['fit_visits'] == 31 and epoch['optimizer_steps'] == 5 * (index + 1)
        assert epoch['fit_consistency_pairs'] == 6
        assert epoch['fit_loss'] == pytest.approx(epoch['fit_gold_ce'] + training.PROFILE['teacher_distribution_regularization'] * epoch['fit_teacher_kl']
                                                + .001 * epoch['fit_correction_penalty'] + .05 * epoch['fit_consistency_js']
                                                + .15 * epoch['fit_ordinal_rps'],
                                                rel=2e-7, abs=2e-7)
        manifest = json.loads((Path(epoch['directory']) / 'manifest.json').read_text())
        assert manifest['learning_rate'] == rates[index] and manifest['schema_version'] == 25
        assert manifest['projection_size'] == 512 and manifest['profile'] == training.PROFILE
        assert manifest['group_learning_rates'] == expected_groups[index]
        assert manifest['optimization'] == protocol['optimization']
    assert result['calibration'] == 'not_started' and not (output / 'calibration-attempt.json').exists()
