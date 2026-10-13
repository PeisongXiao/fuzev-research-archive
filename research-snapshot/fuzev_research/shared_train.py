"""Fresh CUDA training of a single option-conditioned decision head.

One parameter set and optimizer receive all three tasks and both modalities.
CAL is read only after DEV accepts and locks the shared checkpoint. No LLM loads.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .acquire import write_json
from .dimension_augmentation import ADAPTATIONS as DIMENSION_ADAPTATIONS
from .features import FeatureStore, file_sha256
from .metrics import paired_group_interval, per_record
from .paths import project_path
from .preference_head_retrain import CHOICE_TRACKS as ORIGINAL_CHOICE_TRACKS
from .preference_head_retrain import NOUL_TRACKS, check_rows, read_roles, track
from .scope import contract
from .shared_head import SharedDecisionHead, save_shared_bundle, load_shared_bundle
from .shared_head import (ARCHITECTURE, _adapter_geometry, _key_adapter_geometry,
                          KEY_ADAPTER_SIZE, KEY_ADAPTER_PARENT_COMMIT, KEY_ADAPTER_REFERENCE_COMMIT)
from .shared_training_data import SharedFeatureCache
from .state import Store
from .train import membership_identity


PRIMITIVES = ('choice', 'noul', 'ordinal')
DIMENSION_CHOICE_TRACKS = {
    f'{source}/choice/{modality}/{DIMENSION_ADAPTATIONS[source]}'
    for source, modality in (('mm-rlhf', 'image'), ('helpsteer2', 'text'))
}
CHOICE_TRACKS = ORIGINAL_CHOICE_TRACKS | DIMENSION_CHOICE_TRACKS
ORDINAL_TRACKS = {f'helpsteer2/ordinal/text/{name}' for name in
                  ('helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity')}
TRACKS = CHOICE_TRACKS | NOUL_TRACKS | ORDINAL_TRACKS
PROFILE = {'seed': 43, 'learning_rate': 1e-5, 'batch_size': 256,
           'max_epochs': 8, 'patience': 2, 'minimum_improvement': 1e-4,
           'weight_decay': .01, 'clip_grad_norm': 1.,
           'loss': 'normalized candidate cross-entropy plus correction mean-square and FIT-only normalized teacher KL',
           'correction_regularization': .01,
           'teacher_distribution_regularization': 0.,
           'preference_consistency_beta': 0.,
           'bootstrap_draws': 2000, 'bootstrap_seed': 17}
BASE_PROFILE = dict(PROFILE)
BASE_PROFILE['key_gain'] = 1e-3
BASE_PROFILE['balanced_noul'] = False
SEARCH_PROFILES = {
    'key-adapter-ordinal-rps015-light-consistency-medium': {
        'balanced_noul': False, 'fit_track_weighting': 'equal-track-macro',
        'ordinal_rps_coefficient': .15,
        'teacher_distribution_regularization': .03, 'correction_regularization': .001,
        'max_epochs': 12, 'patience': 3, 'preference_consistency_beta': .05,
        'adapter_learning_rate_multiplier': 10.,
        'channel_gate_learning_rate_multiplier': 10.,
        'key_adapter_learning_rate_multiplier': 10.,
        'optimizer_fused': True, 'optimizer_foreach': False,
        'learning_rate_schedule': {
            'epoch_indexing': 'one-based', 'update': 'before-epoch-FIT',
            'stages': [{'start_epoch': 1, 'learning_rate': 1e-5},
                       {'start_epoch': 5, 'learning_rate': 3e-6}],
        },
    },
}
PROFILE.update(BASE_PROFILE | SEARCH_PROFILES['key-adapter-ordinal-rps015-light-consistency-medium'])
SOURCE_PARENT_COMMIT = KEY_ADAPTER_PARENT_COMMIT
RETAINED_CONTROL_COMMIT = KEY_ADAPTER_PARENT_COMMIT
FEATURES = 'data/features/native-4.05-dimension-choice-v4'
FEATURE_SHA = '498ae19fde58540c0961425bec00e5f2eead1f4fa8539631415750be99a9d69a'
TASK = 'training.shared_lm_assisted'
OPTIMIZER_GROUP_PARAMETERS = {
    'scorer_and_gate': ('scorer.query.weight', 'scorer.query.bias', 'scorer.key.weight',
                        'scorer.key.bias', 'gate.weight', 'gate.bias'),
    'query_adapter': ('adapter_down.weight', 'adapter_down.bias', 'adapter_up.weight'),
    'channel_gate': ('channel_a', 'channel_b', 'channel_c'),
    'key_adapter': ('key_adapter_down.weight', 'key_adapter_down.bias', 'key_adapter_up.weight'),
}
FUSED_ADAMW_TOLERANCES = {'parameters': {'atol': 2e-6, 'rtol': 2e-5},
                        'moments': {'atol': 1e-7, 'rtol': 2e-5}}


def require_feature_pin(features=None):
    return file_sha256(features.directory / 'identity.json') if features is not None else FEATURE_SHA


def source_identity():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in
            ('shared_train.py', 'shared_head.py', 'shared_training_data.py',
             'features.py', 'metrics.py', 'scope.py', 'preference_head_retrain.py', 'head.py')}


def learning_rate_for_epoch(epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Learning-rate schedule requires a positive one-based epoch')
    schedule = PROFILE.get('learning_rate_schedule')
    if (not isinstance(schedule, dict) or set(schedule) != {'epoch_indexing', 'update', 'stages'}
            or schedule['epoch_indexing'] != 'one-based' or schedule['update'] != 'before-epoch-FIT'):
        raise ValueError('Invalid epoch learning-rate schedule')
    stages = schedule['stages']
    if (not isinstance(stages, list) or len(stages) != 2
            or any(not isinstance(stage, dict) or set(stage) != {'start_epoch', 'learning_rate'}
                   or type(stage['start_epoch']) is not int or type(stage['learning_rate']) is not float
                   or not math.isfinite(stage['learning_rate']) or stage['learning_rate'] <= 0
                   for stage in stages)
            or [stage['start_epoch'] for stage in stages] != [1, 5]
            or [stage['learning_rate'] for stage in stages] != [1e-5, 3e-6]
            or type(PROFILE.get('learning_rate')) is not float or PROFILE['learning_rate'] != stages[0]['learning_rate']):
        raise ValueError('Invalid epoch learning-rate schedule; require 1e-5 for epochs 1-4 and 3e-6 thereafter')
    return stages[0 if epoch < 5 else 1]['learning_rate']


def optimization_protocol():
    if (type(PROFILE.get('ordinal_rps_coefficient')) is not float or PROFILE['ordinal_rps_coefficient'] != .15
            or 'gold_target_smoothing' in PROFILE):
        raise ValueError('Require FIT-only ordinal RPS coefficient 0.15 and original unsmoothed gold')
    if PROFILE.get('balanced_noul') is not False or PROFILE.get('fit_track_weighting') != 'equal-track-macro':
        raise ValueError('Require the predeclared FIT-only equal-track-macro weighting and balanced_noul=False')
    teacher_kl = PROFILE.get('teacher_distribution_regularization')
    if type(teacher_kl) is not float or teacher_kl != .03:
        raise ValueError('Require the predeclared rank64 additive-family teacher KL weight 0.03')
    if PROFILE.get('optimizer_fused') is not True or PROFILE.get('optimizer_foreach') is not False:
        raise ValueError('Require native AdamW fused=True and foreach=False')
    multiplier = PROFILE.get('adapter_learning_rate_multiplier')
    if type(multiplier) is not float or multiplier != 10.:
        raise ValueError('Require the predeclared adapter learning-rate multiplier 10.0')
    channel_multiplier = PROFILE.get('channel_gate_learning_rate_multiplier')
    if type(channel_multiplier) is not float or channel_multiplier != 10.:
        raise ValueError('Require the predeclared channel-gate learning-rate multiplier 10.0')
    key_multiplier = PROFILE.get('key_adapter_learning_rate_multiplier')
    if type(key_multiplier) is not float or key_multiplier != 10.:
        raise ValueError('Require the predeclared key-adapter learning-rate multiplier 10.0')
    learning_rate_for_epoch(1)
    multipliers = {'scorer_and_gate': 1., 'query_adapter': multiplier, 'channel_gate': channel_multiplier,
                   'key_adapter': key_multiplier}
    groups = [{'name': name, 'parameter_names': list(parameters),
               'lr_multiplier': multipliers[name]}
              for name, parameters in OPTIMIZER_GROUP_PARAMETERS.items()]
    return {'optimizer': 'AdamW', 'implementation': 'native-fused-disjoint-parameter-groups-v1',
            'optimizer_flags': {'fused': True, 'foreach': False},
            'numerical_admission': {
                'control_commit': 'e2b88f1d3bd09a17e3651046cd3663574ae68707',  # unpublished lab commit ID
                'torch_version': '2.13.0+cu130',
                'reference_flags': {'fused': False, 'foreach': False},
                'reference_flags_provenance': 'Provenance-only: actual check_additive_channel_kl_cuda.py replay uses fused=False, foreach=False; training optimizer unchanged.',
                'tolerances': FUSED_ADAMW_TOLERANCES,
                'review': 'Pending source-specific MAIN synthetic CUDA admission; prior source receipts do not admit this source; no bitwise optimizer equivalence claim',
            },
            'epoch_timing': {
                'clock': 'monotonic wall seconds; CUDA synchronized only at phase boundaries',
                'fit_seconds': 'shuffle, FIT updates, exact-once guards and component transfer',
                'development_seconds': 'DEV forward, metrics and bootstrap',
                'checkpoint_seconds': 'checkpoint serialization and head SHA256',
                'seconds': 'sum of the three phases; excludes setup, LR update, selection and receipt writes',
            },
            'parameter_groups': groups,
            'learning_rate_schedule': [
                {'start_epoch': stage['start_epoch'], 'base_learning_rate': stage['learning_rate'],
                 'group_learning_rates': {group['name']: stage['learning_rate'] * group['lr_multiplier']
                                         for group in groups}}
                for stage in PROFILE['learning_rate_schedule']['stages']],
            'schedule_update': 'before-epoch-FIT', 'weight_decay': PROFILE['weight_decay'],
            'decay_semantics': 'decoupled AdamW; each group uses its actual learning rate',
            'gradient_scaling': False, 'output_scaling': False, 'resume': False,
            'clipping': 'one global norm over all fifteen parameters before one optimizer step',
            'persisted_moment_updates_per_step': 1, 'additional_supervised_visits': 0,
            'ordinal_rps': {
                'coefficient': .15, 'roles': ['fit'], 'primitives': ['ordinal'],
                'formula': 'sum(j<K-1, cumsum(masked_softmax(logits)-original_soft_gold)[j]^2)/(K-1)',
                'levels': [0, 1, 2, 3, 4], 'level_admission': 'owned FIT cache on CPU before GPU upload',
                'gold_loss_coefficient': 1., 'divide_by_one_plus_auxiliary': False,
                'normalization': 'existing FIT track/group weights; mean over all FIT rows, nonordinal zero',
                'fit_ordinal_rps': 'same per-row RPS vector used for updates; no extra softmax/head forward',
                'stored_targets_changed': False, 'dev_cal_evaluation_targets_changed': False,
                'js_pair_admission_targets_changed': False},
            'fit_dev_hypothesis': {
                'source_parent_commit': SOURCE_PARENT_COMMIT,
                'control_commit': RETAINED_CONTROL_COMMIT,
                'control_profile': 'ordinal-rps015-light-consistency-medium',
                'control_dev_macro_normalized_nll': .6827463680556892,
                'retained_geometry_source_commit': 'ff13fbf4e44a78aae8377e8233e9774266e627a3',  # unpublished lab commit ID
                'basis': 'Structural transfer based only on the feature-only F42 eligible DEV gain: test a candidate-side nonlinear key residual on the retained assisted A41 scorer.',
                'comparison': 'Append one rank16 key adapter and its actual LR10x group; preserve every A41 loss, FIT share1/15, original DEV/CAL masses and targets, existing optimizer group, visit and schedule.',
                'key_adapter_size': KEY_ADAPTER_SIZE, 'key_adapter_learning_rate_multiplier': key_multiplier,
                'key_adapter_source_only_reference_commit': KEY_ADAPTER_REFERENCE_COMMIT,
                'parent_initial_scores_preserved': True, 'parent_initial_tensors_preserved': True,
                'parent_jacobian_preserved': False, 'parent_clipping_norm_preserved': False,
                'query_adapter_size': 64, 'control_query_adapter_size': 64,
                'balanced_noul': False, 'fit_track_weighting': 'equal-track-macro',
                'fit_track_weighting_roles': ['fit'],
                'teacher_distribution_regularization': teacher_kl,
                'ordinal_rps_coefficient': .15, 'ordinal_rps_roles': ['fit'],
                'intervention_control_commit': SOURCE_PARENT_COMMIT,
                'intervention_control_coefficient': .15,
                'promotion_minimum_gain': 1e-4, 'promotion_required_dev_checks': 26,
                'effective_profile': 'Matched retained A41 with the key adapter only; unsmoothed original gold, RPS0.15, teacher/correction/JS and original visits remain unchanged. Parent initialization outputs are preserved, but the new Jacobian can change global clipping.',
                'predictions': ['Cross-channel candidate-key features may improve DEV without another output head.',
                                'Extra FIT capacity may overfit; no width/rate sweep or presumed generalization.'],
                'caveat': 'Group fairness and teacher KL/correction/JS terms remain, so the entire training objective is not equal to the DEV macro. Coefficient is predeclared, not fitted or claimed optimal; no external, CAL, TEST or E2E selection evidence or presumed retention.',
            }}


def _optimizer_parameters(bundle):
    named = dict(bundle.named_parameters(remove_duplicate=False))
    expected = {name for names in OPTIMIZER_GROUP_PARAMETERS.values() for name in names}
    if (sum(map(len, OPTIMIZER_GROUP_PARAMETERS.values())) != len(expected) or len(expected) != 15
            or set(named) != expected or len({id(parameter) for parameter in named.values()}) != len(expected)
            or any(parameter.dtype != torch.float32 or not parameter.requires_grad for parameter in named.values())):
        raise ValueError('Require exactly fifteen distinct trainable FP32 scorer/gate/adapter parameters')
    _adapter_geometry(bundle)
    _key_adapter_geometry(bundle)
    return named


def make_optimizer(bundle):
    protocol = optimization_protocol()
    named = _optimizer_parameters(bundle)
    initial_rates = protocol['learning_rate_schedule'][0]['group_learning_rates']
    groups = [{**group, 'params': [named[name] for name in group['parameter_names']],
               'lr': initial_rates[group['name']]} for group in protocol['parameter_groups']]
    return torch.optim.AdamW(groups, lr=learning_rate_for_epoch(1), weight_decay=PROFILE['weight_decay'],
                             **protocol['optimizer_flags'])


def set_optimizer_learning_rates(optimizer, bundle, epoch):
    base_rate = learning_rate_for_epoch(epoch)
    expected = {group['name']: group for group in optimization_protocol()['parameter_groups']}
    named = _optimizer_parameters(bundle)
    groups = optimizer.param_groups
    if len(groups) != 4 or {group.get('name') for group in groups} != set(expected):
        raise ValueError('Require exactly the four declared optimizer parameter groups')
    rates = {}
    for group in groups:
        spec = expected[group['name']]
        parameters = group.get('params')
        if (group.get('parameter_names') != spec['parameter_names']
                or type(group.get('lr_multiplier')) is not float or group['lr_multiplier'] != spec['lr_multiplier']
                or not isinstance(parameters, list)
                or [id(parameter) for parameter in parameters] != [id(named[name]) for name in spec['parameter_names']]
                or group.get('weight_decay') != PROFILE['weight_decay']
                or group.get('fused') is not True or group.get('foreach') is not False):
            raise ValueError('Optimizer groups must be disjoint, exhaustive and match their declared multipliers/decay')
        rates[group['name']] = base_rate * group['lr_multiplier']
    # Validate all groups before changing any, and never reset their moments at the boundary.
    for group in groups:
        group['lr'] = rates[group['name']]
    return rates


def phase_timestamp(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.monotonic()


def track_masses(role='fit'):
    if role not in ('fit', 'development', 'calibration'):
        raise ValueError('Track masses require a supported training role')
    if role == 'fit':
        if PROFILE.get('balanced_noul') is not False or PROFILE.get('fit_track_weighting') != 'equal-track-macro':
            raise ValueError('Require the predeclared FIT-only equal-track-macro weighting and balanced_noul=False')
        return {key: 1 / len(TRACKS) for key in TRACKS}
    noul = {key: 1 / 6 if key.startswith('boolq/') else 1 / 18 for key in NOUL_TRACKS}
    return {**{key: 1 / 18 for key in CHOICE_TRACKS},
            **noul,
            **{key: 1 / 15 for key in ORDINAL_TRACKS}}


def fitting_weights(rows):
    check_rows(rows, 'fit')
    return role_weights(rows)


def role_weights(rows):
    if not rows:
        raise ValueError('Need nonempty training-role rows')
    check_rows(rows, rows[0]['role'])
    counts, groups = Counter(), defaultdict(set)
    for row in rows:
        key = track(row)
        counts[key, row['group_']] += 1
        groups[key].add(row['group_'])
    masses = track_masses(rows[0]['role'])
    if set(groups) != set(masses):
        raise ValueError('Joint fitting requires every declared task track')
    weights = {row['id']: len(rows) * masses[track(row)] /
               (len(groups[track(row)]) * counts[track(row), row['group_']]) for row in rows}
    if not math.isclose(sum(weights.values()), len(rows), rel_tol=1e-9):
        raise ValueError('Invalid joint fitting weight normalization')
    return weights


def fit_shared_temperature(logits, targets, mask, weights):
    if (weights.shape != (len(logits),) or not torch.isfinite(weights).all()
            or not (weights > 0).all()):
        raise ValueError('Need positive finite calibration weights')
    finite = logits.detach().cpu().masked_fill(~mask, 0)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(.03), math.log(100)).exp()
        scaled = (finite / temperature).masked_fill(~mask, -1e9)
        losses = -(targets * torch.log_softmax(scaled, dim=1)).sum(dim=1) / mask.sum(dim=1).float().log()
        loss = (losses * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite shared calibration loss')
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().clamp(math.log(.03), math.log(100)).exp().item()


def macro_interval(evidence, *, draws=2000, seed=17):
    keys = sorted(evidence)
    groups = sorted({group for entry in evidence.values() for group in entry['groups']})
    positions = {group: index for index, group in enumerate(groups)}
    totals, counts = np.zeros((len(groups), len(keys))), np.zeros((len(groups), len(keys)))
    for column, key in enumerate(keys):
        entry = evidence[key]
        before, after = (np.asarray(entry[method]['normalized_nll']) for method in ('control', 'head'))
        delta = after - before
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite paired DEV NLL')
        indices = np.asarray([positions[group] for group in entry['groups']])
        totals[:, column] = np.bincount(indices, weights=delta, minlength=len(groups))
        counts[:, column] = np.bincount(indices, minlength=len(groups))
    rng, samples = np.random.default_rng(seed), []
    for _ in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        denominators = multiplicity @ counts
        if (denominators == 0).any():
            raise ValueError('Group bootstrap omitted a complete DEV track')
        samples.append(float(np.mean((multiplicity @ totals) / denominators)))
    low, high = np.quantile(samples, [.025, .975])
    return {'difference': float(np.mean(totals.sum(axis=0) / counts.sum(axis=0))),
            'low': float(low), 'high': float(high), 'groups': len(groups), 'draws': draws, 'seed': seed,
            'method': 'paired shared-group bootstrap; equal task-track means'}


def decision_loss(logits, batch, primitives):
    if len(primitives) != logits.shape[0] or not set(primitives) <= set(PRIMITIVES):
        raise ValueError('Invalid joint task batch')
    mask, target = batch['mask'], batch['targets']
    logp = torch.log_softmax(logits, dim=1).masked_fill(~mask, 0)
    ce = -(target * logp).sum(dim=1) / mask.sum(dim=1).float().log()
    return ce


def correction_penalty(logits, batch):
    mask = batch['mask']
    baseline = batch['baseline']
    if baseline.requires_grad or baseline.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError('Need aligned frozen LM scores and valid candidate masks')
    # Exclude padding before arithmetic; it must not shrink or dominate the penalty.
    delta = logits.masked_fill(~mask, 0) - baseline.masked_fill(~mask, 0)
    counts = mask.sum(dim=1)
    delta = (delta - delta.sum(dim=1, keepdim=True) / counts[:, None]).masked_fill(~mask, 0)
    return delta.square().sum(dim=1) / counts


def teacher_distribution_penalty(logits, batch):
    baseline, mask = batch['baseline'], batch['mask']
    if (logits.ndim != 2 or baseline.shape != logits.shape or mask.shape != logits.shape
            or mask.dtype != torch.bool or baseline.device != logits.device or mask.device != logits.device
            or not logits.is_floating_point() or baseline.dtype != logits.dtype):
        raise ValueError('Need aligned floating teacher/student scores and boolean candidate masks')
    counts = mask.sum(dim=1)
    if not (counts >= 2).all():
        raise ValueError('Teacher KL requires at least two valid candidates per row')
    if not torch.isfinite(baseline[mask]).all() or not torch.isfinite(logits[mask]).all():
        raise ValueError('Nonfinite valid teacher/student scores')
    # Mask before normalization and detach the cached T=1 native LM teacher.
    teacher_logp = torch.log_softmax(baseline.detach().masked_fill(~mask, -torch.inf), dim=1)
    student_logp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    teacher_logp = teacher_logp.masked_fill(~mask, 0)
    student_logp = student_logp.masked_fill(~mask, 0)
    teacher_p = teacher_logp.exp().masked_fill(~mask, 0)
    divergence = (teacher_p * (teacher_logp - student_logp)).sum(dim=1)
    return divergence.clamp_min(0) / counts.to(logits.dtype).log()


def ordinal_cdf_penalty(logits, batch, primitives):
    mask, target = batch['mask'], batch['targets'].detach()
    if (logits.ndim != 2 or len(primitives) != len(logits) or not set(primitives) <= set(PRIMITIVES)
            or mask.shape != logits.shape or target.shape != logits.shape or mask.dtype != torch.bool
            or target.dtype != logits.dtype or mask.device != logits.device or target.device != logits.device
            or not logits.is_floating_point()):
        raise ValueError('Need aligned original soft gold, boolean masks and declared primitives')
    counts = mask.sum(dim=1)
    probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1).masked_fill(~mask, 0)
    cumulative = (probabilities - target.masked_fill(~mask, 0)).cumsum(dim=1)
    thresholds = torch.arange(logits.shape[1], device=logits.device)[None, :] < counts[:, None] - 1
    penalty = cumulative.square().masked_fill(~thresholds, 0).sum(dim=1) / (counts - 1)
    ordinal = torch.tensor([primitive == 'ordinal' for primitive in primitives], dtype=torch.bool, device=logits.device)
    return penalty.masked_fill(~ordinal, 0)


def validate_fit_ordinal_cache(cache):
    if cache.role != 'fit' or any(value.device.type != 'cpu' for value in cache.data.values()):
        raise ValueError('Canonical ordinal admission requires the owned FIT cache on CPU')
    indices = torch.tensor([i for i, row in enumerate(cache.rows) if row['primitive'] == 'ordinal'], dtype=torch.long)
    if not len(indices):
        raise ValueError('Canonical FIT ordinal levels must include all five ordered values')
    masks = cache.data['mask'].index_select(0, indices)
    values = cache.data['ordinal_values'].index_select(0, indices)
    if (masks.ndim != 2 or masks.shape != values.shape or masks.dtype != torch.bool
            or values.dtype != torch.float32 or masks.shape[1] < 5):
        raise ValueError('Canonical FIT ordinal levels require aligned FP32 levels and masks')
    expected_mask = (torch.arange(masks.shape[1]) < 5).expand_as(masks)
    expected_values = torch.arange(5, dtype=torch.float32).expand(len(indices), -1)
    if not torch.equal(masks, expected_mask) or not torch.equal(values[:, :5], expected_values):
        raise ValueError('Canonical FIT ordinal levels require exactly [0,1,2,3,4] with five valid options')


def _fitting_loss_components(logits, batch, primitives, *, role):
    if role != 'fit':
        raise ValueError('Ordinal RPS is FIT-only')
    if (type(PROFILE.get('ordinal_rps_coefficient')) is not float or PROFILE['ordinal_rps_coefficient'] != .15
            or 'gold_target_smoothing' in PROFILE):
        raise ValueError('Require FIT-only ordinal RPS coefficient 0.15 and original unsmoothed gold')
    per_row = decision_loss(logits, batch, primitives)
    if PROFILE['correction_regularization']:
        per_row = per_row + PROFILE['correction_regularization'] * correction_penalty(logits, batch)
    if PROFILE['teacher_distribution_regularization']:
        per_row = per_row + PROFILE['teacher_distribution_regularization'] * teacher_distribution_penalty(logits, batch)
    ordinal_rps = ordinal_cdf_penalty(logits, batch, primitives)
    return per_row + PROFILE['ordinal_rps_coefficient'] * ordinal_rps, ordinal_rps


def fitting_loss(logits, batch, primitives, *, role):
    return _fitting_loss_components(logits, batch, primitives, role=role)[0]


@torch.no_grad()
def development(bundle, cache, temperatures=None):
    if cache.role != 'development':
        raise ValueError('Selection reads DEV only')
    bundle.eval()
    evidence = defaultdict(lambda: {'groups': [], 'control': defaultdict(list), 'head': defaultdict(list)})
    for offset in range(0, len(cache.rows), PROFILE['batch_size']):
        rows = cache.rows[offset:offset + PROFILE['batch_size']]
        batch = cache.batch(rows)
        # Primitive determines loss/calibration semantics, never the scorer's parameters.
        output = shared_logits(bundle, batch)
        for primitive in PRIMITIVES:
            indices = [index for index, row in enumerate(rows) if row['primitive'] == primitive]
            if not indices:
                continue
            tensors = {key: value[indices].cpu().numpy() for key, value in batch.items() if key != 'candidates' and key != 'query'}
            extra = {'ordinal_values': tensors['ordinal_values']} if primitive == 'ordinal' else {}
            variants = {'control': per_record(tensors['baseline'], tensors['targets'], tensors['mask'], **extra),
                        'head': per_record(output[indices].cpu().numpy(), tensors['targets'], tensors['mask'],
                                           temperature=(temperatures or {}).get(primitive, 1.), **extra)}
            for position, index in enumerate(indices):
                row = rows[index]
                entry = evidence[track(row)]
                entry['groups'].append(row['group_'])
                for method, values in variants.items():
                    for metric in ('accuracy', 'tie_aware_accuracy', 'nll', 'normalized_nll', 'ordinal_mae'):
                        if metric in values:
                            entry[method][metric].append(float(values[metric][position]))
    if set(evidence) != TRACKS:
        raise ValueError('Joint DEV must include every declared task track')
    checks, summary = [], {}
    for key, entry in sorted(evidence.items()):
        summary[key] = {method: {metric: float(np.nanmean(value)) if np.isfinite(value).any() else None
                                for metric, value in entry[method].items()} for method in ('control', 'head')}
        metrics = ('ordinal_mae',) if '/ordinal/' in key else ('accuracy', 'tie_aware_accuracy')
        for metric in metrics:
            before, after = np.asarray(entry['control'][metric]), np.asarray(entry['head'][metric])
            if not (np.isfinite(before) & np.isfinite(after)).any() and metric == 'accuracy':
                continue
            interval = paired_group_interval(before, after, entry['groups'],
                                              draws=PROFILE['bootstrap_draws'], seed=PROFILE['bootstrap_seed'])
            passed = interval['high'] <= .1 if metric == 'ordinal_mae' else interval['low'] >= -.02
            checks.append({'track': key, 'metric': metric, 'interval': interval,
                           'passed': interval['groups'] >= 2 and passed})
    # Track means drive selection; retain per-track confidence gates independently.
    macro = {method: sum(values[method]['normalized_nll'] for values in summary.values()) / len(summary)
             for method in ('control', 'head')}
    gain = macro['control'] - macro['head']
    interval = macro_interval(evidence, draws=PROFILE['bootstrap_draws'], seed=PROFILE['bootstrap_seed'])
    checks.append({'track': 'macro', 'metric': 'normalized_nll', 'interval': interval,
                   'passed': interval['groups'] >= 2 and interval['high'] < 0})
    eligible = all(check['passed'] for check in checks) and gain >= PROFILE['minimum_improvement']
    return {'eligible': eligible, 'selection_role': 'development', 'external_selection': False,
            'calibration_selection': False, 'membership': cache.membership, 'tracks': summary,
            'checks': checks, 'macro_normalized_nll': macro, 'gain_vs_label_token': gain,
            'temperature_by_primitive': temperatures or {name: 1. for name in PRIMITIVES}}


def shared_logits(bundle, batch, primitive='choice'):
    kwargs = {'baseline': batch['baseline']} if getattr(bundle, 'requires_lm_scores', False) else {}
    return bundle(primitive, batch['query'], batch['candidates'], batch['mask'], **kwargs)


def fit_preference_pairs(features, rows):
    check_rows(rows, 'fit')
    if not features.read_only:
        raise ValueError('Preference consistency requires read-only FIT features')
    owned = {row['id']: row for row in rows}
    choices = {'helpsteer2/choice/text/main', 'helpsteer3/choice/text/main',
               'mm-rlhf/choice/image/pairwise_preference'}
    admitted = {}
    for child in rows:
        if child['primitive'] != 'noul' or child.get('adaptation') != 'pairwise_preference_yes_no':
            continue
        raw = features.db.execute('SELECT metadata FROM records WHERE id=? AND role=?',
                                  (child['id'], 'fit')).fetchone()
        if raw is None:
            raise ValueError('Missing owned FIT preference child')
        metadata = json.loads(raw['metadata'])
        parent = owned.get(metadata.get('parent_id'))
        permutation = metadata.get('candidate_permutation')
        if (parent is None or track(parent) not in choices
                or any(metadata.get(key) != child[key] for key in ('id', 'role', 'source', 'primitive'))
                or metadata.get('group') != child['group_']
                or metadata.get('parent_role') != 'fit' or metadata.get('parent_primitive') != 'choice'
                or metadata.get('parent_source') != parent['source']
                or any(parent[key] != child[key] for key in ('source', 'group_', 'modality'))
                or not isinstance(permutation, list) or len(permutation) != 2
                or any(type(index) is not int for index in permutation) or sorted(permutation) != [0, 1]
                or len(metadata.get('candidates', [])) != 2):
            raise ValueError('Preference consistency requires exact equivalent FIT views')
        raw_parent = features.db.execute('SELECT metadata FROM records WHERE id=? AND role=?',
                                         (parent['id'], 'fit')).fetchone()
        if raw_parent is None:
            raise ValueError('Missing owned FIT preference parent')
        original = json.loads(raw_parent['metadata'])
        if (any(original.get(key) != parent[key] for key in ('id', 'role', 'source', 'primitive'))
                or original.get('group') != parent['group_'] or len(original.get('candidates', [])) != 2
                or metadata.get('parent_candidate_ids') != [item['id'] for item in original['candidates']]
                or metadata.get('parent_target') != original.get('target')
                or metadata.get('target') != [original['target'][index] for index in permutation]):
            raise ValueError('Preference view target or candidate permutation changed')
        pair = {'parent': parent, 'child': child, 'permutation': permutation}
        group = (child['source'], child['group_'])
        previous = admitted.get(group)
        if previous is None or (parent['id'], child['id']) < (previous['parent']['id'], previous['child']['id']):
            admitted[group] = pair
    if not admitted:
        raise ValueError('No admitted equivalent FIT preference pairs')
    return {pair['child']['id']: pair for _, pair in sorted(admitted.items())}


def preference_js(parent, child, permutation, *, role):
    if role != 'fit':
        raise ValueError('Preference consistency reads FIT only')
    if (not isinstance(parent, torch.Tensor) or not isinstance(child, torch.Tensor)
            or parent.ndim != 2 or parent.shape[0] < 1 or parent.shape[1] != 2
            or child.shape != parent.shape or parent.dtype != torch.float32 or child.dtype != torch.float32
            or parent.device != child.device or not torch.isfinite(parent).all() or not torch.isfinite(child).all()
            or not isinstance(permutation, torch.Tensor) or permutation.dtype != torch.long
            or permutation.shape != parent.shape or permutation.device != parent.device
            or not (permutation.sort(dim=1).values == torch.tensor([0, 1], device=parent.device)).all()):
        raise ValueError('Need finite aligned binary FIT scores and exact option permutations')
    left = torch.log_softmax(parent.gather(1, permutation), dim=1)
    right = torch.log_softmax(child, dim=1)
    mixture = torch.logaddexp(left, right) - math.log(2)
    divergence = .5 * ((left.exp() * (left - mixture)).sum(1)
                        + (right.exp() * (right - mixture)).sum(1)) / math.log(2)
    return divergence.clamp_min(0)


def batch_preference_consistency(bundle, cache, rows, pairs):
    if cache.role != 'fit':
        raise ValueError('Preference consistency reads FIT only')
    check_rows(rows, 'fit')
    if any(row['id'] in pairs and row != pairs[row['id']]['child'] for row in rows):
        raise ValueError('Preference batch differs from its owned FIT pair')
    selected = [pairs[row['id']] for row in rows if row['id'] in pairs]
    if not selected:
        return cache.data['query'].new_zeros(()), []
    parent = cache.batch([pair['parent'] for pair in selected])
    child = cache.batch([pair['child'] for pair in selected])
    permutation = torch.tensor([pair['permutation'] for pair in selected],
                               dtype=torch.long, device=parent['query'].device)
    if (not (parent['mask'].sum(1) == 2).all() or not (child['mask'].sum(1) == 2).all()
            or not parent['mask'][:, :2].all() or not child['mask'][:, :2].all()
            or not torch.equal(parent['targets'][:, :2].gather(1, permutation), child['targets'][:, :2])):
        raise ValueError('Cached equivalent FIT targets or valid options changed')
    losses = preference_js(shared_logits(bundle, parent)[:, :2], shared_logits(bundle, child, 'noul')[:, :2],
                            permutation, role=cache.role)
    return losses.sum(), [pair['child']['id'] for pair in selected]


def run_trial(features, roles, output, *, device='cuda:0', development_only=False):
    optimization = optimization_protocol()
    require_feature_pin(features)
    output.mkdir(parents=True, exist_ok=False)
    store = Store()
    protocol = {'profile': PROFILE, 'feature_identity_sha256': file_sha256(features.directory / 'identity.json'),
                'recorded_feature_identity_sha256': FEATURE_SHA,
                'roles': {role: membership_identity(rows) for role, rows in roles.items()},
                'architecture': ARCHITECTURE,
                'learning_rate_schedule': PROFILE['learning_rate_schedule'],
                'optimization': optimization,
                'development_only': development_only,
                'primitive_masses': {name: sum(f'/{name}/' in key for key in TRACKS) / len(TRACKS)
                                     for name in PRIMITIVES}, 'track_masses': track_masses(),
                'source': source_identity(), 'execution': {'device': device, 'torch': torch.__version__,
                'cuda_version': torch.version.cuda, 'llm_loaded': False, 'host_memory_guards': False},
                'fitting_order': 'every FIT row once per epoch; shuffled jointly, no replay',
                'scope': 'FIT supervised updates; DEV selection; CAL only after locked DEV acceptance',
                'test_selection': False, 'e2e_selection': False, 'supersedes': 'independent primitive-head design'}
    write_json(output / 'protocol.json', protocol)
    store.update(TASK, 'in_progress', 'Preloading owned FIT/DEV feature slices; LLM remains unloaded.', progress=.05)
    fitting = SharedFeatureCache(features, roles['fit'], 'fit', 'cpu')
    validate_fit_ordinal_cache(fitting)
    fitting.data = {key: value.to(device) for key, value in fitting.data.items()}
    dev = SharedFeatureCache(features, roles['development'], 'development', device)
    weights = fitting_weights(fitting.rows)
    beta = PROFILE['preference_consistency_beta']
    if type(beta) is not float or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('Invalid FIT preference-consistency weight')
    pairs = fit_preference_pairs(features, fitting.rows) if beta else {}
    pairing = [{'parent_id': pair['parent']['id'], 'child_id': pair['child']['id'],
                'source': pair['child']['source'], 'group': pair['child']['group_'],
                'permutation': pair['permutation']} for pair in pairs.values()]
    write_json(output / 'cache-provenance.json', {'fit': fitting.provenance(), 'development': dev.provenance(),
               'consistency': {'role': 'fit', 'pairs': len(pairs), 'beta': beta,
                 'selection': 'lexicographic parent/child ID; one pair per source/input group',
                 'normalization': 'mean symmetric JS/log(2) over unique paired FIT groups',
                 'pairing_sha256': hashlib.sha256(json.dumps(pairing, sort_keys=True).encode()).hexdigest(),
                 'extra_supervised_visits': 0}})
    bundle = SharedDecisionHead(seed=PROFILE['seed'], key_gain=PROFILE['key_gain']).to(device)
    optimizer = make_optimizer(bundle)
    initial = development(bundle, dev)
    metadata = {'feature_identity': features.identity, 'feature_origins': features.producer_origins(),
                'preparation': features.require_ready('calibrate'), 'stage': 'mixed',
                'fit_membership': fitting.membership, 'development_membership': dev.membership,
                'training_code': source_identity(), 'shared_training_code': source_identity(),
                'operator_sha256': file_sha256(__file__),
                'profile': PROFILE, 'learned_head_count': 1, 'frozen_lm_head_trained': False,
                'optimization': optimization,
                'artifact_kind': 'training-checkpoint'}
    epochs, stale, best_loss, selected, steps, nonzero_steps = [], 0, None, None, 0, 0
    for epoch in range(PROFILE['max_epochs']):
        group_learning_rates = set_optimizer_learning_rates(optimizer, bundle, epoch + 1)
        learning_rate = group_learning_rates['scorer_and_gate']
        started = phase_timestamp(device)
        ordered = list(fitting.rows)
        random.Random(PROFILE['seed'] + epoch).shuffle(ordered)
        visits, pair_visits, losses = Counter(), Counter(), 0.
        components = torch.zeros(5, dtype=torch.float64, device=device)
        bundle.train()
        for offset in range(0, len(ordered), PROFILE['batch_size']):
            rows = ordered[offset:offset + PROFILE['batch_size']]
            batch = fitting.batch(rows)
            logits = shared_logits(bundle, batch)
            per_row, ordinal_rps = _fitting_loss_components(logits, batch, [row['primitive'] for row in rows], role=fitting.role)
            factors = torch.tensor([weights[row['id']] for row in rows], dtype=torch.float32, device=device)
            loss = (per_row * factors).sum() / PROFILE['batch_size']
            js_sum, visited = batch_preference_consistency(bundle, fitting, rows, pairs) if beta else (loss.new_zeros(()), [])
            # This scale makes the epoch objective beta * mean(JS) without replaying gold losses.
            if beta:
                loss = loss + beta * len(fitting.rows) / (len(pairs) * PROFILE['batch_size']) * js_sum
            pair_visits.update(visited)
            with torch.no_grad():
                gold = (decision_loss(logits, batch, [row['primitive'] for row in rows]) * factors).sum()
                teacher = (teacher_distribution_penalty(logits, batch) * factors).sum()
                correction = (correction_penalty(logits, batch) * factors).sum()
                rps = (ordinal_rps.detach() * factors).sum()
                components += torch.stack((gold, teacher, js_sum, correction, rps)).double()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in bundle.parameters()]
            if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError('Missing or nonfinite shared-head gradients')
            nonzero_steps += int(any(bool((gradient != 0).any()) for gradient in gradients))
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), PROFILE['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            visits.update(row['id'] for row in rows)
            losses += float(loss.detach()) * PROFILE['batch_size']
        if set(visits) != set(weights) or any(count != 1 for count in visits.values()):
            raise ValueError('Every FIT row must be visited exactly once')
        if set(pair_visits) != set(pairs) or any(count != 1 for count in pair_visits.values()):
            raise ValueError('Every admitted FIT preference group must contribute exactly once')
        gold_sum, teacher_sum, js_total, correction_sum, rps_sum = components.cpu().tolist()
        fit_finished = phase_timestamp(device)
        gate = development(bundle, dev)
        development_finished = phase_timestamp(device)
        checkpoint = output / 'checkpoints' / f'epoch-{epoch + 1:03d}'
        save_shared_bundle(checkpoint, bundle, {**metadata, 'epoch': epoch + 1, 'learning_rate': learning_rate,
                           'group_learning_rates': group_learning_rates,
                           'optimizer_steps': steps, 'development': gate}, {name: 1. for name in PRIMITIVES})
        checkpoint_sha = file_sha256(checkpoint / 'head.safetensors')
        checkpoint_finished = phase_timestamp(device)
        result = {'epoch': epoch + 1, 'directory': str(checkpoint), 'sha256': checkpoint_sha,
                  'learning_rate': learning_rate, 'group_learning_rates': group_learning_rates,
                  'fit_loss': losses / len(fitting.rows), 'fit_visits': len(visits), 'optimizer_steps': steps,
                  'fit_gold_ce': gold_sum / len(fitting.rows), 'fit_teacher_kl': teacher_sum / len(fitting.rows),
                  'fit_correction_penalty': correction_sum / len(fitting.rows),
                  'fit_ordinal_rps': rps_sum / len(fitting.rows),
                  'fit_consistency_js': js_total / len(pairs) if pairs else 0., 'fit_consistency_pairs': len(pair_visits),
                  'nonzero_gradient_steps': nonzero_steps, 'seconds': checkpoint_finished - started,
                  'fit_seconds': fit_finished - started,
                  'development_seconds': development_finished - fit_finished,
                  'checkpoint_seconds': checkpoint_finished - development_finished, 'development': gate}
        epochs.append(result)
        eligible = [item for item in epochs if item['development']['eligible']]
        selected = min(eligible, key=lambda item: (item['development']['macro_normalized_nll']['head'], item['epoch'])) if eligible else None
        current = gate['macro_normalized_nll']['head']
        improved = best_loss is None or current < best_loss - PROFILE['minimum_improvement']
        best_loss, stale = (current, 0) if improved else (best_loss, stale + 1)
        write_json(output / 'trial.json', {'completed': False, 'initial': initial, 'epochs': epochs, 'selected': selected})
        print(json.dumps({'epoch': epoch + 1, 'learning_rate': learning_rate,
                          'group_learning_rates': group_learning_rates,
                          'fit_loss': result['fit_loss'], 'seconds': result['seconds'],
                          'fit_seconds': result['fit_seconds'], 'development_seconds': result['development_seconds'],
                          'checkpoint_seconds': result['checkpoint_seconds'],
                          'dev_eligible': gate['eligible'], 'dev_gain': gate['gain_vs_label_token'],
                          'failed_checks': sum(not check['passed'] for check in gate['checks'])}), flush=True)
        store.update(TASK, 'in_progress', f'One shared CUDA head: epoch {epoch + 1}; DEV eligible={gate["eligible"]}. No TEST/E2E.',
                     progress=.2 + .6 * (epoch + 1) / PROFILE['max_epochs'])
        if stale >= PROFILE['patience']:
            break
    trial = {'completed': True, 'initial': initial, 'epochs': epochs, 'selected': selected,
             'status': 'dev_accepted' if selected else 'dev_rejected', 'learned_head_count': 1,
             'main_quality': 'not_run', 'calibration': 'not_started'}
    write_json(output / 'trial.json', trial)
    del fitting, optimizer, bundle
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    if selected is None:
        store.update(TASK, 'failed', 'Shared CUDA trial complete but DEV retention rejected; no CAL, TEST or E2E.', progress=1)
        return trial
    locked = {'directory': selected['directory'], 'sha256': selected['sha256'], 'selection_role': 'development',
              'manifest_sha256': file_sha256(Path(selected['directory']) / 'manifest.json')}
    write_json(output / 'selection-lock.json', locked)
    if development_only:
        store.update(TASK, 'in_progress', 'FIT/DEV profile complete; CAL/TEST/E2E remain unread.', progress=.8)
        return trial
    return calibrate_trial(features, roles, output, trial, device=device, dev=dev)


def calibrate_trial(features, roles, output, trial, *, device='cuda:0', dev=None):
    require_feature_pin(features)
    store = Store()
    protocol = json.loads((output / 'protocol.json').read_text())
    if (protocol['profile'] != PROFILE
            or protocol.get('optimization') != optimization_protocol()
            or protocol['feature_identity_sha256'] != file_sha256(features.directory / 'identity.json')
            or protocol['roles'] != {role: membership_identity(rows) for role, rows in roles.items()}
            or trial.get('completed') is not True or trial.get('status') != 'dev_accepted'
            or trial.get('calibration') != 'not_started'
            or (output / 'reference').exists() or (output / 'calibrated-dev.json').exists()
            or (output / 'calibration-attempt.json').exists()):
        raise ValueError('Require unchanged DEV-selected profile before a single CAL-only fit')
    locked = json.loads((output / 'selection-lock.json').read_text())
    selected = trial['selected']
    if (locked['directory'] != selected['directory'] or locked['sha256'] != selected['sha256']
            or locked['selection_role'] != 'development' or not selected['development']['eligible']):
        raise ValueError('Development selection lock changed')
    bundle, selected_manifest, _ = load_shared_bundle(
        selected['directory'], expected_manifest_sha256=locked['manifest_sha256'])
    if selected_manifest['weights_sha256'] != locked['sha256']:
        raise ValueError('Selected shared manifest differs from the locked checkpoint')
    expected_rates = {group['name']: learning_rate_for_epoch(selected['epoch']) * group['lr_multiplier']
                      for group in optimization_protocol()['parameter_groups']}
    if (selected_manifest['profile'] != PROFILE
            or selected_manifest.get('optimization') != protocol['optimization']
            or selected_manifest.get('group_learning_rates') != expected_rates
            or selected.get('group_learning_rates') != expected_rates
            or selected_manifest.get('learning_rate') != expected_rates['scorer_and_gate']
            or selected.get('learning_rate') != expected_rates['scorer_and_gate']):
        raise ValueError('Selected training provenance changed')
    if (selected_manifest['feature_identity'] != features.identity
            or selected_manifest['preparation'] != features.require_ready('calibrate')
            or selected_manifest['feature_origins'] != features.producer_origins()
            or selected_manifest['fit_membership'] != protocol['roles']['fit']
            or selected_manifest['development_membership'] != protocol['roles']['development']
            or selected_manifest['epoch'] != selected['epoch']
            or selected_manifest['development'] != selected['development']):
        raise ValueError('Selected checkpoint is not bound to the current FIT/DEV corpus and evidence')
    from .shared_qualification import _gate
    _gate(selected['development'], protocol['roles']['development'], {name: 1. for name in PRIMITIVES})
    start_calibration(output, locked, protocol)
    metadata = {key: selected_manifest[key] for key in (
        'feature_identity', 'feature_origins', 'preparation', 'stage', 'fit_membership',
        'development_membership', 'training_code', 'shared_training_code', 'operator_sha256',
        'profile', 'learned_head_count', 'frozen_lm_head_trained',
        'optimization', 'learning_rate', 'group_learning_rates')}
    if dev is None:
        dev = SharedFeatureCache(features, roles['development'], 'development', device)
    bundle.to(device).eval()
    bundle.requires_grad_(False)
    calibration = SharedFeatureCache(features, roles['calibration'], 'calibration', device)
    calibration_weights = role_weights(calibration.rows)
    temperatures = {}
    for primitive in PRIMITIVES:
        rows = [row for row in calibration.rows if row['primitive'] == primitive]
        logits, targets, masks = [], [], []
        for offset in range(0, len(rows), PROFILE['batch_size']):
            batch = calibration.batch(rows[offset:offset + PROFILE['batch_size']])
            with torch.no_grad():
                value = shared_logits(bundle, batch, primitive)
            logits.append(value.cpu())
            targets.append(batch['targets'].cpu())
            masks.append(batch['mask'].cpu())
        factors = torch.tensor([calibration_weights[row['id']] for row in rows], dtype=torch.float32)
        temperatures[primitive] = fit_shared_temperature(torch.cat(logits), torch.cat(targets), torch.cat(masks), factors)
    if file_sha256(Path(selected['directory']) / 'head.safetensors') != locked['sha256']:
        raise ValueError('Locked shared checkpoint changed during calibration')
    final_gate = development(bundle, dev, temperatures)
    write_json(output / 'calibrated-dev.json', final_gate)
    if not final_gate['eligible']:
        trial.update(calibration='completed_but_dev_rejected', temperatures=temperatures)
        write_json(output / 'trial.json', trial)
        finish_calibration(output, 'dev_rejected', temperatures, locked)
        store.update(TASK, 'failed', 'Temperature-only CAL fit completed; final DEV retention rejected. No TEST/E2E.', progress=1)
        return trial
    metadata.update(artifact_kind='calibrated-reference-head', calibration_role='calibration',
                    calibration_code=source_identity(), calibration_membership=calibration.membership,
                    selected_checkpoint=locked,
                    calibration_source=calibration.provenance(), shared_development_gate=final_gate,
                    shared_development_selection=selected['development'],
                    calibration_procedure='per-primitive positive scalar temperature; balanced track/group normalized CE; shared weights frozen')
    save_shared_bundle(output / 'reference', bundle, metadata, temperatures)
    if file_sha256(output / 'reference/head.safetensors') != locked['sha256']:
        raise ValueError('Calibration altered trained shared weights')
    trial.update(calibration='completed', temperatures=temperatures, reference=str(output / 'reference'))
    write_json(output / 'trial.json', trial)
    finish_calibration(output, 'completed', temperatures, locked)
    store.update(TASK, 'completed', 'One shared decision head DEV-accepted and calibrated; main quality still required.', progress=1)
    return trial


def start_calibration(output, locked, protocol):
    # Persist before any CAL tensor access; a crash or competing call cannot refit CAL.
    with (output / 'calibration-attempt.json').open('x') as handle:
        json.dump({'started_at': time.time(), 'checkpoint': locked,
                   'source': protocol['source'], 'profile': protocol['profile'],
                   'calibration_membership': protocol['roles']['calibration']},
                  handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def finish_calibration(output, status, temperatures, locked):
    with (output / 'calibration-outcome.json').open('x') as handle:
        json.dump({'finished_at': time.time(), 'status': status, 'temperatures': temperatures,
                   'locked_weights_sha256': locked['sha256'],
                   'attempt_sha256': file_sha256(output / 'calibration-attempt.json')},
                  handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--profile', choices=SEARCH_PROFILES, default='key-adapter-ordinal-rps015-light-consistency-medium')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--development-only', action='store_true')
    modes.add_argument('--calibrate-existing', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = f'weights/cuda-shared-lm-ordinal-rps-20261006/{args.profile}'
    PROFILE.clear()
    PROFILE.update(BASE_PROFILE | SEARCH_PROFILES[args.profile])
    optimization_protocol()
    require_feature_pin()
    learning_rate_for_epoch(1)
    directory = project_path(FEATURES)
    identity = json.loads((directory / 'identity.json').read_text())
    if identity.get('dataset_scope') != contract(augmentation=True, dimensions=True):
        raise ValueError('Require the complete approved dimension-choice v4 dataset scope')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    features = FeatureStore(directory, identity, read_only=True)
    try:
        roles = read_roles(features)
        if args.calibrate_existing:
            output = project_path(args.output)
            result = calibrate_trial(features, roles, output, json.loads((output / 'trial.json').read_text()))
        else:
            result = run_trial(features, roles, project_path(args.output), development_only=args.development_only)
        print(json.dumps({'output': args.output, 'status': result['status'], 'learned_head_count': 1,
                          'main_quality': 'not_run', 'llm_loaded': False}), flush=True)
    except BaseException as exc:
        Store().update(TASK, 'failed', f'Shared-head CUDA job failed: {type(exc).__name__}: {exc}')
        raise
    finally:
        features.close()


if __name__ == '__main__':
    main()
