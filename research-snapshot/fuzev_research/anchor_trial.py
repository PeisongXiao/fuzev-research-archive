"""One predeclared CUDA-only FIT/DEV residual trial; not release admission.

No calibration, heldout scoring, backbone loading or server changes occur here.
Full frozen-reference export/reload parity must precede the trial.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import save_file

from .acquire import write_json
from .anchor_head import AnchorResidualHead, load_anchor
from .anchor_training_data import OwnedFeatureCache
from .features import FeatureStore, file_sha256
from .metrics import paired_group_interval, per_record
from .paths import project_path
from .preference_head_retrain import CHOICE_TRACKS, NOUL_TRACKS, check_rows, read_roles, track
from .state import Store
from .train import membership_identity


PROFILE = {'choice': {'seed': 29, 'max_epochs': 3}, 'noul': {'seed': 43, 'max_epochs': 5}}
COMMON = {'learning_rate': 1e-4, 'batch_size': 256, 'weight_decay': .01, 'clip_grad_norm': 1.,
          'patience': 2, 'residual_bound': .25, 'centered_residual_l2': .01,
          'minimum_development_gain': 1e-4, 'bootstrap_draws': 2000, 'bootstrap_seed': 17}
OUTPUT = 'weights/cuda-anchor-residual-dev-20261005'
ANCHOR = 'weights/native4-full-label-anchor-20261005'
PARITY = 'runs/native-real/qualification/native4-full-anchor-parity-20261005.json'


def track_masses(primitive):
    if primitive == 'choice':
        return {key: .25 for key in CHOICE_TRACKS}
    if primitive == 'noul':
        return {key: .5 if key.startswith('boolq/') else 1 / 6 for key in NOUL_TRACKS}
    raise ValueError('Only the declared fresh Choice/Noul residuals may be fitted')


def fitting_weights(rows, primitive):
    check_rows(rows, 'fit', primitive)
    counts, groups = Counter(), defaultdict(set)
    for row in rows:
        key = track(row)
        counts[key, row['group_']] += 1
        groups[key].add(row['group_'])
    masses = track_masses(primitive)
    if set(groups) != set(masses):
        raise ValueError('All predeclared FIT tracks are required')
    # Every FIT row is visited once; expansion within a group cannot increase its loss share.
    result = {row['id']: len(rows) * masses[track(row)] /
              (len(groups[track(row)]) * counts[track(row), row['group_']]) for row in rows}
    if not math.isclose(sum(result.values()), len(rows), rel_tol=1e-9):
        raise ValueError('Invalid fitting weight normalization')
    return result


def cached_forward(head, cache, rows, batch):
    return head.forward_cached(batch['query'], batch['candidates'], batch['mask'],
                               baseline=batch['baseline'], rows=rows,
                               anchor_sha256=head.anchor.weights_sha256,
                               feature_identity_sha256=cache.identity_sha256,
                               producer_identity_sha256=head.anchor.producer_identity_sha256)


@torch.no_grad()
def pre_tanh_arguments(head, batch):
    raw = head.residual(batch['query'], batch['candidates'])
    mask = batch['mask']
    mean = raw.masked_fill(~mask, 0).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
    return ((raw - mean) / head.residual_bound)[mask].float().cpu()


def pre_tanh_stats(arguments):
    values = torch.cat(arguments)
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite pre-tanh diagnostic')
    absolute = values.abs()
    derivative = 1 - torch.tanh(values).square()
    return {'candidate_slots': len(values), 'abs_argument_p50': float(absolute.median()),
            'abs_argument_p95': float(torch.quantile(absolute, .95)), 'abs_argument_max': float(absolute.max()),
            'zero_tanh_derivative_fraction': float((derivative == 0).float().mean())}


@torch.no_grad()
def development(head, cache):
    if cache.role != 'development':
        raise ValueError('Checkpoint selection uses DEV only')
    head.eval()
    evidence = defaultdict(lambda: {'groups': [], 'anchor': defaultdict(list), 'head': defaultdict(list)})
    max_margin, nonzero, flips = 0., 0, 0
    raw_arguments = []
    for offset in range(0, len(cache.rows), COMMON['batch_size']):
        rows = cache.rows[offset:offset + COMMON['batch_size']]
        batch = cache.batch(rows)
        raw_arguments.append(pre_tanh_arguments(head, batch))
        output = cached_forward(head, cache, rows, batch)
        baseline, targets, mask = (batch[name].cpu().numpy() for name in ('baseline', 'targets', 'mask'))
        variants = {'anchor': per_record(baseline, targets, mask),
                    'head': per_record(output.cpu().numpy(), targets, mask)}
        diagnostic = head.margin_diagnostics(batch['query'], batch['candidates'], batch['mask'])
        max_margin = max(max_margin, diagnostic['max_candidate_margin'])
        nonzero += diagnostic['nonzero_centered_candidates']
        flips += int((output.argmax(1) != batch['baseline'].masked_fill(~batch['mask'], -torch.inf).argmax(1)).sum())
        for index, row in enumerate(rows):
            entry = evidence[track(row)]
            entry['groups'].append(row['group_'])
            for name, values in variants.items():
                for metric in ('accuracy', 'tie_aware_accuracy', 'nll', 'normalized_nll'):
                    entry[name][metric].append(float(values[metric][index]))
    checks, summary = [], {}
    for key, entry in sorted(evidence.items()):
        summary[key] = {name: {metric: float(np.nanmean(value)) for metric, value in metrics.items()}
                        for name, metrics in ((name, entry[name]) for name in ('anchor', 'head'))}
        for metric in ('accuracy', 'tie_aware_accuracy'):
            before, after = np.asarray(entry['anchor'][metric]), np.asarray(entry['head'][metric])
            valid = np.isfinite(before) & np.isfinite(after)
            if not valid.any() and metric == 'accuracy':
                continue
            interval = paired_group_interval(before, after, entry['groups'],
                                              draws=COMMON['bootstrap_draws'], seed=COMMON['bootstrap_seed'])
            checks.append({'track': key, 'metric': metric, 'interval': interval,
                           'passed': interval['groups'] >= 2 and interval['low'] >= -.02})
        checks.append({'track': key, 'metric': 'raw_nll',
                       'passed': summary[key]['head']['nll'] <= summary[key]['anchor']['nll'] + 1e-6})
    if set(evidence) != set(track_masses(head.primitive)):
        raise ValueError('Incomplete DEV tracks')
    macro = {name: sum(value[name]['normalized_nll'] for value in summary.values()) / len(summary)
             for name in ('anchor', 'head')}
    gain = macro['anchor'] - macro['head']
    learned = {'nonzero_centered_candidates': nonzero, 'max_candidate_margin': max_margin,
               'argmax_changes_vs_anchor': flips}
    eligible = (all(check['passed'] for check in checks) and math.isfinite(gain)
                and gain >= COMMON['minimum_development_gain'] and max_margin > 0)
    return {'eligible': eligible, 'selection_role': 'development', 'external_selection': False,
            'calibration_selection': False, 'membership': cache.membership, 'tracks': summary,
            'checks': checks, 'macro_normalized_nll': macro, 'gain_vs_anchor_only': gain,
            'learned_margin': learned, 'anchor_ablation_temperature': 1.,
            'pre_tanh': pre_tanh_stats(raw_arguments)}


def train_primitive(anchor, features, roles, primitive, output, *, learning_rate=None,
                    task='training.anchor_residual'):
    fit_rows = [row for row in roles['fit'] if row['primitive'] == primitive]
    dev_rows = [row for row in roles['development'] if row['primitive'] == primitive]
    fitting = OwnedFeatureCache(features, fit_rows, 'fit', primitive)
    dev = OwnedFeatureCache(features, dev_rows, 'development', primitive)
    weights = fitting_weights(fit_rows, primitive)
    fit_sample_rows, sample_groups = [], defaultdict(set)
    for row in fitting.rows:
        groups = sample_groups[track(row)]
        if len(groups) < 8 and row['group_'] not in groups:
            groups.add(row['group_'])
            fit_sample_rows.append(row)
    profile = PROFILE[primitive]
    learning_rate = COMMON['learning_rate'] if learning_rate is None else learning_rate
    if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Learning rate must be a finite positive float')
    head = AnchorResidualHead(anchor, primitive=primitive, residual_bound=COMMON['residual_bound'],
                              seed=profile['seed'], cache_feature_identity_sha256=fitting.identity_sha256).cuda()
    initial_sha = head.residual_sha256()
    optimizer = torch.optim.AdamW(head.residual.parameters(), lr=learning_rate,
                                 weight_decay=COMMON['weight_decay'])
    directory = output / primitive
    directory.mkdir(exist_ok=False)
    write_json(directory / 'cache-provenance.json', {'fit': fitting.provenance(), 'development': dev.provenance(),
                                                   'loss_weights': weights, 'track_masses': track_masses(primitive)})
    initial = development(head, dev)
    if initial['gain_vs_anchor_only'] != 0 or initial['learned_margin']['max_candidate_margin'] != 0:
        raise ValueError('Fresh residual does not exactly start at the anchor')
    epochs, steps, supervised_steps, total_gradient_steps, stale, best = [], 0, 0, 0, 0, None
    for epoch in range(profile['max_epochs']):
        ordered = list(fit_rows)
        random.Random(profile['seed'] + epoch).shuffle(ordered)
        visits, weighted_loss = Counter(), 0.
        head.train()
        for offset in range(0, len(ordered), COMMON['batch_size']):
            rows = ordered[offset:offset + COMMON['batch_size']]
            batch = fitting.batch(rows)
            logits = cached_forward(head, fitting, rows, batch)
            logp = torch.log_softmax(logits, dim=1).masked_fill(~batch['mask'], 0)
            ce = -(batch['targets'] * logp).sum(dim=1) / batch['mask'].sum(dim=1).float().log()
            delta = (logits - batch['baseline']).masked_fill(~batch['mask'], 0)
            mean = delta.sum(dim=1, keepdim=True) / batch['mask'].sum(dim=1, keepdim=True)
            penalty = ((delta - mean).masked_fill(~batch['mask'], 0).square().sum(dim=1)
                       / batch['mask'].sum(dim=1))
            factors = torch.tensor([weights[row['id']] for row in rows], device='cuda:0', dtype=torch.float32)
            loss = ((ce + COMMON['centered_residual_l2'] * penalty) * factors).sum() / COMMON['batch_size']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite anchored training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = head.residual.key.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise ValueError('Missing or nonfinite supervised key-weight gradient')
            nonzero_gradient = bool((gradient != 0).any())
            total_gradient_steps += int(nonzero_gradient)
            if steps == 0:
                # At the exact zero residual, the L2 derivative is zero: this is a CE-only proof.
                if bool((delta[batch['mask']] != 0).any()) or not nonzero_gradient:
                    raise ValueError('No proven first-step supervised residual gradient')
                supervised_steps += 1
            torch.nn.utils.clip_grad_norm_(head.residual.parameters(), COMMON['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            visits.update(row['id'] for row in rows)
            weighted_loss += float(loss.detach()) * COMMON['batch_size']
        if set(visits) != {row['id'] for row in fit_rows} or any(count != 1 for count in visits.values()):
            raise ValueError('Every FIT row must be visited exactly once per epoch')
        gate = development(head, dev)
        dev_sample_rows = dev.rows[:COMMON['batch_size']]
        sample = dev.batch(dev_sample_rows)
        with torch.no_grad():
            learning = head.require_learned_residual(initial_sha, optimizer_steps=steps,
                                                     supervised_gradient_steps=supervised_steps,
                                                     query=sample['query'], candidates=sample['candidates'],
                                                     mask=sample['mask'], rows=dev_sample_rows)
        checkpoint = directory / f'epoch-{epoch + 1:03d}.safetensors'
        save_file({key: value.detach().cpu().contiguous() for key, value in head.residual.state_dict().items()}, str(checkpoint))
        result = {'epoch': epoch + 1, 'checkpoint': str(checkpoint), 'sha256': file_sha256(checkpoint),
                  'fit_loss': weighted_loss / len(fit_rows), 'fit_membership': fitting.membership,
                  'fit_visits_this_epoch': len(visits), 'cumulative_fit_visits': (epoch + 1) * len(fit_rows),
                  'total_nonzero_key_gradient_steps': total_gradient_steps,
                  'supervised_gradient_proof': 'first update at exact zero residual; L2 derivative zero',
                  'fit_pre_tanh': {'membership': membership_identity(fit_sample_rows),
                                   **pre_tanh_stats([pre_tanh_arguments(head, fitting.batch(fit_sample_rows))])},
                  'learning': learning, 'development': gate}
        epochs.append(result)
        eligible = [item for item in epochs if item['development']['eligible']]
        selected = min(eligible, key=lambda item: (item['development']['macro_normalized_nll']['head'], item['epoch'])) if eligible else None
        current = gate['macro_normalized_nll']['head']
        improved = best is None or current < best - COMMON['minimum_development_gain']
        best, stale = (current, 0) if improved else (best, stale + 1)
        write_json(directory / 'trial.json', {'primitive': primitive, 'completed': False,
                                            'initialization': initial, 'initial_sha256': initial_sha,
                                            'epochs': epochs, 'selected': selected,
                                            'qualification': 'unqualified FIT/DEV trial; no CAL or TEST'})
        print(json.dumps({'primitive': primitive, 'epoch': epoch + 1, 'dev_eligible': gate['eligible'],
                          'gain': gate['gain_vs_anchor_only'], 'selected_epoch': selected['epoch'] if selected else None}), flush=True)
        progress = (.2 + (.4 if primitive == 'noul' else 0) + .4 * (epoch + 1) / profile['max_epochs']
                    if task == 'training.anchor_residual' else .4 + .6 * (epoch + 1) / profile['max_epochs'])
        Store().update(task, 'in_progress',
                       f'CUDA {primitive} epoch {epoch + 1}/{profile["max_epochs"]}; '
                       f'LR={learning_rate}; DEV eligible={gate["eligible"]}. No CAL/TEST or main quality claim.',
                       progress=progress)
        if stale >= COMMON['patience']:
            break
    report = {'primitive': primitive, 'completed': True, 'profile': {**COMMON, **profile, 'learning_rate': learning_rate},
              'initial_sha256': initial_sha, 'epochs': epochs, 'selected': selected,
              'status': 'dev_accepted' if selected else 'dev_rejected', 'calibration_status': 'not_started',
              'qualification': 'unqualified FIT/DEV trial; no main quality, E2E or production claim'}
    write_json(directory / 'trial.json', report)
    del optimizer, head, fitting, dev
    torch.cuda.empty_cache()
    return report


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    parity_path = project_path(PARITY)
    parity = json.loads(parity_path.read_text())
    if parity.get('numerical_gate') != 'pass' or parity.get('llm_loaded') is not False:
        raise ValueError('Full frozen readout export/reload exact parity must pass first')
    anchor = load_anchor(project_path(ANCHOR), manifest_sha256=parity['anchor_manifest_sha256'])
    if anchor.weights_sha256 != parity['anchor_weights_sha256']:
        raise ValueError('Qualified frozen anchor changed')
    directory = project_path('data/features/native-4.05-noul-preference-v3')
    if file_sha256(directory / 'identity.json') != parity['feature_identity_sha256']:
        raise ValueError('Qualified feature identity changed')
    features = FeatureStore(directory, json.loads((directory / 'identity.json').read_text()), read_only=True)
    try:
        # Preserve all existing corpus, ownership and parent-provenance qualification guards.
        roles = read_roles(features)
        output = project_path(OUTPUT)
        output.mkdir(exist_ok=False)
        protocol = {'started_at': time.time(), 'profile': PROFILE, 'common': COMMON,
                    'feature_identity_sha256': file_sha256(directory / 'identity.json'),
                    'anchor_manifest_sha256': anchor.manifest_sha256, 'anchor_weights_sha256': anchor.weights_sha256,
                    'producer_identity_sha256': anchor.producer_identity_sha256,
                    'parity_proof_sha256': file_sha256(parity_path), 'operator_sha256': file_sha256(__file__),
                    'code': {name: file_sha256(Path(__file__).with_name(name)) for name in
                             ('anchor_head.py', 'anchor_training_data.py', 'features.py', 'metrics.py', 'scope.py')},
                    'roles': {role: membership_identity(rows) for role, rows in roles.items()},
                    'device': 'cuda:0', 'torch': torch.__version__,
                    'scope': 'training-source FIT/DEV only; CAL membership audited but no CAL tensors or fitting',
                    'external_selection': False, 'ordinal': 'untouched; not part of this prototype trial',
                    'qualification': 'not qualified; no server changes, CAL, TEST, E2E, or promotion'}
        write_json(output / 'protocol.json', protocol)
        results = {primitive: train_primitive(anchor, features, roles, primitive, output)
                   for primitive in ('choice', 'noul')}
        write_json(output / 'run.json', {'completed_at': time.time(), 'results': results,
                                        'status': 'trial_complete_not_qualified'})
        print(json.dumps({'output': str(output), 'stages': {key: item['status'] for key, item in results.items()},
                          'main_quality': 'not_run'}), flush=True)
    finally:
        features.close()


if __name__ == '__main__':
    main()
