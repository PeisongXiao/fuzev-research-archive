"""Observe unchanged CUDA FIT updates without editing their gradients or state.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.
"""

import argparse
from collections import defaultdict
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = ROOT / 'reports/2026-10-10-quality-mechanism'
WEIGHTS = ROOT / 'weights/quality-mature-displacement-20261010'
OLD_WORKER = Path(os.environ['FUZEV_WORKLOAD_SOURCE']).expanduser().resolve() / 'scripts/run_workload_robust.py'
OLD_SHA = 'ed398e1283ff1909e0e63899e1d20a57411a726f017677bd182c43be3a45fcd3'
TASK = 'training.generalization.mature-direction'
OBSERVED_EPOCHS, OBSERVED_STEPS = (1, 4, 5), (159, 318, 477)
REFERENCES = {
    'feature-only': (ROOT / 'weights/workload-robust-20261010/feature-only/control/fit/head.safetensors',
                     '4e94e11c01450f216449ad28cc20932fefac84ccb07f7d1288c57582dfe6e5cd'),
    'lm-assisted': (ROOT / 'weights/score-confidence-20261010/control/fit/head.safetensors',
                    '692e1cdaad5bf42b200580ca515417594862d4ae448fe58888d8be1e7bd796bd'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def gold_ce(torch, logits, batch):
    mask, y = batch['mask'], batch['targets']
    require(logits.is_cuda and mask.is_cuda and y.is_cuda and not y.requires_grad
            and bool((mask.sum(1) >= 2).all()), 'Frozen CUDA gold with K>=2 required')
    logp = logits.masked_fill(~mask, -torch.inf).log_softmax(1).masked_fill(~mask, 0)
    return -(y * logp).sum(1) / mask.sum(1).float().log()


def dot(torch, left, right):
    require(len(left) == len(right) > 0 and all(a.is_cuda and b.is_cuda
            and a.shape == b.shape for a, b in zip(left, right)), 'CUDA vector geometry mismatch')
    return torch.stack([(a.double() * b.double()).sum() for a, b in zip(left, right)]).sum()


def cosine(torch, left, right):
    denominator = (dot(torch, left, left) * dot(torch, right, right)).sqrt()
    return float(dot(torch, left, right) / denominator) if float(denominator) > 0 else None


def frozen_snapshot(torch, model):
    return {'parameters': [p.detach().clone() for p in model.parameters()],
            'gradients': [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
            'cpu_rng': torch.get_rng_state().clone(), 'cuda_rng': torch.cuda.get_rng_state().clone(),
            'mode': model.training}


def assert_unchanged(torch, model, snapshot, *, parameters=True):
    current = list(model.parameters())
    require(all(torch.equal(p.grad, saved) if saved is not None else p.grad is None
                for p, saved in zip(current, snapshot['gradients'])), 'Observer changed original .grad')
    require(not parameters or all(torch.equal(p.detach(), saved)
                for p, saved in zip(current, snapshot['parameters'])), 'Observer changed parameters')
    require(torch.equal(torch.get_rng_state(), snapshot['cpu_rng'])
            and torch.equal(torch.cuda.get_rng_state(), snapshot['cuda_rng'])
            and model.training == snapshot['mode'], 'Observer changed RNG or training mode')


def optimizer_snapshot(optimizer, model):
    return [{key: value.detach().clone() for key, value in optimizer.state[p].items()}
            for p in model.parameters()]


def assert_optimizer_unchanged(torch, optimizer, model, snapshot):
    require(all(set(optimizer.state[p]) == set(saved)
                and all(torch.equal(optimizer.state[p][key], value) for key, value in saved.items())
                for p, saved in zip(model.parameters(), snapshot)), 'Observer changed actual optimizer state')


def gold_panel(torch, training, model, cache, grouped, factors, *, gradients):
    parameters = tuple(model.parameters())
    losses, vectors, references = {}, {}, {}
    with torch.set_grad_enabled(gradients):
        for track, rows in grouped.items():
            loss_total = torch.zeros((), dtype=torch.float64, device='cuda')
            reference_total = torch.zeros_like(loss_total)
            accumulated = [torch.zeros_like(p) for p in parameters] if gradients else None
            for offset in range(0, len(rows), 256):
                selected = rows[offset:offset + 256]
                batch = cache.batch(selected)
                logits = training.shared_logits(model, batch)
                per_row = gold_ce(torch, logits, batch)
                weight = torch.tensor([factors[r['id']] for r in selected], device='cuda', dtype=torch.float32)
                loss_total += (per_row.detach().double() * weight.double()).sum()
                mask, y = batch['mask'], batch['targets'].double()
                ref_logp = logits.detach().double().masked_fill(~mask, -torch.inf).log_softmax(1).masked_fill(~mask, 0)
                reference_total += (-(y * ref_logp).sum(1) / mask.sum(1).double().log() * weight.double()).sum()
                if gradients:
                    partial = torch.autograd.grad((per_row * weight).sum(), parameters, allow_unused=True)
                    for destination, value in zip(accumulated, partial):
                        if value is not None:
                            destination.add_(value)
            require(bool(torch.isfinite(loss_total)), 'Nonfinite full FIT gold statistic')
            if gradients:
                require(all(bool(torch.isfinite(g).all()) for g in accumulated), 'Nonfinite gold gradient')
                vectors[track] = accumulated
            losses[track] = float(loss_total)
            references[track] = float(reference_total)
    return losses, vectors, references


class Observer:
    def __init__(self, torch, training, model, cache, grouped, factors, priors, emit,
                 epochs=OBSERVED_EPOCHS, steps=OBSERVED_STEPS):
        self.torch, self.training, self.model, self.cache = torch, training, model, cache
        self.grouped, self.factors, self.emit = grouped, factors, emit
        self.priors = priors
        self.epochs, self.steps = epochs, steps
        self.epoch, self.step, self.pending = 0, 0, None
        self.observations, self.seconds = [], 0.

    def pre(self, optimizer, args, kwargs):
        self.step += 1
        if self.epoch not in self.epochs or self.step not in self.steps:
            return None
        torch = self.torch
        started = time.monotonic()
        snapshot = frozen_snapshot(torch, self.model)
        states = optimizer_snapshot(optimizer, self.model)
        inputs = {key: (value.data_ptr(), value._version) for key, value in self.cache.data.items()}
        require(all(g is not None and g.is_cuda for g in snapshot['gradients']), 'Original clipped CUDA grads required')
        before, vectors, references = gold_panel(torch, self.training, self.model, self.cache,
                                                 self.grouped, self.factors, gradients=True)
        repeat, _, _ = gold_panel(torch, self.training, self.model, self.cache,
                                  self.grouped, self.factors, gradients=False)
        global_factors = {r['id']: self.factors[r['id']] * self.priors[track]
                          for track, owned in self.grouped.items() for r in owned}
        global_losses, global_vectors, _ = gold_panel(torch, self.training, self.model, self.cache,
            {'all': self.cache.rows}, global_factors, gradients=True)
        reconstructed = [sum(self.priors[t] * vectors[t][i] for t in vectors)
                         for i in range(len(snapshot['parameters']))]
        require(all(torch.allclose(a, b, atol=2e-6, rtol=2e-5)
                    for a, b in zip(reconstructed, global_vectors['all'])), 'Full gold gradient reconstruction failed')
        require(abs(sum(self.priors[t] * before[t] for t in before) - global_losses['all']) < 2e-6,
                'Full gold objective reconstruction failed')
        assert_unchanged(torch, self.model, snapshot)
        assert_optimizer_unchanged(torch, optimizer, self.model, states)
        require(inputs == {key: (value.data_ptr(), value._version) for key, value in self.cache.data.items()},
                'Observer changed cache storage/version')
        self.pending = snapshot, before, vectors, references, repeat, inputs, started
        return None

    def post(self, optimizer, args, kwargs):
        if self.pending is None:
            return
        torch = self.torch
        snapshot, before, vectors, references, repeat, inputs, started = self.pending
        delta = [p.detach().double() - old.double() for p, old in zip(self.model.parameters(), snapshot['parameters'])]
        post_snapshot = frozen_snapshot(torch, self.model)
        states = optimizer_snapshot(optimizer, self.model)
        after, _, after_reference = gold_panel(torch, self.training, self.model, self.cache,
                                               self.grouped, self.factors, gradients=False)
        assert_unchanged(torch, self.model, post_snapshot)
        assert_optimizer_unchanged(torch, optimizer, self.model, states)
        assert_unchanged(torch, self.model, snapshot, parameters=False)
        require(inputs == {key: (value.data_ptr(), value._version) for key, value in self.cache.data.items()},
                'Observer changed cache storage/version')
        tracks = {}
        for track, gradient in vectors.items():
            actual_alignment = cosine(torch, gradient, delta)
            resolution = max(abs(before[track] - references[track]), abs(after[track] - after_reference[track]),
                             abs(before[track] - repeat[track]))
            tracks[track] = {'gold_ce_before': before[track], 'gold_ce_after': after[track],
                'observed_gold_ce_delta': after[track] - before[track],
                'ce_arithmetic_resolution': resolution, 'no_update_repeat_delta': repeat[track] - before[track],
                'screening_epsilon': max(1e-6, 10 * resolution),
                'gradient_dot_actual_displacement': float(dot(torch, gradient, delta)),
                'actual_descent_cosine': None if actual_alignment is None else -actual_alignment,
                'clipped_gradient_descent_cosine': cosine(torch, gradient, snapshot['gradients']),
                'gold_gradient_norm': math.sqrt(float(dot(torch, gradient, gradient)))}
        torch.cuda.synchronize()
        duration = time.monotonic() - started
        self.seconds += duration
        observation = {'epoch': self.epoch, 'step_in_epoch': self.step,
            'actual_parameter_displacement_norm': math.sqrt(float(dot(torch, delta, delta))),
            'observer_seconds': duration, 'tracks': tracks,
            'original_gradients_rng_mode_optimizer_cache_unchanged': True,
            'full_gold_gradient_objective_reconstructed': True}
        self.emit(observation)
        self.observations.append(observation)
        self.pending = None


def mechanism_gate(observations):
    mature = [o for o in observations if o['epoch'] in (4, 5)]
    require(len(mature) == 6, 'Six fixed mature updates required')
    counts, by_epoch = {}, {}
    for track in mature[0]['tracks']:
        by_epoch[track] = {}
        for epoch in (4, 5):
            by_epoch[track][str(epoch)] = sum(v['actual_descent_cosine'] is not None
                and v['clipped_gradient_descent_cosine'] is not None
                and v['actual_descent_cosine'] < -.02
                and v['clipped_gradient_descent_cosine'] >= .01
                and v['gradient_dot_actual_displacement'] > v['screening_epsilon']
                and v['observed_gold_ce_delta'] > v['screening_epsilon']
                for o in mature if o['epoch'] == epoch for v in [o['tracks'][track]])
        counts[track] = sum(by_epoch[track].values())
    supported = [t for t, count in counts.items() if '/ordinal/' not in t
                 and all(n >= 2 for n in by_epoch[t].values())]
    return {'supported': len(supported) >= 2, 'qualifying_classification_tracks': supported,
            'harmful_mature_observations_by_track': counts, 'harmful_counts_by_epoch': by_epoch,
            'paired_family_admission': False, 'quality_pass': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--lane', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--proof-sha256', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_direction_admitted_setup',
                                                 Path(__file__).with_name('run_workload_robust.py'))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    shared, core, _, control, admission, cases, proof, pins = old.setup(SimpleNamespace(worker_sha256=OLD_SHA))
    pins.update({str(Path(__file__).resolve()): args.worker_sha256,
                 str(HERE / 'predeclaration.md'): args.declaration_sha256,
                 str(HERE / 'measurement-proof-v2.json'): args.proof_sha256})
    reference, reference_sha = REFERENCES[args.lane]
    pins[str(reference)] = reference_sha
    measurement = shared.read(HERE / 'measurement-proof-v2.json')
    require(measurement['passed'] is measurement['cuda'] is True
            and measurement['cpu_head_execution'] is False, 'Actual CUDA measurement proof required')
    require(os.environ.get('FUZEV_ROOT') == str(ROOT), 'Explicit FuZev root required')
    directory = HERE / 'fit-output-v1' / args.lane
    weight_directory = WEIGHTS / args.lane
    require(not directory.exists() and not weight_directory.exists(), 'Never overwrite a diagnostic replay')
    case = cases['cases'][args.lane]['control']
    from fuzev_research.paths import source_revision
    observed_commit = source_revision(case['worktree'], 'fuzev-research')
    training = old.load_training(case)
    import torch
    from safetensors.torch import load_file, save_file
    from fuzev_research.features import FeatureStore
    from fuzev_research.shared_training_data import SharedFeatureCache
    from fuzev_research.state import Store
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    store, task = Store(), TASK + '.' + args.lane
    store.update(task, 'in_progress', 'CUDA owned FIT replay; nine fixed complete-FIT update observations.', .01)
    directory.mkdir(parents=True)
    weight_directory.mkdir(parents=True)
    core.require_empty_wal()
    features = FeatureStore(core.FEATURES, shared.read(core.FEATURES / 'identity.json'), read_only=True)
    try:
        fit_records = features.db.execute("SELECT count(*) FROM records WHERE role='fit'").fetchone()[0]
        control.begin_corpus_snapshot(features, fit_records)
        rows = core.admitted_rows(features, admission)
        cache = SharedFeatureCache(features, rows, 'fit', 'cpu')
        if args.lane == 'lm-assisted':
            training.validate_fit_ordinal_cache(cache)
        pins.update({str(core.FEATURES / name): value for name, value in cache.files.items()})
        cache.data = {key: value.to('cuda:0') for key, value in cache.data.items()}
        require(all(value.is_cuda and not value.requires_grad for value in cache.data.values()), 'Frozen CUDA cache required')
        model = training.SharedDecisionHead(seed=43, key_gain=training.PROFILE['key_gain']).cuda()
        require(sum(p.numel() for p in model.parameters()) == case['parameters']
                and len(model.state_dict()) == case['state_tensors'], 'Original retained geometry required')
        base, pairs = training.fitting_weights(rows), training.fit_preference_pairs(features, rows)
        admit = old.helper('_direction_prior_reference', old.OLD / 'admit.py')
        require(base == admit.base_weights(rows, args.lane), 'Exact original group/track weights required')
        control.check_updates(cache, base, pairs)
        priors = training.track_masses() if args.lane == 'feature-only' else training.track_masses('fit')
        grouped = defaultdict(list)
        for row in rows:
            grouped[core.row_track(row)].append(row)
        require(set(grouped) == set(training.TRACKS) == set(priors) and len(grouped) == 15, 'All15 original tracks required')
        grouped = {track: grouped[track] for track in sorted(grouped)}
        factors = {row['id']: base[row['id']] / (len(rows) * priors[track])
                   for track, owned in grouped.items() for row in owned}
        require(all(abs(math.fsum(factors[row['id']] for row in owned) - 1) < 1e-12
                    for owned in grouped.values()), 'Complete equal-group mean required per track')
        optimizer = training.make_optimizer(model)
        protocol = {'kind': 'mature-displacement-cuda-fit-diagnostic-v1', 'lane': args.lane,
            'worker_sha256': args.worker_sha256, 'declaration_sha256': args.declaration_sha256,
            'measurement_proof_sha256': args.proof_sha256, 'source_commit': case['commit'], 'source8': case['source8'],
            'setup_source': str(Path(old.__file__).resolve()), 'recorded_setup_source': str(OLD_WORKER),
            'profile': case['profile'], 'optimization_protocol': case['optimization_protocol'],
            'reference_checkpoint_sha256': reference_sha, 'cache_membership': cache.membership,
            'base_weights_sha256': shared.digest(sorted(base.items())), 'pairs': len(pairs),
            'training_records': len(rows), 'seed': 43, 'epochs': 5, 'batch_size': 256,
            'observed_epochs': OBSERVED_EPOCHS, 'observed_steps': OBSERVED_STEPS,
            'gold_diagnostic': 'complete owned FIT unsmoothed CE/logK; original group means',
            'cuda': True,
            'tf32': False, 'cpu_head_execution': False,
            'torch': torch.__version__, 'cuda_version': torch.version.cuda,
            'llm_loaded': False, 'learned_head_count': 1, 'quality_evaluated': False,
            'main_quality_evaluated': False, 'calibration_performed': False, 'qualified': False, 'servable': False}
        shared.write(directory / 'protocol.json', protocol)
        def emit(observation):
            shared.write(directory / f"observation-{observation['epoch']}-{observation['step_in_epoch']}.json", observation)
            print({'lane': args.lane, 'epoch': observation['epoch'], 'step': observation['step_in_epoch'],
                   'observer_seconds': observation['observer_seconds']}, flush=True)
        observer = Observer(torch, training, model, cache, grouped, factors, priors, emit)
        before_handle = optimizer.register_step_pre_hook(observer.pre)
        after_handle = optimizer.register_step_post_hook(observer.post)
        epochs = []
        started = time.monotonic()
        try:
            for epoch in range(1, 6):
                observer.epoch, observer.step = epoch, 0
                result = core.train_epoch(training, model, cache, base, pairs, optimizer, epoch, args.lane, 'control')
                require(observer.step == result['optimizer_steps_this_epoch'] == 477
                        and result['fit_records'] == 121953, 'Original477 exact-once updates required')
                shared.write(directory / f'epoch-{epoch:03d}.json', result)
                epochs.append(result)
                store.update(task, 'in_progress', f'{epoch}/5 unchanged CUDA epochs; no held data.', .1 + .85 * epoch / 5)
        finally:
            before_handle.remove()
            after_handle.remove()
        torch.cuda.synchronize()
        replay_seconds = time.monotonic() - started
        require(len(observer.observations) == 9 and observer.pending is None, 'All nine fixed observations required')
        steps = {name: int(optimizer.state[p]['step'].item()) for name, p in model.named_parameters()}
        require(set(steps.values()) == {2385}, 'Every parameter requires exactly2385 original updates')
        expected = load_file(str(reference), device='cuda')
        require(set(expected) == set(model.state_dict())
                and all(torch.equal(value, expected[name]) for name, value in model.state_dict().items()),
                'Observer must preserve the exact saved fresh control endpoint')
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                  str(weight_directory / 'head.safetensors'))
        features.db.execute('COMMIT')
        core.require_empty_wal()
        gate = mechanism_gate(observer.observations)
        shared.write(directory / 'completion.json', {**protocol, 'completed': True,
            'observed_source_commit': observed_commit, 'recorded_source_commit': case['commit'],
            'observed_source8': training.source_identity(), 'recorded_source8': case['source8'],
            'observed_input_sha256': {str(path): shared.sha(path) for path in pins},
            'actual_optimizer_updates': 2385, 'optimizer_steps_by_parameter': steps,
            'head_sha256': shared.sha(weight_directory / 'head.safetensors'),
            'recorded_head_sha256': reference_sha, 'exact_reference_endpoint': True,
            'observations': observer.observations, 'mechanism_gate': gate,
            'replay_seconds': replay_seconds, 'observer_seconds': observer.seconds,
            'input_sha256_before_and_after': pins})
        store.update(task, 'completed', 'Exact CUDA replay and actual displacement diagnostic complete; not a quality pass.', 1.)
        print({'lane': args.lane, 'mechanism_gate': gate}, flush=True)
    finally:
        features.db.close()


if __name__ == '__main__':
    main()
