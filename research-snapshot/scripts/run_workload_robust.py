"""Fixed CUDA excess-risk curriculum; model libraries and datasets are read-only.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.
"""

import argparse
from collections import defaultdict
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = ROOT / 'reports/2026-10-10-workload-robust'
TREE = Path(os.environ['FUZEV_WORKLOAD_SOURCE']).expanduser().resolve()
OPERATOR = TREE / 'fuzev_research/workload_weights.py'
OLD = ROOT / 'reports/2026-10-09-shared-head-generalization'
GENERAL = ROOT / 'reports/2026-10-06-generalization'
CLEAN = ROOT / 'reports/2026-10-10-retained-clean-refit'
WEIGHTS = ROOT / 'weights/workload-robust-20261010'
RESULTS = HERE / 'trial-v1'
REUSED = ROOT / 'weights/score-confidence-20261010/control/fit'
REUSED_HELD = ROOT / 'reports/2026-10-10-score-confidence-gate/trial-v1/control/held'
LANES, ARMS = ('feature-only', 'lm-assisted'), ('control', 'curriculum')
TASK = 'training.generalization.workload-robust'
IMAGE = os.environ.get('FUZEV_RESEARCH_CUDA_IMAGE')
PINS = {
    HERE / 'predeclaration.md': '13520718d0167d5b89d39e520c7d6fce67557d642f6c5e76ec2029b33dff95b0',
    OPERATOR: '61f2803e811520c35132245fc0b6f272b5b08c4ed7c6aa2f266350a393179b8f',
    HERE / 'prove_weights_cuda.py': '55c488bcdf9c0708884c1234128811172ce603bbac58a18626eba4dd7a2b660d',
    HERE / 'weights-proof-v1.json': '0d7253b8b88dcb4760452b37d49decc922f11acb1d6543d234e25047e4b10117',
    CLEAN / 'refit.py': 'e200bbfb01006f5ef58eba8f620ef8b9a67c53ad8897e86da276228ec0e3826d',
    OLD / 'capacity_train.py': '979fab0ec539b4d6f3c9f4b03a7282b9572a9dd4db046b0606808cb424f87531',
    OLD / 'capacity-sources-v1.json': '325f5c1008d73b814ad8c72cafe49fde1780292916530a319a9f27a5618a7206',
    OLD / 'capacity-cuda-admission-v7.json': 'c5265960b5f14a7bab40ec11e0860bf8603f511b6e62660be653e844501bf3fa',
    OLD / 'input-admission-v2.json': '1283335df26de930722ce1df550edabc887d38b81985ad2e37833cc946f7d693',
    OLD / 'admit.py': 'f74f81ade3c5cdbe23cdd1322d8e58f9e51ea2067ebf3a1396d728c55ca71393',
    OLD / 'compare.py': '01e7fea445f60749e5574ceb25a5fe8c13805e66712dd3c7c67ccf96aa4fdd0e',
    GENERAL / 'coverage_compare.py': '576ed078dfbe9c8f04ce861c975cc337b5dda8fea1f690ac9de19dd6d895ea22',
    REUSED / 'completion.json': 'fb830e0e3f17f99d5e2d1cb124de7b5273aca301784c969e0a4ebf84a2bbb231',
    REUSED / 'head.safetensors': '692e1cdaad5bf42b200580ca515417594862d4ae448fe58888d8be1e7bd796bd',
}


def helper(name='_robust_shared_helpers', path=None):
    spec = importlib.util.spec_from_file_location(name, path if path is not None else CLEAN / 'refit.py')
    shared = importlib.util.module_from_spec(spec)
    sys.modules[name] = shared
    spec.loader.exec_module(shared)
    return shared


def load_training(case):
    package = ModuleType('_robust_training')
    package.__name__ = f'_robust_training_{id(package):x}'
    package.__path__ = [str(Path(case['worktree']) / 'fuzev_research')]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + '.shared_train')


def setup(args):
    shared = helper()
    pins = {str(path): value for path, value in PINS.items()}
    pins[str(Path(__file__).resolve())] = args.worker_sha256
    weights_proof = shared.read(HERE / 'weights-proof-v1.json')
    shared.require(weights_proof['passed'] is weights_proof['cuda'] is True
        and weights_proof['gold_risk_synthetic_rows'] == 4 and weights_proof['cpu_head_execution'] is False,
        'Actual CUDA controller and original-gold statistic proof required')
    operator = helper('_robust_weight_controller', OPERATOR)
    core = helper('_robust_original_cuda_core', OLD / 'capacity_train.py')
    cases = shared.read(OLD / 'capacity-sources-v1.json')
    proof = shared.read(OLD / 'capacity-cuda-admission-v7.json')
    pins.update(proof['input_sha256_before_and_after'])
    helper('folds', GENERAL / 'folds.py')
    control = helper('controls', GENERAL / 'controls.py')
    # The epoch helper resolves controls dynamically; supply the loaded math module.
    core.controls = lambda: control
    admission = shared.read(OLD / 'input-admission-v2.json')
    pins.update(admission['input_sha256_before_and_after'])
    reused = shared.read(REUSED / 'completion.json')
    case = cases['cases']['lm-assisted']['control']
    shared.require(reused['completed'] is reused['cuda'] is True and reused['cpu_head_execution'] is False
        and reused['profile'] == case['profile']
        and reused['optimization_protocol'] == case['optimization_protocol']
        and reused['training_records'] == 121953 and reused['actual_optimizer_updates'] == 2385
        and reused['seed'] == 43 and reused['epochs'][-1]['epoch'] == 5,
        'Reused A control must be the exact fresh five-epoch matched recipe, not a historical milestone')
    return shared, core, operator, control, admission, cases, proof, pins


def fit_directory(lane, arm):
    return REUSED if (lane, arm) == ('lm-assisted', 'control') else WEIGHTS / lane / arm / 'fit'


def barrier(shared, cases, pins, worker_sha):
    finals = {}
    for lane in LANES:
        for arm in ARMS:
            path = fit_directory(lane, arm)
            value = shared.read(path / 'completion.json')
            shared.require(value['completed'] is True and value['training_records'] == 121953
                and value['actual_optimizer_updates'] == 2385, 'All fixed fits must finish before held access')
            if path != REUSED:
                shared.require(value['lane'] == lane and value['arm'] == arm,
                    'Fresh final identity differs')
            pins[str(path / 'head.safetensors')] = value['checkpoint_sha256']
            pins[str(path / 'completion.json')] = shared.sha(path / 'completion.json')
            finals[lane, arm] = value
    return finals


def phase(args):
    shared, core, operator, control, admission, cases, proof, pins = setup(args)
    if args.phase == 'compare':
        comparison(args, shared, core, admission, cases, pins)
        return
    lane, arm = args.lane, args.arm
    shared.require(lane in LANES and arm in ARMS and (lane, arm) != ('lm-assisted', 'control'), 'Explicit fresh case required')
    finals = barrier(shared, cases, pins, args.worker_sha256) if args.phase == 'held' else None
    directory = fit_directory(lane, arm) if args.phase == 'fit' else RESULTS / lane / arm / 'held'
    shared.require(not directory.exists() and os.environ.get('FUZEV_ROOT') == str(ROOT), 'Exclusive phase and explicit root required')
    case = cases['cases'][lane]['control']
    from fuzev_research.paths import source_revision
    observed_commit = source_revision(case['worktree'], 'fuzev-research')
    training = load_training(case)
    import torch
    from safetensors.torch import load_file, save_file
    from fuzev_research.features import FeatureStore
    from fuzev_research.shared_training_data import SharedFeatureCache
    from fuzev_research.state import Store
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    store, task = Store(), TASK + '.' + lane + '.' + arm + '.' + args.phase
    store.update(task, 'in_progress', 'Owned cache loading; all head/risk/controller execution stays on CUDA.', .01)
    directory.mkdir(parents=True, exist_ok=False)
    core.require_empty_wal()
    features = FeatureStore(core.FEATURES, shared.read(core.FEATURES / 'identity.json'), read_only=True)
    try:
        fit_records = features.db.execute("SELECT count(*) FROM records WHERE role='fit'").fetchone()[0]
        control.begin_corpus_snapshot(features, fit_records)
        if args.phase == 'fit':
            rows = core.admitted_rows(features, admission)
        else:
            allowed = set(admission['held_ids'])
            rows = [row for row in features.records('fit') if row['id'] in allowed]
            shared.require(len(rows) == len(allowed) == 63225 and {row['id'] for row in rows} == allowed, 'Complete original internal FIT2 required')
        cache = SharedFeatureCache(features, rows, 'fit', 'cpu')
        if args.phase == 'fit' and lane == 'lm-assisted':
            training.validate_fit_ordinal_cache(cache)
        pins.update({str(core.FEATURES / name): value for name, value in cache.files.items()})
        cache.data = {name: value.to('cuda:0') for name, value in cache.data.items()}
        shared.require(all(value.is_cuda and not value.requires_grad for value in cache.data.values()), 'Frozen CUDA cache required')
        model = training.SharedDecisionHead(seed=43, key_gain=training.PROFILE['key_gain']).cuda()
        shared.require(sum(p.numel() for p in model.parameters()) == case['parameters']
            and len(model.state_dict()) == case['state_tensors'], 'Unchanged retained head geometry required')
        protocol = {'kind': 'bounded-excess-risk-cuda-' + args.phase + '-v1', 'lane': lane, 'arm': arm,
            'source_commit': case['commit'], 'source8': case['source8'], 'base_library_profile': case['profile'],
            'base_optimizer_and_loss_protocol': case['optimization_protocol'],
            'effective_policy': 'original prior' if arm == 'control' else 'epoch classification excess-risk curriculum',
            'worker_sha256': shared.sha(__file__), 'recorded_worker_sha256': args.worker_sha256,
            'controller_sha256': shared.sha(OPERATOR), 'recorded_controller_sha256': PINS[OPERATOR],
            'declaration_sha256': shared.sha(HERE / 'predeclaration.md'),
            'recorded_declaration_sha256': PINS[HERE / 'predeclaration.md'], 'cuda': True,
            'tf32': False, 'cpu_head_execution': False, 'cache': cache.provenance(),
            'learned_head_count': 1, 'official_role': 'fit', 'qualified': False, 'servable': False,
            'main_quality_evaluated': False, 'calibration_performed': False, 'llm_loaded': False,
            'torch': torch.__version__, 'cuda_version': torch.version.cuda, 'producer_lineage': features.producer_origins()}
        if args.phase == 'fit':
            base = training.fitting_weights(rows)
            reference = helper('_robust_prior_reference', OLD / 'admit.py')
            shared.require(base == reference.base_weights(rows, lane), 'Original source/group prior weights required')
            pairs = training.fit_preference_pairs(features, rows)
            control.check_updates(cache, base, pairs)
            if lane == 'lm-assisted':
                reused = shared.read(REUSED / 'completion.json')
                shared.require(cache.membership == reused['cache']['membership']
                    and len(pairs) == reused['js_pairs'],
                    'Reusable A control must match actual owned membership, base factors and JS pairs')
            tracks = sorted(training.TRACKS)
            positions = {track: i for i, track in enumerate(tracks)}
            row_track = [core.row_track(row) for row in rows]
            prior_dict = training.track_masses() if lane == 'feature-only' else training.track_masses('fit')
            prior = torch.tensor([prior_dict[track] for track in tracks], device='cuda', dtype=torch.float64)
            current = prior.clone()
            mask = torch.tensor(['/ordinal/' not in track for track in tracks], device='cuda')
            index = torch.tensor([positions[track] for track in row_track], device='cuda')
            risk_factor = torch.tensor([base[row['id']] / (len(rows) * prior_dict[track])
                for row, track in zip(rows, row_track)], device='cuda', dtype=torch.float64)
            protocol.update(training_records=121953, epochs=5, seed=43, batch_size=256,
                eta=1., ratio_bounds=[.25, 4.], track_order=tracks,
                prior_track_mass=prior.cpu().tolist(), base_weights_sha256=shared.digest(sorted(base.items())),
                js_pairs=len(pairs), pairs_sha256=shared.digest(sorted((key, pair['parent']['id'], pair['permutation']) for key, pair in pairs.items())),
                group_risk_averaging='rows/original source-group, then equal original groups/track',
                controller_inputs='complete owned FIT only; original-target normalized KL', epoch_selection=False)
            shared.write(directory / 'protocol.json', protocol)
            optimizer = training.make_optimizer(model)
            epochs = []
            for epoch in range(1, 6):
                ratios = (current / prior).cpu().tolist()
                effective = {row['id']: base[row['id']] * ratios[positions[track]] for row, track in zip(rows, row_track)}
                control.check_updates(cache, effective, pairs)
                torch.cuda.synchronize()
                started = time.monotonic()
                result = core.train_epoch(training, model, cache, effective, pairs, optimizer, epoch, lane, 'control')
                torch.cuda.synchronize()
                result['fit_seconds'] = time.monotonic() - started
                shared.require(result['fit_records'] == 121953 and result['optimizer_steps_this_epoch'] == 477, 'Exact-once 477-update epoch required')
                risk = torch.zeros(15, device='cuda', dtype=torch.float64)
                model.eval()
                risk_started = time.monotonic()
                with torch.no_grad():
                    for offset in range(0, len(rows), 256):
                        selected = rows[offset:offset + 256]
                        batch = cache.batch(selected)
                        losses = operator.original_gold_risk(torch, training.shared_logits(model, batch), batch['targets'], batch['mask'])
                        risk.index_add_(0, index[offset:offset + len(selected)], losses * risk_factor[offset:offset + len(selected)])
                    next_mass = operator.update(torch, prior, current, risk, mask) if arm == 'curriculum' and epoch < 5 else current.clone()
                torch.cuda.synchronize()
                result.update(risk_seconds=time.monotonic() - risk_started, track_mass=current.cpu().tolist(),
                    original_gold_kl=risk.cpu().tolist(), next_training_mass=next_mass.cpu().tolist(),
                    effective_weights_sha256=shared.digest(sorted(effective.items())))
                current = next_mass
                shared.write(directory / f'epoch-{epoch:03d}.json', result)
                epochs.append(result)
                store.update(task, 'in_progress', f'{epoch}/5 CUDA epochs and complete FIT risk passes; no held inputs.', .1 + .85 * epoch / 5)
                print({'lane': lane, 'arm': arm, 'epoch': epoch, 'fit_seconds': result['fit_seconds'], 'risk_seconds': result['risk_seconds']}, flush=True)
            steps = {name: int(optimizer.state[p]['step'].item()) for name, p in model.named_parameters()}
            shared.require(set(steps.values()) == {2385}, 'All retained parameters need all2385 actual updates')
            optimizer.zero_grad(set_to_none=True)
            save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(directory / 'head.safetensors'))
            result = {'epochs': epochs, 'actual_optimizer_updates': 2385, 'optimizer_steps_by_parameter': steps,
                'checkpoint_sha256': shared.sha(directory / 'head.safetensors')}
        else:
            final = finals[lane, arm]
            model.load_state_dict(load_file(str(fit_directory(lane, arm) / 'head.safetensors'), device='cuda'), strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            ledger = shared.read(ROOT / 'reports/2026-10-08-postcandidate-assessment/fit-exclusions-input-only.json')
            desc = {row['id']: row for row in ledger['input_only_descriptors'] if row['id'] in allowed}
            records = []
            with torch.no_grad():
                for offset in range(0, len(rows), 256):
                    selected = rows[offset:offset + 256]
                    batch = cache.batch(selected)
                    logits = training.shared_logits(model, batch).cpu().tolist()
                    fields = {key: batch[key].cpu().tolist() for key in ('mask', 'targets', 'baseline', 'ordinal_values')}
                    for i, row in enumerate(selected):
                        raw = desc[row['id']]
                        shared.require(raw['original_fold'] == 2 and not raw['quarantined_fit'], 'Only unchanged internal FIT2 allowed')
                        count = sum(fields['mask'][i])
                        records.append(control.describe(row, {'component_sha256': raw['global_component_sha256'], 'length': raw['length']},
                            fields['baseline'][i][:count], logits[i][:count], fields['targets'][i][:count], fields['ordinal_values'][i][:count]))
                    store.update(task, 'in_progress', f'CUDA FIT2 {min(offset + 256, len(rows))}/63225; not main quality.', .1 + .85 * min(offset + 256, len(rows)) / len(rows))
            protocol.update(held_records=63225, checkpoint_sha256=final['checkpoint_sha256'])
            shared.write(directory / 'protocol.json', protocol)
            shared.write(directory / 'predictions.json', {**protocol, 'records': records})
            result = {'prediction_sha256': shared.sha(directory / 'predictions.json'), 'records': 63225}
        features.db.execute('COMMIT')
        core.require_empty_wal()
        shared.write(directory / 'completion.json', {**protocol, **result, 'completed': True,
            'protocol_sha256': shared.sha(directory / 'protocol.json'), 'input_sha256_before_and_after': pins,
            'observed_input_sha256': {str(path): shared.sha(path) for path in pins},
            'observed_source_commit': observed_commit, 'recorded_source_commit': case['commit'],
            'observed_source8': training.source_identity(), 'recorded_source8': case['source8']})
        store.update(task, 'completed', 'Fixed CUDA phase complete; main quality remains unqualified.', 1.)
    finally:
        features.db.close()


def comparison(args, shared, core, admission, cases, pins):
    barrier(shared, cases, pins, args.worker_sha256)
    pins[str(REUSED_HELD / 'completion.json')] = '09736638285f8f0624c12a2ba9f449ff725dc1bda494449856b6872bd1e15db4'
    pins[str(REUSED_HELD / 'predictions.json')] = '311515205fbd858e64ebc8f5e4ab0adf4d8a2fa831f7dd08d7541de9e11bde19'
    sys.path.insert(0, str(GENERAL))
    stats = helper('_robust_statistics', GENERAL / 'coverage_compare.py')
    decision = helper('_robust_fixed_decision', OLD / 'compare.py')
    arrays = {}
    for lane in LANES:
        for arm in ARMS:
            directory = REUSED_HELD if (lane, arm) == ('lm-assisted', 'control') else RESULTS / lane / arm / 'held'
            completion = shared.read(directory / 'completion.json')
            shared.require(completion['completed'] is True and completion['records'] == 63225, 'Both lanes/arms require complete held predictions')
            path = directory / 'predictions.json'
            pins[str(path)] = completion['prediction_sha256']
            arrays[lane, arm] = shared.read(path)['records']
    pairs, metrics = {}, {}
    anchor = arrays['lm-assisted', 'control']
    for lane in LANES:
        before, after = arrays[lane, 'control'], arrays[lane, 'curriculum']
        shared.require(len(before) == len(after) == len(anchor) == 63225
            and all({k: v for k, v in a.items() if k != 'student'} == {k: v for k, v in b.items() if k != 'student'}
                == {k: v for k, v in c.items() if k != 'student'} for a, b, c in zip(before, after, anchor)),
            'All fresh/reused cases must share identical complete input/teacher/gold semantics')
        pairs[lane] = [(2, a, b) for a, b in zip(before, after)]
        metrics[lane] = stats.build_metrics(pairs[lane])
    ledger = shared.read(ROOT / 'reports/2026-10-08-postcandidate-assessment/fit-exclusions-input-only.json')
    signatures = {key: value['sources'] for key, value in ledger['global_component_certificates'].items()}
    draws = decision.replicas(metrics, signatures)
    verdicts = {lane: decision.decision(metrics[lane], draws[lane], stats.CLASS_TRACKS, stats.ORDINAL_TRACKS, minimum_support=200)
                for lane in LANES}
    shared.write(RESULTS / 'comparison.json', {'kind': 'bounded-excess-risk-internal-comparison-v1', 'completed': True,
        'lanes': verdicts, 'reused_a_control': True, 'records_per_case': 63225,
        'worker_sha256': shared.sha(__file__), 'recorded_worker_sha256': args.worker_sha256,
        'recorded_input_sha256': pins, 'observed_input_sha256': {str(path): shared.sha(path) for path in pins},
        'bootstrap': {'draws': 1000, 'seed': 17, 'unit': 'global input component', 'strata': 'source signatures'},
        'main_quality_evaluated': False, 'calibration_performed': False, 'qualified': False, 'servable': False})
    print({lane: value['primary'] for lane, value in verdicts.items()}, flush=True)


def all_phases(args):
    shared = helper()
    shared.require(not WEIGHTS.exists() and not RESULTS.exists(), 'Never overwrite a prior trial')
    phases = [('fit', 'feature-only', 'control'), ('fit', 'feature-only', 'curriculum'), ('fit', 'lm-assisted', 'curriculum'),
              ('held', 'feature-only', 'control'), ('held', 'feature-only', 'curriculum'), ('held', 'lm-assisted', 'curriculum'),
              ('compare', 'feature-only', 'control')]
    source_mounts = []
    for source in sorted({Path(__file__).resolve().parents[1], TREE}):
        if not source.is_relative_to(ROOT):
            source_mounts.extend(('-v', f'{source}:{source}:ro'))
    for phase_name, lane, arm in phases:
        options = []
        for flag, value in (('--name', args.container_name.format(lane=lane, arm=arm, phase=phase_name) if args.container_name else None),
                            ('--cpuset-cpus', args.cpuset_cpus), ('--user', args.user)):
            if value is not None:
                options.extend((flag, value))
        command = ['docker', 'run', '--rm', *options, '--gpus', 'all',
            '--network', 'none', '--ipc', 'host',
            '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'NVIDIA_VISIBLE_DEVICES=all', '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1', '-e', f'FUZEV_ROOT={ROOT}',
            '-e', f'FUZEV_WORKLOAD_SOURCE={TREE}', '-v', f'{ROOT}:{ROOT}:ro', *source_mounts,
            '-v', f'{WEIGHTS.parent}:{WEIGHTS.parent}:rw', '-v', f'{HERE}:{HERE}:rw',
            '-v', f'{ROOT / "state"}:{ROOT / "state"}:rw', '--entrypoint', 'python3', args.image, '-B', str(Path(__file__).resolve()),
            '--phase', phase_name, '--lane', lane, '--arm', arm, '--worker-sha256', args.worker_sha256]
        print(f'Starting sequential CUDA phase {lane}/{arm}/{phase_name}', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--phase', choices=('all', 'fit', 'held', 'compare'), required=True)
    parser.add_argument('--lane', choices=LANES)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--image', default=IMAGE, help="The lab's CUDA build image (Torch 2.13.0+cu130); required for --phase all")
    parser.add_argument('--container-name', help='Optional name format with {lane}, {arm}, {phase}')
    parser.add_argument('--cpuset-cpus', help='Optional Docker CPU set')
    parser.add_argument('--user', help='Optional Docker user')
    args = parser.parse_args()
    if args.phase == 'all' and not args.image:
        parser.error('--phase all requires --image or FUZEV_RESEARCH_CUDA_IMAGE')
    all_phases(args) if args.phase == 'all' else phase(args)
