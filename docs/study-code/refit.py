"""Quarantine-clean retained controls: CUDA FIT, then endpoint DEV only."""

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'reports/2026-10-09-shared-head-generalization'
ASSESS = ROOT / 'reports/2026-10-08-postcandidate-assessment'
FEATURES = ROOT / 'data/features/native-4.05-dimension-choice-v4'
OUTPUT = ROOT / 'weights/retained-clean-20261010'
TASK = 'training.generalization.retained-clean'
IMAGE = os.environ.get('TRAINING_IMAGE')
LANES = {'feature-only': 42, 'lm-assisted': 43}
PINS = {
    HERE / 'predeclaration.md': 'c7e45b50aeec3bb07bee6ca4031fda77e6582e46d6bbae574159f8f1bec90268',
    ASSESS / 'fit-exclusions-input-only.json': '80d05f4d8c7d1b7e9272004d7665558b9b3afcfcdb640463f2f1c514e3d3b0b6',
    ASSESS / 'separation-admission-result.json': 'aa365ba52113a11c15f72e473562eada7f01ad2ba26af6322f2a27e242a4c767',
    ASSESS / 'training-metadata-admission.json': '9cbebeef74a683958c27e4d6943d1cae6decd9084b081243258c58f6868c644e',
    FEATURES / 'identity.json': '498ae19fde58540c0961425bec00e5f2eead1f4fa8539631415750be99a9d69a',
    FEATURES / 'index.sqlite3': 'd45c19700fe637a34eb2e1627ff1cb154d5f0955a9ef0c00c198ff1027735123',
    OLD / 'capacity_train.py': '979fab0ec539b4d6f3c9f4b03a7282b9572a9dd4db046b0606808cb424f87531',
    OLD / 'capacity-cuda-admission-v7.json': 'c5265960b5f14a7bab40ec11e0860bf8603f511b6e62660be653e844501bf3fa',
    OLD / 'capacity-sources-v1.json': '325f5c1008d73b814ad8c72cafe49fde1780292916530a319a9f27a5618a7206',
    ROOT / 'weights/shared-quality-search-20261005/round-42/finalist-lock.json': '5238f80eadb1223ad39563ef16e102b4e9a0573213699f8954c8fc76285af0ab',
    ROOT / 'weights/shared-quality-search-20261005/round-43/finalist-lock.json': '32f256b6703e490e013c74bbdaef2bd0f7797dc531f47980122b7a613d361d14',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def bookend(pins):
    for path, pin in pins.items():
        require(sha(path) == pin, 'Changed immutable source/input: ' + str(path))


def load(name, path, pin):
    bookend({path: pin})
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    bookend({path: pin})
    return module


def track(row):
    return '/'.join((row['source'], row['primitive'], row['modality'], row.get('attribute') or row.get('adaptation') or 'main'))


def membership(rows):
    return digest(sorted((row['id'], row['group_'], track(row)) for row in rows))


def prepare(args):
    require(args.output.resolve() == HERE / 'admission-v1.json' and not args.output.exists(), 'Exclusive current admission path required')
    pins = {str(path): pin for path, pin in PINS.items()}
    pins[str(Path(__file__))] = args.worker_sha256
    bookend(pins)
    separation = read(ASSESS / 'separation-admission-result.json')
    require(separation['universal_closure_complete'] is separation['universal_protected_input_admitted'] is True
        and separation['unresolved_binding_blockers'] == []
        and separation['quarantined_fit_records'] == 12, 'Completed universal input separation required')
    pins.update(separation['input_sha256_before_and_after'])
    ledger = read(ASSESS / 'fit-exclusions-input-only.json')
    descriptors = {row['id']: row for row in ledger['input_only_descriptors']}
    certificates = ledger['global_component_certificates']
    require(len(descriptors) == len(ledger['input_only_descriptors']) == 259539
        and ledger['fixed_api_input_binding']['actual_request_inputs_bound'] is True
        and ledger['fixed_api_input_binding']['all_original_test'] is True, 'Original complete input ledger/API binding required')
    protected = {row['global_component_sha256'] for row in descriptors.values() if row['role'] != 'fit'}
    fit_members = {row[0]: row for row in ledger['fit_input_membership']}
    require(len(fit_members) == 185190, 'Complete original FIT membership required')
    excluded, eligible = set(), set()
    for identifier, member in fit_members.items():
        row = descriptors[identifier]
        require(row['role'] == 'fit' and member[1] == row['group']
            and member[3] == row['global_component_sha256'] and member[4] == row['original_fold']
            and member[5] is row['quarantined_fit'], 'Original quarantine/member binding differs')
        component = member[3]
        if member[5]:
            excluded.add(identifier)
            require(component in protected, 'Quarantine must have a protected overlap')
        else:
            eligible.add(identifier)
            cert = certificates[component]
            require(component not in protected and cert['roles'] == ['fit']
                and cert['contains_protected_nonfit'] is False
                and cert['contains_inner_fold_conflict'] is False, 'Protected or inconsistent component entered full eligible FIT')
    require(len(eligible) == 185178 and len(excluded) == 12, 'Do not restore quarantined rows')
    eligible_components = {descriptors[i]['global_component_sha256'] for i in eligible}
    require({i for i, row in descriptors.items() if row['global_component_sha256'] in eligible_components} == eligible,
        'Universal linked-component closure must be complete and FIT-only')
    wal = FEATURES / 'index.sqlite3-wal'
    require(not wal.exists() or wal.stat().st_size == 0, 'Unbound index WAL')
    with sqlite3.connect((FEATURES / 'index.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        require(db.execute('SELECT 1 FROM failures LIMIT 1').fetchone() is None, 'Extraction failures remain')
        rows = [dict(row) for row in db.execute("SELECT id,group_,role,primitive,modality,source,shard,row_,"
            "json_extract(metadata,'$.adaptation') AS adaptation,json_extract(metadata,'$.attribute') AS attribute,"
            "json_extract(metadata,'$.execution.branch_prefix_tokens') AS length,"
            "json_extract(metadata,'$.execution.query_token_sha256') AS query_sha256,"
            "json_array_length(json_extract(metadata,'$.candidates')) AS option_count "
            "FROM records WHERE role IN ('fit','development') ORDER BY shard,row_")]
    require(len(rows) == 185190 + 24937, 'Complete FIT/DEV metadata required')
    for row in rows:
        raw = descriptors[row['id']]
        require(raw['group'] == row['group_'] and raw['track'] == track(row)
            and all(raw[key] == row[key] for key in ('role','source','primitive','modality','length','query_sha256','option_count')),
            'Index/input descriptor binding changed')
    training = [row for row in rows if row['id'] in eligible]
    development = [row for row in rows if row['role'] == 'development']
    metadata = read(ASSESS / 'training-metadata-admission.json')
    require(metadata['passed'] is True and metadata['original_full_readiness_reused'] is True, 'Original complete readiness required')
    pins.update(metadata['input_sha256_before_and_after'])
    cases = read(OLD / 'capacity-sources-v1.json')['cases']
    references = {}
    for lane, round_number in LANES.items():
        lock = read(ROOT / f'weights/shared-quality-search-20261005/round-{round_number}/finalist-lock.json')['finalists'][lane]
        case = cases[lane]['control']
        tree = Path(case['worktree'])
        require(case['commit'] == lock['source_commit'] and case['source8'] == lock['source']
            and case['profile'] == metadata['retained_sources'][lane]['profile'], 'Retained control source differs')
        pins.update({str(tree / 'fuzev_research' / name): pin for name, pin in case['source8'].items()})
        trial = Path(lock['trial'])
        checkpoint = trial / 'checkpoints/epoch-005'
        reference_manifest = read(checkpoint / 'manifest.json')
        require(lock['epoch'] == reference_manifest['epoch'] == 5
            and reference_manifest['weights_sha256'] == lock['weights_sha256'], 'Exact retained epoch-5 reference required')
        pins[str(trial / 'protocol.json')] = lock['protocol_sha256']
        pins[str(trial / 'selection-lock.json')] = lock['selection_lock_sha256']
        pins[str(checkpoint / 'head.safetensors')] = lock['weights_sha256']
        references[lane] = {'checkpoint': str(checkpoint / 'head.safetensors'), 'sha256': lock['weights_sha256'],
            'manifest_sha256': sha(checkpoint / 'manifest.json'), 'feature_origins': reference_manifest['feature_origins']}
        pins[str(checkpoint / 'manifest.json')] = references[lane]['manifest_sha256']
    bookend(pins)
    result = {'kind': 'retained-clean-full-fit-admission-v1', 'passed': True, 'worker_sha256': args.worker_sha256,
        'declaration_sha256': PINS[HERE / 'predeclaration.md'], 'training_ids': [row['id'] for row in training],
        'training_records': len(training), 'training_membership_sha256': membership(training),
        'development_ids': [row['id'] for row in development], 'development_records': len(development),
        'development_membership_sha256': membership(development), 'excluded_ids_sha256': digest(sorted(excluded)),
        'quarantined_fit_records': 12, 'original_fit2_now_training': True, 'official_roles_changed': False,
        'component_closed': True, 'protected_disjoint': True, 'cases': {lane: cases[lane]['control'] for lane in LANES},
        'references': references, 'preparation': metadata['preparation'], 'input_sha256_before_and_after': pins,
        'targets_accessed': False, 'head_executed': False, 'cuda_initialized': False, 'qualified': False, 'servable': False}
    require(result['training_records'] == 185178 and result['development_records'] == 24937, 'Exact full eligible membership required')
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ('kind','passed','training_records','development_records','quarantined_fit_records')}), flush=True)


def run(args):
    require(os.environ.get('FUZEV_ROOT') == str(ROOT), 'Explicit project root required')
    require(args.output.resolve() == OUTPUT / args.lane / args.phase and not args.output.exists(), 'Exclusive phase-specific artifact directory required')
    require(args.admission.resolve() == HERE / 'admission-v1.json', 'Only the current full-FIT admission is allowed')
    bookend({Path(__file__): args.worker_sha256, args.admission: args.admission_sha256})
    admission = read(args.admission)
    require(admission['passed'] is True and admission['worker_sha256'] == args.worker_sha256
        and admission['kind'] == 'retained-clean-full-fit-admission-v1' and admission['protected_disjoint'] is True,
        'Current clean-full-FIT scope required')
    pins = dict(admission['input_sha256_before_and_after'])
    pins[str(args.admission)] = args.admission_sha256
    bookend(pins)
    sys.path.insert(0, str(OLD))
    core = load('_retained_unchanged_cuda_core', OLD / 'capacity_train.py', PINS[OLD / 'capacity_train.py'])
    proof = read(OLD / 'capacity-cuda-admission-v7.json')
    require(proof['passed'] is proof['worker_train_epoch_admitted'] is True
        and proof['worker_sha256'] == PINS[OLD / 'capacity_train.py'], 'Actual original update CUDA proof required')
    core.validate_proof(proof, read(OLD / 'capacity-sources-v1.json'),
        {key: proof[key] for key in ('worker_sha256','sources_sha256','declaration_sha256','operator_sha256')})
    pins.update(proof['input_sha256_before_and_after'])
    bookend(pins)
    case = admission['cases'][args.lane]
    training, control = core.import_training(case), core.controls()
    import torch
    from safetensors.torch import load_file, save_file
    from fuzev_research.features import FeatureStore
    from fuzev_research.shared_training_data import SharedFeatureCache
    from fuzev_research.state import Store
    core.configure_threads(torch)
    core.configure_cuda(torch)
    store, task = Store(), TASK + '.' + args.lane + '.' + args.phase
    store.update(task, 'in_progress', 'Original read-only feature cache loading; all model execution stays on CUDA.', .01)
    args.output.mkdir(parents=True, exist_ok=False)
    core.require_empty_wal()
    features = FeatureStore(FEATURES, read(FEATURES / 'identity.json'), read_only=True)
    try:
        control.require_corpus_pin(training, features, args.lane)
        control.begin_corpus_snapshot(features, 185190)
        require(features.producer_origins() == admission['references'][args.lane]['feature_origins'], 'Producer lineage changed')
        role = 'fit' if args.phase == 'fit' else 'development'
        allowed = set(admission['training_ids' if role == 'fit' else 'development_ids'])
        rows = [row for row in features.records(role) if row['id'] in allowed]
        require(len(rows) == len(allowed) and {row['id'] for row in rows} == allowed
            and membership(rows) == admission['training_membership_sha256' if role == 'fit' else 'development_membership_sha256'],
            'Complete current phase ownership required')
        cache = SharedFeatureCache(features, rows, role, 'cpu')
        if args.phase == 'fit' and args.lane == 'lm-assisted':
            training.validate_fit_ordinal_cache(cache)
        pins.update({str(FEATURES / name): pin for name, pin in cache.files.items()})
        bookend(pins)
        cache.data = {key: value.to('cuda:0') for key, value in cache.data.items()}
        require(all(value.device.type == 'cuda' and not value.requires_grad for value in cache.data.values()), 'Frozen CUDA inputs required')
        torch.manual_seed(43)
        model = training.SharedDecisionHead(seed=43, key_gain=training.PROFILE['key_gain']).to('cuda:0')
        require(sum(p.numel() for p in model.parameters()) == case['parameters']
            and len(model.state_dict()) == case['state_tensors']
            and all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters()), 'Exact original FP32 CUDA head required')
        protocol = {'kind': 'retained-clean-cuda-' + args.phase + '-v1', 'lane': args.lane,
            'admission_sha256': args.admission_sha256, 'worker_sha256': args.worker_sha256,
            'source_commit': case['commit'], 'source8': case['source8'], 'profile': case['profile'],
            'optimization': case['optimization_protocol'], 'cache': cache.provenance(),
            'tf32': False, 'cpu_head_execution': False, 'llm_loaded': False,
            'calibration_performed': False, 'main_quality_evaluated': False, 'qualified': False, 'servable': False,
            'official_roles_changed': False, 'held_epoch_selection': False, 'learned_head_count': 1,
            'producer_lineage': features.producer_origins(), 'torch': torch.__version__, 'cuda_version': torch.version.cuda}
        if args.phase == 'fit':
            weights = training.fitting_weights(rows)
            audit_weights = load('_retained_weight_reference', OLD / 'admit.py', 'f74f81ade3c5cdbe23cdd1322d8e58f9e51ea2067ebf3a1396d728c55ca71393')
            require(weights == audit_weights.base_weights(rows, args.lane), 'Original full-eligible group/track masses required')
            pairs = training.fit_preference_pairs(features, rows)
            control.check_updates(cache, weights, pairs)
            protocol.update(epochs=5, seed=43, batch_size=256, training_records=len(rows),
                base_weights_sha256=digest(sorted(weights.items())), js_pairs=len(pairs),
                js_pairs_sha256=digest(sorted((key, pair['parent']['id'], pair['permutation']) for key,pair in pairs.items())),
                original_fit2_now_training=True, optimizer_core_sha256=PINS[OLD / 'capacity_train.py'])
            write(args.output / 'protocol.json', protocol)
            optimizer = training.make_optimizer(model)
            epochs = []
            for epoch in range(1,6):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(0)
                started = time.monotonic()
                result = core.train_epoch(training, model, cache, weights, pairs, optimizer, epoch, args.lane, 'control')
                torch.cuda.synchronize()
                result.update(seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                require(result['fit_records'] == 185178 and result['optimizer_steps_this_epoch'] == 724,
                    'Exactly one complete eligible visit per epoch required')
                write(args.output / f'epoch-{epoch:03d}.json', result)
                epochs.append(result)
                store.update(task, 'in_progress', f'{epoch}/5 fixed CUDA epochs complete; no DEV selection.', .1+.85*epoch/5)
                print(json.dumps({k:result[k] for k in ('epoch','fit_records','fit_pairs','fit_loss','seconds','peak_allocated_bytes')}), flush=True)
            steps = {name:int(optimizer.state[p]['step'].item()) for name,p in model.named_parameters()}
            require(set(steps.values()) == {3620}, 'Every original tensor needs all 3,620 actual updates')
            optimizer.zero_grad(set_to_none=True)
            save_file({name:value.detach().cpu().contiguous() for name,value in model.state_dict().items()}, str(args.output / 'head.safetensors'))
            result = {'epochs':epochs, 'actual_optimizer_updates':3620, 'optimizer_steps_by_parameter':steps,
                'checkpoint_sha256':sha(args.output / 'head.safetensors'), 'training_records':len(rows)}
        else:
            # Both fits must finish before either endpoint is inspected.
            for lane in LANES:
                fit = OUTPUT / lane / 'fit'
                completion = read(fit / 'completion.json')
                require(completion['completed'] is True and completion['worker_sha256'] == args.worker_sha256
                    and completion['admission_sha256'] == args.admission_sha256 and completion['actual_optimizer_updates'] == 3620,
                    'Two complete fixed CUDA fits are required before DEV')
                bookend({fit / 'head.safetensors':completion['checkpoint_sha256'], fit / 'protocol.json':completion['protocol_sha256']})
                pins[str(fit / 'completion.json')] = sha(fit / 'completion.json')
            fit = OUTPUT / args.lane / 'fit'
            final = read(fit / 'completion.json')
            model.load_state_dict(load_file(str(fit / 'head.safetensors'), device='cuda:0'), strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            canonical = training.development(model, cache)
            require(len(canonical['checks']) == 26, 'All original canonical DEV checks required')
            reference = training.SharedDecisionHead(seed=43, key_gain=training.PROFILE['key_gain']).to('cuda:0')
            reference.load_state_dict(load_file(admission['references'][args.lane]['checkpoint'], device='cuda:0'), strict=True)
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            ledger = read(ASSESS / 'fit-exclusions-input-only.json')
            metadata = {row['id']: {'component_sha256':row['global_component_sha256'], 'length':row['length']}
                for row in ledger['input_only_descriptors'] if row['role'] == 'development'}
            signatures = {key:value['sources'] for key,value in ledger['global_component_certificates'].items()}
            del ledger
            before, after = [], []
            model.eval()
            reference.eval()
            with torch.no_grad():
                for offset in range(0,len(rows),256):
                    selected = rows[offset:offset+256]
                    batch = cache.batch(selected)
                    old = training.shared_logits(reference,batch).cpu().tolist()
                    new = training.shared_logits(model,batch).cpu().tolist()
                    values = {key:batch[key].cpu().tolist() for key in ('mask','targets','baseline','ordinal_values')}
                    for i,row in enumerate(selected):
                        count = sum(values['mask'][i])
                        for output,logits in ((before,old),(after,new)):
                            output.append(control.describe(row, metadata[row['id']], values['baseline'][i][:count],
                                logits[i][:count], values['targets'][i][:count],values['ordinal_values'][i][:count]))
                    store.update(task, 'in_progress', f'CUDA DEV {min(offset+256,len(rows))}/{len(rows)}; main quality remains unqualified.',
                        .1+.65*min(offset+256,len(rows))/len(rows))
            pairs = list(zip(before,after))
            panel = load('_retained_dev_panel', HERE / 'dev_panel.py', args.panel_sha256).panel(pairs, signatures)
            write(args.output / 'predictions.json', {'role':'development','lane':args.lane,'records':after,
                'reference_records':before, 'checkpoint_sha256':final['checkpoint_sha256'],
                'reference_checkpoint_sha256':admission['references'][args.lane]['sha256']})
            protocol.update(development_records=len(rows), checkpoint_sha256=final['checkpoint_sha256'],
                reference_checkpoint_sha256=admission['references'][args.lane]['sha256'], panel_sha256=args.panel_sha256)
            pins[str(HERE / 'dev_panel.py')] = args.panel_sha256
            write(args.output / 'protocol.json', protocol)
            result = {'canonical':canonical, 'component_panel':panel, 'development_records':len(rows),
                'confirmation_passed':bool(canonical['eligible'] and panel['passed']),
                'prediction_sha256':sha(args.output / 'predictions.json')}
        features.db.execute('COMMIT')
        core.require_empty_wal()
        bookend(pins)
        require(training.source_identity() == case['source8'], 'Training sources changed during execution')
        write(args.output / 'completion.json', {**protocol, **result, 'completed':True,
            'protocol_sha256':sha(args.output / 'protocol.json'), 'source_bookends_passed':True,
            'input_sha256_before_and_after':pins})
        store.update(task, 'completed', 'CUDA phase complete. DEV confirmation is not a release-quality pass.', 1.)
        print(json.dumps({'completed':True,'phase':args.phase,'lane':args.lane,
            'confirmation_passed':result.get('confirmation_passed'), 'main_quality':'not_run'}), flush=True)
    finally:
        features.db.close()
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--phase',choices=('prepare','fit','dev'),required=True)
    parser.add_argument('--lane',choices=LANES)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker-sha256',required=True)
    parser.add_argument('--admission',type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--panel-sha256')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args)
    else:
        require(args.lane in LANES and args.admission is not None and args.admission_sha256 is not None,
            'Explicit lane/admission required')
        if args.phase == 'dev':
            require(args.panel_sha256 is not None, 'Bound component panel required')
        run(args)
