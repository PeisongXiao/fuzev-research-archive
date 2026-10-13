"""Historical CUDA epoch implementation imported by the five-epoch refit.

The retained refit calls train_epoch() directly. This dependency's independent
CLI describes an earlier eight-epoch study. The study-code index records its
original hash, required historical modules and omitted input ledgers.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = Path(__file__).resolve().parent
GENERAL = ROOT / 'reports/2026-10-06-generalization'
FEATURES = ROOT / 'data/features/native-4.05-dimension-choice-v4'
EPOCHS, BATCH_SIZE, SEED = 8, 256, 43
LANES = ('feature-only', 'lm-assisted')
ARMS = ('control', 'candidate')
PRIMITIVES = ('choice', 'noul', 'ordinal')
TORCH_VERSION = '2.13.0+cu130'
TOLERANCES = {'loss': {'atol': 2e-6, 'rtol': 2e-5}, 'gradients': {'atol': 2e-6, 'rtol': 2e-5},
              'parameters': {'atol': 2e-6, 'rtol': 2e-5}, 'moments': {'atol': 1e-7, 'rtol': 2e-5}}
CONSTANT_SHA = '0304a34773d0509c427e2eea1763c08d29c1941a229d817bd006fdb6b27cb556'
CONSTANT_OPERATOR_SHA = 'cec8bb7eba34ab48cd1dcd2483a509276a46b9c852e9e0d1de09ce581844f315'
PROOF_REUSE_SHA = '3fa40ef460d480932df8019dd3c71d9e73fe8ebb12d8ef73180cdd2de1c4409b'
EXCLUDED_PROOF_CLAIMS = ('real_data_used', 'real_targets_read', 'protected_outputs_read',
    'llm_loaded', 'head_checkpoint_loaded', 'quality_evaluated', 'quality', 'qualified', 'servable',
    'real_membership_admitted', 'full_fit_health_admitted', 'full_worker_run_admitted', 'cpu_tests_admitted')
FP32_EPSILON = 2. ** -23
ADMISSION_SHA = '1283335df26de930722ce1df550edabc887d38b81985ad2e37833cc946f7d693'
OBJECTIVE_SHA = 'cebec415b35c6f6a9bddc5696598b4e1a34e3ebe13ff3f0ca3e536fb7e695bba'
DECLARATION_SHA = '1de57992ece67acce913e68227248625a7e7bed055c3450aa86f890098fc9555'
SOURCES_SHA = '325f5c1008d73b814ad8c72cafe49fde1780292916530a319a9f27a5618a7206'
SOURCE_FILES = ('shared_train.py', 'shared_head.py', 'shared_training_data.py',
                'features.py', 'metrics.py', 'scope.py', 'preference_head_retrain.py', 'head.py')
NEW_PARAMETERS = ('joint_down.weight', 'joint_down.bias', 'joint_mid.weight',
                  'joint_mid.bias', 'joint_out.weight')
CONTROL_COMMITS = {
    'feature-only': '6721adaf03b0598839c2d59c6059632912bc883a',
    'lm-assisted': '9728b98b205d4f9af8ecac4e28ee678ba5ab758b',
}
SOURCE_VARIABLES = {
    'feature-only': {'control': 'FEATURE_ONLY_TRAINING_SOURCE',
                     'candidate': 'FEATURE_ONLY_CANDIDATE_SOURCE'},
    'lm-assisted': {'control': 'LM_ASSISTED_TRAINING_SOURCE',
                    'candidate': 'LM_ASSISTED_CANDIDATE_SOURCE'},
}
GEOMETRY = {'feature-only': {'control': (1402144, 13), 'candidate': (2714400, 18)},
            'lm-assisted': {'control': (2872401, 15), 'candidate': (5233233, 20)}}
HELPER_PINS = {
    HERE / 'train.py': '918bd3ea59f362103009ae5e2a60f910670a7d955f8fd7f4ee7320c0506054e1',
    HERE / 'admit.py': 'f74f81ade3c5cdbe23cdd1322d8e58f9e51ea2067ebf3a1396d728c55ca71393',
    GENERAL / 'controls.py': 'f289dff9876b47584c245b4fd4cf5ab7dd393992ecd929769c233f9eb5354eb8',
    GENERAL / 'folds.py': '5d6eeee1914ea236cac08fcd1a89efa90a9b0d0e61ae4ff41a76de6b08de6805',
}


class OptimizationInconclusive(RuntimeError):
    """Numerical inactivity or unhealthy updates are not a capacity verdict."""

    def __init__(self, message, receipt=None):
        super().__init__('optimization-inconclusive: ' + message)
        self.receipt = receipt


def require(value, message):
    if not value:
        raise ValueError(message)


def health_require(value, message, receipt=None):
    if not value:
        raise OptimizationInconclusive(message, receipt)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def bookend_files(pins):
    for path, expected in pins.items():
        require(is_sha(expected) and file_sha(path) == expected, 'Source/input changed: ' + str(path))


def load_pinned(name, path):
    path = Path(path).resolve()
    bookend_files({path: HELPER_PINS[path]})
    sys.dont_write_bytecode = True
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path, 'Unexpected helper import: ' + name)
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    bookend_files({path: HELPER_PINS[path]})
    return module


def controls():
    load_pinned('folds', GENERAL / 'folds.py')
    return load_pinned('controls', GENERAL / 'controls.py')


def require_admission(path, expected):
    require(Path(path).resolve() == HERE / 'input-admission-v2.json' and expected == ADMISSION_SHA,
            'Only immutable original v4 FIT0/1 input-admission-v2 membership is admitted')
    load_pinned('admit', HERE / 'admit.py')
    original = load_pinned('_capacity_original_membership_train', HERE / 'train.py')
    # This validator hashes old objective bytes as provenance; it never imports it.
    return original.require_admission(path, expected, OBJECTIVE_SHA)


def validate_sources(value, admission, declaration_sha):
    require(value.get('kind') == 'shared-capacity-source-manifest-v1'
            and value.get('passed') is True and value.get('admission_sha256') == ADMISSION_SHA
            and value.get('declaration_sha256') == declaration_sha
            and set(value.get('cases', {})) == set(LANES), 'New frozen capacity source manifest required')
    for lane in LANES:
        require(set(value['cases'][lane]) == set(ARMS), 'All four capacity cases required')
        parent = admission['sources'][lane]
        for arm in ARMS:
            case = value['cases'][lane][arm]
            expected_tree = Path(os.environ[SOURCE_VARIABLES[lane][arm]]).expanduser().resolve()
            require(isinstance(case.get('worktree'), str) and Path(case['worktree']).is_absolute()
                    and Path(case['worktree']).resolve() == expected_tree,
                    'Exact absolute capacity worktree required')
            commit = case.get('commit')
            require(isinstance(commit, str) and len(commit) == 40
                    and set(commit) <= set('0123456789abcdef'), 'Full source commit required')
            require(set(case.get('source8', {})) == set(SOURCE_FILES)
                    and all(is_sha(pin) for pin in case['source8'].values()), 'Complete Source8 required')
            parameters, tensors = GEOMETRY[lane][arm]
            require(type(case.get('parameters')) is int and case['parameters'] == parameters
                    and type(case.get('state_tensors')) is int and case['state_tensors'] == tensors,
                    'Full capacity geometry required')
            profile = dict(parent['profile'])
            if arm == 'candidate':
                profile['joint_hidden_learning_rate_multiplier'] = 10.
                profile['joint_readout_learning_rate_multiplier' if lane == 'feature-only'
                        else 'joint_out_learning_rate_multiplier'] = 100.
                require(commit != CONTROL_COMMITS[lane], 'Candidate cannot relabel parent source')
                require(all(case['source8'][name] == parent['source8'][name]
                            for name in SOURCE_FILES if name not in ('shared_train.py', 'shared_head.py')),
                        'Candidate changed a non-head/non-trainer Source8 file')
            else:
                require(commit == parent['source_commit'] == CONTROL_COMMITS[lane]
                        and case['source8'] == parent['source8'], 'Exact original F42/A43 control required')
            require(canonical(case.get('profile')) == canonical(profile), 'Original lane losses/profile changed')
            require(isinstance(case.get('optimization_protocol'), dict), 'Frozen optimization protocol required')
    return value


def source_bookend(sources):
    for lane in LANES:
        for arm in ARMS:
            case = sources['cases'][lane][arm]
            tree = Path(case['worktree'])
            bookend_files({tree / 'fuzev_research' / name: pin for name, pin in case['source8'].items()})


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def proof_group_rates(lane, arm, epoch):
    return {name: (1e-5 if epoch == 4 else 3e-6) * multiplier
            for name, multiplier in group_multipliers(lane, arm).items()}


def validate_proof(value, sources, bindings):
    require(set(bindings) == {'sources_sha256', 'declaration_sha256', 'worker_sha256', 'operator_sha256'}
            and all(is_sha(pin) for pin in bindings.values()), 'Four exact proof source bindings required')
    require(value.get('kind') == 'shared-capacity-cuda-admission-v1' and value.get('passed') is True
            and value.get('cuda') is True and value.get('admission_sha256') == ADMISSION_SHA
            and all(value.get(key) == pin for key, pin in bindings.items())
            and value.get('cpu_threads') == 4 and value.get('interop_threads') == 1
            and value.get('tf32') is False and value.get('float32_matmul_precision') == 'highest'
            and canonical(value.get('tolerances')) == canonical(TOLERANCES)
            and value.get('constant_clip_probe_sha256') == CONSTANT_SHA
            and value.get('constant_clip_operator_sha256') == CONSTANT_OPERATOR_SHA
            and all(value.get(key) is False for key in EXCLUDED_PROOF_CLAIMS)
            and all(value.get(key) is True for key in ('worker_train_epoch_admitted', 'synthetic_only',
                                                      'source_bookends_passed', 'constant_clip_probe_passed'))
            and value.get('max_retained_step_history') == 1
            and set(value.get('cases', {})) == set(LANES), 'New random-only source-bound capacity CUDA proof required')
    pins = value.get('input_sha256_before_and_after', {})
    evidence = {HERE / 'capacity_train.py': bindings['worker_sha256'],
                HERE / 'capacity_prove_cuda.py': bindings['operator_sha256'],
                HERE / 'capacity-constant-clip-cuda-v1.json': CONSTANT_SHA,
                HERE / 'capacity_clip_probe.py': CONSTANT_OPERATOR_SHA, HERE / 'prove_cuda.py': PROOF_REUSE_SHA}
    require(isinstance(pins, dict) and all(is_sha(pin) for pin in pins.values())
            and all(pins.get(str(path)) == pin for path, pin in evidence.items()), 'Missing worker/operator/evidence bookends')
    for field in ('sources', 'declaration'):
        require(isinstance(value.get(field + '_path'), str)
                and pins.get(value[field + '_path']) == bindings[field + '_sha256'], 'Missing artifact path bookend')
    flags = ('independent_logits', 'independent_primitive_loss', 'independent_js_gradients', 'independent_fp64_clip',
             'initial_parent_tensors_exact', 'initial_parent_logits_exact', 'moments_preserved_at_lr_transition',
             'frozen_inputs_unchanged', 'baseweights_frozen', 'global_clipping_exercised')
    for lane in LANES:
        require(set(value['cases'][lane]) == set(ARMS), 'CUDA proof must cover all four cases')
        for arm in ARMS:
            admitted = value['cases'][lane][arm]
            case = sources['cases'][lane][arm]
            require(admitted.get('passed') is True and admitted.get('status') == 'passed'
                    and admitted.get('commit') == admitted.get('source_commit') == case['commit']
                    and admitted.get('source8') == case['source8']
                    and canonical(admitted.get('profile')) == canonical(case['profile'])
                    and admitted.get('parameters') == case['parameters']
                    and admitted.get('state_tensors') == case['state_tensors'] and admitted.get('head_count') == 1
                    and admitted.get('actual_updates') == admitted.get('optimizer_updates') == 6
                    and admitted.get('epochs_replayed') == [4, 5]
                    and admitted.get('batch_sizes_per_epoch') == [256, 256, 138]
                    and admitted.get('synthetic_rows') == 650 and admitted.get('training_tracks') == 15
                    and admitted.get('synthetic_pairs') == 50 and admitted.get('pad_width') == 16
                    and admitted.get('fused_groups') == (6 if arm == 'candidate' else 4)
                    and admitted.get('reference_fused') is False and admitted.get('reference_foreach') is False
                    and all(admitted.get(key) is True for key in flags)
                    and admitted.get('joint_clipping_exercised') is (arm == 'candidate')
                    and admitted.get('active_joint_tensors_after_step1') == (sorted(NEW_PARAMETERS) if arm == 'candidate' else [])
                    and admitted.get('max_retained_step_history') == 1, 'Incomplete actual four-case CUDA numerical admission')
            require(all(pins.get(str(Path(case['worktree']) / 'fuzev_research' / name)) == pin
                        for name, pin in case['source8'].items()), 'Missing proof Source8 bookend')
            steps, epochs = admitted.get('steps'), admitted.get('epochs')
            require(isinstance(steps, list) and len(steps) == 6 and isinstance(epochs, list) and len(epochs) == 2,
                    'Six actual updates and two epoch receipts required')
            for index, step in enumerate(steps):
                epoch = 4 + index // 3
                require(step.get('step') == index + 1 and step.get('epoch') == epoch
                        and step.get('batch_size') == (256, 256, 138)[index % 3]
                        and type(step.get('pairs')) is int and 0 < step['pairs'] <= 50
                        and step.get('group_learning_rates') == proof_group_rates(lane, arm, epoch)
                        and all(step.get(key) is True for key in ('passed', 'all_finite', 'actual_parameters_changed'))
                        and finite_number(step.get('raw_global_norm_fp64')) and step['raw_global_norm_fp64'] > 0
                        and finite_number(step.get('clipped_global_norm_fp64'))
                        and 0 < step['clipped_global_norm_fp64'] <= 1 + 3e-6,
                        'Wrong actual update/rate/finite-gradient evidence')
                clip = step.get('reference_clip_norms_fp64', {})
                require(set(clip) == ({'joint', 'global'} if arm == 'candidate' else {'global'})
                        and all(finite_number(n) and n >= 0 for n in clip.values()), 'Missing independent clip norms')
                require((finite_number(step.get('raw_joint_norm_fp64')) and step['raw_joint_norm_fp64'] > 0)
                        if arm == 'candidate' else step.get('raw_joint_norm_fp64') is None, 'Wrong joint gradient domain')
            probe = admitted.get('isolated_joint_clip_probe', {}) or {}
            isolated = (probe.get('kind') == 'isolated-full-geometry-joint-clip-v1' and probe.get('passed') is True
                and probe.get('same_state_independent_fp64_reference') is True
                and finite_number(probe.get('raw_joint_norm_fp64')) and probe['raw_joint_norm_fp64'] > 1
                and finite_number(probe.get('post_joint_global_norm_fp64')) and probe['post_joint_global_norm_fp64'] > 1
                and finite_number(probe.get('clipped_global_norm_fp64')) and 0 < probe['clipped_global_norm_fp64'] <= 1 + 3e-6)
            require(any(s['raw_global_norm_fp64'] > 1 for s in steps)
                    and (arm == 'control' or any(s['raw_joint_norm_fp64'] > 1 for s in steps) or isolated), 'Clipping not exercised')
            for index, epoch in enumerate(epochs):
                require(epoch.get('epoch') == index + 4 and epoch.get('fit_records') == 650
                        and epoch.get('fit_pairs') == 50 and epoch.get('optimizer_steps_this_epoch') == 3
                        and finite_number(epoch.get('fit_loss'))
                        and sum(s['pairs'] for s in steps[index * 3:index * 3 + 3]) == 50,
                        'Incomplete exact-once epoch loss/JS evidence')
            diagnostics = admitted.get('diagnostics', {})
            require({'logits', 'js', 'loss', 'epoch_loss', 'raw', 'clipped', 'parameters', 'moments'} <= set(diagnostics)
                    and all(finite_number(n) and n >= 0 for n in diagnostics.values()), 'Missing strict comparisons')
    return value


def require_artifacts(args):
    require(args.declaration_sha256 == DECLARATION_SHA and args.sources_sha256 == SOURCES_SHA,
            'Require MAIN frozen new capacity declaration and source manifest')
    pins = {Path(__file__).resolve(): args.worker_sha256, args.admission: args.admission_sha256,
            args.declaration: args.declaration_sha256, args.sources: args.sources_sha256,
            args.proof: args.proof_sha256, **HELPER_PINS}
    require(len({Path(p).resolve() for p in
                 (args.admission, args.declaration, args.sources, args.proof, Path(__file__))}) == 5,
            'Admission, declaration, sources, proof and worker must be distinct artifacts')
    bookend_files(pins)
    admission = require_admission(args.admission, args.admission_sha256)
    sources = validate_sources(read_json(args.sources), admission, args.declaration_sha256)
    proof = read_json(args.proof)
    bindings = {key: getattr(args, key) for key in ('worker_sha256', 'sources_sha256', 'declaration_sha256')}
    bindings['operator_sha256'] = proof.get('operator_sha256')
    validate_proof(proof, sources, bindings)
    require(proof['sources_path'] == str(args.sources.resolve())
            and proof['declaration_path'] == str(args.declaration.resolve()), 'Proof must bind actual CLI artifact paths')
    # Do not import the proof operator: it imports this worker for replay.
    bookend_files(proof['input_sha256_before_and_after'])
    bookend_files({HERE / 'capacity_prove_cuda.py': proof['operator_sha256']})
    source_bookend(sources)
    return admission, sources, proof


def row_track(row):
    return '/'.join((row['source'], row['primitive'], row['modality'],
                     row.get('attribute') or row.get('adaptation') or 'main'))


def admitted_rows(features, admission):
    ids = admission['training_ids']
    allowed = set(ids)
    require(len(allowed) == len(ids) == 121953, 'Unique complete admitted FIT0/1 population required')
    rows = [row for row in features.records('fit') if row['id'] in allowed]
    require(len(rows) == len(allowed) and {r['id'] for r in rows} == allowed
            and all(row['role'] == 'fit' for row in rows)
            and digest(sorted((r['id'], r['group_'], row_track(r)) for r in rows))
                == admission['training_membership_sha256'], 'Frozen eligible FIT membership changed')
    return rows


def group_multipliers(lane, arm):
    require(lane in LANES and arm in ARMS, 'Explicit capacity lane/arm required')
    result = {'scorer' if lane == 'feature-only' else 'scorer_and_gate': 1.,
              'query_adapter': 10., 'channel_gate': 10., 'key_adapter': 10.}
    if arm == 'candidate':
        result.update({'joint_hidden': 10., 'joint_readout': 100.} if lane == 'feature-only'
                      else {'j_h': 10., 'j_out': 100.})
    return result


def set_rates(training, bundle, optimizer, epoch, lane, arm):
    require(type(epoch) is int and 1 <= epoch <= EPOCHS, 'Exactly eight one-based fixed epochs required')
    if lane == 'feature-only':
        training.set_epoch_learning_rate(optimizer, epoch, bundle)
    else:
        training.set_optimizer_learning_rates(optimizer, bundle, epoch)
    base_rate = 1e-5 if epoch <= 4 else 3e-6
    multipliers = group_multipliers(lane, arm)
    named = dict(bundle.named_parameters())
    groups = optimizer.param_groups
    require(len(groups) == len(multipliers) and {g.get('name') for g in groups} == set(multipliers),
            'Require original four or candidate six optimizer groups')
    owned = []
    for group in groups:
        names = list(training.OPTIMIZER_GROUP_PARAMETERS[group['name']])
        require(group.get('parameter_names') == names
                and group.get('lr_multiplier') == multipliers[group['name']]
                and group['lr'] == base_rate * multipliers[group['name']]
                and [id(p) for p in group['params']] == [id(named[name]) for name in names]
                and group.get('fused') is True and group.get('foreach') is False
                and group.get('weight_decay') == .01
                and group.get('betas') == (.9, .999) and group.get('eps') == 1e-8,
                'Actual rates, parameters or native fused AdamW contract changed')
        owned.extend(id(p) for p in group['params'])
    require(len(owned) == len(set(owned)) == len(named) and set(owned) == {id(p) for p in named.values()},
            'Optimizer groups must be disjoint and exhaustive')
    return {group['name']: group['lr'] for group in groups}


def norm_summary(norms):
    parent = math.hypot(*(norm for name, norm in norms.items() if name not in NEW_PARAMETERS))
    joint = math.hypot(*(norms.get(name, 0.) for name in NEW_PARAMETERS))
    combined = math.hypot(parent, joint)
    return {'per_tensor': norms, 'parent_combined_l2': parent, 'joint_combined_l2': joint,
            'all_combined_l2': combined, 'joint_l2_fraction': joint / combined if combined else 0.,
            'joint_squared_norm_share': (joint / combined) ** 2 if combined else 0.}


def gradient_snapshot(bundle):
    import torch
    named = dict(bundle.named_parameters())
    health_require(all(p.grad is not None and p.dtype == torch.float32 and p.grad.dtype == torch.float32
                       and bool(torch.isfinite(p.grad).all()) for p in named.values()),
                   'Missing/nonfinite or non-FP32 data gradient')
    # Observation only. Clipping below always receives the original FP32 gradients.
    norms = torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in named.values()])
    health_require(bool(torch.isfinite(norms).all()), 'Nonfinite FP32 gradient norm')
    return norm_summary(dict(zip(named, norms.cpu().tolist())))


def clip_and_measure(training, bundle, lane, arm):
    import torch
    before = gradient_snapshot(bundle)
    try:
        if arm == 'candidate':
            receipt = training.clip_gradients(bundle)
            if lane == 'feature-only':
                joint_norm, global_norm = receipt
            else:
                joint_norm = receipt['joint_norm_before_cap']
                global_norm = receipt['global_norm_after_joint_cap_before_global_cap']
            joint_norm, global_norm = float(joint_norm), float(global_norm)
        else:
            joint_norm = None
            global_norm = float(torch.nn.utils.clip_grad_norm_(
                bundle.parameters(), training.PROFILE['clip_grad_norm'], error_if_nonfinite=True))
    except (ValueError, RuntimeError) as exc:
        raise OptimizationInconclusive('Actual source clipping rejected: ' + str(exc)) from exc
    health_require(math.isfinite(global_norm) and (joint_norm is None or math.isfinite(joint_norm)),
                   'Actual source clip returned a nonfinite norm')
    after = gradient_snapshot(bundle)
    return {'before': before, 'after': after, 'joint_norm_before_cap': joint_norm,
            'global_norm_after_joint_cap_before_global_cap': global_norm,
            'joint_cap_factor': min(1., 1. / (joint_norm + 1e-6)) if joint_norm is not None else None,
            'global_cap_factor': min(1., 1. / (global_norm + 1e-6))}


def record_new_gradients(first_seen, norms, step):
    if step > 1:
        for name in NEW_PARAMETERS:
            value = norms.get(name, 0.)
            if math.isfinite(value) and value > 0.:
                first_seen.setdefault(name, step)


def require_gradient_health(first_seen):
    receipt = {'kind': 'shared-capacity-epoch1-data-gradient-health-v1',
               'after_update1_only': True, 'first_nonzero_finite_step': dict(first_seen),
               'passed': set(first_seen) == set(NEW_PARAMETERS)}
    health_require(receipt['passed'] and all(type(step) is int and step > 1 for step in first_seen.values()),
                   'Epoch1 requires finite nonzero data gradients for every new tensor after update1', receipt)
    return receipt


def train_epoch(training, bundle, cache, base, pairs, optimizer, epoch, lane, arm):
    """Actual proof-replayable FIT update loop, not the legacy DEV-selected runner."""
    import torch
    control = controls()
    control.check_updates(cache, base, pairs)
    profile = training.PROFILE
    require(profile['seed'] == SEED and profile['batch_size'] == BATCH_SIZE
            and profile['preference_consistency_beta'] == .05 and profile['clip_grad_norm'] == 1.
            and pairs, 'Original seed, batch, JS and norm1 bound required')
    if lane == 'feature-only':
        require(profile.get('gold_target_smoothing') == .05 and profile.get('distillation_lambda') == 0.,
                'Retain F42 FIT smoothing and zero distillation')
    else:
        require(lane == 'lm-assisted' and 'gold_target_smoothing' not in profile
                and profile.get('teacher_distribution_regularization') == .03
                and profile.get('correction_regularization') == .001
                and profile.get('ordinal_rps_coefficient') == .15, 'Retain all original A43 losses')
    rates = set_rates(training, bundle, optimizer, epoch, lane, arm)
    ordered = list(cache.rows)
    random.Random(SEED + epoch - 1).shuffle(ordered)
    visits, pair_visits, loss_sum = Counter(), Counter(), 0.
    clipping, first_seen = [], {}
    bundle.train()
    for offset in range(0, len(ordered), BATCH_SIZE):
        rows = ordered[offset:offset + BATCH_SIZE]
        try:
            batch = cache.batch(rows)
            logits = training.shared_logits(bundle, batch)
            per_row = training.fitting_loss(logits, batch, [r['primitive'] for r in rows], role=cache.role)
            factors = torch.tensor([base[r['id']] for r in rows], dtype=torch.float32, device=logits.device)
            loss = (per_row * factors).sum() / BATCH_SIZE
            js_sum, visited = training.batch_preference_consistency(bundle, cache, rows, pairs)
            loss = loss + .05 * len(ordered) / (len(pairs) * BATCH_SIZE) * js_sum
            health_require(bool(torch.isfinite(loss)), 'Nonfinite retained FIT/JS objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = clip_and_measure(training, bundle, lane, arm)
            optimizer.step()
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, OptimizationInconclusive):
                raise
            raise OptimizationInconclusive('Actual FIT backward/update rejected: ' + str(exc)) from exc
        step = offset // BATCH_SIZE + 1
        clip['step_in_epoch'] = step
        clipping.append(clip)
        if epoch == 1 and arm == 'candidate':
            record_new_gradients(first_seen, clip['before']['per_tensor'], step)
        visits.update(r['id'] for r in rows)
        pair_visits.update(visited)
        loss_sum += float(loss.detach()) * BATCH_SIZE
    require(set(visits) == set(base) and all(n == 1 for n in visits.values())
            and set(pair_visits) == set(pairs) and all(n == 1 for n in pair_visits.values()),
            'Original exact-once FIT/JS ownership failed')
    result = {'epoch': epoch, 'fit_records': len(visits), 'fit_pairs': len(pair_visits),
              'fit_loss': loss_sum / len(ordered), 'group_learning_rates': rates,
              'optimizer_steps_this_epoch': math.ceil(len(ordered) / BATCH_SIZE),
              'clipping': clipping, 'joint_cap_active_steps': sum(c['joint_cap_factor'] is not None
                  and c['joint_cap_factor'] < 1. for c in clipping),
              'global_cap_active_steps': sum(c['global_cap_factor'] < 1. for c in clipping)}
    if epoch == 1 and arm == 'candidate':
        result['new_tensor_data_gradient_health'] = require_gradient_health(first_seen)
    return result


def raw_scores(training, bundle, batch, arm):
    """Observe one actual forward; reuse its adapted terms, before LM mixing.

    Parent means the parent formula with this model's CURRENT parent parameters,
    not a second checkpoint or a claim of parent Jacobian/update identity.
    """
    import torch
    health_require(not torch.is_grad_enabled(), 'Raw FIT health must be no-gradient')
    captured, handles = {}, []

    def capture(key):
        def hook(module, inputs, output):
            captured[key] = output
        return hook

    try:
        if arm == 'candidate':
            def capture_joint(module, inputs):
                captured['joint_input'] = inputs[0]
            handles.append(bundle.joint_down.register_forward_pre_hook(capture_joint))
            handles.append(bundle.joint_out.register_forward_hook(capture('residual')))
        else:
            for key, module in (('query', bundle.scorer.query), ('keys', bundle.scorer.key),
                                ('query_adapter', bundle.adapter_up), ('key_adapter', bundle.key_adapter_up)):
                handles.append(module.register_forward_hook(capture(key)))
        training.shared_logits(bundle, batch)
        if arm == 'candidate':
            x = captured['joint_input']
            p = bundle.projection_size
            u, v = x[:, 0, :p].contiguous(), x[:, :, p:2 * p].contiguous()
            residual = captured['residual'].squeeze(-1)
        else:
            u = captured['query'] + captured['query_adapter']
            v = captured['keys'] + captured['key_adapter']
            residual = torch.zeros(v.shape[:2], dtype=v.dtype, device=v.device)
        channel = bundle.channel_a * u[:, None, :] + bundle.channel_b * v + bundle.channel_c
        scale = math.sqrt(bundle.projection_size)
        parent = (torch.einsum('bd,bkd->bk', u, v) / scale
                  + (u[:, None, :] * v * (2 * torch.sigmoid(channel) - 1)).sum(-1) / scale)
        new = parent + residual
        health_require(bool(torch.isfinite(parent).all()) and bool(torch.isfinite(new).all()),
                       'Nonfinite live parent/new raw scores')
        return parent, new
    except (ValueError, RuntimeError) as exc:
        if isinstance(exc, OptimizationInconclusive):
            raise
        raise OptimizationInconclusive('Actual raw FIT diagnostic rejected: ' + str(exc)) from exc
    finally:
        for handle in handles:
            handle.remove()


def centered_mean_squares(values, mask):
    counts = mask.sum(1)
    health_require(bool((counts >= 2).all()), 'FIT health requires at least two valid options')
    valid = values.masked_fill(~mask, 0)
    centered = (valid - valid.sum(1, keepdim=True) / counts[:, None]).masked_fill(~mask, 0)
    return centered.square().sum(1) / counts


def add_rms(totals, counts, rows, base, parent_ms, residual_ms):
    require(len(rows) == len(parent_ms) == len(residual_ms), 'Aligned per-row FIT health required')
    for row, parent, residual in zip(rows, parent_ms, residual_ms):
        primitive, weight = row['primitive'], base[row['id']]
        health_require(primitive in PRIMITIVES and math.isfinite(weight) and weight > 0
                       and all(math.isfinite(x) and x >= 0 for x in (parent, residual)),
                       'Nonfinite primitive FIT health aggregation')
        mass, parent_sum, residual_sum = totals.get(primitive, (0., 0., 0.))
        totals[primitive] = (mass + weight, parent_sum + weight * parent, residual_sum + weight * residual)
        counts[primitive] += 1


def finish_rms(totals, counts, arm):
    health_require(set(totals) == set(PRIMITIVES), 'Full FIT diagnostic must contain every primitive')
    cells = {}
    for primitive in PRIMITIVES:
        mass, parent_sum, residual_sum = totals[primitive]
        health_require(all(math.isfinite(x) and x >= 0 for x in (mass, parent_sum, residual_sum)) and mass > 0,
                       'Invalid complete-FIT RMS totals')
        parent, residual = math.sqrt(parent_sum / mass), math.sqrt(residual_sum / mass)
        threshold = 32 * FP32_EPSILON * max(1., parent)
        cells[primitive] = {'records': counts[primitive], 'base_weight_mass': mass,
                            'parent_centered_rms': parent, 'residual_centered_rms': residual,
                            'activity_floor': threshold, 'active': residual > threshold}
    receipt = {'kind': 'shared-capacity-fit-residual-health-v1', 'arm': arm,
               'aggregation': 'original base weights of per-row mean valid-option centered squares',
               'residual_definition': 'raw_new minus current-parent-form raw, before mask and LM trust mixing',
               'activity_gate_applicable': arm == 'candidate', 'primitives': cells,
               'passed': arm == 'control' or all(c['active'] for c in cells.values())}
    health_require(receipt['passed'], 'Complete-FIT centered residual inactive in at least one primitive', receipt)
    return receipt


def fit_residual_health(training, bundle, cache, base, lane, arm, epoch):
    """Matched full owned FIT no-grad passes at epoch1 and epoch8, never held."""
    import torch
    require(epoch in (1, EPOCHS) and lane in LANES and arm in ARMS, 'Fixed FIT diagnostic epochs required')
    controls().check_updates(cache, base, {})
    totals, counts, visits = {}, Counter(), Counter()
    previous_mode = bundle.training
    bundle.eval()
    try:
        with torch.no_grad():
            for offset in range(0, len(cache.rows), BATCH_SIZE):
                rows = cache.rows[offset:offset + BATCH_SIZE]
                batch = cache.batch(rows)
                parent, new = raw_scores(training, bundle, batch, arm)
                parent_ms = centered_mean_squares(parent, batch['mask']).cpu().tolist()
                # Subtract actual FP32 raw scores, not an idealized unrounded branch.
                residual_ms = centered_mean_squares(new - parent, batch['mask']).cpu().tolist()
                add_rms(totals, counts, rows, base, parent_ms, residual_ms)
                visits.update(row['id'] for row in rows)
        require(set(visits) == set(base) and all(n == 1 for n in visits.values()),
                'Full FIT no-grad diagnostic ownership failed')
        result = finish_rms(totals, counts, arm)
        return {**result, 'epoch': epoch, 'lane': lane, 'fit_records': len(visits),
                'no_grad': True, 'quality_evaluated': False}
    finally:
        bundle.train(previous_mode)


def import_training(case):
    tree = Path(case['worktree'])
    require(not any(name == 'fuzev' or name.startswith('fuzev.') for name in sys.modules),
            'Capacity worker needs a fresh process, not an already imported lane/head')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tree))
    training = importlib.import_module('fuzev.shared_train')
    require(Path(training.__file__).resolve() == tree / 'fuzev/shared_train.py'
            and training.source_identity() == case['source8']
            and canonical(training.PROFILE) == canonical(case['profile'])
            and canonical(training.optimization_protocol()) == canonical(case['optimization_protocol']),
            'Imported source/profile/optimization protocol differs from frozen capacity case')
    return training


def configure_threads(torch):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def configure_cuda(torch):
    require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def require_empty_wal():
    wal = FEATURES / 'index.sqlite3-wal'
    require(not wal.exists() or wal.stat().st_size == 0, 'Unbound feature-index WAL')


def run(args):
    admission, sources, _ = require_artifacts(args)
    case = sources['cases'][args.lane][args.arm]
    training = import_training(case)
    control = controls()
    import torch
    from safetensors.torch import save_file
    from fuzev_research.features import FeatureStore
    from fuzev_research.shared_training_data import SharedFeatureCache
    from fuzev_research.state import Store
    from fuzev_research.train import membership_identity

    configure_threads(torch)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    store = Store()
    task = f'training.generalization.capacity.{args.lane}.{args.arm}'
    store.update(task, 'in_progress', 'Loading admitted FIT0/1 CPU cache; exactly eight CUDA epochs, no held consumer.', .01)
    require_empty_wal()
    features = FeatureStore(FEATURES, read_json(FEATURES / 'identity.json'), read_only=True)
    transaction = False
    try:
        control.require_corpus_pin(training, features, args.lane)
        control.begin_corpus_snapshot(features, 185190)
        transaction = True
        version = features.db.execute('PRAGMA data_version').fetchone()[0]
        rows = admitted_rows(features, admission)
        membership = membership_identity(rows)
        base = training.fitting_weights(rows)
        original_admit = sys.modules['admit']
        require(base == original_admit.base_weights(rows, args.lane), 'Original track/group base weights changed')
        cache = SharedFeatureCache(features, rows, 'fit', 'cpu')
        if args.lane == 'lm-assisted':
            training.validate_fit_ordinal_cache(cache)
        require(cache.membership == membership and all(v.device.type == 'cpu' for v in cache.data.values()),
                'Owned complete CPU cache required before CUDA upload')
        pairs = training.fit_preference_pairs(features, rows)
        control.check_updates(cache, base, pairs)
        files = {str(FEATURES / name): pin for name, pin in cache.files.items()}
        bookend_files(files)
        configure_cuda(torch)
        cache.data = {key: value.to('cuda:0') for key, value in cache.data.items()}
        bundle = training.SharedDecisionHead(seed=SEED, key_gain=training.PROFILE['key_gain']).to('cuda:0')
        require(sum(p.numel() for p in bundle.parameters()) == case['parameters']
                and len(bundle.state_dict()) == case['state_tensors']
                and all(p.dtype == torch.float32 for p in bundle.parameters()), 'Fresh full FP32 geometry required')
        optimizer = training.make_optimizer(bundle)
        protocol = {'kind': 'shared-capacity-fresh-fit-v1', 'lane': args.lane, 'arm': args.arm,
                    **{key: getattr(args, key) for key in ('worker_sha256', 'admission_sha256',
                         'declaration_sha256', 'sources_sha256', 'proof_sha256')},
                    'source_commit': case['commit'], 'source8': case['source8'], 'profile': case['profile'],
                    'optimization_protocol': case['optimization_protocol'], 'parameters': case['parameters'],
                    'state_tensors': case['state_tensors'], 'training_membership': membership,
                    'training_records': len(rows), 'base_weights_sha256': digest(sorted(base.items())),
                    'feature_identity_sha256': admission['feature_identity_sha256'],
                    'index_sha256': admission['index_sha256'], 'cache': cache.provenance(),
                    'js_pairs': len(pairs), 'js_pairs_sha256': digest(sorted(
                        (key, p['parent']['id'], p['permutation']) for key, p in pairs.items())),
                    'preparation': admission['preparation'], 'fresh_initialization_seed': SEED,
                    'epochs': EPOCHS, 'checkpoint': 'final-only', 'fit_diagnostic_epochs': [1, EPOCHS],
                    'torch': torch.__version__, 'recorded_torch_version': TORCH_VERSION, 'threads': 4,
                    'cuda_version': torch.version.cuda, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                    'tf32': False, 'official_role': 'fit', 'official_roles_changed': False,
                    'risk_weighting_used': False, 'held_tensors_accessed': False, 'held_epoch_selection': False,
                    'development_tensors_accessed': False, 'llm_loaded': False, 'quality_evaluated': False,
                    'qualified': False, 'servable': False}
        write_json(output / 'protocol.json', protocol)
        epochs, health = [], []
        for epoch in range(1, EPOCHS + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            started = time.monotonic()
            result = train_epoch(training, bundle, cache, base, pairs, optimizer, epoch, args.lane, args.arm)
            torch.cuda.synchronize()
            result['fit_seconds'] = time.monotonic() - started
            result['fit_records_per_second'] = len(rows) / result['fit_seconds']
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
            if epoch in (1, EPOCHS):
                diagnostic = fit_residual_health(training, bundle, cache, base, args.lane, args.arm, epoch)
                health.append(diagnostic)
                write_json(output / f'fit-health-epoch-{epoch:03d}.json', diagnostic)
                result['fit_residual_health_sha256'] = file_sha(output / f'fit-health-epoch-{epoch:03d}.json')
            epochs.append(result)
            write_json(output / f'epoch-{epoch:03d}.json', result)
            store.update(task, 'in_progress', f'{epoch}/8 fixed CUDA epochs complete; FIT health only, no held scoring.',
                         .1 + .8 * epoch / EPOCHS)
            print(json.dumps({key: result[key] for key in ('epoch', 'fit_records', 'fit_loss', 'fit_seconds')}), flush=True)

        require(len(epochs) == EPOCHS and [h['epoch'] for h in health] == [1, EPOCHS],
                'No checkpoint before all eight fixed epochs and both complete FIT diagnostics')
        health_require(all(bool(torch.isfinite(p).all()) for p in bundle.parameters()), 'Nonfinite final parameters')
        require(cache.membership == membership == membership_identity(cache.rows)
                and membership_identity(admitted_rows(features, admission)) == membership
                and training.fitting_weights(cache.rows) == base
                and features.db.execute('PRAGMA data_version').fetchone()[0] == version,
                'FIT/cache membership or corpus changed during training')
        features.db.execute('COMMIT')
        transaction = False
        require_empty_wal()
        # data_version is connection/snapshot-local; byte identity plus ownership
        # is the durable post-transaction invariant (the live snapshot check stays).
        require(file_sha(FEATURES / 'index.sqlite3') == admission['index_sha256']
                and membership_identity(admitted_rows(features, admission)) == membership,
                'Post-snapshot FIT index bytes or population changed')
        bookend_files(files)
        require_artifacts(args)
        require(training.source_identity() == case['source8']
                and canonical(training.PROFILE) == canonical(case['profile'])
                and canonical(training.optimization_protocol()) == canonical(case['optimization_protocol']),
                'Runtime source/profile changed during fitting')
        save_file({key: value.detach().cpu().contiguous() for key, value in bundle.state_dict().items()},
                  str(output / 'head.safetensors'))
        checkpoint_sha = file_sha(output / 'head.safetensors')
        bookend_files(files)
        require_artifacts(args)
        write_json(output / 'completion.json', {'kind': 'shared-capacity-fit-completion-v1', 'completed': True,
                   'lane': args.lane, 'arm': args.arm, **{key: getattr(args, key) for key in
                       ('worker_sha256', 'admission_sha256', 'declaration_sha256', 'sources_sha256', 'proof_sha256')},
                   'protocol_sha256': file_sha(output / 'protocol.json'), 'checkpoint_sha256': checkpoint_sha,
                   'source_commit': case['commit'], 'source8': case['source8'], 'training_membership': membership,
                   'training_records': len(rows), 'epochs': epochs, 'fit_health': health,
                   'input_sha256_before_and_after': files, 'source_bookends_verified': True,
                   'held_tensors_accessed': False,
                   'development_tensors_accessed': False, 'llm_loaded': False, 'quality_evaluated': False,
                   'qualified': False, 'servable': False})
        store.update(task, 'completed', 'Eight fixed CUDA epochs and both full-FIT health gates finalized; no held/quality claim.', 1.)
    except OptimizationInconclusive as exc:
        write_json(output / 'optimization-inconclusive.json', {'kind': 'shared-capacity-optimization-inconclusive-v1',
                   'completed': False, 'lane': args.lane, 'arm': args.arm, 'reason': str(exc),
                   'health_receipt': exc.receipt, 'held_tensors_accessed': False, 'quality_evaluated': False,
                   'qualified': False, 'servable': False})
        store.update(task, 'failed', str(exc))
        raise
    finally:
        if transaction:
            features.db.execute('ROLLBACK')
        features.db.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane', choices=LANES, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--output', type=Path, required=True)
    for name in ('worker', 'admission', 'declaration', 'sources', 'proof'):
        if name != 'worker':
            p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    return p


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
