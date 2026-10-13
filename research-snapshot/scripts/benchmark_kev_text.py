"""One frozen, evaluation-only Kev breadth test read on native 4.05-bpw CUDA.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.

This is offline quality evaluation, not API E2E or head tuning. Reuse Kev's
unmodified scoring and whole-record bootstrap; retain newly extracted features.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/2026-10-10-kev-text-evaluation'
KEV = Path(os.environ.get('FUZEV_KEV_SOURCE', ROOT / 'kev')).expanduser().resolve()
SUITE = ROOT / 'data/evaluation/kev-breadth-v1'
OUTPUT = REPORT / 'native-results-v1'
PLAN = REPORT / 'frozen-plan-v1.json'
TEST_SHA = '91a64f0a9448dcaece421c051a4b3fd079c15b84ea6cb9cfa8155237d8d1fa18'
TASK = 'validation.kev-text'
HEADS = {
    'lm-assisted': ('FUZEV_LM_ASSISTED_SOURCE',
                    '9728b98b205d4f9af8ecac4e28ee678ba5ab758b',  # unpublished lab commit ID
                    'f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c'),
    'feature-only': ('FUZEV_FEATURE_ONLY_SOURCE',
                     '6721adaf03b0598839c2d59c6059632912bc883a',  # unpublished lab commit ID
                     '862cd81d9837de046dd7247c2a6b466f59eea313c5005b71e4891f44b4c0becb'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def verify(pins):
    return {str(path): sha(path) for path in pins}


def register():
    from fuzev_research.state import Store
    store = Store()
    store.initialize()
    with store.connect() as db:
        db.execute('INSERT OR IGNORE INTO tasks(id,parent,title,updated) VALUES(?,?,?,?)',
                   (TASK, 'validation', 'Frozen text generalization: Kev breadth-v1', time.time()))
    return store


def freeze():
    require(not PLAN.exists() and not OUTPUT.exists(), 'Do not overwrite a benchmark declaration/result')
    from fuzev_research.paths import source_revision
    kev_revision = source_revision(KEV, 'kev')
    cases = read(ROOT / 'reports/2026-10-09-shared-head-generalization/capacity-sources-v1.json')['cases']
    pins = {str(Path(__file__)): sha(__file__),
            str(HERE / 'kev_requests_adapter.py'): sha(HERE / 'kev_requests_adapter.py')}
    candidates = {}
    for lane, (source_env, commit, head_sha) in HEADS.items():
        tree = Path(os.environ[source_env]).expanduser().resolve()
        case = cases[lane]['control']
        pins.update({str(tree / 'fuzev_research' / file): pin for file, pin in case['source8'].items()})
        directory = ROOT / 'weights/retained-clean-20261010' / lane / 'fit'
        receipt = read(directory / 'completion.json')
        require(receipt['completed'] is True
                and receipt['training_records'] == 185178 and receipt['actual_optimizer_updates'] == 3620
                and receipt['calibration_performed'] is False, 'Different clean full-FIT checkpoint')
        pins[str(directory / 'head.safetensors')] = head_sha
        pins[str(directory / 'protocol.json')] = sha(directory / 'protocol.json')
        candidates[lane] = {'tree': str(tree), 'commit': source_revision(tree, 'fuzev-research'), 'recorded_commit': commit,
                            'weights': str(directory / 'head.safetensors'), 'recorded_weights_sha256': head_sha,
                            'weights_sha256': sha(directory / 'head.safetensors'), 'source8': case['source8'],
                            'profile': case['profile'], 'parameters': case['parameters']}
    for file in ('native.py', 'preparation.py', 'encoding.py', 'templates/chat_template.jinja'):
        path = Path(os.environ['FUZEV_RECIPES_SOURCE']).expanduser().resolve() / 'fuzev_research' / file
        pins[str(path)] = sha(path)
    for file in ('scripts/build_breadth_v1.py', 'scripts/breadth_report.py',
                 'kev/benchmark.py', 'kev/suite.py', 'evals/breadth-v1/manifest.json',
                 'runs/fam-breadth-test-report/report.json'):
        path = KEV / file
        pins[str(path)] = sha(path)
    verify(pins)
    plan = {'kind': 'frozen-kev-text-evaluation-v1', 'primary': 'lm-assisted',
            'diagnostic': 'feature-only', 'temperature': 1., 'candidate_selection_allowed': False,
            'training_allowed': False, 'calibration_allowed': False, 'suite': str(SUITE),
            'partition': 'test', 'test_sha256': TEST_SHA, 'records': 1990, 'questions': 3089,
            'bootstrap_samples': 2000, 'bootstrap_seed': 0,
            'baseline_index': 40.98328748655934, 'baseline_index_ci': [38.82186660975127, 43.90236579809118],
            'baseline_question_accuracy': 2157 / 3089,
            'gate': 'primary index CI lower > released Kev-9B v2 index CI upper; no paired claim',
            'candidates': candidates, 'input_sha256': pins,
            'observed_input_sha256': verify(pins), 'kev_revision': kev_revision,
            'recorded_kev_revision': 'fc4a17e194bca35a48c57211392ec49ea54a94ac',
            'original_feature_identity': str(ROOT / 'data/features/native-4.05-dimension-choice-v4/identity.json'),
            'raw_probabilities_not_calibrated': True,
            'qualified': False, 'servable': False}
    write(PLAN, plan)
    register().update(TASK, 'in_progress', 'Rebuilding official pinned eval-only panel; F/A heads frozen, no new training.', 0.)
    print(json.dumps({'plan_sha256': sha(PLAN), 'primary': plan['primary'], 'test_sha256': TEST_SHA}))


def load_heads(plan):
    import torch
    from safetensors.torch import load_file
    models = {}
    for lane, case in plan['candidates'].items():
        name = '_kev_' + lane.replace('-', '_')
        package = types.ModuleType(name)
        package.__path__ = [str(Path(case['tree']) / 'fuzev_research')]
        sys.modules[name] = package
        module = importlib.import_module(name + '.shared_head')
        model = module.SharedDecisionHead(seed=43, key_gain=case['profile']['key_gain']).to('cuda:0')
        model.load_state_dict(load_file(case['weights'], device='cuda:0'), strict=True)
        require(sum(p.numel() for p in model.parameters()) == case['parameters'], 'Wrong head parameter count')
        model.eval().requires_grad_(False)
        models[lane] = model
    return models


def run():
    import torch
    from safetensors.torch import save_file
    from fuzev_research.native import NativeBackend
    from kev_requests_adapter import adapt_record, prediction_metadata

    require(os.environ.get('FUZEV_ROOT') == str(ROOT), 'Explicit project root required')
    require(torch.cuda.is_available(), 'CUDA runtime required')
    plan = read(PLAN)
    verify(plan['input_sha256'])
    plan.update(observed_test_sha256=sha(SUITE / 'test.jsonl'), recorded_test_sha256=TEST_SHA,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                observed_input_sha256=verify(plan['input_sha256']))
    sys.path.insert(0, str(KEV))
    from kev.suite import load_split, read_manifest
    from scripts.breadth_report import score_system, bootstrap_index
    records = load_split(str(SUITE), 'test', allow_test=True)
    manifest = read_manifest(str(SUITE))
    require(len(records) == 1990 and sum(len(r['questions']) for r in records) == 3089,
            'Incomplete official test population')
    expected_manifest = read(KEV / 'evals/breadth-v1/manifest.json')
    require(manifest['datasets'] == expected_manifest['datasets'] and manifest['areas'] == expected_manifest['areas'],
            'Official dataset mappings or area weights changed')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT / 'plan.json', plan)
    store = register()
    store.update(TASK, 'in_progress', 'Loading original optimized native 4.05-bpw backbone; held test never used for tuning.', 0.)
    acquired = read(ROOT / 'data/manifests/acquisition.json')['models']['native-4.05']
    backend = None
    journal = None
    started = time.monotonic()
    rows = {'lm-assisted': [], 'feature-only': [], 'frozen-lm-label-control': []}
    try:
        torch.set_num_threads(10)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backend = NativeBackend(acquired['local_snapshot'], candidate_batch_size=8, batch_capacity=8)
        original = read(plan['original_feature_identity'])
        from fuzev_research.encoding import identity
        require(identity() == original['encoding'], 'Changed training input encoding')
        write(OUTPUT / 'environment.json', {'runtime': backend.readout_runtime, 'encoding': identity(),
              'gpu': torch.cuda.get_device_name(0), 'model': acquired['repo'], 'revision': acquired['revision'],
              'PLE': 'NVMe mmap', 'precision': 'original backbone; FP32 heads; float16 KV',
              'cuda_allocation_fraction': backend.memory_fraction,
              'recorded_readout_runtime': original['readout_runtime'],
              'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
              'plan_sha256': sha(PLAN)})
        models = load_heads(plan)
        journal = (OUTPUT / 'predictions.jsonl').open('x')
        feature_directory = ROOT / 'data/features/kev-breadth-v1-native-4.05'
        feature_directory.mkdir(parents=True, exist_ok=False)
        write(feature_directory / 'identity.json', {'role': 'external-evaluation-only', 'training_allowed': False,
              'calibration_allowed': False, 'test_sha256': TEST_SHA, 'plan_sha256': sha(PLAN),
              'native_original_identity': original, 'adapter_sha256': sha(HERE / 'kev_requests_adapter.py')})
        pending = {}
        count = 0
        with torch.inference_mode():
            for record in records:
                for item in adapt_record(record):
                    q, keys, baseline, execution = backend.extract(item, baseline=True)
                    require(q.shape == (2560,) and keys.shape == (len(item['candidates']), 2560)
                            and baseline.shape == (keys.shape[0],), 'Unexpected native readout geometry')
                    query_cuda, keys_cuda = q.float().cuda().unsqueeze(0), keys.float().cuda().unsqueeze(0)
                    baseline_cuda = baseline.float().cuda().unsqueeze(0)
                    predictions = {}
                    for lane, model in models.items():
                        kwargs = {'baseline': baseline_cuda} if model.requires_lm_scores else {}
                        logits = model(item['primitive'], query_cuda, keys_cuda, **kwargs)
                        probability = torch.softmax(logits, dim=-1)[0].cpu().tolist()
                        result_row = prediction_metadata(item, probability, input_order='fuzev')
                        rows[lane].append(result_row)
                        predictions[lane] = result_row
                    control = prediction_metadata(item, torch.softmax(baseline_cuda, dim=-1)[0].cpu().tolist(), input_order='fuzev')
                    rows['frozen-lm-label-control'].append(control)
                    predictions['frozen-lm-label-control'] = control
                    journal.write(json.dumps({'index': count, 'id': item['id'],
                                  'execution': execution, 'predictions': predictions}, allow_nan=False) + '\n')
                    journal.flush()
                    pending[f'{count}.query'] = q.float().contiguous()
                    pending[f'{count}.keys'] = keys.float().contiguous()
                    pending[f'{count}.baseline'] = baseline.float().contiguous()
                    count += 1
                    if count % 128 == 0 or count == 3089:
                        save_file(pending, str(feature_directory / f'shard-{count:05d}.safetensors'))
                        pending.clear()
                    if count % 25 == 0 or count == 3089:
                        elapsed = time.monotonic() - started
                        store.update(TASK, 'in_progress', f'CUDA text evaluation: {count}/3089 questions; no benchmark fitting.', count / 3089)
                        print(json.dumps({'completed_questions': count, 'questions': 3089,
                                          'elapsed_seconds': elapsed, 'questions_per_second': count / elapsed}), flush=True)
        journal.close()
        journal = None
        backend.close()
        backend = None
        verify(plan['input_sha256'])
        systems = {}
        for lane, values in rows.items():
            directory = OUTPUT / lane
            directory.mkdir()
            write(directory / 'rows.json', values)
            systems[lane] = score_system(records, values, manifest)
        uncertainty = bootstrap_index(records, rows, manifest, samples=2000, seed=0)
        lower = uncertainty['index']['lm-assisted'][0]
        passed = lower > plan['baseline_index_ci'][1]
        result = {'kind': 'frozen-kev-text-quality-result-v1', 'status': 'pass' if passed else 'stalled',
                  'primary': 'lm-assisted', 'systems': systems, 'uncertainty': uncertainty,
                  'test_sha256': TEST_SHA, 'plan_sha256': sha(PLAN), 'elapsed_seconds': time.monotonic() - started,
                  'comparison': {'Kev-9B-v2': {'index': plan['baseline_index'], 'index_ci': plan['baseline_index_ci'],
                                  'question_accuracy': plan['baseline_question_accuracy']},
                                 'paired_baseline_available': False, 'passed': passed, 'gate': plan['gate']},
                  'held_data_used_for_tuning': False, 'calibrated': False, 'qualified': False, 'servable': False}
        write(OUTPUT / 'result.json', result)
        store.record_result(TASK, {'validation_pass_rate': systems['lm-assisted']['overall']['question_accuracy_answered'],
                                 'validation_kind': 'Kev breadth-v1 test, offline CUDA 4.05-bpw',
                                 'quality_gate': 'pass' if passed else 'fail', 'index': systems['lm-assisted']['overall']['index'],
                                 'e2e_allowed': passed, 'qualified': False, 'servable': False})
        store.update(TASK, 'completed' if passed else 'blocked',
                     'Kev comparison passed; existing main/native gates still apply.' if passed else
                     'Stalled: frozen text head did not clearly exceed released Kev-9B; E2E not started.', 1.)
        print(json.dumps({'status': result['status'], 'primary_overall': systems['lm-assisted']['overall'],
                          'primary_index_ci': uncertainty['index']['lm-assisted'], 'result': str(OUTPUT / 'result.json')}), flush=True)
    except BaseException as error:
        store.update(TASK, 'blocked', 'Benchmark infrastructure failed; no quality verdict: ' + str(error))
        raise
    finally:
        if journal is not None:
            journal.close()
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
