"""Run the two admitted CUDA replay diagnostics sequentially, without held data.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
TREE = Path(os.environ['FUZEV_WORKLOAD_SOURCE']).expanduser().resolve()
HERE = ROOT / 'reports/2026-10-10-quality-mechanism'
WORKER = Path(__file__).resolve().with_name('probe_mature_direction.py')
TASK = 'training.generalization.mature-direction'
IMAGE = os.environ.get('FUZEV_RESEARCH_CUDA_IMAGE')
LANES = ('feature-only', 'lm-assisted')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--queue-sha256', required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--proof-sha256', required=True)
    parser.add_argument('--image', default=IMAGE, required=IMAGE is None, help="The lab's CUDA build image (Torch 2.13.0+cu130)")
    parser.add_argument('--container-name', help='Optional name format with {lane}')
    parser.add_argument('--cpuset-cpus', help='Optional Docker CPU set')
    parser.add_argument('--user', help='Optional Docker user')
    args = parser.parse_args()
    pins = {Path(__file__).resolve(): args.queue_sha256, WORKER: args.worker_sha256,
            HERE / 'predeclaration.md': args.declaration_sha256,
            HERE / 'measurement-proof-v2.json': args.proof_sha256}
    observed_pins = {str(path): sha(path) for path in pins}
    if (HERE / 'fit-output-v1').exists():
        raise ValueError('Never overwrite or implicitly resume a prior replay')
    os.environ['FUZEV_ROOT'] = str(ROOT)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fuzev_research.state import Store
    store = Store()
    with store.connect() as db:
        db.executemany('INSERT INTO tasks(id,parent,title,updated) VALUES(?,?,?,?)', [
            (TASK, 'training.generalization', 'CUDA mature optimizer direction diagnostic', time.time()),
            *[(TASK + '.' + lane, TASK, lane + ' unchanged CUDA replay', time.time()) for lane in LANES]])
    store.update(TASK, 'in_progress', 'Two unchanged CUDA control replays; no held outcomes.', .01)
    source_mounts = []
    for source in sorted({Path(__file__).resolve().parents[1], TREE}):
        if not source.is_relative_to(ROOT):
            source_mounts.extend(('-v', f'{source}:{source}:ro'))
    for lane in LANES:
        options = []
        for flag, value in (('--name', args.container_name.format(lane=lane) if args.container_name else None),
                            ('--cpuset-cpus', args.cpuset_cpus), ('--user', args.user)):
            if value is not None:
                options.extend((flag, value))
        command = ['docker', 'run', '--rm', *options,
            '--gpus', 'all', '--network', 'none', '--ipc', 'host',
            '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'NVIDIA_VISIBLE_DEVICES=all', '-e', 'OMP_NUM_THREADS=4',
            '-e', 'MKL_NUM_THREADS=4', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
            '-e', 'FUZEV_ROOT=' + str(ROOT), '-e', f'FUZEV_WORKLOAD_SOURCE={TREE}',
            '-v', f'{ROOT}:{ROOT}:ro', *source_mounts,
            '-v', f'{HERE}:{HERE}:rw', '-v', f'{ROOT / "weights"}:{ROOT / "weights"}:rw',
            '-v', f'{ROOT / "state"}:{ROOT / "state"}:rw', '--entrypoint', 'python3', args.image,
            '-B', str(WORKER), '--lane', lane, '--worker-sha256', args.worker_sha256,
            '--declaration-sha256', args.declaration_sha256, '--proof-sha256', args.proof_sha256]
        print('Starting unchanged CUDA replay: ' + lane, flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            store.update(TASK + '.' + lane, 'failed', 'Authoritative CUDA process exited unsuccessfully; no admission.')
            store.update(TASK, 'failed', 'Diagnostic process failed; original controls and quality gates untouched.')
            raise
    receipts = {lane: HERE / 'fit-output-v1' / lane / 'completion.json' for lane in LANES}
    values = {lane: json.loads(path.read_text()) for lane, path in receipts.items()}
    if not all(v['completed'] and v['exact_reference_endpoint'] and v['cuda']
               and v['actual_optimizer_updates'] == 2385 for v in values.values()):
        raise ValueError('Both unchanged CUDA recurrences must complete exactly')
    common = sorted(set.intersection(*(set(v['mechanism_gate']['qualifying_classification_tracks'])
                                       for v in values.values())))
    result = {'kind': 'mature-displacement-paired-screen-v1', 'completed': True,
        'common_qualifying_classification_tracks': common, 'mechanism_supported': len(common) >= 2,
        'receipt_sha256': {lane: sha(path) for lane, path in receipts.items()},
        'queue_sha256': args.queue_sha256, 'worker_sha256': args.worker_sha256,
        'recorded_input_sha256': {str(path): pin for path, pin in pins.items()},
        'observed_input_sha256_before': observed_pins, 'observed_input_sha256_after': {str(path): sha(path) for path in pins},
        'head_quality_pass': False, 'qualified': False, 'servable': False, 'held_data_accessed': False,
        'cpu_head_execution': False, 'main_quality_evaluated': False}
    with (HERE / 'paired-screen-v1.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    store.update(TASK, 'completed', 'CUDA exact replays and paired mechanism screen complete; not a quality pass.', 1.)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
