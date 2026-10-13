"""Input-only audit of the original exact-once FIT minibatch exposure.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = ROOT / 'reports/2026-10-10-quality-mechanism'
OLD = ROOT / 'reports/2026-10-09-shared-head-generalization'
INDEX = ROOT / 'data/features/native-4.05-dimension-choice-v4/index.sqlite3'
LEDGER = ROOT / 'reports/2026-10-08-postcandidate-assessment/fit-exclusions-input-only.json'
PINS = {
    INDEX: 'd45c19700fe637a34eb2e1627ff1cb154d5f0955a9ef0c00c198ff1027735123',
    LEDGER: '80d05f4d8c7d1b7e9272004d7665558b9b3afcfcdb640463f2f1c514e3d3b0b6',
    OLD / 'input-admission-v2.json': '1283335df26de930722ce1df550edabc887d38b81985ad2e37833cc946f7d693',
    OLD / 'admit.py': 'f74f81ade3c5cdbe23cdd1322d8e58f9e51ea2067ebf3a1396d728c55ca71393',
}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bookend(pins):
    return {str(path): sha(path) for path in pins}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--worker-sha256', required=True)
    args = parser.parse_args()
    pins = {**PINS, Path(__file__).resolve(): args.worker_sha256}
    bookend(pins)
    output = HERE / 'sampling-exposure-v1.json'
    require(not output.exists(), 'Never overwrite an exposure audit')
    admission = json.loads((OLD / 'input-admission-v2.json').read_text())
    ledger = json.loads(LEDGER.read_text())
    allowed = set(admission['training_ids'])
    descriptors = {d['id']: d for d in ledger['input_only_descriptors'] if d['fitting_eligible_fold01']}
    require(len(allowed) == len(descriptors) == 121953 and set(descriptors) == allowed
            and all(d['role'] == 'fit' and d['original_fold'] in (0, 1) and not d['quarantined_fit']
                    for d in descriptors.values()), 'Exact safe original FIT0/1 only')
    wal = Path(str(INDEX) + '-wal')
    require(not wal.exists() or wal.stat().st_size == 0, 'Immutable input index must have no pending WAL')
    db = sqlite3.connect(INDEX.as_uri() + '?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    try:
        # Do not select or decode targets, candidate text, images or predictions.
        fields = ("id,shard,row_,group_,role,primitive,modality,source,"
                  "json_extract(metadata,'$.adaptation') AS adaptation,"
                  "json_extract(metadata,'$.attribute') AS attribute")
        rows = [dict(row) for row in db.execute('SELECT ' + fields + " FROM records WHERE role='fit' ORDER BY shard,row_")
                if row['id'] in allowed]
    finally:
        db.close()
    require(len(rows) == len(allowed) and {r['id'] for r in rows} == allowed, 'Complete original cache order required')
    membership = hashlib.sha256()
    for row in rows:
        membership.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
        d = descriptors[row['id']]
        require(row['group_'] == d['group'] and row['source'] == d['source']
                and row['primitive'] == d['primitive'] and row['modality'] == d['modality'], 'Original metadata changed')
    expected_order = '194bcc3dd960816c0194c9611dba7283c3c97311db6d860c3ed93d3aa98e22c0'
    require(membership.hexdigest() == expected_order, 'Actual fresh control cache order required')
    spec = importlib.util.spec_from_file_location('_exposure_original_weights', OLD / 'admit.py')
    weights = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weights)
    track_rows = defaultdict(list)
    for row in rows:
        track = weights.row_track(row)
        require(track == descriptors[row['id']]['track'], 'Original track metadata changed')
        track_rows[track].append(row)
    require(len(track_rows) == 15, 'All15 workloads required')
    results = {}
    for lane in ('feature-only', 'lm-assisted'):
        base = weights.base_weights(rows, lane)
        priors = {track: math.fsum(base[r['id']] for r in owned) / len(rows)
                  for track, owned in track_rows.items()}
        exposure, tails, epoch_orders = defaultdict(list), defaultdict(list), {}
        for epoch in range(1, 6):
            ordered = list(rows)
            random.Random(43 + epoch - 1).shuffle(ordered)
            epoch_orders[str(epoch)] = hashlib.sha256(json.dumps([r['id'] for r in ordered],
                separators=(',', ':')).encode()).hexdigest()
            visits = Counter()
            for offset in range(0, len(ordered), 256):
                batch = ordered[offset:offset + 256]
                mass = defaultdict(float)
                for row in batch:
                    mass[weights.row_track(row)] += base[row['id']] / 256
                    visits[row['id']] += 1
                target = exposure if len(batch) == 256 else tails
                for track in track_rows:
                    target[track].append(mass[track] / priors[track])
            require(set(visits) == allowed and all(n == 1 for n in visits.values()), 'Exact-once shuffle changed')
        tracks = {}
        for track, values in exposure.items():
            require(len(values) == 2380 and len(tails[track]) == 5, '476 full plus one97-row tail per epoch required')
            mean = statistics.fmean(values)
            owned = track_rows[track]
            tracks[track] = {'records': len(owned), 'original_groups': len({r['group_'] for r in owned}),
                'prior_mass': priors[track], 'zero_full_batch_fraction': values.count(0.) / len(values),
                'relative_mass_mean': mean, 'relative_mass_cv': statistics.pstdev(values) / mean,
                'relative_mass_min': min(values), 'relative_mass_max': max(values),
                'relative_mass_tail_each_epoch': tails[track]}
        results[lane] = {'tracks': tracks, 'epoch_order_sha256': epoch_orders}
    bookend(pins)
    result = {'kind': 'input-only-exact-once-sampling-exposure-v1', 'completed': True,
        'training_records': 121953, 'batch_size': 256, 'seed': 43, 'epochs': 5,
        'ordered_cache_membership_sha256': expected_order, 'lanes': results,
        'input_sha256_before_and_after': {str(p): h for p, h in pins.items()},
        'observed_input_sha256': bookend(pins),
        'targets_accessed': False, 'features_accessed': False, 'model_executed': False,
        'new_schedule_applied': False, 'quality_evaluated': False, 'qualified': False, 'servable': False}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'completed': True, 'records': len(rows),
        'sparse_track_exposure': {t: v for t, v in results['lm-assisted']['tracks'].items()
                                if v['zero_full_batch_fraction'] >= .05}}, indent=2))


if __name__ == '__main__':
    main()
