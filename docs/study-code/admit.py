"""Freeze broad original-FIT input strata, without opening targets or tensors."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'reports/2026-10-08-postcandidate-assessment'
GENERAL = ROOT / 'reports/2026-10-06-generalization'
FEATURES = ROOT / 'data/features/native-4.05-dimension-choice-v4'
PINS = {
    HERE / 'predeclaration.md': '9081c09dd94fb53330e45bbb62b6117bb574d4625788d1e886650499611ba68a',
    OLD / 'separation-admission-result.json': 'aa365ba52113a11c15f72e473562eada7f01ad2ba26af6322f2a27e242a4c767',
    OLD / 'fit-exclusions-input-only.json': '80d05f4d8c7d1b7e9272004d7665558b9b3afcfcdb640463f2f1c514e3d3b0b6',
    OLD / 'training-metadata-admission.json': '9cbebeef74a683958c27e4d6943d1cae6decd9084b081243258c58f6868c644e',
    GENERAL / 'folds.py': '5d6eeee1914ea236cac08fcd1a89efa90a9b0d0e61ae4ff41a76de6b08de6805',
    GENERAL / 'folds.json': '1367120fb0906de72f40e5ed195275f37110682a09407ef8d87c8a37b8832a5d',
    GENERAL / 'controls.py': 'f289dff9876b47584c245b4fd4cf5ab7dd393992ecd929769c233f9eb5354eb8',
    GENERAL / 'coverage_train.py': 'ab59f68a5a31cec460e8d034fa8a7df84de2eb16ec9b272b3bc82aa6fe6ec962',
    FEATURES / 'identity.json': '498ae19fde58540c0961425bec00e5f2eead1f4fa8539631415750be99a9d69a',
    FEATURES / 'index.sqlite3': 'd45c19700fe637a34eb2e1627ff1cb154d5f0955a9ef0c00c198ff1027735123',
}
LANES = {
    'feature-only': ('FEATURE_ONLY_TRAINING_SOURCE', '6721adaf03b0598839c2d59c6059632912bc883a'),  # unpublished lab commit ID
    'lm-assisted': ('LM_ASSISTED_TRAINING_SOURCE', '9728b98b205d4f9af8ecac4e28ee678ba5ab758b'),  # unpublished lab commit ID
}


def require(value, message):
    if not value:
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


def load(name, path, expected):
    require(sha(path) == expected, 'Changed helper: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(sha(path) == expected, 'Helper changed during import')
    return module


def row_track(row):
    return '/'.join((row['source'], row['primitive'], row['modality'], row.get('attribute') or row.get('adaptation') or 'main'))


def base_weights(rows, lane):
    require(lane in LANES and rows and all(r['role'] == 'fit' for r in rows), 'Explicit lane and FIT rows required')
    counts, groups = Counter(), defaultdict(set)
    for row in rows:
        key = row_track(row)
        counts[key, row['group_']] += 1
        groups[key].add(row['group_'])
    require(len(groups) == 15, 'Every original workload required')
    masses = {key: 1 / 15 if lane == 'lm-assisted' or '/ordinal/' in key else
              1 / 6 if key.startswith('boolq/') else 1 / 18 for key in groups}
    return {row['id']: len(rows) * masses[row_track(row)] /
            (len(groups[row_track(row)]) * counts[row_track(row), row['group_']]) for row in rows}


def bookend(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, 'Source/input changed: ' + str(path))


def build(objective_sha, operator_sha):
    pins = dict(PINS)
    pins[HERE / 'objective.py'] = objective_sha
    pins[Path(__file__).resolve()] = operator_sha
    bookend(pins)
    separation = read(OLD / 'separation-admission-result.json')
    require(separation['kind'] == 'research-fit-separation-admission-v1'
        and separation['universal_protected_input_admitted'] is True
        and separation['universal_closure_complete'] is True
        and separation['safe_training_records'] == 121953 and separation['full_fit_records'] == 185190
        and separation['quarantined_fit_records'] == separation['quarantined_fit_fold01_records'] == 12
        and separation['remaining_training_cross_inner_fold_components'] == 0
        and separation['remaining_training_protected_overlap_components'] == 0
        and separation['unresolved_binding_blockers'] == []
        and all(separation[k] is False for k in ('annotations_decoded', 'outcomes_accessed', 'official_roles_changed')),
        'Complete unchanged input-only separation admission required')
    pins.update({Path(k): v for k, v in separation['input_sha256_before_and_after'].items()})
    metadata = read(OLD / 'training-metadata-admission.json')
    require(metadata['passed'] is True and metadata['original_full_readiness_reused'] is True
        and metadata['original_readiness_receipt_sha256'] == 'fe0d34f052347e075fc262fe7a1f708226237023d1c501e3664ad229058973fa',
        'Original completed full-readiness audit required')
    readiness = ROOT / 'weights/generalization-20261007/isolated-representation/feature-only/full-set/protocol.json'
    pins[readiness] = metadata['original_readiness_receipt_sha256']
    sources = {}
    for lane, (variable, commit) in LANES.items():
        tree = Path(os.environ[variable]).expanduser().resolve()
        source = metadata['retained_sources'][lane]
        require(source['source_commit'] == commit, 'Recipe metadata commit mismatch')
        for file, expected in source['source8'].items():
            pins[tree / 'fuzev_research' / file] = expected
        sources[lane] = {k: source[k] for k in ('source_commit', 'source8', 'profile')}
    bookend(pins)
    exclusions = read(OLD / 'fit-exclusions-input-only.json')
    members = {r[0]: r for r in exclusions['fit_input_membership']}
    require(len(members) == 185190 and len(exclusions['fit_input_membership']) == len(members), 'Unique complete FIT membership required')
    descriptors = {r['id']: r for r in exclusions['input_only_descriptors'] if r['role'] == 'fit'}
    require(set(descriptors) == set(members), 'Complete input-only FIT descriptors required')
    folds = read(GENERAL / 'folds.json')
    index = FEATURES / 'index.sqlite3'
    wal = index.with_name(index.name + '-wal')
    require(not wal.exists() or wal.stat().st_size == 0, 'Unbound feature WAL')
    db = sqlite3.connect(index.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        require(db.execute('SELECT 1 FROM failures LIMIT 1').fetchone() is None, 'Original extraction failures')
        sql = ("SELECT id,shard,row_,group_,role,primitive,modality,source,"
            "json_extract(metadata,'$.adaptation') AS adaptation,json_extract(metadata,'$.attribute') AS attribute,"
            "json_extract(metadata,'$.execution.branch_prefix_tokens') AS length,"
            "json_extract(metadata,'$.execution.query_token_sha256') AS query_sha256,"
            "json_array_length(json_extract(metadata,'$.candidates')) AS option_count "
            "FROM records WHERE role='fit' ORDER BY shard,row_")
        indexed = [dict(r) for r in db.execute(sql)]
    finally:
        db.close()
    require(len(indexed) == 185190 and {r['id'] for r in indexed} == set(members), 'Original FIT index membership changed')
    training, held, inputs, component_by_group = [], [], {}, {}
    certificates = exclusions['global_component_certificates']
    for row in indexed:
        member, descriptor = members[row['id']], descriptors[row['id']]
        require(member[1] == row['group_'] == descriptor['group']
            and member[4] == descriptor['fold'] == folds['group_fold'][row['group_']]
            and member[3] == descriptor['global_component_sha256']
            and descriptor['track'] == row_track(row)
            and all(descriptor[k] == row[k] for k in ('source', 'primitive', 'modality', 'length', 'query_sha256', 'option_count')),
            'Input descriptor, original fold or exact index binding changed')
        component = member[3]
        component_by_group[row['group_']] = component
        if member[6]:
            cert = certificates[component]
            require(member[4] in (0, 1) and member[5] is False and cert['roles'] == ['fit']
                and cert['original_fit_folds'] == [member[4]], 'Protected or cross-fold component entered fitting')
            training.append({k: v for k, v in row.items() if k not in ('length', 'query_sha256', 'option_count')})
            inputs[row['id']] = {k: descriptor[k] for k in ('id', 'group', 'source', 'track', 'length', 'option_count')}
        elif member[4] == 2:
            require(member[5] is False, 'Unexpected fold2 quarantine; stop, do not change held population')
            held.append(row['id'])
    require(len(training) == 121953 and len(held) == 63225, 'Declared broad training or blocked held count differs')
    require(len({r['id'] for r in training}) == len(training) and not set(held) & set(inputs), 'Duplicate or overlapping partitions')
    objective = load('_regime_objective_admission', HERE / 'objective.py', objective_sha)
    plans = {lane: objective.build_strata(training, inputs, component_by_group, base_weights(training, lane)) for lane in LANES}
    bookend(pins)
    return {'kind': 'input-regime-broad-fit-admission-v1', 'passed': True,
        'operator_sha256': operator_sha, 'objective_sha256': objective_sha,
        'declaration_sha256': PINS[HERE / 'predeclaration.md'], 'sources': sources,
        'original_full_readiness_reused': True, 'preparation': metadata['preparation'],
        'feature_identity_sha256': PINS[FEATURES / 'identity.json'], 'index_sha256': PINS[index],
        'training_records': len(training), 'held_records': len(held), 'official_role': 'fit',
        'training_ids': [r['id'] for r in training], 'held_ids': held,
        'training_membership_sha256': digest(sorted((r['id'], r['group_'], row_track(r)) for r in training)),
        'held_ids_sha256': digest(sorted(held)), 'plans': plans,
        'training_component_by_group': {r['group_']: component_by_group[r['group_']] for r in training},
        'input_sha256_before_and_after': {str(k): v for k, v in pins.items()},
        'targets_accessed': False, 'feature_tensors_opened': False, 'head_loaded': False,
        'llm_loaded': False, 'cuda_initialized': False, 'fitting_performed': False,
        'quality_evaluated': False, 'official_roles_changed': False, 'qualified': False, 'servable': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--operator-sha256', required=True)
    p.add_argument('--objective-sha256', required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'Admission output already exists')
    result = build(args.objective_sha256, args.operator_sha256)
    write(args.output, result)
    print(json.dumps({k: result[k] for k in ('kind', 'passed', 'training_records', 'held_records', 'training_membership_sha256')}), flush=True)


if __name__ == '__main__':
    main()
