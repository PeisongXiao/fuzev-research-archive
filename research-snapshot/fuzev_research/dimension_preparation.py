"""Prepare approved human-rating views without changing original ownership."""

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from . import dimension_augmentation as adapter
from .acquire import write_json
from .data import acquisition, download, json_lines, ordinal_records
from .features import file_sha256
from .paths import project_path
from .scope import validate_row


REVISION = '2026-10-05-v4-dimension-choice'
SOURCE = 'data/features/native-4.05-noul-preference-v3'
SOURCE_SHA = '2d1eb2f6129bf70d2d9f38992062dae22d774528d47436376f39b822c048276b'
FILES = {'text': 'dimension-choice-text.jsonl', 'image': 'dimension-choice-images.jsonl'}


def _bytes(value):
    # Original parent IDs predate canonical JSON and include original field order.
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode()


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _parents(root, database):
    parents, descriptors, stamps = {}, {}, {}
    ownership = dict(database.execute('SELECT id,role FROM groups_'))
    for modality, filename in (('text', 'text.jsonl'), ('image', 'images.jsonl')):
        path = root / 'data/prepared' / filename
        manifest_path = root / 'data/manifests' / ('text-splits.json' if modality == 'text' else 'image-splits.json')
        manifest = json.loads(manifest_path.read_text())
        registered = database.execute('SELECT sha256,completed,examined FROM preparations WHERE path=?',
                                      (str(path.resolve()),)).fetchone()
        if (not registered or registered[1] != 1 or registered[2] != manifest['records']
                or registered[0] != manifest['sha256'] or Path(manifest['path']).resolve() != path):
            raise ValueError('Original parent preparation is not complete and bound')
        stamps[path], stamps[manifest_path] = _stamp(path), _stamp(manifest_path)
        descriptors[modality] = {'path': str(path), 'sha256': manifest['sha256'],
                                'manifest_sha256': file_sha256(manifest_path), 'records': manifest['records']}
        digest, examined = hashlib.sha256(), 0
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line)
                examined += 1
                row = json.loads(line)
                if row['role'] not in adapter.ROLES:
                    continue
                eligible = (row['source'] == 'helpsteer2' and row['primitive'] == 'ordinal'
                            and row.get('attribute') == 'correctness' if modality == 'text' else
                            row['source'] == 'mm-rlhf' and row['primitive'] == 'choice'
                            and row.get('adaptation') == 'best_response')
                if not eligible:
                    continue
                indexed = database.execute('SELECT role,group_,source,primitive,modality FROM records WHERE id=?',
                                           (row['id'],)).fetchone()
                if (not indexed or tuple(indexed) != (row['role'], row['group'], row['source'], row['primitive'], modality)
                        or ownership.get(row['group']) != row['role'] or row['id'] in parents):
                    raise ValueError('Original parent membership or group ownership changed')
                parents[row['id']] = row
        if digest.hexdigest() != manifest['sha256'] or examined != manifest['records']:
            raise ValueError('Original preparation bytes changed')
    return parents, ownership, descriptors, stamps


def _view_stream(parents, ownership, provenance, raw_paths, exclusions):
    mm = {}
    for parent in parents.values():
        if parent['source'] == 'mm-rlhf':
            key = (parent.get('annotation_id'), parent['source_image_id'],
                   parent['question'].removeprefix(adapter.MM_PREFIX),
                   tuple(candidate['text'] for candidate in parent['candidates']))
            if key in mm:
                raise ValueError('Ambiguous admitted MM annotation locator')
            mm[key] = parent
    matched, repeated = {}, set()
    for raw in json_lines(raw_paths['mm-rlhf']):
        responses = raw.get('models_output')
        if not isinstance(responses, list) or any(not isinstance(value, str) for value in responses):
            continue
        key = (raw.get('id'), raw.get('image'), raw.get('question'), tuple(responses))
        if key not in mm:
            continue
        if key in matched:
            repeated.add(key)
            exclusions['mm_repeated_raw_annotation_records'] += 1
            continue
        matched[key] = raw
    if set(matched) != set(mm):
        raise ValueError('Admitted MM parents are missing from the pinned raw annotations')
    exclusions['mm_repeated_annotation_inputs_excluded'] = len(repeated)
    for key, raw in matched.items():
        # Different human annotations can share the exact inputs. Use no copy.
        if key in repeated:
            continue
        parent = mm[key]
        ratings = raw.get('faithfulness')
        if (not isinstance(ratings, list) or len(ratings) != len(raw.get('models_output', []))
                or any(type(value) not in (int, float) or not 1 <= value <= 5 for value in ratings)):
            exclusions['mm_missing_or_unsupported_faithfulness'] += 1
            continue
        response_ratings = {}
        conflict = False
        for response, rating in zip(raw['models_output'], ratings):
            if response in response_ratings and response_ratings[response] != rating:
                conflict = True
            response_ratings[response] = rating
        if conflict:
            exclusions['mm_conflicting_identical_response_ratings'] += 1
            continue
        if len(response_ratings) < 2:
            exclusions['mm_no_distinct_response_pair'] += 1
            continue
        yield from adapter.mm_faithfulness_views(parent, raw, provenance['mm-rlhf'], group_roles=ownership)
    grouped, matched, repeated = defaultdict(list), {}, set()
    for index, raw in enumerate(json_lines(raw_paths['helpsteer2'])):
        generated = [row for row in ordinal_records(raw, 'train') if row.get('attribute') == 'correctness']
        if not generated:
            exclusions['hs2_missing_or_unsupported_correctness'] += 1
            continue
        parent = parents.get(generated[0]['id'])
        if parent is None:
            continue
        if parent['id'] in matched:
            repeated.add(parent['id'])
            exclusions['hs2_repeated_raw_correctness_records'] += 1
            continue
        matched[parent['id']] = parent, raw, index
    expected = {key for key, parent in parents.items() if parent['source'] == 'helpsteer2'}
    if set(matched) != expected:
        raise ValueError('Admitted HS2 parents are missing from the original training annotations')
    exclusions['hs2_repeated_correctness_parent_inputs_excluded'] = len(repeated)
    for identifier, row in matched.items():
        if identifier not in repeated:
            grouped[row[0]['group']].append(row)
    for rows in grouped.values():
        response_ratings = {}
        conflict = False
        for _, raw, _ in rows:
            response, rating = raw['response'], raw['correctness']
            if response in response_ratings and response_ratings[response] != rating:
                conflict = True
            response_ratings[response] = rating
        if conflict:
            exclusions['hs2_conflicting_identical_response_ratings_groups'] += 1
            continue
        for a, b in combinations(rows, 2):
            if a[1]['response'] == b[1]['response']:
                exclusions['hs2_identical_response_pair'] += 1
                continue
            yield adapter.hs2_correctness_view([a[0], b[0]], [a[1], b[1]], provenance['helpsteer2'],
                                               row_indices=[a[2], b[2]], group_roles=ownership)


def prepare(root=None):
    root = Path(root).expanduser().resolve() if root is not None else project_path('')
    targets = {modality: (root / 'data/prepared' / filename,
                          root / 'data/manifests' / (Path(filename).stem + '-splits.json'))
               for modality, filename in FILES.items()}
    if any(path.exists() or path.is_symlink() for pair in targets.values() for path in pair):
        raise FileExistsError('Preserve previous dimension preparations; refuse overwrite')
    source = root / SOURCE
    source_sha = file_sha256(source / 'identity.json')
    database = sqlite3.connect((source / 'index.sqlite3').as_uri() + '?mode=ro', uri=True)
    database.execute('PRAGMA query_only=ON')
    try:
        database.execute('BEGIN')
        if (database.execute('SELECT count(*) FROM failures').fetchone()[0]
                or database.execute('SELECT group_ FROM records GROUP BY group_ HAVING count(DISTINCT role)>1 LIMIT 1').fetchone()):
            raise ValueError('Unresolved original failures or group leakage')
        parents, ownership, descriptors, stamps = _parents(root, database)
    finally:
        database.close()
    raw_paths = {'mm-rlhf': download('mm-rlhf', 'data.jsonl'),
                 'helpsteer2': download('helpsteer2', 'train.jsonl.gz')}
    provenance = {source: {'source': source, 'revision': acquisition()[source]['revision'],
                          'filename': path.name, 'sha256': file_sha256(path)}
                  for source, path in raw_paths.items()}
    identity = json.loads((source / 'identity.json').read_text())
    if any(identity['datasets'][name] != pin['revision'] for name, pin in provenance.items()):
        raise ValueError('Raw annotation revision differs from original native4 provenance')
    stamps.update({path: _stamp(path) for path in raw_paths.values()})
    exclusions = Counter()
    selected, selection = adapter.select_dimension_views(
        _view_stream(parents, ownership, provenance, raw_paths, exclusions), group_roles=ownership)
    manifests, published = {}, []
    try:
        with tempfile.TemporaryDirectory(prefix='.dimension-choice-', dir=root / 'data') as temporary:
            staged = []
            for modality, (destination, manifest_path) in targets.items():
                output = Path(temporary) / destination.name
                counts, labels, groups, digest, count = Counter(), Counter(), set(), hashlib.sha256(), 0
                with output.open('xb') as writer:
                    for supplied in selected:
                        if supplied['modality'] != modality:
                            continue
                        row = dict(supplied, dataset_scope_revision=REVISION,
                                   parent_preparation=descriptors[modality],
                                   parent_feature_identity_sha256=source_sha)
                        validate_row(row, destination.name)
                        encoded = _bytes(row) + b'\n'
                        writer.write(encoded)
                        digest.update(encoded)
                        counts[f"{row['source']}/choice/{row['role']}"] += 1
                        labels[f"{row['source']}/{row['role']}/" + ('tie' if row['target'] == [.5, .5]
                              else 'first' if row['target'] == [1., 0.] else 'second')] += 1
                        groups.add(row['group'])
                        count += 1
                manifest = {'records': count, 'counts': dict(counts), 'groups': len(groups),
                            'sha256': digest.hexdigest(), 'path': str(destination),
                            'dataset_scope_revision': REVISION, 'dataset_purpose': 'training',
                            'evaluation_scope': 'training', 'purpose': 'training',
                            'adapter_version': adapter.ADAPTER_VERSION, 'seed': adapter.SEED,
                            'budgets': adapter.PHASE1_BUDGETS, 'label_counts': dict(labels),
                            'source_provenance': provenance, 'parent_preparations': [descriptors[modality]],
                            'source_feature_identity_sha256': source_sha, 'recorded_source_feature_identity_sha256': SOURCE_SHA,
                            'selection': selection,
                            'exclusions': dict(exclusions), 'new_test_records': 0,
                            'external_examples_used': False, 'labels_are_absolute_truth': False}
                manifest_output = Path(temporary) / manifest_path.name
                write_json(manifest_output, manifest)
                manifests[destination.name] = manifest
                staged.extend(((output, destination), (manifest_output, manifest_path)))
            if any(_stamp(path) != stamp for path, stamp in stamps.items()):
                raise ValueError('Pinned annotation or parent preparation changed')
            with sqlite3.connect((source / 'index.sqlite3').as_uri() + '?mode=ro', uri=True) as check:
                check.execute('PRAGMA query_only=ON')
                check.execute('BEGIN')
                for view in selected:
                    for parent in view['parent_records']:
                        actual = check.execute('SELECT role,group_,source,primitive,modality FROM records WHERE id=?',
                                               (parent['id'],)).fetchone()
                        if (actual != (parent['role'], parent['group'], parent['source'], parent['primitive'], parent['modality'])
                                or check.execute('SELECT role FROM groups_ WHERE id=?',
                                                 (parent['group'],)).fetchone() != (parent['role'],)):
                            raise ValueError('Selected parent ownership changed during preparation')
            for origin, destination in staged:
                os.link(origin, destination)
                stat = destination.stat()
                published.append((destination, stat.st_dev, stat.st_ino))
    except BaseException:
        for path, device, inode in published:
            if path.is_file() and (path.stat().st_dev, path.stat().st_ino) == (device, inode):
                path.unlink()
        raise
    return manifests


if __name__ == '__main__':
    result = prepare()
    print(json.dumps({name: {key: manifest[key] for key in ('records', 'counts', 'sha256', 'label_counts', 'exclusions')}
                      for name, manifest in result.items()}, sort_keys=True), flush=True)
