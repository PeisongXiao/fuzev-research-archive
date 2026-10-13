"""Synthetic provenance/IO tests; no real dataset, LLM or CUDA access."""

import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from fuzev_research import data, dimension_augmentation as adapter, dimension_preparation as preparation
from fuzev_research.acquire import write_json
from fuzev_research.features import FeatureStore, file_sha256, verify_preparation
from fuzev_research.scope import DIMENSION_METADATA, contract, validate_row


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    revisions = {'helpsteer2': {'revision': '1' * 40}, 'mm-rlhf': {'revision': '2' * 40}}
    monkeypatch.setattr(data, 'acquisition', lambda: revisions)
    monkeypatch.setattr(preparation, 'acquisition', lambda: revisions)
    monkeypatch.setattr(adapter, 'PHASE1_BUDGETS', {role: {source: 1 for source in adapter.SOURCES}
                                                  for role in adapter.ROLES})
    for name in ('data/prepared', 'data/manifests', preparation.SOURCE, 'annotations'):
        (tmp_path / name).mkdir(parents=True)
    texts, images, hs_raw, mm_raw, ownership = [], [], [], [], {}
    for index, role in enumerate(adapter.ROLES):
        for response, score in (('Accurate response.', 4), ('Inaccurate response.', 1)):
            raw = {'prompt': f'Synthetic request {index}', 'response': response, 'correctness': score}
            parent = next(data.ordinal_records(raw, 'train'))
            parent['role'] = role
            texts.append(parent)
            hs_raw.append(raw)
            ownership[parent['group']] = role
        raw = {'id': index, 'image': f'synthetic-{index}.png', 'question': f'Synthetic image query {index}',
               'models_output': ['Faithful description.', 'Unfaithful description.'],
               'faithfulness': [5, 1], 'final_ranking': [2, 1]}
        parent = {'id': hashlib.sha256(f'mm-parent-{index}'.encode()).hexdigest(),
                  'source': 'mm-rlhf', 'revision': revisions['mm-rlhf']['revision'], 'group': f'visual:{index}',
                  'role': role, 'original_split': 'train', 'modality': 'image', 'context': [],
                  'question': adapter.MM_PREFIX + raw['question'], 'primitive': 'choice',
                  'candidates': [{'id': str(i), 'text': value} for i, value in enumerate(raw['models_output'])],
                  'target': [0., 1.], 'adaptation': 'best_response', 'annotation_id': index,
                  'image_path': f'/synthetic/{index}.png', 'image_sha256': str(index + 3) * 64,
                  'source_image_id': raw['image'], 'official_holdout_overlap': False,
                  'evaluation_scope': 'training'}
        parent['id'] = data.stable_id(parent['source'], parent['group'], parent['context'], parent['question'],
                                      json.dumps(parent['candidates']), parent['primitive'], parent['target'])
        images.append(parent)
        mm_raw.append(raw)
        ownership[parent['group']] = role
    raw_paths = {'helpsteer2': tmp_path / 'annotations/train.jsonl.gz',
                 'mm-rlhf': tmp_path / 'annotations/data.jsonl'}
    with gzip.open(raw_paths['helpsteer2'], 'wt') as handle:
        for row in hs_raw:
            handle.write(json.dumps(row) + '\n')
    raw_paths['mm-rlhf'].write_text(''.join(json.dumps(row) + '\n' for row in mm_raw))
    monkeypatch.setattr(preparation, 'download', lambda source, filename: raw_paths[source])
    source = tmp_path / preparation.SOURCE
    write_json(source / 'identity.json', {'synthetic_fixture': True,
                                        'datasets': {key: value['revision'] for key, value in revisions.items()}})
    monkeypatch.setattr(preparation, 'SOURCE_SHA', file_sha256(source / 'identity.json'))
    with sqlite3.connect(source / 'index.sqlite3') as database:
        database.executescript('''
            CREATE TABLE records(id TEXT,role TEXT,group_ TEXT,source TEXT,primitive TEXT,modality TEXT);
            CREATE TABLE groups_(id TEXT,role TEXT);
            CREATE TABLE failures(id TEXT,reason TEXT);
            CREATE TABLE preparations(path TEXT,sha256 TEXT,completed INT,examined INT);
        ''')
        database.executemany('INSERT INTO groups_ VALUES(?,?)', ownership.items())
        for modality, rows, filename, manifest_name in (
                ('text', texts, 'text.jsonl', 'text-splits.json'),
                ('image', images, 'images.jsonl', 'image-splits.json')):
            path = tmp_path / 'data/prepared' / filename
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            write_json(tmp_path / 'data/manifests' / manifest_name,
                       {'path': str(path), 'sha256': file_sha256(path), 'records': len(rows)})
            database.execute('INSERT INTO preparations VALUES(?,?,1,?)',
                             (str(path), file_sha256(path), len(rows)))
            database.executemany('INSERT INTO records VALUES(?,?,?,?,?,?)',
                                 [(row['id'], row['role'], row['group'], row['source'], row['primitive'], modality)
                                  for row in rows])
    return tmp_path


def test_real_bridge_preserves_ownership_and_refuses_overwrite(synthetic):
    root = synthetic
    before = {path: file_sha256(path) for path in (root / 'data/prepared').iterdir()}
    manifests = preparation.prepare(root)
    assert sum(item['records'] for item in manifests.values()) == 6
    assert all(item['new_test_records'] == 0 and item['external_examples_used'] is False
               for item in manifests.values())
    for filename in manifests:
        rows = [json.loads(line) for line in (root / 'data/prepared' / filename).read_text().splitlines()]
        assert {row['role'] for row in rows} == set(adapter.ROLES)
        for row in rows:
            validate_row(row, filename)
            assert row['target'] == ([1., 0.] if row['dimension_scores'][0] > row['dimension_scores'][1] else [0., 1.])
            assert all(parent['group'] == row['group'] and parent['role'] == row['role'] for parent in row['parent_records'])
    assert all(file_sha256(path) == sha for path, sha in before.items())
    with pytest.raises(FileExistsError, match='overwrite'):
        preparation.prepare(root)


def test_bad_original_group_never_produces_new_views(synthetic):
    with sqlite3.connect(synthetic / preparation.SOURCE / 'index.sqlite3') as database:
        database.execute("UPDATE groups_ SET role='test' WHERE id='visual:0'")
    with pytest.raises(ValueError, match='ownership'):
        preparation.prepare(synthetic)
    assert not list((synthetic / 'data/prepared').glob('dimension-choice*'))


def test_missing_authentic_visual_ratings_fail_quota_without_fallback(synthetic):
    path = synthetic / 'annotations/data.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['faithfulness'] = None
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(adapter.QuotaInfeasible):
        preparation.prepare(synthetic)
    assert not list((synthetic / 'data/prepared').glob('dimension-choice*'))


def test_repeated_mm_ids_for_other_inputs_do_not_match_admitted_parents(synthetic):
    path = synthetic / 'annotations/data.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append(dict(rows[0], image='not-an-admitted-image.png'))
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifests = preparation.prepare(synthetic)
    assert sum(item['records'] for item in manifests.values()) == 6
    assert all(item['exclusions']['mm_repeated_annotation_inputs_excluded'] == 0 for item in manifests.values())


@pytest.mark.parametrize('conflicting', [False, True])
def test_all_copies_of_repeated_mm_inputs_are_excluded_without_first_row_wins(synthetic, conflicting):
    path = synthetic / 'annotations/data.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append(dict(rows[0], faithfulness=[1, 5]) if conflicting else rows[0])
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(adapter.QuotaInfeasible):
        preparation.prepare(synthetic)
    assert not list((synthetic / 'data/prepared').glob('dimension-choice*'))


def test_identical_visual_response_text_cannot_receive_conflicting_new_targets(synthetic):
    parents_path = synthetic / 'data/prepared/images.jsonl'
    parents = [json.loads(line) for line in parents_path.read_text().splitlines()]
    raw_path = synthetic / 'annotations/data.jsonl'
    raws = [json.loads(line) for line in raw_path.read_text().splitlines()]
    raws[0]['models_output'][1] = raws[0]['models_output'][0]
    parents[0]['candidates'][1]['text'] = parents[0]['candidates'][0]['text']
    parents[0]['id'] = data.stable_id(parents[0]['source'], parents[0]['group'], parents[0]['context'],
                                     parents[0]['question'], json.dumps(parents[0]['candidates']),
                                     parents[0]['primitive'], parents[0]['target'])
    parents_path.write_text(''.join(json.dumps(row) + '\n' for row in parents))
    raw_path.write_text(''.join(json.dumps(row) + '\n' for row in raws))
    manifest_path = synthetic / 'data/manifests/image-splits.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256'] = file_sha256(parents_path)
    write_json(manifest_path, manifest)
    with sqlite3.connect(synthetic / preparation.SOURCE / 'index.sqlite3') as database:
        database.execute("UPDATE records SET id=? WHERE group_='visual:0'", (parents[0]['id'],))
        database.execute('UPDATE preparations SET sha256=? WHERE path=?',
                         (manifest['sha256'], str(parents_path)))
    with pytest.raises(adapter.QuotaInfeasible):
        preparation.prepare(synthetic)
    assert not list((synthetic / 'data/prepared').glob('dimension-choice*'))


def test_repeated_text_correctness_annotations_are_not_arbitrarily_selected(synthetic):
    path = synthetic / 'annotations/train.jsonl.gz'
    with gzip.open(path, 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    rows.append(dict(rows[0]))
    with gzip.open(path, 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    with pytest.raises(adapter.QuotaInfeasible):
        preparation.prepare(synthetic)
    assert not list((synthetic / 'data/prepared').glob('dimension-choice*'))


@pytest.mark.parametrize('mutation', ['dimension_scores', 'question', 'dataset_scope_revision'])
def test_new_view_context_and_ratings_survive_feature_metadata(synthetic, mutation):
    torch = pytest.importorskip('torch')
    manifests = preparation.prepare(synthetic)
    path = synthetic / 'data/prepared/dimension-choice-text.jsonl'
    row = json.loads(path.read_text().splitlines()[0])
    identity = {'dataset_scope': contract(augmentation=True, dimensions=True),
                'datasets': {'helpsteer2': row['revision'], 'mm-rlhf': '2' * 40}}
    store = FeatureStore(synthetic / 'new-features', identity)
    try:
        store.add(row, torch.ones(2560), torch.ones(2, 2560), torch.zeros(2))
        store.flush()
        metadata = json.loads(store.records(include_metadata=True)[0]['metadata'])
        assert 'context' in DIMENSION_METADATA
        for key in DIMENSION_METADATA:
            assert metadata.get(key) == row.get(key)
        validate_row(metadata)
        one = synthetic / 'dimension-choice-text.jsonl'
        one.write_text(json.dumps(row) + '\n')
        verify_preparation(store.db, one, {'sha256': file_sha256(one), 'records': 1},
                           'text', identity['datasets'], scoped=True)
        changed = dict(metadata)
        changed[mutation] = (list(reversed(metadata['dimension_scores'])) if mutation == 'dimension_scores'
                             else 'changed-and-not-the-pinned-preparation')
        store.db.execute('UPDATE records SET metadata=?', (json.dumps(changed),))
        store.db.commit()
        with pytest.raises(ValueError, match='matching features'):
            verify_preparation(store.db, one, {'sha256': file_sha256(one), 'records': 1},
                               'text', identity['datasets'], scoped=True)
    finally:
        store.close()


def test_v3_identity_cannot_accept_dimension_rows(synthetic):
    torch = pytest.importorskip('torch')
    preparation.prepare(synthetic)
    row = json.loads((synthetic / 'data/prepared/dimension-choice-images.jsonl').read_text().splitlines()[0])
    store = FeatureStore(synthetic / 'old-scope-fixture', {'dataset_scope': contract(augmentation=True)})
    try:
        with pytest.raises(ValueError, match='not enabled'):
            store.add(row, torch.ones(2560), torch.ones(2, 2560), torch.zeros(2))
    finally:
        store.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_extraction_route_requires_explicit_dimension_opt_in(synthetic, monkeypatch, enabled):
    pytest.importorskip('torch')
    from fastapi.testclient import TestClient
    from fuzev_research import server

    preparation.prepare(synthetic)
    write_json(synthetic / 'data/manifests/acquisition.json',
               {'models': {'native-4.05': {'local_snapshot': 'synthetic-no-model'}}})
    monkeypatch.setattr(server, 'project_path', lambda path: synthetic / path)
    monkeypatch.setattr(server, 'runtime_sources', lambda: {'recipes': 'synthetic'})
    monkeypatch.setenv('FUZEV_NOUL_PREFERENCE', '1')
    monkeypatch.setenv('FUZEV_DIMENSION_CHOICE', '1' if enabled else '0')
    calls = []

    class Backend:
        candidate_batch_size = 1
        readout_runtime = {'candidate_batch_size': 1}

        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(server, 'NativeBackend', Backend)
    monkeypatch.setattr(server, 'extract_features', lambda *_args: calls.append('synthetic-job-only'))
    with TestClient(server.create_app()) as client:
        response = client.post('/v1/jobs/extract', json={'prepared': ['dimension-choice-text.jsonl']})
        assert response.status_code == (202 if enabled else 400), response.text
    assert bool(calls) is enabled
