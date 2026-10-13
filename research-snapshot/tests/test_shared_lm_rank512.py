"""Synthetic rank-512 geometry, initialization and strict shared dispatch checks."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from fuzev_research import evaluate, shared_head, shared_qualification, shared_train as training
from fuzev_research.head import load_bundle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata():
    code = training.source_identity()
    return {'artifact_kind': 'training-checkpoint', 'stage': 'mixed',
            'feature_identity': {'synthetic_only': True}, 'preparation': {'synthetic_only': True},
            'fit_membership': {'records': 2, 'ordered_sha256': 'b' * 64},
            'development_membership': {'records': 1, 'ordered_sha256': 'c' * 64},
            'training_code': code, 'shared_training_code': dict(code),
            'operator_sha256': code['shared_train.py'], 'profile': dict(training.PROFILE)}


@pytest.fixture(scope='module')
def production():
    return shared_head.SharedDecisionHead(seed=43)


@pytest.fixture
def artifact(tmp_path, production):
    directory = tmp_path / 'rank512'
    shared_head.save_shared_bundle(directory, production, metadata(),
                                   {name: 1. for name in training.PRIMITIVES})
    return directory


def repin(directory, **changes):
    path = directory / 'manifest.json'
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))


def test_actual_sqrt512_scale_and_frozen_lm_centered_formula_for_every_primitive(production):
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(2, 2560, generator=generator)
    candidates = torch.randn(2, 4, 2560, generator=generator)
    baseline = torch.randn(2, 4, generator=generator)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    dot = torch.einsum('bd,bkd->bk', production.scorer.query(query), production.scorer.key(candidates))
    raw = dot / math.sqrt(512)
    mean = raw.masked_fill(~mask, 0).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    expected = (baseline + raw - mean).masked_fill(~mask, -torch.inf)
    wrong = raw * math.sqrt(2)
    wrong = baseline + wrong - wrong.masked_fill(~mask, 0).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    for primitive in training.PRIMITIVES:
        actual = production(primitive, query, candidates, mask, baseline=baseline)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        assert not torch.allclose(actual[mask], wrong[mask], rtol=1e-5, atol=1e-8)
        assert torch.isneginf(actual[~mask]).all()
    assert baseline.grad is None


@pytest.mark.parametrize('loader', [load_bundle, shared_head.load_shared_bundle])
@pytest.mark.parametrize('change', [
    {'schema_version': 3}, {'projection_size': 256}, {'projection_size': 512.},
    {'schema_version': 12.}, {'hidden_size': 1280},
    {'architecture': 'shared-lm-assisted-candidate-dot-product-v1', 'schema_version': 3, 'projection_size': 256},
])
def test_schema_or_geometry_downgrade_rejected_before_weight_access(artifact, monkeypatch, loader, change):
    repin(artifact, **change)
    monkeypatch.setattr(shared_head, 'load', lambda *args: pytest.fail('Invalid geometry reached weight decode'))
    with pytest.raises(ValueError, match='Incompatible'):
        loader(artifact)


def test_rank256_tensors_cannot_be_relabelled_as_rank512_even_with_fresh_hashes(artifact):
    path = artifact / 'head.safetensors'
    state = {key: value[:256].contiguous() for key, value in load_file(str(path)).items()}
    save_file(state, str(path))
    repin(artifact, weights_sha256=sha(path))
    with pytest.raises(ValueError, match='state shape'):
        load_bundle(artifact)


def test_rank256_geometry_cannot_be_saved_under_new_schema(tmp_path):
    head = shared_head.SharedDecisionHead(seed=43, projection_size=256)
    with pytest.raises(ValueError, match='Synthetic geometry'):
        shared_head.save_shared_bundle(tmp_path / 'denied', head, metadata(),
                                       {name: 1. for name in training.PRIMITIVES})
    assert not (tmp_path / 'denied').exists()


def test_old_calibration_schema_rejected_even_with_recomputed_pin(artifact):
    path = artifact / 'calibration.json'
    value = json.loads(path.read_text())
    value.update(schema_version=3, architecture='shared-lm-assisted-candidate-dot-product-v1')
    path.write_text(json.dumps(value))
    repin(artifact, calibration_sha256=sha(path))
    with pytest.raises(ValueError, match='typed primitive calibration'):
        load_bundle(artifact)


@pytest.mark.parametrize('change', [{'schema_version': 3}, {'projection_size': 256},
                                   {'projection_size': 512.}, {'hidden_size': 4}])
def test_qualification_geometry_rejects_old_or_untyped_manifest(production, change):
    manifest = {'architecture': shared_head.ARCHITECTURE, 'schema_version': 18,
                'hidden_size': 2560, 'projection_size': 512, **change}
    with pytest.raises(ValueError, match='rank-512 schema or scorer geometry'):
        shared_qualification.validate_shared_reference(production, manifest, {})


def test_qualification_rejects_old_rank256_object_despite_rank512_manifest():
    head = shared_head.SharedDecisionHead(seed=43, projection_size=256)
    manifest = {'architecture': shared_head.ARCHITECTURE, 'schema_version': 18,
                'hidden_size': 2560, 'projection_size': 512}
    with pytest.raises(ValueError, match='rank-512 schema or scorer geometry'):
        shared_qualification.validate_shared_reference(head, manifest, {})


@pytest.mark.parametrize('architecture,shared', [
    (shared_head.ARCHITECTURE, True),
    ('shared-candidate-dot-product-v1', True),
    ('shared-lm-assisted-candidate-dot-product-v1', True),
    ('candidate-dot-product-v1', False),
])
def test_evaluation_report_records_exact_shared_source_hashes(tmp_path, monkeypatch, architecture, shared):
    (tmp_path / 'identity.json').write_text('{}')
    identity_pin = sha(tmp_path / 'identity.json')
    database = SimpleNamespace(execute=lambda sql: SimpleNamespace(fetchone=lambda: [1]))
    features = SimpleNamespace(directory=tmp_path, identity={}, db=database, close=lambda: None)
    reference = {'features': str(tmp_path), 'feature_identity_sha256': identity_pin,
                 'head': str(tmp_path / 'head'), 'sha256': {name: 'a' * 64 for name in evaluate.HEAD_FILES}}
    monkeypatch.setattr(evaluate, 'open_features', lambda *args: features)
    monkeypatch.setattr(evaluate, 'load_reference', lambda *args: (None, {'architecture': architecture}, {}))
    monkeypatch.setattr(evaluate, 'require_final_ready', lambda *args: {'synthetic_only': True})
    monkeypatch.setattr(evaluate, 'check_hashes', lambda directory, expected: expected)
    report = evaluate.evaluate({'schema_version': 1, 'reference': reference, 'cells': []}, draws=1)
    names = {'evaluate.py', 'evaluation_metrics.py', 'metrics.py', 'features.py', 'head.py', 'train.py'}
    if shared:
        names |= {'shared_head.py', 'shared_train.py', 'shared_training_data.py', 'shared_qualification.py'}
    assert report['evaluation_code_sha256'] == {
        name: sha(Path(evaluate.__file__).with_name(name)) for name in names}
