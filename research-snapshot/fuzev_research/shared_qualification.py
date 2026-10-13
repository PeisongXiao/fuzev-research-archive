"""Architecture-specific admission of one DEV-accepted, CAL-frozen scorer."""

import math
from pathlib import Path

from torch import nn

from .features import file_sha256
from .shared_head import (ARCHITECTURE, CODE_FILES, FEATURE_TAP, HIDDEN_SIZE, PRIMITIVES,
                          PROJECTION_SIZE, SCHEMA_VERSION, SharedDecisionHead, _adapter_geometry,
                          _channel_geometry, _key_adapter_geometry, _profile, _state)


def _require(condition, message):
    if not condition:
        raise ValueError('Shared reference: ' + message)


def _gate(gate, membership, temperatures):
    from .shared_train import PROFILE, TRACKS

    _require(isinstance(gate, dict) and gate.get('eligible') is True
             and gate.get('selection_role') == 'development'
             and gate.get('external_selection') is False
             and gate.get('calibration_selection') is False
             and gate.get('membership') == membership
             and gate.get('temperature_by_primitive') == temperatures,
             'missing complete DEV-only acceptance')
    tracks = gate.get('tracks', {})
    _require(set(tracks) == TRACKS, 'partial DEV task-track evidence')
    macro = {method: sum(tracks[key][method]['normalized_nll'] for key in sorted(TRACKS)) / len(TRACKS)
             for method in ('control', 'head')}
    _require(all(math.isfinite(value) for value in macro.values())
             and gate.get('macro_normalized_nll') == macro
             and math.isclose(gate.get('gain_vs_label_token', math.nan), macro['control'] - macro['head'], abs_tol=1e-12)
             and macro['control'] - macro['head'] >= PROFILE['minimum_improvement'],
             'missing supported DEV NLL gain')
    required = {(key, 'ordinal_mae') if '/ordinal/' in key else (key, 'tie_aware_accuracy') for key in TRACKS}
    required |= {(key, 'accuracy') for key in TRACKS if '/ordinal/' not in key
                 and tracks[key]['control'].get('accuracy') is not None}
    required.add(('macro', 'normalized_nll'))
    checks = gate.get('checks', [])
    _require(len(checks) == len(required) and {(item.get('track'), item.get('metric')) for item in checks} == required,
             'partial or duplicate DEV retention checks')
    for item in checks:
        interval = item.get('interval', {})
        _require(item.get('passed') is True and interval.get('groups', 0) >= 2
                 and interval.get('draws') == PROFILE['bootstrap_draws']
                 and interval.get('seed') == PROFILE['bootstrap_seed']
                 and all(type(interval.get(key)) in (float, int) and math.isfinite(interval[key])
                         for key in ('difference', 'low', 'high'))
                 and interval['low'] <= interval['high'], 'invalid DEV confidence evidence')
        metric = item['metric']
        passed = (interval['high'] < 0 if metric == 'normalized_nll' else
                  interval['high'] <= .1 if metric == 'ordinal_mae' else interval['low'] >= -.02)
        _require(passed, 'DEV retention failed despite accepted flag')


def validate_shared_reference(bundle, manifest, temperatures, *, features=None, corpus=False):
    _require(isinstance(bundle, SharedDecisionHead) and manifest.get('architecture') == ARCHITECTURE,
             'one shared decision scorer is required')
    _require(all(type(manifest.get(key)) is int and manifest[key] == value for key, value in
                 (('schema_version', SCHEMA_VERSION), ('hidden_size', HIDDEN_SIZE), ('projection_size', PROJECTION_SIZE)))
             and (bundle.hidden_size, bundle.projection_size) == (HIDDEN_SIZE, PROJECTION_SIZE)
             and (bundle.scorer.hidden_size, bundle.scorer.projection_size) == (HIDDEN_SIZE, PROJECTION_SIZE),
             'incompatible rank-512 schema or scorer geometry')
    _require(isinstance(getattr(bundle, 'gate', None), nn.Linear)
             and (bundle.gate.in_features, bundle.gate.out_features) == (HIDDEN_SIZE, 1),
             'one shared query-only scalar context gate is required')
    try:
        _adapter_geometry(bundle)
        _key_adapter_geometry(bundle)
        _channel_geometry(bundle)
        _profile(manifest, FEATURE_TAP)
        _state(bundle.state_dict())
    except ValueError as exc:
        raise ValueError('Shared reference: invalid query-adapter/channel-gate state or provenance') from exc
    _require(bundle.initialization == manifest['initialization']
             and bundle.lm_assistance == manifest['lm_assistance'],
             'scorer initialization or assistance differs from its manifest')
    _require(manifest.get('learned_head_count') == 1 and manifest.get('frozen_lm_head_trained') is False,
             'unexpected additional learned heads or LM-head updates')
    _require(manifest.get('artifact_kind') == 'calibrated-reference-head'
             and manifest.get('stage') == 'mixed' and manifest.get('calibration_role') == 'calibration',
             'training checkpoint is not a calibrated reference')
    code = manifest.get('training_code', {})
    _require(set(code) == CODE_FILES
             and code == manifest.get('shared_training_code') == manifest.get('calibration_code')
             and manifest.get('operator_sha256') == code.get('shared_train.py'),
             'actual training/calibration source identity mismatch')
    _gate(manifest.get('shared_development_selection'), manifest.get('development_membership'),
          {name: 1. for name in PRIMITIVES})
    _gate(manifest.get('shared_development_gate'), manifest.get('development_membership'), temperatures)
    lock = manifest.get('selected_checkpoint', {})
    _require(lock.get('selection_role') == 'development'
             and lock.get('sha256') == manifest.get('weights_sha256')
             and isinstance(lock.get('manifest_sha256'), str) and len(lock['manifest_sha256']) == 64,
             'shared weights do not match the locked DEV checkpoint')
    calibration = manifest.get('calibration_source', {})
    _require(calibration.get('role') == 'calibration'
             and calibration.get('membership') == manifest.get('calibration_membership')
             and calibration.get('tensor_access') == 'owned row slices only',
             'missing CAL-only frozen-weight provenance')
    if not corpus:
        return
    _require(features is not None and features.read_only, 'full read-only corpus required')
    from .paths import project_path
    from .preference_head_retrain import read_roles
    from .shared_train import development, source_identity
    from .shared_training_data import SharedFeatureCache
    from .train import membership_identity

    roles = read_roles(features)
    for role, field in (('fit', 'fit_membership'), ('development', 'development_membership'),
                        ('calibration', 'calibration_membership')):
        _require(manifest[field] == membership_identity(roles[role]), 'full role membership changed')
    _require(manifest['feature_identity'] == features.identity
             and manifest['preparation'] == features.require_ready('calibrate')
             and manifest['feature_origins'] == features.producer_origins()
             and code == source_identity(), 'full corpus or qualified operator source changed')
    directory = Path(lock.get('directory', '')).resolve()
    _require(directory.is_relative_to(project_path('weights')), 'locked checkpoint escaped project weights')
    _require(file_sha256(directory / 'head.safetensors') == lock['sha256']
             and file_sha256(directory / 'manifest.json') == lock['manifest_sha256'],
             'locked checkpoint files changed')
    _require(calibration['feature_identity_sha256'] == file_sha256(features.directory / 'identity.json'),
             'CAL feature identity changed')
    cache = SharedFeatureCache(features, roles['development'], 'development', 'cpu')
    for recorded, values in ((manifest['shared_development_selection'], None),
                             (manifest['shared_development_gate'], temperatures)):
        actual = development(bundle, cache, values)
        _require(actual['eligible'], 'recomputed full DEV acceptance failed')
        for key in ('control', 'head'):
            _require(math.isclose(actual['macro_normalized_nll'][key], recorded['macro_normalized_nll'][key],
                                  rel_tol=1e-5, abs_tol=1e-6), 'DEV CPU/CUDA scores disagree')
