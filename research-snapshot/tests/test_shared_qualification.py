"""Synthetic strict eligibility checks for the single shared architecture."""

import copy

import pytest

from fuzev_research.shared_head import (ARCHITECTURE, FEATURE_TAP, PARAMETER_SHARING, PRIMITIVES,
                               RAW_FORMULA, PROJECTION_FORMULA, EVALUATION_FORMULA,
                               SharedDecisionHead)
from fuzev_research.shared_qualification import validate_shared_reference
from fuzev_research.shared_train import PROFILE, TRACKS, source_identity


def accepted_gate(temperatures):
    tracks, checks = {}, []
    for key in sorted(TRACKS):
        tracks[key] = {'control': {'normalized_nll': 1., 'accuracy': .6},
                       'head': {'normalized_nll': .5, 'accuracy': .8}}
        for metric in (('ordinal_mae',) if '/ordinal/' in key else ('accuracy', 'tie_aware_accuracy')):
            interval = {'difference': -.2 if metric == 'ordinal_mae' else .2,
                        'low': -.3 if metric == 'ordinal_mae' else .1,
                        'high': -.1 if metric == 'ordinal_mae' else .3,
                        'groups': 10, 'draws': PROFILE['bootstrap_draws'], 'seed': PROFILE['bootstrap_seed']}
            checks.append({'track': key, 'metric': metric, 'passed': True, 'interval': interval})
    checks.append({'track': 'macro', 'metric': 'normalized_nll', 'passed': True,
                   'interval': {'difference': -.5, 'low': -.6, 'high': -.4, 'groups': 20,
                                'draws': PROFILE['bootstrap_draws'], 'seed': PROFILE['bootstrap_seed']}})
    return {'eligible': True, 'selection_role': 'development', 'external_selection': False,
            'calibration_selection': False, 'membership': {'records': 100, 'ordered_sha256': 'a' * 64},
            'temperature_by_primitive': temperatures, 'tracks': tracks, 'checks': checks,
            'macro_normalized_nll': {'control': 1., 'head': .5}, 'gain_vs_label_token': .5}


def manifest():
    code = source_identity()
    temperatures = {'choice': 1., 'noul': 2., 'ordinal': 3.}
    head = SharedDecisionHead(seed=43)
    return {'architecture': ARCHITECTURE, 'schema_version': 25, 'hidden_size': 2560, 'projection_size': 512,
            'query_adapter_size': 64, 'key_adapter_size': 16, 'key_adapter': head.key_adapter,
            'candidate_gate_input_scale': 1., 'state_tensor_count': 15, 'parameter_count': 2872401,
            'raw_formula': RAW_FORMULA, 'projection_formula': PROJECTION_FORMULA,
            'evaluation_formula': EVALUATION_FORMULA,
            'feature_tap': FEATURE_TAP, 'precision': 'float32', 'primitives': list(PRIMITIVES),
            'parameter_sharing': PARAMETER_SHARING, 'initialization': head.initialization,
            'lm_assistance': head.lm_assistance, 'requires_lm_scores': True,
            'feature_identity': {'synthetic_only': True}, 'preparation': {'synthetic_only': True},
            'fit_membership': {'records': 200, 'ordered_sha256': 'e' * 64},
            'shared_head_code_sha256': code['shared_head.py'], 'calibration_sha256': 'f' * 64,
            'artifact_kind': 'calibrated-reference-head',
            'learned_head_count': 1, 'frozen_lm_head_trained': False, 'stage': 'mixed',
            'calibration_role': 'calibration', 'training_code': code, 'calibration_code': code,
            'shared_training_code': code, 'operator_sha256': code['shared_train.py'],
            'development_membership': {'records': 100, 'ordered_sha256': 'a' * 64},
            'calibration_membership': {'records': 50, 'ordered_sha256': 'b' * 64},
            'shared_development_selection': accepted_gate({'choice': 1., 'noul': 1., 'ordinal': 1.}),
            'shared_development_gate': accepted_gate(temperatures),
            'weights_sha256': 'c' * 64,
            'selected_checkpoint': {'selection_role': 'development', 'sha256': 'c' * 64,
                                    'manifest_sha256': 'd' * 64},
            'calibration_source': {'role': 'calibration', 'membership': {'records': 50, 'ordered_sha256': 'b' * 64},
                                   'tensor_access': 'owned row slices only'}}, temperatures


def test_shared_gate_accepts_one_scorer_without_reading_corpus():
    head = SharedDecisionHead(seed=43)
    value, temperatures = manifest()
    validate_shared_reference(head, value, temperatures)


@pytest.mark.parametrize('failure', ['three_heads', 'lm_training', 'checkpoint', 'source', 'cal_source',
                                     'false_flag', 'missing_track', 'missing_check', 'failed_ci',
                                     'macro_ci', 'changed_weights', 'temperature', 'group_count'])
def test_bad_admission_rejected(failure):
    head = SharedDecisionHead(seed=43)
    value, temperatures = manifest()
    if failure == 'three_heads':
        value['learned_head_count'] = 3
    elif failure == 'lm_training':
        value['frozen_lm_head_trained'] = True
    elif failure == 'checkpoint':
        value['artifact_kind'] = 'training-checkpoint'
    elif failure == 'source':
        value['operator_sha256'] = 'f' * 64
    elif failure == 'cal_source':
        value['calibration_source']['role'] = 'development'
    elif failure == 'false_flag':
        value['shared_development_gate']['eligible'] = False
    elif failure == 'missing_track':
        value['shared_development_gate']['tracks'].pop(next(iter(TRACKS)))
    elif failure == 'missing_check':
        value['shared_development_gate']['checks'].pop()
    elif failure == 'failed_ci':
        value['shared_development_gate']['checks'][0]['interval']['low'] = -.03
    elif failure == 'macro_ci':
        value['shared_development_gate']['checks'][-1]['interval']['high'] = .01
    elif failure == 'changed_weights':
        value['selected_checkpoint']['sha256'] = 'f' * 64
    elif failure == 'temperature':
        temperatures = copy.deepcopy(temperatures)
        temperatures['noul'] = 1.
    else:
        value['shared_development_gate']['checks'][0]['interval']['groups'] = 1
    with pytest.raises(ValueError, match='Shared reference:'):
        validate_shared_reference(head, value, temperatures)


def test_full_admission_never_bypasses_corpus_readiness():
    head = SharedDecisionHead(seed=43)
    value, temperatures = manifest()
    with pytest.raises(ValueError, match='full read-only corpus'):
        validate_shared_reference(head, value, temperatures, corpus=True)
