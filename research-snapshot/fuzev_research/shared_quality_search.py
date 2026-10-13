"""Sequential CUDA FIT/DEV searches; no CAL, TEST, E2E or backbone inference."""

import argparse
import json

import torch

from .acquire import write_json
from .features import FeatureStore, file_sha256
from .paths import project_path
from .preference_head_retrain import read_roles
from . import shared_train as training
from .shared_head import ARCHITECTURE
from .state import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--profiles', nargs='+', choices=training.SEARCH_PROFILES, required=True)
    args = parser.parse_args()
    if len(args.profiles) != len(set(args.profiles)):
        raise ValueError('Duplicate profiles waste the fitting budget')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    directory = project_path(training.FEATURES)
    observed_feature_sha = file_sha256(directory / 'identity.json')
    source = training.source_identity()
    write_json(output / 'search-protocol.json', {
        'architecture': ARCHITECTURE, 'profiles': args.profiles,
        'configured_profiles': {name: training.BASE_PROFILE | training.SEARCH_PROFILES[name]
                                for name in args.profiles},
        'source': source, 'search_operator_sha256': file_sha256(__file__),
        'recorded_feature_identity_sha256': training.FEATURE_SHA,
        'feature_identity_sha256': observed_feature_sha,
        'selection': 'all canonical DEV gates, then lowest raw DEV macro normalized NLL',
        'calibration_used': False, 'test_used': False, 'external_selection': False,
        'e2e_used': False, 'llm_loaded': False, 'learned_head_count': 1})
    features = FeatureStore(directory, json.loads((directory / 'identity.json').read_text()), read_only=True)
    results = []
    store = Store()
    try:
        roles = read_roles(features)
        for index, name in enumerate(args.profiles):
            training.PROFILE.clear()
            training.PROFILE.update(training.BASE_PROFILE | training.SEARCH_PROFILES[name])
            store.update(training.TASK, 'in_progress',
                         f'Quality search {index + 1}/{len(args.profiles)}: {name}; FIT/DEV only.',
                         progress=index / len(args.profiles))
            trial = training.run_trial(features, roles, output / name, development_only=True)
            selected = trial['selected']
            results.append({'profile': name, 'directory': str(output / name), 'status': trial['status'],
                            'selected': selected, 'calibration': trial['calibration']})
            eligible = [item for item in results if item['selected'] is not None]
            best = min(eligible, key=lambda item: (
                item['selected']['development']['macro_normalized_nll']['head'], item['profile'])) if eligible else None
            write_json(output / 'search-results.json', {
                'completed': False, 'selection_role': 'development', 'results': results, 'selected': best,
                'external_selection': False, 'calibration_selection': False})
            torch.cuda.empty_cache()
        write_json(output / 'search-results.json', {
            'completed': True, 'selection_role': 'development', 'results': results, 'selected': best,
            'source': training.source_identity(), 'recorded_source': source,
            'feature_identity_sha256': file_sha256(directory / 'identity.json'),
            'recorded_feature_identity_sha256': training.FEATURE_SHA,
            'external_selection': False, 'calibration_selection': False})
        store.update(training.TASK, 'completed',
                     'Initial FIT/DEV profile search finished; no CAL/TEST/E2E read. Family not yet qualified.', progress=1)
        print(json.dumps({'output': str(output), 'selected': None if best is None else best['profile'],
                          'calibration': 'not_started', 'main_quality': 'not_run'}), flush=True)
    except BaseException as exc:
        store.update(training.TASK, 'failed', f'Quality search failed: {type(exc).__name__}: {exc}')
        raise
    finally:
        features.close()


if __name__ == '__main__':
    main()
