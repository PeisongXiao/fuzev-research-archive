"""One fresh lower-LR Noul repeat after FIT/DEV-confirmed tanh saturation."""

import json
from pathlib import Path
import time

import torch

from .acquire import write_json
from .anchor_head import load_anchor
from .anchor_trial import ANCHOR, COMMON, PARITY, PROFILE, train_primitive
from .features import FeatureStore, file_sha256
from .paths import project_path
from .preference_head_retrain import read_roles
from .state import Store


LEARNING_RATE = 1e-6
PREVIOUS = 'weights/cuda-anchor-residual-dev-20261005'
OUTPUT = 'weights/cuda-anchor-noul-lr1e6-20261005'


def previous_selection():
    previous = project_path(PREVIOUS)
    choice_path = previous / 'choice/trial.json'
    noul_path = previous / 'noul/trial.json'
    choice, noul = (json.loads(path.read_text()) for path in (choice_path, noul_path))
    if (choice['status'] != 'dev_accepted' or not choice['completed'] or not choice['selected']
            or not choice['selected']['development']['eligible']
            or noul['status'] != 'dev_rejected' or not noul['completed'] or noul['selected'] is not None
            or noul['epochs'][-1]['total_nonzero_key_gradient_steps'] != 2):
        raise ValueError('Expected preserved accepted Choice and saturated rejected Noul trial')
    selected = choice['selected']
    if file_sha256(selected['checkpoint']) != selected['sha256']:
        raise ValueError('Retained Choice checkpoint changed')
    return {'choice': {'trial_sha256': file_sha256(choice_path), 'checkpoint': selected['checkpoint'],
                       'checkpoint_sha256': selected['sha256'], 'selected_epoch': selected['epoch'],
                       'status': 'retained_DEV_accepted_not_retrained'},
            'failed_noul_trial_sha256': file_sha256(noul_path),
            'previous_protocol_sha256': file_sha256(previous / 'protocol.json')}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Fresh lower-LR repeat requires CUDA; no CPU fallback')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    retained = previous_selection()
    parity_path = project_path(PARITY)
    parity = json.loads(parity_path.read_text())
    if parity['numerical_gate'] != 'pass' or parity['llm_loaded'] is not False:
        raise ValueError('Full reference readout must remain exact')
    anchor = load_anchor(project_path(ANCHOR), manifest_sha256=parity['anchor_manifest_sha256'])
    if anchor.weights_sha256 != parity['anchor_weights_sha256']:
        raise ValueError('Frozen reference readout changed')
    directory = project_path('data/features/native-4.05-noul-preference-v3')
    if file_sha256(directory / 'identity.json') != parity['feature_identity_sha256']:
        raise ValueError('Qualified feature identity changed')
    features = FeatureStore(directory, json.loads((directory / 'identity.json').read_text()), read_only=True)
    store = Store()
    try:
        roles = read_roles(features)
        output = project_path(OUTPUT)
        output.mkdir(exist_ok=False)
        protocol = {'started_at': time.time(), 'profile': {**COMMON, **PROFILE['noul'], 'learning_rate': LEARNING_RATE},
                    'primitive': 'noul', 'fresh_initialization': True, 'resume_rejected_checkpoint': False,
                    'reason': 'FIT/DEV-only diagnosis: tanh saturation; 2 of 123 key-weight gradient updates live',
                    'retained': retained, 'selection_role': 'development', 'external_selection': False,
                    'calibration_selection': False, 'scope': 'one fresh Noul CUDA FIT/DEV repeat only; no CAL/TEST/E2E',
                    'torch': torch.__version__,
                    'feature_identity_sha256': file_sha256(directory / 'identity.json'),
                    'parity_proof_sha256': file_sha256(parity_path),
                    'anchor_manifest_sha256': anchor.manifest_sha256, 'anchor_weights_sha256': anchor.weights_sha256,
                    'operator_sha256': file_sha256(__file__),
                    'training_operator_sha256': file_sha256(Path(__file__).with_name('anchor_trial.py')),
                    'no_automatic_profile_bound_or_normalization_rescue': True,
                    'qualification': 'unqualified prototype repeat; no head quality or production claim'}
        write_json(output / 'protocol.json', protocol)
        store.update('training.anchor_saturation', 'in_progress',
                     'Confirmed FIT/DEV tanh saturation; running one fresh CUDA Noul LR=1e-6 repeat. Choice bytes, anchor, .25 cap and gates unchanged.', progress=.4)
        result = train_primitive(anchor, features, roles, 'noul', output,
                                 learning_rate=LEARNING_RATE, task='training.anchor_saturation')
        if previous_selection() != retained:
            raise ValueError('Retained Choice or failed evidence changed during the repeat')
        write_json(output / 'run.json', {'completed_at': time.time(), 'noul': result, 'retained': retained,
                                        'status': 'DEV_accepted_not_qualified' if result['selected'] else 'DEV_rejected',
                                        'main_quality': 'not_run', 'calibration_status': 'not_started'})
        store.update('training.anchor_saturation', 'completed' if result['selected'] else 'failed',
                     'Fresh Noul repeat finished: ' + result['status'] + '; not calibrated or quality-qualified.', progress=1)
        print(json.dumps({'output': str(output), 'status': result['status'],
                          'selected_epoch': result['selected']['epoch'] if result['selected'] else None,
                          'choice': 'unchanged', 'main_quality': 'not_run'}), flush=True)
    except BaseException as exc:
        store.update('training.anchor_saturation', 'failed', f'Fresh Noul repeat error: {type(exc).__name__}: {exc}')
        raise
    finally:
        features.close()


if __name__ == '__main__':
    main()
