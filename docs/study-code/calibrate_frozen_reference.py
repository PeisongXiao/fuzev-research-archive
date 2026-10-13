"""One CAL-only CUDA temperature fit and full frozen processed evaluation.

No backbone, head optimizer, new checkpoint selection, external fitting, or
production qualification. Existing head/training Source8 stays unchanged.
"""

import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import types

from benchmark_kev_text import PLAN, ROOT, read, sha, verify, write


REPORT = ROOT / 'reports/2026-10-10-reference-calibration'
OUTPUT = ROOT / 'weights/frozen-reference-calibration-20261010'
FEATURES = ROOT / 'data/features/native-4.05-dimension-choice-v4'
ADMISSION = ROOT / 'reports/2026-10-10-retained-clean-refit/admission-v1.json'
ADMISSION_SHA = '4acfb1b4d4c054beb910d0bd6e4ea86b43bc784916dd13c049d1fbe347235b31'
TASK = 'training.frozen-reference-calibration'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fit_temperature(logits, targets, mask, weights):
    import torch
    require(all(t.device.type == 'cuda' for t in (logits, targets, mask, weights)), 'CUDA scalar fitting required')
    require(weights.shape == (len(logits),) and torch.isfinite(weights).all() and (weights > 0).all(),
            'Positive finite original CAL weights required')
    finite = logits.detach().masked_fill(~mask, 0)
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(.03), math.log(100)).exp()
        scaled = (finite / temperature).masked_fill(~mask, -1e9)
        losses = -(targets * torch.log_softmax(scaled, dim=1)).sum(dim=1) / mask.sum(dim=1).float().log()
        loss = (losses * weights).sum() / weights.sum()
        require(torch.isfinite(loss), 'Nonfinite CUDA calibration objective')
        loss.backward()
        return loss

    with torch.no_grad():
        before = float((-(targets * torch.log_softmax(finite.masked_fill(~mask, -1e9), dim=1)).sum(1)
                        / mask.sum(1).float().log() * weights).sum() / weights.sum())
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(math.log(.03), math.log(100)).exp())
    with torch.no_grad():
        after = float((-(targets * torch.log_softmax((finite / temperature).masked_fill(~mask, -1e9), dim=1)).sum(1)
                       / mask.sum(1).float().log() * weights).sum() / weights.sum())
    require(after <= before + 1e-6, 'CAL solver worsened its fixed objective')
    return temperature, {'records': len(logits), 'before_normalized_ce': before, 'after_normalized_ce': after,
                         'device': str(logits.device), 'optimizer': 'LBFGS', 'maximum_iterations': 50}


def run():
    import torch
    from safetensors.torch import load_file
    from fuzev_research.state import Store
    from fuzev_research.evaluate import evaluate_cell

    require(not OUTPUT.exists(), 'Never overwrite a calibration attempt or fit temperatures twice')
    plan = read(PLAN)
    pins = dict(plan['input_sha256'])
    pins.update({str(PLAN): sha(PLAN), str(ADMISSION): ADMISSION_SHA,
                 str(REPORT / 'predeclaration.md'): sha(REPORT / 'predeclaration.md'),
                 str(Path(__file__)): sha(__file__)})
    verify(pins)
    case = plan['candidates']['lm-assisted']
    admission = read(ADMISSION)
    require(admission['protected_disjoint'] is True and admission['component_closed'] is True
            and admission['quarantined_fit_records'] == 12, 'Original quarantine/leakage admission required')
    fit_receipt = read(ROOT / 'weights/retained-clean-20261010/lm-assisted/fit/completion.json')
    dev_receipt = read(ROOT / 'weights/retained-clean-20261010/lm-assisted/dev/completion.json')
    require(fit_receipt['checkpoint_sha256'] == dev_receipt['checkpoint_sha256'] == case['weights_sha256']
            and fit_receipt['training_records'] == 185178 and fit_receipt['actual_optimizer_updates'] == 3620
            and dev_receipt['canonical']['eligible'] is True, 'Same frozen clean full-FIT endpoint and canonical DEV evidence required')
    pins.update({str(ROOT / 'weights/retained-clean-20261010/lm-assisted' / phase / 'completion.json'):
                 sha(ROOT / 'weights/retained-clean-20261010/lm-assisted' / phase / 'completion.json')
                 for phase in ('fit', 'dev')})
    package = types.ModuleType('_frozen_reference_a')
    package.__path__ = [str(Path(case['tree']) / 'fuzev_research')]
    sys.modules[package.__name__] = package
    training = importlib.import_module(package.__name__ + '.shared_train')
    require(training.source_identity() == case['source8'] and training.PROFILE == case['profile'], 'Frozen training/scoring source differs')
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), 'CUDA runtime required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = training.SharedDecisionHead(seed=43, key_gain=case['profile']['key_gain']).to('cuda:0')
    model.load_state_dict(load_file(case['weights'], device='cuda:0'), strict=True)
    model.eval().requires_grad_(False)
    frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
    store = Store()
    store.initialize()
    with store.connect() as db:
        db.execute('INSERT OR IGNORE INTO tasks(id,parent,title,updated) VALUES(?,?,?,?)',
                   (TASK, 'training', 'Frozen reference CAL-only CUDA temperatures and processed quality', time.time()))
    store.update(TASK, 'in_progress', 'Original CAL-only positive temperatures; head and protected evaluation data remain frozen.', 0.)
    OUTPUT.mkdir()
    write(OUTPUT / 'attempt.json', {'head_sha256': case['weights_sha256'], 'plan_sha256': sha(PLAN),
          'calibration_role': 'calibration', 'expected_cal_records': 23813, 'device': 'cuda:0',
          'calibration_worker_sha256': sha(__file__), 'head_training': False,
          'external_fitting': False, 'qualified': False, 'servable': False})
    features = training.FeatureStore(FEATURES, read(FEATURES / 'identity.json'), read_only=True)
    try:
        training.require_feature_pin(features)
        roles = training.read_roles(features)
        eligible = set(admission['training_ids'])
        roles['fit'] = [row for row in roles['fit'] if row['id'] in eligible]
        require(len(roles['fit']) == len(eligible) == 185178 and len(roles['development']) == 24937
                and len(roles['calibration']) == 23813, 'Complete original safe FIT/DEV/CAL membership required')
        training.check_rows(roles['calibration'], 'calibration')
        ready = features.require_ready('calibrate')
        origins = features.producer_origins()
        cache = training.SharedFeatureCache(features, roles['calibration'], 'calibration', 'cuda:0')
        pins.update({str(FEATURES / name): pin for name, pin in cache.files.items()})
        weights = training.role_weights(cache.rows)
        temperatures, fits = {}, {}
        for index, primitive in enumerate(training.PRIMITIVES):
            rows = [row for row in cache.rows if row['primitive'] == primitive]
            logits, targets, masks = [], [], []
            with torch.inference_mode():
                for offset in range(0, len(rows), 256):
                    batch = cache.batch(rows[offset:offset + 256])
                    logits.append(training.shared_logits(model, batch, primitive).clone())
                    targets.append(batch['targets'].clone())
                    masks.append(batch['mask'].clone())
            factors = torch.tensor([weights[row['id']] for row in rows], device='cuda:0', dtype=torch.float32)
            temperatures[primitive], fits[primitive] = fit_temperature(torch.cat(logits), torch.cat(targets), torch.cat(masks), factors)
            store.update(TASK, 'in_progress', f'{primitive} CAL temperature frozen; no shared-head updates.', .1 + .1 * (index + 1))
        require(all(torch.equal(value, frozen[key]) for key, value in model.state_dict().items())
                and all(p.grad is None for p in model.parameters()), 'Calibration altered the shared head')
        write(OUTPUT / 'temperature-lock.json', {'temperatures': temperatures, 'fits': fits, 'head_sha256': case['weights_sha256'],
              'calibration_membership': cache.membership, 'calibration_cache': cache.provenance(),
              'worker_sha256': sha(__file__), 'external_fitting': False, 'head_training': False})
        del cache
        dev = training.SharedFeatureCache(features, roles['development'], 'development', 'cuda:0')
        final_dev = training.development(model, dev, temperatures)
        pins.update({str(FEATURES / name): pin for name, pin in dev.files.items()})
        write(OUTPUT / 'calibrated-dev.json', final_dev)
        del dev
        metadata = {'artifact_kind': 'calibrated-reference-head', 'stage': 'mixed', 'calibration_role': 'calibration',
                    'feature_identity': features.identity, 'feature_origins': origins, 'preparation': ready,
                    'fit_membership': training.membership_identity(roles['fit']),
                    'development_membership': training.membership_identity(roles['development']),
                    'calibration_membership': training.membership_identity(roles['calibration']),
                    'training_code': case['source8'], 'shared_training_code': case['source8'], 'calibration_code': case['source8'],
                    'operator_sha256': case['source8']['shared_train.py'], 'calibration_worker_sha256': sha(__file__),
                    'actual_fit_operator_sha256': fit_receipt['worker_sha256'], 'fit_receipt_sha256': pins[str(ROOT / 'weights/retained-clean-20261010/lm-assisted/fit/completion.json')],
                    'profile': case['profile'], 'learned_head_count': 1, 'frozen_lm_head_trained': False,
                    'weights_sha256': case['weights_sha256'], 'seed': 43,
                    'shared_development_selection': dev_receipt['canonical'], 'shared_development_gate': final_dev,
                    'same_family_clean_refit_confirmation_passed': dev_receipt['confirmation_passed'],
                    'same_family_clean_refit_confirmation': dev_receipt['component_panel'],
                    'selection_policy': 'unchanged predeclared primary, fixed clean full-FIT epoch-5 endpoint',
                    'calibration_procedure': 'CUDA CAL-only positive per-primitive scalar LBFGS; original balanced group/track normalized CE',
                    'experimental': True, 'qualified': False, 'servable': False}
        training.save_shared_bundle(OUTPUT / 'reference', model, metadata, temperatures)
        require(sha(OUTPUT / 'reference/head.safetensors') == case['weights_sha256'], 'Calibrated bundle changed checkpoint bytes')
        store.update(TASK, 'in_progress', 'Temperatures locked; evaluating complete original native4 TEST on CUDA, no LLM.', .5)

        def progress(processed, passed, total):
            if processed % 256 == 0 or processed == total:
                store.update(TASK, 'in_progress', f'Processed CUDA TEST: {processed}/{total}; evaluation-only.', .5 + .5 * processed / total)
                from fuzev_research.acquire import write_json
                write_json(ROOT / 'runs/validation-progress.json', {'run_id': 'frozen-reference-main-20261010',
                           'stage': 'processed', 'status': 'in_progress', 'processed': processed, 'passed': passed,
                           'total': total, 'pass_rate': passed / processed if processed else None, 'updated_at': time.time()})
        quality = evaluate_cell(features, features, model, temperatures, device='cuda:0', progress=progress)
        require(quality['test_membership']['records'] == 25599, 'Do not narrow original main quality to an easier subset')
        write(REPORT / 'processed-quality.json', quality)
        require(all(torch.equal(value, frozen[key]) for key, value in model.state_dict().items()), 'Processed evaluation altered head')
        verify(pins)
        result = {'status': 'completed', 'temperatures': temperatures, 'fits': fits, 'canonical_calibrated_dev_pass': final_dev['eligible'],
                  'processed_main_gates': quality['gates'], 'head_sha256': case['weights_sha256'],
                  'device': 'cuda:0', 'torch': torch.__version__, 'recorded_torch_version': '2.13.0+cu130',
                  'LLM_loaded': False, 'cpu_head_execution': False, 'external_fitting': False,
                  'head_training': False, 'learned_head_count': 1, 'input_sha256_before_and_after': pins,
                  'qualified': False, 'servable': False}
        write(OUTPUT / 'completion.json', result)
        store.record_result(TASK, {'temperatures': temperatures, 'quality_gate': quality['gates']['head_calibrated']['status'],
                                 'device': 'cuda:0', 'qualified': False, 'servable': False})
        store.update(TASK, 'completed', 'Single CAL fit and full processed CUDA assessment complete; all failures remain visible.', 1.)
        print(json.dumps({'status': 'completed', 'temperatures': temperatures,
                          'main_raw': quality['gates']['head_raw']['status'],
                          'main_calibrated': quality['gates']['head_calibrated']['status'], 'output': str(OUTPUT)}), flush=True)
    except BaseException as error:
        store.update(TASK, 'failed', str(error))
        raise
    finally:
        features.close()


if __name__ == '__main__':
    run()
