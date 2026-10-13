"""Append an evaluation-only Kev development panel without replacing fixed E2E.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.

Preparation emits no predictions and loads no tensors or backbone. Expected
answers are kept outside HTTP request bodies. No E2E execution is authorized here.
"""

import hashlib
import json
import os
from pathlib import Path

from kev_requests_adapter import iter_request_rows


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
OLD = ROOT / 'runs/native-real/qualification/shared-e2e-inputs.json'
SUITE = ROOT / 'data/evaluation/kev-breadth-v1'
OUT = ROOT / 'data/evaluation/e2e-with-kev-v1'
PINS = {
    OLD: 'bea7571002c4e659a797204799bfcf01b6d7ab0334f1d9a2549f34d744bc0667',
    SUITE / 'development.jsonl': '9aad8f4a7374988b2768d31135d5690137df3a3d6cbc1e771db4ee95ca515a94',
    SUITE / 'test.jsonl': '91a64f0a9448dcaece421c051a4b3fd079c15b84ea6cb9cfa8155237d8d1fa18',
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare():
    observed_pins = {path: sha(path) for path in PINS}
    old = json.loads(OLD.read_text())
    records = [json.loads(line) for line in (SUITE / 'development.jsonl').read_text().splitlines()]
    if len(old['rows']) != 2000 or len(records) != 1990 or sum(len(r['questions']) for r in records) != 3075:
        raise ValueError('Incomplete fixed original or Kev development population')
    # Verify the native split boundary by identifiers, without scoring either set.
    test = [json.loads(line) for line in (SUITE / 'test.jsonl').read_text().splitlines()]
    dev_groups = {r['_meta']['group_id'] for r in records}
    test_groups = {r['_meta']['group_id'] for r in test}
    if dev_groups & test_groups:
        raise ValueError('Kev development/test group leakage')
    OUT.mkdir(parents=True, exist_ok=False)
    requests, expected = OUT / 'kev-development-requests.jsonl', OUT / 'kev-development-expected.jsonl'
    with requests.open('x') as request_file, expected.open('x') as answer_file:
        count = 0
        for row in iter_request_rows(records, split='development'):
            request = {'model': 'native-4.05', 'head_id': 'reference', 'messages': row['context'],
                       'primitive': row['primitive'], 'question': row['question'], 'candidates': row['candidates']}
            common = {'id': row['id'], 'record_id': row['official_record_id'],
                      'question_id': row['official_question_id'], 'source': row['source']}
            request_file.write(json.dumps({**common, 'request': request}, ensure_ascii=True, allow_nan=False) + '\n')
            answer_file.write(json.dumps({**common, 'target': row['target'], 'official_keys': row['official_keys'],
                                         'candidate_keys': row['candidate_keys'],
                                         'official_label_index': row['official_label_index']}, allow_nan=False) + '\n')
            count += 1
    if count != 3075:
        raise ValueError('Incomplete Kev API question panel')
    for path, pin in PINS.items():
        if sha(path) != pin:
            raise ValueError('Evaluation source changed during preparation')
    manifest = {'kind': 'fixed-e2e-multipanel-v1', 'external_evaluation_only': True,
                'training_allowed': False, 'model_selection_allowed': False, 'calibration_allowed': False,
                'run_allowed_now': False, 'gate': 'Main head quality and frozen Kev comparison must pass before real Jev API E2E',
                'same_inputs_across': ['native-4.05', 'native-3.05', 'native-5.05',
                                       'vllm-nvidia', 'vllm-radixark', 'vllm-exl3-4.05'],
                'combined_records': 3990, 'combined_questions_or_requests': 5075,
                'report_panels_separately': True, 'new_data_fitting': False, 'e2e_executed': False,
                'recorded_input_sha256': {str(path): pin for path, pin in PINS.items()},
                'observed_input_sha256': {str(path): pin for path, pin in observed_pins.items()},
                'panels': {'original-fixed': {'path': str(OLD), 'sha256': observed_pins[OLD], 'records': 2000, 'requests': 2000,
                                             'sampling_and_weights': 'Retain original quotas, seed, inclusion probabilities and targets unchanged'},
                           'kev-development': {'source': str(SUITE / 'development.jsonl'),
                                               'source_sha256': observed_pins[SUITE / 'development.jsonl'],
                                               'records': 1990, 'questions': 3075,
                                               'requests': str(requests), 'requests_sha256': sha(requests),
                                               'expected': str(expected), 'expected_sha256': sha(expected),
                                               'selection': 'Entire official frozen development partition, original record/question order',
                                               'metric': 'Official breadth-v1 per-dataset metrics and chance-corrected index; no training DEV use'}},
                'preparation_source_sha256': sha(__file__),
                'adapter_sha256': sha(Path(__file__).with_name('kev_requests_adapter.py')),
                'qualified': False, 'servable': False}
    with (OUT / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'manifest': str(OUT / 'manifest.json'), 'manifest_sha256': sha(OUT / 'manifest.json'),
                      'records': 3990, 'requests': 5075, 'e2e_executed': False}))


if __name__ == '__main__':
    prepare()
