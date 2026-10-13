"""CUDA adapter replay on existing external features; no LLM or new evaluation."""

import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

from fuzev_research.kev_vllm_research_head import KevVllmResearchHead, PLAN_SHA, REPORT, ROOT


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run():
    output = ROOT / 'reports/2026-10-10-cross-cell/adapter-cuda-replay.json'
    if output.exists():
        raise FileExistsError('Never overwrite a previous replay')
    receipt = json.loads((REPORT / 'processed-cuda-replay.json').read_text())
    if receipt['status'] != 'pass' or receipt['plan_sha256'] != PLAN_SHA:
        raise ValueError('Original processed replay must match the frozen plan')
    references = json.loads((REPORT / 'native-results-v1/lm-assisted/rows.json').read_text())
    if len(references) != 3089:
        raise ValueError('Require the complete original Kev question population')
    torch.set_num_threads(10)
    head = KevVllmResearchHead()
    source = Path(sys.modules[KevVllmResearchHead.__module__].__file__)
    source_sha = sha(source)
    seen, maximum, flips = set(), 0., 0
    for name, expected_sha in sorted(receipt['feature_shards_sha256'].items()):
        path = Path(name)
        if not path.is_relative_to(ROOT / 'data/features/kev-breadth-v1-native-4.05') or sha(path) != expected_sha:
            raise ValueError('Changed original external feature shard')
        tensors = load_file(str(path), device='cpu')
        for index in sorted(int(name.split('.')[0]) for name in tensors if name.endswith('.query')):
            if index in seen or not 0 <= index < 3089:
                raise ValueError('Duplicated or out-of-range question')
            seen.add(index)
            row = references[index]
            _, probabilities, legacy = head.score(
                row['type'], tensors[f'{index}.query'], tensors[f'{index}.keys'],
                baseline=tensors[f'{index}.baseline'])
            if probabilities != legacy:
                raise ValueError('Raw T1 fields differ')
            if row['type'] == 'noul':
                probabilities.reverse()
            expected = row['p']
            if len(probabilities) != len(expected):
                raise ValueError('Wrong candidate population')
            maximum = max(maximum, max(abs(a - b) for a, b in zip(probabilities, expected)))
            flips += int(max(range(len(expected)), key=probabilities.__getitem__) !=
                         max(range(len(expected)), key=expected.__getitem__))
        del tensors
        if sha(path) != expected_sha:
            raise ValueError('Feature shard changed during replay')
    if seen != set(range(3089)) or maximum > 1e-6 or flips or sha(source) != source_sha:
        raise ValueError('CUDA adapter differs from the frozen original readout')
    if any(name == 'exllamav3' or name.startswith('exllamav3.') or name == 'fuzev_research.native' for name in sys.modules):
        raise ValueError('Adapter replay must not load an LLM engine')
    result = {'status': 'pass', 'questions': len(seen), 'maximum_probability_error': maximum,
              'selection_flips': flips, 'LLM_loaded': False, 'cpu_head_execution': False,
              'adapter_sha256': source_sha,
              'worker_sha256': sha(__file__), 'original_replay_sha256': sha(REPORT / 'processed-cuda-replay.json'),
              'training_performed': False, 'calibration_performed': False,
              'provenance': head.provenance, 'qualified': False, 'servable': False}
    with os.fdopen(os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'questions', 'maximum_probability_error',
                                                'selection_flips', 'LLM_loaded')}))


if __name__ == '__main__':
    run()
