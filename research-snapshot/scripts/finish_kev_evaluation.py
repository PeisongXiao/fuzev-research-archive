"""Head-only CUDA replay and descriptive scoring after fixed research E2E."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from benchmark_kev_text import HEADS, KEV, PLAN, REPORT, ROOT, SUITE, load_heads, read, sha, verify, write


def replay():
    import torch
    from safetensors.torch import load_file

    plan = read(PLAN)
    verify(plan['input_sha256'])
    torch.set_num_threads(10)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = load_heads(plan)
    directory = ROOT / 'data/features/kev-breadth-v1-native-4.05'
    identity = read(directory / 'identity.json')
    if identity['test_sha256'] != plan['test_sha256'] or identity['plan_sha256'] != sha(PLAN):
        raise ValueError('Wrong external feature lineage')
    references = {lane: read(REPORT / 'native-results-v1' / lane / 'rows.json') for lane in HEADS}
    maximum = dict.fromkeys(HEADS, 0.)
    flips = dict.fromkeys(HEADS, 0)
    seen, feature_pins = set(), {}
    with torch.inference_mode():
        for path in sorted(directory.glob('shard-*.safetensors')):
            feature_pins[str(path)] = sha(path)
            tensors = load_file(str(path), device='cuda:0')
            indices = sorted(int(name.split('.')[0]) for name in tensors if name.endswith('.query'))
            for index in indices:
                if index in seen:
                    raise ValueError('Duplicate feature row')
                seen.add(index)
                query, keys, baseline = [tensors[f'{index}.{name}'].float().unsqueeze(0)
                                         for name in ('query', 'keys', 'baseline')]
                for lane, model in models.items():
                    reference = references[lane][index]
                    kwargs = {'baseline': baseline} if model.requires_lm_scores else {}
                    logits = model(reference['type'], query, keys, **kwargs)
                    probabilities = torch.softmax(logits, dim=-1)[0]
                    if reference['type'] == 'noul':
                        probabilities = probabilities.flip(0)
                    expected = torch.tensor(reference['p'], dtype=torch.float32, device='cuda:0')
                    maximum[lane] = max(maximum[lane], float((probabilities - expected).abs().max()))
                    flips[lane] += int(probabilities.argmax() != expected.argmax())
            del tensors
    if seen != set(range(3089)) or any(maximum[lane] > 1e-6 or flips[lane] for lane in HEADS):
        raise ValueError('Processed head-only replay differs from original text evaluation')
    if any(name == 'exllamav3' or name.startswith('exllamav3.') or name == 'fuzev_research.native' for name in sys.modules):
        raise ValueError('Head-only replay must not load the LLM engine')
    verify(plan['input_sha256'])
    result = {'status': 'pass', 'records': 1990, 'questions': 3089, 'head_execution_device': 'cuda:0',
              'LLM_loaded': False, 'training_performed': False,
              'calibration_performed': False, 'selection_flips': flips, 'maximum_probability_error': maximum,
              'feature_shards_sha256': feature_pins, 'plan_sha256': sha(PLAN), 'worker_sha256': sha(__file__),
              'qualified': False, 'servable': False}
    write(REPORT / 'processed-cuda-replay.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'questions', 'LLM_loaded', 'selection_flips', 'maximum_probability_error')}))


def score_e2e(directory=None):
    sys.path.insert(0, str(KEV))
    from kev.suite import load_split, read_manifest
    from scripts.breadth_report import score_system, bootstrap_index

    directory = Path(directory or REPORT / 'e2e-native-v2').resolve()
    if directory.parent != REPORT or not directory.name.startswith('e2e-'):
        raise ValueError('Score only private fixed-panel research E2E attempts')
    summary = read(directory / 'summary.json')
    if summary['processed'] != 5075 or summary['status'] != 'completed' or summary['frozen_postcheck_status'] != 'pass':
        raise ValueError('Incomplete/unbound E2E cannot be summarized as complete')
    model = summary.get('model', 'native-4.05')
    if model not in ('native-3.05', 'native-4.05', 'native-5.05',
                     'vllm-nvidia', 'vllm-radixark', 'vllm-exl3-4.05'):
        raise ValueError('Unrecognized cross-check cell')
    system = 'research-' + model
    records = load_split(str(SUITE), 'development')
    rows = read(directory / 'kev-development-rows.json')
    if len(records) != 1990 or len(rows) != 3075:
        raise ValueError('Incomplete Kev development E2E')
    official = score_system(records, rows, read_manifest(str(SUITE)))
    uncertainty = bootstrap_index(records, {system: rows}, read_manifest(str(SUITE)), samples=2000, seed=0)
    weighted, totals = defaultdict(list), defaultdict(lambda: [0, 0])
    with (directory / 'http.jsonl').open() as handle:
        count = 0
        for line in handle:
            record = json.loads(line)
            if record['type'] != 'decision':
                continue
            count += 1
            panel = record['panel']
            totals[panel][0] += int(record['selection_passed'])
            totals[panel][1] += 1
            if panel != 'original-fixed':
                continue
            item = record['original_item']
            weight = 1 / item['subset_inclusion_probability']
            external = item.get('original_external_sampling')
            if external is not None:
                weight *= external['sampling_weight']
            weighted[item['task_track']].append((weight, float(record['selection_passed'])))
    if count != 5075:
        raise ValueError('Incomplete HTTP journal')
    # Different original quotas and label-stratified inclusion probabilities are
    # not collapsed into a purported whole-population accuracy.
    result = {'status': 'completed', 'model': model, 'http_requests': count, 'api_response_validation_errors': 0,
              'panels': {panel: {'correct_or_tie_aware': values[0], 'requests': values[1],
                                 'descriptive_selection_accuracy': values[0] / values[1]}
                         for panel, values in totals.items()},
              'original_fixed_weighted_per_track': {
                  track: {'n': len(values), 'weighted_tie_aware_accuracy': sum(w * x for w, x in values) / sum(w for w, _ in values)}
                  for track, values in weighted.items()},
              'weighting_scope': 'Conditional API-eligible represented strata; subset inverse inclusion multiplied by original external sampling weight. No whole-population or causal claim.',
              'kev_development_official': official, 'kev_development_uncertainty': uncertainty,
              'summary_sha256': sha(directory / 'summary.json'), 'rows_sha256': sha(directory / 'kev-development-rows.json'),
              'worker_sha256': sha(__file__), 'training_performed': False, 'calibration_performed': False,
              'qualified': False, 'servable': False}
    write(directory / 'panel-report.json', result)
    print(json.dumps({'panels': result['panels'], 'kev_development': official['overall'],
                      'kev_development_index_ci': uncertainty['index'][system],
                      'external_weighted': {k: v for k, v in result['original_fixed_weighted_per_track'].items()
                                            if k.startswith(('qasper/', 'vl-rewardbench/'))}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('replay', 'score-e2e'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.action == 'replay' and args.output_dir is not None:
        parser.error('--output-dir applies only to descriptive E2E scoring')
    replay() if args.action == 'replay' else score_e2e(args.output_dir)
