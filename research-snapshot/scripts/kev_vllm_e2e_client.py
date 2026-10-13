"""Research-only vLLM verifier using the unchanged full native E2E panel."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import kev_e2e_client as canonical

from fuzev_research.kev_vllm_research_server import CELLS, CORE_COMMIT, EXL3_VIEW, MEMORY_FRACTION
from fuzev_research.vllm_receipts import READOUT_FIELDS, readout_sources
from fuzev_research.vllm_service import VllmDecisionResponse


MODEL = None
BINDING = None


def wire_request(request):
    canonical.require(MODEL in CELLS and request['model'] == canonical.REFERENCE_MODEL,
                      'Validate canonical native4 bodies before selecting a vLLM cell')
    return {**request, 'model': MODEL}


def health_check(health, original, binding=None):
    canonical.require(health.get('status') == 'loaded' and health.get('model') == MODEL,
                      'Actual selected vLLM research service must be loaded')
    provenance = health.get('provenance', {})
    canonical.require(all(provenance.get(key) is value for key, value in canonical.FLAGS.items()),
                      'Not an explicit unqualified/raw-T1 research service')
    canonical.require(provenance.get('learned_head_count') == 1
                      and provenance.get('head_lane') == 'lm-assisted'
                      and provenance.get('calibration_id') == 'uncalibrated:T1'
                      and provenance.get('raw_inference_temperature') == 1.
                      and provenance.get('reference_feature_identity') == original,
                      'Frozen reference head/T1/feature identity differs')
    runtime, sources = provenance.get('runtime_cell', {}), provenance.get('runtime_sources', {})
    key, repo, revision, quantization, image = CELLS[MODEL]
    acquired = canonical.read(canonical.ACQUISITION)['models'][key]
    snapshot = Path(acquired['local_snapshot'])
    directory = EXL3_VIEW if quantization == 'exl3' else snapshot
    expected = {'engine': 'vllm-fuzev', 'revision': revision, 'tokenizer_revision': revision,
                'quantization': quantization, 'dtype': 'torch.bfloat16', 'cache_dtype_config': 'auto',
                'memory_fraction': MEMORY_FRACTION, 'context_window': 262144, 'hidden_size': 2560,
                'feature_tap': original['feature_tap'], 'encoding': original['encoding'],
                'max_num_seqs': 4, 'max_num_batched_tokens': 16384, 'prefix_caching': True,
                'runner': 'generate', 'async_scheduling': False, 'speculative_decoding': False,
                'cache_transfer': False, 'image_processor_kwargs': {'truncation': False}}
    image_binding = provenance.get('image_binding', {})
    actual_image = image_binding.get('image_id')
    pins = provenance.get('source_pins', {})
    canonical.require(isinstance(pins, dict), 'Source provenance must be an object')
    server_sources = [Path(path) for path in pins if Path(path).name == 'kev_vllm_research_server.py']
    recipe = server_sources[0].parent.parent if server_sources else Path(__file__).parent.parent
    provenance = {**provenance, 'observed_recipe_revision': canonical.git_head(recipe),
                  'recorded_core_commit': CORE_COMMIT, 'observed_source_pins': canonical.verify(pins),
                  'recorded_runtime_cell': expected,
                  'observed_model_artifact': acquired, 'recorded_model_artifact': {'repo': repo, 'revision': revision},
                  'recorded_model_directory': str(directory), 'recorded_parent_image': image,
                  'observed_image_id': actual_image, 'recorded_head_sha256': canonical.HEAD_SHA,
                  'recorded_plan_sha256': canonical.PLAN_SHA, 'recorded_quality_sha256': canonical.QUALITY_SHA}
    plan = canonical.read(canonical.PLAN)
    provenance['recorded_source8'] = plan['candidates']['lm-assisted']['source8']
    worker = health.get('worker_readout', {})
    loaded_sources = readout_sources(worker.get('expected_loaded_sources'))
    canonical.require(provenance.get('worker_readout', {}).get('expected_loaded_sources') == loaded_sources
                      and worker.get('required_mode') == 'query-lm-assisted'
                      and worker.get('status') in ('awaiting-worker-receipt', 'worker-verified'),
                      'Actual worker protocol binding differs')
    capabilities = health.get('capabilities', {})
    canonical.require(all(capabilities.get(name) is True for name in
                      ('typed_decisions', 'text_generation', 'shared_lm_assisted_head', 'trusted_still_images')),
                      'Required shared-runtime APIs are unavailable')
    if binding is not None:
        canonical.require(provenance == binding and worker.get('status') == 'worker-verified'
                          and {'query-lm-assisted', 'feature-only'} <= worker.get('verified_receipts', {}).keys(),
                          'Loaded provenance changed or actual worker reads were not verified')
    return provenance


def response_check(response, request, request_id):
    validated = VllmDecisionResponse.model_validate(response)
    canonical.require(validated.id == request_id and all(response.get(key) == request[key]
                      for key in ('model', 'primitive', 'head_id')), 'Response does not match the HTTP request')
    canonical.require(all(response.get(key) is value for key, value in canonical.FLAGS.items())
                      and validated.calibration_id == 'uncalibrated:T1'
                      and response.get('provenance') == BINDING, 'Response head/source/T1 provenance changed')
    scores = [score.model_dump() for score in validated.scores]
    ids = [candidate['id'] for candidate in request['candidates']]
    logits, raw, legacy = ([score[key] for score in scores] for key in
                          ('logit', 'raw_probability', 'calibrated_probability'))
    canonical.require([score['id'] for score in scores] == ids and all(math.isfinite(z) for z in logits)
                      and raw == legacy and all(math.isfinite(p) and 0 <= p <= 1 for p in raw)
                      and math.isclose(sum(raw), 1., rel_tol=0., abs_tol=1e-6), 'Invalid ordered raw-T1 scores')
    exps = [math.exp(z - max(logits)) for z in logits]
    canonical.require(all(math.isclose(p, e / sum(exps), rel_tol=1e-5, abs_tol=1e-6)
                      for p, e in zip(raw, exps))
                      and validated.selected_id == ids[max(range(len(ids)), key=logits.__getitem__)],
                      'Scores, softmax and selected option disagree')
    execution = validated.execution
    canonical.require(execution.mode == 'fresh'
                      and 0 <= execution.logical_prefix_tokens < execution.logical_query_tokens <= 262144
                      and execution.consumed_tokens is None and execution.replayed_tokens is None
                      and len(execution.requests) == len(ids) + 1,
                      'Do not fabricate native cache/consumption evidence for vLLM')
    expected_sources = readout_sources(BINDING['worker_readout']['expected_loaded_sources'])
    submitted = 0
    for index, receipt in enumerate(execution.requests):
        proof = receipt.get('fuzev_readout_receipt', {})
        labels = proof.get('baseline_token_ids')
        count = receipt.get('submitted_tokens')
        canonical.require(set(proof) == READOUT_FIELDS and type(count) is int and count > 0
                          and all(type(proof[name]) is int for name in ('schema_version', 'prompt_tokens',
                              'feature_readout_position', 'feature_width', 'generated_token_count', 'lm_projection_calls'))
                          and proof['schema_version'] == 1
                          and proof['request_id'] == receipt['request_id']
                          and isinstance(proof['worker_request_id'], str) and 1 <= len(proof['worker_request_id']) <= 1024
                          and proof['prompt_tokens'] == count and proof['feature_readout_position'] == count - 1
                          and proof['prompt_token_sha256'] == receipt['token_sha256']
                          and proof['feature_width'] == 2560 and proof['generated_token_count'] == 0
                          and proof['lm_projection_calls'] == (1 if index == 0 else 0)
                          and readout_sources(proof['worker_loaded_sources']) == expected_sources,
                          'Missing or incompatible actual request-local worker receipt')
        canonical.require(count <= 262144 and (count == execution.logical_query_tokens if index == 0
                          else count > execution.logical_query_tokens), 'Wrong query/candidate branch input lengths')
        canonical.require(isinstance(labels, list) and (len(labels) == len(ids) if index == 0 else labels == [])
                          and all(type(token) is int and token >= 0 for token in labels)
                          and len(set(labels)) == len(labels), 'Wrong query labels or candidate LM projection')
        submitted += count
    canonical.require(execution.submitted_tokens == submitted, 'Engine input-position totals disagree')
    return scores


def main():
    global MODEL, BINDING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-inference', required=True, action='store_true')
    parser.add_argument('--model', required=True, choices=tuple(CELLS))
    parser.add_argument('--timeout', type=float, default=120.)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume-from', type=Path, help='Reuse an inactive, compatible attempt into a new output directory')
    args = parser.parse_args()
    MODEL = args.model
    canonical.require(os.environ.get('FUZEV_ROOT') == str(canonical.ROOT), 'Explicit root required')
    canonical.require(math.isfinite(args.timeout) and 0 < args.timeout <= 3600, 'Invalid timeout')
    inputs = canonical.load_inputs()
    items, pins, original = inputs
    inputs = (items, {**pins, str(Path(__file__)): canonical.sha(__file__)}, original)
    canonical.MODEL, canonical.TASK = MODEL, 'validation.transfer.' + MODEL
    canonical.OUTPUT = (args.output_dir or canonical.REPORT / ('e2e-' + MODEL + '-v1')).resolve()
    canonical.require(canonical.OUTPUT.parent == canonical.REPORT
                      and canonical.OUTPUT.name.startswith('e2e-' + MODEL + '-v'), 'Use a new exclusive cell attempt')

    def bind_health(health, original, binding=None):
        global BINDING
        BINDING = health_check(health, original, binding)
        return BINDING

    return canonical.run(args.timeout, port=args.port, input_loader=lambda: inputs, request_mapper=wire_request,
                         health_validator=bind_health, response_validator=response_check,
                         task_parent='validation.transfer', task_title='Research CUDA HTTP E2E: ' + MODEL,
                         resume_from=args.resume_from)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        sys.exit(1)
