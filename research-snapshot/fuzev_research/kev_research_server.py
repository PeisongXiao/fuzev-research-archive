"""Explicit CUDA/raw-T1 research E2E, not a calibrated production head.

Keep the production ServingHead loader and its qualification checks intact.
Admission binds this experiment to the already-passing, frozen Kev test read.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import types

import torch
from safetensors.torch import load_file

from .native_service import NativeService
from .paths import ROOT
from .server import create_app
from .telemetry import MemoryGuard


REPORT = ROOT / 'reports/2026-10-10-kev-text-evaluation'
PLAN = REPORT / 'frozen-plan-v1.json'
QUALITY = REPORT / 'native-results-v1/result.json'
TASK = 'validation.kev-e2e'
ACQUISITION = ROOT / 'data/manifests/acquisition.json'
NATIVE_REPO = 'turboderp/Qwen3.8-Flash-Next-exl3'
MODEL_REVISIONS = {
    'native-3.05': '69e33439ae950f17bcbe95c98f117d80f759ab6d',
    'native-4.05': '55a732e0c4c3d4614bc42b68493bb930d9b02c0a',
    'native-5.05': '7cef615f7fd3681295b68848018876dfabc336c7',
}
HEAD_SHA = 'f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serving_identity_for(model_name, model, original, runtime):
    require(model_name in MODEL_REVISIONS, 'Unknown native cell')
    require(original['hidden_size'] == 2560 and original['cache_precision'] == 'float16'
            and original['feature_tap'] == 'final-mixer/post-collapse/before-lm-head',
            'Incompatible canonical native4 reference')
    directory = Path(model['local_snapshot'])
    metadata = read(directory / 'config.json')
    text = metadata.get('text_config', metadata)
    require(text.get('hidden_size') == 2560 and text.get('max_position_embeddings') == 262144
            and metadata.get('quantization_config', {}).get('quant_method') == 'exl3',
            'Acquired native configuration has incompatible geometry/quantization')
    return {'cell': model_name, 'model': model['repo'], 'revision': model['revision'],
            'recorded_model': NATIVE_REPO, 'recorded_revision': MODEL_REVISIONS[model_name],
            'local_snapshot': str(directory), 'encoding': original['encoding'],
            'feature_tap': original['feature_tap'], 'hidden_size': 2560, 'cache_precision': 'float16',
            'readout_runtime': dict(runtime), 'ple': 'NVMe mmap',
            'config_sha256': sha(directory / 'config.json'),
            'quantization_config_sha256': sha(directory / 'quantization_config.json')}


def memory_snapshot(phase, runtime):
    import psutil
    torch.cuda.synchronize(0)
    free, total = torch.cuda.mem_get_info(0)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'phase': phase, 'cuda_allocated_bytes': torch.cuda.memory_allocated(0),
            'cuda_reserved_bytes': torch.cuda.memory_reserved(0),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
            'cuda_free_bytes': free, 'cuda_total_bytes': total,
            'ram_rss_bytes': psutil.Process().memory_info().rss,
            'ram_peak_rss_bytes': peak * (1 if sys.platform == 'darwin' else 1024),
            'readout_runtime': dict(runtime)}


class ResearchHead:
    def __init__(self, backend, plan, quality):
        require(torch.cuda.is_available(), 'CUDA research target required')
        self.backend = backend
        case = plan['candidates']['lm-assisted']
        tree = Path(case['tree'])
        package = types.ModuleType('_kev_research_a')
        package.__path__ = [str(tree / 'fuzev_research')]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + '.shared_head')
        self.bundle = module.SharedDecisionHead(seed=43, key_gain=case['profile']['key_gain']).to('cuda:0')
        self.bundle.load_state_dict(load_file(case['weights'], device='cuda:0'), strict=True)
        self.bundle.eval().requires_grad_(False)
        require(sum(p.numel() for p in self.bundle.parameters()) == case['parameters'], 'Wrong shared head geometry')
        self.lm = next(module for module, _, _ in backend.model.fwd_modules if module.caps.get('logits_output'))
        self.labels = []
        for index in range(16):
            ids = backend.preparer.encode(' ' + chr(ord('A') + index))
            require(len(ids) == 1, 'Native label controls must be single tokens')
            self.labels.append(ids[0])
        self.head_id = 'reference'
        self.calibration_id = 'uncalibrated:T1'
        self.manifest_sha256 = sha(PLAN)
        self.weights_sha256 = sha(case['weights'])

    @torch.inference_mode()
    def score(self, primitive, query, candidates):
        require(query.shape == (2560,) and candidates.ndim == 2 and candidates.shape[1] == 2560,
                'Invalid original native features')
        query_cuda = query.detach().to(device='cuda:0', dtype=torch.float32).unsqueeze(0)
        keys_cuda = candidates.detach().to(device='cuda:0', dtype=torch.float32).unsqueeze(0)
        vocabulary = self.lm.forward(query_cuda.reshape(1, 1, 2560).half(), {})[0, -1]
        baseline = vocabulary[self.labels[:len(candidates)]].float().unsqueeze(0)
        logits = self.bundle(primitive, query_cuda, keys_cuda, baseline=baseline)[0]
        probabilities = torch.softmax(logits, dim=0)
        require(torch.isfinite(logits).all() and torch.isfinite(probabilities).all(), 'Nonfinite CUDA scores')
        raw = probabilities.cpu().tolist()
        # The legacy field is retained for schema compatibility, not a CAL claim.
        return logits.cpu().tolist(), raw, raw[:]


class ResearchService(NativeService):
    def decide(self, request, cancel, request_id):
        result = super().decide(request, cancel, request_id)
        return {**result, 'experimental': True, 'calibrated': False,
                'qualified': False, 'servable': False, 'head_execution_device': 'cuda:0',
                'head_sha256': self.head.weights_sha256,
                'qualification': 'research E2E only; CUDA head, raw T1; no production qualification'}


def build_research_service(backend, model_name, model, sources, config):
    plan, quality = read(PLAN), read(QUALITY)
    require(sha(PLAN) == quality['plan_sha256'] and quality['status'] == 'pass'
            and quality['comparison']['passed'] is True and quality['primary'] == 'lm-assisted'
            and quality['test_sha256'] == plan['test_sha256'], 'Frozen main text quality must pass before research E2E')
    require(quality['uncertainty']['index']['lm-assisted'][0] > plan['baseline_index_ci'][1], 'Kev quality comparison no longer passes')
    observed_pins = {path: sha(path) for path in plan['input_sha256']}
    acquired = read(ACQUISITION)['models'].get(model_name)
    require(acquired == model, 'Server cell differs from the acquired model metadata')
    original = read(plan['original_feature_identity'])
    serving = serving_identity_for(model_name, model, original, backend.readout_runtime)
    require(Path(backend.config.directory).resolve() == Path(model['local_snapshot']).resolve(),
            'Loaded native directory differs from acquired snapshot')
    require(backend.config.infer_params.ngram_stream_from_disk is True, 'NVMe PLE streaming must remain enabled')
    config.reference_feature_identity, config.serving_identity = original, serving
    from .encoding import identity
    require(identity() == original['encoding'], 'Research input encoding changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded_memory = [memory_snapshot('backbone-cache-loaded', backend.readout_runtime)]
    print(json.dumps({'model': model_name, 'loaded_memory': loaded_memory[0]}), flush=True)
    head = ResearchHead(backend, plan, quality)
    loaded_memory.append(memory_snapshot('backbone-cache-cuda-head-loaded', backend.readout_runtime))
    provenance = {'experimental': True, 'calibrated': False, 'qualified': False, 'servable': False,
                  'head_execution_device': 'cuda:0', 'learned_head_count': 1, 'head_lane': 'lm-assisted',
                  'head_sha256': head.weights_sha256, 'plan_sha256': sha(PLAN), 'quality_sha256': sha(QUALITY),
                  'readout_runtime': backend.readout_runtime, 'source_pins': config.source_pins,
                  'feature_identity': original, 'recipe_sources': sources,
                  'reference_feature_identity': original, 'serving_identity': serving,
                  'acquisition_manifest_sha256': sha(ACQUISITION), 'loaded_memory': loaded_memory,
                  'observed_source_pins': observed_pins, 'recorded_source_pins': plan['input_sha256'],
                  'recorded_head_sha256': HEAD_SHA, 'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                  'readout_math': {'hidden_size': 2560, 'head_dtype': 'float32',
                                   'lm_projection_input_dtype': 'float16', 'raw_temperature': 1.},
                  'calibration_id': head.calibration_id, 'raw_inference_temperature': 1.}
    guard = MemoryGuard(fraction=backend.readout_runtime['memory_fraction'])
    guard.start()
    service = ResearchService(backend, model_name, head, provenance, guard,
                              capacity=config.prefix_capacity, ttl=config.prefix_ttl_seconds)
    return service


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(MODEL_REVISIONS), default='native-4.05')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--memory-fraction', type=float, default=None, help='Optional host memory fraction (default: no cap)')
    parser.add_argument('--max-num-seqs', type=int, default=None, help='Native batch capacity (default: backend default)')
    args = parser.parse_args()
    model_name = args.model
    torch.set_num_threads(10)
    source_pins = {str(Path(__file__)): sha(__file__), str(ACQUISITION): sha(ACQUISITION)}
    for file in ('server.py', 'native_service.py', 'native.py', 'contracts.py', 'service_contracts.py',
                 'service_images.py', 'encoding.py', 'preparation.py'):
        path = Path(__file__).with_name(file)
        source_pins[str(path)] = sha(path)
    config = types.SimpleNamespace(prefix_capacity=8, prefix_ttl_seconds=900., source_pins=source_pins,
                                   reference_feature_identity=None, serving_identity=None)
    options = {} if args.max_num_seqs is None else {'batch_capacity': args.max_num_seqs, 'candidate_batch_size': args.max_num_seqs}
    app = create_app(model_name, **options, memory_fraction=args.memory_fraction,
                     service_config=config, service_factory=build_research_service)

    @app.get('/research-health')
    def health():
        service = app.state.service
        return {'status': 'ok' if app.state.ready else 'stopping', 'model': model_name,
                'provenance': service.provenance, 'readout_runtime': app.state.backend.readout_runtime,
                'memory': memory_snapshot('research-health', app.state.backend.readout_runtime),
                'host_memory_guard': service.guard.snapshot() if hasattr(service.guard, 'snapshot') else {'enabled': True},
                'capabilities': {'typed_decisions': True, 'trusted_still_images': True, 'shared_lm_assisted_head': True}}

    uvicorn.run(app, host=args.host, port=args.port, log_level='warning', access_log=False)


if __name__ == '__main__':
    main()
