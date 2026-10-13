"""CPU-only synthetic trace fixtures; no real vLLM, GPU, models or datasets."""

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from fuzev_research import kev_vllm_state_trace as trace


def config_file(root):
    directory = root / 'reports'
    directory.mkdir(exist_ok=True)
    path = directory / 'synthetic-trace.json'
    value = {'output': str(directory / 'states.jsonl'), 'max_readouts': 12,
             'token_sha256': [trace.token_hash(tuple(range(i * 16, i * 16 + 8))) for i in range(6)]}
    path.write_text(json.dumps(value))
    return path, value


@pytest.fixture(autouse=True)
def disabled_by_default(monkeypatch):
    monkeypatch.delenv(trace.ENV, raising=False)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    path, value = config_file(tmp_path)
    monkeypatch.setenv(trace.ENV, str(path))
    cache = {name: NS(kv_cache=(torch.arange(24, dtype=torch.float32, device='cpu').reshape(4, 2, 3) + i * 100,))
             for i, name in enumerate(('ssm', 'conv'))}
    batch = NS(req_ids=['known'], num_accepted_tokens_cpu=torch.ones(1, dtype=torch.int32, device='cpu'))
    requests = {'known': NS(prompt_token_ids=list(range(8)), pooling_params=NS(),
                            num_computed_tokens=4, block_ids=[[0, 1, 2, 3]])}
    groups = NS(kv_cache_groups=[NS(layer_names=['ssm', 'conv'])])
    buffers = NS(mamba_group_ids=[0], mamba_spec=NS(block_size=4, num_speculative_blocks=0))
    indices, calls = {}, []
    meta = NS(num_prefills=1, num_prefill_tokens=4, num_decodes=0, num_actual_tokens=4,
              has_initial_states_p=torch.tensor([True], device='cpu'),
              has_initial_states_d=torch.tensor([False], device='cpu'),
              prefill_state_indices=torch.tensor([1], device='cpu'),
              query_start_loc=torch.tensor([0, 4], device='cpu'))
    forward_context = NS(attn_metadata={name: meta for name in cache},
                         batch_descriptor=NS(num_tokens=4, num_reqs=1), cudagraph_runtime_mode='NONE')
    forward_module = ModuleType('vllm.forward_context')
    forward_module.get_forward_context = lambda: forward_context
    monkeypatch.setitem(sys.modules, 'vllm.forward_context', forward_module)

    def preprocess(scheduler_output, kv_cache_config, cache_config, mamba_state_idx,
                   input_batch, requests, forward_context, mamba_state_copy_funcs,
                   copy_bufs, align_ctx=None):
        calls.append('preprocess')
        for rid in input_batch.req_ids:
            request = requests[rid]
            previous = mamba_state_idx.get(rid, (request.num_computed_tokens - 1) // 4)
            current = (request.num_computed_tokens + scheduler_output.num_scheduled_tokens[rid] + 3) // 4 - 1
            mamba_state_idx[rid] = current
            if previous >= 0 and previous != current:
                for layer in forward_context.values():
                    layer.kv_cache[0][request.block_ids[0][current]].copy_(
                        layer.kv_cache[0][request.block_ids[0][previous]])

    utils = NS(preprocess_mamba=preprocess, _get_mamba_spec_for_layer=lambda group, name: NS(mamba_type=name))

    class Runner:
        input_ids = torch.arange(4, dtype=torch.int64, device='cpu')
        positions = torch.arange(4, 8, dtype=torch.int64, device='cpu')
        query_start_loc = torch.tensor([0, 4], device='cpu')
        ngram_context = torch.arange(12, device='cpu').reshape(3, 4).t()

        def _model_forward(self, input_ids=None, positions=None, intermediate_tensors=None,
                           inputs_embeds=None, **model_kwargs):
            calls.append('forward')
            for rid in batch.req_ids:
                for layer in cache.values():
                    layer.kv_cache[0][requests[rid].block_ids[0][indices[rid]]].add_(1)
            return input_ids.float() * 2 + positions

        def execute_model(self, scheduler_output, *args, **kwargs):
            calls.append('execute')
            utils.preprocess_mamba(scheduler_output, groups, NS(enable_prefix_caching=True),
                                   indices, batch, requests, cache, {}, buffers, align_ctx=None)
            return self._model_forward(self.input_ids, self.positions,
                                       query_start_loc=self.query_start_loc, ngram_context=self.ngram_context)

    runner = Runner()
    state = trace.StateTrace(runner, trace.configuration(tmp_path), utils)
    state.install()
    result = NS(runner=runner, state=state, utils=utils, cache=cache, batch=batch,
                requests=requests, buffers=buffers, indices=indices, calls=calls,
                context=forward_context, output=Path(value['output']))
    result.run = lambda: runner.execute_model(NS(num_scheduled_tokens={rid: 4 for rid in batch.req_ids},
                                                 total_num_scheduled_tokens=4 * len(batch.req_ids)))
    result.rows = lambda: [json.loads(line) for line in result.output.read_text().splitlines()]
    try:
        yield result
    finally:
        os.close(state.fd)


@pytest.mark.parametrize('change', ['count', 'duplicate', 'nonhex', 'nonstr', 'bool_bound', 'wrong_bound', 'output'])
def test_configuration_guards(tmp_path, monkeypatch, change):
    path, value = config_file(tmp_path)
    if change == 'count':
        value['token_sha256'].pop()
    elif change == 'duplicate':
        value['token_sha256'][1] = value['token_sha256'][0]
    elif change == 'nonhex':
        value['token_sha256'][0] = 'G' * 64
    elif change == 'nonstr':
        value['token_sha256'][0] = {}
    elif change == 'bool_bound':
        value['max_readouts'] = True
    elif change == 'wrong_bound':
        value['max_readouts'] = 13
    else:
        value['output'] = str(tmp_path / 'outside.jsonl')
    path.write_text(json.dumps(value))
    monkeypatch.setenv(trace.ENV, str(path))
    with pytest.raises(ValueError):
        trace.configuration(tmp_path)
    assert not Path(value['output']).exists()


def test_configuration_rejects_config_outside_reports(tmp_path, monkeypatch):
    path, _ = config_file(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_bytes(path.read_bytes())
    monkeypatch.setenv(trace.ENV, str(outside))
    with pytest.raises(ValueError):
        trace.configuration(tmp_path)


def test_disabled_install_never_imports_vllm_or_installs_hooks(tmp_path, monkeypatch):
    worker = NS(model_runner=NS(execute_model=object(), _model_forward=object()))
    before = vars(worker.model_runner).copy()
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker', None)
    assert trace.install(worker, tmp_path) is None
    assert vars(worker.model_runner) == before
    assert not hasattr(worker, '_fuzev_state_trace')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64])
def test_snapshot_is_owned_noncontiguous_cpu_bytes_without_mutation(dtype):
    tensor = torch.arange(12, dtype=dtype, device='cpu').reshape(3, 4).t()
    original, version = tensor.clone(), tensor._version
    value = trace.snapshot(tensor)
    body = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    assert value == {'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
                     'shape': [4, 3], 'dtype': str(dtype)}
    assert torch.equal(tensor, original) and tensor._version == version
    tensor.add_(1)
    assert trace.snapshot(tensor)['sha256'] != value['sha256']


def test_trace_preserves_original_forward_math_and_source_tensors(rig):
    sources = {name: layer.kv_cache[0][0].clone() for name, layer in rig.cache.items()}
    inputs = {name: getattr(rig.runner, name).clone() for name in ('input_ids', 'positions', 'query_start_loc', 'ngram_context')}
    result = rig.run()
    assert torch.equal(result, inputs['input_ids'].float() * 2 + inputs['positions'])
    assert rig.calls == ['execute', 'preprocess', 'forward']
    for name, layer in rig.cache.items():
        assert torch.equal(layer.kv_cache[0][0], sources[name])
        assert torch.equal(layer.kv_cache[0][1], sources[name] + 1)
    for name, before in inputs.items():
        assert torch.equal(getattr(rig.runner, name), before)
    row = rig.rows()[0]
    assert len(row['snapshots']) == 2
    assert all(entry['copy_equal'] and entry['source_unchanged'] for entry in row['snapshots'])
    assert all(entry['source_ptr'] != entry['destination_ptr'] for entry in row['snapshots'])
    assert stat.S_IMODE(rig.output.stat().st_mode) == 0o600
    assert rig.state.active is None and rig.state.pending is None


def test_forward_geometry_ple_hashes_and_actual_metadata_are_recorded(rig):
    assert not hasattr(rig.context, 'num_tokens')
    rig.run()
    row = rig.rows()[0]
    assert row['batch_shape']['input_tokens'] == 4
    assert row['total_scheduled_tokens'] == 4
    for name in ('input_ids', 'positions', 'query_start_loc', 'ngram_context'):
        assert row['forward'][name] == trace.snapshot(getattr(rig.runner, name))
    for name in ('ssm', 'conv'):
        assert row['attention_metadata'][name]['has_initial_states_p'] == [True]
        assert row['attention_metadata'][name]['has_initial_states_d'] == [False]
        assert row['attention_metadata'][name]['prefill_state_indices'] == [1]
    rig.context.attn_metadata['ssm'].has_initial_states_p.fill_(False)
    assert row['attention_metadata']['ssm']['has_initial_states_p'] == [True]


def test_fresh_readout_keeps_attention_metadata_even_without_copy_source(rig):
    rig.requests['known'].num_computed_tokens = 0
    rig.run()
    row = rig.rows()[0]
    assert row['snapshots'] == []
    assert all(mapping['source_id'] is None for mapping in row['mapping'])
    assert set(row['attention_metadata']) == {'ssm', 'conv'}


def test_unknown_only_batch_keeps_original_fastpath_without_trace(rig):
    rig.requests['known'].prompt_token_ids = [999]
    assert torch.equal(rig.run(), torch.tensor([4., 7., 10., 13.]))
    assert rig.rows() == [] and rig.state.count == 0


@pytest.mark.parametrize('generated', [[], [7], [7, 8]])
def test_fixed_synthetic_prefix_and_suffix_accept_actual_generated_token_count(rig, generated):
    prefix, suffix = [3] * 1980, [5, 6]
    rig.state.layout = {'prefix_sha256': trace.token_hash(prefix),
                        'suffix_sha256': [trace.token_hash(suffix)]}
    rig.requests['known'].prompt_token_ids = prefix + generated + suffix
    rig.run()
    assert len(rig.rows()) == 1 and rig.state.count == 1


@pytest.mark.parametrize('change', ['prefix', 'suffix', 'three_generated'])
def test_synthetic_layout_does_not_admit_unknown_work(rig, change):
    prefix, suffix = [3] * 1980, [5, 6]
    rig.state.layout = {'prefix_sha256': trace.token_hash(prefix),
                        'suffix_sha256': [trace.token_hash(suffix)]}
    tokens = prefix + [7] + suffix
    if change == 'prefix':
        tokens[0] = 4
    elif change == 'suffix':
        tokens[-1] = 9
    else:
        tokens = prefix + [7, 8, 9] + suffix
    rig.requests['known'].prompt_token_ids = tokens
    rig.run()
    assert rig.rows() == [] and rig.state.count == 0


def test_unknown_mixed_batch_is_rejected_before_copy_or_forward(rig):
    rig.batch.req_ids.append('unknown')
    rig.requests['unknown'] = NS(prompt_token_ids=[999], pooling_params=NS())
    before = {name: layer.kv_cache[0].clone() for name, layer in rig.cache.items()}
    with pytest.raises(ValueError, match='unlisted'):
        rig.run()
    assert rig.calls == ['execute'] and rig.rows() == []
    assert all(torch.equal(layer.kv_cache[0], before[name]) for name, layer in rig.cache.items())
    assert rig.state.active is None and rig.state.pending is None


def test_seen_request_and_thirteenth_readout_are_rejected(rig):
    for index in range(12):
        rid = 'synthetic_' + str(index)
        rig.requests[rid] = rig.requests['known']
        rig.batch.req_ids[:] = [rid]
        rig.indices.clear()
        rig.run()
    assert rig.state.count == 12 and len(rig.rows()) == 12
    with pytest.raises(ValueError, match='budget'):
        rig.run()
    rig.requests['thirteenth'] = rig.requests['known']
    rig.batch.req_ids[:] = ['thirteenth']
    with pytest.raises(ValueError, match='budget'):
        rig.run()
    assert len(rig.rows()) == 12


def test_speculative_copy_path_is_rejected_before_original_math(rig):
    rig.buffers.mamba_spec.num_speculative_blocks = 1
    with pytest.raises(ValueError, match='no-speculation'):
        rig.run()
    assert rig.calls == ['execute'] and rig.rows() == []


def test_trace_output_is_exclusive_and_cannot_overwrite_prior_evidence(rig):
    with pytest.raises(FileExistsError):
        trace.StateTrace(rig.runner, (rig.state.path, rig.output, rig.state.allowed), rig.utils)


def test_disabled_capture_cleanup_does_not_install_trace_or_import_server(monkeypatch):
    from fuzev_research import kev_vllm_capture as capture
    cuda = NS(is_initialized=lambda: True, current_device=lambda: 0, mem_get_info=lambda device: (100, 200),
              memory_allocated=lambda device: 10, memory_reserved=lambda device: 20,
              max_memory_allocated=lambda device: 10, max_memory_reserved=lambda device: 20,
              synchronize=Mock(), empty_cache=Mock())
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=cuda))
    monkeypatch.setitem(sys.modules, 'fuzev_research.kev_vllm_research_server', None)
    monkeypatch.setattr(capture, 'worker_observation', lambda worker: {'synthetic': True})
    installer = Mock(side_effect=AssertionError('disabled trace must not install'))
    monkeypatch.setattr(trace, 'install', installer)
    result = capture.RuntimeObservationWorkerExtension().fuzev_runtime_observation(release_unused_startup_cache=True)
    assert 'synthetic_state_trace' not in result
    assert result['startup_allocator_cleanup']['before']['allocated_bytes'] == 10
    installer.assert_not_called()
