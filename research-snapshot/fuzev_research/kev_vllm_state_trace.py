"""Opt-in, bounded synthetic recurrent-cache diagnostics; never alter tensors."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import re

from .encoding import token_hash


ENV = 'FUZEV_SYNTHETIC_STATE_TRACE'


class ReplayIntegrityError(ValueError):
    pass


def configuration(root):
    raw = os.environ.get(ENV)
    if not raw:
        return None
    path = Path(raw).resolve()
    value = json.loads(path.read_text())
    output = Path(value['output']).resolve()
    hashes = value['token_sha256']
    if (not path.is_relative_to(root / 'reports') or output.parent != path.parent
            or not isinstance(hashes, list) or len(hashes) != 6
            or any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes)
            or len(set(hashes)) != 6
            or type(value.get('max_readouts')) is not int or value['max_readouts'] != 12
            or type(value.get('private_moe_replay', False)) is not bool):
        raise ValueError('State trace requires six synthetic token hashes and a twelve-readout bound')
    layout = value.get('synthetic_layout')
    if layout is not None:
        suffixes = layout.get('suffix_sha256', [])
        if (layout.get('prefix_tokens') != 1980 or layout.get('max_generated_tokens') != 2
                or layout.get('prefix_sha256') != 'c7aa04d8cb6747cdfa462db059576ab2f7f675392ab748eff746a356345956de'
                or not isinstance(suffixes, list) or len(suffixes) != 6
                or any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in suffixes)
                or len(set(suffixes)) != 6):
            raise ValueError('Synthetic layout must bind the fixed 1980-token fixture and six exact suffixes')
    return path, output, frozenset(hashes)


def snapshot(tensor):
    import torch
    owned = tensor.detach().to(device='cpu').clone().contiguous()
    body = owned.view(torch.uint8).numpy().tobytes()
    return {'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
            'shape': list(owned.shape), 'dtype': str(owned.dtype)}


class StateTrace:
    def __init__(self, runner, spec, mamba_utils):
        self.runner, self.utils = runner, mamba_utils
        self.path, output, self.allowed = spec
        self.fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self.count = 0
        self.active = None
        self.pending = None
        self.seen = set()
        self.replayed = False
        self.layout = json.loads(self.path.read_text()).get('synthetic_layout')

    def install(self):
        original_execute = self.runner.execute_model
        original_preprocess = self.utils.preprocess_mamba
        original_forward = self.runner._model_forward
        signature = inspect.signature(original_preprocess)

        def execute(scheduler_output, *args, **kwargs):
            if self.active is not None:
                raise ValueError('Synthetic state trace requires non-overlapping worker execution')
            self.active, self.pending = scheduler_output, None
            try:
                result = original_execute(scheduler_output, *args, **kwargs)
                if self.pending is not None:
                    self.finish()
                return result
            finally:
                self.active, self.pending = None, None

        def preprocess(*args, **kwargs):
            traced = self.active is not None and args and args[0] is self.active
            if traced:
                self.begin(signature.bind(*args, **kwargs).arguments)
            result = original_preprocess(*args, **kwargs)
            if traced and self.pending is not None:
                for row in self.pending['snapshots']:
                    row['source_after_copy'] = snapshot(row.pop('_source_tensor_after_copy'))
                    row['destination_after_copy'] = snapshot(row.pop('_destination_tensor'))
            return result

        def forward(*args, **kwargs):
            if self.pending is not None:
                bound = inspect.signature(original_forward).bind(*args, **kwargs).arguments
                inputs = {**bound, **bound.get('model_kwargs', {})}
                self.pending['forward'] = {name: snapshot(inputs[name]) for name in
                    ('input_ids', 'positions', 'query_start_loc', 'ngram_context') if inputs.get(name) is not None}
                from vllm.forward_context import get_forward_context
                context = get_forward_context()
                self.pending['batch_shape'] = {'input_tokens': int(inputs['input_ids'].shape[0]),
                    'batch_descriptor': repr(context.batch_descriptor),
                    'cudagraph_runtime_mode': str(context.cudagraph_runtime_mode)}
                self.pending['attention_metadata'] = {}
                for name in self.pending['representative_layers']:
                    meta = context.attn_metadata[name]
                    values = {}
                    for field in ('num_prefills', 'num_prefill_tokens', 'num_decodes', 'num_actual_tokens',
                                  'has_initial_state', 'prefill_has_initial_state', 'has_initial_states_p', 'has_initial_states_d',
                                  'non_spec_state_indices_tensor', 'prefill_state_indices', 'state_indices_tensor',
                                  'non_spec_query_start_loc', 'prefill_query_start_loc', 'query_start_loc'):
                        value = getattr(meta, field, None)
                        if value is not None:
                            values[field] = value.detach().cpu().tolist() if hasattr(value, 'detach') else value
                    self.pending['attention_metadata'][name] = values
            return original_forward(*args, **kwargs)

        self.runner.execute_model = execute
        self.runner._model_forward = forward
        self.utils.preprocess_mamba = preprocess
        if json.loads(self.path.read_text()).get('private_moe_replay') is True:
            from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe
            original_moe = flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe

            def moe(*args, **kwargs):
                result = original_moe(*args, **kwargs)
                if (self.pending is not None and not self.replayed and not args
                        and len(self.pending['requests']) == 1
                        and 458 <= kwargs['input'].shape[0] <= 460):
                    self.replayed = True
                    try:
                        report = private_moe_replay(original_moe, kwargs)
                    except ReplayIntegrityError:
                        raise
                    except Exception as error:
                        report = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                                  'scope': 'Private diagnostic failed; no correctness qualification'}
                    path = self.path.with_name('moe-replay.json')
                    body = (json.dumps(report, allow_nan=False, ensure_ascii=True) + '\n').encode('ascii')
                    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as handle:
                        handle.write(body)
                        handle.flush()
                        os.fsync(handle.fileno())
                return result

            flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe = moe

    def begin(self, args):
        batch, requests = args['input_batch'], args['requests']
        selected = []
        for index, rid in enumerate(batch.req_ids):
            request = requests[rid]
            tokens = request.prompt_token_ids
            if request.pooling_params is None or tokens is None or len(tokens) > 32768:
                continue
            digest = token_hash(tuple(tokens))
            matching_layout = (self.layout is not None and len(tokens) <= 2048
                and token_hash(tokens[:1980]) == self.layout['prefix_sha256']
                and any(token_hash(tokens[1980 + n:]) in self.layout['suffix_sha256'] for n in range(3)))
            if digest in self.allowed or matching_layout:
                if rid in self.seen or self.count >= 12:
                    raise ValueError('Synthetic state trace readout budget exhausted')
                self.seen.add(rid)
                self.count += 1
                selected.append((index, rid, request, digest))
        if not selected:
            return
        if len(selected) != len(batch.req_ids) or self.pending is not None:
            raise ValueError('Synthetic trace must not share a worker batch with unlisted work')
        buffers = args['copy_bufs']
        block_size = buffers.mamba_spec.block_size
        if buffers.mamba_spec.num_speculative_blocks or args.get('align_ctx') is not None:
            raise ValueError('Synthetic state trace requires the unchanged no-speculation align path')
        record = {'requests': [], 'snapshots': [], 'mapping': [], 'representative_layers': [],
                  'total_scheduled_tokens': self.active.total_num_scheduled_tokens,
                  'batch_request_ids': list(batch.req_ids)}
        self.pending = record
        groups = args['kv_cache_config'].kv_cache_groups
        context = args['forward_context']
        for index, rid, request, digest in selected:
            scheduled = self.active.num_scheduled_tokens[rid]
            previous = args['mamba_state_idx'].get(rid, (request.num_computed_tokens - 1) // block_size)
            current = (request.num_computed_tokens + scheduled + block_size - 1) // block_size - 1
            record['requests'].append({'request_id': rid, 'token_sha256': digest,
                'prompt_tokens': len(request.prompt_token_ids), 'computed_tokens': request.num_computed_tokens,
                'scheduled_tokens': scheduled, 'previous_column': previous, 'running_column': current,
                'accepted_tokens': int(batch.num_accepted_tokens_cpu[index])})
            representatives = set()
            for gid in buffers.mamba_group_ids:
                group = groups[gid]
                for name in group.layer_names:
                    spec = self.utils._get_mamba_spec_for_layer(group, name)
                    kind = str(spec.mamba_type)
                    source_id = request.block_ids[gid][previous] if previous >= 0 else None
                    destination_id = request.block_ids[gid][current]
                    record['mapping'].append({'request_id': rid, 'group_id': gid, 'layer': name,
                        'mamba_type': kind, 'source_id': source_id, 'destination_id': destination_id})
                    if kind in representatives:
                        continue
                    representatives.add(kind)
                    if name not in record['representative_layers']:
                        record['representative_layers'].append(name)
                    if source_id is None:
                        continue
                    for state_index, state in enumerate(context[name].kv_cache):
                        source, destination = state[source_id], state[destination_id]
                        record['snapshots'].append({'request_id': rid, 'group_id': gid, 'layer': name,
                            'mamba_type': kind, 'state_index': state_index,
                            'source_id': source_id, 'destination_id': destination_id,
                            'source_ptr': source.data_ptr(), 'destination_ptr': destination.data_ptr(),
                            'source_before_copy': snapshot(source), '_source_tensor_after_copy': source,
                            '_source_tensor_after_forward': source, '_destination_tensor': destination})

    def finish(self):
        for row in self.pending['snapshots']:
            row['source_after_forward'] = snapshot(row.pop('_source_tensor_after_forward'))
            row['source_unchanged'] = (row['source_before_copy'] == row['source_after_copy'] == row['source_after_forward'])
            row['copy_equal'] = row['source_before_copy'] == row['destination_after_copy']
        body = (json.dumps(self.pending, allow_nan=False, ensure_ascii=True) + '\n').encode('ascii')
        if os.write(self.fd, body) != len(body):
            raise ValueError('Incomplete synthetic state trace')
        os.fsync(self.fd)


def install(worker, root):
    spec = configuration(root)
    if spec is None:
        return None
    if getattr(worker, '_fuzev_state_trace', None) is not None:
        raise ValueError('Synthetic state trace is startup-only')
    from vllm.v1.worker import mamba_utils
    trace = StateTrace(worker.model_runner, spec, mamba_utils)
    trace.install()
    worker._fuzev_state_trace = trace
    return {'config': str(spec[0]), 'sha256': hashlib.sha256(spec[0].read_bytes()).hexdigest(),
            'output': str(spec[1]), 'max_readouts': 12,
            'scope': 'Synthetic synchronized state hashes only; timing perturbed; tensors unchanged'}


def private_moe_replay(original, arguments):
    """Compare private outputs with frozen inputs and explicit tactics or default dispatch."""
    import torch
    from flashinfer.autotuner import AutoTuner

    def freeze(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, list):
            return [freeze(item) for item in value]
        return value

    weights = {name: arguments[name] for name in ('fc1_expert_weights', 'fc2_expert_weights')}
    weight_hashes = {name: snapshot(value) for name, value in weights.items()}
    frozen = {name: (value if name in weights else freeze(value)) for name, value in arguments.items()
              if name != 'output'}
    tensor_inputs = {name: value for name, value in frozen.items() if isinstance(value, torch.Tensor) and name not in weights}
    tensor_inputs.update({'quant_scale_' + str(i): value for i, value in enumerate(frozen['quant_scales'])
                          if isinstance(value, torch.Tensor)})
    before = {name: snapshot(value) for name, value in tensor_inputs.items()}
    modes = []
    tuner = AutoTuner.get()
    choose = tuner.choose_one
    try:
        for fused in (True, False):
            profiles, outputs = {}, []
            mode = {'use_fused_finalize': fused, 'status': 'failed'}
            modes.append(mode)

            def record_tactic(*args, **kwargs):
                result = choose(*args, **kwargs)
                index = kwargs.get('gemm_idx')
                if index in (1, 2):
                    profiles[index] = result[1]
                return result

            call = {**frozen, 'use_fused_finalize': fused}
            call.pop('profile_ids', None)
            try:
                tuner.choose_one = record_tactic
                try:
                    original(**call, output=torch.empty_like(arguments['output']))
                    torch.cuda.synchronize()
                finally:
                    tuner.choose_one = choose
                if set(profiles) != {1, 2} or any(type(value) is not int or value < -1 for value in profiles.values()):
                    raise ValueError('Private replay requires observed valid GEMM tactics for each finalization mode')
                default = -1 in profiles.values()
                call['profile_ids'] = [-1, -1] if default else [profiles[1], profiles[2]]
                mode.update(tactic_mode='default_dispatch' if default else 'explicit',
                            profile_ids=call['profile_ids'])
                for _ in range(4):
                    output = torch.empty_like(arguments['output'])
                    original(**call, output=output)
                    outputs.append(output.detach().cpu().float().clone())
                mode['status'] = 'complete'
            except ReplayIntegrityError:
                raise
            except Exception as error:
                mode.update(error_type=type(error).__name__, error=str(error))
                break
            finally:
                mode['observed_profile_ids'] = [profiles.get(1), profiles.get(2)]
                mode['output_hashes'] = [snapshot(value)['sha256'] for value in outputs]
                deltas = [(value - outputs[0]) for value in outputs[1:]]
                mode['max_abs_deltas'] = [float(delta.abs().max()) for delta in deltas]
                mode['rms_deltas'] = [float(delta.square().mean().sqrt()) for delta in deltas]
                if outputs:
                    mode['shape'] = list(outputs[0].shape)
    finally:
        tuner.choose_one = choose
        after = {name: snapshot(value) for name, value in tensor_inputs.items()}
        if before != after or weight_hashes != {name: snapshot(value) for name, value in weights.items()}:
            raise ReplayIntegrityError('Private MoE replay mutated frozen inputs or loaded weights')
    return {'status': 'complete' if all(mode['status'] == 'complete' for mode in modes) else 'failed',
            'scope': 'Private fixed-input kernel diagnostic only; original forward output unchanged; no correctness qualification',
            'tactic_scope': 'Explicit holds observed tactic IDs; default_dispatch holds dispatch policy, not actual selected tactic IDs',
            'inputs': before, 'weights': weight_hashes,
            'inputs_unchanged': True, 'modes': modes}
