"""MAIN-only synthetic CUDA admission: PYTHONPATH=$PWD python3 scripts/check_shared_lm_fused_adamw.py.

No datasets, feature reads, backbone, checkpoint loads or output files. Fixed 24
steps cover epochs 1-6, including zero gradients, decay and the epoch-5 boundary.
Tolerances are predeclared in the training protocol and cannot be CLI-overridden.
CUDA event timings exclude clipping/comparison and the first four warmup steps;
they are a short optimizer-only diagnostic, not an end-to-end speedup claim.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from fuzev_research import shared_train as training
from fuzev_research.shared_head import PARAMETER_COUNT, SharedDecisionHead


STEPS = 24
STEPS_PER_EPOCH = 4
ZERO_GRADIENT_STEPS = (1, 6, 11, 16, 21)


@torch.no_grad()
def compare_updates(device, *, hidden_size=2560, projection_size=512):
    device = torch.device(device)
    candidate = SharedDecisionHead(seed=43, hidden_size=hidden_size, projection_size=projection_size).to(device)
    reference = copy.deepcopy(candidate)
    actual = training.make_optimizer(candidate)
    ref_named = dict(reference.named_parameters())
    expected = torch.optim.AdamW([
        {'params': [ref_named[name] for name in training.OPTIMIZER_GROUP_PARAMETERS['scorer_and_gate']], 'lr': 1e-5},
        {'params': [ref_named[name] for name in training.OPTIMIZER_GROUP_PARAMETERS['query_adapter']], 'lr': 1e-4},
        {'params': [ref_named[name] for name in training.OPTIMIZER_GROUP_PARAMETERS['channel_gate']], 'lr': 1e-4},
    ], weight_decay=.01, fused=False, foreach=True)
    named = dict(candidate.named_parameters())
    assert len(named) == 12 and len({id(p) for p in named.values()}) == 12
    generator = torch.Generator(device=device).manual_seed(119)
    templates = {name: torch.randn(p.shape, device=device, generator=generator) for name, p in named.items()}
    max_errors = {name: {'parameters': 0., 'moments': 0.} for name in named}
    events = {'fused': [], 'reference_foreach': []}
    clipped_steps, rates = [], []
    tolerance = training.optimization_protocol()['numerical_admission']['tolerances']

    def check(left, right, kind, label):
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f'Nonfinite {label}')
        torch.testing.assert_close(left, right, **tolerance[kind], msg=label)
        return float((left - right).abs().max())

    for step in range(1, STEPS + 1):
        epoch = (step - 1) // STEPS_PER_EPOCH + 1
        base_lr = 1e-5 if epoch < 5 else 3e-6
        actual_rates = training.set_optimizer_learning_rates(actual, candidate, epoch)
        assert actual_rates == {'scorer_and_gate': base_lr, 'query_adapter': base_lr * 10, 'channel_gate': base_lr * 10}
        expected.param_groups[0]['lr'], expected.param_groups[1]['lr'] = base_lr, base_lr * 10
        expected.param_groups[2]['lr'] = base_lr * 10
        rates.append({'step': step, 'epoch': epoch, **actual_rates})
        actual.zero_grad(set_to_none=True)
        expected.zero_grad(set_to_none=True)
        for name, parameter in named.items():
            scale = (1e-4 if step % 2 else 10.) * (-1 if step % 3 else 1)
            gradient = (torch.zeros_like(parameter) if step in ZERO_GRADIENT_STEPS else
                        (templates[name] + (step % 5 - 2) * .01) * scale)
            parameter.grad = gradient.clone()
            ref_named[name].grad = gradient.clone()
        # Exactly one global clip per head, never separate clips for the three groups.
        norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        if float(norm) > 1.:
            clipped_steps.append(step)
        gradients = {name: p.grad.clone() for name, p in named.items()}
        decay_only = {name: p.detach().clone() for name, p in named.items()} if step == 1 else {}
        ordered = [('fused', actual), ('reference_foreach', expected)]
        for label, optimizer in ordered if step % 2 else reversed(ordered):
            if device.type == 'cuda' and step > STEPS_PER_EPOCH:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                optimizer.step()
                end.record()
                events[label].append((begin, end))
            else:
                optimizer.step()
        for name, parameter in named.items():
            label = f'step={step} epoch={epoch} parameter={name}'
            max_errors[name]['parameters'] = max(max_errors[name]['parameters'],
                check(parameter, ref_named[name], 'parameters', label))
            assert torch.equal(parameter.grad, gradients[name]), f'Gradient changed: {label}'
            assert torch.equal(ref_named[name].grad, gradients[name]), f'Reference gradient changed: {label}'
            if step == 1:
                rate = base_lr * (10 if name.startswith(('adapter_', 'channel_')) else 1)
                check(parameter, decay_only[name] * (1 - rate * .01), 'parameters', f'zero-gradient decay: {label}')
            assert actual.state[parameter]['step'].item() == expected.state[ref_named[name]]['step'].item() == step
            for key in ('exp_avg', 'exp_avg_sq'):
                max_errors[name]['moments'] = max(max_errors[name]['moments'],
                    check(actual.state[parameter][key], expected.state[ref_named[name]][key], 'moments', f'{label} {key}'))
    assert clipped_steps and len(clipped_steps) < STEPS - len(ZERO_GRADIENT_STEPS)
    timings = None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        timings = {name: sum(start.elapsed_time(end) for start, end in pairs) / 1000 for name, pairs in events.items()}
    return {'passed': True, 'scope': 'synthetic optimizer only; no data or backbone',
            'torch': torch.__version__, 'device': str(device),
            'parameter_count': sum(p.numel() for p in candidate.parameters()),
            'parameters': list(named), 'steps': STEPS, 'global_clips_per_head_per_step': 1,
            'zero_gradient_steps': list(ZERO_GRADIENT_STEPS), 'clipped_steps': clipped_steps,
            'group_learning_rates': rates, 'maximum_absolute_errors': max_errors,
            'optimization': training.optimization_protocol(), 'source': training.source_identity(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'optimizer_cuda_event_seconds': timings, 'timed_steps_per_optimizer': STEPS - STEPS_PER_EPOCH}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    admission = training.optimization_protocol()['numerical_admission']
    if not torch.cuda.is_available():
        raise ValueError('MAIN-only CUDA comparator requires CUDA; no CPU fallback admission')
    torch.set_num_threads(1)
    result = compare_updates('cuda:0')
    result['recorded_torch_version'] = admission['torch_version']
    result['torch_version'] = str(torch.__version__)
    result['cuda_version'] = torch.version.cuda
    assert result['parameter_count'] == PARAMETER_COUNT
    result['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(json.dumps({'passed': False, 'error': str(exc)}), flush=True)
        raise SystemExit(1)
