"""Data-free CUDA proof of gold-gradient and actual-displacement observation."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--observer-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    observer_path = source.with_name('probe_mature_direction.py')
    require(sha(source) == args.worker_sha256 and sha(observer_path) == args.observer_sha256,
            'Source SHA mismatch before import')
    require(not args.output.exists(), 'Never overwrite a numerical proof')
    spec = importlib.util.spec_from_file_location('_direction_measurement', observer_path)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator(device='cuda').manual_seed(20261010)
    query = torch.randn((60, 4), generator=generator, device='cuda')
    masks = torch.tensor([[True, True, False, False], [True, True, True, False],
                          [True, True, True, True], [True, True, True, True]] * 15, device='cuda')
    targets = torch.tensor([[1., 0., 0., 0.], [0., .5, .5, 0.],
                            [.25, .25, .25, .25], [.1, .2, .3, .4]] * 15, device='cuda')
    rows = [{'id': str(i), 'primitive': 'choice'} for i in range(60)]
    grouped = {f'synthetic/choice/track-{i}': rows[i * 4:i * 4 + 4] for i in range(15)}
    factors = {row['id']: .25 for row in rows}
    class Cache:
        data = {'query': query, 'mask': masks, 'targets': targets}

        def __init__(self):
            self.rows = rows

        def batch(self, selected):
            indices = torch.tensor([int(row['id']) for row in selected], device='cuda')
            return {'query': query[indices], 'mask': masks[indices], 'targets': targets[indices]}
    cache = Cache()
    training = SimpleNamespace(shared_logits=lambda model, batch: model(batch['query']))
    with torch.device('cuda'):
        first = torch.nn.Linear(4, 4)
        second = torch.nn.Linear(4, 4)
    second.load_state_dict(first.state_dict())
    first.train()
    second.train()
    snapshots = [query.clone(), masks.clone(), targets.clone()]
    snapshot = operator.frozen_snapshot(torch, first)
    losses, gradients, _ = operator.gold_panel(torch, training, first, cache, grouped, factors, gradients=True)
    operator.assert_unchanged(torch, first, snapshot)
    errors = []
    for track, selected in grouped.items():
        batch = cache.batch(selected)
        logits = first(batch['query']).detach()
        probability = logits.masked_fill(~batch['mask'], -torch.inf).softmax(1)
        y = batch['targets']
        derivative = (probability * y.sum(1, keepdim=True) - y) / batch['mask'].sum(1).float().log()[:, None] / 4
        independent = [derivative.double().T @ batch['query'].double(), derivative.double().sum(0)]
        for actual, expected in zip(gradients[track], independent):
            errors.append(float((actual.double() - expected).abs().max()))
            require(torch.allclose(actual.double(), expected, atol=2e-6, rtol=2e-5), 'Analytic softmax gold gradient mismatch')
        stable = logits.double().masked_fill(~batch['mask'], -torch.inf).log_softmax(1).masked_fill(~batch['mask'], 0)
        expected_loss = -(stable * y.double()).sum(1).div(batch['mask'].sum(1).double().log()).mean()
        require(abs(losses[track] - float(expected_loss)) < 2e-6, 'Independent gold CE mismatch')
    # Check direction signs and exact zero directions independently on CUDA.
    left = [torch.tensor([1., 2., -3.], device='cuda'), torch.tensor([4.], device='cuda')]
    right = [torch.tensor([2., -1., 1.], device='cuda'), torch.tensor([.5], device='cuda')]
    require(float(operator.dot(torch, left, right)) == -1., 'Displacement dot sign mismatch')
    require(operator.cosine(torch, left, [torch.zeros_like(x) for x in left]) is None, 'Zero norm must be undefined')
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01, fused=True, foreach=False)
                  for m in (first, second)]
    priors = {track: 1 / 15 for track in grouped}
    observer = operator.Observer(torch, training, first, cache, grouped, factors, priors, lambda result: None,
                                 epochs=(4,), steps=(3, 6, 9))
    observer.epoch = 4
    pre = optimizers[0].register_step_pre_hook(observer.pre)
    post = optimizers[0].register_step_post_hook(observer.post)
    try:
        for step in range(9):
            for model, optimizer in zip((first, second), optimizers):
                batch = cache.batch(rows)
                loss = operator.gold_ce(torch, model(batch['query']), batch).mean() * 100
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            require(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())),
                    'Observer changed actual fused AdamW parameter trajectory')
            for a, b in zip(first.parameters(), second.parameters()):
                state_a, state_b = optimizers[0].state[a], optimizers[1].state[b]
                require(set(state_a) == set(state_b), 'Optimizer state geometry changed')
                require(all(value.is_cuda and torch.equal(value, state_b[key]) for key, value in state_a.items()),
                        'Observer changed actual mature CUDA AdamW moments/steps')
    finally:
        pre.remove()
        post.remove()
    require(len(observer.observations) == 3 and observer.pending is None, 'Missing mature hook observations')
    require(all(torch.equal(x, old) for x, old in zip((query, masks, targets), snapshots)), 'Synthetic inputs changed')
    require(sha(source) == args.worker_sha256 and sha(observer_path) == args.observer_sha256,
            'Source SHA mismatch after CUDA proof')
    torch.cuda.synchronize()
    result = {'kind': 'mature-displacement-cuda-measurement-proof-v1', 'passed': True, 'cuda': True,
        'worker_sha256': args.worker_sha256, 'observer_sha256': args.observer_sha256,
        'torch': torch.__version__, 'cuda_version': torch.version.cuda, 'tf32': False,
        'cpu_head_execution': False, 'corpus_accessed': False,
        'retained_head_proof': False, 'synthetic_tracks': 15, 'synthetic_rows': 60,
        'masked_soft_targets': True, 'max_analytic_gradient_error': max(errors),
        'gradients_rng_mode_inputs_unchanged': True, 'mature_optimizer_steps': 9,
        'hook_observations': 3, 'fused_hook_unhooked_parameters_moments_steps_exact': True}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
