# Native Generation Sampling

The native research sampler applies temperature, top-k and top-p to the
existing LM head's logits. Extraction does not call this sampler.

## Bounded Untied Head

`NativeService._sample()` previously copied every float32 vocabulary logit to
CPU, applied the top-k cutoff there, and sorted the full vocabulary. It still
uses the existing CPU top-p calculation, renormalization and multinomial, but
can avoid that full transfer and sort when all these conditions hold:

- Temperature is exactly `1`.
- `1 <= top_k <= 64`, and the vocabulary is larger than `4 * top_k`.
- The selected top-k values are finite and strictly decreasing.
- The kth value is strictly greater than the next value.

The device-side top-k obtains `k+1` values to witness the cutoff. Only those
values and the k selected token indices are copied to CPU. For the agreed
`temperature=1, top_p=.95, top_k=20`, this is 21 float32 values and 20 int64
indices (244 bytes), plus a scalar validation result. This count excludes
device-internal work and transfer overhead; the CPU fixture only checks the
explicit `.cpu()` operands, not actual CUDA transfer traffic.

The CPU ordered vector still has the full vocabulary length: the selected
values followed by `-inf`. The unchanged full-length softmax/cumsum, shifted
top-p removal mask, second softmax and full-length multinomial preserve the
reference population and draw order. The selected values have a unique rank;
all tail probabilities are zero, so their unused token ordering is irrelevant.
There is no compact-k categorical draw, extra draw, GPU RNG or input mutation.
Full-vocabulary CPU allocations and multinomial work remain; this is not an
O(k)-only sampler.

Any selected/cutoff tie, insufficient finite head, other positive temperature,
disabled/large/full-vocabulary top-k falls back to the original CPU copy,
inclusive cutoff and full sort. Small populations also fall back so the head
values plus int64 indices cannot exceed the original full-logit transfer budget.
Ties outside the selected head do not matter.
Greedy mode retains the original CPU argmax and first-index tie rule without
spending RNG. Validation covers the entire float32 logit vector on its current
device before sampling: NaN, positive infinity, empty/invalid vector shape and
all-negative-infinity inputs reject; negative-infinity tails remain legal.
Float conversion and temperature-overflow failure behavior are unchanged.

## CPU Verification

`tests/test_native_sampling.py` keeps an independent copy of the unpublished lab commit `fea0428`
algorithm. CPU probabilities are checked with `rtol=0, atol=0`, alongside exact
tokens and CPU Generator states for seeds including `2**63-1`. Coverage includes
FP16/BF16/FP32/FP64 input, k=1/20/64, tiny/one/exact-float32-boundary top-p,
inclusive cutoff/conversion ties, signed zero, subnormal/extreme/narrow finite
heads, nonfinite values, greedy ties, repeated draws, float-conversion/temperature
overflow and a 262144-element strided input.
Instrumentation checks that eligible inputs copy only the bounded head and do
not call full sort; fallback cases do. A compact-k counterexample confirms that
shortening the multinomial population is not a seeded-equivalence strategy.

Historical focused run: **76 passed in 2.44 seconds**, comprising 72 sampling
cases and four generation smoke cases.

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_native_sampling.py \
  tests/test_native_service.py::test_exact_accepted_tokens_include_hidden_eos_and_partial_token_stops \
  tests/test_native_service.py::test_length_stop_and_chat_render_only_a \
  tests/test_native_service.py::test_single_sample_has_no_decode_work
```

## Bounded CPU Microbenchmark

One synthetic CPU-only run used Torch `2.13.0+cu130`,
one Torch/BLAS thread, float32 `unique_logits`, k=20/p=.95/T=1, four warmups per
implementation and 24 paired single-sample trials per vocabulary. Trial order
alternated reference/optimized; each implementation had its own seed-17 CPU
Generator. Outputs and RNG states matched after every pair. Timing included the
whole sampler, with equality checks outside the measured region.

| Synthetic vocabulary | Reference median | Optimized median | Reference/optimized |
| --- | --- | --- | --- |
| 65536 | 2.121 ms | 1.534 ms | 1.38 |
| 262144 | 9.429 ms | 6.726 ms | 1.40 |

The report retains medians but not trial dispersion. These are short synthetic
CPU observations, not a live/GPU speedup, transfer
benchmark, model-vocabulary assertion or end-to-end tokens/second result. No
profiling flags were introduced. Device validation/top-k and extra synchronization
may change the real tradeoff; CUDA equivalence, transfer costs and runtime
performance were not measured. Exact CPU results
on this build do not claim byte identity across devices or Torch versions. The
eligible-head fraction is unknown; tied fallbacks also pay the bounded head
probe before the original algorithm.

The benchmark functions are `reference_sample`, `unique_logits` and `parameters`
in the focused test file, loaded with `runpy.run_path`; the optimized function
is `NativeService._sample`. The benchmark used only those synthetic fixtures,
not a native service instance, backbone or acquired data.

[Snapshot methods and environment](../README.md)
