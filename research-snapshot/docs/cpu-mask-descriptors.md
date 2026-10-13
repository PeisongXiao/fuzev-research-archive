# CPU Mask Descriptors

## Scope and Evidence

`FeatureStore.mask_width()` previously reopened each represented shard absent
from the tensor LRU and reduced its selected masks on every call. The following
`batch_tensors()` pass separately loads the features. Grouped copying, compact
metadata, readiness caching/version guards, and the 64 MiB/four-shard tensor LRU
already exist; none are reimplemented or enlarged here. No throughput bottleneck
has been measured on the reference corpus.

The only runtime change is a process-local, byte- and entry-bounded LRU of
immutable per-shard row widths in `fuzev_research/features.py`. Descriptors use one byte
per row: last true position plus one, or zero for fewer than two candidates.
Only selected invalid rows fail, including on hits. Nonprefix masks still use
their last true position, not their population count. No tensor, mmap, file
handle, prepared context, or feature vector is retained in this cache.

Each descriptor lookup stats device, inode, size, mtime_ns and ctime_ns. Changes
discard the descriptor; missing files fail. A miss reads only the mask and
checks the stamp again before caching. Concurrent changes fail closed. The
existing loaded-tensor path is unchanged and still checks the selected masks;
this is not a new full-tensor-cache invalidator or atomic corpus snapshot.
Committed shards must remain immutable, as before. `close()` clears descriptors.

`train.py`, `head.py`, the schedule, readiness checks, role membership checks,
collation/copy order and FP32 operations are unchanged. There are no workers,
RNG calls or asynchronous batches. Defaults remain CPU four threads, batch 256,
12 profiles per stage (17/29/43, two learning rates, two Brier weights), ten
epochs and patience two. The training-code identity includes `features.py`, so
the change produced new trial identities.

## Budget and Expected Bound

- Default descriptor payload cap: 1 MiB; independent entry cap: 8,192 shards.
  `mask_cache_bytes=0` disables retention. Oversized entries are not cached.
- At the supplied 562,773-row corpus size, descriptors need 562,773 payload
  bytes (about 0.537 MiB), with roughly 4,397 shards at the source default of
  128 rows before extra partial flushes. This is arithmetic, not a corpus scan.
- Allow conservatively 8 MiB including bounded Python entry overhead on
  64-bit CPython, not a measured RSS guarantee. A normal 128-by-16 mask creates
  only a small transient Python list on misses. Tensor LRU and batch allocations
  are unchanged; nothing is preloaded or allocated on GPU.
- Each warm hit replaces one mask-only open/read plus Torch mask reductions
  with one stat and byte lookups. Full feature loads/copies and head work are
  unchanged. For a tensor-LRU miss, two shard opens become one after warmup.
  There is **no guaranteed wall-time gain**: cold misses process the whole tiny
  mask, stats still cost time, and oversized/thrashing caches miss. The ideal
  time saving cannot exceed the old mask-width pass; if it was fraction `p` of
  runtime, the ideal upper speedup bound is `1/(1-p)`, not a measured estimate.

Batch lookahead is deliberately deferred: an extra maximum-width FP32 batch
alone costs about 42.5 MiB before autograd, while worker contention, SQLite
thread ownership and concurrent tensor-LRU access add risk. No evidence yet
justifies that complexity, a preload, or another profiling harness.

## Verification

Executed only `python3 -B tests/test_mask_descriptors.py`: 12 stdlib-only fake
tests passed in 0.006 seconds on the final run. Coverage includes cold/hot
lookup counts, immutable payloads/no retained mask objects, row subsets,
nonprefix masks, input row order/Python RNG preservation, selected invalid masks,
byte/entry LRU limits, disabled/oversized entries, all stamp fields, same-size
same-mtime replacement, deletion, mid-read mutation/corruption and close cleanup.
Fixtures are temporary; acquisition, project paths and full-shard loads are
stubbed or forbidden. No Torch, safetensors runtime, native/GPU import or real
corpus read was used.

The real-tensor regression is the standalone
`tests/test_mask_descriptors_cpu.py`. It imports only features and
heads from the project, never train/native/exllamav3/vllm. It compares every
warm-cache batch tensor exactly against the existing uncached row-reader
collation path, including reordered/duplicate rows, nonprefix masks and an
unselected wider row. It also compares FP32 logits, normalized NLL plus Brier
loss and every parameter gradient exactly, and checks CPU Torch/Python RNG
states. There are only two two-row shards, one four-row batch and one default
CandidateHead, with one CPU thread and no optimizer/checkpoint. Fixture payload
is about 0.24 MiB. Thread count and RNG states are restored on exit.

CPU command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_mask_descriptors_cpu.py
```

## Recorded Comparison

A subsequent check ran 12 fake descriptor checks (0.020 seconds), the
standalone real CPU tensor/FP32 logit/loss/gradient/RNG case (1.09 seconds), and
the existing fresh-process rehearsal/readiness/orchestration suites (25 passed
in 10.89 seconds). The latter includes all 24 one-epoch synthetic profiles,
calibration, exact nonterminal resume and durable patience-stop behavior. No
actual reference head was fitted; the corpus-completion barriers are unchanged.

`tools/bench_cpu_mask_descriptors.py` then compared against the actual unchanged
unpublished lab commit `035184b` feature reader, verified by its source SHA256 rather than substituting
another local optimization or disabling retention in the new implementation.
The comparison validated the control's host-verified source digest directly.

The bounded read-only comparison used 122 fitting-row locators over 32 committed
shards, CPU four threads, warm OS caches and three ABBA blocks. It did not select
heldouts, train, decode, preload the full corpus or initialize CUDA. Every batch
tensor matched the original control exactly. Median results:

| Scope | Original control | Descriptor reader |
| --- | ---: | ---: |
| Mask-width pass | 1.5530 ms | 0.2850 ms |
| Complete collation | 8.7596 ms | 7.3777 ms |

This is **1.1873x warm collation**, not 18.7% faster extraction or whole training.
It removes about 1.38 ms for this small batch; real training includes head
forward/backward, optimizer work and different shard/input distributions.
Only 4,096 descriptor payload bytes were retained in this run. The original feature files were opened read-only.
Three ABBA blocks were measured; per-run timing spread was not recorded.

[Snapshot methods and environment](../README.md)
