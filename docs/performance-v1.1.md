# v1.1 Matched Prefill Performance

Recorded on 2026-10-16 on one NVIDIA GB10 (128 GB unified memory).
The [public summary](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/performance.md)
keeps the current table and settings; [historical short-workload results](performance-v1.0.md)
retain the earlier standalone-server measurements.

The [aggregate](results/performance/v1.1/final-prefill-upstream-master-results.json)
and six source records contain timings, memory, token counts, settings,
warmups and W4 controls. The checkpoint is
`turboderp/Qwen3.8-Flash-Next-exl3` 4.05 bpw, snapshot
`55a732e0c4c3d4614bc42b68493bb930d9b02c0a`.

**Matched prefill:** all five cells have lower final native
and wall mean times, and all ten matched-offset comparisons point the same
way. Wall mean-time ratios W1-W5 are **1.60 / 1.40 / 1.44 / 1.72 / 1.30**.

## Recipe And Provenance

Reference: ExLlamaV3 master `058b8630` (version string 1.6.2, one commit
after the v1.6.2 tag `4c842731`), plain prefill, no FuZev hooks.
Candidate engine: exllamav3-fuzev at unpublished lab commit ID `3a3e1d02`
plus the hyper-connection table cache and profile changes that ship in
exllamav3-fuzev v1.1. The candidate is based on ExLlamaV3 1.6.1
(`3a513ac9`); the ratios compare complete builds.

Both sides record PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer
0.6.18.post1 and JIT cache 0.6.18.post1+cu130. Python is unrecorded in this
panel; unrecorded package versions are omitted. Settings are batch 1, Q8 KV,
chunk and loader 32,768, load allowance 90 GiB, FP32 recurrent state and
zero draft steps during prefill. Per-layer (n-gram) embeddings are streamed
from disk ([PLE and QSA glossary](research-provenance.md#glossary)).
The candidate's [profile](https://github.com/PeisongXiao/exllamav3-fuzev/blob/v1.1/exllamav3/optimized/profile.json)
sets `prefill_chunk=32768`, `gdn_native_layers=[0]`, `moe_batch_cap=4`,
`moe_max_rows=16384`, `qsa_stage=true` and `hc_cache_tables=true`.

### Reproduce

The [archived harness](../tools/prefill_bench.py) reads the Apache-2.0
vLLM-documentation corpus assembled in the
[comparison method](v1.0-v1.1-comparison.md#speed-protocol-and-corpus):
1,923,478 bytes, SHA-256
`130f1671eb3bcc81331939dc40fead0944d008a0c0e242e33d454b3b9336e7e8`.
Install each measured engine build separately. Set `MODE=plain` for
ExLlamaV3 master and `MODE=fork-opt` for FuZev; use the recorded candidate
profile above and the separately acquired model snapshot. From the archive
root, place that snapshot in `model`, build `corpus.txt` with the linked
recipe, and set these paths once for the performance and F1-F3 commands:

```bash
MODEL="$PWD/model"
CORPUS="$PWD/corpus.txt"
OUT="$PWD/prefill-results"

python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/fresh.json" --mode "$MODE" --state fp32 --chunks 32768 \
  --lengths 2048 16384 65536 --context-tokens 98304 --cache-tokens 98304 \
  --use-gib 90 --reps 2 --warm-ple
python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/w4.json" --mode "$MODE" --state fp32 --chunks 32768 \
  --prefix-tokens 32768 --append-tokens 512 \
  --context-tokens 66048 --cache-tokens 66048 --use-gib 90 --reps 2 --warm-ple
python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/w5.json" --mode "$MODE" --state fp32 --chunks 32768 \
  --prefix-tokens 32768 --append-tokens 8192 \
  --context-tokens 73728 --cache-tokens 73728 --use-gib 90 --reps 2 --warm-ple
```

The recorded candidate W4 runs interleave the full profile, an 8,192-token
chunk, disabled hyper-connection table caching, then the full profile again.
Reproduce that order by adding
`--variants '{"Full":{},"minus_C":{"chunk":8192},"minus_H0":{"hc_cache_tables":false},"repeat_Full":{}}'`
to the candidate W4 command. The primary comparison uses the first full-profile
arm; the table retains its repeat separately.

| Cell | Prompt / Cached / Processed Tokens | Context / Cache | Matched Offsets r1 / r2 |
| --- | --- | --- | --- |
| W1 | 2048 / 0 / 2047 | 98304 / 98304 | 15838 / 23757 |
| W2 | 16384 / 0 / 16383 | 98304 / 98304 | 39595 / 47514 |
| W3 | 65536 / 0 / 65535 | 98304 / 98304 | 63352 / 71271 |
| W4 | 33280 / 32768 / 511 | 66048 / 66048 | 15838 / 23757 |
| W5 | 40960 / 32768 / 8191 | 73728 / 73728 | 15838 / 23757 |

Each cell has two measured repetitions per side plus one excluded rep0
warmup: **20 primary measured rows / ten selected warmups**. Other W4
leave-one-out arms are in [final-loo-w4.json](results/performance/v1.1/final-loo-w4.json).
Every measured
arm gets one same-prompt zero-KV PLE prime before a fresh Generator.
W4/W5 seed the same 32,768-token prefix using a 32,769-token setup prompt and
one generated token; measured forwarded counts are 512/8192, with one new
token each. Prime/setup time is excluded. Processed counts are saved
harness accounting. Metadata,
offsets, token/prime/prefix counts and actual recurrent dtypes match;
all 20 measured timed input-block deltas are observed 0.

**W4 reuse is explicit:** candidate rows are only `variant=Full,
measured=true` from final-loo-w4.
Settings/offsets/reps match upstream W4, but candidate blocks interleave
the full profile, two leave-one-out arms, then a repeat of the full profile,
while upstream is a standalone ordinary run.

## Means

Ratios are **mean upstream seconds / mean candidate seconds**.
Positive time-ratio gain is 100*(ratio-1); latency
reduction is 100*(1-1/ratio). Native prefill and host
wall are kept separate; mean per-row rates are descriptive.

| Cell | Native Mean s Upstream / Final | Wall Mean s Upstream / Final | Native / Wall Time Ratio | Mean Row Tokens/s Upstream / Final |
| --- | --- | --- | --- | --- |
| W1 | 2.457761 / 1.526967 | 2.492377 / 1.561329 | 1.609570 / 1.596317 | 833.322 / 1340.763 |
| W2 | 11.177960 / 7.991264 | 11.213152 / 8.026130 | 1.398772 / 1.397081 | 1465.662 / 2050.123 |
| W3 | 42.645931 / 29.571555 | 42.682343 / 29.608360 | 1.442127 / 1.441564 | 1537.268 / 2216.188 |
| W4 | 1.496147 / 0.852778 | 1.533909 / 0.893150 | 1.754440 / 1.717415 | 341.692 / 599.890 |
| W5 | 6.035254 / 4.619482 | 6.073800 / 4.659410 | 1.306479 / 1.303555 | 1357.416 / 1773.164 |

Native time-ratio gains W1-W5 are **+60.957047% / +39.877248% /
+44.212678% / +75.443950% / +30.647856%**; wall gains are
**+59.631716% / +39.708085% / +44.156392% / +71.741480% / +30.355542%**.

## All Matched Samples And Variation

All ten pairs are shown; raw precision, token counts, priming/setup,
per-row rates, allocated bytes and options are retained in the
[aggregate sidecar](results/performance/v1.1/final-prefill-upstream-master-results.json).

| Cell / Rep | Offset | Native s Upstream / Final | Wall s Upstream / Final | Native / Wall Time Ratio |
| --- | --- | --- | --- | --- |
| W1 / 1 | 15838 | 2.400653 / 1.508455 | 2.435642 / 1.542332 | 1.591465 / 1.579195 |
| W1 / 2 | 23757 | 2.514868 / 1.545479 | 2.549111 / 1.580326 | 1.627242 / 1.613028 |
| W2 / 1 | 39595 | 11.207027 / 7.974500 | 11.242268 / 8.009146 | 1.405358 / 1.403679 |
| W2 / 2 | 47514 | 11.148892 / 8.008028 | 11.184036 / 8.043114 | 1.392214 / 1.390511 |
| W3 / 1 | 63352 | 41.843092 / 29.448535 | 41.879872 / 29.485853 | 1.420889 / 1.420338 |
| W3 / 2 | 71271 | 43.448769 / 29.694574 | 43.484814 / 29.730866 | 1.463189 / 1.462615 |
| W4 / 1 | 15838 | 1.527313 / 0.824241 | 1.564757 / 0.864973 | 1.852993 / 1.809024 |
| W4 / 2 | 23757 | 1.464981 / 0.881314 | 1.503061 / 0.921327 | 1.662268 / 1.631409 |
| W5 / 1 | 15838 | 5.957745 / 4.635409 | 5.997009 / 4.674605 | 1.285269 / 1.282891 |
| W5 / 2 | 23757 | 6.112762 / 4.603555 | 6.150590 / 4.644215 | 1.327835 / 1.324355 |

Candidate W4 has an additional same-offset control witness:

| Rep / Offset | Full -> Repeat Native s | Native Latency Shift % | Full -> Repeat Wall s | Wall Latency Shift % |
| --- | --- | --- | --- | --- |
| 1 / 15838 | 0.824241 -> 0.791142 | -4.016 | 0.864973 -> 0.831542 | -3.865 |
| 2 / 23757 | 0.881314 -> 0.907538 | +2.975 | 0.921327 -> 0.947487 | +2.839 |

Using repeated Full instead of initial Full gives W4 native/wall
mean-time ratios **1.761541 / 1.724434**.
The per-offset native shifts are -4.015773% and +2.975490%; the
repeat-Full mean native time is 0.403167% lower than the Full mean. The
[source W4 LOO artifact](results/performance/v1.1/final-loo-w4.json) preserves
both controls and every ablated arm.

## Memory

| Cell | Upstream Measured Peak GiB | Final Measured Peak GiB | Final Minus Upstream GiB |
| --- | --- | --- | --- |
| W1 | 63.497902 | 64.800358 | +1.302456 |
| W2 | 65.926583 | 67.169414 | +1.242831 |
| W3 | 68.402045 | 70.127307 | +1.725263 |
| W4 | 62.738453 | 64.052942 | +1.314489 |
| W5 | 64.228159 | 65.417879 | +1.189719 |

These are maximum recorded Torch allocated bytes across each selected
cell's two measured rows, including live persistent state. Warmup rows remain
in the sidecar and are excluded from timing means. Overall selected-row
maxima, including warmups, are upstream 68.402045 / final 70.127307 GiB.

The saved [host-memory summary](results/performance/v1.1/final-prefill-upstream-master-results.json) has
**740 samples** in the **07:10:48..08:12:31.635 UTC** window. Minimum host
MemAvailable is **39222432 KiB / 37.405426 GiB** at **07:41:38 UTC**;
first/last samples are 07:10:50 / 08:12:30,
largest adjacent interval 6 s. The archive records summary statistics of
this host-memory log.

## Decision-Root Timing

A separate [SystemOne record](results/integration/v1.1/decision-root-timing.json)
from 2026-10-17 contains one before/after pair for four independent two-option questions over
a shared 2,048-token context. It uses one NVIDIA GB10 (128 GB unified memory),
Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.7.2,
JIT cache 0.7.2+cu130 and exllamav3-fuzev 1.6.1+fuzev.v1.1. Settings are
batch 4, Q8 KV, FP32 recurrent state, MTP3, 2,048-token chunks and
16,384-token context/cache. The release-candidate runtime/plugin sources are
identical in both arms; only the fork's decision-root scheduling fix changes.

The context file contains the first 2,048 retokenized tokens from the
[public vLLM documentation corpus](v1.0-v1.1-comparison.md#speed-protocol-and-corpus).
Both arms report `usage.input_tokens` from the release-candidate runtime and
plugin, which count every logical branch: grouped **27,302**; concurrent
uppercase **6,306**, number **6,320**, color **6,309**, weekday **6,314**. The
v1.1 API counts the shared encoded state once.

| Request Shape | Before Wall s | After Wall s |
| --- | ---: | ---: |
| One grouped SystemOne request, first after server start | 16.51 | 16.90 |
| Four concurrent SystemOne requests | 2.02 | 1.23 |

Each arm has one sample. Grouped requests run first, concurrent requests next
and separate counter probes last, so the grouped row includes first-use
initialization. The sequential probes have identical execution counters across
arms.

The [warm timing record](results/integration/v1.1/decision-root-timing-warm.json)
repeats the fixture four times on the final v1.1 sources (same GB10 and
software versions) with the launcher defaults: 262,143-token context,
batch 16, FP16 KV, FP32 recurrent state, 32,768-token chunk and MTP off.
Warm samples take 1.23-1.30 s grouped and 1.15 s concurrent; the first
grouped request after start takes 2.81 s.

Concurrent and sequential decisions differ by up to 0.048 raw / 0.042
calibrated probability on v1.1 (0.089 / 0.080 before the fix;
[method and limits](integration-v1.1.md#decision-root-checks)).

## Saved Sources And Scope

| Side | Fresh W1-W3 | W4 | W5 |
| --- | --- | --- | --- |
| ExLlamaV3 master | [upstream-master-fresh.json](results/performance/v1.1/upstream-master-fresh.json) | [upstream-master-w4.json](results/performance/v1.1/upstream-master-w4.json) | [upstream-master-w5.json](results/performance/v1.1/upstream-master-w5.json) |
| FuZev final profile | [final-fresh.json](results/performance/v1.1/final-fresh.json) | [final-loo-w4.json](results/performance/v1.1/final-loo-w4.json), Full only for the primary pair | [final-w5.json](results/performance/v1.1/final-w5.json) |

Upstream fresh/W4/W5 finish 07:17:02 / 07:27:07.775 / 07:33:17.672 UTC;
candidate fresh/W4/W5 finish 07:42:19.787 / 07:57:24.268 / 08:12:31.635 UTC,
on 2026-10-16.

## Limitations

- Two offsets per cell share one load per arm. The builds run sequentially;
  this panel has no upstream self-repeat or confidence interval.
- Candidate W4 interleaves ablations; the upstream W4 runs alone.
- W1-W5 timings exclude loading, startup, prime and prefix setup. Decode and MTP
  results are in the [served comparison](v1.0-v1.1-comparison.md#speed).
- Allocations measure Torch tensors; host-memory statistics cover a broad
  window including idle periods and intervening work.
- The Kev runs in [head results](v1.1-head-results.md) use 8,192-token chunks;
  the [served comparison](v1.0-v1.1-comparison.md) uses the 32,768-token default.
- Reproduction needs equivalent hardware and data.
