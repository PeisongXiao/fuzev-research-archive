# v1.1 Integration Evidence

The original panels were recorded on 2026-10-16 on one NVIDIA GB10
(128 GB unified memory), ARM64. The
[public integration page](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/integration.md)
summarizes engine-owned behavior; [historical integration results](integration-v1.0.md)
preserve the older pre-release/cross-engine checks.

## Recipe And Scope

This recipe describes F1-F3 and the recorded settings of F4, F5 and D1. The
later [decision-root checks](#decision-root-checks) have separate settings.

Candidate engine: exllamav3-fuzev at unpublished lab commit ID `3a3e1d02`
plus the hyper-connection table cache and profile changes that ship in
exllamav3-fuzev v1.1. It is based on ExLlamaV3 1.6.1 (`3a513ac9`).
The [profile](https://github.com/PeisongXiao/exllamav3-fuzev/blob/v1.1/exllamav3/optimized/profile.json)
sets `prefill_chunk=32768`, `gdn_native_layers=[0]`, `hc_cache_tables=true`,
`moe_batch_cap=4`, `moe_max_rows=16384` and `qsa_stage=true`.
The model is
`turboderp/Qwen3.8-Flash-Next-exl3`, 4.05 bpw, snapshot
`55a732e0c4c3d4614bc42b68493bb930d9b02c0a`.
The four prefill records hold PyTorch 2.13.0+cu130, Triton 3.7.1,
FlashInfer 0.6.18.post1 and JIT cache 0.6.18.post1+cu130. They omit Python;
the API record omits software versions and the head identity.

The five original [curated artifacts](#saved-artifacts) retain recorded measurements,
options, warmups, priming/setup, input-token checks, native accounting,
all decision scores/ranks and cleanup counters. Private absolute-path metadata,
request corpus/question/candidate text, generated text and response request IDs
are omitted following the [archive policy](results/README.md).
Missing native fields remain null.

## Long Prefills: F1-F3

### Reproduce

Use the [prefill harness](../tools/prefill_bench.py), measured candidate profile
and [model, corpus and output path setup](performance-v1.1.md#reproduce).
Build the corpus using the linked
[public recipe](v1.0-v1.1-comparison.md#speed-protocol-and-corpus).
The corpus is 1,923,478 bytes, SHA-256
`130f1671eb3bcc81331939dc40fead0944d008a0c0e242e33d454b3b9336e7e8`.
Each command below has one excluded warmup and one measured repetition:

```bash
python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/f1.json" --mode fork-opt --state fp32 --chunks 32768 \
  --lengths 131072 --context-tokens 163840 --cache-tokens 163840 \
  --use-gib 90 --reps 1 --warm-ple
python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/f2-short.json" --mode fork-opt --state fp32 --chunks 32768 \
  --prefix-tokens 131072 --append-tokens 512 \
  --context-tokens 164352 --cache-tokens 164352 --use-gib 90 --reps 1 --warm-ple
python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/f2-long.json" --mode fork-opt --state fp32 --chunks 32768 \
  --prefix-tokens 131072 --append-tokens 8192 \
  --context-tokens 172032 --cache-tokens 172032 --use-gib 90 --reps 1 --warm-ple
python tools/prefill_bench.py --model "$MODEL" --corpus "$CORPUS" \
  --out "$OUT/f3.json" --mode fork-opt --state fp32 --chunks 32768 \
  --prefix-tokens 249856 --append-tokens 8192 \
  --context-tokens 262144 --cache-tokens 262144 --use-gib 90 --reps 1 --warm-ple
```

### Measurements

The four jobs use the [prefill harness](../tools/prefill_bench.py), batch 1,
Q8 KV, FP32 recurrent state, zero draft steps, load allowance 90 GiB and
per-layer (n-gram) embeddings streamed from disk ([glossary](research-provenance.md#glossary)).

Each cell has one measured row at corpus offset 15,838, an excluded rep0
warmup at 7,919 and one same-prompt uncached prime. Prefix setup and priming
are outside the measured interval. Native prefill time and host job wall
time are separate.

| Cell | Context = Cache | Prompt / Cached / Processed Tokens | Forwarded | Native / Wall s | Measured Allocated Peak B / GiB |
| --- | ---: | --- | ---: | --- | --- |
| F1 | 163840 | 131072 / 0 / 131071 | Unrecorded | 58.887054 / 58.926258 | 76403345408 / 71.156160 |
| F2 short | 164352 | 131584 / 131072 / 511 | 512 | 0.923896 / 0.967444 | 70651965440 / 65.799770 |
| F2 long | 172032 | 139264 / 131072 / 8191 | 8192 | 4.816422 / 4.861232 | 71913055232 / 66.974252 |
| F3 | 262144 | 258048 / 249856 / 8191 | 8192 | 5.046858 / 5.093011 | 73438415360 / 68.394854 |

Processed tokens exclude cached tokens and the final prompt token, which the
first decode step consumes. F3's prompt ends at 258,048 = 262,144 - 4,096.
Appends generate one new token each. Processed and forwarded counts come
from harness accounting; F1 has no forwarded field. Measured input-block
deltas are zero.

Peaks are measured-region Torch allocated bytes, including live persistent
state. Warmup values remain
visible; for F2 long its warmup allocation is higher than its measured row.

## In-Process API: F4, F5 And D1

The final API panel uses the published loader and actual application through
`httpx.ASGITransport`. It loads a calibrated shared head, Q8 target/draft KV,
FP32 target recurrent state,
native MTP3, batch 4, context 32,768/cache 98,304, chunk 32,768 and load
allowance 90 GiB. The saved draft recurrent-dtype list is empty. The API
record contains no runtime/plugin version or head identity.

The F4/F5/D1 panel's exact producer is unpublished. The settings and
measurements below preserve the recorded method, but the archive does not
provide a command that reproduces this panel. The public F1-F3 prefill
commands and component installation checks are different workloads.

| Cell | Recorded Work | Phase Wall s | Peak Allocated GiB |
| --- | --- | ---: | ---: |
| F4 | Four HTTP 200 completions; each 16384 prompt / 0 cached / 1 output token; four unique prompts and API in-flight peak 4. | 45.852939 | 70.266059 |
| F5 | HTTP 200; 16384 prompt / 0 cached / 256 output; 190 accepted / 8 rejected drafts; native prefill 8.293971 s / generation 7.664097 s. | 15.992372 | 70.287297 |
| D1 K8 | HTTP 200; 2048-token base / 2129-token templated root; eight 9-token tails, each caching 2129; eight complete finite calibrated scores. | 5.573794 | 67.136335 |

F4/F5 input tokenization is exact, with zero boundary adjustment, roundtrip
checks and API usage matching native counts. F5 is greedy, seed 43, and
reaches its 256-token cap (`natural_eos=false`). Task accuracy is not
measured.

D1's saved execution accounting is **19,233 submitted / 17,032 cached /
2,201 forwarded tokens / 8 completed root branches**. Base-token count,
all eight logits, raw/calibrated probabilities, ranks and native root/tail
counts remain in the artifact. Its missing native timing/MTP fields remain
null.

After each phase and finally, every recorded cleanup counter is zero:
native jobs, readout roots, pinned checkpoints, target/draft recurrent slots,
service jobs, owner jobs and retained prefixes. Allocator peaks are shared
per phase and include resident model allocations.

Per-request wall times, phase wall times and native intervals are retained
separately.

## Earlier v1.1 Head Bitrate Checks

The existing [head-results report](v1.1-head-results.md#paired-bitrate-api)
and [paired aggregate](results/v1.1-head/bitrate-api.json) record the same
v1.1 head (bundle 1.1.0) on 4.05/3.05 bpw, MTP 3/batch 4,
context 65,536/cache 73,728 and chunk 8,192.

Image decisions are an experimental extension; run at your own risk.
Both bitrate runs use the **in-process ASGI** application.
Each completes 24 successful finite decisions /
12 fresh-retained pairs: four high-K, one Noul, one Ordinal and six visual
cases. Within each bitrate all 12 pairs have exact winners, top-two, ranks,
logits and raw/calibrated probabilities. Native jobs, readout roots,
service jobs, owner jobs and retained prefixes are zero at cleanup.

Cross-bitrate winners agree 10/12 and top-two membership 9/12. Choice
K128/K255 winners change; Choice K64/K128 and Ordinal255 top-two membership
change. Before/after generation probes each produce 16 tokens, with
accepted/rejected counts 8/16 on 4.05 and 6/24 on 3.05. These checks measure
capacity and retained-path portability.

## Decision-Root Checks

The 2026-10-17 checks use one NVIDIA GB10 (128 GB unified memory), Python
3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.7.2 and
exllamav3-fuzev 1.6.1+fuzev.v1.1. Native checks use **MTP off**, batch 4,
Q8 KV, FP32 recurrent state, a 2,048-token prefill chunk and checkpoint
interval, and a 16,384-token cache. The before/after HTTP timing and
cancellation checks use **MTP3** with the same batch, KV, recurrent-state and
chunk settings and a 16,384-token context/cache; their records also report
JIT cache 0.7.2+cu130.

The correctness record and the after timing arm use the v1.1 release-candidate
fork plus the decision-root scheduling fix, with the release-candidate
runtime/plugin sources; the before timing arm omits the fix. The concurrency
record compares the release candidate before the fix with the final v1.1
sources. The warm timing and mixed HTTP records use the final v1.1 sources.

### Concurrent And Sequential Decisions

The [concurrency record](results/integration/v1.1/decision-root-concurrency.json)
decides four three-option Choice questions over one shared state on one loaded
model, one question at a time in three passes and all four concurrently in two
passes. The requested 2,048-token prefix decodes to text that re-encodes to
1,878 tokens; the encoded shared state has 1,883 tokens. Sequential passes
match each other exactly, whether they start cold or reuse the cached shared
state, and the two concurrent passes match each other exactly.

Maximum concurrent-versus-sequential differences in features, raw and
calibrated probabilities are **2.75 / 0.048 / 0.042** on the final v1.1
sources and **3.91 / 0.089 / 0.080** on the release candidate before the fix.
On the release candidate, three of the four concurrent questions complete no
root branches; with the fix, every question completes all three root branches.
No question changes its selected option.

### Matched-Geometry Control

The [correctness record](results/integration/v1.1/decision-root-correctness.json)
holds a same-process control that enqueues all four query jobs on one loaded
model. Their four roots coexist, sharing seven complete KV pages, while
**one root's children at a time** drain in fixed four-to-two-row geometry.
Each isolated reference uses a fresh Generator on that same model with
matching query/candidate geometry and exact token sequences. Queries have
1,939 tokens, all forwarded without a cached prefix; candidate sequences have
1,950/1,950/1,949/1,949 tokens. The three original candidates are scored; a
duplicate supplies the fourth row. Maximum query-feature, candidate-feature,
raw-probability and calibrated-probability differences are all **zero**.
Final active/pending jobs, roots, pins and referenced pages are zero, with all
four slots free.

The record also keeps two earlier diagnostics. In the initial
varying-geometry diagnostic (four three-candidate questions over the same
1,883-token shared state), three isolated repetitions match each other
exactly and the concurrent pass differs from them by up to
**2.546875 / 0.08900523 / 0.07610142** in features, raw and calibrated
probabilities. The other compares a separate model load against saved
tensors: query and candidate features differ by up to 1.66 and 6.09, and raw
probabilities by up to 0.065.

### Timing And Mixed HTTP Cancellation

The [timing record](results/integration/v1.1/decision-root-timing.json)
contains one before/after pair for four independent two-option SystemOne
questions over a shared 2,048-token context. The release-candidate
runtime/plugin sources are held fixed; only the decision-root scheduling fix
changes. Grouped requests run first after server start, concurrent requests
next, then separate sequential decision probes, so the grouped wall time
(**16.51 -> 16.90 s**) includes first-use initialization. Concurrent wall time
is **2.02 -> 1.23 s**. The probes retain identical execution counters;
[timing details](performance-v1.1.md#decision-root-timing) and the
[HTTP timing tool](../tools/shared_prefix_timing.py) describe that panel.

The [warm timing record](results/integration/v1.1/decision-root-timing-warm.json)
repeats the fixture four times on the final v1.1 sources with the launcher
defaults (262,143-token context, batch 16, FP16 KV, FP32 recurrent state,
32,768-token chunk, MTP off). Warm samples take 1.23-1.30 s grouped and
1.15 s concurrent; the first grouped request after start takes 2.81 s.

The separate [mixed HTTP record](results/integration/v1.1/mixed-http-cancellation.json)
runs a nine-candidate decision, six short chats and an uncapped streaming
chat. The decision completes nine branches, and it and the six chats return
HTTP 200 before the streaming chat is cancelled at 64 client-retokenized
output tokens. The stream transport returns HTTP 200 with the expected 499
SSE cancellation error and completion marker. All eight request IDs become
inactive; subsequent health and short-chat requests return HTTP 200. This
check measures HTTP request cleanup and subsequent responsiveness.

## Saved Artifacts

| Check | Curated JSON |
| --- | --- |
| F1 fresh 131072 | [final-f1.json](results/integration/v1.1/final-f1.json) |
| F2 short append | [final-f2-short.json](results/integration/v1.1/final-f2-short.json) |
| F2 long append | [final-f2-long.json](results/integration/v1.1/final-f2-long.json) |
| F3 near-context append | [final-f3.json](results/integration/v1.1/final-f3.json) |
| F4/F5/D1 API | [api-long-prompt-checks.json](results/integration/v1.1/api-long-prompt-checks.json) |
| Decision-root concurrent versus sequential | [decision-root-concurrency.json](results/integration/v1.1/decision-root-concurrency.json) |
| Decision-root correctness and geometry | [decision-root-correctness.json](results/integration/v1.1/decision-root-correctness.json) |
| Decision-root before/after timing | [decision-root-timing.json](results/integration/v1.1/decision-root-timing.json) |
| Decision-root warm timing | [decision-root-timing-warm.json](results/integration/v1.1/decision-root-timing-warm.json) |
| Mixed HTTP completion and cancellation | [mixed-http-cancellation.json](results/integration/v1.1/mixed-http-cancellation.json) |

## Limitations

- These are single functional measurements without matched F-series controls
  or confidence intervals.
- F4/F5/D1 and bitrate API requests use in-process ASGI; network latency is
  outside those panels.
- Allocations are phase-level Torch peaks. Host page residency is unmeasured.
- The bitrate panel establishes capacity and retained-path behavior; quality
  results are in the [served comparison](v1.0-v1.1-comparison.md).
- Earlier decision results were measured on the v1.1 release candidate before
  the [decision-root scheduling fix](#decision-root-checks); prefill and decode
  do not use that path.
- Matched-geometry equality covers the recorded token sequences, per-root
  four-to-two-row windows and query jobs without a cached prefix. Other
  geometries, including 16 candidate rows from several questions in one batch,
  are unmeasured by that control.
- Concurrent and sequential execution of the same decisions differ by up to
  0.048 raw probability in the recorded probe (0.089 before the decision-root
  scheduling fix); each order repeats exactly. Other prompts, option counts and
  batch sizes can differ by other amounts.
- The native correctness and concurrency probes are unpublished.
- Internal page and root release are unmeasured by the HTTP cancellation check.
- Image quality is outside these checks; see the
  [head-results limits](v1.1-head-results.md#provenance-and-limits).
- Reproduction needs equivalent hardware and data.
