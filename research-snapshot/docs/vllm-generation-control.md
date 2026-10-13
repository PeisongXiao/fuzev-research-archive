# vLLM Generation Control

`fuzev_research.vllm_generation_control.run_control` consumes an existing engine and
rendered procedural token IDs to measure generation without the decision head.

`describe_control` recorded source revisions and loaded engine settings. The
lab used the 262144 window, up to four synchronous V1 sequences, no speculation
or cache transfer, and allocation fraction 0.85. Raw PLE used
`ple_storage=mmap`; packed EXL3 used `VLLM_EXL3_NGRAM_TABLE=disk` with the
graph boundary. MemoryGuard and host admission were recorded resource settings.

**This is a fusion-disabled generation path in the same pinned fork, not pristine
upstream.** The core's `fuzev.enabled` is false and supported tasks must exclude
`fuzev_features`. Disk PLE remains enabled independently. An already fused
generation-only server is rejected as a control. No pristine upstream baseline
with the same disk-PLE capability was executed.

The lab used one, two or four requests per wave. The caller sequences repetitions/warmup
and the fused generation/decision/mixed harness on identical model payload,
encoding, explicit sampling, cache policy, geometry and source/kernel settings.
The serving alias is independent of the actual snapshot path; both are recorded.
Prefix reset requires an idle engine acknowledgement, never resets running
requests and does not imply cold image-encoder, processor or NVMe caches.
Within-wave prefix reuse and unknown cache counters are reported honestly.

Control records retain API-accepted IDs including a stopping token and exact prompt
hashes, sampling, elapsed wall intervals, and the same validated engine-core
phase timestamps used by fused serving. Missing/preempted stats have no nominal
phase throughput. No physical forwarded-token count is invented for stock
generation, and request-rate sums are not called GPU-kernel throughput.
Actual AsyncLLM stats must be enabled. Requested sampling and normalized
submission parameters are recorded separately; generation-config EOS
augmentation happens afterward inside the engine. Direct-engine client intervals
exclude HTTP rendering/retention and cannot be compared as equivalent wall
latencies to the HTTP harness. Only matched engine phases share that scope.

Cancellation/failure aborts only owned request IDs and closes/drains readers and
iterators before returning. vLLM abort submission is not a GPU-free
acknowledgement. The caller retains exclusive engine ownership and must verify
actual scheduler/worker cleanup in a live run.
Any abort closes control admission on that engine until the caller shuts it
down; an empty frontend request map is not proof of backend cleanup. Sibling
abort/iterator failures are retained independently of the primary error and
repeated cancellation. Images require one admitted matching RGB digest;
submission-time binding records and borrowed-payload postflight checks prevent
later mutation from silently changing the witness.

Every record has `qualified:false`. CPU fakes cover lifecycle/contracts only;
real generation equality, paired timing/memory, images and cleanup remain
unmeasured by these fixtures.
The outer operator used the canonical artifact/source preflight from the
comparison executor before constructing an engine. This callable inspects
loaded configuration/source pins, not tensor-file checksums or a GPU loader.

[Snapshot methods and environment](../README.md)
