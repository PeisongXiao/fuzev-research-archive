# Frozen Reference Native 4.05 Qualification Binding

The research binding connects the CUDA shared head to the native procedural
and full-window comparison methods.

`fuzev_research.native_research_qualification` connects the reference head to
the native 4.05 research service. The lab ran the frozen reference head in FP32
on CUDA:0 with raw temperature 1. The lab run recorded:

- HEAD `f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c`.
- PLAN `252f43ac3d7eab82c3d3be8db468da17798f4766e6b083e505da72c98bffcf3e`.
- Quality `83464ee250883e186876195c4bd5421d27bce07a1968cd13a39b5243da34eb9b`.
- Native unpublished lab commit `ba60611d031f673a12ebbf2f133c583b4e12df7f`, prefill optimizations
  `59d2d45db19d642e457717bbf96d9523ca00fc1005315ff37b90dddc28448144`,
  and actual imported extension
  `e1398c20e1b1c4a0c91c8661790cd2543156c3556af2d5963c75dfbe4fc6c05b`.

Source8 remains immutable. Frozen checkpoint bytes must equal all loaded head
tensors. CPU snapshots are owned copies for mutation checks, not CPU inference.
The actual LM module and all 16 single-token label IDs are verified. Both the
recorded decision and fresh oracle delegate to `ResearchHead.score`: exactly one
query LM projection per score, the same label baseline, and no candidate LM
projection or alternate scoring math. Criterion, evidence and ordered labels
use canonical `query_suffix`; the oracle replays exact accepted token IDs and
the recorded suffix, without re-rendering/re-tokenizing generated B.

## Operator Interface

The original launcher and exact case files are not distributed. The module
accepted `--allow-inference`, `--mode procedural`, `--cases` and `--output`.

This is the small existing non-heldout procedural suite: accepted EOS/length/stop
boundaries, primitives, ordered/reordered candidates, exact accepted-AB/C replay,
image controls, and cancellation/failure cleanup. `observe_native_state: true`
is required; complete active hybrid-state observation and parent/branch/resource
invariants are retained, not replaced by feature agreement. Procedural mode
explicitly omits any full-window plan and records `operator_mode: procedural`.
Inputs are attested by the operator; no heldout dataset is opened or synthesized.

For the separate existing full-window plan, use the same launcher with
`--mode full-window`, `--cases` pointing to
`$FUZEV_ROOT/runs/native-real/qualification/native4-full-window-cases.json`, and a
distinct new output. This mode requires the explicit plan and enables it without
reducing the 262144-token context. Recorded B8/cache/prefill-optimization/memory settings come
from the frozen reference identity, not new CLI overrides. Full-window serving
capacity and simultaneous fresh-oracle capacity remain separate records; an
infeasible oracle is not an observed state comparison or permission to shrink
the window. No performance branch or measurement module is modified here.

## Evidence Limits

Reports are exclusive new `.json` files in the experiment's report directory; existing files are
not overwritten. Sources, pins, loaded head and resource restoration are checked
again after the cases. Service and backend close before the final report write;
startup/close failures produce failed execution, never a published success.

Exit 0 means only successful harness execution, not runtime qualification.
`qualified: false`, experimental/uncalibrated/nonservable provenance, incomplete
checks and limitations remain intact. Inspect actual per-case and per-check results.
Images retain the expected-answer/sensitivity controls; experimental vision can
fail them and must not be relabelled passed. Full-window, forward/prefill coverage,
complete hybrid state, and performance still require their real runtime evidence.
Backbone weight-byte verification is not claimed by this binding.

CPU fixtures test binding refusals, snapshot ownership, canonical baseline and
encoding delegation, and cleanup/report order only. They do not prove CUDA math,
native state correctness, image quality, capacity or speed.

Historical result: **278 focused CPU tests passed**, including the legacy
qualification, CLI, state integration and observer regressions.

[Snapshot methods and environment](../README.md)
