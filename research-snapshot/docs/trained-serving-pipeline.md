# Synthetic Trained Head Serving

## What This Exercises

The existing service tests use mocked scorers or manually fabricated head
metadata. `test_authentic_trained_serving_pipeline` instead trains and calibrates
a genuine local head, then hands its unchanged bundle to actual `ServingHead`,
`NativeLane`/`NativeService`, and `VllmService` code.

This is CPU integration evidence from explicitly synthetic inputs and backends.
Actual runtime qualification remains unrun. No real model is loaded, no image
is decoded, and no dataset, GPU, endpoint, or HTTP request is used. Generation
calls below consume predetermined CPU fake outputs solely to establish accepted
prefixes.

## Authentic Fixture and Source Identities

The test reused `_build_corpus`, `_train`, and the no-engine import guard from
the trained-review pipeline test. The only change to that helper added
optional `encoding` and `runtime_sources` arguments at corpus construction;
its existing defaults and lifecycle were preserved. There is no duplicate corpus
builder and no manifest repair after training.

Before any feature is written, this fixture declares the real portable encoding
descriptor from `fuzev_research.encoding.identity()` and a content-hashed extraction
recipe record. Its engine/model identity remains explicitly synthetic. The
encoding descriptor selects the admission contract; the stored random vectors
are synthetic fixtures rather than measured outputs of that encoding or a
backbone.

There are 48 random synthetic text/image-feature rows, complete JSONL roots,
manifests, preparations, SQLite ownership, and 2,560-dimensional Safetensors
features. Noul uses `yes`/`no`, ordinal values are increasing, and finite
candidate-aligned synthetic baseline logits are present. All 12 text and all
12 mixed profiles fit for one explicitly recorded synthetic epoch, followed by
actual calibration. Production ten epochs, patience two, batching, optimization,
readiness, admission, and MemoryGuard remain unchanged. Inherited audits verify
all fitting/development/calibration memberships and exclude test roles from
training, selection, and calibration.

Serving receives a separate content-hashed record covering the serving modules
and new serving test. It is distinct from the extraction helper record and the
trainer's actual `training_code` hashes. The distinction uses actual source
contents, without inventing a real engine execution or rewriting the head's
recorded source identity.

## Scoring Checks

- Actual `ServingHead` scores each primitive on stored synthetic features.
  Its logits, raw softmax, and calibrated softmax agree with the trained
  `HeadBundle` and its authentic temperatures. Heads stay FP32, frozen, and
  on CPU.
- Actual vLLM `build_service` accepts the original extraction identity with
  separate, explicitly synthetic serving cell/source pins. Only its injected
  runtime inspector and CPU API/engine are fake. The default `ServingHead`,
  memory admission, and MemoryGuard run unchanged.
- `VllmService` makes standalone and retained-prefix decisions for Choice,
  Noul, and ordinal. Tests verify candidate order, logits/probabilities, exact
  query/tail tokens, calibration/manifest IDs, and unchanged accepted AB.
  Reused prefixes use full token replay; no hybrid checkpoint or inferred
  forward counts are claimed. The existing fake engine mutates output scratch
  after yielding, so oracle agreement also checks feature/token ownership.
- Actual `NativeLane` performs branch creation, freezing, forking, partial-page
  copy-on-write, batched tails, and final-token resolution under `NativeService`.
  All three primitives run standalone and from one retained prefix. Scores
  match the authentic head oracle, retained tokens/checkpoint remain unchanged,
  and all branches, pages, and fake hybrid slots are returned at cleanup.
- Serving weights/temperatures remain identical in memory. The source
  `head.safetensors`, `manifest.json`, and `calibration.json` bytes and hashes,
  prepared/feature files, training/serving source hashes, and native source
  hashes survive the full lifecycle unchanged.

## Native Admission Boundary

The native builder requires an `optimized-native-exllamav3-fuzev` extraction
identity and matching native geometry/source pins. The authentic fixture records
`synthetic-cpu-no-engine`, so that builder rejects it before head admission.
This expected rejection is tested; no native identity is substituted to force
a pass. The successful native decisions use the existing dependency-injection
constructor with actual `ServingHead`, actual `NativeLane`, and a CPU fake backend.
Successful production native-builder admission requires genuine native features
and remains outside this synthetic check.

`NativeLane` itself is read from the pinned native tree, copied byte-for-byte into
the owned fixture with `constants.py`, and loaded under an isolated Python
namespace. Those files import only standard Python and Torch. The native package
bootstrap and extensions are never imported. The fake `State`, `Cache`, and
`Model` helpers are reused from `tests/test_fuzev_lanes.py` by parsing its AST
and executing only those class definitions. A small CPU model subclass provides
2,560-wide feature scratch and predetermined vocabulary outputs. The recipes
fake Backend and vLLM Engine/API helpers are also reused.

`NativeLane` source exercised on the initial run:
`cc269b9cdb63dab1b32077872ceda674001afaa48ad4838e8c419061b318694f`.
Source copies and original files are checksum-checked; no native file is edited.

## Rejections and Bounds

The test rejects an incompatible extraction encoding in `ServingHead`, an
incompatible portable encoding at the vLLM builder, serving-source pins
substituted for extraction provenance, a mismatched serving record, and an
incompatible manifest even when that owned copy is correctly repinned. Builder
failures perform no fake engine forward. Only the negative manifest copy is
modified; the authentic calibrated bundle remains untouched.

A fresh worker preserves the existing no-native/no-vLLM import guard throughout
fitting and serving. CUDA is hidden and remains uninitialized; Torch uses one
intra-op and one inter-op thread. HF is offline and bytecode is disabled. All
custom artifacts and caches use a unique temporary `FUZEV_ROOT` below this
fixture's `.pytest_cache`, removed on success, failure, or worker timeout. The
worker has a 180-second deadline and a final artifact-size check below 2 GiB.
The size check is not a disk quota; abrupt termination of outer pytest cannot
guarantee Python cleanup. Existing model and Docker caches are never deleted.

## Recorded Runs

The serving fixture read `exllamav3/fuzev.py`, `exllamav3/constants.py`
and `tests/test_fuzev_lanes.py` from
[historical-native-source](../historical-native-source/). These unchanged files
come from unpublished lab commit `ba60611d031f673a12ebbf2f133c583b4e12df7f`
of the ExLlamaV3 fork. The tree includes upstream and FuZev source under the
[MIT license](../historical-native-source/LICENSE). The published snapshot omits
this test, because it needs the unpublished dataset scope of the training run.

Initial serving run: **1 passed in 7.17 seconds**, six native and six vLLM
decisions, 1,167,334,011 temporary bytes, cleanup confirmed. The native builder boundary
above is the only limitation to a synthetic positive admission test.

## Recorded Follow-Up

The [qualification-candidate record](qualification-candidate.md) holds the
combined timing and per-worker RSS observations.

[Snapshot methods and environment](../README.md)
