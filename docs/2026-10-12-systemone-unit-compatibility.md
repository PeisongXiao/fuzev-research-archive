# SystemOne CPU Compatibility History

The 2026-10-12 transport checks compared the public `typesafe-sdk` client
(tested with 0.6.0) against the existing decision services through in-process
ASGI. This record describes the earlier measured revisions; the
[API reference](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md)
documents current behavior.

## Recorded Coverage

| Suite | Passed | Coverage |
| --- | ---: | --- |
| Common SystemOne | 14 | Request mapping, model attribution, structured legends and independent questions |
| Native routes and service parity | 54 | SDK versus Generator service, model discovery, IDs, prefixes, cancellation and cleanup |
| vLLM fixtures | 51 | SDK versus decisions route, feature inputs, active IDs and cancellation |
| Shared head | 25 | Tensor scoring, artifact loading and calibration |
| vLLM pooling | 2 | Readout control and label validation |
| Native Generator image fixtures | 24 | Standalone images, retained text plus new image, embeddings, isolation and cleanup |

Counts apply to the recorded revisions. Current suites cover the released
Noul ordering, rendering, concurrency and input-validation contracts.

The SDK comparison checked distributions, selection, Yes probability, expected
rating, confidence and logical input usage.

- Ordinal JSON keys were strings; the SDK converted them to integers.
- Structured Score legends kept their JSON values.
- Responses reported the configured model.

Cancellation fixtures:

- mapped the public request ID to active work
- rejected duplicate active IDs with 409
- accepted repeated cancellation with 202 while draining
- returned 499 with the original request ID
- returned 404 after cleanup

Ordinary generation remained available on the injected engine fixture.

## Historical Sources

All revisions below are **unpublished lab commit IDs**.

| Component | Revision |
| --- | --- |
| Runtime | `f4a4e64467da068819d317952920106c24572700` |
| vLLM plugin | `1d9ca0905dc44d09effa194495861439d7ee1104` |
| Native route/parity suite | `e36fd72d6b08c5aabb9ee232d532c8d0eed883d6` |
| Native image fixtures | `72c03c08f61ed625a7b6312d4bef3bed6dee8465` |
| vLLM pooling/readout | `8e1cf93c9dd256b0bb4aa6dc0bfc0a0ca8529e41` |

- Lightweight route checks: Python 3.13.13.
- Tensor fixtures: Python 3.12.3, PyTorch 2.13.0+cu130, Linux aarch64, CUDA hidden.

## Reproduce Current CPU Checks

Use downloaded source directories, which include the tests. From their parent
directory, create an environment and install the runtime, plugin and test
dependencies together:

```bash
python3 -m venv .venv-systemone
. .venv-systemone/bin/activate
python -m pip install './fuzev-runtime[test]' \
  './fuzev-exllamav3-plugin[test]' typesafe-sdk
export CUDA_VISIBLE_DEVICES=""
(cd fuzev-runtime && python -m pytest -q tests/test_systemone.py tests/test_head.py)
(cd fuzev-exllamav3-plugin && python -m pytest -q)
```

Run `(cd fuzev-vllm-plugin && python -m pytest -q tests/test_adapter.py)` in a
v1.0 serving environment with the vllm-fuzev fork installed
([v1.0 installation guide](https://github.com/PeisongXiao/fuzev/blob/v1.0/docs/install.md#vllm)).
Engine-fork pooling checks use the fork's own test environment.

These commands target the supplied source, so counts can differ from the
recorded table.

## Limitations

- The recorded image fixtures supported standalone images and new images after
  a retained text prefix. They did not cover prefixes retained from generation
  with an input image.
- The checks used synthetic CPU backends, with no listening server or inference engine.

Current image and end-to-end evidence scope is in [evaluation](research-evaluation.md).
