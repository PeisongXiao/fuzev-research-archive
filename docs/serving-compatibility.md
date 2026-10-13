# Serving Compatibility

FuZev attaches a shared decision head to a running vLLM engine or ExLlamaV3
Generator.

- [API reference](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md): the public
  SystemOne contract, limits and optional extensions.
- Engine setup: [vLLM plugin](https://github.com/PeisongXiao/fuzev-vllm-plugin) and
  [ExLlamaV3 plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin).

## Feature Contract

- **Inputs:** final hidden states before the vocabulary projection, plus the
  existing LM head's query-label logits. Candidate continuations start from the
  same query prefix.
- **Bundle:** declares geometry, primitive types, maximum options and encoding
  version. Matching width alone does not establish useful quality on another
  backbone; train a new head with the [training guide](train-a-shared-head.md).
- **Lifecycle:** an injected engine stays under application control. Close the
  routes from `add_systemone_routes` before the decision service; the
  application closes its Generator.
- **Prefixes:** retained prefixes store exact accepted token sequences; every
  decision is scored afresh.
- **Images:** native image decisions use the application's vision callback and
  Generator jobs. Standalone images and new images after retained text prefixes
  are supported. ExLlamaV3 prefixes from generation requests with an image keep
  the image embeddings with the accepted tokens.

## Measurement Scope

- [Evaluation](research-evaluation.md): measured builds, batch geometry and image coverage.
- [Benchmarks](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md): decision
  quality and repeat variation.
- [Integration](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md):
  installation checks and generation-only comparisons.

## Post-Fix Executed Scope

The corrected NVIDIA NVFP4 checks measured these builds
(**unpublished lab commit IDs**):

| Component | Commit |
| --- | --- |
| vLLM | `5ccd64a0ac1bd7d27f71255faa3439c7e4f25e0e` |
| Runtime | `46de6161561ce55dbc3bd1e4d4736b04240c0be0` |
| vLLM plugin | `38e446d7101f6f6e101ad7ef90eaebc00a611b5d` |

## Limitations

- The [API limitations](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#limitations)
  list current option limits, supported topology and experimental image behavior.
- The [CPU transport checks](2026-10-12-systemone-unit-compatibility.md) establish
  client and route behavior.
