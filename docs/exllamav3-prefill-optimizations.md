# ExLlamaV3 Prefill Optimizations

[exllamav3-fuzev](https://github.com/PeisongXiao/exllamav3-fuzev/tree/v1.1) builds the
prefill optimizations into its attention, GDN, expert and recurrent-cache
modules; `exllamav3.optimized` holds their options. The
[canonical install guide](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md#exllamav3) and
the fork's [tuning guide](https://github.com/PeisongXiao/exllamav3-fuzev/blob/v1.1/doc/fuzev-fork.md#prefill-optimizations)
cover default activation, kernel options, per-device fallback and the
`EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1` opt-out.

- `enable_prefill_optimizations` sets process-wide options before a model loads.
- Generator chunk size is an engine setting; the FuZev launcher defaults to
  32,768 tokens from the GB10-tuned profile, capped at the cache tokens
  (`--max-chunk-size`).
- `fuzev-native --prefill-profile` takes a JSON object or file path and sets
  profile options before the model loads.
- FlashInfer GDN prefill takes BF16 q, k and v, with FP32 gates, beta and state.
- The [ExLlamaV3 plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1)
  loads the decision head.

## Historical Measurement Scope

- The recorded measurements used an extension predating the
  [shipped CUDA-graph kernel-argument fix](research-history.md#superseded-serving-results).
  See [integration results](integration-v1.0.md).
- Earlier packaging fixtures covered the source and wheel import layout with
  stand-in engine APIs. They did not compile or exercise the CUDA extension.
- `EXLLAMA_NOCOMPILE` controls build-time compilation only. Importing an engine
  without its compiled extension may invoke upstream JIT compilation.
