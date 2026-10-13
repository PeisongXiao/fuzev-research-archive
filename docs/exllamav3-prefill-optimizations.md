# ExLlamaV3 Prefill Optimizations

[exllamav3-fuzev](https://github.com/PeisongXiao/exllamav3-fuzev) packages the
prefill optimizations as `exllamav3.optimized`. The
[canonical install guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#exllamav3) and
the fork's [tuning guide](https://github.com/PeisongXiao/exllamav3-fuzev/blob/main/doc/fuzev-fork.md#prefill-optimizations)
cover default activation, kernel options, per-device fallback and the
`EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1` opt-out.

- Activation is idempotent and applies process-wide prefill patches.
- Generator chunk size is an engine setting; the FuZev launcher defaults to
  8,192 tokens (`--max-chunk-size`).
- FlashInfer runs in the model's activation dtype.
- The [ExLlamaV3 plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin)
  loads the decision head.

## Historical Measurement Scope

- The recorded measurements used an extension predating the
  [shipped CUDA-graph kernel-argument fix](research-history.md#superseded-serving-results).
  See [integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md).
- Earlier packaging fixtures covered the source and wheel import layout with
  stand-in engine APIs. They did not compile or exercise the CUDA extension.
- `EXLLAMA_NOCOMPILE` controls build-time compilation only. Importing an engine
  without its compiled extension may invoke upstream JIT compilation.
