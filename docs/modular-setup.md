# Component Setup

Install the runtime and the ExLlamaV3 integration with the
[index installation guide](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md), which covers
unpacked source directories and engine-specific prerequisites.

| Component | Install and use |
| --- | --- |
| Shared Python head | [fuzev-runtime](https://github.com/PeisongXiao/fuzev-runtime/tree/v1.1) |
| ExLlamaV3 decisions and launcher | [fuzev-exllamav3-plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1) |
| ExLlamaV3 engine build and optional prefill tuning | [exllamav3-fuzev](https://github.com/PeisongXiao/exllamav3-fuzev/blob/v1.1/doc/fuzev-fork.md) |

v1.0 research also used fuzev-vllm-plugin, vllm-fuzev and vllm-exl3-plugin
(AGPL-3.0-only); fuzev-vllm-plugin 1.0 pairs with fuzev-runtime 1.0 only.

- [API guide](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md): SDK/curl requests to
  `POST /v1/systemone`, responses and optional FuZev extensions.
- [Head card](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev/tree/v1.1): the head bundle.
- [Training walkthrough](train-a-shared-head.md): training a head.
- [component-install-smoke.md](component-install-smoke.md): historical component
  installation results.

## Limitations

- GB10 is the tested platform.
- The engines require CUDA; the shared head runs on CPU or CUDA.
- [Installation history](component-install-smoke.md) gives the prepared-environment
  check scope; [evaluation](research-evaluation.md) gives the measured image and
  decision-quality scope.
