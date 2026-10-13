# Component Setup

Install the runtime and one engine integration with the
[index installation guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md), which covers wheels,
unpacked source directories and engine-specific prerequisites.

| Component | Install and use |
| --- | --- |
| Shared Python head | [fuzev-runtime](https://github.com/PeisongXiao/fuzev-runtime) |
| vLLM decisions and server | [fuzev-vllm-plugin](https://github.com/PeisongXiao/fuzev-vllm-plugin) |
| ExLlamaV3 decisions and launcher | [fuzev-exllamav3-plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin) |
| vLLM engine build | [vllm-fuzev](https://github.com/PeisongXiao/vllm-fuzev/blob/main/docs/fuzev-fork.md) |
| ExLlamaV3 engine build and optional prefill tuning | [exllamav3-fuzev](https://github.com/PeisongXiao/exllamav3-fuzev/blob/main/doc/fuzev-fork.md) |
| EXL3 weights in vLLM (optional, AGPL-3.0-only) | [vllm-exl3-plugin](https://github.com/PeisongXiao/vllm-exl3-plugin/blob/main/docs/fuzev-serving.md) |

- [API guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md): SDK/curl requests to
  `POST /v1/systemone`, responses and optional FuZev extensions.
- [Head card](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev): the head bundle.
- [Training walkthrough](train-a-shared-head.md): training a head.
- [component-install-smoke.md](component-install-smoke.md): historical component
  installation results.

## Limitations

- GB10 is the tested platform.
- The engines require CUDA; the shared head runs on CPU or CUDA.
- [Installation history](component-install-smoke.md) gives the prepared-environment
  check scope; [evaluation](research-evaluation.md) gives the measured image and
  decision-quality scope.
