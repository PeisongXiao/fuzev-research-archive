# Third-Party Notices

FuZev Research Archive
Copyright 2026 Peisong Xiao

Authored source and documentation use [Apache-2.0](LICENSE), including the
[research snapshot](research-snapshot/LICENSE).
`research-snapshot/historical-native-source/` contains ExLlamaV3 source by
turboderp with FuZev additions by Peisong Xiao, under its
[MIT license](research-snapshot/historical-native-source/LICENSE).
Copyright and template attribution also appear in [NOTICE](NOTICE) and
[research-snapshot/NOTICE](research-snapshot/NOTICE).

FuZev's learned head uses Apache-2.0. Backbone weights, source datasets and
underlying text and images retain their own terms. This archive does not
redistribute backbone weights or dataset contents.


## Models And Encoding

- [Qwen3.8-Flash-Next, Qwen team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
- [EXL3 quantizations, turboderp](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3):
  [Qwen Community License 1.0](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/main/LICENSE).
- [NVFP4 quantization, NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4):
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
  and the additional Qwen Community License 1.0 terms stated in its card.
- [NVFP4 quantization, RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4):
  source-model Qwen Community License 1.0 terms.
- [Qwen Fixed Chat Templates, Frederic Guigand (froggeric),
  Juan Calderon-Perez and contributors](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates):
  [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). The training decision template is v22.3,
  source revision `3be7669fac73085a458e20e81ee9dc5c322aefb2`.
- [Kev, Jared Palmer and contributors](https://github.com/jaredpalmer/kev/blob/fc4a17e194bca35a48c57211392ec49ea54a94ac/LICENSE):
  Apache-2.0 for the evaluation code and suite; underlying datasets retain their terms.

## Training Datasets

| Source | Credit | Declared Terms |
| --- | --- | --- |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a) | NVIDIA and contributors | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3/tree/f6d145777bcbde96137596340fab89793acd1031) | NVIDIA and contributors | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [MM-RLHF](https://huggingface.co/datasets/yifanzhang114/MM-RLHF/tree/29cfeea5929979d12b42a8f30201e03c188408a6) | Yifan Zhang and contributors | [MIT](https://opensource.org/license/mit), as declared in the dataset card |
| [BoolQ](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5) | Clark et al., Google Research | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| [WANDS](https://github.com/wayfair/WANDS) | Chen, Liu, Liu, Sun, Baltrunas and Schroeder, Wayfair (ECIR 2022) | MIT, Copyright (c) 2021 ecir2022; v1.1 head FIT/DEV/CAL and held-out TEST rows |
| [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) `replace_att` | Hsieh, Zhang, Ma, Kembhavi and Krishna (NeurIPS 2023 Datasets and Benchmarks) | MIT, Copyright (c) 2023 RAIVN Lab; FIT/DEV select the v1.1 image residual strength and temperature, EVAL reports the image result |
| [COCO 2017 validation images](https://cocodataset.org/#termsofuse) | COCO Consortium | Annotations CC BY 4.0; images under the Flickr Terms of Use |

## Evaluation-Only Datasets

| Source | Credit | Declared Terms |
| --- | --- | --- |
| [QASPER](https://huggingface.co/datasets/allenai/qasper/tree/fdc9d8214fbab5dd782958601db4d678e6934a54) | Dasigi et al., Allen Institute for AI | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench/tree/0e6e62701eba92818a69ce95af0ed7aa0648b176) | VL-RewardBench authors | MIT metadata; the card also restricts use to research and invokes GPT-4o and Claude terms |

The following sources form Kev `breadth-v1`. Declarations and credits follow
[Kev's dataset manifest](https://github.com/jaredpalmer/kev/blob/fc4a17e194bca35a48c57211392ec49ea54a94ac/evals/breadth-v1/manifest.json).

| Source | Credit | Declared Terms |
| --- | --- | --- |
| [MuSR](https://huggingface.co/datasets/TAUR-Lab/MuSR) | Sprague et al. (ICLR 2024) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code [MIT](https://opensource.org/license/mit) |
| [SATA-Bench](https://huggingface.co/datasets/sata-bench/sata-bench) | Xu et al. (2025) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); code [MIT](https://opensource.org/license/mit) |
| [ChessBench](https://github.com/google-deepmind/searchless_chess) | Ruoss et al. (NeurIPS 2024) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); lichess positions [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | Koreeda and Manning (EMNLP 2021) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | Zellers et al. (ACL 2019) | [MIT](https://opensource.org/license/mit) (code and data release); Kev uses only the ActivityNet-derived items |
| [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) | Larson et al. (EMNLP 2019) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| [SGD / SGD-X](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | Rastogi et al. (AAAI 2020); Lee et al. (AAAI 2022) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) | Su et al. (ICLR 2025) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Patil et al. (ICML 2025) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [ToolRet](https://github.com/mangopy/tool-retrieval-benchmark) | Shi et al. (ACL 2025) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); underlying source terms also apply |
| [API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank) | Li et al. (EMNLP 2023) | [MIT](https://opensource.org/license/mit) |
| [RouterBench](https://github.com/withmartian/routerbench) | Hu et al., Martian (2024) | [MIT](https://opensource.org/license/mit) |
| [Humicroedit](https://cs.rochester.edu/u/nhossain/humicroedit.html) | Hossain et al. (SemEval 2020) | No license stated; upstream requests paper attribution |
| [cfcolor](https://www.dgp.toronto.edu/~donovan/cfcolor/) | O'Donovan, Agarwala and Hertzmann (CAe 2014) | Permissive program notice; no separate dataset license stated |

## Historical Acquisition

The research pipeline acquired and indexed A-OKVQA before the dataset-scope
change. A-OKVQA v1.0 question-to-COCO image-ID metadata (allenai/aokvqa) was
read to exclude previously used images from the SugarCrepe image sample.
No A-OKVQA question, answer or image is a training, calibration or evaluation
example.

| Source | Credit | Declared Terms |
| --- | --- | --- |
| [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA/tree/d1b0efa3a436e9101dfbde3752db7607da696c35) | Schwenk et al. (ECCV 2022) | No license declared in the acquired dataset card; images are from COCO |
| [Official A-OKVQA v1.0 metadata](https://github.com/allenai/aokvqa) | Schwenk et al., Allen Institute for AI | Repository Apache-2.0; no dataset license stated |

These declarations describe the upstream releases. Rights in underlying images,
text and generated responses remain governed by their source terms. Evaluation-only
data do not supply head-fitting examples.

The v1.0 research servers used the [vllm-exl3 plugin](https://github.com/PeisongXiao/vllm-exl3-plugin/tree/v1.0)
(AGPL-3.0-only); this archive contains none of its code.
