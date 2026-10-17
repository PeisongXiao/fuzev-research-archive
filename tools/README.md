# Research Tools

Run these tools from the unpacked archive. The
[training walkthrough](../docs/train-a-shared-head.md) supplies feature schemas
and commands; [installation](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md) prepares serving.

| Tool | Reproduces | Packages | Inputs |
| --- | --- | --- | --- |
| `extract_hf_features.py` | Generic text-encoding feature extraction | Torch, Transformers, safetensors | Local HF model, prepared JSONL examples and output directory; writes backbone and encoding metadata |
| `train_shared_head.py` | Fitting, calibration and loader-format export | fuzev-runtime, Torch, safetensors | FIT/CAL feature JSONL, backbone config, extractor encoding metadata, geometry and output directory |
| `small_http_performance.py` | Repeated generation/decision/mixed HTTP latency samples | Python standard library | Running server URL, model alias, optional API key, stream/sample counts and output JSON |
| `smoke_head.py` | Discovery, generation and SystemOne checks | Python standard library | Running server URL, model alias, optional API key and output JSON |
| `smoke_staggered_workloads.py` | Decisions during an active generation request | Python standard library | Running server URL, model alias, optional API key, delay, output budget and output JSON |
| `train_shared_head_v2.py` | Ragged fitting, calibration and held-out evaluation | fuzev-runtime 1.1.0, `pip install './research-snapshot[data]'`, PyTorch, safetensors | Bundle/config and FIT/DEV/CAL/EVAL ragged features; `train`, `calibrate`, `evaluate` |
| `extract_shared_head_v2.py` | Native final-token and candidate feature extraction | fuzev-exllamav3-plugin 1.1.0, exllamav3-fuzev v1.1 | Prepared JSONL, head bundle, model and shard output directory |
| `ragged_features.py` | Ragged shard read/write and historical replay import | fuzev-runtime 1.1.0, PyTorch, safetensors; the replay importer also needs Transformers and `pip install './research-snapshot[data]'` | Feature tensors and row metadata; the v1.0 feature cache read by the replay importer is unpublished |
| `representation_capture.py` | Capture of the control representation used by visual selection | fuzev-exllamav3-plugin 1.1.0, PyTorch | Native service decision inputs and final-token events |
| `representation_probe.py` | CPU `rescore` of saved visual features | fuzev-runtime 1.1.0, `pip install './research-snapshot[data]'`, PyTorch, safetensors | Capture output, head bundle and result path |
| `prepare_wands.py` | Grouped WANDS FIT/DEV/CAL/TEST data | Python standard library | WANDS query, product and label files, earlier prepared JSONL to exclude, seed and output directory |
| `prepare_sugarcrepe.py` | SugarCrepe replace-att image groups and AB/BA views | Pillow, fuzev-runtime 1.1.0 | Exclusion COCO IDs, seed, role group counts and output; images resolved against an image directory |
| `visual_capture.py` | Visual FIT/DEV or EVAL control logits | fuzev-exllamav3-plugin 1.1.0, exllamav3-fuzev v1.1 | `--representation-input`, `--image-dir`, `--output` and native server options |
| `residual_blend.py` | FIT residual/temperature fitting, DEV selection and EVAL reporting | `pip install './research-snapshot[data]'` (NumPy, SciPy) | Prepared records and capture scores, temperature bounds, bootstrap draws and seed |
| `comparison_kev.py` | Served breadth-v1 metrics and paired bootstrap | Kev at tag `kev-1.0`, installed in its own environment | Kev suite directory, running server and optional API key, or saved `rows.json`; `PYTHONPATH="$KEV/scripts"` |
| `comparison_vision.py` | Image metrics, backbone-label control and lossless-WebP retries | `pip install './research-snapshot[data]'`, httpx, Pillow; `serve` also needs fuzev-exllamav3-plugin 1.1.0 | Repeated `--set NAME=PATH`, `--image-root`, server URL and optional API key, or recorded attempts |
| `prefill_bench.py` | Fresh, append and comparison C/D timing | PyTorch, ExLlamaV3 build under test, flashinfer-python | Model, corpus, output, required `--chunks`; optional memory/cache/context and profile variants |
| `make_plots.py` | Six comparison charts, PNG and SVG | NumPy, matplotlib | `--results` published comparison JSON directory, `--out` chart directory |
| `shared_prefix_timing.py` | Grouped/concurrent SystemOne shared-prefix wall time and separate execution counters | Python standard library | `--url`, `--model`, optional `--api-key`, shared `--context`, JSON `--questions`, `--samples`, `--output`; optional `--environment` (device, software and server settings JSON) and `--context-source` |

The HTTP clients use the existing server. Model loading, compilation and
scheduling belong to that server's engine. Recorded runs are indexed under
[results](../docs/results/README.md).

The [v1.1 walkthrough](../docs/train-v1.1-head.md) gives the WANDS and
SugarCrepe preparation, extraction, replay-import, training, calibration,
`evaluate`, visual capture and `rescore` commands. The
[comparison method](../docs/v1.0-v1.1-comparison.md#method) gives the Kev,
speed, image and plot commands.

Run the CPU tool tests from the unpacked archive after installing the packages
in the table. `comparison_kev.py` and its test run in a separate environment
because Kev 1.0 declares PyTorch >=2.6,<2.9; `$KEV` is the unpacked `kev-1.0`
source:

```bash
PYTHONPATH=tools:research-snapshot python -m pytest tools -q --ignore=tools/test_comparison_kev.py
python -m venv .venv-kev
.venv-kev/bin/pip install "$KEV" pytest
PYTHONPATH="$KEV/scripts" .venv-kev/bin/python -m pytest tools/test_comparison_kev.py -q
```

`prefill_bench.py` defaults to two repetitions, fresh prompt lengths
2k/4k/8k/32k/64k and FP32 recurrent state. Comparison decode cells use the
prompt lengths up to `--decode-max-length` (default 32,768). Cache and context
default to the page-rounded largest prompt plus outputs and draft margin.
`--use-gib` supplies an optional loader allowance. Profile variants are applied
before model construction and loading, including model-bound MoE limits and GDN
layer selection. Each chunk/length/variant cell loads one target model outside
timing; its warmup and measured repetitions share that model. Corpus offsets
remain paired across variants.

The image tools (`prepare_sugarcrepe.py`, `visual_capture.py`,
`representation_capture.py`, `representation_probe.py`, `residual_blend.py`,
`comparison_vision.py`) serve an experimental extension; run them at your own
risk.
