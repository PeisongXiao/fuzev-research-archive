# Research Tools

Run these tools from the unpacked archive. The
[training walkthrough](../docs/train-a-shared-head.md) supplies feature schemas
and commands; [installation](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md) prepares serving.

| Tool | Reproduces | Packages | Inputs |
| --- | --- | --- | --- |
| `extract_hf_features.py` | Generic text-encoding feature extraction | Torch, Transformers, safetensors | Local HF model, prepared JSONL examples and output directory; writes backbone and encoding metadata |
| `train_shared_head.py` | Fitting, calibration and loader-format export | fuzev-runtime, Torch, safetensors | FIT/CAL feature JSONL, backbone config, extractor encoding metadata, geometry and output directory |
| `small_http_performance.py` | Repeated generation/decision/mixed HTTP latency samples | Python standard library | Running server URL, model alias, optional API key, stream/sample counts and output JSON |
| `smoke_head.py` | Discovery, generation and SystemOne checks | Python standard library | Running server URL, model alias, optional API key and output JSON |
| `smoke_staggered_workloads.py` | Decisions during an active generation request | Python standard library | Running server URL, model alias, optional API key, delay, output budget and output JSON |

The HTTP clients use the existing server. Model loading, compilation and
scheduling belong to that server's engine. Recorded runs are indexed under
[results](../docs/results/README.md).
