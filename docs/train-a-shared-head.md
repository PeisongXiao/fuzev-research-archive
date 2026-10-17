# Train One Shared Decision Head

- [train_shared_head.py](../tools/train_shared_head.py) trains a shared head from frozen
  features, calibrates per-primitive temperatures and exports a `fuzev-runtime` bundle.
  It reads width from `text_config.hidden_size`, falling back to `hidden_size`;
  projection/adapter sizes are arguments.
- [extract_hf_features.py](../tools/extract_hf_features.py) extracts local HF causal LM
  features on the selected device and dtype.

## Environment

Use Python 3.11 or newer (tested with 3.12). Run from the unpacked archive root and set the runtime source path:

```bash
export FUZEV_RUNTIME_SOURCE=/absolute/path/to/fuzev-runtime
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "$FUZEV_RUNTIME_SOURCE"
python -m pip install 'transformers>=4.57' 'huggingface_hub>=1' 'pyarrow>=20'
```

## Tiny Public Backbone

This example uses the public [tiny random Llama](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/tree/9fb191250dd56d0ba7ec9785a025ed29c03d5998)
(hidden size **16**, two layers, 2,048-token context) and 320 real
[BoolQ](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5) rows.
Download revisions identify the example; model weights stay in the HF cache.

```bash
MODEL=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("hf-internal-testing/tiny-random-LlamaForCausalLM", revision="9fb191250dd56d0ba7ec9785a025ed29c03d5998", allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]))')
DATA=$(python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("google/boolq", "data/train-00000-of-00001.parquet", repo_type="dataset", revision="35b264d03638db9f4ce671b711558bf7ff0f80d5"))')
export RUN="$(mktemp -d)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/extract_hf_features.py prepare-boolq \
  --input "$DATA" --output "$RUN/requests" \
  --fit-rows 256 --dev-rows 32 --calibration-rows 32
python tools/extract_hf_features.py extract \
  --model-directory "$MODEL" \
  --input "$RUN/requests/fit.jsonl" "$RUN/requests/development.jsonl" "$RUN/requests/calibration.jsonl" \
  --output "$RUN/features" --device cpu --dtype float32
python tools/train_shared_head.py train \
  --backbone-config "$RUN/features/backbone-config.json" \
  --fit "$RUN/features/fit.jsonl" --development "$RUN/features/development.jsonl" \
  --calibration "$RUN/features/calibration.jsonl" --output "$RUN/head" \
  --projection-size 8 --query-adapter-size 4 --key-adapter-size 2 \
  --max-options 2 \
  --epochs 1 --batch-size 32 --learning-rate 0.001 --calibration-steps 10 \
  --device cpu --inspect
```

Preparation takes the first 320 distinct whitespace-normalized passages/questions,
in file order, from BoolQ's training split:

- 256 fitting (FIT), 32 development (DEV) and 32 calibration (CAL), with disjoint
  IDs and passage groups.
- Each is a Choice in **[Yes, No]** order with its Boolean label as target.
- DEV supplies post-export inspection. These subsets are separate from BoolQ's
  official development/test sets.

The CPU run extracted all 320 rows (maximum sequence length 1,082), performed
one epoch / **8 AdamW updates**, then exported and loaded through `ServingHead`.
First DEV Choice: logits `[0.07548210, -0.14430124]`, raw `[0.55472571, 0.44527429]`,
temperature `0.33768919`, calibrated `[0.65720087, 0.34279913]`.

| Tested step | Versions |
| --- | --- |
| Extraction and training | Python 3.12.3, PyTorch 2.13.0+cu130, Transformers 5.17.0, safetensors 0.8.0, Hub 1.32.0 |
| Preparation | Python 3.13.13, PyArrow 25.0.1, Hub 1.33.0 |

## Feature And Split Contract

Features:

- Keep the backbone and vocabulary head frozen.
- Append candidates independently to the identical query token prefix, in option order.
- Read final hidden states at the last consumed token before vocabulary projection.
  Copy the FP32 feature vectors before buffer reuse.
- `baseline` is the frozen LM's actual candidate-label logits at the query endpoint.
- Labels are single tokens (A/B IDs 319/350 here). Keep gold labels, targets and
  rationales separate from model inputs.

Extractor:

- Uses its versioned plain-text prompt without generation or caching, and checks
  context limits without truncation.
- Writes `encoding.json` beside `backbone-config.json`; training copies that
  encoding into the exported config.
- Defaults to `--device cpu --dtype float32`. For a CUDA backbone, use
  `--device cuda --dtype bfloat16` (or `float16` when supported). `--dtype auto`
  uses the backbone's configured weight dtype.
- Exported feature vectors remain FP32.

Each FIT, DEV or CAL JSONL line contains:

| Field | Meaning |
| --- | --- |
| `id` | Unique record identifier |
| `group` | Optional related-example group for split separation |
| `primitive` | `choice`, `noul`, or `ordinal` (Score) |
| `query` | Finite FP32-compatible list of length H |
| `candidates` | K finite lists of length H, in option order |
| `baseline` | K finite frozen LM label logits |
| `target` | K nonnegative probabilities summing to one; preserve votes and ties |

H is the backbone width; K is 2 through `--max-options`. Noul has exactly two entries,
**[Yes, No]**; ordinal levels stay increasing.

Splits:

- Group related conversations, documents, images and semantic duplicates before
  extraction. FIT/DEV/CAL IDs and groups are disjoint.
- FIT and CAL cover the same primitives; DEV uses those in FIT.
- FIT supplies gradients; CAL fits positive temperatures after freezing.
- Training ends at a fixed epoch. DEV is only for `--inspect` (CAL if omitted).
- Reserve DEV/TEST for selection and evaluation; external benchmarks stay held out.

## Train Another Backbone

Supply existing features and the backbone config with its extractor-written `encoding.json`:

```bash
export FUZEV_BACKBONE_CONFIG=/absolute/path/to/features/backbone-config.json
export FUZEV_FEATURES=/absolute/path/to/features
NEW_RUN=$(mktemp -d)
python tools/train_shared_head.py train \
  --backbone-config "$FUZEV_BACKBONE_CONFIG" \
  --fit "$FUZEV_FEATURES/fit.jsonl" --calibration "$FUZEV_FEATURES/calibration.jsonl" \
  --output "$NEW_RUN/head" \
  --projection-size 128 --query-adapter-size 32 --key-adapter-size 8 \
  --max-options 16 \
  --epochs 5 --batch-size 32 --learning-rate 0.001 \
  --calibration-steps 50 --device cpu --inspect
```

The geometry and optimization values above are starting choices.

- FP32 training uses shared query/key projections, SiLU adapters, channel gates and
  a query-only context gate.
- AdamW updates head parameters with gradient clipping at 1. Soft-target
  cross-entropy is divided by `log(K)` and averaged equally over rows.
- CPU is the default; `--device cuda` (or `cuda:0`) selects CUDA head training.

## Export And Score

The bundle contains `head.safetensors`, `config.json` and `calibration.json`.

- Config declares `hidden_size`, `projection_size`, `query_adapter_size`,
  `key_adapter_size`, `primitives`, `max_options`, `encoding_version` and `release_date`.
- The runtime reads the geometry and loads weights strictly.
- Raw temperature-1 scoring is the default.

To score the first DEV row:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import torch
from fuzev.head import ServingHead
run = Path(os.environ["RUN"])
row = json.loads((run / "features/development.jsonl").read_text().splitlines()[0])
with ServingHead(run / "head", device="cpu", calibration="calibrated") as head:
    logits, raw, calibrated = head.score(
        row["primitive"], torch.tensor(row["query"]), torch.tensor(row["candidates"]),
        baseline=torch.tensor(row["baseline"]),
    )
    print({"logits": logits, "raw": raw, "calibrated": calibrated})
PY
```

## Limitations

- This portable baseline uses the training procedure above. The v1.0 head's
  specialized loss and five-epoch refit are in the [research history](research-history.md);
  the v1.1 head's recipe is [train-v1.1-head.md](train-v1.1-head.md).
- The tiny random model checks extraction, training and loading. This recorded run
  verifies CPU execution.
- Heads using the walkthrough's `fuzev-hf-text-branches-v1` encoding load and score
  through the runtime Python API in v1.1. The v1.1 ExLlamaV3 plugin serves only the
  `fuzev-candidate-branches-v1` and `fuzev-list-aware-v2` Qwen chat encodings
  ([Encodings](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/models.md#encodings)).
- JSONL stays in host memory, and CAL batches one primitive at a time. Size inputs
  and batches accordingly.
- For distribution, supply a model card and notices. Keep data, features, optimizer
  state and backbone weights separate.
- BoolQ uses [CC BY-SA 3.0](https://github.com/google-research-datasets/boolean-questions);
  the tiny model card specifies no license.
