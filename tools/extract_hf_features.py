"""Prepare a small BoolQ split or extract frozen features from a local HF causal LM."""

import argparse
import json
from pathlib import Path
import shutil
import sys


ENCODING_VERSION = "fuzev-hf-text-branches-v1"


def prepare_boolq(args):
    if args.input.suffix == ".parquet":
        import pyarrow.parquet as parquet
        source = parquet.read_table(args.input).to_pylist()
    else:
        source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    sizes = dict(fit=args.fit_rows, development=args.dev_rows, calibration=args.calibration_rows)
    rows, passages, questions = [], set(), set()
    for index, row in enumerate(source):
        passage = " ".join(row["passage"].split())
        question = " ".join(row["question"].split())
        if passage in passages or question in questions:
            continue
        passages.add(passage)
        questions.add(question)
        rows.append(dict(id=f"boolq-train-{index}", group=passage, primitive="choice",
                         question=row["question"], context=row["passage"],
                         candidates=[{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
                         target=[1.0, 0.0] if row["answer"] else [0.0, 1.0]))
        if len(rows) == sum(sizes.values()):
            break
    if len(rows) != sum(sizes.values()):
        raise ValueError("Not enough distinct passages/questions for the requested split sizes")
    args.output.mkdir(parents=True, exist_ok=False)
    start = 0
    for role, count in sizes.items():
        with (args.output / f"{role}.jsonl").open("w") as handle:
            for row in rows[start:start + count]:
                handle.write(json.dumps(row) + "\n")
        start += count
    return dict(output=str(args.output), rows=sizes, source_rows=len(source))


def tokenize(row, tokenizer):
    choices = "\n".join(f"({chr(65 + i)}) {candidate['text']}" for i, candidate in enumerate(row["candidates"]))
    prompt = ("Read the context and answer the question. Return only the candidate label.\n"
              f"Context:\n{row.get('context', '')}\nQuestion: {row['question']}\nCandidates:\n{choices}\nAnswer:")
    query = tokenizer.encode(prompt, add_special_tokens=True)
    tails = [tokenizer.encode("\nCandidate under consideration:\n" + candidate["text"], add_special_tokens=False)
             for candidate in row["candidates"]]
    labels = [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(len(tails))]
    if any(len(label) != 1 for label in labels):
        raise ValueError("This encoding requires one-token candidate labels for the selected tokenizer")
    return query, tails, [label[0] for label in labels]


def extract(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    config = AutoConfig.from_pretrained(args.model_directory, local_files_only=True)
    text_config = config.get_text_config()
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory, local_files_only=True)
    prepared = {}
    longest = 0
    for path in args.input:
        prepared[path.stem] = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            query, tails, labels = tokenize(row, tokenizer)
            length = len(query) + max(map(len, tails))
            if length > text_config.max_position_embeddings:
                raise ValueError(f"{row['id']} exceeds the backbone context; input is not truncated")
            longest = max(longest, length)
            prepared[path.stem].append((row, query, tails, labels))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_directory, local_files_only=True, use_safetensors=True,
        dtype="auto" if args.dtype == "auto" else getattr(torch, args.dtype),
    ).to(args.device).eval().requires_grad_(False)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.model_directory / "config.json", args.output / "backbone-config.json")
    (args.output / "encoding.json").write_text(
        json.dumps({"encoding_version": ENCODING_VERSION}, indent=2) + "\n")
    counts = {}
    with torch.inference_mode():
        for role, records in prepared.items():
            with (args.output / f"{role}.jsonl").open("w") as handle:
                for row, query_ids, tails, labels in records:
                    query = model(input_ids=torch.tensor([query_ids], device=device), use_cache=False,
                                  output_hidden_states=True, logits_to_keep=1, return_dict=True)
                    keys = [model.base_model(input_ids=torch.tensor([query_ids + tail], device=device), use_cache=False,
                                             return_dict=True).last_hidden_state[0, -1].float().cpu().clone()
                            for tail in tails]
                    features = {key: row[key] for key in ("id", "primitive", "target")}
                    if "group" in row:
                        features["group"] = row["group"]
                    features.update(query=query.hidden_states[-1][0, -1].float().cpu().tolist(),
                                    candidates=torch.stack(keys).tolist(), baseline=query.logits[0, -1, labels].float().cpu().tolist())
                    handle.write(json.dumps(features, allow_nan=False) + "\n")
            counts[role] = len(records)
            print(f"Extracted {role}: {len(records)} rows", file=sys.stderr, flush=True)
    return dict(output=str(args.output), rows=counts, hidden_size=text_config.hidden_size,
                encoding_version=ENCODING_VERSION, max_sequence_tokens=longest,
                device=str(device), dtype=str(model.dtype),
                cuda_initialized=torch.cuda.is_initialized())


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive count")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-boolq", help="Split distinct passages from public BoolQ training rows")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("fit-rows", "dev-rows", "calibration-rows"):
        prepare.add_argument("--" + name, type=positive_int, required=True)
    features = commands.add_parser("extract", help="Extract FP32 query/candidate features and actual LM controls")
    features.add_argument("--model-directory", type=Path, required=True)
    features.add_argument("--input", type=Path, nargs="+", required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--device", default="cpu", help="Backbone device (default: cpu)")
    features.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                          default="float32", help="Backbone weight dtype (default: float32)")
    args = parser.parse_args(argv)
    result = prepare_boolq(args) if args.command == "prepare-boolq" else extract(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
