"""Extract native final-token features into resumable, role-separated ragged shards.

Use the public native server arguments, plus --input, --output and --shard-bytes.
--head accepts a v1 or v2 bundle. A v1 bundle gets a v2 extraction copy under
output/extraction-head with unchanged weights. Completed shards and the
extraction copy are reused on resume. No scores or metrics are saved.
"""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from fuzev.encoding import LIST_AWARE_VERSION, VERSION, candidate_suffix
from fuzev.head import V2_SCHEMA, read_artifacts, read_v2
from fuzev.service_contracts import ServiceDecisionRequest
from ragged_features import GEOMETRY, ROLES, RaggedWriter, json_rows


def extraction_head(source, directory):
    source, directory = Path(source).resolve(), Path(directory)
    metadata, _, _ = read_artifacts(source)
    if metadata.get("schema") == V2_SCHEMA:
        return metadata
    if directory.exists():
        return json.loads((directory / "config.json").read_text())
    feature = {"tap": "final-pre-lm", "pooling": "last-token"}
    config = {key: metadata[key] for key in GEOMETRY}
    config.update(schema=V2_SCHEMA, architecture_version="fuzev-shared-decision-head-v1",
        encoding_version=LIST_AWARE_VERSION, precision="float32",
        features={"query": feature, "candidate": feature},
        baseline={"labelled": "query-label-logits", "label_free": "zero"},
        capacities={"choice": {"text": {"implemented_max": 255, "validated_max": None}}},
        residual_strength={"text": 1.},
        extraction={"purpose": "feature-extraction", "source_head": str(source),
                    "weights": "unchanged", "source_max_options": metadata["max_options"]})
    if "reference_backbone" in metadata:
        config["reference_backbone"] = metadata["reference_backbone"]
    calibration = {"strata": []}
    read_v2(config, calibration)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".extraction-head-") as temporary:
        staging = Path(temporary)
        for filename, value in (("config.json", config), ("calibration.json", calibration)):
            (staging / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        shutil.copyfile(source / "head.safetensors", staging / "head.safetensors")
        staging.rename(directory)
    return config


async def prepare_row(service, source, producer):
    if source.get("primitive") != "choice" or source.get("modality") != "text":
        raise ValueError("This extractor accepts prepared text choice rows only")
    request = ServiceDecisionRequest(primitive=source["primitive"], messages=source["context"],
        question=source["question"], additional_context=source.get("additional_context", ""),
        candidates=source["candidates"])
    route, modality = service._admit_decision(request)
    state = await service.prepare_state(request)
    prepared = await asyncio.to_thread(service._prepare_decision, request, state, "candidates",
                                       route=route, modality=modality)
    candidates = [candidate.model_dump() for candidate in request.candidates]
    row = deepcopy(source)
    row["request"] = {"primitive": request.primitive, "messages": deepcopy(request.messages),
                      "question": request.question, "additional_context": request.additional_context,
                      "candidates": candidates}
    row["rendering"] = {
        "prefix": await asyncio.to_thread(service.preparer.render, request.messages),
        "query_suffix": service.preparer.query_suffix(request.question, candidates, state.additional_context, route=route),
        "candidate_suffixes": [candidate_suffix(candidate) for candidate in candidates],
        "encoding_version": service.head.encoding_version}
    row["producer"] = deepcopy(producer)
    return row, request, prepared


async def extract_rows(service, rows, writers, producer):
    groups, identifiers = {}, {}
    counts = {role: {"written": 0, "skipped": 0} for role in writers}
    for role, writer in writers.items():
        for ref in writer.completed.rows:
            if ref["role"] != role or groups.get(ref["group"], role) != role:
                raise ValueError("Existing feature directories have group role leakage")
            if identifiers.get(ref["id"], role) != role:
                raise ValueError("Existing record ID occurs in different roles")
            groups[ref["group"]], identifiers[ref["id"]] = role, role
    for source in rows:
        role = source["role"]
        if role not in writers:
            raise ValueError(f"Unknown or unselected prepared role: {role}")
        row, request, prepared = await prepare_row(service, source, producer)
        writer = writers[role]
        exists = writer.contains(row)
        if groups.get(row["group"], role) != role or identifiers.get(row["id"], role) != role:
            raise ValueError("Prepared rows have record or group role leakage")
        groups[row["group"]], identifiers[row["id"]] = role, role
        if exists:
            counts[role]["skipped"] += 1
            continue
        captured = []
        head = service.head.head
        original = head.score

        def capture(primitive, query, candidates, *, baseline, modality="text"):
            if captured or primitive != request.primitive or modality != prepared.modality:
                raise ValueError("Expected exactly one whole-slate score capture")
            width = head.metadata["hidden_size"]
            for value, shape in ((query, (width,)), (candidates, (len(request.candidates), width)),
                                 (baseline, (len(request.candidates),))):
                if value.shape != shape or value.dtype != torch.float32:
                    raise ValueError("Native scoring inputs differ from the declared final-feature contract")
            captured.append(tuple(value.detach().to(device="cpu").clone() for value in (query, candidates, baseline)))
            return original(primitive, query, candidates, baseline=baseline, modality=modality)

        with patch.object(head, "score", capture):
            result = await service.decide(request, prepared=prepared)
        if len(captured) != 1 or [item["id"] for item in result["scores"]] != [c.id for c in request.candidates]:
            raise ValueError("Missing whole-slate features or changed candidate order")
        row["execution"] = deepcopy(result["execution"])
        writer.add(row, *captured[0])
        counts[role]["written"] += 1
        print(json.dumps({"id": row["id"], "role": role, "execution": row["execution"]}, allow_nan=False),
              flush=True)
    for writer in writers.values():
        writer.flush()
    return counts


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("Expected a positive count")
    return value


def main(argv=None):
    import uvicorn
    from fuzev_exllamav3.server import parser, serve

    cli = parser()
    cli.description = __doc__
    cli.add_argument("--input", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--shard-bytes", type=positive_int, required=True)
    args = cli.parse_args(argv)
    source_head = str(Path(args.head).resolve())
    directory = args.output / "extraction-head"
    config = extraction_head(source_head, directory)
    source_is_v2 = json.loads((Path(source_head) / "config.json").read_text()).get("schema") == V2_SCHEMA
    args.head = source_head if source_is_v2 else str(directory)
    writers = {role: RaggedWriter(args.output / role, config, shard_bytes=args.shard_bytes) for role in ROLES}

    async def extract(server):
        service = server.config.app.state.fuzev_routes.get_service()
        gen = service.generator
        producer = {"model": args.model, "source_head": source_head, "extraction_head": args.head,
            "imports": {name: getattr(sys.modules.get(name), "__file__", None)
                        for name in ("exllamav3", "exllamav3_ext", "fuzev", "fuzev_exllamav3")},
            "bounds": {"context_tokens": service.context_tokens, "cache_tokens": gen.cache.max_num_tokens,
                       "max_batch_size": gen.max_batch_size, "max_chunk_size": gen.max_chunk_size,
                       "mtp_tokens": gen.num_draft_tokens, "cache_bits": args.cache_bits,
                       "prefill_optimizations": gen.model.config.infer_params.prefill_optimizations}}
        rows = (row for path in args.input for row in json_rows(path))
        counts = await extract_rows(service, rows, writers, producer)
        print(json.dumps({"output": str(args.output), "roles": counts}, allow_nan=False), flush=True)

    # The public launcher owns the only engine and all shutdown; no listener starts.
    with patch.object(uvicorn.Server, "serve", extract):
        asyncio.run(serve(args))


if __name__ == "__main__":
    main()
