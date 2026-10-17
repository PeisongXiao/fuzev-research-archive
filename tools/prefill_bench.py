"""Measure fresh prefill, shared-prefix append and comparison prefill/decode.

The comparison method and recorded commands are in docs/v1.0-v1.1-comparison.md.
"""

import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time


OFFSET_STRIDE = 7919


def recurrent_state_dtypes(value):
    return {str(key): str(layer.recurrent_state.dtype)
            for key, layer in getattr(value, "recurrent_layers", {}).items()
            if hasattr(layer, "recurrent_state")}


def run(generator, ids, timing=None, **kw):
    from exllamav3 import Job

    job = Job(input_ids=ids.view(1, -1), max_new_tokens=kw.pop("max_new_tokens", 1), **kw)
    t0 = time.perf_counter()
    if timing is not None:
        receive_sample = job.receive_sample

        def first_sample(*sample_args, **sample_kwargs):
            value = receive_sample(*sample_args, **sample_kwargs)
            if job.new_tokens > 0:
                timing["first_sample_epoch_s"] = time.time()
                timing["ttft_s"] = time.perf_counter() - t0
                job.receive_sample = receive_sample
            return value

        job.receive_sample = first_sample
    generator.enqueue(job)
    last = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "error":
                raise r["error"]
            if r.get("eos"):
                last = r
    if timing is not None and "first_sample_epoch_s" in timing:
        timing["last_sample_epoch_s"] = job.time_last_token
        timing["native_first_decode_epoch_s"] = job.time_first_token
        timing["time_after_first_s"] = (job.time_last_token - timing["first_sample_epoch_s"]
                                        if job.new_tokens > 1 else 0.0)
    return time.perf_counter() - t0, last


def comparison_offset(corpus_tokens, length, rep):
    return ((rep + 1) * OFFSET_STRIDE) % (corpus_tokens - length - 1)


def comparison_measurement(ids, wall, last, expected_outputs, mtp, timing):
    if last is None:
        raise RuntimeError("The native Job ended without a terminal result")
    job = last["job"]
    sequence = job.sequences[0].sequence_ids.torch().reshape(-1).tolist()
    prompt = ids.reshape(-1).tolist()
    generated = sequence[len(prompt):]
    native = {key: last.get(key) for key in ("prompt_tokens", "cached_tokens", "new_tokens",
              "time_enqueued", "time_prefill", "time_generate", "eos_reason",
              "accepted_draft_tokens", "rejected_draft_tokens")}
    rounds = [list(value) for value in getattr(job, "draft_stats", [])]
    accepted = getattr(job, "accepted_draft_tokens", 0)
    rejected = getattr(job, "rejected_draft_tokens", 0)
    processed = native["prompt_tokens"] - native["cached_tokens"] - 1
    prefill_s = native["time_prefill"]
    decode_s = timing.get("time_after_first_s")
    row = {"wall_s": wall, "native": native, "timing": timing,
           "prompt_tokens": native["prompt_tokens"], "cached_tokens": native["cached_tokens"],
           "processed_tokens": processed, "time_prefill_s": prefill_s,
           "prefill_tok_per_s": processed / prefill_s if prefill_s and prefill_s > 0 else None,
           "ttft_s": timing.get("ttft_s"),
           "native_prefill_ready_s": native["time_enqueued"] + prefill_s,
           "completion_tokens": len(generated), "reported_completion_tokens": native["new_tokens"],
           "expected_completion_tokens": expected_outputs,
           "tokens_after_first": max(0, len(generated) - 1), "time_after_first_s": decode_s,
           "decode_tok_per_s": (len(generated) - 1) / decode_s
                               if len(generated) > 1 and decode_s and decode_s > 0 else None,
           "accepted_draft_tokens": accepted, "rejected_draft_tokens": rejected,
           "proposed_draft_tokens": sum(value[1] for value in rounds) if mtp else 0,
           "draft_stats": rounds}
    issues = []
    if sequence[:len(prompt)] != prompt:
        issues.append("Native accepted sequence has a different prompt")
    if len(generated) != expected_outputs or native["new_tokens"] != len(generated):
        issues.append("Native accepted output length differs from the requested length or terminal count")
    if native["prompt_tokens"] != len(prompt) or native["cached_tokens"] != 0:
        issues.append("Fresh native request reports a different prompt length or cached tokens")
    if issues:
        row["errors"] = issues
    return row


def comparison_speed():
    from exllamav3.generator.sampler import ComboSampler
    import gc
    import traceback

    results.update(loader={"max_chunk_size": load_chunk, "max_batch_size": 1, "use_per_device_gib": args.use_gib},
                   generator_options={"max_chunk_size": args.chunks[0], "max_batch_size": 1,
                                      "dynamic_draft_tokens": False, "record_draft_stats": True},
                   sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                             "rep_p": 1.0, "pres_p": 0.0, "freq_p": 0.0, "seed": args.seed},
                   cache={"tokens": cache.max_num_tokens, "batch_slots": cache.num_slots,
                          "max_history": cache.max_history, "k_bits": 8, "v_bits": 8,
                          "recurrent_constructor_kwargs": {key: str(value) for key, value in cache_kwargs.items()}},
                   protocol={"warmups_per_cell": 1, "measured_reps": args.reps,
                             "offset_stride_tokens": OFFSET_STRIDE, "fresh_generator_per_request": True,
                             "stop_tokens": list(config.eos_token_id_list),
                             "stop_tokens_suppressed_through_requested_output": True,
                             "draft_component_loaded_during_mtp_off": draft_model is not None,
                             "prefill_timer": "native time_prefill: first prefill through first decode start",
                             "decode_timer": "native last-token timestamp minus first accepted-sample timestamp"})

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    save()
    cells = []
    if "C" in args.comparison_phases:
        cells.extend(("C", length, 0, 1) for length in args.lengths)
    if "D" in args.comparison_phases:
        cells.extend(("D", length, mtp, 1024) for length in args.lengths
                     if length <= args.decode_max_length for mtp in (3, 0))
    for phase, length, mtp, output_tokens in cells:
        for rep in range(args.reps + 1):
            offset = comparison_offset(corpus.numel(), length, rep)
            row = {"phase": phase, "length": length, "mtp_tokens": mtp, "rep": rep,
                   "measured": rep > 0, "offset": offset, "chunk": args.chunks[0]}
            results["runs"].append(row)
            save()
            generator = None
            last = None
            started = time.perf_counter()
            try:
                generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    max_chunk_size=args.chunks[0], max_batch_size=1,
                    draft_model=draft_model if mtp else None, draft_cache=draft_cache if mtp else None,
                    num_draft_tokens=mtp, dynamic_draft_tokens=False, record_draft_stats=True)
                row.update(chunk=generator.max_chunk_size, actual_draft_tokens=generator.num_draft_tokens,
                           mtp_draft=bool(generator.mtp_draft), dynamic_draft=bool(generator.dynamic_draft))
                ids = corpus[offset:offset + length].clone()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timing = {}
                wall, last = run(generator, ids, timing=timing,
                    max_new_tokens=output_tokens, min_new_tokens=output_tokens,
                    stop_conditions=list(config.eos_token_id_list), token_healing=False,
                    sampler=ComboSampler(temperature=1.0, top_p=0.95, top_k=20), seed=args.seed)
                row.update(comparison_measurement(ids, wall, last, output_tokens, mtp, timing),
                           peak_gpu_bytes=torch.cuda.max_memory_allocated())
            except Exception:
                row.update(error=traceback.format_exc(), elapsed_to_error_s=time.perf_counter() - started)
            finally:
                if generator is not None:
                    try:
                        generator.clear_queue()
                        close = getattr(generator, "close", None)
                        if close is not None:
                            close()
                            row["generator_cleanup"] = "native close"
                        else:
                            generator.filter_pool.shutdown(wait=False)
                            row["generator_cleanup"] = "native clear_queue and Python garbage collection"
                    except Exception:
                        row["cleanup_error"] = traceback.format_exc()
                generator = last = None
                gc.collect()
                save()
                print(json.dumps(row), flush=True)
    print("wrote", args.out, flush=True)


def seed_prefix(generator, ids, prefix_tokens=None, *, run_job=run, synchronize=None):
    # N+1 inputs checkpoint all N prefix tokens before the final-token forward.
    setup_tokens = (args.prefix_tokens if prefix_tokens is None else prefix_tokens) + 1
    wall, last = run_job(generator, ids[:setup_tokens].clone())
    if (last is None or last.get("prompt_tokens") != setup_tokens
            or last.get("cached_tokens") != 0 or last.get("new_tokens") != 1):
        raise RuntimeError(f"Prefix setup requires the full uncached prefix plus one input: {last}")
    if synchronize is not None:
        synchronize()
    return {"wall_s": wall, "prompt_tokens": last["prompt_tokens"], "cached_tokens": last["cached_tokens"],
            "processed_tokens": last["prompt_tokens"] - last["cached_tokens"] - 1,
            "forwarded_tokens": last["prompt_tokens"] - last["cached_tokens"], "new_tokens": last["new_tokens"]}


def prefill_measurement(last, wall, peak_gpu_bytes, *, chunk, length, rep, offset):
    processed = last["prompt_tokens"] - last["cached_tokens"] - 1
    return {"chunk": chunk, "length": length, "rep": rep, "measured": rep > 0,
            "offset": offset, "wall_s": wall, "time_prefill_s": last["time_prefill"],
            "prompt_tokens": last["prompt_tokens"], "cached_tokens": last["cached_tokens"],
            "processed_tokens": processed, "peak_gpu_bytes": peak_gpu_bytes,
            "tok_per_s": processed / last["time_prefill"]}


@contextmanager
def variant_generator(chunk, overrides):
    from exllamav3 import Cache, Config, Model
    from exllamav3.cache import CacheLayer_quant

    generator = variant_model = None
    settings.CONFIG.clear()
    settings.CONFIG.update(options)
    settings.CONFIG.update({key: value for key, value in overrides.items() if key != "chunk"})

    def fresh_generator():
        nonlocal generator
        if generator is not None:
            generator.close()
            generator = None
        generator = Generator(model=variant_model, cache=variant_cache, tokenizer=tokenizer,
                              max_chunk_size=overrides.get("chunk", chunk), max_batch_size=1)
        return generator

    try:
        variant_config = Config.from_directory(args.model)
        variant_config.infer_params.ngram_stream_from_disk = True
        variant_model = Model.from_config(variant_config)
        variant_cache = Cache(variant_model, max_num_tokens=args.cache_tokens, max_batch_size=1,
            max_history=0, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, **cache_kwargs)
        loader = {"max_chunk_size": load_chunk, "max_batch_size": 1}
        if args.use_gib is not None:
            loader["use_per_device"] = args.use_gib
        variant_model.load(**loader)
        states = recurrent_state_dtypes(variant_cache)
        results["recurrent_state_dtypes"] = {"target": states, "draft": {}}
        results["environment"]["recurrent_state"] = sorted(set(states.values()))
        yield fresh_generator
    finally:
        try:
            if generator is not None:
                generator.close()
        finally:
            try:
                if variant_model is not None:
                    variant_model.unload()
            finally:
                settings.CONFIG.clear()
                settings.CONFIG.update(options)


def ordinary_timing():
    counter = 0
    results["means"] = []
    results["protocol"] = {"offset_stride_tokens": OFFSET_STRIDE,
        "warmups_per_cell": 1, "measured_reps": args.reps,
        "variant_model_lifecycle": "One model load per chunk, length and variant; warmup and measured repetitions share that model." if args.variants else "one model load"}
    timing_chunks = args.chunks
    if args.variants and all("chunk" in overrides for overrides in args.variants.values()):
        timing_chunks = args.chunks[:1]
    for chunk in timing_chunks:
        if not args.variants:
            generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                                  max_chunk_size=chunk, max_batch_size=1)
            try:
                run(generator, corpus[:512])
            finally:
                generator.close()
        for length in args.lengths:
            offsets = []
            for _ in range(args.reps + 1):
                counter += 1
                offsets.append((counter * OFFSET_STRIDE) % (corpus.numel() - length - 1))
            for variant, overrides in (args.variants.items() if args.variants else ((None, {}),)):
                cell_start = len(results["runs"])
                with variant_generator(chunk, overrides) if args.variants else nullcontext() as fresh_generator:
                    generator = None
                    try:
                        for rep, start in enumerate(offsets):
                            if args.variants:
                                generator = fresh_generator()
                            else:
                                if generator is not None:
                                    generator.close()
                                generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                                                      max_chunk_size=chunk, max_batch_size=1)
                            real_chunk = generator.max_chunk_size
                            priming = None
                            if args.warm_ple and rep > 0:
                                prime_wall, prime_last = run(generator, corpus[start:start + length].clone())
                                if (prime_last is None or prime_last.get("prompt_tokens") != length
                                        or prime_last.get("cached_tokens") != 0):
                                    raise RuntimeError(f"PLE priming requires the identical full prompt: {prime_last}")
                                priming = {"wall_s": prime_wall, "input_tokens": length,
                                    "prompt_tokens": prime_last["prompt_tokens"], "cached_tokens": prime_last["cached_tokens"],
                                    "processed_tokens": prime_last["prompt_tokens"] - prime_last["cached_tokens"] - 1}
                                if args.variants:
                                    generator = fresh_generator()
                                else:
                                    generator.close()
                                    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                                                          max_chunk_size=chunk, max_batch_size=1)
                                real_chunk = generator.max_chunk_size
                            if append_mode:
                                prefix_setup = seed_prefix(generator, corpus[start:start + length],
                                                           synchronize=torch.cuda.synchronize)
                            torch.cuda.reset_peak_memory_stats()
                            if priming is not None:
                                affinity = sorted(os.sched_getaffinity(0))
                                inblock0 = resource.getrusage(resource.RUSAGE_SELF).ru_inblock
                            wall, last = run(generator, corpus[start:start + length].clone())
                            if priming is not None:
                                inblock1 = resource.getrusage(resource.RUSAGE_SELF).ru_inblock
                            expected_cached = args.prefix_tokens if append_mode else 0
                            if (not last or not last.get("time_prefill")
                                    or last.get("cached_tokens") != expected_cached
                                    or last.get("prompt_tokens") != length):
                                raise RuntimeError(f"Prefill requires exactly {expected_cached} cached tokens, "
                                                   f"{length} prompt tokens and engine timing: {last}")
                            row = prefill_measurement(last, wall, torch.cuda.max_memory_allocated(),
                                chunk=real_chunk, length=length, rep=rep, offset=start)
                            if append_mode:
                                row.update(prefix_tokens=args.prefix_tokens, append_tokens=args.append_tokens,
                                    forwarded_tokens=last["prompt_tokens"] - last["cached_tokens"],
                                    new_tokens=last["new_tokens"], prefix_setup=prefix_setup)
                            if priming is not None:
                                row.update(ple_priming=priming, cpu_affinity=affinity,
                                    timed_ru_inblock_before=inblock0, timed_ru_inblock_after=inblock1,
                                    timed_ru_inblock_delta=inblock1 - inblock0)
                            if args.variants:
                                row.update(variant=variant, operator_chunk=chunk,
                                           effective_prefill_options=dict(settings.CONFIG))
                            results["runs"].append(row)
                            print(json.dumps(row), flush=True)
                    finally:
                        if not args.variants and generator is not None:
                            generator.close()
                measured = [row for row in results["runs"][cell_start:] if row["measured"]]
                mean = {"chunk": measured[0]["chunk"], "length": length, "reps": len(measured),
                    "tok_per_s": statistics.mean(row["tok_per_s"] for row in measured),
                    "time_prefill_s": statistics.mean(row["time_prefill_s"] for row in measured),
                    "wall_s": statistics.mean(row["wall_s"] for row in measured)}
                if append_mode:
                    mean.update(prefix_tokens=args.prefix_tokens, append_tokens=args.append_tokens)
                if args.variants:
                    mean.update(variant=variant, operator_chunk=chunk,
                                effective_prefill_options=measured[0]["effective_prefill_options"])
                results["means"].append(mean)
                print(json.dumps({"mean": mean}), flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True)
    cli.add_argument("--corpus", required=True)
    cli.add_argument("--out", required=True)
    cli.add_argument("--mode", choices=("plain", "fork-opt"), default="plain")
    cli.add_argument("--chunks", type=int, nargs="+", required=True)
    cli.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 32768, 65536])
    cli.add_argument("--reps", type=int, default=2)
    cli.add_argument("--cache-tokens", type=int)
    cli.add_argument("--context-tokens", type=int)
    cli.add_argument("--use-gib", type=float)
    cli.add_argument("--state", choices=("fp32", "default"), default="fp32")
    cli.add_argument("--prefix-tokens", type=int)
    cli.add_argument("--append-tokens", type=int)
    cli.add_argument("--warm-ple", action="store_true")
    cli.add_argument("--variants", type=json.loads)
    cli.add_argument("--comparison-speed", action="store_true")
    cli.add_argument("--comparison-phases", nargs="+", choices=("C", "D"), default=["C", "D"])
    cli.add_argument("--decode-max-length", type=int, default=32768)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--label")
    return cli


def main(argv=None):
    global args, results, torch, Generator, model, cache, tokenizer, config
    global draft_model, draft_cache, cache_kwargs, corpus, load_chunk, settings, options, append_mode
    cli = parser()
    args = cli.parse_args(argv)
    if args.mode == "plain":
        os.environ["EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS"] = "1"
    import torch
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.version import __version__

    append_mode = args.prefix_tokens is not None
    if append_mode:
        args.lengths = [args.prefix_tokens + args.append_tokens]
    config = Config.from_directory(args.model)
    config.infer_params.ngram_stream_from_disk = True
    tokenizer = Tokenizer.from_config(config)
    corpus = tokenizer.encode(Path(args.corpus).read_text(), encode_special_tokens=False).reshape(-1)
    model = Model.from_config(config)
    draft_model = draft_cache = None
    if args.comparison_speed and "D" in args.comparison_phases:
        draft_model = Model.from_config(config, component="mtp")
    history = max(3, draft_model.caps.get("default_draft_size", 3)) if draft_model else 0
    outputs = 1024 if draft_model is not None else 1
    needed = max(args.lengths) + outputs + history
    rounded = (needed + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    args.context_tokens = args.context_tokens or rounded
    args.cache_tokens = args.cache_tokens or rounded
    if not (needed <= args.context_tokens <= config.max_position_embeddings and needed <= args.cache_tokens):
        cli.error("The prompt, outputs and draft margin must fit the cache and model context")
    options, settings = {}, None
    if args.mode == "fork-opt":
        try:
            from exllamav3.optimized import settings
        except ImportError:
            # exllamav3-fuzev 1.0 names this module exllamav3.optimized.hooks.
            from exllamav3.optimized import hooks as settings
        options = dict(settings.CONFIG)
    load_chunk = max(args.chunks + [value.get("chunk", max(args.chunks))
                                   for value in (args.variants or {}).values()])
    cache_kwargs = {"recurrent_dtype": torch.float} if args.state == "fp32" else {}
    if "recurrent_dtype" not in inspect.signature(Cache).parameters:
        # exllamav3-fuzev Cache takes no recurrent_dtype and keeps FP32 recurrent
        # state; the record stores the observed per-layer dtypes.
        cache_kwargs = {}
    cache = Cache(model, max_num_tokens=args.cache_tokens, max_batch_size=1,
        max_history=history, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, **cache_kwargs)
    loader = {"max_chunk_size": load_chunk, "max_batch_size": 1}
    if args.use_gib is not None:
        loader["use_per_device"] = args.use_gib
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens=args.cache_tokens, max_batch_size=1,
            max_history=history, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, **cache_kwargs)
        draft_model.load(**loader)
    if not args.variants or args.comparison_speed:
        model.load(**loader)
    versions = {}
    for name in ("torch", "triton", "flashinfer-python", "flashinfer-jit-cache"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    actual_state = {"target": recurrent_state_dtypes(cache), "draft": recurrent_state_dtypes(draft_cache)}
    environment = {"device": torch.cuda.get_device_name(),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": platform.python_version(), "engine": __version__, "kv_cache_bits": [8, 8],
        "prefill_chunk": args.chunks, "max_batch_size": 1,
        "recurrent_state": sorted(set(actual_state["target"].values())),
        "mtp_tokens": [0, 3] if draft_model is not None else [0]}
    for source, target in (("torch", "torch"), ("triton", "triton"),
                           ("flashinfer-python", "flashinfer_python"),
                           ("flashinfer-jit-cache", "flashinfer_jit_cache")):
        if source in versions:
            environment[target] = versions[source]
    recorded_settings = {**vars(args), "model": Path(args.model).name,
        "corpus": Path(args.corpus).name, "out": Path(args.out).name}
    results = {"mode": args.mode, "label": args.label, "settings": recorded_settings,
        "environment": environment, "versions": versions, "effective_prefill_options": options,
        "recurrent_state_dtypes": actual_state, "ngram_stream_from_disk": True,
        "batch_size": 1, "runs": []}
    try:
        if args.comparison_speed:
            comparison_speed()
        else:
            ordinary_timing()
    finally:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        if draft_model is not None:
            draft_model.unload()
        if not args.variants or args.comparison_speed:
            model.unload()
    return results


if __name__ == "__main__":
    main()
