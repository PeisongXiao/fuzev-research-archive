"""Opt-in stock generation control for an already loaded, exclusive offline owner.

No loader, engine import, head, forward interception or qualification claim.
"""

import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from weakref import WeakSet

from .encoding import token_hash
from .paths import source_revision
from .telemetry import MemoryGuard, admit


ENGINE_COMMIT = "ba60611d031f673a12ebbf2f133c583b4e12df7f"  # unpublished lab commit ID
WINDOW = 262144
PAGE = 256
_owners = WeakSet()
_owner_lock = RLock()


@dataclass(frozen=True)
class GenerationControlRequest:
    name: str
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    add_bos: bool
    encode_special_tokens: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    native_service_completion_token_ids: tuple[int, ...] | None = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stock_api(expected_engine_commit=ENGINE_COMMIT):
    names = ("exllamav3.generator.generator", "exllamav3.generator.job",
             "exllamav3.generator.sampler.presets", "exllamav3.modules.ngram_embedding",
             "exllamav3.architecture.qwen4_exp")
    modules = [sys.modules.get(name) for name in names]
    _require(all(modules) and "torch" in sys.modules, "Stock engine must already be imported and loaded")
    root = Path(modules[0].__file__).resolve().parents[2]
    for module, name in zip(modules, names):
        _require(Path(module.__file__).resolve() == root / (name.replace(".", "/") + ".py"),
                 "Stock engine bindings have mixed source roots")
    return (sys.modules["torch"], modules[0].Generator, modules[1].Job,
            modules[2].ComboSampler, modules[3].NGramEmbedding, modules[4].Qwen4ExpModel,
            "native_stock_observation")


def _ownership(generator):
    cache, table = generator.cache, generator.pagetable
    _require(type(cache.owner_serial) is int and cache.owner_serial > 0
             and generator.cache_owner_serial == cache.owner_serial,
             "Stock Generator no longer owns its cache; no takeover or cleanup is safe")
    _require(table.generator is generator and table.cache is cache, "Misbound stock page table")


def _idle(generator):
    _ownership(generator)
    cache, table = generator.cache, generator.pagetable
    _require(cache.initialized is True, "Cache must already be allocated")
    _require(not generator.pending_jobs and not generator.active_jobs and not generator.filter_queue,
             "Refusing occupied stock queues")
    _require(type(cache.num_slots) is int and 1 <= cache.num_slots <= 8
             and type(generator.max_batch_size) is int
             and 1 <= generator.max_batch_size <= cache.num_slots,
             "Unsafe stock batch/slot capacity")
    _require(all(type(slot) is int for slot in cache.free_list)
             and sorted(cache.free_list) == list(range(cache.num_slots)), "Refusing occupied recurrent slots")
    _require(type(table.max_pages) is int and table.max_pages > 0
             and table.max_pages == cache.max_num_tokens // PAGE
             and generator.max_total_tokens == table.max_pages * PAGE, "Contradictory cache capacity")
    _require(not table.referenced_pages and len(table.unreferenced_pages) == table.max_pages
             and len(table.all_pages) == table.max_pages
             and {id(p) for p in table.all_pages} == {id(p) for p in table.unreferenced_pages.values()}
             and all(type(p.ref_count) is int and p.ref_count == 0 for p in table.all_pages),
             "Refusing referenced or contradictory cache pages")


def _ple_modes(generator, ngram_type):
    tables = [module for module in generator.model if type(module) is ngram_type]
    _require(tables, "Missing stock Qwen4Exp PLE streaming table")
    for table in tables:
        _require(table.mode in ("trellis_disk", "fp16_disk") and table.tables is None and table.handles,
                 "Stock PLE must already be streaming from disk")
        _require(not table._pending and not any(pin.held for pin in table._pins),
                 "Stock PLE staging must be quiescent")
    return [table.mode for table in tables]


def _ids(tensor, torch):
    _require(type(tensor) is torch.Tensor and tensor.device.type == "cpu"
             and tensor.dtype == torch.int64 and tensor.ndim == 2 and tensor.shape[0] == 1,
             "Expected stock CPU int64 [1, n] token history")
    return tensor[0].tolist()


def _phase_timing(result):
    raw = {key: result[key] for key in ("time_enqueued", "time_prefill", "time_generate")}
    valid = all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in raw.values())
    return {"clock": "stock time.time", "status": "available" if valid else "invalid_wall_clock",
            "raw_seconds": raw, "ttft_seconds": None,
            "scope": "prefill ends before first generation forward; generate includes first sample"}


def run_native_generation_control(generator, requests, *, allow_generation=False, exclusive=False,
                                  additional_peak_bytes, estimate_basis, memory_fraction=None,
                                  timeout_seconds=300., cancel=None, expected_engine_commit=None):
    """Run one idle 1/2/4-job stock batch; caller loads and owns the offline engine.

    Cancellation/deadlines are checked between stock calls, not inside GPU work.
    Only this invocation's jobs are cancelled/released; queues are never cleared.
    """
    _require(allow_generation is True and exclusive is True,
             "Stock control requires explicit offline generation and exclusive ownership")
    _require(type(additional_peak_bytes) is int and additional_peak_bytes > 0
             and isinstance(estimate_basis, str) and estimate_basis.strip(), "Missing additional peak-memory estimate")
    _require(memory_fraction is None or (type(memory_fraction) in (int, float) and math.isfinite(memory_fraction)
                                        and 0 < memory_fraction <= 1), "Memory fraction must be in (0, 1]")
    _require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
             and timeout_seconds > 0, "Invalid stock-control deadline")
    _require(type(requests) in (list, tuple) and bool(requests)
             and all(type(r) is GenerationControlRequest for r in requests),
             "Stock control requires explicit requests")
    requests = tuple(requests)
    _require(all(isinstance(r.name, str) and 0 < len(r.name) <= 128 for r in requests)
             and len({r.name for r in requests}) == len(requests), "Request names must be distinct")
    module = sys.modules.get("exllamav3.generator.generator")
    directory = Path(module.__file__).resolve().parents[2] if module is not None else None
    actual_commit = source_revision(directory, "exllamav3")
    api = _stock_api() if expected_engine_commit is None else _stock_api(expected_engine_commit)
    torch, generator_type, job_type, sampler_type, ngram_type, model_type, evidence = api
    _require(type(generator) is generator_type, "Expected the original stock Generator, not NativeService or a wrapper")
    _require(type(generator.model) is model_type, "Expected the original stock Qwen4Exp model")
    _require(not any(name in generator.__dict__ for name in ("enqueue", "iterate", "cancel"))
             and "forward" not in generator.model.__dict__, "Refusing instance-wrapped stock paths")
    with _owner_lock:
        _require(generator not in _owners, "Stock control is already running for this owner")
        _owners.add(generator)
    jobs, rows, guard = [], [], None
    try:
        _idle(generator)
        _require(generator.draft_model is None and generator.draft_cache is None
                 and generator.num_draft_tokens == 0 and generator.ngram_match_min == 0
                 and generator.dynamic_draft is False, "Speculative stock decoding is unsupported")
        model, config = generator.model, generator.model.config
        _require(config.architecture == "Qwen4ExpForConditionalGeneration" and model.loaded_tp is False
                 and model.output_device is not None and model.caps.get("recurrent_states"),
                 "Expected already-loaded non-TP stock Qwen4Exp with recurrent cache")
        _require(generator.recurrent_cache is not None, "Missing stock recurrent checkpoint cache")
        if evidence != "cpu_fake":
            device = torch.device(model.output_device)
            _require(device.type == "cuda" and device.index is not None
                     and model.active_devices in ([], [device.index]), "Expected a single explicit stock CUDA device")
        modes = _ple_modes(generator, ngram_type)
        _require(len(requests) <= generator.max_batch_size, "Requested batch exceeds already-loaded stock capacity")
        vocab, eos = config.vocab_size, tuple(config.eos_token_id_list)
        _require(type(vocab) is int and vocab > 0 and eos
                 and all(type(token) is int and 0 <= token < vocab for token in eos), "Invalid stock vocabulary/EOS metadata")
        contexts, pages = 0, 0
        for request in requests:
            _require(type(request.prompt_token_ids) is tuple and 0 < len(request.prompt_token_ids) <= WINDOW
                     and all(type(token) is int and 0 <= token < vocab for token in request.prompt_token_ids),
                     "Explicit text prompt IDs must be nonempty and in vocabulary")
            _require(isinstance(request.rendered_prompt, str) and 0 < len(request.rendered_prompt) <= WINDOW * 64
                     and type(request.add_bos) is bool and type(request.encode_special_tokens) is bool,
                     "Explicit rendered prompt and tokenizer flags are required")
            _require(type(request.max_new_tokens) is int and 0 < request.max_new_tokens <= WINDOW
                     and type(request.seed) is int and 0 <= request.seed <= 2**63 - 1,
                     "Invalid completion budget or seed")
            _require(type(request.temperature) in (int, float) and math.isfinite(request.temperature)
                     and 0 <= request.temperature <= 2
                     and type(request.top_p) in (int, float) and math.isfinite(request.top_p)
                     and 0 < request.top_p <= 1 and type(request.top_k) is int and request.top_k >= 0,
                     "Invalid explicit sampling settings")
            reference = request.native_service_completion_token_ids
            _require(reference is None or (type(reference) is tuple and len(reference) <= request.max_new_tokens
                     and all(type(token) is int and 0 <= token < vocab for token in reference)),
                     "Invalid optional NativeService accepted-ID reference")
            total = len(request.prompt_token_ids) + request.max_new_tokens
            _require(total <= WINDOW, "Per-request prompt plus completion budget exceeds 262144 tokens")
            contexts += total
            # Stock max_rq_tokens=None adds one token before Sequence.prepare rounds pages.
            pages += (total + 1 + PAGE - 1) // PAGE
        _require(pages <= generator.pagetable.max_pages, "Conservative simultaneous stock page budget does not fit")
        receipt_reserve = contexts * 192 + sum(4 * len(r.rendered_prompt) for r in requests) + (1 << 20)
        admit(additional_peak_bytes + receipt_reserve, memory_fraction)
        guard = MemoryGuard(fraction=memory_fraction)
        guard.start()
        for request in requests:
            encoded = generator.tokenizer.encode(request.rendered_prompt, add_bos=request.add_bos,
                                                 encode_special_tokens=request.encode_special_tokens)
            _require(_ids(encoded, torch) == list(request.prompt_token_ids), "Rendered prompt/token IDs disagree")
        cache_before = generator.get_cache_stats()
        started = time.monotonic()

        def check():
            _ownership(generator)
            _require(all(any(job is owned for owned in jobs)
                         for job in generator.pending_jobs + generator.active_jobs), "Foreign stock job appeared")
            guard.check()
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Stock control cancelled between iterations")
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("Stock control deadline expired between iterations")

        for request in requests:
            check()
            sampler = sampler_type(temperature=request.temperature, top_p=request.top_p, top_k=request.top_k,
                                   min_p=0., rep_p=1., freq_p=0., pres_p=0., dry_multiplier=0.,
                                   temp_last=False, adaptive_target=1., logit_bias=None)
            job = job_type(input_ids=torch.tensor([request.prompt_token_ids], dtype=torch.long, device="cpu"),
                           sampler=sampler, max_new_tokens=request.max_new_tokens, min_new_tokens=0,
                           seed=request.seed, stop_conditions=list(eos), token_healing=False,
                           decode_special_tokens=False, filters=None, embeddings=None, max_rq_tokens=None,
                           stop_on_loop=None, identifier=request.name,
                           return_probs=False, return_logits=False, return_top_tokens=0)
            jobs.append(job)
            _require(len(job.sequences) == 1 and _ids(job.sequences[0].input_ids.torch(), torch)
                     == list(request.prompt_token_ids), "Stock Job changed the procedural prompt")
            serial = generator.enqueue(job)
            _require(type(serial) is int and serial == job.serial_number
                     and serial not in [row["serial"] for row in rows], "Invalid stock job serial")
            rows.append({"name": request.name, "serial": serial, "accepted": [], "streamed": [],
                         "first_observed": None, "last_observed": None, "final": None,
                         "sampler_fused_only": sampler.fused_only,
                         "sampler_steps": [type(step).__name__ for step in sampler.steps]})
        iterations = 0
        while generator.num_remaining_jobs():
            check()
            events = generator.iterate()
            iterations += 1
            observed = time.monotonic()
            check()
            for job, request, row in zip(jobs, requests, rows):
                sequence = job.sequences[0].sequence_ids
                offset = len(request.prompt_token_ids) + len(row["accepted"])
                _require(len(sequence) >= offset, "Stock accepted history rewound unexpectedly")
                new = _ids(sequence.torch_slice(offset, None), torch)
                _require(all(0 <= token < vocab for token in new), "Stock accepted invalid token IDs")
                if new:
                    row["accepted"].extend(new)
                    row["first_observed"] = row["first_observed"] if row["first_observed"] is not None else observed
                    row["last_observed"] = observed
                _require(len(row["accepted"]) <= request.max_new_tokens, "Stock exceeded completion budget")
            for event in events:
                _require(isinstance(event, dict) and any(event.get("job") is job for job in jobs),
                         "Unknown stock event owner")
                index = next(i for i, job in enumerate(jobs) if event["job"] is job)
                row = rows[index]
                _require(event.get("serial") == row["serial"] and not event.get("requeue"),
                         "Unexpected stock serial or requeue")
                if event.get("stage") == "error":
                    raise RuntimeError(f"Stock job {row['name']} failed") from event.get("error")
                _require(event.get("stage") in ("started", "prefill", "streaming"), "Unknown stock stage")
                if "token_ids" in event:
                    row["streamed"].extend(_ids(event["token_ids"], torch))
                if event.get("eos"):
                    _require(row["final"] is None and event["stage"] == "streaming", "Duplicate or premature stock EOS")
                    row["final"] = event
        ended = time.monotonic()
        check()
        _idle(generator)
        _ple_modes(generator, ngram_type)
        output = []
        for job, request, row in zip(jobs, requests, rows):
            final, accepted = row.pop("final"), row.pop("accepted")
            _require(final is not None and job.is_finished and accepted
                     and final["prompt_tokens"] == len(request.prompt_token_ids)
                     and final["new_tokens"] == len(accepted) == job.new_tokens,
                     "Incomplete or contradictory stock completion counts")
            _require(_ids(job.sequences[0].sequence_ids.torch_slice(0, len(request.prompt_token_ids)), torch)
                     == list(request.prompt_token_ids), "Stock prompt history mutated")
            reason = final["eos_reason"]
            _require((reason == "stop_token" and final["eos_triggering_token_id"] == accepted[-1] in eos)
                     or (reason == "max_new_tokens" and len(accepted) == request.max_new_tokens
                         and accepted[-1] not in eos), "Unexpected stock termination")
            _require(row["streamed"] == accepted[:len(row["streamed"])], "Streamed IDs disagree with accepted history")
            _require(type(final["cached_pages"]) is int and type(final["cached_tokens"]) is int
                     and 0 <= final["cached_pages"] * PAGE <= final["cached_tokens"] <= len(request.prompt_token_ids),
                     "Invalid stock prompt-cache counts")
            reference = request.native_service_completion_token_ids
            comparison = {"reference": "operator-supplied NativeService accepted IDs, not independently attested",
                          "status": "not_supplied" if reference is None else ("equal" if list(reference) == accepted else "different"),
                          "mode": "greedy" if request.temperature == 0 else "stochastic",
                          "rng_equivalence_claimed": False}
            output.append({**row, "prompt_token_ids": list(request.prompt_token_ids),
                           "prompt_token_sha256": token_hash(request.prompt_token_ids),
                           "accepted_completion_token_ids": accepted, "completion_token_sha256": token_hash(accepted),
                           "prompt_tokens": len(request.prompt_token_ids), "completion_tokens": len(accepted),
                           "rendering": {"rendered_prompt": request.rendered_prompt, "add_bos": request.add_bos,
                                         "encode_special_tokens": request.encode_special_tokens, "token_healing": False},
                           "sampling": {"class": "ComboSampler", "temperature": request.temperature,
                                        "top_p": request.top_p, "top_k": request.top_k, "seed": request.seed,
                                        "min_p": 0., "penalties": "disabled", "speculative": False},
                           "stop_token_ids": list(eos), "eos_reason": reason, "text": final["full_completion"],
                           "stock_cached_pages": final["cached_pages"], "stock_cached_tokens": final["cached_tokens"],
                           "physical_forwarded_tokens": None, "stock_phase_timing": _phase_timing(final),
                           "native_service_comparison": comparison})
            output[-1]["first_accepted_observed_seconds"] = output[-1].pop("first_observed") - started
            output[-1]["last_accepted_observed_seconds"] = output[-1].pop("last_observed") - started
        _require(math.isfinite(ended - started) and ended > started, "Invalid outer monotonic timing")
        return {"qualified": False, "evidence_kind": evidence, "engine_commit": actual_commit,
                "recorded_engine_commit": expected_engine_commit or ENGINE_COMMIT,
                "jobs": output, "batch_size": len(jobs), "iterations": iterations,
                "outer_monotonic_seconds": ended - started,
                "outer_timing_scope": "stock Job/Sampler creation, enqueue and iterate through queue drain; not GPU-kernel-only",
                "aggregate_context_budget": contexts, "conservative_pages": pages,
                "additional_peak_bytes": additional_peak_bytes, "receipt_reserve_bytes": receipt_reserve,
                "estimate_basis": estimate_basis, "memory_fraction": memory_fraction,
                "peak_memory_qualified": False, "ple_modes": modes,
                "cache_before": cache_before, "cache_after": generator.get_cache_stats()}
    finally:
        failure, cleanup_errors = sys.exc_info()[1], []
        try:
            # Never release slots/pages belonging to a replacement generator.
            for job in jobs:
                if job.generator is not generator:
                    continue
                try:
                    _ownership(generator)
                except Exception as error:
                    cleanup_errors.append(error)
                    break
                for release in (lambda: generator.cancel(job), job.deallocate_pages):
                    try:
                        release()
                    except Exception as error:
                        cleanup_errors.append(error)
                if job in generator.pending_jobs or job in generator.active_jobs:
                    try:
                        generator.cancel(job)
                    except Exception as error:
                        cleanup_errors.append(error)
        finally:
            try:
                if guard is not None:
                    try:
                        guard.close()
                    except Exception as error:
                        cleanup_errors.append(error)
            finally:
                with _owner_lock:
                    _owners.discard(generator)
        if cleanup_errors:
            message = "Stock cleanup could not be verified: " + "; ".join(str(error) for error in cleanup_errors)
            if failure is not None:
                failure.add_note(message)
            else:
                raise RuntimeError(message) from cleanup_errors[0]
