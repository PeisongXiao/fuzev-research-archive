"""Future, exclusive native stock/fused performance; no qualification claim.

Only the explicitly authorized entrypoint imports Torch, the engine or the head.
Written against unpublished lab engine commits requiring exllamav3/fuzev.py.
Four fused pipelines queue through one GPU owner, not four generation forwards.
"""

import argparse
import asyncio
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, get_ident
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .acquire import write_json
from .encoding import (
    TEMPLATE,
    candidate_suffix,
    identity as encoding_identity,
    query_suffix,
    token_hash,
)
from .native_generation_control import (
    GenerationControlRequest,
    run_native_generation_control,
)
from .paths import ROOT, source_revision
from .runtime_harness import Workload, generation_phases, require, summarize_samples
from .runtime_measurement import EvidencePin
from .runtime_performance_counts import counter_delta
from .service_contracts import GenerationRequest, ServiceDecisionRequest


NATIVE_COMMIT = "5e1ec65733f76c3b5e2f796c50daabb4d345aba2"  # unpublished lab commit ID
FOUNDATION_COMMIT = "ba60611d031f673a12ebbf2f133c583b4e12df7f"  # unpublished lab commit ID
SOURCE8_COMMIT = "9728b98b205d4f9af8ecac4e28ee678ba5ab758b"  # unpublished lab commit ID
HEAD_SHA = "f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c"
EXTENSION_SHA = "e1398c20e1b1c4a0c91c8661790cd2543156c3556af2d5963c75dfbe4fc6c05b"
# Manifest sha256 of the unpublished lab `fuzev-optimized` bundle used by the recorded runs.
FOUNDATION_SHA = "59d2d45db19d642e457717bbf96d9523ca00fc1005315ff37b90dddc28448144"
MODEL_REVISION = "55a732e0c4c3d4614bc42b68493bb930d9b02c0a"
MODEL_REPO = "turboderp/Qwen3.8-Flash-Next-exl3"
WINDOW = 262144
MODES = ("stock-generation", "generation-only", "decision-only", "mixed")
FILLER = "ALPHA BETA GAMMA.\n"


class OperatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: Literal["native-4.05"] = "native-4.05"
    expected_engine_commit: str | None = None
    expected_recipes_commit: str | None = None
    native_tree: Path
    native_extension: Path
    workloads: EvidencePin
    plan: EvidencePin
    quality: EvidencePin
    correctness_evidence: EvidencePin
    memory_fraction: float | None = Field(default=None, gt=0, le=1)
    additional_peak_bytes: int = Field(strict=True, gt=0)
    estimate_basis: str = Field(min_length=1)
    timeout_seconds: float = Field(default=3600, ge=1, le=3600)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean_source(tree, revision):
    tree = Path(tree).resolve()
    package = "fuzev-research" if tree == Path(__file__).resolve().parents[1] else "exllamav3"
    return {"path": str(tree), "commit": source_revision(tree, package), "recorded_commit": revision}


def load_workloads(config):
    raw = config.workloads.read()
    require(
        isinstance(raw, list) and len(raw) == 3,
        "Pin the original three procedural workloads",
    )
    workloads = [Workload.model_validate(item) for item in raw]
    require(
        len({item.name for item in workloads}) == 3, "Duplicate procedural workload"
    )
    for workload in workloads:
        request = workload.generation
        require(
            request.model == config.model
            and request.messages is not None
            and request.image is None
            and request.temperature == 1.0
            and request.top_p == 0.95
            and request.top_k == 20
            and request.max_tokens == 128
            and request.seed == 17
            and not request.stop,
            "Keep the original standard sampling, seed, budget and text-only workload",
        )
        require(
            all(spec.image is None for spec in workload.decisions),
            "Text-only performance operator",
        )
    return workloads


@dataclass(frozen=True)
class PreparedWorkload:
    name: str
    nominal_context_tokens: int
    generation: GenerationRequest
    standalone_messages: list
    decisions: list
    stock: GenerationControlRequest
    receipt: dict


def prepare_workload(workload, tier, backend):
    """Smallest repeated literal context meeting the nominal loaded-token tier."""
    require(
        tier in (2048, 32768), "Only declared 2K/32K tiers; full-window is separate"
    )
    original = [message.model_dump() for message in workload.generation.messages]

    def padded(messages, repetitions):
        messages = deepcopy(messages)
        user = next(message for message in messages if message["role"] == "user")
        user["content"] = FILLER * repetitions + user["content"]
        return messages

    def render(repetitions):
        messages = padded(original, repetitions)
        text = backend.tokenizer.hf_render_chat_template(
            messages,
            add_generation_prompt=True,
            chat_template=TEMPLATE.read_text(),
            enable_thinking=False,
            reasoning_effort="none",
            preserve_thinking=True,
        )
        return text, tuple(backend.encode(text)), messages

    low, high = 0, 1
    while len(render(high)[1]) < tier:
        high *= 2
        require(
            high <= WINDOW, "Literal padding cannot reach the token tier within bounds"
        )
    while low < high:
        middle = (low + high) // 2
        if len(render(middle)[1]) < tier:
            low = middle + 1
        else:
            high = middle
    rendered, tokens, messages = render(low)
    require(
        tier <= len(tokens) <= tier + 256,
        "Tokenizer did not produce a bounded nominal tier",
    )
    generation = GenerationRequest.model_validate(
        {**workload.generation.model_dump(), "messages": messages}
    )
    standalone = padded(
        [message.model_dump() for message in workload.standalone_messages], low
    )
    base, base_hash = backend.preparer.base(standalone)
    specs = []
    for spec in workload.decisions:
        candidates = [candidate.model_dump() for candidate in spec.candidates]
        suffix = backend.encode(
            query_suffix(spec.question, candidates, spec.additional_context)
        )
        tails = [
            backend.encode(candidate_suffix(candidate)) for candidate in candidates
        ]
        total = (
            max(len(base), len(tokens) + generation.max_tokens)
            + len(suffix)
            + max(map(len, tails))
        )
        require(total <= WINDOW, "ABC plus candidate tail cannot fit; no truncation")
        specs.append(
            {
                "primitive": spec.primitive,
                "candidate_ids": [item["id"] for item in candidates],
                "suffix_token_ids": list(suffix),
                "candidate_tail_token_ids": [list(tail) for tail in tails],
                "suffix_token_sha256": token_hash(suffix),
                "tail_token_sha256": [token_hash(tail) for tail in tails],
                "maximum_abc_tail_tokens": total,
            }
        )
    stock = GenerationControlRequest(
        name=f"{workload.name}-{tier}",
        rendered_prompt=rendered,
        prompt_token_ids=tokens,
        add_bos=False,
        encode_special_tokens=True,
        max_new_tokens=generation.max_tokens,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        seed=generation.seed,
    )
    receipt = {
        "name": stock.name,
        "nominal_context_tokens": tier,
        "padding_literal": FILLER,
        "padding_repetitions": low,
        "generation_prompt_tokens": len(tokens),
        "generation_prompt_sha256": token_hash(tokens),
        "rendered_prompt": rendered,
        "generation_prompt_token_ids": list(tokens),
        "standalone_base_token_ids": list(base),
        "standalone_base_sha256": base_hash,
        "decision_layouts": specs,
        "tokenizer_flags": {"add_bos": False, "encode_special_tokens": True},
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 17,
            "maximum_completion_tokens": 128,
            "early_eos_actual_counts_required": True,
        },
    }
    return PreparedWorkload(
        stock.name, tier, generation, standalone, workload.decisions, stock, receipt
    )


def observed_classes(service_type, head_type):
    """Observe original CPU-return boundaries, never wrap model.forward."""

    def completed(counters, call):
        mark = counters.mark()
        try:
            output = call()
        except BaseException:
            counters.completion_failed()
            raise
        counters.acknowledge(mark)
        return output

    class ObservedHead(head_type):
        def score(self, *args, **kwargs):
            return completed(
                self.backend.model.performance_counters,
                lambda: super(ObservedHead, self).score(*args, **kwargs),
            )

    class ObservedService(service_type):
        def _sample(self, *args, **kwargs):
            return completed(
                self.backend.model.performance_counters,
                lambda: super(ObservedService, self)._sample(*args, **kwargs),
            )

    return ObservedService, ObservedHead


def idle_lane(backend):
    lane, cache = backend.lane, backend.cache
    require(
        not backend.prefixes
        and not lane.branches
        and not any(lane.pool.references)
        and sorted(lane.pool.free) == list(range(cache.max_num_tokens // 256))
        and sorted(cache.free_list) == list(range(cache.num_slots)),
        "Cannot transfer an occupied or contradictory native cache",
    )


class LoadedOwner:
    def __init__(self, config):
        self.config, self.owner = config, get_ident()
        self.backend = self.generator = self.service = None
        self.cancel = Event()

    def check_owner(self):
        require(
            get_ident() == self.owner,
            "All model/cache/head work requires the single owner",
        )

    def load(self, workloads):
        self.check_owner()
        config = self.config
        require(
            config.native_tree.is_absolute() and config.estimate_basis.strip(),
            "Absolute native source and a nonblank memory estimate basis required",
        )
        recipes = Path(__file__).resolve().parents[1]
        sources = {
            "recipes": clean_source(recipes, config.expected_recipes_commit),
            "native": clean_source(config.native_tree, NATIVE_COMMIT),
            "foundation_origin": clean_source(
                ROOT / "src/exllamav3-fuzev", FOUNDATION_COMMIT
            ),
        }
        require(
            not any(
                name == "exllamav3" or name.startswith("exllamav3.")
                for name in sys.modules
            ),
            "Do not borrow a pre-imported native engine or resident model",
        )
        require(
            config.native_extension.is_absolute(),
            "Use the genuine cached native e139 binary, never JIT or rebuild",
        )
        sys.path[:0] = [
            str(config.native_tree.resolve()),
            str(config.native_extension.parent.resolve()),
        ]
        extension = importlib.util.find_spec("exllamav3_ext")
        require(
            extension is not None
            and Path(extension.origin).resolve() == config.native_extension.resolve(),
            "Actual extension import must bind the attested binary before model imports",
        )
        from .native import NativeBackend, runtime_sources
        from .kev_research_server import PLAN, QUALITY
        import torch

        require(
            torch.cuda.is_available(),
            "Original CUDA Torch target required",
        )
        torch.set_num_threads(10)
        require(
            config.plan.path.resolve() == PLAN.resolve()
            and config.quality.path.resolve() == QUALITY.resolve(),
            "Canonical research head must use its original frozen plan and quality artifacts",
        )
        require(
            config.workloads.path.resolve()
            == (
                ROOT
                / "runs/native-real/qualification/native4-performance-workloads.json"
            ).resolve(),
            "Use only the original three procedural workload definitions",
        )
        plan, quality = config.plan.read(), config.quality.read()
        case = plan["candidates"]["lm-assisted"]
        require(
            quality["status"] == "pass"
            and quality["plan_sha256"] == config.plan.sha256
            and quality["primary"] == "lm-assisted"
            and quality["test_sha256"] == plan["test_sha256"]
            and quality["comparison"]["passed"] is True,
            "Frozen A/raw-T1 research prerequisites changed; no new selection or CAL",
        )
        require(
            quality["uncertainty"]["index"]["lm-assisted"][0]
            > plan["baseline_index_ci"][1],
            "Original published-level Kev research admission changed",
        )
        sources["source8"] = clean_source(case["tree"], SOURCE8_COMMIT)
        original = json.loads(Path(plan["original_feature_identity"]).read_text())
        require(
            original["encoding"] == encoding_identity()
            and original["hidden_size"] == 2560
            and original["feature_tap"] == "final-mixer/post-collapse/before-lm-head",
            "Canonical head reference encoding or tap differs",
        )
        model = json.loads((ROOT / "data/manifests/acquisition.json").read_text())[
            "models"
        ][config.model]
        directory = Path(model["local_snapshot"]).resolve()
        metadata = json.loads((directory / "config.json").read_text())
        text = metadata.get("text_config", metadata)
        require(
            text.get("hidden_size") == 2560
            and text.get("max_position_embeddings") == WINDOW
            and metadata.get("quantization_config", {}).get("quant_method") == "exl3",
            "Acquired native4 configuration has incompatible geometry/quantization",
        )
        self.plan, self.quality = plan, quality
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.backend = NativeBackend(
            model["local_snapshot"],
            cache_tokens=294912,
            memory_fraction=config.memory_fraction,
            optimized=True,
            candidate_batch_size=4,
            batch_capacity=4,
        )
        imported = runtime_sources()
        self.backend.model.enable_performance_counters()
        prepared = [
            prepare_workload(workload, tier, self.backend)
            for workload in workloads
            for tier in (2048, 32768)
        ]
        idle_lane(self.backend)
        self.backend.lane.close()
        self.backend.lane = None
        from exllamav3.generator.generator import Generator

        self.generator = Generator(
            self.backend.model,
            self.backend.cache,
            self.backend.tokenizer,
            max_batch_size=4,
            max_chunk_size=8192,
        )
        self.provenance = {
            "sources": sources,
            "imported_sources": imported,
            "model": model,
            "model_config_sha256": sha(directory / "config.json"),
            "quantization_config_sha256": sha(directory / "quantization_config.json"),
            "acquisition_sha256": sha(ROOT / "data/manifests/acquisition.json"),
            "native_extension_path": str(config.native_extension.resolve()),
            "native_extension_sha256": sha(config.native_extension),
            "recorded_native_extension_sha256": EXTENSION_SHA,
            "recorded_foundation_sha256": FOUNDATION_SHA,
            "foundation_origin_tree": str(ROOT / "src/exllamav3-fuzev"),
            "readout_runtime": self.backend.readout_runtime,
            "head_sha256": sha(case["weights"]), "recorded_head_sha256": HEAD_SHA,
            "head_source_commit": case.get("commit"), "recorded_head_source_commit": SOURCE8_COMMIT,
            "recorded_model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "reference_feature_identity": original,
            "calibration_id": "uncalibrated:T1",
            "raw_temperature": 1.0,
            "native_original_ba606_qualification_inherited": False,
        }
        return prepared, self.provenance

    def counters(self):
        self.check_owner()
        return self.backend.model.get_performance_counters()

    def memory(self):
        self.check_owner()
        import psutil
        import torch

        return {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(0),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(0),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "ram_rss_bytes": psutil.Process().memory_info().rss,
            "ram_available_bytes": psutil.virtual_memory().available,
            "swap": psutil.swap_memory()._asdict(),
            "scope": "allocator/host observations outside requests; no added CUDA synchronization",
        }

    def stock(self, workload, streams):
        self.check_owner()
        require(
            self.service is None
            and self.backend.lane is None
            and self.generator is not None,
            "Stock and lane owners must never coexist",
        )
        requests = [
            replace(workload.stock, name=f"{workload.name}-{index}")
            for index in range(streams)
        ]
        started, before = time.perf_counter(), self.counters()
        receipt = run_native_generation_control(
            self.generator,
            requests,
            allow_generation=True,
            exclusive=True,
            additional_peak_bytes=self.config.additional_peak_bytes,
            estimate_basis=self.config.estimate_basis,
            memory_fraction=self.config.memory_fraction,
            timeout_seconds=self.config.timeout_seconds,
            cancel=self.cancel,
            expected_engine_commit=NATIVE_COMMIT,
        )
        wall = time.perf_counter() - started
        samples = [
            {
                "status": "complete",
                "workload": workload.name,
                "mode": "stock-generation",
                "elapsed_seconds": job["last_accepted_observed_seconds"],
                "latency_scope": "CPU accepted-token observation from common stock batch start",
                "completion_tokens": job["completion_tokens"],
                "logical_prefix_tokens": job["prompt_tokens"],
                "forwarded_tokens": None,
                "forward_counts_status": None,
                "generation_phase_timing": None,
                "generation": job,
            }
            for job in receipt["jobs"]
        ]
        after = self.counters()
        return (
            samples,
            wall,
            {
                **counter_delta(before, after),
                "counter_before": before,
                "counter_after": after,
            },
            receipt,
        )

    def enter_fused(self):
        self.check_owner()
        from .native_generation_control import _idle

        require(
            self.service is None
            and self.backend.lane is None
            and self.generator is not None,
            "Invalid exclusive stock-to-lane transition",
        )
        _idle(self.generator)
        self.generator.filter_pool.shutdown(wait=True)
        # Retire the stock owner serial before any fresh lane can use its pages.
        self.backend.cache.owner_serial += 1
        self.generator = None
        gc.collect()
        try:
            from exllamav3.fuzev import NativeLane
        except ImportError as error:
            raise ImportError("Requires the historical engine module exllamav3/fuzev.py "
                              "from research-snapshot/historical-native-source") from error
        from .kev_research_server import ResearchHead, ResearchService
        from .telemetry import MemoryGuard

        self.backend.lane = NativeLane(
            self.backend.model,
            self.backend.cache,
            context_limit=WINDOW,
            chunk_size=8192,
        )
        idle_lane(self.backend)
        service_type, head_type = observed_classes(ResearchService, ResearchHead)
        head = head_type(self.backend, self.plan, self.quality)
        require(
            head.calibration_id == "uncalibrated:T1",
            "Frozen raw head changed",
        )
        guard = MemoryGuard(fraction=self.config.memory_fraction)
        try:
            guard.start()
            self.service = service_type(
                self.backend,
                self.config.model,
                head,
                self.provenance,
                guard,
                capacity=4,
                ttl=3600.0,
            )
        except BaseException:
            guard.close()
            raise

    def generate(self, workload, request_id):
        self.check_owner()
        result = self.service.generate(workload.generation, self.cancel, request_id)
        expected = tuple(workload.stock.prompt_token_ids) + tuple(
            result["completion_token_ids"]
        )
        try:
            require(
                result["accepted_token_sha256"] == token_hash(expected)
                and result["usage"]["prompt_tokens"]
                == len(workload.stock.prompt_token_ids)
                and tuple(self.service._get_prefix(result["prefix_id"]).accepted)
                == expected,
                "Stock/fused generation rendering or exact accepted IDs differ",
            )
        except BaseException:
            self.service.release_prefix(result["prefix_id"])
            raise
        return result

    def decide(self, workload, index, generation, request_id):
        self.check_owner()
        spec = workload.decisions[index % 3]
        context = (
            {"prefix_id": generation["prefix_id"]}
            if generation
            else {"messages": workload.standalone_messages}
        )
        request = ServiceDecisionRequest.model_validate(
            spec.request(self.config.model, "reference", **context)
        )
        result = self.service.decide(request, self.cancel, request_id)
        require(
            not result["qualified"]
            and not result["calibrated"]
            and result["calibration_id"] == "uncalibrated:T1",
            "Research head policy changed",
        )
        return result

    def release(self, prefix_id):
        self.check_owner()
        require(
            self.service.release_prefix(prefix_id),
            "Owned performance prefix was not released",
        )

    def postcheck(self):
        self.check_owner()
        require(not self.service.prefixes, "Retained performance prefix leaked")
        idle_lane(self.backend)
        clean_source(self.config.native_tree, NATIVE_COMMIT)
        clean_source(
            Path(__file__).resolve().parents[1], self.config.expected_recipes_commit
        )
        case = self.plan["candidates"]["lm-assisted"]
        clean_source(case["tree"], SOURCE8_COMMIT)
        self.provenance["observed_head_sha256_after"] = sha(case["weights"])
        return self.counters()

    def close(self):
        self.check_owner()
        errors = []
        for name, call in (
            ("service", lambda: self.service.close() if self.service else None),
            (
                "stock_filter_pool",
                lambda: (
                    self.generator.filter_pool.shutdown(wait=True)
                    if self.generator
                    else None
                ),
            ),
            (
                "lane",
                lambda: (
                    self.backend.lane.close()
                    if self.backend and self.backend.lane
                    else None
                ),
            ),
            ("model", lambda: self.backend.model.unload() if self.backend else None),
        ):
            try:
                call()
            except Exception as error:
                errors.append(
                    {"stage": name, "error": f"{type(error).__name__}: {error}"}
                )
        self.service = self.generator = self.backend = None
        return errors


def wave_summary(samples, wall, counts):
    complete = [sample for sample in samples if sample["status"] == "complete"]
    all_complete = len(complete) == len(samples)
    summary = (
        summarize_samples(samples, wall)
        if samples and all_complete
        else {
            "requests": len(samples),
            "wall_seconds_including_cleanup": wall,
            "completion_tokens_per_pipeline_second_including_prefill": None,
            "latency_p50_seconds": None,
            "latency_p95_seconds": None,
            "scope": "failed wave retained whole; no success-only throughput",
        }
    )
    physical = counts["physical_forwarded_tokens"] if all_complete else None
    return {
        **summary,
        "requested_pipelines": len(samples),
        "failed_pipelines": len(samples) - len(complete),
        "counter_delta": counts,
        "physical_forwarded_tokens": physical,
        "physical_forwarded_positions_per_wall_second": physical / wall
        if physical is not None and wall > 0
        else None,
        "logical_prefix_tokens_per_wall_second": summary["logical_prefix_tokens"] / wall
        if all_complete and samples and wall > 0
        else None,
        "per_request_physical_attribution": "unavailable; genuine totals are exclusive-wave owner-wide",
    }


async def fused_wave(owner, pool, workload, mode, streams, wave):
    loop = asyncio.get_running_loop()

    async def call(method, *args):
        return await loop.run_in_executor(pool, method, *args)

    async def pipeline(index):
        started, generation, decision = time.perf_counter(), None, None
        sample = {
            "status": "failed",
            "workload": workload.name,
            "mode": mode,
            "index": index,
        }
        try:
            request_id = f"performance-{workload.name}-{mode}-{wave}-{index}"
            if mode != "decision-only":
                generation = await call(
                    owner.generate, workload, request_id + "-generation"
                )
            if mode != "generation-only":
                decision = await call(
                    owner.decide,
                    workload,
                    max(0, wave) * streams + index,
                    generation,
                    request_id + "-decision",
                )
            sample.update(
                status="complete",
                elapsed_seconds=time.perf_counter() - started,
                completion_tokens=generation["usage"]["completion_tokens"]
                if generation
                else 0,
                logical_prefix_tokens=(
                    generation["usage"]["prompt_tokens"] if generation else 0
                )
                + (decision["execution"]["logical_prefix_tokens"] if decision else 0),
                generation_phase_timing=generation_phases(generation, "native")
                if generation
                else None,
                forwarded_tokens=None,
                forward_counts_status=None,
                generation=generation,
                decision=decision,
                service_arithmetic_execution=decision["execution"]
                if decision
                else None,
            )
        except Exception as error:
            sample["error"] = f"{type(error).__name__}: {error}"
            sample.update(generation=generation, decision=decision)
        finally:
            if generation is not None:
                try:
                    await call(owner.release, generation["prefix_id"])
                except Exception as error:
                    sample.update(
                        status="failed",
                        cleanup_error=f"{type(error).__name__}: {error}",
                    )
        return sample

    before, started = await call(owner.counters), time.perf_counter()
    samples = await asyncio.gather(*(pipeline(index) for index in range(streams)))
    wall = time.perf_counter() - started
    after = await call(owner.postcheck)
    return (
        samples,
        wall,
        {
            **counter_delta(before, after),
            "counter_before": before,
            "counter_after": after,
        },
        None,
    )


async def run_operator(
    config, output, *, allow_inference=False, exclusive=False, owner_factory=LoadedOwner
):
    require(
        allow_inference is True and exclusive is True,
        "Future only: explicitly authorize inference and exclusive offline ownership; no live panel overlap",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output / "config.json", config.model_dump(mode="json"))
    report = {
        "schema_version": 1,
        "status": "in_progress",
        "qualified": False,
        "servable": False,
        "runtime_qualified": False,
        "scope": "sequential one-resident-model native stock/fused offline performance",
        "matrix": {
            "contexts": [2048, 32768],
            "streams": [1, 4],
            "modes": list(MODES),
            "warmup_waves": 1,
            "measured_waves": 3,
        },
        "rng_equivalence_claimed": False,
        "full_window_scope": "separate feasibility operator; not measured here",
        "ownership_scope": "one language backbone; stock retired before lane/head; fused requests serialize on one owner; candidate tails batch <=4",
        "warmup": [],
        "performance": [],
        "cleanup_failures": [],
    }
    write_json(output / "report.json", report)
    owner = None
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="native-performance-owner"
    ) as pool:
        loop = asyncio.get_running_loop()

        async def call(method, *args):
            return await loop.run_in_executor(pool, method, *args)

        try:
            evidence = config.correctness_evidence.read()
            require(
                isinstance(evidence, dict)
                and evidence.get("status")
                in {"passed", "passed-scoped-checks", "completed", "complete", "failed"}
                and evidence.get("model") == config.model,
                "Pin terminal original native correctness, including failures",
            )
            report.update(
                existing_correctness_status=evidence["status"],
                correctness_evidence=config.correctness_evidence.model_dump(
                    mode="json"
                ),
            )
            workloads = load_workloads(config)
            owner = await call(owner_factory, config)
            prepared, report["provenance"] = await call(owner.load, workloads)
            write_json(
                output / "workloads-rendered.json",
                [workload.receipt for workload in prepared],
            )
            report["memory_before_stock"] = await call(owner.memory)
            for mode in MODES:
                if mode == "generation-only":
                    await call(owner.enter_fused)
                    report["memory_before_fused"] = await call(owner.memory)
                for workload in prepared:
                    for streams in (1, 4):
                        group = {
                            "workload": workload.name,
                            "mode": mode,
                            "streams": streams,
                            "waves": [],
                        }
                        report["performance"].append(group)
                        for wave in range(4):
                            report["active_wave"] = {
                                "workload": workload.name,
                                "mode": mode,
                                "streams": streams,
                                "wave": max(0, wave - 1),
                                "warmup": wave == 0,
                            }
                            write_json(output / "report.json", report)
                            if mode == "stock-generation":
                                samples, wall, counts, receipt = await call(
                                    owner.stock, workload, streams
                                )
                            else:
                                samples, wall, counts, receipt = await asyncio.wait_for(
                                    fused_wave(
                                        owner, pool, workload, mode, streams, wave - 1
                                    ),
                                    timeout=config.timeout_seconds,
                                )
                            result = {
                                **report.pop("active_wave"),
                                "samples": samples,
                                "stock_receipt": receipt,
                                **wave_summary(samples, wall, counts),
                            }
                            (report["warmup"] if wave == 0 else group["waves"]).append(
                                result
                            )
                            write_json(output / "report.json", report)
                            require(
                                result["failed_pipelines"] == 0,
                                "Failed wave retained; do not retry or exclude",
                            )
                        group["summary"] = wave_summary(
                            [
                                sample
                                for item in group["waves"]
                                for sample in item["samples"]
                            ],
                            sum(
                                item["wall_seconds_including_cleanup"]
                                for item in group["waves"]
                            ),
                            counter_delta(
                                group["waves"][0]["counter_delta"]["counter_before"],
                                group["waves"][-1]["counter_delta"]["counter_after"],
                            ),
                        )
            report["counter_after"] = await call(owner.postcheck)
            report["memory_after"] = await call(owner.memory)
            config.workloads.read()
            config.plan.read()
            config.quality.read()
            config.correctness_evidence.read()
            report["status"] = "measurement-complete"
        except BaseException as error:
            if owner is not None:
                owner.cancel.set()
            report.update(
                status="cancelled"
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            if owner is not None:
                try:
                    report["counter_at_cleanup"] = await call(owner.counters)
                except Exception as error:
                    report["counter_at_cleanup_unavailable"] = (
                        f"{type(error).__name__}: {error}"
                    )
                report["cleanup_failures"] = await call(owner.close)
            if report["cleanup_failures"]:
                report["status"] = "failed"
            write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-inference", action="store_true")
    parser.add_argument("--exclusive", action="store_true")
    args = parser.parse_args()
    if not (args.allow_inference and args.exclusive):
        parser.error(
            "Explicit exclusive offline inference permission required; never overlap a live panel"
        )
    config = OperatorConfig.model_validate_json(args.config.read_text())
    report = asyncio.run(
        run_operator(config, args.output, allow_inference=True, exclusive=True)
    )
    print(
        json.dumps({"status": report["status"], "qualified": False, "servable": False})
    )
    raise SystemExit(0 if report["status"] == "measurement-complete" else 1)


if __name__ == "__main__":
    main()
