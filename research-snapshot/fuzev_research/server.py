"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks."""

import argparse
import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from threading import Event
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .contracts import DecisionRequest
from .extraction import FeatureProxy, PIPELINE_POLICY
from .native import NativeBackend, extract_features, runtime_sources
from .native_service import build_service
from .owner_queue import OwnerQueue
from .paths import project_path
from .service_contracts import GenerationRequest, ServiceConfig, ServiceDecisionRequest, TrustedImage
from .service_images import IMAGE_ADMISSION_BYTES, checked_image
from .telemetry import admit
from .scope import ALL_PREPARATIONS as PREPARATIONS, contract, validate_manifest


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: dict
    baseline: bool = False


class ExtractionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prepared: list[str] = Field(min_length=1, max_length=6)


def create_app(model_name="native-4.05", batch_capacity=4, candidate_batch_size=4,
               *, service_config: ServiceConfig | None = None, service_factory=None, memory_fraction=None):
    @asynccontextmanager
    async def lifespan(app):
        manifest = json.loads(project_path("data/manifests/acquisition.json").read_text())
        model = manifest["models"][model_name]
        if not model.get("local_snapshot"):
            raise RuntimeError("Pinned model acquisition is not complete")
        app.state.sources = runtime_sources()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-gpu-owner")
        writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fuzev-feature-writer")
        app.state.executor, app.state.cancel = executor, Event()
        app.state.queue = OwnerQueue(executor)
        app.state.writer = writer
        app.state.active, app.state.jobs, app.state.job_cancel = 0, {}, {}
        app.state.requests, app.state.service = {}, None
        loop = asyncio.get_running_loop()
        backend, sweeper = None, None
        try:
            backend = await loop.run_in_executor(executor, partial(NativeBackend, model["local_snapshot"],
                batch_capacity=batch_capacity, candidate_batch_size=candidate_batch_size, memory_fraction=memory_fraction))
            if service_config is not None:
                factory = service_factory or build_service
                app.state.service = await loop.run_in_executor(executor, partial(
                    factory, backend, model_name, model, app.state.sources, service_config))
                sweeper = asyncio.create_task(expire_prefixes())
            app.state.backend, app.state.ready = backend, True
            yield
        finally:
            app.state.ready = False
            app.state.cancel.set()
            for cancel in app.state.job_cancel.values():
                cancel.set()
            for cancel in list(app.state.requests.values()):
                cancel.set()
            if sweeper is not None:
                sweeper.cancel()
                await asyncio.gather(sweeper, return_exceptions=True)
            writer.shutdown(wait=True, cancel_futures=True)
            if app.state.service is not None:
                await loop.run_in_executor(executor, app.state.service.close)
            if backend is not None:
                await loop.run_in_executor(executor, backend.close)
            executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="FuZev native experiment", lifespan=lifespan)

    async def expire_prefixes():
        while True:
            await asyncio.sleep(min(30., service_config.prefix_ttl_seconds / 2))
            future = app.state.queue.submit(app.state.service.expire)
            await asyncio.shield(asyncio.wrap_future(future))

    async def owner_request(http, function):
        request_id = http.headers.get("x-fuzev-request-id", "req_" + uuid4().hex)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            raise HTTPException(400, "Invalid X-Fuzev-Request-ID")
        if request_id in app.state.requests:
            raise HTTPException(409, "Request ID is already active")
        cancel = Event()

        def execute():
            if cancel.is_set():
                raise InterruptedError("Cancelled before native admission")
            return function(cancel, request_id)

        future = app.state.queue.submit(execute)
        app.state.requests[request_id] = cancel
        loop = asyncio.get_running_loop()
        future.add_done_callback(lambda _: loop.call_soon_threadsafe(
            app.state.requests.pop, request_id, None))
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            while True:
                done, _ = await asyncio.wait({wrapped}, timeout=.1)
                if done:
                    return wrapped.result()
                if await http.is_disconnected():
                    cancel.set()
        except asyncio.CancelledError:
            cancel.set()
            raise
        except InterruptedError as error:
            raise HTTPException(499, str(error)) from error
        except KeyError as error:
            raise HTTPException(404, "Unknown, expired or evicted prefix") from error
        except (ValueError, MemoryError) as error:
            raise HTTPException(507 if isinstance(error, MemoryError) else 400, str(error)) from error

    def require_service():
        if app.state.service is None:
            raise HTTPException(503, "Shared service is disabled; calibrated artifact pins are required")
        return app.state.service

    @app.post("/v1/completions")
    @app.post("/v1/chat/completions")
    async def generate(request: GenerationRequest, http: Request):
        service = require_service()
        if (http.url.path == "/v1/chat/completions") != (request.messages is not None):
            raise HTTPException(400, "Chat requires messages; completion requires a prompt")
        return await owner_request(http, lambda cancel, key: service.generate(request, cancel, key))

    @app.post("/v1/decisions")
    async def decide(request: ServiceDecisionRequest, http: Request):
        service = require_service()
        return await owner_request(http, lambda cancel, key: service.decide(request, cancel, key))

    @app.delete("/v1/prefixes/{prefix_id}")
    async def release_prefix(prefix_id: str, http: Request):
        service = require_service()
        return await owner_request(http, lambda _cancel, _key: {
            "id": prefix_id, "released": service.release_prefix(prefix_id)})

    @app.post("/v1/requests/{request_id}/cancel", status_code=202)
    async def cancel_request(request_id: str):
        cancel = app.state.requests.get(request_id)
        if cancel is None:
            raise HTTPException(404, "Unknown or completed request")
        cancel.set()
        return {"id": request_id, "cancel_requested": True,
                "boundary": "next target-forward boundary; admission held until cleanup"}

    @app.get("/health")
    def health():
        return {"status": "ok" if app.state.ready else "stopping", "model": model_name,
                "phase": "shared-native-unqualified" if app.state.service else "feature-pilot",
                "context_window": 262144, "memory_fraction": app.state.backend.readout_runtime["memory_fraction"],
                "cache_precision": "float16", "ple": "NVMe row streaming",
                "gpu_owner_threads": 1,
                "candidate_batch_size": app.state.backend.candidate_batch_size,
                "readout_runtime": app.state.backend.readout_runtime,
                "extraction_scheduling": dict(PIPELINE_POLICY),
                "generation_and_decision_service": "implemented, not GPU-qualified" if app.state.service else "disabled",
                "service_capabilities": {"text_generation": app.state.service is not None,
                                         "typed_decisions": app.state.service is not None,
                                         "trusted_still_images": app.state.service is not None,
                                         "image_input_contract": "acquired digest.png path plus RGB sha256; no URLs or uploads",
                                         "streaming": False, "speculative_decoding": False},
                "sources": app.state.sources}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "fuzev-local-experiment"}]}

    @app.post("/v1/features")
    async def features(request: FeatureRequest, http: Request):
        row = request.row
        source = None
        try:
            DecisionRequest(model=model_name, primitive=row["primitive"], question=row["question"],
                            candidates=row["candidates"], messages=row["context"])
            if row.get("image_path"):
                source = TrustedImage(path=row["image_path"], sha256=row["image_sha256"])
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

        def readout(cancel, _key):
            if source is None:
                return app.state.backend.extract(row, request.baseline)
            admit(IMAGE_ADMISSION_BYTES, app.state.backend.readout_runtime["memory_fraction"])
            with checked_image(source, project_path("data/images")) as snapshot:
                if cancel.is_set():
                    raise InterruptedError("Cancelled after image validation")
                # Only HTTP inputs need this boundary. The pinned prepared
                # corpus and its CPU preparation/write overlap are unchanged.
                return app.state.backend.extract({**row, "image_path": snapshot}, request.baseline)

        try:
            result = await owner_request(http, readout)
            query, candidates, baseline, execution = result
            return {"feature_tap": "final-mixer/post-collapse/before-lm-head", "precision": "float32",
                    "query": query.tolist(), "candidates": candidates.tolist(),
                    "baseline_logits": baseline.tolist() if baseline is not None else None,
                    "execution": execution}
        except (ValueError, MemoryError) as error:
            raise HTTPException(507 if isinstance(error, MemoryError) else 400, str(error)) from error

    @app.post("/v1/jobs/extract", status_code=202)
    async def start_extraction(request: ExtractionJob):
        if any(not future.done() for future in app.state.jobs.values()):
            raise HTTPException(409, "A frozen-feature job is already active")
        active = contract(augmentation=os.environ.get("FUZEV_NOUL_PREFERENCE") == "1",
                          dimensions=os.environ.get("FUZEV_DIMENSION_CHOICE") == "1")
        if len(set(request.prepared)) != len(request.prepared) or set(request.prepared) - set(active["preparations"]):
            raise HTTPException(400, "Only registered prepared corpus files can be extracted")
        paths = [project_path("data/prepared") / filename for filename in request.prepared]
        if any(not path.exists() for path in paths):
            raise HTTPException(400, "Requested preparation is incomplete")
        try:
            for path in paths:
                validate_manifest(path.name, json.loads(project_path(
                    "data/manifests/" + PREPARATIONS[path.name][0]).read_text()))
        except (ValueError, OSError) as error:
            raise HTTPException(400, "Preparation is not verified for the active scope: " + str(error)) from error
        if runtime_sources() != app.state.sources:
            raise HTTPException(409, "Sources changed since runtime startup; do not misidentify cached features")
        job = str(uuid4())
        cancel = Event()
        proxy = FeatureProxy(app.state.backend, app.state.executor,
                             app.state.queue if app.state.service is not None else None)
        future = asyncio.get_running_loop().run_in_executor(app.state.writer,
            partial(extract_features, model_name, paths, proxy, cancel))
        # Observe exceptions rather than leaving an unhandled background future.
        future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        app.state.jobs[job] = future
        app.state.job_cancel[job] = cancel
        return {"id": job, "status": "in_progress", "prepared": request.prepared}

    @app.post("/v1/jobs/{job}/cancel", status_code=202)
    def cancel_job(job: str):
        if job not in app.state.jobs:
            raise HTTPException(404, "Unknown experimental job")
        app.state.job_cancel[job].set()
        return {"id": job, "cancel_requested": True,
                "boundary": "after the current exact readout; committed shards retained"}

    @app.get("/v1/jobs/{job}")
    def job_status(job: str):
        if job not in app.state.jobs:
            raise HTTPException(404, "Unknown experimental job")
        future = app.state.jobs[job]
        if not future.done():
            return {"id": job, "status": "in_progress"}
        error = future.exception()
        status = "cancelled" if isinstance(error, InterruptedError) else "failed" if error else "completed"
        return {"id": job, "status": status, "error": str(error) if error else None}

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="native-4.05")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--service-config", help="Opt-in operator-owned artifact/source pins; no implicit head fallback")
    parser.add_argument("--batch-capacity", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--candidate-batch-size", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--memory-fraction", type=float, default=None, help="Optional host memory fraction (default: no cap)")
    args = parser.parse_args()
    if args.candidate_batch_size > args.batch_capacity:
        parser.error("Candidate batch size must not exceed batch capacity")
    config = None
    if args.service_config:
        from pathlib import Path
        config = ServiceConfig.model_validate_json(Path(args.service_config).read_text())
    uvicorn.run(create_app(args.model, args.batch_capacity, args.candidate_batch_size,
                          service_config=config, memory_fraction=args.memory_fraction),
                host=args.host, port=args.port, timeout_graceful_shutdown=30)


if __name__ == "__main__":
    main()
