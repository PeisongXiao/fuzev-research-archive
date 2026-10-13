"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks.

Nothing in this module is imported by the frozen native extraction server.
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from .service_contracts import GenerationRequest, ServiceDecisionRequest
from .vllm_ple_storage import validate_ple_storage_config, validate_ple_storage_options
from .vllm_service import (CONTEXT, MAX_INPUT_BYTES,
                           QUALIFICATION, VllmServiceConfig, build_service,
                           drain_tasks, validate_engine_config)


class BoundedBody:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        parts, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            size += len(part)
            if size > MAX_INPUT_BYTES:
                return await JSONResponse({"detail": "Request exceeds 8 MiB resource bound"},
                                          status_code=413)(scope, receive, send)
            parts.append(part)
            if not message.get("more_body", False):
                break
        body = b"".join(parts)
        sent = False

        async def bounded_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(engine_context, config, *, builder=build_service):
    """engine_context yields ONE client; attaching never creates another model."""
    @asynccontextmanager
    async def lifespan(app):
        async with engine_context() as engine_client:
            service = await builder(engine_client, config)
            app.state.service = service

            async def expire():
                while True:
                    await asyncio.sleep(min(30., config.prefix_ttl_seconds / 2))
                    service.expire()

            sweeper = asyncio.create_task(expire())
            try:
                yield
            finally:
                sweeper.cancel()
                await drain_tasks([sweeper])
                await service.close()

    app = FastAPI(title="FuZev shared vLLM experiment", lifespan=lifespan)
    app.add_middleware(BoundedBody)

    async def execute(http, function):
        request_id = http.headers.get("x-fuzev-request-id", "req_" + uuid4().hex)
        task = asyncio.create_task(function(request_id))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=.1)
                if done:
                    return task.result()
                if await http.is_disconnected():
                    task.cancel()
                    await drain_tasks([task])
                    raise HTTPException(499, "Client disconnected; abort submitted")
        except asyncio.CancelledError:
            task.cancel()
            await drain_tasks([task])
            if asyncio.current_task().cancelling():
                raise
            raise HTTPException(499, "Request cancelled; abort submitted") from None
        except FileExistsError as error:
            raise HTTPException(409, str(error)) from error
        except KeyError as error:
            raise HTTPException(404, "Unknown, expired or evicted token prefix") from error
        except MemoryError as error:
            raise HTTPException(507, str(error)) from error
        except TimeoutError as error:
            raise HTTPException(504, "Service request deadline exceeded; abort submitted") from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    @app.post("/v1/completions")
    @app.post("/v1/chat/completions")
    async def generate(request: GenerationRequest, http: Request):
        if (http.url.path == "/v1/chat/completions") != (request.messages is not None):
            raise HTTPException(400, "Chat requires messages; completion requires prompt")
        return await execute(http, lambda key: app.state.service.generate(request, key))

    @app.post("/v1/decisions")
    async def decide(request: ServiceDecisionRequest, http: Request):
        return await execute(http, lambda key: app.state.service.decide(request, key))

    @app.delete("/v1/prefixes/{prefix_id}")
    async def release(prefix_id: str):
        return {"id": prefix_id, "released": app.state.service.release_prefix(prefix_id),
                "scope": "service token receipt only; engine prefix cache is engine-owned"}

    @app.post("/v1/requests/{request_id}/cancel", status_code=202)
    async def cancel(request_id: str):
        if not app.state.service.cancel(request_id):
            raise HTTPException(404, "Unknown or completed request")
        return {"id": request_id, "cancel_requested": True,
                "cleanup": "await abort API and iterator cleanup; no GPU-free acknowledgement"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": config.model_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        service = app.state.service
        try:
            service._check()
            await service.engine.check_health()
        except (MemoryError, RuntimeError) as error:
            return JSONResponse({"status": "unavailable", "reason": str(error)}, status_code=503)
        return {"status": "ok", "qualification": QUALIFICATION,
                "context_window": CONTEXT,
                "active_engine_requests": len(service.engine_requests),
                "retained_prefixes": len(service.prefixes), "retained_tokens": service.retained_tokens,
                "retained_multimodal_bytes": service.retained_multimodal_bytes,
                "capabilities": {"text_generation": True, "typed_decisions": True,
                                 "trusted_still_images": service.images is not None,
                                 "image_input_contract": "acquired digest.png path plus RGB sha256",
                                 "images": service.images is not None,
                                 "streaming": False, "checkpoint_retention": False},
                "worker_readout": service.readout_health(),
                "provenance": service.provenance}

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-config", required=True, help="Local JSON AsyncEngineArgs")
    parser.add_argument("--service-config", required=True, help="Local JSON VllmServiceConfig pins")
    args = parser.parse_args()
    config = VllmServiceConfig.model_validate_json(Path(args.service_config).read_text())
    engine_options = json.loads(Path(args.engine_config).read_text())
    validate_ple_storage_options(engine_options, quantization=config.runtime_cell.get("quantization"))

    @asynccontextmanager
    async def engine_context():
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.fuzev import fuzev_enabled
        from .telemetry import admit
        engine_config = AsyncEngineArgs(**engine_options).create_engine_config()
        validate_engine_config(engine_config)
        validate_ple_storage_options(engine_options, quantization=engine_config.model_config.quantization)
        validate_ple_storage_config(engine_config)
        if not fuzev_enabled(engine_config):
            raise ValueError("This existing generation runner must have FuZev enabled")
        admit(128 << 20)
        from vllm.v1.engine.async_llm import AsyncLLM
        engine = AsyncLLM.from_vllm_config(engine_config)
        try:
            yield engine
        finally:
            engine.shutdown()

    uvicorn.run(create_app(engine_context, config), host=args.host, port=args.port,
                timeout_graceful_shutdown=30)


if __name__ == "__main__":
    main()
