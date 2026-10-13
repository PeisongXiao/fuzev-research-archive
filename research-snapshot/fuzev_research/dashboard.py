import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from .state import Store
from .telemetry import Sampler


@asynccontextmanager
async def lifespan(app):
    sampler = Sampler()
    stop = asyncio.Event()

    async def collect():
        next_tick = asyncio.get_running_loop().time()
        while not stop.is_set():
            await asyncio.to_thread(app.state.store.record_metric, await asyncio.to_thread(sampler.sample))
            next_tick += 1
            delay = max(0, next_tick - asyncio.get_running_loop().time())
            if delay == 0:
                next_tick = asyncio.get_running_loop().time()
            try:
                await asyncio.wait_for(stop.wait(), delay)
            except asyncio.TimeoutError:
                pass

    collector = asyncio.create_task(collect())
    try:
        yield
    finally:
        stop.set()
        await collector


def create_app(store=None):
    app = FastAPI(title="FuZev observability", lifespan=lifespan)
    app.state.store = store or Store()
    app.state.store.initialize()

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "read_only": True}

    @app.get("/api/state")
    def state():
        return app.state.store.snapshot()

    @app.get("/api/metrics")
    def metrics(since: float = 0, limit: int = 900):
        return app.state.store.metrics(since, max(1, limit))

    @app.get("/api/events")
    async def events():
        async def stream():
            while True:
                data = await asyncio.to_thread(app.state.store.snapshot)
                yield "data: " + json.dumps(data, allow_nan=False) + "\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return app
