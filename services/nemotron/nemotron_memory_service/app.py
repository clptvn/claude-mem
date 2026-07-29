from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .durable_queue import DurableWriteQueue
from .model import EmbeddingBatcher, ModelEngine
from .settings import ServiceSettings
from .vector_store import NemotronVectorStore


class VectorCallRequest(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    input_type: Literal["query", "passage"] = "passage"


@dataclass
class Runtime:
    settings: ServiceSettings
    engine: ModelEngine
    batcher: EmbeddingBatcher
    store: NemotronVectorStore
    jobs: DurableWriteQueue
    started_at: float
    eager_task: asyncio.Task[None] | None = None


async def _swallow_eager_load(engine: ModelEngine) -> None:
    try:
        await engine.ensure_loaded()
    except Exception:
        # Health exposes the error and requests retain the normal retry path.
        pass


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    effective = settings or ServiceSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = ModelEngine(effective)
        batcher = EmbeddingBatcher(engine, effective)
        batcher.start()
        store = NemotronVectorStore(effective.chroma_dir, batcher)
        jobs = DurableWriteQueue(
            effective.queue_path,
            store.call,
            effective.queue_batch_size,
            effective.write_timeout_seconds,
        )
        jobs.start()
        runtime = Runtime(effective, engine, batcher, store, jobs, time.time())
        app.state.runtime = runtime
        if effective.eager_load:
            runtime.eager_task = asyncio.create_task(
                _swallow_eager_load(engine), name="eager-model-load"
            )
        try:
            yield
        finally:
            if runtime.eager_task is not None:
                runtime.eager_task.cancel()
            await jobs.stop()
            await batcher.stop()
            engine.close()

    app = FastAPI(
        title="claude-mem Nemotron service",
        version=__version__,
        lifespan=lifespan,
    )

    def runtime(request: Request) -> Runtime:
        return request.app.state.runtime

    @app.get("/healthz")
    async def health(request: Request) -> dict:
        state = runtime(request)
        return {
            "ok": True,
            "service": "claude-mem-nemotron",
            "version": __version__,
            "uptime_seconds": round(time.time() - state.started_at, 3),
            "model": state.engine.status(),
            "embedding_queue": state.batcher.status(),
            "durable_write_queue": await state.jobs.status(),
            "vector_path": str(state.settings.chroma_dir),
            "privacy": {
                "chroma_anonymized_telemetry": (
                    state.store.client.get_settings().anonymized_telemetry
                ),
            },
        }

    @app.get("/readyz")
    async def ready(request: Request) -> dict:
        state = runtime(request)
        if state.engine.state != "ready":
            raise HTTPException(
                status_code=503,
                detail={
                    "state": state.engine.state,
                    "error": state.engine.error,
                },
            )
        return {"ready": True, "model": state.engine.status()}

    @app.post("/v1/embeddings")
    async def embeddings(payload: EmbeddingsRequest, request: Request) -> dict:
        state = runtime(request)
        texts = [payload.input] if isinstance(payload.input, str) else payload.input
        try:
            vectors = await state.batcher.submit(texts, payload.input_type)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        return {
            "object": "list",
            "model": state.settings.model_id,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/vector/call")
    async def vector_call(payload: VectorCallRequest, request: Request) -> dict:
        state = runtime(request)
        try:
            if payload.tool_name in NemotronVectorStore.WRITE_TOOLS:
                result = await state.jobs.enqueue_and_wait(
                    payload.tool_name, payload.arguments
                )
            else:
                result = await state.store.call(payload.tool_name, payload.arguments)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        return {"result": result}

    return app


app = create_app()
