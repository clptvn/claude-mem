from __future__ import annotations

import asyncio
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .settings import ServiceSettings

InputType = Literal["query", "passage"]


@dataclass
class _EmbeddingRequest:
    texts: list[str]
    input_type: InputType
    future: asyncio.Future[list[list[float]]]


class ModelEngine:
    """Owns the sole in-process model instance and its Metal execution thread."""

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.model: SentenceTransformer | None = None
        self.state = "not_loaded"
        self.error: str | None = None
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None
        self.device = self._select_device()
        self.dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nemotron-metal"
        )
        self._load_lock = asyncio.Lock()

    @staticmethod
    def _select_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    async def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        async with self._load_lock:
            if self.model is not None:
                return
            self.state = "loading"
            self.error = None
            started = time.monotonic()
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(self._executor, self._load_sync)
            except Exception as exc:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
                raise
            self.load_seconds = time.monotonic() - started
            self.loaded_at = time.time()
            self.state = "ready"

    def _load_sync(self) -> None:
        model = SentenceTransformer(
            self.settings.model_id,
            device=self.device,
            model_kwargs={
                "dtype": self.dtype,
                "attn_implementation": "sdpa",
            },
        )
        model.max_seq_length = self.settings.max_sequence_length
        self.model = model

    async def encode(
        self, texts: list[str], input_type: InputType
    ) -> list[list[float]]:
        if not texts:
            return []
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._encode_sync, texts, input_type
        )

    def _encode_sync(
        self, texts: list[str], input_type: InputType
    ) -> list[list[float]]:
        assert self.model is not None
        encode = (
            self.model.encode_query
            if input_type == "query"
            else self.model.encode_document
        )
        embeddings = encode(
            texts,
            batch_size=self.settings.inference_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Chroma expects ordinary float vectors. Converting to float32 also
        # avoids serializing numpy float16 values through FastAPI.
        return np.asarray(embeddings, dtype=np.float32).tolist()

    def status(self) -> dict[str, object]:
        return {
            "state": self.state,
            "model": self.settings.model_id,
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "max_sequence_length": self.settings.max_sequence_length,
            "embedding_dimensions": 2048,
            "loaded_at": self.loaded_at,
            "load_seconds": self.load_seconds,
            "error": self.error,
            "platform": platform.platform(),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class EmbeddingBatcher:
    """Batches query and passage requests arriving from every local client."""

    def __init__(self, engine: ModelEngine, settings: ServiceSettings):
        self.engine = engine
        self.settings = settings
        self.queue: asyncio.Queue[_EmbeddingRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.requests_processed = 0
        self.texts_processed = 0
        self.batches_processed = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="embedding-batcher")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def submit(
        self, texts: list[str], input_type: InputType
    ) -> list[list[float]]:
        clean = [text if isinstance(text, str) else str(text) for text in texts]
        if not clean:
            return []
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[list[float]]] = loop.create_future()
        await self.queue.put(_EmbeddingRequest(clean, input_type, future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            requests = [first]
            if self.settings.batch_wait_ms:
                await asyncio.sleep(self.settings.batch_wait_ms / 1000)
            while len(requests) < self.settings.queue_batch_size:
                try:
                    requests.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            for input_type in ("query", "passage"):
                typed = [
                    request
                    for request in requests
                    if request.input_type == input_type
                ]
                if not typed:
                    continue
                flat = [text for request in typed for text in request.texts]
                try:
                    vectors = await self.engine.encode(flat, input_type)
                    cursor = 0
                    for request in typed:
                        end = cursor + len(request.texts)
                        if not request.future.done():
                            request.future.set_result(vectors[cursor:end])
                        cursor = end
                    self.requests_processed += len(typed)
                    self.texts_processed += len(flat)
                    self.batches_processed += 1
                except Exception as exc:
                    for request in typed:
                        if not request.future.done():
                            request.future.set_exception(exc)

            for _ in requests:
                self.queue.task_done()

    def status(self) -> dict[str, int]:
        return {
            "pending_requests": self.queue.qsize(),
            "requests_processed": self.requests_processed,
            "texts_processed": self.texts_processed,
            "batches_processed": self.batches_processed,
        }
