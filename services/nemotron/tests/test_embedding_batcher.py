from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from nemotron_memory_service.model import EmbeddingBatcher
from nemotron_memory_service.settings import ServiceSettings


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    async def encode(self, texts: list[str], input_type: str) -> list[list[float]]:
        self.calls.append((texts, input_type))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class EmbeddingBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_batches_requests_from_concurrent_clients_by_input_type(self) -> None:
        settings = replace(
            ServiceSettings.from_env(),
            batch_wait_ms=20,
            queue_batch_size=8,
        )
        engine = FakeEngine()
        batcher = EmbeddingBatcher(engine, settings)  # type: ignore[arg-type]
        batcher.start()
        try:
            first, second = await asyncio.gather(
                batcher.submit(["one"], "passage"),
                batcher.submit(["two", "three"], "passage"),
            )
            self.assertEqual(len(engine.calls), 1)
            self.assertEqual(engine.calls[0], (["one", "two", "three"], "passage"))
            self.assertEqual(first, [[0.0, 1.0]])
            self.assertEqual(second, [[1.0, 1.0], [2.0, 1.0]])
        finally:
            await batcher.stop()


if __name__ == "__main__":
    unittest.main()
