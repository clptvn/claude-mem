from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from nemotron_memory_service.durable_queue import DurableWriteQueue


SCHEMA = """
CREATE TABLE vector_jobs (
  id TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  result_json TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
)
"""


class DurableWriteQueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_recovers_an_interrupted_write_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(SCHEMA)
            connection.execute(
                """
                INSERT INTO vector_jobs(
                  id, tool_name, payload_json, status, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, 'processing', 1, ?, ?)
                """,
                (
                    "interrupted-job",
                    "chroma_add_documents",
                    json.dumps({"ids": ["obs_1"]}),
                    time.time(),
                    time.time(),
                ),
            )
            connection.commit()
            connection.close()

            processed: list[tuple[str, dict]] = []

            async def handler(tool_name: str, payload: dict) -> object:
                processed.append((tool_name, payload))
                return {"written": len(payload["ids"])}

            queue = DurableWriteQueue(path, handler, batch_size=8, timeout_seconds=5)
            queue.start()
            try:
                for _ in range(100):
                    status = await queue.status()
                    if status["completed"] == 1:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(status["completed"], 1)
                self.assertEqual(
                    processed,
                    [("chroma_add_documents", {"ids": ["obs_1"]})],
                )
            finally:
                await queue.stop()

    async def test_preserves_fifo_order_for_vector_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.sqlite3"
            completed: list[str] = []

            async def handler(tool_name: str, _payload: dict) -> object:
                if tool_name == "chroma_add_documents":
                    await asyncio.sleep(0.05)
                completed.append(tool_name)
                return {"ok": True}

            queue = DurableWriteQueue(path, handler, batch_size=8, timeout_seconds=5)
            try:
                add = asyncio.create_task(
                    queue.enqueue_and_wait("chroma_add_documents", {"ids": ["obs_1"]})
                )
                delete = asyncio.create_task(
                    queue.enqueue_and_wait(
                        "chroma_delete_documents", {"ids": ["obs_1"]}
                    )
                )
                # Let both callers durably enqueue before the consumer claims
                # the batch; this is the ordering case that concurrent
                # processing used to invert.
                await asyncio.sleep(0.01)
                queue.start()
                await asyncio.gather(add, delete)
                self.assertEqual(
                    completed,
                    ["chroma_add_documents", "chroma_delete_documents"],
                )
            finally:
                await queue.stop()


if __name__ == "__main__":
    unittest.main()
