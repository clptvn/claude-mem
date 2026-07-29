from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

WriteHandler = Callable[[str, dict], Awaitable[object]]


class DurableWriteQueue:
    """SQLite-backed write queue recovered after service or machine restarts."""

    def __init__(
        self,
        path: Path,
        handler: WriteHandler,
        batch_size: int,
        timeout_seconds: int,
    ):
        self.path = path
        self.handler = handler
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._db_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._initialize()

    def _initialize(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        migration_path = (
            Path(__file__).resolve().parent
            / "migrations"
            / "001_create_vector_jobs.sql"
        )
        self._connection.executescript(migration_path.read_text(encoding="utf-8"))
        # Any interrupted write is safe to retry: claude-mem uses deterministic
        # document IDs and already reconciles duplicate adds with updates.
        self._connection.execute(
            "UPDATE vector_jobs SET status='pending', updated_at=? "
            "WHERE status='processing'",
            (time.time(),),
        )
        self._connection.commit()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="durable-write-queue")
            self._wake.set()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._connection.close()

    async def enqueue_and_wait(self, tool_name: str, payload: dict) -> object:
        job_id = str(uuid.uuid4())
        now = time.time()
        async with self._db_lock:
            self._connection.execute(
                """
                INSERT INTO vector_jobs(
                  id, tool_name, payload_json, status, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
                """,
                (job_id, tool_name, json.dumps(payload), now, now),
            )
            self._connection.commit()
        self._wake.set()

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            async with self._db_lock:
                row = self._connection.execute(
                    "SELECT status, result_json, error FROM vector_jobs WHERE id=?",
                    (job_id,),
                ).fetchone()
            if row is None:
                raise RuntimeError(f"Durable vector job {job_id} disappeared")
            if row["status"] == "completed":
                return json.loads(row["result_json"]) if row["result_json"] else None
            if row["status"] == "failed":
                raise RuntimeError(row["error"] or f"Vector job {job_id} failed")
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"Durable vector job {job_id} exceeded {self.timeout_seconds}s"
        )

    async def _claim(self) -> list[sqlite3.Row]:
        async with self._db_lock:
            rows = self._connection.execute(
                """
                SELECT id, tool_name, payload_json, attempts
                FROM vector_jobs
                WHERE status='pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (self.batch_size,),
            ).fetchall()
            if not rows:
                return []
            now = time.time()
            self._connection.executemany(
                """
                UPDATE vector_jobs
                SET status='processing', attempts=attempts+1, updated_at=?
                WHERE id=? AND status='pending'
                """,
                [(now, row["id"]) for row in rows],
            )
            self._connection.commit()
            return rows

    async def _run(self) -> None:
        while True:
            rows = await self._claim()
            if not rows:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue
            # Chroma mutations are order-sensitive. In particular, an add
            # followed by a delete must never be allowed to finish in reverse
            # order merely because the add spent longer embedding documents.
            # EmbeddingBatcher still coalesces inference across reads and all
            # connected clients; this loop only serializes durable mutations.
            for row in rows:
                await self._process(row)
            await self._cleanup()

    async def _process(self, row: sqlite3.Row) -> None:
        job_id = row["id"]
        try:
            result = await self.handler(
                row["tool_name"], json.loads(row["payload_json"])
            )
            async with self._db_lock:
                self._connection.execute(
                    """
                    UPDATE vector_jobs
                    SET status='completed', result_json=?, error=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (json.dumps(result), time.time(), job_id),
                )
                self._connection.commit()
        except Exception as exc:
            attempts = int(row["attempts"]) + 1
            status = "pending" if attempts < 3 else "failed"
            async with self._db_lock:
                self._connection.execute(
                    """
                    UPDATE vector_jobs
                    SET status=?, error=?, updated_at=?
                    WHERE id=?
                    """,
                    (status, f"{type(exc).__name__}: {exc}", time.time(), job_id),
                )
                self._connection.commit()
            if status == "pending":
                await asyncio.sleep(min(2**attempts, 8))
                self._wake.set()

    async def _cleanup(self) -> None:
        cutoff = time.time() - 7 * 24 * 60 * 60
        async with self._db_lock:
            self._connection.execute(
                "DELETE FROM vector_jobs "
                "WHERE status IN ('completed', 'failed') AND updated_at < ?",
                (cutoff,),
            )
            self._connection.commit()

    async def status(self) -> dict[str, int]:
        async with self._db_lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM vector_jobs GROUP BY status"
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        return {
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
        }
