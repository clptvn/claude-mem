from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class ServiceSettings:
    model_id: str
    host: str
    port: int
    data_dir: Path
    chroma_dir: Path
    queue_path: Path
    max_sequence_length: int
    inference_batch_size: int
    queue_batch_size: int
    batch_wait_ms: int
    write_timeout_seconds: int
    eager_load: bool

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        data_dir = Path(
            os.getenv(
                "CLAUDE_MEM_NEMOTRON_DATA_DIR",
                str(Path.home() / ".claude-mem" / "nemotron"),
            )
        ).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir = data_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            model_id=os.getenv(
                "CLAUDE_MEM_NEMOTRON_MODEL",
                "nvidia/Nemotron-3-Embed-1B-BF16",
            ),
            host=os.getenv("CLAUDE_MEM_NEMOTRON_HOST", "127.0.0.1"),
            port=_env_int("CLAUDE_MEM_NEMOTRON_PORT", 37901, 1024, 65535),
            data_dir=data_dir,
            chroma_dir=chroma_dir,
            queue_path=data_dir / "jobs.sqlite3",
            # The checkpoint supports 32,768 tokens. 4,096 is a deliberately
            # conservative default for a 32 GB unified-memory Mac; users can
            # opt into larger records without changing the service.
            max_sequence_length=_env_int(
                "CLAUDE_MEM_NEMOTRON_MAX_TOKENS", 4096, 128, 32768
            ),
            inference_batch_size=_env_int(
                "CLAUDE_MEM_NEMOTRON_BATCH_SIZE", 8, 1, 64
            ),
            queue_batch_size=_env_int(
                "CLAUDE_MEM_NEMOTRON_QUEUE_BATCH_SIZE", 16, 1, 128
            ),
            batch_wait_ms=_env_int(
                "CLAUDE_MEM_NEMOTRON_BATCH_WAIT_MS", 12, 0, 250
            ),
            write_timeout_seconds=_env_int(
                "CLAUDE_MEM_NEMOTRON_WRITE_TIMEOUT_SECONDS", 180, 10, 900
            ),
            eager_load=os.getenv("CLAUDE_MEM_NEMOTRON_EAGER_LOAD", "true").lower()
            not in {"0", "false", "no"},
        )
