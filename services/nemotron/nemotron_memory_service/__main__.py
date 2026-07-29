from __future__ import annotations

import uvicorn

from .settings import ServiceSettings


def main() -> None:
    settings = ServiceSettings.from_env()
    uvicorn.run(
        "nemotron_memory_service.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
