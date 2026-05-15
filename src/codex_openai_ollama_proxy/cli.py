from __future__ import annotations

import sys

import uvicorn

from .app import create_app
from .core.config import Settings


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_sources(cli_args=argv if argv is not None else sys.argv[1:])
    print(f"DEBUG={'on' if settings.debug else 'off'}")
    print(
        "DISABLE_COPILOT_ADAPTATIONS="
        f"{'on' if settings.disable_copilot_adaptations else 'off'}"
    )
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
