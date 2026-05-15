from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi import Response
from fastapi.responses import FileResponse, PlainTextResponse

from codex_openai_ollama_proxy.api.deps import get_settings
from codex_openai_ollama_proxy.core.config import Settings

router = APIRouter(tags=["meta"])


@router.get("/", include_in_schema=False)
def ollama_root() -> PlainTextResponse:
    return PlainTextResponse("Ollama is running")


@router.head("/", include_in_schema=False)
def ollama_root_head() -> Response:
    return Response(status_code=200, media_type="text/plain")


@router.api_route("/api/version", methods=["GET", "HEAD"])
def api_version(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"version": settings.ollama_compat_version}


@router.get("/chat-test", include_in_schema=False)
@router.get("/chat-test.html", include_in_schema=False)
def chat_test(settings: Settings = Depends(get_settings)) -> FileResponse:
    return FileResponse(settings.project_root / "tests" / "chat-test.html")
