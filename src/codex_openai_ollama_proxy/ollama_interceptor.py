from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .core.config import load_dotenv_file, normalize_bool, normalize_optional
from .core.security import extract_incoming_api_key, unauthorized_response

INTERCEPTOR_LOGGER_NAME = "codex_openai_ollama_proxy.ollama_interceptor"
DEFAULT_INTERCEPTOR_PORT = 8889
DEFAULT_UPSTREAM_BASE_URL = "http://127.0.0.1:11434"
STREAMING_MEDIA_TYPES = frozenset({"text/event-stream", "application/x-ndjson"})
RESPONSE_EXCLUDED_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
REQUEST_EXCLUDED_HEADERS = {"host", "content-length", "connection", "transfer-encoding"}

_request_id_var: ContextVar[str | None] = ContextVar("interceptor_request_id", default=None)
_endpoint_var: ContextVar[str | None] = ContextVar("interceptor_endpoint", default=None)


def _normalize_debug_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _normalize_debug_value(value.model_dump(by_alias=True, exclude_none=True))

    if isinstance(value, dict):
        return {str(key): _normalize_debug_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_normalize_debug_value(item) for item in value]

    if isinstance(value, bytes):
        try:
            return _normalize_debug_value(value.decode("utf-8"))
        except UnicodeDecodeError:
            return value.hex()

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    if value is None or isinstance(value, bool | int | float):
        return value

    return str(value)


def configure_interceptor_logging(*, debug: bool, project_root: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    debug_logger = logging.getLogger(INTERCEPTOR_LOGGER_NAME)
    for handler in list(debug_logger.handlers):
        handler.close()
        debug_logger.removeHandler(handler)
    debug_logger.propagate = False

    if debug:
        log_path = project_root / "logs" / "interceptor.debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        debug_logger.setLevel(logging.INFO)
        debug_logger.addHandler(handler)


def log_interceptor_event(event: str, **payload: Any) -> None:
    logger = logging.getLogger(INTERCEPTOR_LOGGER_NAME)
    if not logger.handlers:
        return

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": _request_id_var.get(),
        "endpoint": _endpoint_var.get(),
        "event": event,
        **{key: _normalize_debug_value(value) for key, value in payload.items()},
    }
    logger.info(json.dumps(entry, ensure_ascii=False))


def start_interceptor_request(endpoint: str, request_payload: Any) -> tuple[str, str]:
    request_id = uuid4().hex
    _request_id_var.set(request_id)
    _endpoint_var.set(endpoint)
    log_interceptor_event("incoming_request", payload=request_payload)
    return request_id, endpoint


def finish_interceptor_request(tokens: tuple[str, str] | None = None) -> None:  # noqa: ARG001
    _request_id_var.set(None)
    _endpoint_var.set(None)


def split_response_headers(headers: dict[str, str]) -> tuple[str | None, dict[str, str]]:
    media_type: str | None = None
    forwarded_headers: dict[str, str] = {}
    for name, value in headers.items():
        normalized_name = name.lower()
        if normalized_name == "content-type":
            media_type = value
            continue
        if normalized_name in RESPONSE_EXCLUDED_HEADERS:
            continue
        forwarded_headers[name] = value
    return media_type, forwarded_headers


def is_streaming_media_type(media_type: str | None) -> bool:
    if media_type is None:
        return False
    return media_type.split(";", 1)[0].strip().lower() in STREAMING_MEDIA_TYPES


def filter_request_headers(headers: httpx.Headers) -> dict[str, str]:
    forwarded_headers: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in REQUEST_EXCLUDED_HEADERS:
            continue
        forwarded_headers[name] = value
    return forwarded_headers


@dataclass(slots=True)
class InterceptorSettings:
    port: int = DEFAULT_INTERCEPTOR_PORT
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    debug: bool = True
    required_client_api_key: str | None = None
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_sources(
        cls,
        cli_args: Sequence[str] | None = None,
        cwd: Path | None = None,
    ) -> "InterceptorSettings":
        working_dir = cwd or Path.cwd()
        load_dotenv_file(working_dir / ".env")

        parser = argparse.ArgumentParser(prog="ollama-interceptor")
        parser.add_argument("-p", "--port", dest="port")
        parser.add_argument("--upstream", dest="upstream")
        args = parser.parse_args(list(cli_args) if cli_args is not None else None)

        debug_env = os.getenv("OLLAMA_INTERCEPTOR_DEBUG")
        debug = normalize_bool(debug_env) if debug_env is not None else True

        return cls(
            port=int(args.port or os.getenv("OLLAMA_INTERCEPTOR_PORT") or DEFAULT_INTERCEPTOR_PORT),
            upstream_base_url=args.upstream or os.getenv("OLLAMA_INTERCEPTOR_UPSTREAM") or DEFAULT_UPSTREAM_BASE_URL,
            debug=debug,
            required_client_api_key=normalize_optional(os.getenv("OLLAMA_INTERCEPTOR_API_KEY")),
            project_root=working_dir,
        )


def create_interceptor_app(settings: InterceptorSettings | None = None) -> FastAPI:
    resolved_settings = settings or InterceptorSettings.from_sources()
    configure_interceptor_logging(
        debug=resolved_settings.debug,
        project_root=resolved_settings.project_root,
    )

    client = httpx.AsyncClient(follow_redirects=False, timeout=None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.http_client = client
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(
        title="ollama-interceptor",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.http_client = client

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    async def intercept(request: Request, full_path: str = ""):
        required_api_key = resolved_settings.required_client_api_key
        if required_api_key is not None:
            provided = extract_incoming_api_key(request.headers)
            if provided != required_api_key:
                return unauthorized_response("Invalid or missing API key.")

        request_body = await request.body()
        request_payload: Any
        try:
            request_payload = json.loads(request_body) if request_body else None
        except json.JSONDecodeError:
            request_payload = request_body.decode("utf-8", errors="replace") if request_body else None

        debug_tokens = start_interceptor_request(request.url.path, request_payload)

        upstream_url = f"{resolved_settings.upstream_base_url.rstrip('/')}/{full_path}" if full_path else resolved_settings.upstream_base_url.rstrip("/")
        forwarded_headers = filter_request_headers(request.headers)
        log_interceptor_event(
            "upstream_request",
            method=request.method,
            url=upstream_url,
            query=dict(request.query_params),
            headers=forwarded_headers,
            body=request_body,
        )

        upstream_request = client.build_request(
            method=request.method,
            url=upstream_url,
            params=request.query_params,
            headers=forwarded_headers,
            content=request_body,
        )

        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except Exception as exc:  # noqa: BLE001
            log_interceptor_event("upstream_error", error=str(exc))
            finish_interceptor_request(debug_tokens)
            return JSONResponse(
                status_code=502,
                content={"error": f"Interceptor upstream error: {exc}"},
            )

        media_type, response_headers = split_response_headers(dict(upstream_response.headers))

        if is_streaming_media_type(media_type):

            async def stream_passthrough():
                emitted_chunks: list[bytes] = []
                try:
                    async for chunk in upstream_response.aiter_raw():
                        if await request.is_disconnected():
                            break
                        emitted_chunks.append(chunk)
                        yield chunk
                finally:
                    body = b"".join(emitted_chunks)
                    log_interceptor_event(
                        "upstream_response_stream",
                        status_code=upstream_response.status_code,
                        headers=dict(upstream_response.headers),
                        body=body,
                    )
                    log_interceptor_event(
                        "client_response",
                        status_code=upstream_response.status_code,
                        media_type=media_type or "application/octet-stream",
                        body=body,
                    )
                    finish_interceptor_request(debug_tokens)
                    await upstream_response.aclose()

            return StreamingResponse(
                stream_passthrough(),
                status_code=upstream_response.status_code,
                media_type=media_type,
                headers=response_headers,
            )

        body = b""
        try:
            body = await upstream_response.aread()
        finally:
            await upstream_response.aclose()

        log_interceptor_event(
            "upstream_response",
            status_code=upstream_response.status_code,
            headers=dict(upstream_response.headers),
            body=body,
        )
        log_interceptor_event(
            "client_response",
            status_code=upstream_response.status_code,
            media_type=media_type or "application/octet-stream",
            body=body,
        )
        finish_interceptor_request(debug_tokens)

        return Response(
            content=body,
            status_code=upstream_response.status_code,
            media_type=media_type,
            headers=response_headers,
        )

    return app


def main(argv: list[str] | None = None) -> None:
    settings = InterceptorSettings.from_sources(
        cli_args=argv if argv is not None else os.sys.argv[1:]
    )
    app = create_interceptor_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
