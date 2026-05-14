from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.health import router as health_router
from .api.routes.meta import router as meta_router
from .api.routes.models import router as models_router
from .api.routes.openai import router as openai_router
from .api.routes.ollama import router as ollama_router
from .core.config import Settings
from .core.debug_trace import log_debug_event
from .core.logging import configure_logging
from .core.security import is_public_path, unauthorized_response
from .services.auth_store import AuthStore
from .services.backend_client import BackendClient
from .services.model_catalog import ModelCatalogService
from .services.proxy_service import ProxyService

SENSITIVE_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}
SENSITIVE_KEY_PARTS = ("token", "secret", "password", "credential", "auth")
MAX_LOG_BODY_CHARS = 16000


def _should_redact_key(key: str, *, is_header: bool = False) -> bool:
    normalized = key.strip().lower()
    if is_header:
        return normalized in SENSITIVE_HEADER_NAMES

    collapsed = normalized.replace("-", "_")
    if collapsed in SENSITIVE_HEADER_NAMES:
        return True
    if collapsed.endswith("_key"):
        return True
    return any(part in collapsed for part in SENSITIVE_KEY_PARTS)


def _truncate_string(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_LOG_BODY_CHARS:
        return value, False
    return value[:MAX_LOG_BODY_CHARS], True


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _should_redact_key(key_str):
                sanitized[key_str] = "[REDACTED]"
            else:
                sanitized[key_str] = _sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
        truncated, _ = _truncate_string(decoded)
        return truncated
    if isinstance(value, str):
        truncated, _ = _truncate_string(value)
        return truncated
    return value


def _sanitize_headers(headers: Request) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for name, value in headers.headers.items():
        sanitized[name] = "[REDACTED]" if _should_redact_key(name, is_header=True) else value
    return sanitized


def _sanitize_query_params(request: Request) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        sanitized[key] = "[REDACTED]" if _should_redact_key(key) else value
    return sanitized


async def _extract_request_body(request: Request) -> tuple[Any, bool]:
    raw_body = await request.body()
    if not raw_body:
        return None, False

    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    truncated_raw, body_truncated = _truncate_string(
        raw_body.decode("utf-8", errors="replace")
    )

    if content_type.endswith("/json") or content_type.endswith("+json"):
        try:
            parsed = json.loads(truncated_raw)
        except json.JSONDecodeError:
            return truncated_raw, body_truncated
        return _sanitize_value(parsed), body_truncated

    return truncated_raw, body_truncated


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_sources()
    configure_logging(debug=resolved_settings.debug, project_root=resolved_settings.project_root)
    auth_store = AuthStore(resolved_settings.auth_path)
    backend_client = BackendClient(resolved_settings, auth_store)
    model_catalog = ModelCatalogService(resolved_settings, backend_client)
    proxy_service = ProxyService(resolved_settings, backend_client, model_catalog)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.auth_store = auth_store
        app.state.backend_client = backend_client
        app.state.model_catalog = model_catalog
        app.state.proxy_service = proxy_service

        try:
            yield
        finally:
            await backend_client.aclose()

    app = FastAPI(
        title="codex-openai-ollama-proxy",
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.auth_store = auth_store
    app.state.backend_client = backend_client
    app.state.model_catalog = model_catalog
    app.state.proxy_service = proxy_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_private_network=True,
        allow_headers=[
            "authorization",
            "content-type",
            "accept",
            "accept-encoding",
            "x-stainless-arch",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-retry-count",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-stainless-timeout",
            "x-api-key",
            "api-key",
        ],
    )

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):  # type: ignore[override]
        required_api_key = resolved_settings.required_client_api_key
        if (
            required_api_key
            and request.method != "OPTIONS"
            and not is_public_path(request.url.path, resolved_settings)
        ):
            from .core.security import extract_incoming_api_key

            provided_api_key = extract_incoming_api_key(request.headers)
            if provided_api_key != required_api_key:
                return unauthorized_response(
                    "Invalid or missing API key. Set Authorization: Bearer <API_KEY> to access this proxy."
                )

        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @app.middleware("http")
    async def inbound_monitoring_middleware(request: Request, call_next):  # type: ignore[override]
        if not resolved_settings.debug:
            return await call_next(request)

        headers = _sanitize_headers(request)
        query = _sanitize_query_params(request)
        body, body_truncated = await _extract_request_body(request)
        started_at = time.perf_counter()
        log_debug_event(
            "inbound_request",
            method=request.method,
            path=request.url.path,
            query=query,
            headers=headers,
            body=body,
            body_truncated=body_truncated,
        )

        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log_debug_event(
            "inbound_response",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            duration_ms=duration_ms,
        )
        return response

    @app.options("/{full_path:path}", include_in_schema=False)
    async def preflight_handler(full_path: str) -> Response:  # noqa: ARG001
        return Response(status_code=204)

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(meta_router)
    app.include_router(ollama_router)
    app.include_router(openai_router)

    return app
