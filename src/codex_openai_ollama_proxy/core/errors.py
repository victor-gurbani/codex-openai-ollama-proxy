from __future__ import annotations

from fastapi.responses import JSONResponse


class ProxyError(Exception):
    """Base exception for proxy failures."""


class BackendHTTPError(ProxyError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"ChatGPT backend error {status_code}: {body}")


class AuthenticationRefreshError(ProxyError):
    """Raised when refresh_token based renewal fails."""


class EmptyBackendResponseError(ProxyError):
    """Raised when backend emits no text and no tool calls."""


class BackendSSEError(ProxyError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        error_type: str | None = None,
        param: str | None = None,
        code: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        super().__init__(message)


def status_code_for_error(exc: Exception) -> int:
    if isinstance(exc, BackendSSEError):
        return exc.status_code
    if isinstance(exc, BackendHTTPError):
        return exc.status_code
    if isinstance(exc, ValueError):
        return 400
    if isinstance(exc, AuthenticationRefreshError):
        return 502
    if isinstance(exc, EmptyBackendResponseError):
        return 502
    if isinstance(exc, ProxyError):
        return 500
    return 500


def openai_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, BackendSSEError):
        return JSONResponse(
            status_code=exc.status_code,
            headers={
                "Access-Control-Allow-Origin": "*",
            },
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type or "invalid_request_error",
                    "param": exc.param,
                    "code": exc.code,
                }
            },
        )
    return JSONResponse(
        status_code=status_code_for_error(exc),
        headers={
            "Access-Control-Allow-Origin": "*",
        },
        content={
            "error": {
                "message": f"Proxy error: {exc}",
                "type": "proxy_error",
                "code": "internal_error",
            }
        },
    )


def ollama_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, BackendSSEError):
        return JSONResponse(
            status_code=exc.status_code,
            headers={
                "Access-Control-Allow-Origin": "*",
            },
            content={"error": exc.message},
        )
    return JSONResponse(
        status_code=status_code_for_error(exc),
        headers={
            "Access-Control-Allow-Origin": "*",
        },
        content={"error": f"proxy error: {exc}"},
    )
