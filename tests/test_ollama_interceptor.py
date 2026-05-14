from __future__ import annotations

import json
from pathlib import Path

import respx
from fastapi.testclient import TestClient
from httpx import Response

from codex_openai_ollama_proxy.ollama_interceptor import (
    InterceptorSettings,
    create_interceptor_app,
)


def build_settings(tmp_path: Path) -> InterceptorSettings:
    return InterceptorSettings(
        port=8889,
        upstream_base_url="http://127.0.0.1:11434",
        debug=True,
        required_client_api_key=None,
        project_root=tmp_path,
    )


def read_debug_log(project_root: Path) -> list[dict[str, object]]:
    log_path = project_root / "logs" / "interceptor.debug.log"
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_interceptor_forwards_json_requests_and_logs_bodies(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    app = create_interceptor_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post("http://127.0.0.1:11434/v1/chat/completions").mock(
            return_value=Response(
                200,
                text='{"ok":true}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions?source=test",
                headers={"X-Test": "1"},
                json={"model": "qwen3.5:latest", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert dict(route.calls.last.request.url.params) == {"source": "test"}
    assert json.loads(route.calls.last.request.content.decode("utf-8")) == {
        "model": "qwen3.5:latest",
        "messages": [{"role": "user", "content": "hi"}],
    }

    entries = read_debug_log(tmp_path)
    assert {entry["event"] for entry in entries} >= {
        "incoming_request",
        "upstream_request",
        "upstream_response",
        "client_response",
    }


def test_interceptor_streams_sse_passthrough_verbatim(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    app = create_interceptor_app(settings)

    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post("http://127.0.0.1:11434/v1/chat/completions").mock(
            return_value=Response(
                200,
                text=sse_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "qwen3.5:latest", "stream": True},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == sse_body

    entries = read_debug_log(tmp_path)
    assert {entry["event"] for entry in entries} >= {
        "incoming_request",
        "upstream_request",
        "upstream_response_stream",
        "client_response",
    }


def test_interceptor_streams_ndjson_passthrough_verbatim(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    app = create_interceptor_app(settings)

    ndjson_body = '{"done":false,"message":{"content":"Hello"}}\n{"done":true}\n'

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=Response(
                200,
                text=ndjson_body,
                headers={"content-type": "application/x-ndjson"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={"model": "qwen3.5:latest", "stream": True},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.text == ndjson_body
