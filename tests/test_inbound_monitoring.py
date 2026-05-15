from __future__ import annotations

import json
from pathlib import Path

import respx
from fastapi.testclient import TestClient
from httpx import Response

from codex_openai_ollama_proxy.app import create_app
from codex_openai_ollama_proxy.core.config import Settings


def write_auth_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_settings(
    auth_path: Path,
    *,
    required_api_key: str | None = None,
    debug: bool = True,
    project_root: Path | None = None,
) -> Settings:
    return Settings(
        port=8888,
        auth_path=auth_path,
        required_client_api_key=required_api_key,
        debug=debug,
        project_root=project_root or auth_path.parent,
        service_name="codex-openai-ollama-proxy",
        service_version="0.1.0",
        ollama_compat_version="0.22.0",
    )


def read_debug_log(project_root: Path) -> list[dict[str, object]]:
    log_path = project_root / "logs" / "debug.log"
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_inbound_monitor_logs_request_and_keeps_body_readable(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, project_root=tmp_path)
    app = create_app(settings)

    catalog_body = {
        "models": [{"slug": "gpt-5.4", "context_window": 272000}]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(
            return_value=Response(200, json=catalog_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/show?source=vscode",
                headers={
                    "Authorization": "Bearer secret-token",
                    "X-Api-Key": "hidden-key",
                    "User-Agent": "VSCode-Test",
                },
                json={
                    "model": "gpt-5.4",
                    "access_token": "top-secret",
                },
            )

    assert response.status_code == 200
    entries = read_debug_log(tmp_path)
    inbound_request = next(entry for entry in entries if entry["event"] == "inbound_request")
    inbound_response = next(entry for entry in entries if entry["event"] == "inbound_response")

    assert inbound_request["method"] == "POST"
    assert inbound_request["path"] == "/api/show"
    assert inbound_request["query"] == {"source": "vscode"}
    assert inbound_request["headers"]["authorization"] == "[REDACTED]"
    assert inbound_request["headers"]["x-api-key"] == "[REDACTED]"
    assert inbound_request["headers"]["user-agent"] == "VSCode-Test"
    assert inbound_request["body"] == {
        "model": "gpt-5.4",
        "access_token": "[REDACTED]",
    }
    assert inbound_request["body_truncated"] is False

    assert inbound_response["method"] == "POST"
    assert inbound_response["path"] == "/api/show"
    assert inbound_response["status_code"] == 200
    assert "application/json" in str(inbound_response["media_type"])


def test_inbound_monitor_logs_unauthorized_requests_before_route_handler(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(
        auth_path,
        required_api_key="expected-secret",
        project_root=tmp_path,
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get(
            "/v1/models?client=vscode",
            headers={
                "Authorization": "Bearer wrong-secret",
                "Cookie": "session=abc",
            },
        )

    assert response.status_code == 401
    entries = read_debug_log(tmp_path)
    inbound_request = next(entry for entry in entries if entry["event"] == "inbound_request")
    inbound_response = next(entry for entry in entries if entry["event"] == "inbound_response")

    assert inbound_request["method"] == "GET"
    assert inbound_request["path"] == "/v1/models"
    assert inbound_request["query"] == {"client": "vscode"}
    assert inbound_request["headers"]["authorization"] == "[REDACTED]"
    assert inbound_request["headers"]["cookie"] == "[REDACTED]"
    assert inbound_request["body"] is None
    assert inbound_response["status_code"] == 401
