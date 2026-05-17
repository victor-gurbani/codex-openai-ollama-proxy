from __future__ import annotations

import json
from pathlib import Path

import respx
from fastapi.testclient import TestClient
from httpx import Response

from codex_openai_ollama_proxy.api.routes.ollama import (
    SYNTHETIC_MODEL_MODIFIED_AT,
    SYNTHETIC_MODEL_SIZE_BYTES,
    ollama_model_details,
)
from codex_openai_ollama_proxy.app import create_app
from codex_openai_ollama_proxy.core.config import Settings


def write_auth_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_settings(auth_path: Path) -> Settings:
    return Settings(
        port=8888,
        auth_path=auth_path,
        required_client_api_key=None,
        service_name="codex-openai-ollama-proxy",
        service_version="0.1.0",
        ollama_compat_version="0.22.0",
    )


def backend_sse_body() -> str:
    return (
        'data: {"type":"response.output_text.delta","delta":"Hello from ollama"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )


def backend_tool_call_body() -> str:
    return (
        'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"list_files","arguments":"{\\"path\\":\\".\\",\\"recursive\\":false}"}}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )


def test_api_version_returns_ollama_compat_semver(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": "0.22.0"}


def test_ollama_root_get_matches_native_liveness_probe(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text == "Ollama is running"
    assert response.headers["content-type"].startswith("text/plain")


def test_ollama_root_head_succeeds_for_cli_probe(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.head("/")

    assert response.status_code == 200
    assert response.text == ""


def test_api_version_head_succeeds(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.head("/api/version")

    assert response.status_code == 200
    assert response.text == ""


def test_ollama_show_known_model_returns_metadata(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/show", json={"model": "gpt-5.4"})

    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload.keys()) == [
        "capabilities",
        "details",
        "license",
        "model_info",
        "modelfile",
        "modified_at",
        "parameters",
        "requires",
        "template",
        "tensors",
    ]
    assert payload["details"]["format"] == "proxy"
    assert payload["modelfile"] == "FROM gpt-5.4\n"
    assert payload["model_info"]["general.basename"] == "gpt-5.4"
    assert payload["model_info"]["general.file_type"] == 0
    assert "general.context_length" not in payload["model_info"]
    assert payload["capabilities"] == ["completion", "tools", "thinking"]
    assert payload["modified_at"] == SYNTHETIC_MODEL_MODIFIED_AT
    assert payload["requires"] == "0.17.1"
    assert payload["tensors"] == []


def test_ollama_show_uses_context_window_from_backend_catalog(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "gpt-5.4",
                "context_window": 272000,
                "max_context_window": 1000000,
            },
            {"slug": "gpt-5.3-codex", "display_name": "gpt-5.3-codex", "context_window": 272000},
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(return_value=Response(200, json=catalog_body))
        with TestClient(app) as client:
            response = client.post("/api/show", json={"model": "gpt-5.4"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_info"]["general.context_length"] == 1000000
    assert payload["model_info"]["gpt.context_length"] == 1000000
    assert payload["parameters"] == "num_ctx 1000000"


def test_ollama_show_adds_vision_and_namespaced_codex_metadata(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "GPT-5.4",
                "description": "General-purpose Codex-backed model",
                "context_window": 272000,
                "max_context_window": 1000000,
                "input_modalities": ["text", "image"],
                "supported_reasoning_levels": ["low", "medium", "high", "xhigh"],
                "default_reasoning_level": "medium",
                "default_reasoning_summary": "auto",
                "default_verbosity": "medium",
                "support_verbosity": True,
                "additional_speed_tiers": ["fast"],
                "service_tiers": ["default", "flex"],
                "truncation_policy": "disabled",
            }
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(return_value=Response(200, json=catalog_body))
        with TestClient(app) as client:
            response = client.post("/api/show", json={"model": "gpt-5.4"})

    assert response.status_code == 200
    payload = response.json()

    assert sorted(payload.keys()) == [
        "capabilities",
        "details",
        "license",
        "model_info",
        "modelfile",
        "modified_at",
        "parameters",
        "requires",
        "template",
        "tensors",
    ]
    assert payload["capabilities"] == ["completion", "tools", "thinking", "vision"]
    assert payload["model_info"]["general.context_length"] == 1000000
    assert payload["model_info"]["gpt.context_length"] == 1000000
    assert payload["model_info"]["codex.display_name"] == "GPT-5.4"
    assert payload["model_info"]["codex.description"] == "General-purpose Codex-backed model"
    assert payload["model_info"]["codex.reasoning.supported_levels"] == [
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert payload["model_info"]["codex.reasoning.default_level"] == "medium"
    assert payload["model_info"]["codex.reasoning.default_summary"] == "auto"
    assert payload["model_info"]["codex.verbosity.default"] == "medium"
    assert payload["model_info"]["codex.verbosity.supported"] is True
    assert payload["model_info"]["codex.additional_speed_tiers"] == ["fast"]
    assert payload["model_info"]["codex.fast_tier_available"] is True
    assert payload["model_info"]["codex.service_tiers"] == ["default", "flex"]
    assert payload["model_info"]["codex.truncation_policy"] == "disabled"
    assert payload["parameters"] == "num_ctx 1000000"
    assert "display_name" not in payload
    assert "description" not in payload
    assert "service_tiers" not in payload
    assert "default_reasoning_level" not in payload


def test_ollama_show_preserves_existing_capabilities_without_vision(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "GPT-5.4",
                "input_modalities": ["text"],
            }
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(return_value=Response(200, json=catalog_body))
        with TestClient(app) as client:
            response = client.post("/api/show", json={"model": "gpt-5.4"})

    assert response.status_code == 200
    assert response.json()["capabilities"] == ["completion", "tools", "thinking"]


def test_ollama_show_adds_vision_from_support_flag_without_new_top_level_fields(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "GPT-5.4",
                "supports_image": True,
            }
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(return_value=Response(200, json=catalog_body))
        with TestClient(app) as client:
            response = client.post("/api/show", json={"model": "gpt-5.4"})

    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload.keys()) == [
        "capabilities",
        "details",
        "license",
        "model_info",
        "modelfile",
        "modified_at",
        "parameters",
        "requires",
        "template",
        "tensors",
    ]
    assert payload["capabilities"] == ["completion", "tools", "thinking", "vision"]


def test_ollama_show_normalizes_latest_suffix(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/show", json={"model": "gpt-5.4:latest"})

    assert response.status_code == 200
    assert response.json()["model_info"]["general.basename"] == "gpt-5.4"


def test_ollama_show_unknown_model_returns_404(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/show", json={"model": "missing-model:latest"})

    assert response.status_code == 404
    assert response.json() == {"error": "model 'missing-model' not found"}


def test_ollama_show_accepts_legacy_name_field(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/show", json={"name": "gpt-5.4"})

    assert response.status_code == 200
    assert response.json()["model_info"]["general.basename"] == "gpt-5.4"


def test_ollama_show_prefers_legacy_name_when_model_is_blank(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/show", json={"model": "", "name": "gpt-5.4"})

    assert response.status_code == 200
    assert response.json()["model_info"]["general.basename"] == "gpt-5.4"


def test_ollama_pull_known_model_returns_streaming_success(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/pull", json={"model": "gpt-5.4"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.text == '{"status":"success"}\n'


def test_ollama_pull_known_model_can_return_non_streaming_success(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/pull",
            json={"name": "gpt-5.4:latest", "stream": False},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_ollama_pull_prefers_legacy_name_when_model_is_blank(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/pull",
            json={"model": "", "name": "gpt-5.4", "stream": False},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_ollama_pull_unknown_model_returns_404(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/pull", json={"model": "missing-model"})

    assert response.status_code == 404
    assert response.json() == {"error": "model 'missing-model' not found"}


def test_ollama_ps_returns_fallback_models_with_synthetic_runtime_fields(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/ps")

    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload.keys()) == ["models"]
    assert len(payload["models"]) == 10
    first_model = payload["models"][0]
    assert sorted(first_model.keys()) == [
        "details",
        "digest",
        "expires_at",
        "model",
        "name",
        "size",
        "size_vram",
    ]
    assert first_model["model"] == first_model["name"]
    assert first_model["details"]["format"] == "proxy"
    assert first_model["size"] == SYNTHETIC_MODEL_SIZE_BYTES
    assert isinstance(first_model["digest"], str)
    assert len(first_model["digest"]) == 64
    assert first_model["size_vram"] == 0
    assert first_model["expires_at"].endswith("Z")


def test_ollama_ps_uses_context_window_from_backend_catalog(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "gpt-5.4",
                "context_window": 272000,
                "max_context_window": 1000000,
                "size_vram": 8192,
                "expires_at": "2099-01-01T00:00:00Z",
            },
            {
                "slug": "gpt-5.3-codex",
                "display_name": "gpt-5.3-codex",
                "context_window": 272000,
            },
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(
            return_value=Response(200, json=catalog_body)
        )
        with TestClient(app) as client:
            response = client.get("/api/ps")

    assert response.status_code == 200
    payload = response.json()
    models = {item["model"]: item for item in payload["models"]}
    assert models["gpt-5.4"]["context_length"] == 1000000
    assert models["gpt-5.4"]["size_vram"] == 8192
    assert models["gpt-5.4"]["expires_at"].endswith("Z")
    assert models["gpt-5.4-high"]["context_length"] == 1000000


def test_ollama_model_details_normalizes_family_like_real_ollama() -> None:
    details = ollama_model_details("qwen3.5:latest")

    assert details["family"] == "qwen35"
    assert details["families"] == ["qwen35"]


def test_api_tags_returns_ollama_cli_safe_model_metadata(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/tags")

    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload.keys()) == ["models"]
    first_model = payload["models"][0]
    assert sorted(first_model.keys()) == [
        "capabilities",
        "details",
        "digest",
        "model",
        "modified_at",
        "name",
        "size",
    ]
    assert first_model["name"] == first_model["model"]
    assert first_model["modified_at"] == SYNTHETIC_MODEL_MODIFIED_AT
    assert first_model["size"] == SYNTHETIC_MODEL_SIZE_BYTES
    assert isinstance(first_model["digest"], str)
    assert len(first_model["digest"]) == 64
    assert first_model["details"]["format"] == "proxy"
    assert first_model["capabilities"] == ["completion", "tools", "thinking"]


def test_api_tags_advertises_vision_when_backend_metadata_supports_images(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "GPT-5.4",
                "input_modalities": ["text", "image"],
            }
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(
            return_value=Response(200, json=catalog_body)
        )
        with TestClient(app) as client:
            response = client.get("/api/tags")

    assert response.status_code == 200
    payload = response.json()
    models = {item["model"]: item for item in payload["models"]}
    assert models["gpt-5.4"]["capabilities"] == [
        "completion",
        "tools",
        "thinking",
        "vision",
    ]
    assert models["gpt-5.4-low"]["capabilities"] == [
        "completion",
        "tools",
        "thinking",
        "vision",
    ]


def test_ollama_show_uses_codex_family_key_for_codex_models(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    catalog_body = {
        "models": [
            {
                "slug": "gpt-5.3-codex",
                "display_name": "gpt-5.3-codex",
                "context_window": 272000,
            }
        ]
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(
            return_value=Response(200, json=catalog_body)
        )
        with TestClient(app) as client:
            response = client.post("/api/show", json={"model": "gpt-5.3-codex"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["details"]["family"] == "codex"
    assert payload["model_info"]["general.context_length"] == 272000
    assert payload["model_info"]["codex.context_length"] == 272000
    assert payload["parameters"] == "num_ctx 272000"


def test_ollama_chat_route(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": "gpt-5.4:latest",
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "gpt-5.4"
    assert payload["message"]["content"] == "Hello from ollama"


def test_ollama_chat_route_forwards_tools_and_returns_tool_calls(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        }
    ]

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_tool_call_body())
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": "gpt-5.4",
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": request_tools,
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))

    assert backend_payload["tools"][0]["type"] == "function"
    assert backend_payload["tools"][0]["name"] == "list_files"
    assert payload["message"]["content"] == ""
    assert payload["message"]["tool_calls"][0]["function"]["name"] == "list_files"
    assert payload["message"]["tool_calls"][0]["function"]["arguments"] == {
        "path": ".",
        "recursive": False,
    }
    assert "type" not in payload["message"]["tool_calls"][0]
    assert payload["message"]["tool_calls"][0]["id"] == "call_1"
    assert payload["message"]["tool_calls"][0]["function"]["index"] == 0


def test_ollama_chat_route_normalizes_raycast_tool_types(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_tools = [
        {"name": "web_search", "type": "remote_tool"},
        {"name": "search_images", "type": "remote_tool"},
        {
            "function": {
                "description": "Gets the user's current location.",
                "name": "location-get-current-location",
                "parameters": {},
            },
            "type": "local_tool",
        },
    ]

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "gpt-5.4",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": request_tools,
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["tools"] == [
        {"type": "function", "name": "web_search"},
        {"type": "function", "name": "search_images"},
        {
            "type": "function",
            "name": "location-get-current-location",
            "description": "Gets the user's current location.",
            "parameters": {},
        },
    ]


def test_ollama_chat_route_drops_unanswered_replayed_tool_call(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "gpt-5.4",
                    "stream": False,
                    "messages": [
                        {"role": "user", "content": "What's the latest inflation rate?"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "web_search", "arguments": {}}}
                            ],
                        },
                    ],
                    "tools": [{"name": "web_search", "type": "remote_tool"}],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["input"] == [
        {
            "type": "message",
            "id": None,
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What's the latest inflation rate?"}
            ],
        }
    ]
    assert backend_payload["tools"] == [{"type": "function", "name": "web_search"}]


def test_ollama_chat_think_false_maps_to_none_reasoning(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                        "think": False,
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["reasoning"] == {"effort": "none"}


def test_ollama_generate_think_xhigh_maps_to_backend_reasoning(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={
                        "model": "gpt-5.3-codex",
                        "prompt": "hello",
                        "think": "xhigh",
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["reasoning"] == {"summary": "auto", "effort": "xhigh"}


def test_ollama_chat_think_true_maps_to_medium_reasoning(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                        "think": True,
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["reasoning"] == {"summary": "auto", "effort": "medium"}


def test_ollama_chat_without_think_requests_reasoning_by_default(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["reasoning"] == {"summary": "auto"}


def test_ollama_generate_without_think_requests_reasoning_by_default(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": "gpt-5.4",
                    "prompt": "hello",
                    "stream": False,
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["reasoning"] == {"summary": "auto"}


def test_ollama_chat_non_streaming_returns_thinking(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.done","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"Think first."}]}}\n\n'
        'data: {"type":"response.output_text.delta","delta":"Hello from ollama"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": "gpt-5.4:latest",
                        "messages": [{"role": "user", "content": "hello"}],
                        "think": True,
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]["thinking"] == "Think first."
    assert payload["message"]["content"] == "Hello from ollama"


def test_ollama_generate_non_streaming_returns_thinking(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.done","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"Think first."}]}}\n\n'
        'data: {"type":"response.output_text.delta","delta":"Hello from ollama"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={
                        "model": "gpt-5.4:latest",
                        "prompt": "hello",
                        "think": True,
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["thinking"] == "Think first."
    assert payload["response"] == "Hello from ollama"


def test_ollama_chat_non_streaming_returns_thinking_and_tool_calls(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.done","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"Think first."}]}}\n\n'
        'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"list_files","arguments":"{\\"path\\":\\".\\",\\"recursive\\":false}"}}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "description": "List files",
                                "parameters": {"type": "object"},
                            },
                        }
                        ],
                        "think": True,
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]["thinking"] == "Think first."
    assert payload["message"]["tool_calls"][0]["function"]["name"] == "list_files"


def test_ollama_generate_route_streaming(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": "gpt-5.4:latest",
                    "prompt": "hello",
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"done":true' in response.text


def test_ollama_chat_streams_by_default_when_stream_is_omitted(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "gpt-5.4:latest",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"done":true' in response.text


def test_ollama_generate_streams_by_default_when_stream_is_omitted(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": "gpt-5.4:latest",
                    "prompt": "hello",
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"done":true' in response.text


def test_ollama_generate_empty_non_streaming_loads_model(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/generate",
            json={"model": "gpt-5.4:latest", "stream": False},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-5.4"
    assert response.json()["response"] == ""
    assert response.json()["done"] is True


def test_ollama_chat_empty_streaming_loads_model(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"model": "gpt-5.4:latest", "messages": [], "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"model":"gpt-5.4"' in response.text
    assert '"message":{"role":"assistant","content":""}' in response.text
    assert '"done":true' in response.text


def test_ollama_chat_unknown_model_returns_native_not_found_error(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": "missing-model:latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "model 'missing-model' not found"}


def test_ollama_generate_unknown_model_returns_native_not_found_error(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/generate",
            json={
                "model": "missing-model:latest",
                "prompt": "hello",
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "model 'missing-model' not found"}


def test_ollama_generate_translates_legacy_images_field_into_backend_input_images(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": "gpt-5.4",
                    "prompt": "describe this",
                    "images": ["QUJD"],
                    "stream": False,
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["input"][0]["content"] == [
        {"type": "input_text", "text": "describe this"},
        {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
    ]


def test_ollama_chat_translates_message_images_field_into_backend_input_images(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "gpt-5.4",
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": "describe this",
                            "images": ["QUJD"],
                        }
                    ],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["input"][0]["content"] == [
        {"type": "input_text", "text": "describe this"},
        {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
    ]


def test_ollama_chat_json_mode_maps_to_backend_text_format(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "gpt-5.4",
                    "stream": False,
                    "format": "json",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["text"] == {"format": {"type": "json_object"}}


def test_ollama_generate_schema_format_maps_to_backend_json_schema(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": "gpt-5.4",
                    "prompt": "hello",
                    "stream": False,
                    "format": schema,
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "response",
            "schema": schema,
            "strict": True,
        }
    }


def test_ollama_generate_empty_streaming_loads_model(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/generate",
            json={"model": "gpt-5.4:latest", "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"model":"gpt-5.4"' in response.text
    assert '"response":""' in response.text
    assert '"done":true' in response.text


def test_ollama_invalid_think_value_rejected(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hello"}],
                "think": "invalid",
            },
        )

    assert response.status_code == 422
    assert "think must be one of" in response.text


def test_ollama_non_streaming_backend_failed_event_surfaces_backend_message(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress","output":[]}}\n\n'
        'event: error\n'
        'data: {"type":"error","message":"Your input exceeds the context window of this model.","code":"context_length_exceeded","type":"invalid_request_error","param":"messages"}\n\n'
        'event: response.failed\n'
        'data: {"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model."}}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={
                        "model": "gpt-5.4:latest",
                        "prompt": "hello",
                        "stream": False,
                    },
                )

    assert response.status_code == 400
    assert response.json() == {"error": "Your input exceeds the context window of this model."}


def test_ollama_streaming_backend_failed_event_surfaces_backend_message(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress","output":[]}}\n\n'
        'event: error\n'
        'data: {"type":"error","message":"Your input exceeds the context window of this model.","code":"context_length_exceeded","type":"invalid_request_error","param":"messages"}\n\n'
        'event: response.failed\n'
        'data: {"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model."}}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/generate",
                json={
                    "model": "gpt-5.4:latest",
                    "prompt": "hello",
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert '{"error":"Your input exceeds the context window of this model."}' in response.text
