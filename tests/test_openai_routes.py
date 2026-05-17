from __future__ import annotations

import json
from pathlib import Path

import respx
from fastapi.testclient import TestClient
from httpx import Response

from codex_openai_ollama_proxy.app import create_app
from codex_openai_ollama_proxy.core.config import DEFAULT_SYSTEM_INSTRUCTIONS, Settings


def write_auth_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_settings(
    auth_path: Path,
    *,
    debug: bool = False,
    disable_copilot_adaptations: bool = True,
    add_default_responses_instructions: bool = True,
    project_root: Path | None = None,
) -> Settings:
    return Settings(
        port=8888,
        auth_path=auth_path,
        required_client_api_key=None,
        debug=debug,
        project_root=project_root or Path.cwd(),
        service_name="codex-openai-ollama-proxy",
        service_version="0.1.0",
        disable_copilot_adaptations=disable_copilot_adaptations,
        add_default_responses_instructions=add_default_responses_instructions,
    )


def backend_sse_body() -> str:
    return (
        'data: {"type":"response.output_text.delta","delta":"Hello world"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}\n\n'
        "data: [DONE]\n\n"
    )


def test_openai_chat_completions_route(tmp_path: Path) -> None:
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "gpt-5.4"
    assert payload["system_fingerprint"] == "fp_ollama"
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert payload["usage"]["total_tokens"] == 7
    assert route.calls.last.request.headers["authorization"] == "Bearer backend_key"


def test_openai_streaming_route_returns_sse(tmp_path: Path) -> None:
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text


def test_openai_responses_passthrough_route_preserves_json_body_and_headers(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_body = '{"model":"gpt-5.4","input":"hello","reasoning":{"effort":"high"}}'
    response_body = '{"id":"resp_123","object":"response","status":"completed"}'

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=response_body,
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "x-request-id": "req_123",
                },
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "OpenAI-Beta": "responses=v1",
                    "X-Stainless-Arch": "x64",
                    "User-Agent": "OpenAI/Python 1.0",
                },
            )

    assert response.status_code == 200
    assert response.text == response_body
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"] == "req_123"
    assert json.loads(route.calls.last.request.content.decode("utf-8")) == {
        "model": "gpt-5.4",
        "input": "hello",
        "reasoning": {"effort": "high"},
        "instructions": DEFAULT_SYSTEM_INSTRUCTIONS,
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer backend_key"
    assert route.calls.last.request.headers["accept"] == "application/json"
    assert route.calls.last.request.headers["openai-beta"] == "responses=v1"
    assert route.calls.last.request.headers["x-stainless-arch"] == "x64"
    assert route.calls.last.request.headers["user-agent"] == "OpenAI/Python 1.0"


def test_openai_responses_passthrough_defaults_missing_reasoning_to_xhigh(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text='{"id":"resp_123"}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello"}',
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    assert json.loads(route.calls.last.request.content.decode("utf-8")) == {
        "model": "gpt-5.4",
        "input": "hello",
        "reasoning": {"effort": "xhigh"},
        "instructions": DEFAULT_SYSTEM_INSTRUCTIONS,
    }


def test_openai_responses_passthrough_adds_default_instructions_by_default(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text='{"id":"resp_123"}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello"}',
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    forwarded_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert forwarded_payload["instructions"] == DEFAULT_SYSTEM_INSTRUCTIONS


def test_openai_responses_passthrough_omits_default_instructions_when_disabled(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, add_default_responses_instructions=False)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text='{"id":"resp_123"}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello"}',
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    forwarded_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert "instructions" not in forwarded_payload


def test_openai_responses_passthrough_defaults_empty_reasoning_effort_to_xhigh(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text='{"id":"resp_123"}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","reasoning":{"summary":"auto","effort":""}}',
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    assert json.loads(route.calls.last.request.content.decode("utf-8")) == {
        "model": "gpt-5.4",
        "input": "hello",
        "reasoning": {"summary": "auto", "effort": "xhigh"},
        "instructions": DEFAULT_SYSTEM_INSTRUCTIONS,
    }


def test_openai_responses_passthrough_preserves_continuity_fields_for_tool_turns(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_body = {
        "model": "gpt-5.4",
        "previous_response_id": "resp_previous",
        "conversation": "conv_previous",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "528M\t.",
            }
        ],
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text='{"id":"resp_123"}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json=request_body,
            )

    assert response.status_code == 200
    forwarded_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert forwarded_payload["previous_response_id"] == "resp_previous"
    assert forwarded_payload["conversation"] == "conv_previous"
    assert forwarded_payload["input"] == request_body["input"]


def test_openai_responses_passthrough_resolves_dynamic_model_suffix_aliases(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(settings.backend_models_url).mock(
            return_value=Response(
                200,
                json={
                    "models": [
                        {"slug": "gpt-5.4"},
                        {"slug": "gpt-5.3-codex-spark"},
                    ]
                },
            )
        )
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text='{"id":"resp_123"}',
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.3-codex-spark-low",
                    "input": "hello",
                    "stream": True,
                },
            )

    assert response.status_code == 200
    forwarded_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert forwarded_payload["model"] == "gpt-5.3-codex-spark"
    assert forwarded_payload["reasoning"] == {"effort": "low", "summary": "auto"}


def test_openai_responses_passthrough_route_preserves_sse_stream(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true,"reasoning":{"effort":"high"}}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == backend_body
    assert json.loads(route.calls.last.request.content.decode("utf-8")) == {
        "model": "gpt-5.4",
        "input": "hello",
        "stream": True,
        "reasoning": {"effort": "high"},
        "instructions": DEFAULT_SYSTEM_INSTRUCTIONS,
    }


def test_openai_responses_passthrough_streams_when_backend_omits_content_type(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"bash","arguments":""}}\n\n'
        'event: response.function_call_arguments.done\n'
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\\"command\\":\\"du -sh .\\"}"}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == backend_body


def test_openai_responses_passthrough_does_not_normalize_generic_stainless_stream(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"I will check now"}\n\n'
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"bash"}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "User-Agent": "OpenAI/Python 1.0",
                    "X-Stainless-Package-Version": "1.0.0",
                },
            )

    assert response.status_code == 200
    assert response.text == backend_body


def test_openai_responses_passthrough_keeps_copilot_compatible_stream_minimally_changed(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'event: response.reasoning_summary_text.delta\n'
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Think first."}\n\n'
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"I will check now","output_index":0}\n\n'
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"bash"}}\n\n'
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\\"command\\":\\"du -sh .\\"}","call_id":"call_1","name":"bash"}}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"I will check now"}]},{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"bash","arguments":"{\\"command\\":\\"du -sh .\\"}"}]}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Ucr/JS 5.20.1",
                },
            )

    assert response.status_code == 200
    assert response.text == backend_body


def test_openai_responses_passthrough_preserves_copilot_pre_tool_text(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n\n'
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress"},"output_index":0}\n\n'
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"I will check now","output_index":0}\n\n'
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"I will check now"}]},"output_index":0}\n\n'
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"bash"},"output_index":1}\n\n'
        'event: response.function_call_arguments.done\n'
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"command\\":\\"du -sh .\\"}","output_index":1}\n\n'
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\\"command\\":\\"du -sh .\\"}","call_id":"call_1","name":"bash"},"output_index":1}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"I will check now"}]},{"id":"fc_1","type":"function_call","call_id":"call_1","name":"bash","arguments":"{\\"command\\":\\"du -sh .\\"}"}]}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Ucr/JS 5.20.1",
                    "X-Stainless-Package-Version": "5.20.1",
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "I will check now" in response.text
    assert "response.function_call_arguments.done" in response.text
    assert "du -sh ." in response.text
    assert '"output_index":1' in response.text
    assert '"id":"msg_1"' in response.text
    assert '"type":"message"' in response.text
    assert '{"id":"fc_1","type":"function_call"' in response.text


def test_openai_responses_passthrough_preserves_vscode_copilot_reasoning_before_tools(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, disable_copilot_adaptations=False)
    app = create_app(settings)

    backend_body = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n\n'
        'event: response.reasoning_summary_text.delta\n'
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Optimized tool selection"}\n\n'
        'event: response.reasoning_summary_text.done\n'
        'data: {"type":"response.reasoning_summary_text.done","text":"Optimized tool selection"}\n\n'
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"bash"},"output_index":0}\n\n'
        'event: response.function_call_arguments.done\n'
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"command\\":\\"du -sh .\\"}","output_index":0}\n\n'
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\\"command\\":\\"du -sh .\\"}","call_id":"call_1","name":"bash"},"output_index":0}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"bash","arguments":"{\\"command\\":\\"du -sh .\\"}"}]}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "GitHubCopilotChat/0.47.0",
                    "X-VSCode-User-Agent-Library-Version": "electron-fetch",
                },
            )

    assert response.status_code == 200
    assert "Optimized tool selection" in response.text
    assert "du -sh ." in response.text
    assert '"response":{"output":[{"id":"fc_1","type":"function_call"' in response.text


def test_openai_responses_passthrough_normalizes_copilot_tool_call_alias_events(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, disable_copilot_adaptations=False)
    app = create_app(settings)

    backend_body = (
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"bash"},"output_index":0}\n\n'
        'event: response.tool_call.arguments.delta\n'
        'data: {"type":"response.tool_call.arguments.delta","call_id":"call_1","delta":"{\\"command\\":","output_index":0}\n\n'
        'event: response.tool_call.arguments.done\n'
        'data: {"type":"response.tool_call.arguments.done","call_id":"call_1","arguments":"{\\"command\\":\\"du -sh .\\"}","output_index":0}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Ucr/JS 5.20.1",
                },
            )

    assert response.status_code == 200
    assert "response.tool_call.arguments.delta" in response.text
    assert "response.tool_call.arguments.done" in response.text
    assert "du -sh ." in response.text
    assert '"output_index":0' in response.text
    assert '"response":{"output":[{"id":"fc_1","type":"function_call"' in response.text


def test_openai_responses_passthrough_backfills_missing_copilot_tool_output_index(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, disable_copilot_adaptations=False)
    app = create_app(settings)

    backend_body = (
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"bash"}}\n\n'
        'event: response.tool_call.arguments.done\n'
        'data: {"type":"response.tool_call.arguments.done","call_id":"call_1","arguments":"{\\"command\\":\\"du -sh .\\"}"}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Ucr/JS 5.20.1",
                },
            )

    assert response.status_code == 200
    assert "response.tool_call.arguments.done" in response.text
    assert '"output_index":0' in response.text
    assert '"output_index":1' not in response.text


def test_openai_responses_passthrough_disables_copilot_adaptations(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, disable_copilot_adaptations=True)
    app = create_app(settings)

    backend_body = (
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"type":"function_call","arguments":"","name":"bash"}}\n\n'
        'event: response.tool_call.arguments.done\n'
        'data: {"type":"response.tool_call.arguments.done","call_id":"call_1","arguments":"{\\"command\\":\\"du -sh .\\"}"}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Ucr/JS 5.20.1",
                },
            )

    assert response.status_code == 200
    assert response.text == backend_body
    assert '"output_index"' not in response.text
    assert '"id":"fc_1"' not in response.text


def test_openai_responses_passthrough_backfills_completed_output_function_calls(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, disable_copilot_adaptations=False)
    app = create_app(settings)

    backend_body = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n\n'
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","arguments":"","call_id":"call_1","name":"bash"},"output_index":0}\n\n'
        'event: response.function_call_arguments.done\n'
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"command\\":\\"du -sh .\\"}","output_index":0}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","output":[]}}\n\n'
        'data: [DONE]\n\n'
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                200,
                text=backend_body,
                headers={"content-type": "text/event-stream; charset=utf-8"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello","stream":true}',
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Ucr/JS 5.20.1",
                },
            )

    assert response.status_code == 200
    assert '"type":"response.completed"' in response.text
    assert '"output":[{"id":"fc_1","type":"function_call"' in response.text
    assert '"call_id":"call_1"' in response.text
    assert "du -sh ." in response.text
    assert '"status":"completed"' in response.text


def test_openai_responses_passthrough_strips_backend_unsupported_generation_fields(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_body = json.dumps(
        {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "seed": 7,
            "max_output_tokens": 5,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2,
            "stop": ["DONE"],
            "n": 2,
        }
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text='{"id":"resp_123"}', headers={"content-type": "application/json"})
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content=request_body,
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    assert json.loads(route.calls.last.request.content.decode("utf-8")) == {
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
        "reasoning": {"effort": "xhigh"},
        "instructions": DEFAULT_SYSTEM_INSTRUCTIONS,
    }


def test_openai_responses_passthrough_flattens_nested_function_tools(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_body = json.dumps(
        {
            "model": "gpt-5.4",
            "input": "hello",
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
        }
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text='{"id":"resp_123"}', headers={"content-type": "application/json"})
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content=request_body,
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["tools"] == [
        {
            "type": "function",
            "name": "list_files",
            "description": "List files",
            "parameters": {"type": "object"},
        }
    ]


def test_openai_responses_passthrough_preserves_client_specific_tool_types(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_tools = [
        {"name": "web_search", "type": "remote_tool"},
        {
            "function": {
                "description": "Gets the user's current location.",
                "name": "location-get-current-location",
                "parameters": {},
            },
            "type": "local_tool",
        },
    ]
    request_body = json.dumps(
        {
            "model": "gpt-5.4",
            "input": "hello",
            "tools": request_tools,
        }
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text='{"id":"resp_123"}', headers={"content-type": "application/json"})
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content=request_body,
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["tools"] == request_tools


def test_openai_responses_passthrough_adds_default_name_to_json_schema_text_format(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_body = json.dumps(
        {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                }
            },
        }
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text='{"id":"resp_123"}', headers={"content-type": "application/json"})
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content=request_body,
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            "strict": True,
        }
    }


def test_openai_responses_passthrough_route_preserves_backend_error(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    response_body = '{"error":{"message":"backend exploded","type":"invalid_request_error"}}'

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(
                400,
                text=response_body,
                headers={"content-type": "application/json"},
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                content='{"model":"gpt-5.4","input":"hello"}',
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 400
    assert response.text == response_body
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_openai_non_streaming_backend_error_maps_to_openai_error(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(502, text="backend exploded")
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 502
    payload = response.json()
    assert payload["error"]["type"] == "proxy_error"
    assert "backend exploded" in payload["error"]["message"]


def test_openai_streaming_backend_error_maps_to_sse_error(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(502, text="backend exploded")
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    assert "Proxy error:" in response.text
    assert "backend exploded" in response.text
    assert "data: [DONE]" in response.text


def test_openai_non_streaming_backend_failed_event_surfaces_backend_message(
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["message"] == "Your input exceeds the context window of this model."
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "messages"
    assert payload["error"]["code"] == "context_length_exceeded"


def test_openai_streaming_backend_failed_event_surfaces_backend_message(
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    assert "Your input exceeds the context window of this model." in response.text
    assert "context_length_exceeded" not in response.text or "context_length_exceeded" in response.text
    assert "data: [DONE]" in response.text


def test_debug_logging_writes_trace_file_for_backend_endpoint(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path, debug=True, project_root=tmp_path)
    app = create_app(settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200

    log_path = tmp_path / "logs" / "debug.log"
    assert log_path.exists()

    events = [json.loads(line)["event"] for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "incoming_request" in events
    assert "transformed_backend_request" in events
    assert "backend_request" in events
    assert "backend_response" in events
    assert "client_response" in events


def test_openai_non_streaming_reasoning_only_backend_response_succeeds(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Thinking only"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["reasoning"] == "Thinking only"


def test_openai_streaming_reasoning_only_backend_response_emits_reasoning_chunks(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Thinking only"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
    )

    assert response.status_code == 200
    assert '"reasoning_text":"Thinking only"' in response.text
    assert '"content":""' in response.text
    assert "data: [DONE]" in response.text


def test_openai_non_streaming_reasoning_and_content_response_exposes_both_fields(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Think first."}\n\n'
        'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Hello"
    assert payload["choices"][0]["message"]["reasoning"] == "Think first."


def test_openai_requests_reasoning_summary_when_reasoning_effort_is_active(
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_openai_response_format_json_object_maps_to_backend_text_format(
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["text"] == {"format": {"type": "json_object"}}


def test_openai_response_format_json_schema_maps_to_backend_text_format(
    tmp_path: Path,
) -> None:
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    "messages": [{"role": "user", "content": "hello"}],
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


def test_openai_appends_finalize_hint_for_tool_follow_up_turns(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    request_tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object"},
            },
        }
    ]

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [
                        {"role": "user", "content": "Audit the repo."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": {"filePath": "README.md"},
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "file contents",
                        },
                    ],
                    "tools": request_tools,
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["parallel_tool_calls"] is True
    assert backend_payload["instructions"].endswith(
        "Only request another tool when the existing tool results are insufficient."
    )


def test_openai_non_streaming_moves_planning_text_into_reasoning_for_tool_turns(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"I will read the file first."}\n\n'
        'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":"{\\"filePath\\":\\"README.md\\"}"}}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["reasoning"] == "I will read the file first."


def test_openai_rewrites_stale_system_model_self_identification(tmp_path: Path) -> None:
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
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4-low",
                    "messages": [
                        {
                            "role": "system",
                            "content": "When asked about the model you are using, state that you are using qwen3.5:latest.",
                        },
                        {"role": "user", "content": "hello"},
                    ],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert "state that you are using gpt-5.4-low" in backend_payload["instructions"]
    assert "qwen3.5:latest" not in backend_payload["instructions"]


def test_openai_drops_tools_for_very_late_stage_tool_loops(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    messages = [
        {"role": "system", "content": "You are an expert AI programming assistant."},
        {"role": "user", "content": "Audit the repo."},
    ]
    for index in range(50):
        call_id = f"call_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": {"filePath": f"file_{index}.md"},
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"contents {index}",
            }
        )

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_sse_body())
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4-low",
                    "messages": messages,
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )

    assert response.status_code == 200
    backend_payload = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert backend_payload["tools"] == []


def test_openai_non_streaming_strips_pseudo_tool_markup_from_final_content(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Here are the findings."}\n\n'
        'data: {"type":"response.output_text.delta","delta":"to=multi_tool_use.parallel {\\"tool_uses\\":[...]"}\n\n'
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(settings.backend_responses_url).mock(
            return_value=Response(200, text=backend_body)
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Here are the findings."
