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


def build_settings(auth_path: Path) -> Settings:
    return Settings(
        port=8888,
        auth_path=auth_path,
        required_client_api_key=None,
        service_name="codex-openai-ollama-proxy",
        service_version="0.1.0",
    )


def test_openai_streaming_preserves_chunk_order(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Hel"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"lo"}\n\n'
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
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    body = response.text
    first_delta_index = body.find('"role":"assistant","content":"Hel"')
    second_delta_index = body.find('"role":"assistant","content":"lo"')
    final_index = body.find('"finish_reason":"stop"')

    assert first_delta_index != -1
    assert second_delta_index > first_delta_index
    assert final_index > second_delta_index
    assert '"system_fingerprint":"fp_ollama"' in body
    assert '"total_tokens":6' not in body
    assert "data: [DONE]" in body


def test_openai_streaming_emits_tool_call_snapshot_before_done(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"list_files"}}\n\n'
        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"path\\""}\n\n'
        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":":\\".\\"}"}\n\n'
        'data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"list_files","arguments":"{\\"path\\":\\".\\"}"}}\n\n'
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
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
    )

    assert response.status_code == 200
    body = response.text
    tool_chunk = '"role":"assistant","content":"","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"list_files","arguments":"{\\"path\\":\\".\\"}"}}]'
    assert body.find(tool_chunk) != -1
    assert body.count('"tool_calls":[') == 1
    assert body.find('"finish_reason":"tool_calls"') > body.find(tool_chunk)
    assert '"choices":[],"usage":{' in body
    assert "data: [DONE]" in body


def test_openai_streaming_emits_usage_chunk_only_when_requested(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
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
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert '"choices":[],"usage":{' in body
    assert '"total_tokens":6' in body


def test_openai_streaming_moves_planning_content_into_reasoning_before_tool_calls(
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
                    "stream": True,
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
    body = response.text
    reasoning_index = body.find('"role":"assistant","content":"","reasoning":"I will read the file first."')
    tool_index = body.find('"tool_calls":[')
    assert reasoning_index != -1
    assert tool_index > reasoning_index


def test_openai_streaming_strips_pseudo_tool_markup_from_content_chunks(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Here are the findings."}\n\n'
        'data: {"type":"response.output_text.delta","delta":"to=functions.exec_command {\\"cmd\\":\\"pwd\\"}"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"This should be suppressed too."}\n\n'
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
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert 'Here are the findings.' in body
    assert 'to=functions.exec_command' not in body
    assert 'This should be suppressed too.' not in body


def test_openai_streaming_reasoning_only_backend_response_emits_reasoning_deltas(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Thinking only"}\n\n'
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
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
    )

    assert response.status_code == 200
    body = response.text
    assert '"role":"assistant","content":"","reasoning":"Thinking only"' in body
    assert '"finish_reason":"stop"' in body
    assert '"content":""' in body
    assert "data: [DONE]" in body


def test_openai_streaming_reasoning_and_content_are_split_like_ollama_v1(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Think"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"OK"}\n\n'
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
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    body = response.text
    reasoning_index = body.find('"role":"assistant","content":"","reasoning":"Think"')
    content_index = body.find('"role":"assistant","content":"OK"')
    final_index = body.find('"finish_reason":"stop"')

    assert reasoning_index != -1
    assert content_index > reasoning_index
    assert final_index > content_index
    assert "data: [DONE]" in body
