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


def test_openai_streaming_emits_tool_call_from_arguments_done_event(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"list_files"}}\n\n'
        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"path\\""}\n\n'
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","name":"list_files","arguments":"{\\"path\\":\\".\\"}"}\n\n'
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
                            "function": {"name": "list_files", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert '"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"list_files","arguments":"{\\"path\\":\\".\\"}"}}]' in body
    assert body.count('"tool_calls":[') == 1
    assert '"finish_reason":"tool_calls"' in body


def test_openai_streaming_preserves_added_name_when_arguments_done_omits_name(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"list_files"}}\n\n'
        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"path\\""}\n\n'
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"path\\":\\".\\"}"}\n\n'
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
                            "function": {"name": "list_files", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert '"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"list_files","arguments":"{\\"path\\":\\".\\"}"}}]' in body
    assert body.count('"tool_calls":[') == 1
    assert '"finish_reason":"tool_calls"' in body


def test_openai_streaming_emits_tool_call_from_tool_call_done_alias(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_item.added","item":{"id":"tc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"exec_command"}}\n\n'
        'data: {"type":"response.tool_call.arguments.delta","item_id":"tc_1","delta":"{\\"cmd\\""}\n\n'
        'data: {"type":"response.tool_call.arguments.done","item_id":"tc_1","call_id":"call_1","name":"exec_command","arguments":"{\\"cmd\\":\\"rg --files -uu\\"}"}\n\n'
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
                            "function": {"name": "exec_command", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert '"name":"exec_command","arguments":"{\\"cmd\\":\\"rg --files -uu\\"}"' in body
    assert '"finish_reason":"tool_calls"' in body


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
    reasoning_index = body.find('"role":"assistant","content":"","reasoning_text":"I will read the file first."')
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


def test_openai_streaming_strips_split_pseudo_tool_markup_from_content_chunks(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Here are the findings. to=func"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"tions.exec_command {\\"cmd\\":\\"rg --files\\"}"}\n\n'
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
    assert 'rg --files' not in body
    assert 'This should be suppressed too.' not in body
    assert "data: [DONE]" in body


def test_openai_streaming_strips_unprefixed_exec_command_markup_with_cjk_noise(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"I have the inventory. func"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"tions.exec_command 彩乐彩票 --command=\\"du -sh node_modules\\""}\n\n'
        'data: {"type":"response.output_text.delta","delta":"broken chinese should be suppressed"}\n\n'
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
    assert "I have the inventory." in body
    assert "functions.exec_command" not in body
    assert "du -sh node_modules" not in body
    assert "彩乐彩票" not in body
    assert "broken chinese" not in body
    assert "data: [DONE]" in body


def test_openai_streaming_strips_exact_exec_command_leak_with_command_flag(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Inventory complete. to=functions.exec_command "}\n\n'
        'data: {"type":"response.output_text.delta","delta":"--command=\\"rg --files -uu\\""}\n\n'
        'data: {"type":"response.output_text.delta","delta":" --command=\\"du -sh node_modules\\""}\n\n'
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
                    "model": "gpt-5.4-low",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert "Inventory complete." in body
    assert "to=functions.exec_command" not in body
    assert "rg --files -uu" not in body
    assert "du -sh node_modules" not in body
    assert "data: [DONE]" in body


def test_openai_streaming_strips_terminal_run_markup_with_command_json(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.output_text.delta","delta":"Inventory ready. to=terminal.run 官网群=json"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"{\\"command\\":\\"du -sh node_modules\\",\\"cwd\\":\\"/tmp\\"}"}\n\n'
        'data: {"type":"response.output_text.delta","delta":"tail should be hidden"}\n\n'
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
    assert "Inventory ready." in body
    assert "to=terminal.run" not in body
    assert "du -sh node_modules" not in body
    assert "官网群" not in body
    assert "tail should be hidden" not in body
    assert "data: [DONE]" in body


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
    assert '"role":"assistant","content":"","reasoning_text":"Thinking only"' in body
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
    reasoning_index = body.find('"role":"assistant","content":"","reasoning_text":"Think"')
    content_index = body.find('"role":"assistant","content":"OK"')
    final_index = body.find('"finish_reason":"stop"')

    assert reasoning_index != -1
    assert content_index > reasoning_index
    assert final_index > content_index
    assert "data: [DONE]" in body


def test_openai_streaming_reasoning_summary_part_added_does_not_duplicate_done(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    backend_body = (
        'data: {"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":"Plan"}}\n\n'
        'data: {"type":"response.reasoning_summary_part.done","part":{"type":"summary_text","text":"Plan"}}\n\n'
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
    assert body.count('"role":"assistant","content":"","reasoning_text":"Plan"') == 1
    assert '"role":"assistant","content":"OK"' in body
    assert "data: [DONE]" in body


def test_openai_streaming_reclassifies_post_tool_planning_text_as_reasoning(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    write_auth_file(auth_path, {"OPENAI_API_KEY": "backend_key"})
    settings = build_settings(auth_path)
    app = create_app(settings)

    planning_text = (
        "I'm checking the workspace contents and verifying command outputs so I can "
        "give you a complete file list before the final report."
    )
    backend_body = (
        'data: {"type":"response.output_text.done","text":"'
        + planning_text
        + '","item_id":"msg_1","output_index":0}\n\n'
        'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"'
        + planning_text
        + '"}]},"output_index":0}\n\n'
        'data: {"type":"response.output_text.delta","delta":"Available tools in this environment:"}\n\n'
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
                    "messages": [
                        {"role": "user", "content": "list files"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "run_in_terminal",
                                        "arguments": {"command": "ls"},
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_1", "content": "files"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "run_in_terminal", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )

    assert response.status_code == 200
    body = response.text
    assert body.count('"reasoning_text":"' + planning_text) == 1
    assert '"content":"' + planning_text not in body
    assert '"content":"Available tools in this environment:"' in body
