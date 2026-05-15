from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import Headers

from codex_openai_ollama_proxy.api.deps import get_proxy_service, get_settings
from codex_openai_ollama_proxy.core.config import Settings
from codex_openai_ollama_proxy.core.debug_trace import (
    finish_debug_request,
    log_debug_event,
    start_debug_request,
)
from codex_openai_ollama_proxy.core.errors import BackendSSEError, openai_error_response
from codex_openai_ollama_proxy.schemas.openai import ChatCompletionsRequest
from codex_openai_ollama_proxy.services.event_parser import (
    FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES,
    FUNCTION_CALL_ARGUMENTS_DONE_EVENT_TYPES,
)
from codex_openai_ollama_proxy.services.proxy_service import ProxyService
from codex_openai_ollama_proxy.services.streaming_formatter import build_openai_error_sse

router = APIRouter(tags=["openai"])

PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS = {
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


def _split_passthrough_response_headers(
    headers: dict[str, str],
) -> tuple[str | None, dict[str, str]]:
    media_type: str | None = None
    forwarded_headers: dict[str, str] = {}
    for name, value in headers.items():
        normalized_name = name.lower()
        if normalized_name == "content-type":
            media_type = value
            continue
        if normalized_name in PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS:
            continue
        forwarded_headers[name] = value
    return media_type, forwarded_headers


def _is_sse_media_type(media_type: str | None) -> bool:
    if media_type is None:
        return False
    return media_type.split(";", 1)[0].strip().lower() == "text/event-stream"


def _media_type_base(media_type: str | None) -> str | None:
    if media_type is None:
        return None
    return media_type.split(";", 1)[0].strip().lower()


def _is_responses_stream_request(request_body: bytes) -> bool:
    try:
        payload = json.loads(request_body)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def _is_copilot_responses_client(headers: Headers) -> bool:
    user_agent = headers.get("user-agent", "")
    return (
        user_agent.startswith("Ucr/JS ")
        or user_agent.startswith("GitHubCopilotChat/")
        or headers.get("x-vscode-user-agent-library-version") is not None
    )


def _encode_sse_event(event_name: str | None, data: str | dict) -> bytes:
    lines: list[str] = []
    if event_name:
        lines.append(f"event: {event_name}")
    if isinstance(data, dict):
        data_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        data_text = data
    for line in data_text.splitlines() or [""]:
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _is_visible_output_event(payload: dict) -> bool:
    event_type = payload.get("type")
    if event_type in {
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
    }:
        return True
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = payload.get("item")
        return isinstance(item, dict) and item.get("type") == "message"
    return False


def _has_function_call(payload: dict) -> bool:
    event_type = payload.get("type")
    if event_type in FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES | FUNCTION_CALL_ARGUMENTS_DONE_EVENT_TYPES:
        return True
    item = payload.get("item")
    if isinstance(item, dict) and item.get("type") == "function_call":
        return True
    response = payload.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    return isinstance(output, list) and any(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in output
    )


def _function_call_item_ids(item: dict) -> list[str]:
    ids: list[str] = []
    for key in ("id", "call_id"):
        value = item.get(key)
        if isinstance(value, str) and value and value not in ids:
            ids.append(value)
    return ids


def _function_call_event_id(payload: dict) -> str | None:
    for key in ("item_id", "id", "call_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _completed_response_with_function_calls(
    payload: dict,
    function_call_items: dict[str, dict],
) -> dict:
    response = payload.get("response")
    if not isinstance(response, dict):
        return payload

    output = response.get("output")
    if not isinstance(output, list):
        output = []

    completed_output: list[dict] = []
    seen_call_ids: set[str] = set()
    for item in output:
        if not isinstance(item, dict):
            completed_output.append(item)
            continue
        if item.get("type") != "function_call":
            completed_output.append(item)
            continue
        canonical_item = dict(item)
        canonical_item["status"] = "completed"
        call_id = canonical_item.get("call_id") or canonical_item.get("id")
        if isinstance(call_id, str):
            seen_call_ids.add(call_id)
        completed_output.append(canonical_item)

    for item in function_call_items.values():
        call_id = item.get("call_id") or item.get("id")
        if isinstance(call_id, str) and call_id in seen_call_ids:
            continue
        canonical_item = dict(item)
        canonical_item["status"] = "completed"
        if isinstance(call_id, str):
            seen_call_ids.add(call_id)
        completed_output.append(canonical_item)

    if completed_output == output:
        return payload

    updated_response = dict(response)
    updated_response["output"] = completed_output
    updated_payload = dict(payload)
    updated_payload["response"] = updated_response
    return updated_payload


def _record_function_call_output_index(
    payload: dict,
    function_call_output_indexes: dict[str, int],
) -> None:
    output_index = payload.get("output_index")
    if not isinstance(output_index, int):
        return

    item = payload.get("item")
    if isinstance(item, dict) and item.get("type") == "function_call":
        for item_id in _function_call_item_ids(item):
            function_call_output_indexes[item_id] = output_index

    event_id = _function_call_event_id(payload)
    if event_id is not None:
        function_call_output_indexes[event_id] = output_index


def _function_call_output_index(
    payload: dict,
    function_call_output_indexes: dict[str, int],
    default_output_index: int,
) -> int:
    item = payload.get("item")
    if isinstance(item, dict):
        for item_id in _function_call_item_ids(item):
            known_output_index = function_call_output_indexes.get(item_id)
            if known_output_index is not None:
                return known_output_index

    event_id = _function_call_event_id(payload)
    if event_id is not None:
        known_output_index = function_call_output_indexes.get(event_id)
        if known_output_index is not None:
            return known_output_index

    return default_output_index


def _normalize_copilot_tool_output_index(
    payload: dict,
    function_call_output_indexes: dict[str, int],
    default_output_index: int,
) -> dict:
    event_type = payload.get("type")
    item = payload.get("item")
    is_function_item = isinstance(item, dict) and item.get("type") == "function_call"
    is_function_arguments = event_type in (
        FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES | FUNCTION_CALL_ARGUMENTS_DONE_EVENT_TYPES
    )
    if not is_function_item and not is_function_arguments:
        return payload
    if "output_index" in payload:
        return payload
    updated_payload = dict(payload)
    updated_payload["output_index"] = _function_call_output_index(
        payload,
        function_call_output_indexes,
        default_output_index,
    )
    return updated_payload


def _canonicalize_copilot_function_call(
    payload: dict,
    function_call_items: dict[str, dict],
) -> dict:
    event_type = payload.get("type")
    item = payload.get("item")
    updated_payload = dict(payload)

    if isinstance(item, dict) and item.get("type") == "function_call":
        canonical_item = dict(item)
        if event_type == "response.output_item.added":
            canonical_item.setdefault("status", "in_progress")
        elif event_type == "response.output_item.done":
            canonical_item.setdefault("status", "completed")
        item_id = canonical_item.get("id") or canonical_item.get("call_id")
        if not isinstance(item_id, str) or not item_id:
            item_id = f"fc_{len(function_call_items) + 1}"
        canonical_item["id"] = item_id
        canonical_item.setdefault("call_id", item_id)
        for alias_id in _function_call_item_ids(canonical_item):
            function_call_items[alias_id] = canonical_item
        if canonical_item == item:
            return updated_payload
        updated_payload["item"] = canonical_item
        updated_payload["response"] = {"output": [canonical_item]}
        return updated_payload

    if event_type in FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES | FUNCTION_CALL_ARGUMENTS_DONE_EVENT_TYPES:
        item_id = _function_call_event_id(payload)
        if item_id is None:
            return updated_payload

        canonical_item = dict(
            function_call_items.get(
                item_id,
                {
                    "id": item_id,
                    "type": "function_call",
                    "call_id": payload.get("call_id") or item_id,
                    "name": payload.get("name") or "",
                    "arguments": "",
                },
            )
        )
        if event_type in FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES:
            delta = payload.get("delta")
            if isinstance(delta, str):
                canonical_item["arguments"] = f"{canonical_item.get('arguments', '')}{delta}"
            canonical_item.setdefault("status", "in_progress")
        else:
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                canonical_item["arguments"] = arguments
            canonical_item.setdefault("status", "completed")
        if isinstance(payload.get("name"), str):
            canonical_item["name"] = payload["name"]
        if isinstance(payload.get("call_id"), str):
            canonical_item["call_id"] = payload["call_id"]
        for alias_id in _function_call_item_ids(canonical_item):
            function_call_items[alias_id] = canonical_item
        function_call_items[item_id] = canonical_item
        updated_payload["item"] = canonical_item
        updated_payload["response"] = {"output": [canonical_item]}
        return updated_payload

    return updated_payload


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionsRequest,
    raw_request: Request,
    proxy_service: ProxyService = Depends(get_proxy_service),
):
    debug_tokens = start_debug_request(
        raw_request.url.path,
        request.model_dump(by_alias=True, exclude_none=True),
    )

    if request.stream:
        async def stream_with_error_fallback():
            emitted_chunks: list[str] = []
            try:
                try:
                    async for chunk in await proxy_service.stream_chat_completions(
                        request,
                        is_disconnected=raw_request.is_disconnected,
                    ):
                        emitted_chunks.append(chunk)
                        yield chunk
                except Exception as exc:  # noqa: BLE001
                    message = exc.message if isinstance(exc, BackendSSEError) else f"Proxy error: {exc}"
                    chunk = build_openai_error_sse(request.model, message)
                    emitted_chunks.append(chunk)
                    yield chunk
            finally:
                log_debug_event(
                    "client_response",
                    status_code=200,
                    media_type="text/event-stream",
                    body="".join(emitted_chunks),
                )
                finish_debug_request(debug_tokens)

        return StreamingResponse(
            stream_with_error_fallback(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
            },
        )

    try:
        response = await proxy_service.proxy_chat_completions(
            request,
            allow_thinking_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        error_response = openai_error_response(exc)
        log_debug_event(
            "client_response",
            status_code=error_response.status_code,
            media_type="application/json",
            body=error_response.body,
        )
        finish_debug_request(debug_tokens)
        return error_response

    payload = response.model_dump(by_alias=True, exclude_none=True)
    log_debug_event(
        "client_response",
        status_code=200,
        media_type="application/json",
        body=payload,
    )
    finish_debug_request(debug_tokens)
    return JSONResponse(content=payload)


@router.post("/responses")
@router.post("/v1/responses")
async def responses_passthrough(
    raw_request: Request,
    settings: Settings = Depends(get_settings),
    proxy_service: ProxyService = Depends(get_proxy_service),
):
    request_body = await raw_request.body()
    debug_tokens = start_debug_request(raw_request.url.path, request_body)

    try:
        backend_response = await proxy_service.open_responses_passthrough(
            request_body,
            incoming_headers=raw_request.headers,
        )
    except Exception as exc:  # noqa: BLE001
        error_response = openai_error_response(exc)
        log_debug_event(
            "client_response",
            status_code=error_response.status_code,
            media_type="application/json",
            body=error_response.body,
        )
        finish_debug_request(debug_tokens)
        return error_response

    media_type, response_headers = _split_passthrough_response_headers(
        dict(backend_response.headers)
    )

    media_type_base = _media_type_base(media_type)
    should_stream_response = _is_sse_media_type(media_type) or (
        backend_response.status_code < 400
        and _is_responses_stream_request(request_body)
        and media_type_base in {None, "text/plain", "application/octet-stream"}
    )

    if should_stream_response:
        stream_media_type = media_type if _is_sse_media_type(media_type) else "text/event-stream"
        should_normalize_copilot_stream = (
            not settings.disable_copilot_adaptations
            and _is_copilot_responses_client(raw_request.headers)
        )

        async def stream_passthrough():
            emitted_chunks: list[bytes] = []
            if not should_normalize_copilot_stream:
                try:
                    async for chunk in backend_response.aiter_raw():
                        if await raw_request.is_disconnected():
                            break
                        emitted_chunks.append(chunk)
                        yield chunk
                finally:
                    body = b"".join(emitted_chunks)
                    log_debug_event(
                        "backend_response_stream",
                        status_code=backend_response.status_code,
                        body=body,
                    )
                    log_debug_event(
                        "client_response",
                        status_code=backend_response.status_code,
                        media_type=stream_media_type,
                        body=body,
                    )
                    finish_debug_request(debug_tokens)
                    await backend_response.aclose()
                return

            pending_event_name: str | None = None
            pending_data_lines: list[str] = []
            function_call_items: dict[str, dict] = {}
            function_call_output_indexes: dict[str, int] = {}
            saw_visible_output_before_function_call = False
            saw_function_call = False

            async def emit_event(event_name: str | None, data_text: str):
                nonlocal saw_function_call, saw_visible_output_before_function_call
                if data_text.strip() == "[DONE]":
                    yield _encode_sse_event(event_name, data_text)
                    return

                try:
                    payload = json.loads(data_text)
                except (TypeError, ValueError):
                    chunk = _encode_sse_event(event_name, data_text)
                    yield chunk
                    return

                if not isinstance(payload, dict):
                    yield _encode_sse_event(event_name, data_text)
                    return

                default_output_index = 0
                if _has_function_call(payload):
                    default_output_index = 1 if saw_visible_output_before_function_call else 0
                    saw_function_call = True
                    payload = _canonicalize_copilot_function_call(payload, function_call_items)
                    _record_function_call_output_index(payload, function_call_output_indexes)

                if _is_visible_output_event(payload) and not saw_function_call:
                    saw_visible_output_before_function_call = True

                if payload.get("type") == "response.completed" and saw_function_call:
                    payload = _completed_response_with_function_calls(payload, function_call_items)
                if saw_function_call:
                    payload = _canonicalize_copilot_function_call(payload, function_call_items)
                    _record_function_call_output_index(payload, function_call_output_indexes)
                    payload = _normalize_copilot_tool_output_index(
                        payload,
                        function_call_output_indexes,
                        default_output_index,
                    )
                    _record_function_call_output_index(payload, function_call_output_indexes)

                yield _encode_sse_event(event_name, payload)

            try:
                async for line in backend_response.aiter_lines():
                    if await raw_request.is_disconnected():
                        break
                    if line.startswith("event: "):
                        pending_event_name = line[7:]
                        continue
                    if line.startswith("data: "):
                        pending_data_lines.append(line[6:])
                        continue
                    if line != "":
                        continue

                    if pending_data_lines:
                        data_text = "\n".join(pending_data_lines)
                        async for chunk in emit_event(pending_event_name, data_text):
                            emitted_chunks.append(chunk)
                            yield chunk
                    pending_event_name = None
                    pending_data_lines = []

                if pending_data_lines:
                    data_text = "\n".join(pending_data_lines)
                    async for chunk in emit_event(pending_event_name, data_text):
                        emitted_chunks.append(chunk)
                        yield chunk
            finally:
                body = b"".join(emitted_chunks)
                log_debug_event(
                    "backend_response_stream",
                    status_code=backend_response.status_code,
                    body=body,
                )
                log_debug_event(
                    "client_response",
                    status_code=backend_response.status_code,
                    media_type=stream_media_type,
                    body=body,
                )
                finish_debug_request(debug_tokens)
                await backend_response.aclose()

        return StreamingResponse(
            stream_passthrough(),
            status_code=backend_response.status_code,
            media_type=stream_media_type,
            headers=response_headers,
        )

    body = b""
    try:
        body = await backend_response.aread()
    finally:
        await backend_response.aclose()

    log_debug_event(
        "backend_response",
        status_code=backend_response.status_code,
        body=body,
    )
    log_debug_event(
        "client_response",
        status_code=backend_response.status_code,
        media_type=media_type or "application/octet-stream",
        body=body,
    )
    finish_debug_request(debug_tokens)

    return Response(
        content=body,
        status_code=backend_response.status_code,
        media_type=media_type,
        headers=response_headers,
    )
