from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from codex_openai_ollama_proxy.core.config import DEFAULT_SYSTEM_INSTRUCTIONS, Settings
from codex_openai_ollama_proxy.core.debug_trace import log_debug_event
from codex_openai_ollama_proxy.core.errors import BackendSSEError, EmptyBackendResponseError
from codex_openai_ollama_proxy.schemas.backend import ResponsesApiRequest
from codex_openai_ollama_proxy.schemas.events import (
    ErrorEvent,
    ThinkingDeltaEvent,
    ThinkingDoneEvent,
    TextDeltaEvent,
    TextDoneEvent,
    ToolCallChunkEvent,
)
from codex_openai_ollama_proxy.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from codex_openai_ollama_proxy.schemas.openai import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    ChatResponseMessage,
    Choice,
)
from codex_openai_ollama_proxy.schemas.usage import Usage
from codex_openai_ollama_proxy.services.backend_client import BackendClient
from codex_openai_ollama_proxy.services.content_conversion import convert_messages_to_input
from codex_openai_ollama_proxy.services.event_parser import (
    parse_backend_sse_text,
    stream_events_from_sse_lines,
)
from codex_openai_ollama_proxy.services.model_catalog import ModelCatalogService
from codex_openai_ollama_proxy.services.model_resolution import (
    normalize_ollama_think,
    resolve_model_and_reasoning,
    resolve_temperature,
)
from codex_openai_ollama_proxy.services.stream_state import StreamState
from codex_openai_ollama_proxy.services.tool_conversion import (
    convert_chat_tools_to_responses,
    convert_tool_choice,
)

TOOL_FOLLOW_UP_FINALIZE_HINT = (
    "\n\nIf the latest tool results already provide enough information to answer the user's "
    "request, respond with the final answer directly instead of requesting more tools. "
    "Only request another tool when the existing tool results are insufficient."
)


def normalize_tool_turn_reasoning(
    content: str,
    reasoning: str,
    tool_calls: list[ChatToolCall],
) -> tuple[str, str]:
    if tool_calls and content and not reasoning:
        return "", content
    return content, reasoning


class ProxyService:
    def __init__(
        self,
        settings: Settings,
        backend_client: BackendClient,
        model_catalog: ModelCatalogService,
    ) -> None:
        self._settings = settings
        self._backend_client = backend_client
        self._model_catalog = model_catalog

    async def proxy_chat_completions(
        self,
        chat_req: ChatCompletionsRequest,
        *,
        allow_thinking_only: bool = False,
    ) -> ChatCompletionsResponse:
        requested_model = chat_req.model
        responses_req = await self.convert_chat_to_responses(chat_req)
        if not responses_req.input:
            raise ValueError("No non-system input message found (input is empty)")

        response_text = await self._backend_client.send_responses_request(responses_req)
        response_content, response_thinking, response_tool_calls, usage = parse_backend_sse_text(
            response_text,
            allow_thinking_only=allow_thinking_only,
        )
        response_content, response_thinking = normalize_tool_turn_reasoning(
            response_content,
            response_thinking,
            response_tool_calls,
        )

        finish_reason = "stop" if not response_tool_calls else "tool_calls"
        return ChatCompletionsResponse(
            id=f"chatcmpl-{uuid4()}",
            object="chat.completion",
            created=int(datetime.now(UTC).timestamp()),
            model=requested_model,
            system_fingerprint="fp_ollama",
            choices=[
                Choice(
                    index=0,
                    message=ChatResponseMessage(
                        role="assistant",
                        content=response_content,
                        reasoning=response_thinking or None,
                        tool_calls=response_tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage or Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def stream_chat_completions(
        self,
        chat_req: ChatCompletionsRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        from codex_openai_ollama_proxy.services.streaming_formatter import OpenAIStreamFormatter

        requested_model = chat_req.model
        responses_req = await self.convert_chat_to_responses(chat_req)
        if not responses_req.input:
            raise ValueError("No non-system input message found (input is empty)")

        formatter = OpenAIStreamFormatter(requested_model)
        state = StreamState()
        include_usage = should_include_stream_usage(chat_req)

        async def iterator() -> AsyncIterator[str]:
            disconnected = False
            last_tool_call_signature: str | None = None
            buffered_content: list[str] = []
            emitted_reasoning = False
            saw_tool_calls = False
            if is_disconnected is not None and await is_disconnected():
                return
            lines = await self._backend_client.stream_responses_request(responses_req)
            async for event in stream_events_with_idle_heartbeat(
                stream_events_from_sse_lines(lines),
                self._settings.stream_idle_heartbeat_seconds,
            ):
                if event is HEARTBEAT_SENTINEL:
                    if is_disconnected is not None and await is_disconnected():
                        disconnected = True
                        await maybe_aclose_async_iterator(lines)
                        break
                    yield formatter.heartbeat_chunk()
                    continue
                if is_disconnected is not None and await is_disconnected():
                    disconnected = True
                    await maybe_aclose_async_iterator(lines)
                    break
                if isinstance(event, ErrorEvent):
                    raise BackendSSEError(
                        event.message,
                        status_code=event.status_code,
                        error_type=event.error_type,
                        param=event.param,
                        code=event.code,
                    )
                emit = state.apply(event)
                if isinstance(event, TextDeltaEvent) and emit:
                    buffered_content.append(event.text)
                elif isinstance(event, TextDoneEvent) and emit:
                    buffered_content.append(event.text)
                elif isinstance(event, ThinkingDeltaEvent) and emit:
                    if buffered_content:
                        for text in buffered_content:
                            yield formatter.content_chunk(text)
                        buffered_content.clear()
                    yield formatter.reasoning_chunk(event.text)
                    emitted_reasoning = True
                elif isinstance(event, ThinkingDoneEvent) and emit:
                    if buffered_content:
                        for text in buffered_content:
                            yield formatter.content_chunk(text)
                        buffered_content.clear()
                    yield formatter.reasoning_chunk(event.text)
                    emitted_reasoning = True
                elif (
                    isinstance(event, ToolCallChunkEvent)
                    and event.is_final
                    and state.tool_calls
                ):
                    if buffered_content and not emitted_reasoning:
                        for text in buffered_content:
                            yield formatter.reasoning_chunk(text)
                        buffered_content.clear()
                        emitted_reasoning = True
                    elif buffered_content:
                        for text in buffered_content:
                            yield formatter.content_chunk(text)
                        buffered_content.clear()
                    tool_call_signature = json.dumps(
                        [tool.model_dump(by_alias=True) for tool in state.tool_calls],
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if tool_call_signature != last_tool_call_signature:
                        yield formatter.tool_calls_chunk(state.tool_calls)
                        last_tool_call_signature = tool_call_signature
                    saw_tool_calls = True

            if disconnected:
                return

            if not state.has_any_output:
                raise EmptyBackendResponseError(
                    "Empty content and no tool calls returned from ChatGPT backend"
                )

            if buffered_content:
                chunk_writer = formatter.reasoning_chunk if saw_tool_calls and not emitted_reasoning else formatter.content_chunk
                for text in buffered_content:
                    yield chunk_writer(text)

            yield formatter.final_chunk(state.finish_reason)
            if include_usage and state.usage is not None:
                yield formatter.usage_chunk(state.usage)
            yield formatter.done_chunk()

        return iterator()

    async def convert_chat_to_responses(
        self, chat_req: ChatCompletionsRequest
    ) -> ResponsesApiRequest:
        base_models = await self._model_catalog.get_base_models_for_request(chat_req.model)
        backend_model, reasoning = resolve_model_and_reasoning(
            chat_req.model,
            chat_req.reasoning,
            chat_req.reasoning_effort,
            base_models,
        )
        temperature = resolve_temperature(
            chat_req.model,
            backend_model,
            reasoning,
            chat_req.temperature,
        )
        converted_tools = convert_chat_tools_to_responses(chat_req.tools)
        converted_tool_choice = convert_tool_choice(chat_req.tool_choice)
        text_config = build_responses_text_config(chat_req.response_format)
        input_items, instructions = convert_messages_to_input(
            chat_req.messages,
            default_instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
        )
        if should_append_tool_finalize_hint(chat_req):
            instructions += TOOL_FOLLOW_UP_FINALIZE_HINT

        responses_request = ResponsesApiRequest(
            model=backend_model,
            instructions=instructions,
            input=input_items,
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            parallel_tool_calls=True,
            temperature=temperature,
            reasoning=reasoning,
            text=text_config,
            store=False,
            stream=True,
            include=[],
        )
        log_debug_event("transformed_backend_request", payload=responses_request)
        return responses_request

    async def open_responses_passthrough(
        self,
        request_body: bytes,
        incoming_headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        request_body = normalize_responses_body_for_backend(
            request_body,
            default_effort="xhigh",
            default_instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
        )
        return await self._backend_client.open_responses_passthrough(
            request_body,
            incoming_headers=incoming_headers,
        )

    async def proxy_ollama_chat(self, request: OllamaChatRequest) -> ChatCompletionsResponse:
        chat_request = self._build_chat_request_from_ollama(
            model=request.model,
            messages=request.messages,
            prompt=request.prompt,
            images=request.images,
            format=request.format,
            system=request.system,
            stream=request.stream,
            think=request.think,
            tools=getattr(request, "tools", None),
            tool_choice=getattr(request, "tool_choice", None),
        )
        return await self.proxy_chat_completions(chat_request, allow_thinking_only=True)

    async def stream_ollama_chat(
        self,
        request: OllamaChatRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        return await self._stream_ollama(
            request,
            mode="chat",
            is_disconnected=is_disconnected,
        )

    async def stream_ollama_generate(
        self,
        request: OllamaGenerateRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        return await self._stream_ollama(
            request,
            mode="generate",
            is_disconnected=is_disconnected,
        )

    async def _stream_ollama(
        self,
        request: OllamaChatRequest | OllamaGenerateRequest,
        *,
        mode: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        from codex_openai_ollama_proxy.services.streaming_formatter import OllamaStreamFormatter

        chat_request = self._build_chat_request_from_ollama(
            model=request.model,
            messages=request.messages,
            prompt=request.prompt,
            images=request.images,
            format=request.format,
            system=request.system,
            stream=request.stream,
            think=request.think,
            tools=getattr(request, "tools", None),
            tool_choice=getattr(request, "tool_choice", None),
        )
        requested_model = normalize_ollama_model(request.model)
        responses_req = await self.convert_chat_to_responses(chat_request)
        if not responses_req.input:
            raise ValueError("No non-system input message found (input is empty)")

        formatter = OllamaStreamFormatter(requested_model, mode=mode)
        state = StreamState()

        async def iterator() -> AsyncIterator[str]:
            disconnected = False
            last_tool_call_signature: str | None = None
            if is_disconnected is not None and await is_disconnected():
                return
            lines = await self._backend_client.stream_responses_request(responses_req)
            async for event in stream_events_with_idle_heartbeat(
                stream_events_from_sse_lines(lines),
                self._settings.stream_idle_heartbeat_seconds,
            ):
                if event is HEARTBEAT_SENTINEL:
                    if is_disconnected is not None and await is_disconnected():
                        disconnected = True
                        await maybe_aclose_async_iterator(lines)
                        break
                    yield formatter.heartbeat_chunk()
                    continue
                if is_disconnected is not None and await is_disconnected():
                    disconnected = True
                    await maybe_aclose_async_iterator(lines)
                    break
                if isinstance(event, ErrorEvent):
                    raise BackendSSEError(
                        event.message,
                        status_code=event.status_code,
                        error_type=event.error_type,
                        param=event.param,
                        code=event.code,
                    )
                emit = state.apply(event)
                if isinstance(event, TextDeltaEvent) and emit:
                    yield formatter.content_chunk(event.text)
                elif isinstance(event, TextDoneEvent) and emit:
                    yield formatter.content_chunk(event.text)
                elif isinstance(event, ThinkingDeltaEvent) and emit:
                    yield formatter.thinking_chunk(event.text)
                elif isinstance(event, ThinkingDoneEvent) and emit:
                    yield formatter.thinking_chunk(event.text)
                elif isinstance(event, ToolCallChunkEvent) and mode == "chat" and state.tool_calls:
                    tool_call_signature = json.dumps(
                        [tool.model_dump(by_alias=True) for tool in state.tool_calls],
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if tool_call_signature != last_tool_call_signature:
                        yield formatter.tool_call_snapshot_chunk(state.tool_calls)
                        last_tool_call_signature = tool_call_signature

            if disconnected:
                return

            if not state.has_any_output:
                raise EmptyBackendResponseError(
                    "Empty content and no tool calls returned from ChatGPT backend"
                )

            yield formatter.final_chunk(state.usage)

        return iterator()

    async def proxy_ollama_generate(
        self, request: OllamaGenerateRequest
    ) -> ChatCompletionsResponse:
        chat_request = self._build_chat_request_from_ollama(
            model=request.model,
            messages=request.messages,
            prompt=request.prompt,
            images=request.images,
            format=request.format,
            system=request.system,
            stream=request.stream,
            think=request.think,
            tools=None,
            tool_choice=None,
        )
        return await self.proxy_chat_completions(chat_request, allow_thinking_only=True)

    def _build_chat_request_from_ollama(
        self,
        *,
        model: str,
        messages: list[ChatMessage] | None,
        prompt: str | None,
        images: list[str] | None,
        format: Any | None,
        system: str | None,
        stream: bool | None,
        think: bool | str | None,
        tools: list[Any] | None,
        tool_choice: Any | None,
    ) -> ChatCompletionsRequest:
        normalized_model = normalize_ollama_model(model)
        resolved_messages = list(messages or [])
        reasoning_effort = normalize_ollama_think(think)
        reasoning = None
        if think is None or reasoning_effort not in {None, "none"}:
            reasoning = {"summary": "auto"}

        def merge_images_into_content(content: Any, message_images: list[str] | None) -> Any:
            if not message_images:
                return content

            parts: list[Any] = []
            if isinstance(content, list):
                parts.extend(content)
            elif content is not None:
                parts.append({"type": "text", "text": str(content)})

            parts.extend(
                {"type": "input_image", "image_base64": image}
                for image in message_images
            )
            return parts

        if system and system.strip():
            resolved_messages.insert(
                0,
                ChatMessage(role="system", content=system),
            )

        resolved_messages = [
            message.model_copy(
                update={
                    "content": merge_images_into_content(message.content, message.images)
                }
            )
            for message in resolved_messages
        ]

        if not has_non_system_message(resolved_messages):
            if not prompt or not prompt.strip():
                raise ValueError("missing prompt or messages")
            resolved_messages.append(
                ChatMessage(
                    role="user",
                    content=merge_images_into_content(prompt, images),
                )
            )

        return ChatCompletionsRequest(
            model=normalized_model,
            messages=resolved_messages,
            temperature=None,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            response_format=format,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
        )


def normalize_ollama_model(model: str) -> str:
    return model[:-7] if model.endswith(":latest") else model


def should_include_stream_usage(chat_req: ChatCompletionsRequest) -> bool:
    stream_options = getattr(chat_req, "stream_options", None)
    if not isinstance(stream_options, dict):
        return False
    return stream_options.get("include_usage") is True


def should_append_tool_finalize_hint(chat_req: ChatCompletionsRequest) -> bool:
    if not chat_req.tools:
        return False

    messages = chat_req.messages
    if len(messages) < 2:
        return False

    trailing_tool_messages = 0
    for message in reversed(messages):
        if message.role.lower() == "tool":
            trailing_tool_messages += 1
            continue
        break

    if trailing_tool_messages == 0:
        return False

    assistant_index = len(messages) - trailing_tool_messages - 1
    if assistant_index < 0:
        return False

    assistant_message = messages[assistant_index]
    return (
        assistant_message.role.lower() == "assistant"
        and bool(assistant_message.tool_calls)
    )


def build_responses_text_config(response_format: Any) -> Any:
    if response_format is None:
        return None

    if isinstance(response_format, str):
        if response_format.strip().lower() == "json":
            return {"format": {"type": "json_object"}}
        return None

    if not isinstance(response_format, dict):
        return None

    format_type = response_format.get("type")
    if format_type == "json_object":
        return {"format": {"type": "json_object"}}

    if format_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            return None

        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            return None

        format_payload: dict[str, Any] = {
            "type": "json_schema",
            "name": json_schema.get("name") if isinstance(json_schema.get("name"), str) else "response",
            "schema": schema,
        }
        if "strict" in json_schema:
            format_payload["strict"] = json_schema["strict"]
        if isinstance(json_schema.get("description"), str):
            format_payload["description"] = json_schema["description"]
        return {"format": format_payload}

    return {
        "format": {
            "type": "json_schema",
            "name": "response",
            "schema": response_format,
            "strict": True,
        }
    }


def has_non_system_message(messages: list[ChatMessage]) -> bool:
    return any(message.role.lower() != "system" for message in messages)


async def maybe_aclose_async_iterator(iterator: object) -> None:
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        await aclose()


async def async_anext(iterator: AsyncIterator[Any]) -> Any:
    return await iterator.__anext__()


def apply_default_reasoning_effort_to_responses_body(
    request_body: bytes,
    *,
    default_effort: str,
) -> bytes:
    try:
        payload = json.loads(request_body)
    except (TypeError, ValueError):
        return request_body

    if not isinstance(payload, dict):
        return request_body

    updated_payload = with_default_reasoning_effort(payload, default_effort=default_effort)
    if updated_payload is payload:
        return request_body

    return json.dumps(
        updated_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


RESPONSES_UNSUPPORTED_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "max_output_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "n",
        "previous_response_id",
        "conversation",
        "truncation",
    }
)


def normalize_responses_body_for_backend(
    request_body: bytes,
    *,
    default_effort: str,
    default_instructions: str,
) -> bytes:
    try:
        payload = json.loads(request_body)
    except (TypeError, ValueError):
        return request_body

    if not isinstance(payload, dict):
        return request_body

    updated_payload = dict(payload)

    if not isinstance(updated_payload.get("instructions"), str) or not updated_payload.get(
        "instructions", ""
    ).strip():
        updated_payload["instructions"] = default_instructions

    if "reasoning_effort" in updated_payload and "reasoning" not in updated_payload:
        updated_payload["reasoning"] = {"effort": updated_payload.pop("reasoning_effort")}

    normalized_payload = with_default_reasoning_effort(
        updated_payload,
        default_effort=default_effort,
    )
    if isinstance(normalized_payload, dict):
        updated_payload = dict(normalized_payload)

    tools = updated_payload.get("tools")
    if isinstance(tools, list):
        updated_payload["tools"] = convert_chat_tools_to_responses(tools)

    text_config = updated_payload.get("text")
    if isinstance(text_config, dict):
        format_config = text_config.get("format")
        if isinstance(format_config, dict) and format_config.get("type") == "json_schema":
            format_config = dict(format_config)
            format_config.setdefault("name", "response")
            if "strict" not in format_config:
                format_config["strict"] = True
            updated_text = dict(text_config)
            updated_text["format"] = format_config
            updated_payload["text"] = updated_text

    for field_name in RESPONSES_UNSUPPORTED_FIELDS:
        updated_payload.pop(field_name, None)

    return json.dumps(
        updated_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def with_default_reasoning_effort(
    payload: dict[str, Any],
    *,
    default_effort: str,
) -> dict[str, Any]:
    reasoning = payload.get("reasoning")

    if reasoning is None:
        updated_payload = dict(payload)
        updated_payload["reasoning"] = {"effort": default_effort}
        return updated_payload

    if isinstance(reasoning, str):
        if reasoning.strip():
            return payload
        updated_payload = dict(payload)
        updated_payload["reasoning"] = {"effort": default_effort}
        return updated_payload

    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.strip():
            return payload
        if effort is not None and not isinstance(effort, str):
            return payload

        updated_payload = dict(payload)
        updated_reasoning = dict(reasoning)
        updated_reasoning["effort"] = default_effort
        updated_payload["reasoning"] = updated_reasoning
        return updated_payload

    return payload


HEARTBEAT_SENTINEL = object()


async def stream_events_with_idle_heartbeat(
    events: AsyncIterator[Any],
    interval_seconds: float,
) -> AsyncIterator[Any]:
    if interval_seconds <= 0:
        async for event in events:
            yield event
        return

    pending: asyncio.Task[Any] | None = asyncio.create_task(async_anext(events))
    try:
        while pending is not None:
            done, _ = await asyncio.wait({pending}, timeout=interval_seconds)
            if not done:
                yield HEARTBEAT_SENTINEL
                continue

            try:
                event = pending.result()
            except StopAsyncIteration:
                break

            yield event
            pending = asyncio.create_task(async_anext(events))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
