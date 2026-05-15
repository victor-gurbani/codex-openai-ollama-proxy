from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict
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
    ChatToolCall,
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
    resolve_model_alias,
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
TOOL_FORCE_FINALIZE_HINT = (
    "\n\nYou already have enough tool results to answer the user's request. "
    "Do not request any more tools. Provide the final answer now using only the existing tool results."
)

MODEL_SELF_IDENTIFICATION_PATTERN = re.compile(
    r"(state that you are using\s+)([^.\n]+)",
    re.IGNORECASE,
)
PSEUDO_TOOL_LEAK_MARKERS = (
    "to=terminal.",
    "to=terminal.run",
    "to=functions.",
    "to=functions.exec_command",
    "to=multi_tool_use.",
    "to=multi_tool_use.parallel",
    "terminal.run",
    "functions.exec_command",
    '{"tool_uses":',
    '"command":"',
    '"cwd":"',
    '"to":"terminal.',
    '"to":"functions.',
    '"to":"multi_tool_use.',
    'recipient_name":"functions.',
    'recipient_name":"multi_tool_use.',
    'recipient_name":"terminal.',
)
MAX_PSEUDO_TOOL_LEAK_MARKER_LENGTH = max(len(marker) for marker in PSEUDO_TOOL_LEAK_MARKERS)
MIN_PSEUDO_TOOL_LEAK_MARKER_PREFIX_LENGTH = 3
AGENTIC_PLANNING_STARTS = (
    "i'm ",
    "i’m ",
    "i am ",
    "i'll ",
    "i’ll ",
    "i will ",
)
AGENTIC_PLANNING_ACTIONS = (
    "check",
    "checking",
    "confirm",
    "confirming",
    "gather",
    "gathering",
    "inspect",
    "inspecting",
    "enumerate",
    "enumerating",
    "verify",
    "verifying",
    "run",
    "running",
    "execute",
    "executing",
    "collect",
    "collecting",
)
AGENTIC_PLANNING_PURPOSE_MARKERS = (
    " so i can ",
    " so we can ",
    " then i",
    " before the final ",
    " next i",
    " outcome:",
    " why:",
)


def normalize_tool_turn_reasoning(
    content: str,
    reasoning: str,
    tool_calls: list[ChatToolCall],
) -> tuple[str, str]:
    if tool_calls and content and not reasoning:
        return "", content
    return content, reasoning


def strip_pseudo_tool_markup(text: str) -> str:
    earliest_index: int | None = None
    for marker in PSEUDO_TOOL_LEAK_MARKERS:
        index = text.find(marker)
        if index != -1 and (earliest_index is None or index < earliest_index):
            earliest_index = index
    return text[:earliest_index].rstrip() if earliest_index is not None else text


def looks_like_agentic_planning_text(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized.startswith(AGENTIC_PLANNING_STARTS):
        return False
    if not any(action in normalized[:80] for action in AGENTIC_PLANNING_ACTIONS):
        return False
    return any(marker in normalized for marker in AGENTIC_PLANNING_PURPOSE_MARKERS)


class StreamTextLeakFilter:
    def __init__(self) -> None:
        self.drop_remaining = False
        self._pending = ""

    def sanitize(self, text: str) -> str:
        if self.drop_remaining:
            return ""
        combined = self._pending + text
        self._pending = ""
        cleaned = strip_pseudo_tool_markup(combined)
        if cleaned != combined:
            self.drop_remaining = True
            return cleaned
        marker_prefix = longest_pseudo_tool_marker_prefix_suffix(combined)
        if marker_prefix:
            self._pending = marker_prefix
            return combined[: -len(marker_prefix)]
        return cleaned

    def flush(self) -> str:
        if self.drop_remaining:
            self._pending = ""
            return ""
        pending = self._pending
        self._pending = ""
        return pending


def longest_pseudo_tool_marker_prefix_suffix(text: str) -> str:
    max_length = min(len(text), MAX_PSEUDO_TOOL_LEAK_MARKER_LENGTH - 1)
    for length in range(max_length, MIN_PSEUDO_TOOL_LEAK_MARKER_PREFIX_LENGTH - 1, -1):
        suffix = text[-length:]
        if any(marker.startswith(suffix) for marker in PSEUDO_TOOL_LEAK_MARKERS):
            return suffix
    return ""


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
        response_content = strip_pseudo_tool_markup(response_content)
        response_thinking = strip_pseudo_tool_markup(response_thinking)

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
        should_reclassify_planning_text = bool(chat_req.tools) or any(
            message.role == "tool" for message in chat_req.messages
        )

        async def iterator() -> AsyncIterator[str]:
            disconnected = False
            last_tool_call_signatures: dict[str, str] = {}
            buffered_content: list[str] = []
            emitted_reasoning = False
            saw_tool_calls = False
            content_leak_filter = StreamTextLeakFilter()
            reasoning_leak_filter = StreamTextLeakFilter()

            async def flush_buffered_content_as_content() -> AsyncIterator[str]:
                nonlocal buffered_content
                if not buffered_content:
                    return
                for text in buffered_content:
                    cleaned = content_leak_filter.sanitize(text)
                    if cleaned:
                        yield formatter.content_chunk(cleaned)
                buffered_content = []

            async def flush_buffered_content_at_end() -> AsyncIterator[str]:
                nonlocal buffered_content, emitted_reasoning
                if not buffered_content:
                    return
                for text in buffered_content:
                    if (
                        should_reclassify_planning_text
                        and not emitted_reasoning
                        and looks_like_agentic_planning_text(text)
                    ):
                        cleaned = reasoning_leak_filter.sanitize(text)
                        if cleaned:
                            yield formatter.reasoning_chunk(cleaned)
                        flushed_reasoning = reasoning_leak_filter.flush()
                        if flushed_reasoning:
                            yield formatter.reasoning_chunk(flushed_reasoning)
                        emitted_reasoning = True
                        continue

                    chunk_writer = (
                        formatter.reasoning_chunk
                        if saw_tool_calls and not emitted_reasoning
                        else formatter.content_chunk
                    )
                    leak_filter = (
                        reasoning_leak_filter
                        if chunk_writer is formatter.reasoning_chunk
                        else content_leak_filter
                    )
                    cleaned = leak_filter.sanitize(text)
                    if cleaned:
                        yield chunk_writer(cleaned)
                flushed_content = content_leak_filter.flush()
                if flushed_content:
                    yield formatter.content_chunk(flushed_content)
                flushed_reasoning = reasoning_leak_filter.flush()
                if flushed_reasoning:
                    yield formatter.reasoning_chunk(flushed_reasoning)
                buffered_content = []

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
                        async for chunk in flush_buffered_content_as_content():
                            yield chunk
                        await maybe_aclose_async_iterator(lines)
                        break
                    yield formatter.heartbeat_chunk()
                    continue
                if is_disconnected is not None and await is_disconnected():
                    disconnected = True
                    async for chunk in flush_buffered_content_as_content():
                        yield chunk
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
                            cleaned = content_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.content_chunk(cleaned)
                        flushed_content = content_leak_filter.flush()
                        if flushed_content:
                            yield formatter.content_chunk(flushed_content)
                        buffered_content.clear()
                    cleaned_reasoning = reasoning_leak_filter.sanitize(event.text)
                    if cleaned_reasoning:
                        yield formatter.reasoning_chunk(cleaned_reasoning)
                    emitted_reasoning = True
                elif isinstance(event, ThinkingDoneEvent) and emit:
                    if buffered_content:
                        for text in buffered_content:
                            cleaned = content_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.content_chunk(cleaned)
                        flushed_content = content_leak_filter.flush()
                        if flushed_content:
                            yield formatter.content_chunk(flushed_content)
                        buffered_content.clear()
                    cleaned_reasoning = reasoning_leak_filter.sanitize(event.text)
                    if cleaned_reasoning:
                        yield formatter.reasoning_chunk(cleaned_reasoning)
                    emitted_reasoning = True
                elif (
                    isinstance(event, ToolCallChunkEvent)
                    and event.is_final
                    and state.tool_calls
                ):
                    if buffered_content and not emitted_reasoning:
                        for text in buffered_content:
                            cleaned = reasoning_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.reasoning_chunk(cleaned)
                        flushed_reasoning = reasoning_leak_filter.flush()
                        if flushed_reasoning:
                            yield formatter.reasoning_chunk(flushed_reasoning)
                        buffered_content.clear()
                        emitted_reasoning = True
                    elif buffered_content:
                        for text in buffered_content:
                            cleaned = content_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.content_chunk(cleaned)
                        flushed_content = content_leak_filter.flush()
                        if flushed_content:
                            yield formatter.content_chunk(flushed_content)
                        buffered_content.clear()
                    tool_call_signature = json.dumps(
                        asdict(event),
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if tool_call_signature != last_tool_call_signatures.get(event.item_id):
                        yield formatter.tool_call_chunk(event)
                        last_tool_call_signatures[event.item_id] = tool_call_signature
                    saw_tool_calls = True

            if disconnected:
                return

            if not state.has_any_output:
                raise EmptyBackendResponseError(
                    "Empty content and no tool calls returned from ChatGPT backend"
                )

            async for chunk in flush_buffered_content_at_end():
                yield chunk

            flushed_content = content_leak_filter.flush()
            if flushed_content:
                yield formatter.content_chunk(flushed_content)
            flushed_reasoning = reasoning_leak_filter.flush()
            if flushed_reasoning:
                yield formatter.reasoning_chunk(flushed_reasoning)

            yield formatter.final_chunk(state.finish_reason)
            if include_usage and state.usage is not None:
                yield formatter.usage_chunk(state.usage)
            yield formatter.done_chunk()

        return iterator()

    async def convert_chat_to_responses(
        self, chat_req: ChatCompletionsRequest
    ) -> ResponsesApiRequest:
        normalized_messages = normalize_model_self_identification(
            chat_req.messages,
            chat_req.model,
        )
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
        if should_force_finalize_without_tools(chat_req):
            converted_tools = []
        converted_tool_choice = convert_tool_choice(chat_req.tool_choice)
        text_config = build_responses_text_config(chat_req.response_format)
        input_items, instructions = convert_messages_to_input(
            normalized_messages,
            default_instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
        )
        if should_append_tool_finalize_hint(chat_req):
            instructions += TOOL_FOLLOW_UP_FINALIZE_HINT
        if should_force_finalize_without_tools(chat_req):
            instructions += TOOL_FORCE_FINALIZE_HINT

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
        request_model = extract_responses_model(request_body)
        base_models = (
            await self._model_catalog.get_base_models_for_request(request_model)
            if request_model is not None
            else self._model_catalog.cached_or_fallback_base_models()
        )
        request_body = normalize_responses_body_for_backend(
            request_body,
            default_effort="xhigh",
            default_instructions=(
                DEFAULT_SYSTEM_INSTRUCTIONS
                if self._settings.add_default_responses_instructions
                else None
            ),
            base_models=base_models,
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
        request_has_tools = bool(getattr(request, "tools", None))

        async def iterator() -> AsyncIterator[str]:
            disconnected = False
            last_tool_call_signature: str | None = None
            buffered_content: list[str] = []
            emitted_thinking = False
            saw_tool_calls = False
            content_leak_filter = StreamTextLeakFilter()
            thinking_leak_filter = StreamTextLeakFilter()
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
                    if emitted_thinking or not request_has_tools:
                        cleaned = content_leak_filter.sanitize(event.text)
                        if cleaned:
                            yield formatter.content_chunk(cleaned)
                    else:
                        buffered_content.append(event.text)
                elif isinstance(event, TextDoneEvent) and emit:
                    if emitted_thinking or not request_has_tools:
                        cleaned = content_leak_filter.sanitize(event.text)
                        if cleaned:
                            yield formatter.content_chunk(cleaned)
                    else:
                        buffered_content.append(event.text)
                elif isinstance(event, ThinkingDeltaEvent) and emit:
                    if buffered_content:
                        for text in buffered_content:
                            cleaned = content_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.content_chunk(cleaned)
                        flushed_content = content_leak_filter.flush()
                        if flushed_content:
                            yield formatter.content_chunk(flushed_content)
                        buffered_content.clear()
                    cleaned = thinking_leak_filter.sanitize(event.text)
                    if cleaned:
                        yield formatter.thinking_chunk(cleaned)
                    emitted_thinking = True
                elif isinstance(event, ThinkingDoneEvent) and emit:
                    if buffered_content:
                        for text in buffered_content:
                            cleaned = content_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.content_chunk(cleaned)
                        flushed_content = content_leak_filter.flush()
                        if flushed_content:
                            yield formatter.content_chunk(flushed_content)
                        buffered_content.clear()
                    cleaned = thinking_leak_filter.sanitize(event.text)
                    if cleaned:
                        yield formatter.thinking_chunk(cleaned)
                    emitted_thinking = True
                elif (
                    isinstance(event, ToolCallChunkEvent)
                    and event.is_final
                    and mode == "chat"
                    and state.tool_calls
                ):
                    if buffered_content and not emitted_thinking:
                        for text in buffered_content:
                            cleaned = thinking_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.thinking_chunk(cleaned)
                        flushed_thinking = thinking_leak_filter.flush()
                        if flushed_thinking:
                            yield formatter.thinking_chunk(flushed_thinking)
                        buffered_content.clear()
                        emitted_thinking = True
                    elif buffered_content:
                        for text in buffered_content:
                            cleaned = content_leak_filter.sanitize(text)
                            if cleaned:
                                yield formatter.content_chunk(cleaned)
                        flushed_content = content_leak_filter.flush()
                        if flushed_content:
                            yield formatter.content_chunk(flushed_content)
                        buffered_content.clear()
                    tool_call_signature = json.dumps(
                        [tool.model_dump(by_alias=True) for tool in state.tool_calls],
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if tool_call_signature != last_tool_call_signature:
                        yield formatter.tool_call_snapshot_chunk(state.tool_calls)
                        last_tool_call_signature = tool_call_signature
                    saw_tool_calls = True

            if disconnected:
                return

            if not state.has_any_output:
                raise EmptyBackendResponseError(
                    "Empty content and no tool calls returned from ChatGPT backend"
                )

            if buffered_content:
                chunk_writer = formatter.thinking_chunk if saw_tool_calls and not emitted_thinking else formatter.content_chunk
                leak_filter = thinking_leak_filter if chunk_writer is formatter.thinking_chunk else content_leak_filter
                for text in buffered_content:
                    cleaned = leak_filter.sanitize(text)
                    if cleaned:
                        yield chunk_writer(cleaned)
                flushed = leak_filter.flush()
                if flushed:
                    yield chunk_writer(flushed)

            flushed_content = content_leak_filter.flush()
            if flushed_content:
                yield formatter.content_chunk(flushed_content)
            flushed_thinking = thinking_leak_filter.flush()
            if flushed_thinking:
                yield formatter.thinking_chunk(flushed_thinking)

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


def should_force_finalize_without_tools(chat_req: ChatCompletionsRequest) -> bool:
    if not should_append_tool_finalize_hint(chat_req):
        return False

    tool_message_count = sum(1 for message in chat_req.messages if message.role.lower() == "tool")
    user_message_count = sum(1 for message in chat_req.messages if message.role.lower() == "user")
    assistant_message_count = sum(1 for message in chat_req.messages if message.role.lower() == "assistant")

    return tool_message_count >= 50 and user_message_count <= 2 and assistant_message_count >= 15


def normalize_model_self_identification(
    messages: list[ChatMessage],
    requested_model: str,
) -> list[ChatMessage]:
    normalized_messages: list[ChatMessage] = []
    for message in messages:
        if message.role.lower() != "system" or not isinstance(message.content, str):
            normalized_messages.append(message)
            continue

        updated_content = MODEL_SELF_IDENTIFICATION_PATTERN.sub(
            lambda match: f"{match.group(1)}{requested_model}",
            message.content,
        )
        normalized_messages.append(
            message.model_copy(update={"content": updated_content})
            if updated_content != message.content
            else message
        )
    return normalized_messages


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
        "truncation",
    }
)


def normalize_responses_body_for_backend(
    request_body: bytes,
    *,
    default_effort: str,
    default_instructions: str | None,
    base_models: list[str] | None = None,
) -> bytes:
    try:
        payload = json.loads(request_body)
    except (TypeError, ValueError):
        return request_body

    if not isinstance(payload, dict):
        return request_body

    updated_payload = dict(payload)

    if default_instructions is not None and (
        not isinstance(updated_payload.get("instructions"), str)
        or not updated_payload.get("instructions", "").strip()
    ):
        updated_payload["instructions"] = default_instructions

    reasoning_effort = updated_payload.pop("reasoning_effort", None)

    model = updated_payload.get("model")
    if isinstance(model, str):
        backend_model, model_effort = resolve_model_alias(model, base_models)
        updated_payload["model"] = backend_model
        if model_effort is not None:
            _, resolved_reasoning = resolve_model_and_reasoning(
                model,
                updated_payload.get("reasoning"),
                reasoning_effort if isinstance(reasoning_effort, str) else None,
                base_models,
            )
            if resolved_reasoning is not None:
                updated_payload["reasoning"] = resolved_reasoning
            else:
                updated_payload.pop("reasoning", None)
        elif isinstance(reasoning_effort, str) and "reasoning" not in updated_payload:
            updated_payload["reasoning"] = {"effort": reasoning_effort}
    elif isinstance(reasoning_effort, str) and "reasoning" not in updated_payload:
        updated_payload["reasoning"] = {"effort": reasoning_effort}

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


def extract_responses_model(request_body: bytes) -> str | None:
    try:
        payload = json.loads(request_body)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


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
