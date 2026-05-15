from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
import json
from typing import Any
from uuid import uuid4

from codex_openai_ollama_proxy.core.errors import BackendSSEError, EmptyBackendResponseError
from codex_openai_ollama_proxy.schemas.events import (
    ErrorEvent,
    StreamEvent,
    ThinkingDeltaEvent,
    ThinkingDoneEvent,
    TextDeltaEvent,
    TextDoneEvent,
    ToolCallChunkEvent,
    UsageEvent,
)
from codex_openai_ollama_proxy.schemas.usage import Usage
from codex_openai_ollama_proxy.services.stream_state import StreamState
from codex_openai_ollama_proxy.services.usage_extraction import extract_usage_from_event


TEXT_DELTA_EVENT_TYPES = {"response.output_text.delta"}
TEXT_DONE_EVENT_TYPES = {"response.output_text.done"}
THINKING_DELTA_EVENT_TYPES = {
    "response.reasoning_summary_text.delta",
    "response.reasoning_text.delta",
}
THINKING_DONE_EVENT_TYPES = {
    "response.reasoning_summary_text.done",
    "response.reasoning_text.done",
}
THINKING_SUMMARY_PART_DONE_EVENT_TYPES = {"response.reasoning_summary_part.done"}
FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES = {
    "response.function_call_arguments.delta",
    "response.function_call.arguments.delta",
    "response.function_call.delta",
    "response.tool_call_arguments.delta",
    "response.tool_call.arguments.delta",
    "response.tool_call.delta",
}
FUNCTION_CALL_ARGUMENTS_DONE_EVENT_TYPES = {
    "response.function_call_arguments.done",
    "response.function_call.arguments.done",
    "response.function_call.done",
    "response.tool_call_arguments.done",
    "response.tool_call.arguments.done",
    "response.tool_call.done",
}


@dataclass(slots=True)
class FunctionCallState:
    item_id: str
    index: int
    tool_call_id: str
    name: str
    arguments: str = ""


class BackendEventParser:
    def __init__(self) -> None:
        self._function_calls: dict[str, FunctionCallState] = {}
        self._next_tool_index = 0

    @staticmethod
    def _status_code_from_error(
        *,
        error_type: str | None,
        code: str | None,
        message: str,
        status_code: int | None,
    ) -> int:
        if status_code is not None and 400 <= status_code <= 599:
            return status_code
        normalized_type = (error_type or "").strip().lower()
        normalized_code = (code or "").strip().lower()
        normalized_message = message.strip().lower()
        if normalized_type == "invalid_request_error" or normalized_code == "context_length_exceeded":
            return 400
        if "context window" in normalized_message or "input exceeds the context window" in normalized_message:
            return 400
        if normalized_type in {"authentication_error", "invalid_api_key_error"}:
            return 401
        if normalized_code in {"rate_limit_exceeded", "insufficient_quota"}:
            return 429
        return 502

    def _parse_error_event(
        self, event: dict[str, Any], *, sse_event_name: str | None = None
    ) -> ErrorEvent | None:
        event_type = event.get("type") or sse_event_name
        error_type: str | None = None
        code: str | None = None
        param: str | None = None
        message: str | None = None
        status_code: int | None = parse_optional_int(event.get("status_code"))

        error_obj = event.get("error")
        if isinstance(error_obj, str):
            message = error_obj
        elif isinstance(error_obj, dict):
            message = error_obj.get("message") if isinstance(error_obj.get("message"), str) else None
            error_type = error_obj.get("type") if isinstance(error_obj.get("type"), str) else None
            code = error_obj.get("code") if isinstance(error_obj.get("code"), str) else None
            param = error_obj.get("param") if isinstance(error_obj.get("param"), str) else None
            status_code = parse_optional_int(error_obj.get("status_code")) or status_code

        response_obj = event.get("response")
        if isinstance(response_obj, dict):
            response_error = response_obj.get("error")
            if isinstance(response_error, dict):
                message = message or (
                    response_error.get("message")
                    if isinstance(response_error.get("message"), str)
                    else None
                )
                error_type = error_type or (
                    response_error.get("type")
                    if isinstance(response_error.get("type"), str)
                    else None
                )
                code = code or (
                    response_error.get("code")
                    if isinstance(response_error.get("code"), str)
                    else None
                )
                param = param or (
                    response_error.get("param")
                    if isinstance(response_error.get("param"), str)
                    else None
                )
                status_code = parse_optional_int(response_error.get("status_code")) or status_code

        if error_type is None and isinstance(event.get("type"), str):
            event_type_value = event.get("type")
            if event_type_value != "response.failed":
                error_type = event_type_value

        if code is None and isinstance(event.get("code"), str):
            code = event.get("code")

        if param is None and isinstance(event.get("param"), str):
            param = event.get("param")

        if message is None:
            event_message = event.get("message")
            if isinstance(event_message, str):
                message = event_message

        if message is None and event_type in {"error", "response.failed"}:
            message = json.dumps(event, ensure_ascii=False, separators=(",", ":"))

        if message is None:
            return None

        return ErrorEvent(
            message=message,
            status_code=self._status_code_from_error(
                error_type=error_type,
                code=code,
                message=message,
                status_code=status_code,
            ),
            error_type=error_type,
            param=param,
            code=code,
        )

    def parse_event(
        self, event: dict[str, Any], *, sse_event_name: str | None = None
    ) -> list[StreamEvent]:
        parsed_events: list[StreamEvent] = []

        parsed_usage = extract_usage_from_event(event)
        if parsed_usage is not None:
            parsed_events.append(UsageEvent(parsed_usage))

        error_event = self._parse_error_event(event, sse_event_name=sse_event_name)
        if error_event is not None:
            parsed_events.append(error_event)
            return parsed_events

        event_type = event.get("type") or sse_event_name
        if event_type in TEXT_DELTA_EVENT_TYPES:
            delta = event.get("delta")
            if isinstance(delta, str):
                parsed_events.append(TextDeltaEvent(delta))
            return parsed_events

        if event_type in TEXT_DONE_EVENT_TYPES:
            text = event.get("text")
            if isinstance(text, str):
                parsed_events.append(TextDoneEvent(text))
            return parsed_events

        if event_type in THINKING_DELTA_EVENT_TYPES:
            delta = event.get("delta")
            if isinstance(delta, str):
                parsed_events.append(ThinkingDeltaEvent(delta))
            return parsed_events

        if event_type in THINKING_DONE_EVENT_TYPES:
            text = event.get("text")
            if isinstance(text, str):
                parsed_events.append(ThinkingDoneEvent(text))
            return parsed_events

        if event_type in THINKING_SUMMARY_PART_DONE_EVENT_TYPES:
            text = extract_text_from_part(event.get("part"))
            if text is not None:
                parsed_events.append(ThinkingDoneEvent(text))
            return parsed_events

        if event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict):
                tool_event = self._parse_function_call_added(item)
                if tool_event is not None:
                    parsed_events.append(tool_event)
            return parsed_events

        if event_type in FUNCTION_CALL_ARGUMENTS_DELTA_EVENT_TYPES:
            item_id = event.get("item_id")
            delta = event.get("delta") or event.get("arguments_delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                tool_event = self._parse_function_call_delta(item_id, delta)
                if tool_event is not None:
                    parsed_events.append(tool_event)
            return parsed_events

        if event_type in FUNCTION_CALL_ARGUMENTS_DONE_EVENT_TYPES:
            item_id = event.get("item_id")
            name = event.get("name")
            arguments = event.get("arguments")
            call_id = event.get("call_id")
            if isinstance(item_id, str):
                tool_event = self._parse_function_call_arguments_done(
                    item_id,
                    name if isinstance(name, str) else None,
                    arguments if isinstance(arguments, str) else "",
                    call_id if isinstance(call_id, str) else None,
                )
                if tool_event is not None:
                    parsed_events.append(tool_event)
            return parsed_events

        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                tool_event = self._parse_function_call_done(item)
                if tool_event is not None:
                    parsed_events.append(tool_event)
                    return parsed_events

                reasoning_text = self._parse_reasoning_done(item)
                if reasoning_text is not None:
                    parsed_events.append(ThinkingDoneEvent(reasoning_text))
                    return parsed_events

                content_array = item.get("content")
                if isinstance(content_array, list):
                    text_parts: list[str] = []
                    for content_item in content_array:
                        if isinstance(content_item, dict):
                            text = content_item.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    if text_parts:
                        parsed_events.append(TextDoneEvent("".join(text_parts)))

        return parsed_events

    @staticmethod
    def _parse_reasoning_done(item: dict[str, Any]) -> str | None:
        if item.get("type") != "reasoning":
            return None

        summary = item.get("summary")
        if not isinstance(summary, list):
            return None

        text_parts: list[str] = []
        for summary_item in summary:
            if isinstance(summary_item, dict):
                text = summary_item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

        combined = "".join(text_parts)
        return combined or None

    def _parse_function_call_added(self, item: dict[str, Any]) -> ToolCallChunkEvent | None:
        if item.get("type") != "function_call":
            return None

        item_id = item.get("id")
        name = item.get("name")
        if not isinstance(item_id, str) or not isinstance(name, str):
            return None

        call_id = item.get("call_id") or item_id or f"call_{uuid4()}"
        if not isinstance(call_id, str):
            call_id = f"call_{uuid4()}"

        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            arguments = ""

        state = FunctionCallState(
            item_id=item_id,
            index=self._next_tool_index,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
        )
        self._function_calls[item_id] = state
        self._next_tool_index += 1

        return ToolCallChunkEvent(
            item_id=item_id,
            index=state.index,
            tool_call_id=state.tool_call_id,
            name=state.name,
            arguments=state.arguments,
        )

    def _parse_function_call_delta(
        self, item_id: str, delta: str
    ) -> ToolCallChunkEvent | None:
        state = self._function_calls.get(item_id)
        if state is None:
            return None

        state.arguments += delta
        return ToolCallChunkEvent(
            item_id=item_id,
            index=state.index,
            tool_call_id=state.tool_call_id,
            name=None,
            arguments=state.arguments,
            arguments_delta=delta,
        )

    def _parse_function_call_done(self, item: dict[str, Any]) -> ToolCallChunkEvent | None:
        if item.get("type") != "function_call":
            return None

        item_id = item.get("id")
        name = item.get("name")
        if not isinstance(item_id, str) or not isinstance(name, str):
            return None

        call_id = item.get("call_id") or item_id or f"call_{uuid4()}"
        if not isinstance(call_id, str):
            call_id = f"call_{uuid4()}"

        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            arguments = ""

        state = self._function_calls.get(item_id)
        if state is None:
            state = FunctionCallState(
                item_id=item_id,
                index=self._next_tool_index,
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
            )
            self._function_calls[item_id] = state
            self._next_tool_index += 1
        else:
            state.tool_call_id = call_id
            state.name = name
            state.arguments = arguments

        return ToolCallChunkEvent(
            item_id=item_id,
            index=state.index,
            tool_call_id=state.tool_call_id,
            name=state.name,
            arguments=state.arguments,
            is_final=True,
        )

    def _parse_function_call_arguments_done(
        self,
        item_id: str,
        name: str | None,
        arguments: str,
        call_id: str | None,
    ) -> ToolCallChunkEvent | None:
        state = self._function_calls.get(item_id)
        if state is None:
            if name is None:
                return None
            state = FunctionCallState(
                item_id=item_id,
                index=self._next_tool_index,
                tool_call_id=call_id or item_id,
                name=name,
                arguments=arguments,
            )
            self._function_calls[item_id] = state
            self._next_tool_index += 1
        else:
            if call_id is not None:
                state.tool_call_id = call_id
            if name is not None:
                state.name = name
            state.arguments = arguments

        return ToolCallChunkEvent(
            item_id=item_id,
            index=state.index,
            tool_call_id=state.tool_call_id,
            name=state.name,
            arguments=state.arguments,
            is_final=True,
        )


def parse_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def extract_text_from_part(part: Any) -> str | None:
    if not isinstance(part, dict):
        return None
    text = part.get("text")
    return text if isinstance(text, str) else None


def iter_events_from_sse_lines(lines: Iterable[str]) -> list[StreamEvent]:
    parser = BackendEventParser()
    events: list[StreamEvent] = []
    pending_event_name: str | None = None
    for line in lines:
        if line.startswith("event: "):
            pending_event_name = line[7:].strip() or None
            continue
        if not line.startswith("data: "):
            continue
        json_data = line[6:]
        if json_data == "[DONE]":
            break
        try:
            payload = json.loads(json_data)
        except json.JSONDecodeError:
            pending_event_name = None
            continue
        if isinstance(payload, dict):
            events.extend(parser.parse_event(payload, sse_event_name=pending_event_name))
        pending_event_name = None
    return events


async def stream_events_from_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[StreamEvent]:
    parser = BackendEventParser()
    pending_event_name: str | None = None
    async for line in lines:
        if line.startswith("event: "):
            pending_event_name = line[7:].strip() or None
            continue
        if not line.startswith("data: "):
            continue
        json_data = line[6:]
        if json_data == "[DONE]":
            break
        try:
            payload = json.loads(json_data)
        except json.JSONDecodeError:
            pending_event_name = None
            continue
        if not isinstance(payload, dict):
            pending_event_name = None
            continue
        for event in parser.parse_event(payload, sse_event_name=pending_event_name):
            yield event
        pending_event_name = None


def parse_backend_sse_text(
    response_text: str,
    *,
    allow_thinking_only: bool = False,
) -> tuple[str, str, list[Any], Usage | None]:
    state = StreamState()
    for event in iter_events_from_sse_lines(response_text.splitlines()):
        if isinstance(event, ErrorEvent):
            raise BackendSSEError(
                event.message,
                status_code=event.status_code,
                error_type=event.error_type,
                param=event.param,
                code=event.code,
            )
        state.apply(event)

    has_visible_output = (
        state.has_any_output if allow_thinking_only else state.has_visible_openai_output
    )

    if not has_visible_output:
        raise EmptyBackendResponseError(
            "Empty content and no tool calls returned from ChatGPT backend"
        )

    return state.text, state.thinking, list(state.tool_calls), state.usage
