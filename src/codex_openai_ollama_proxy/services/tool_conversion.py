from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import Any
from uuid import uuid4

from codex_openai_ollama_proxy.schemas.openai import ChatMessage, ChatMessageToolCall


@dataclass(slots=True)
class PendingToolCall:
    call_id: str
    name: str


def normalize_function_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, separators=(",", ":"))


def parse_function_arguments(arguments: str) -> Any:
    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return arguments


def convert_chat_tool_call_to_ollama(tool_call: Any, *, index: int | None = None) -> dict[str, Any]:
    function_payload: dict[str, Any] = {
            "name": tool_call.function.name,
            "arguments": parse_function_arguments(tool_call.function.arguments),
    }
    if index is not None:
        function_payload["index"] = index

    payload: dict[str, Any] = {"function": function_payload}
    if tool_call.id is not None:
        payload["id"] = tool_call.id
    return payload

def convert_chat_tool_calls_to_ollama(tool_calls: list[Any] | None) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None
    return [
        convert_chat_tool_call_to_ollama(tool_call, index=index)
        for index, tool_call in enumerate(tool_calls)
    ]


def convert_chat_tools_to_responses(tools: list[Any] | None) -> list[Any]:
    converted_tools: list[Any] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            converted_tools.append(tool)
            continue

        function_obj = tool.get("function") if isinstance(tool.get("function"), dict) else None
        name = (function_obj or {}).get("name") or tool.get("name")
        if name is None:
            converted_tools.append(tool)
            continue

        converted: dict[str, Any] = {"type": "function", "name": name}
        description = (function_obj or {}).get("description") or tool.get("description")
        parameters = (function_obj or {}).get("parameters") or tool.get("parameters")
        strict = (function_obj or {}).get("strict") or tool.get("strict")
        if description is not None:
            converted["description"] = description
        if parameters is not None:
            converted["parameters"] = parameters
        if strict is not None:
            converted["strict"] = strict
        converted_tools.append(converted)

    return converted_tools


def convert_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        function_obj = tool_choice.get("function")
        if name is None and isinstance(function_obj, dict):
            name = function_obj.get("name")
        if name is not None:
            return {"type": "function", "name": name}
    return tool_choice if tool_choice is not None else "auto"


def assistant_tool_calls_to_input(
    tool_calls: list[ChatMessageToolCall] | None,
    pending_calls: deque[PendingToolCall],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if tool_call.call_type and tool_call.call_type.lower() != "function":
            continue
        call_id = tool_call.id or f"call_{uuid4()}"
        pending_calls.append(PendingToolCall(call_id=call_id, name=tool_call.function.name))
        items.append(
            {
                "type": "function_call",
                "id": None,
                "call_id": call_id,
                "name": tool_call.function.name,
                "arguments": normalize_function_arguments(tool_call.function.arguments),
            }
        )
    return items


def _pop_pending_by_call_id(
    pending_calls: deque[PendingToolCall], call_id: str
) -> PendingToolCall | None:
    for pending_call in pending_calls:
        if pending_call.call_id == call_id:
            pending_calls.remove(pending_call)
            return pending_call
    return None


def _pop_pending_by_tool_name(
    pending_calls: deque[PendingToolCall], tool_name: str
) -> PendingToolCall | None:
    for pending_call in pending_calls:
        if pending_call.name == tool_name:
            pending_calls.remove(pending_call)
            return pending_call
    return None


def tool_message_to_output(
    message: ChatMessage,
    pending_calls: deque[PendingToolCall],
    output: str,
) -> dict[str, Any]:
    pending_call: PendingToolCall | None = None
    tool_name = message.tool_name

    if message.tool_call_id is not None:
        pending_call = _pop_pending_by_call_id(pending_calls, message.tool_call_id)
    elif tool_name is not None:
        pending_call = _pop_pending_by_tool_name(pending_calls, tool_name)
    elif pending_calls:
        pending_call = pending_calls.popleft()

    call_id = (
        message.tool_call_id
        or (pending_call.call_id if pending_call is not None else None)
    )
    if call_id is None:
        call_id = f"call_{uuid4()}"

    if tool_name is None and pending_call is not None:
        tool_name = pending_call.name

    payload = {"type": "function_call_output", "call_id": call_id, "output": output}
    if tool_name is not None:
        payload["name"] = tool_name
    return payload
