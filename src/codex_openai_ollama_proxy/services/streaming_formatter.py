from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from codex_openai_ollama_proxy.schemas.events import ToolCallChunkEvent
from codex_openai_ollama_proxy.schemas.openai import ChatCompletionsResponse, ChatToolCall
from codex_openai_ollama_proxy.schemas.usage import Usage
from codex_openai_ollama_proxy.services.tool_conversion import convert_chat_tool_calls_to_ollama


def build_openai_sse_response(
    model: str,
    message: str,
    tool_calls: list[ChatToolCall] | None,
    finish_reason: str,
    usage: Usage | None,
) -> str:
    chunk_id = f"chatcmpl-{uuid4()}"
    created = int(datetime.now(UTC).timestamp())
    system_fingerprint = "fp_ollama"

    role_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }

    chunks = [f"data: {json.dumps(role_chunk, separators=(',', ':'))}\n\n"]

    if message:
        content_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(content_chunk, separators=(',', ':'))}\n\n")

    if tool_calls:
        tool_calls_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [{"index": 0, "delta": {"tool_calls": [tool.model_dump(by_alias=True) for tool in tool_calls]}, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(tool_calls_chunk, separators=(',', ':'))}\n\n")

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }

    chunks.append(f"data: {json.dumps(final_chunk, separators=(',', ':'))}\n\n")
    if usage is not None:
        usage_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [],
            "usage": usage.model_dump(),
        }
        chunks.append(f"data: {json.dumps(usage_chunk, separators=(',', ':'))}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


class OpenAIStreamFormatter:
    def __init__(self, model: str) -> None:
        self.model = model
        self.chunk_id = f"chatcmpl-{uuid4()}"
        self.created = int(datetime.now(UTC).timestamp())
        self.system_fingerprint = "fp_ollama"

    def role_chunk(self) -> str:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    def content_chunk(self, text: str) -> str:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    def reasoning_chunk(self, text: str) -> str:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "", "reasoning": text},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    def tool_call_chunk(self, event: ToolCallChunkEvent) -> str:
        function_payload: dict[str, object] = {}
        if event.name is not None:
            function_payload["name"] = event.name
        if event.arguments_delta or event.is_final or event.name is not None:
            function_payload["arguments"] = (
                event.arguments_delta if event.arguments_delta else event.arguments
            )

        tool_call_payload: dict[str, object] = {"index": event.index, "function": function_payload}
        if event.name is not None:
            tool_call_payload["id"] = event.tool_call_id
            tool_call_payload["type"] = "function"

        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call_payload],
                    },
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    def tool_calls_chunk(self, tool_calls: list[ChatToolCall]) -> str:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "index": index,
                                **tool_call.model_dump(by_alias=True),
                            }
                            for index, tool_call in enumerate(tool_calls)
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    def final_chunk(self, finish_reason: str) -> str:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    def usage_chunk(self, usage: Usage) -> str:
        payload = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [],
            "usage": usage.model_dump(),
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    @staticmethod
    def done_chunk() -> str:
        return "data: [DONE]\n\n"

    @staticmethod
    def heartbeat_chunk() -> str:
        return ": keep-alive\n\n"


class OllamaStreamFormatter:
    def __init__(self, model: str, mode: str) -> None:
        self.model = model
        self.mode = mode

    def content_chunk(self, text: str) -> str:
        created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if self.mode == "chat":
            payload = {
                "model": self.model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": text},
                "done": False,
            }
        else:
            payload = {
                "model": self.model,
                "created_at": created_at,
                "response": text,
                "done": False,
            }
        return json.dumps(payload, separators=(",", ":")) + "\n"

    def thinking_chunk(self, text: str) -> str:
        created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if self.mode == "chat":
            payload = {
                "model": self.model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": "", "thinking": text},
                "done": False,
            }
        else:
            payload = {
                "model": self.model,
                "created_at": created_at,
                "response": "",
                "thinking": text,
                "done": False,
            }
        return json.dumps(payload, separators=(",", ":")) + "\n"

    def tool_calls_chunk(self, tool_calls: list[ChatToolCall]) -> str:
        created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        payload = {
            "model": self.model,
            "created_at": created_at,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": convert_chat_tool_calls_to_ollama(tool_calls),
            },
            "done": False,
        }
        return json.dumps(payload, separators=(",", ":")) + "\n"

    def tool_call_snapshot_chunk(self, tool_calls: list[ChatToolCall]) -> str:
        return self.tool_calls_chunk(tool_calls)

    def final_chunk(self, usage: Usage | None) -> str:
        created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if self.mode == "chat":
            payload = {
                "model": self.model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": usage.prompt_tokens if usage else 0,
                "prompt_eval_duration": 0,
                "eval_count": usage.completion_tokens if usage else 0,
                "eval_duration": 0,
            }
        else:
            payload = {
                "model": self.model,
                "created_at": created_at,
                "response": "",
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": usage.prompt_tokens if usage else 0,
                "prompt_eval_duration": 0,
                "eval_count": usage.completion_tokens if usage else 0,
                "eval_duration": 0,
            }
        return json.dumps(payload, separators=(",", ":")) + "\n"

    def heartbeat_chunk(self) -> str:
        created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if self.mode == "chat":
            payload = {
                "model": self.model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": False,
            }
        else:
            payload = {
                "model": self.model,
                "created_at": created_at,
                "response": "",
                "done": False,
            }
        return json.dumps(payload, separators=(",", ":")) + "\n"


def build_openai_sse_from_response(response: ChatCompletionsResponse) -> str:
    choice = response.choices[0]
    return build_openai_sse_response(
        model=response.model,
        message=choice.message.content,
        tool_calls=choice.message.tool_calls,
        finish_reason=choice.finish_reason or "stop",
        usage=response.usage,
    )


def build_openai_error_sse(model: str, message: str) -> str:
    return build_openai_sse_response(
        model=model,
        message=message,
        tool_calls=None,
        finish_reason="stop",
        usage=None,
    )


def build_ollama_chat_ndjson(response: ChatCompletionsResponse) -> str:
    content = response.choices[0].message.content
    thinking = response.choices[0].message.reasoning
    tool_calls = response.choices[0].message.tool_calls
    created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    message: dict[str, object] = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = convert_chat_tool_calls_to_ollama(tool_calls)
    content_chunk: dict[str, object] = {
        "model": response.model,
        "created_at": created_at,
        "message": message,
        "done": False,
    }
    done_chunk = {
        "model": response.model,
        "created_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
        "prompt_eval_duration": 0,
        "eval_count": response.usage.completion_tokens if response.usage else 0,
        "eval_duration": 0,
    }
    return "\n".join(
        [
            json.dumps(content_chunk, separators=(",", ":")),
            json.dumps(done_chunk, separators=(",", ":")),
            "",
        ]
    )


def build_ollama_generate_ndjson(response: ChatCompletionsResponse) -> str:
    content = response.choices[0].message.content
    thinking = response.choices[0].message.reasoning
    created_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    content_chunk = {
        "model": response.model,
        "created_at": created_at,
        "response": content,
        "done": False,
    }
    if thinking:
        content_chunk["thinking"] = thinking
    done_chunk = {
        "model": response.model,
        "created_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "response": "",
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
        "prompt_eval_duration": 0,
        "eval_count": response.usage.completion_tokens if response.usage else 0,
        "eval_duration": 0,
    }
    return "\n".join(
        [
            json.dumps(content_chunk, separators=(",", ":")),
            json.dumps(done_chunk, separators=(",", ":")),
            "",
        ]
    )


def build_ollama_error_ndjson(message: str) -> str:
    return json.dumps({"error": message}, separators=(",", ":")) + "\n"
