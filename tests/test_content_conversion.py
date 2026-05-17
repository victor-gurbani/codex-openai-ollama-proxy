from codex_openai_ollama_proxy.schemas.openai import ChatMessage, ChatMessageToolCall, ChatMessageToolFunction
from codex_openai_ollama_proxy.services.content_conversion import (
    convert_messages_to_input,
    parse_chat_content_items,
)


def test_parse_image_url_content() -> None:
    items = parse_chat_content_items(
        [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ],
        is_assistant=False,
    )

    assert {"type": "input_text", "text": "Describe this image"} in items
    assert {
        "type": "input_image",
        "image_url": "https://example.com/cat.png",
        "detail": "high",
    } in items


def test_parse_base64_image_content() -> None:
    items = parse_chat_content_items(
        [{"type": "input_image", "image_base64": "QUJD", "mime_type": "image/jpeg"}],
        is_assistant=False,
    )

    assert items == [{"type": "input_image", "image_url": "data:image/jpeg;base64,QUJD"}]


def test_convert_messages_to_input_infers_tool_call_id() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_from_assistant",
                    type="function",
                    function=ChatMessageToolFunction(name="list_dir", arguments={"path": "."}),
                )
            ],
        ),
        ChatMessage(role="tool", content="[]"),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    assert any(
        item["type"] == "function_call_output" and item["call_id"] == "call_from_assistant"
        for item in input_items
    )
    assert instructions


def test_convert_messages_to_input_prefers_tool_name_for_matching_outputs() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_temperature",
                    type="function",
                    function=ChatMessageToolFunction(
                        name="get_temperature", arguments={"city": "New York"}
                    ),
                ),
                ChatMessageToolCall(
                    id="call_conditions",
                    type="function",
                    function=ChatMessageToolFunction(
                        name="get_conditions", arguments={"city": "New York"}
                    ),
                ),
            ],
        ),
        ChatMessage(role="tool", tool_name="get_conditions", content="Partly cloudy"),
        ChatMessage(role="tool", tool_name="get_temperature", content="22°C"),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    tool_outputs = [item for item in input_items if item["type"] == "function_call_output"]
    assert tool_outputs == [
        {
            "type": "function_call_output",
            "call_id": "call_conditions",
            "name": "get_conditions",
            "output": "Partly cloudy",
        },
        {
            "type": "function_call_output",
            "call_id": "call_temperature",
            "name": "get_temperature",
            "output": "22°C",
        },
    ]
    assert instructions


def test_convert_messages_to_input_tool_call_id_overrides_tool_name() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_a",
                    type="function",
                    function=ChatMessageToolFunction(name="lookup", arguments={"id": 1}),
                ),
                ChatMessageToolCall(
                    id="call_b",
                    type="function",
                    function=ChatMessageToolFunction(name="lookup", arguments={"id": 2}),
                ),
            ],
        ),
        ChatMessage(
            role="tool",
            tool_name="lookup",
            tool_call_id="call_b",
            content="result for b",
        ),
        ChatMessage(role="tool", content="result for a"),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    tool_outputs = [item for item in input_items if item["type"] == "function_call_output"]
    assert tool_outputs == [
        {
            "type": "function_call_output",
            "call_id": "call_b",
            "name": "lookup",
            "output": "result for b",
        },
        {
            "type": "function_call_output",
            "call_id": "call_a",
            "name": "lookup",
            "output": "result for a",
        },
    ]
    assert instructions


def test_convert_messages_to_input_tool_name_handles_duplicate_calls_fifo() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_first",
                    type="function",
                    function=ChatMessageToolFunction(name="lookup", arguments={"id": 1}),
                ),
                ChatMessageToolCall(
                    id="call_second",
                    type="function",
                    function=ChatMessageToolFunction(name="lookup", arguments={"id": 2}),
                ),
            ],
        ),
        ChatMessage(role="tool", tool_name="lookup", content="first result"),
        ChatMessage(role="tool", tool_name="lookup", content="second result"),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    tool_outputs = [item for item in input_items if item["type"] == "function_call_output"]
    assert tool_outputs == [
        {
            "type": "function_call_output",
            "call_id": "call_first",
            "name": "lookup",
            "output": "first result",
        },
        {
            "type": "function_call_output",
            "call_id": "call_second",
            "name": "lookup",
            "output": "second result",
        },
    ]
    assert instructions


def test_convert_messages_to_input_drops_unanswered_assistant_tool_call() -> None:
    messages = [
        ChatMessage(role="user", content="What is the latest inflation rate?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id=None,
                    type="function",
                    function=ChatMessageToolFunction(name="web_search", arguments={}),
                )
            ],
        ),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    assert input_items == [
        {
            "type": "message",
            "id": None,
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is the latest inflation rate?"}
            ],
        }
    ]
    assert instructions


def test_convert_messages_to_input_keeps_answered_tool_call_and_drops_unanswered_one() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_answered",
                    type="function",
                    function=ChatMessageToolFunction(name="lookup", arguments={"id": 1}),
                ),
                ChatMessageToolCall(
                    id="call_unanswered",
                    type="function",
                    function=ChatMessageToolFunction(name="lookup", arguments={"id": 2}),
                ),
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_answered", content="result"),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    assert input_items == [
        {
            "type": "function_call",
            "id": None,
            "call_id": "call_answered",
            "name": "lookup",
            "arguments": '{"id":1}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_answered",
            "name": "lookup",
            "output": "result",
        },
    ]
    assert instructions


def test_convert_messages_to_input_keeps_raycast_tool_output_without_ids() -> None:
    messages = [
        ChatMessage(role="user", content="get my location"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id=None,
                    type="function",
                    function=ChatMessageToolFunction(
                        name="location-get-current-location",
                        arguments={},
                    ),
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content={"city": "Madrid", "country": "Spain"},
        ),
    ]

    input_items, instructions = convert_messages_to_input(messages)

    assert len(input_items) == 3
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["name"] == "location-get-current-location"
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": input_items[1]["call_id"],
        "name": "location-get-current-location",
        "output": '{"city":"Madrid","country":"Spain"}',
    }
    assert instructions


def test_convert_messages_to_input_replays_assistant_thinking_before_content() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            thinking="I should inspect files first.",
            content="Calling the tool now.",
        )
    ]

    input_items, instructions = convert_messages_to_input(messages)

    assert input_items == [
        {
            "type": "message",
            "id": None,
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "I should inspect files first."},
                {"type": "output_text", "text": "Calling the tool now."},
            ],
        }
    ]
    assert instructions


def test_convert_messages_to_input_replays_openai_reasoning_before_content() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            reasoning="I should calculate before responding.",
            content="The answer is 5.",
        )
    ]

    input_items, instructions = convert_messages_to_input(messages)

    assert input_items == [
        {
            "type": "message",
            "id": None,
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "I should calculate before responding."},
                {"type": "output_text", "text": "The answer is 5."},
            ],
        }
    ]
    assert instructions
