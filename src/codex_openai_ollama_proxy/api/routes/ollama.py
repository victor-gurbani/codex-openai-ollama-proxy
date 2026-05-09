from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codex_openai_ollama_proxy.api.deps import get_model_catalog, get_proxy_service, get_settings
from codex_openai_ollama_proxy.core.config import Settings
from codex_openai_ollama_proxy.core.debug_trace import (
    finish_debug_request,
    log_debug_event,
    start_debug_request,
)
from codex_openai_ollama_proxy.core.errors import ollama_error_response
from codex_openai_ollama_proxy.schemas.ollama import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
)
from codex_openai_ollama_proxy.services.model_catalog import ModelCatalogService
from codex_openai_ollama_proxy.services.proxy_service import ProxyService, normalize_ollama_model
from codex_openai_ollama_proxy.services.streaming_formatter import build_ollama_error_ndjson
from codex_openai_ollama_proxy.services.tool_conversion import convert_chat_tool_calls_to_ollama

router = APIRouter(tags=["ollama"])


def parse_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def metadata_context_length(metadata: dict[str, object] | None) -> int | None:
    if metadata is None:
        return None
    for key in ("context_window", "context_length", "max_context_window"):
        parsed = parse_positive_int(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def ollama_model_details(model: str) -> dict[str, object]:
    family = "codex" if "codex" in model else "gpt"
    return {
        "parent_model": "",
        "format": "proxy",
        "family": family,
        "families": [family],
        "parameter_size": "unknown",
        "quantization_level": "unknown",
    }


def ollama_show_payload(
    model: str, metadata: dict[str, object] | None = None
) -> dict[str, object]:
    details = ollama_model_details(model)
    model_info: dict[str, object] = {
        "general.architecture": details["family"],
        "general.basename": model,
        "general.file_type": 0,
        "general.parameter_count": 0,
        "general.quantization_version": 0,
        "codex.proxy.format": details["format"],
        "codex.proxy.variant": "synthetic-metadata",
    }
    context_length = metadata_context_length(metadata)
    if context_length is not None:
        model_info["general.context_length"] = context_length
    return {
        "license": "",
        "modelfile": f"FROM {model}\n",
        "parameters": "",
        "template": "",
        "details": details,
        "model_info": model_info,
        "capabilities": ["completion", "tools", "thinking"],
        "modified_at": "1970-01-01T00:00:00.000Z",
        "requires": "0.17.1",
        "tensors": [],
    }


def ollama_not_found_response(model: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        headers={"Access-Control-Allow-Origin": "*"},
        content={"error": f"model '{model}' not found"},
    )


@router.get("/api/tags")
async def ollama_tags(
    settings: Settings = Depends(get_settings),
    model_catalog: ModelCatalogService = Depends(get_model_catalog),
) -> dict[str, object]:
    models = await model_catalog.get_exposed_models()
    return {
        "models": [
            {
                "name": model,
                "model": model,
                "modified_at": "1970-01-01T00:00:00.000Z",
                "size": 0,
                "digest": "",
                "details": ollama_model_details(model),
            }
            for model in models
        ]
    }


@router.post("/api/show")
async def ollama_show(
    request: OllamaShowRequest,
    model_catalog: ModelCatalogService = Depends(get_model_catalog),
) -> JSONResponse:
    normalized_model = normalize_ollama_model(request.model)
    exposed_models = await model_catalog.get_exposed_models()
    if normalized_model not in exposed_models:
        return ollama_not_found_response(normalized_model)
    metadata = await model_catalog.get_model_metadata_for_request(normalized_model)
    return JSONResponse(content=ollama_show_payload(normalized_model, metadata))


@router.post("/api/chat")
async def ollama_chat(
    request: OllamaChatRequest,
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
                    async for chunk in await proxy_service.stream_ollama_chat(
                        request,
                        is_disconnected=raw_request.is_disconnected,
                    ):
                        emitted_chunks.append(chunk)
                        yield chunk
                except Exception as exc:  # noqa: BLE001
                    chunk = build_ollama_error_ndjson(f"proxy error: {exc}")
                    emitted_chunks.append(chunk)
                    yield chunk
            finally:
                log_debug_event(
                    "client_response",
                    status_code=200,
                    media_type="application/x-ndjson",
                    body="".join(emitted_chunks),
                )
                finish_debug_request(debug_tokens)

        return StreamingResponse(
            stream_with_error_fallback(),
            media_type="application/x-ndjson",
        )

    try:
        response = await proxy_service.proxy_ollama_chat(request)
    except Exception as exc:  # noqa: BLE001
        error_response = ollama_error_response(exc)
        log_debug_event(
            "client_response",
            status_code=error_response.status_code,
            media_type="application/json",
            body=error_response.body,
        )
        finish_debug_request(debug_tokens)
        return error_response

    content = response.choices[0].message.content
    payload = {
        "model": response.model,
        "created_at": "1970-01-01T00:00:00.000Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
        "prompt_eval_duration": 0,
        "eval_count": response.usage.completion_tokens if response.usage else 0,
        "eval_duration": 0,
    }
    if response.choices[0].message.tool_calls:
        payload["message"]["tool_calls"] = convert_chat_tool_calls_to_ollama(
            response.choices[0].message.tool_calls
        )
    log_debug_event(
        "client_response",
        status_code=200,
        media_type="application/json",
        body=payload,
    )
    finish_debug_request(debug_tokens)
    return JSONResponse(content=payload)


@router.post("/api/generate")
async def ollama_generate(
    request: OllamaGenerateRequest,
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
                    async for chunk in await proxy_service.stream_ollama_generate(
                        request,
                        is_disconnected=raw_request.is_disconnected,
                    ):
                        emitted_chunks.append(chunk)
                        yield chunk
                except Exception as exc:  # noqa: BLE001
                    chunk = build_ollama_error_ndjson(f"proxy error: {exc}")
                    emitted_chunks.append(chunk)
                    yield chunk
            finally:
                log_debug_event(
                    "client_response",
                    status_code=200,
                    media_type="application/x-ndjson",
                    body="".join(emitted_chunks),
                )
                finish_debug_request(debug_tokens)

        return StreamingResponse(
            stream_with_error_fallback(),
            media_type="application/x-ndjson",
        )

    try:
        response = await proxy_service.proxy_ollama_generate(request)
    except Exception as exc:  # noqa: BLE001
        error_response = ollama_error_response(exc)
        log_debug_event(
            "client_response",
            status_code=error_response.status_code,
            media_type="application/json",
            body=error_response.body,
        )
        finish_debug_request(debug_tokens)
        return error_response

    content = response.choices[0].message.content
    payload = {
        "model": response.model,
        "created_at": "1970-01-01T00:00:00.000Z",
        "response": content,
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
    log_debug_event(
        "client_response",
        status_code=200,
        media_type="application/json",
        body=payload,
    )
    finish_debug_request(debug_tokens)
    return JSONResponse(content=payload)
