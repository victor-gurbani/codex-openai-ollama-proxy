from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codex_openai_ollama_proxy.api.deps import get_model_catalog, get_proxy_service, get_settings
from codex_openai_ollama_proxy.core.config import Settings
from codex_openai_ollama_proxy.core.debug_trace import (
    finish_debug_request,
    log_debug_event,
    start_debug_request,
)
from codex_openai_ollama_proxy.core.errors import BackendSSEError, ollama_error_response
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

SHOW_CAPABILITIES = ["completion", "tools", "thinking"]


def normalize_family_name(value: str) -> str:
    normalized = "".join(ch for ch in value.strip().lower() if ch.isalnum())
    return normalized


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
    values: list[int] = []
    for key in ("context_window", "context_length", "max_context_window"):
        parsed = parse_positive_int(metadata.get(key))
        if parsed is not None:
            values.append(parsed)
    return max(values) if values else None


def infer_ollama_family(model: str, metadata: dict[str, object] | None) -> str:
    if metadata is not None:
        family = metadata.get("family")
        if isinstance(family, str) and family.strip():
            normalized = normalize_family_name(family)
            if normalized:
                return normalized

        families = metadata.get("families")
        if isinstance(families, list | tuple | set):
            for family_value in families:
                if isinstance(family_value, str) and family_value.strip():
                    normalized = normalize_family_name(family_value)
                    if normalized:
                        return normalized

    normalized_model = model.split(":", 1)[0].strip().lower()
    if "codex" in normalized_model:
        return "codex"

    family_token = normalized_model.split("-", 1)[0]
    normalized = normalize_family_name(family_token)
    return normalized or "model"


def metadata_supports_image_input(metadata: dict[str, object] | None) -> bool:
    if metadata is None:
        return False

    input_modalities = metadata.get("input_modalities")
    if isinstance(input_modalities, list | tuple | set):
        return any(
            isinstance(item, str) and item.strip().lower() == "image"
            for item in input_modalities
        )

    for key in ("supports_image", "image_input", "vision"):
        if metadata.get(key) is True:
            return True

    return False


def add_namespaced_metadata(
    model_info: dict[str, object], metadata: dict[str, object] | None
) -> None:
    if metadata is None:
        return

    metadata_mappings = (
        ("display_name", "codex.display_name"),
        ("description", "codex.description"),
        ("supported_reasoning_levels", "codex.reasoning.supported_levels"),
        ("default_reasoning_level", "codex.reasoning.default_level"),
        ("default_reasoning_summary", "codex.reasoning.default_summary"),
        ("default_verbosity", "codex.verbosity.default"),
        ("support_verbosity", "codex.verbosity.supported"),
        ("service_tiers", "codex.service_tiers"),
        ("truncation_policy", "codex.truncation_policy"),
    )
    for metadata_key, model_info_key in metadata_mappings:
        value = metadata.get(metadata_key)
        if value is not None:
            model_info[model_info_key] = value

    additional_speed_tiers = metadata.get("additional_speed_tiers")
    if isinstance(additional_speed_tiers, list):
        model_info["codex.additional_speed_tiers"] = additional_speed_tiers
        model_info["codex.fast_tier_available"] = "fast" in additional_speed_tiers


def ollama_model_details(
    model: str, metadata: dict[str, object] | None = None
) -> dict[str, object]:
    family = infer_ollama_family(model, metadata)
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
    details = ollama_model_details(model, metadata)
    model_info: dict[str, object] = {
        "general.architecture": details["family"],
        "general.basename": model,
        "general.file_type": 0,
        "general.parameter_count": 0,
        "general.quantization_version": 0,
        "codex.proxy.format": details["format"],
        "codex.proxy.variant": "synthetic-metadata",
    }
    parameters = ""
    context_length = metadata_context_length(metadata)
    if context_length is not None:
        model_info["general.context_length"] = context_length
        model_info[f"{details['family']}.context_length"] = context_length
        parameters = f"num_ctx {context_length}"
    add_namespaced_metadata(model_info, metadata)

    capabilities = list(SHOW_CAPABILITIES)
    if metadata_supports_image_input(metadata):
        capabilities.append("vision")
    return {
        "license": "",
        "modelfile": f"FROM {model}\n",
        "parameters": parameters,
        "template": "",
        "details": details,
        "model_info": model_info,
        "capabilities": capabilities,
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


def format_proxy_expires_at(ttl_seconds: float) -> str:
    expires_at = datetime.now(UTC) + timedelta(seconds=max(ttl_seconds, 0.0))
    return expires_at.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def metadata_size_vram(metadata: dict[str, object] | None) -> int:
    if metadata is None:
        return 0
    for key in ("size_vram", "size_vram_bytes"):
        parsed = parse_positive_int(metadata.get(key))
        if parsed is not None:
            return parsed
    return 0


def ollama_ps_model_payload(
    model: str,
    metadata: dict[str, object] | None,
    *,
    expires_at: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": model,
        "model": model,
        "size": 0,
        "digest": "",
        "details": ollama_model_details(model),
        "expires_at": expires_at,
        "size_vram": metadata_size_vram(metadata),
    }
    context_length = metadata_context_length(metadata)
    if context_length is not None:
        payload["context_length"] = context_length
    return payload


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


@router.get("/api/ps")
async def ollama_ps(
    settings: Settings = Depends(get_settings),
    model_catalog: ModelCatalogService = Depends(get_model_catalog),
) -> dict[str, object]:
    expires_at = format_proxy_expires_at(settings.model_catalog_ttl_seconds)
    models = await model_catalog.get_exposed_models()
    payload_models: list[dict[str, object]] = []
    for model in models:
        metadata = await model_catalog.get_model_metadata_for_request(model)
        payload_models.append(
            ollama_ps_model_payload(model, metadata, expires_at=expires_at)
        )
    return {"models": payload_models}


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
                    message = exc.message if isinstance(exc, BackendSSEError) else f"proxy error: {exc}"
                    chunk = build_ollama_error_ndjson(message)
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
    thinking = response.choices[0].message.reasoning
    message: dict[str, object] = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    payload = {
        "model": response.model,
        "created_at": "1970-01-01T00:00:00.000Z",
        "message": message,
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
                    message = exc.message if isinstance(exc, BackendSSEError) else f"proxy error: {exc}"
                    chunk = build_ollama_error_ndjson(message)
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
    thinking = response.choices[0].message.reasoning
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
    if thinking:
        payload["thinking"] = thinking
    log_debug_event(
        "client_response",
        status_code=200,
        media_type="application/json",
        body=payload,
    )
    finish_debug_request(debug_tokens)
    return JSONResponse(content=payload)
