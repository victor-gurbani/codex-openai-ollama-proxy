from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .openai import ChatMessage


def validate_ollama_think(value: Any) -> bool | str | None:
    if value is None:
        return None
    if value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "false":
            return False
        if normalized == "true":
            return True
        if normalized in {"none", "low", "medium", "high", "xhigh"}:
            return normalized
    raise ValueError("think must be one of: true, false, none, low, medium, high, xhigh")


def prefer_legacy_name_when_model_is_blank(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    model = data.get("model")
    name = data.get("name")
    if isinstance(name, str) and name.strip() and not (
        isinstance(model, str) and model.strip()
    ):
        return {**data, "model": name}
    return data


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] | None = None
    prompt: str | None = None
    images: list[str] | None = None
    format: Any | None = None
    system: str | None = None
    stream: bool | None = None
    think: bool | str | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("think", mode="before")
    @classmethod
    def _validate_think(cls, value: Any) -> bool | str | None:
        return validate_ollama_think(value)


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str | None = None
    images: list[str] | None = None
    format: Any | None = None
    system: str | None = None
    stream: bool | None = None
    messages: list[ChatMessage] | None = None
    think: bool | str | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("think", mode="before")
    @classmethod
    def _validate_think(cls, value: Any) -> bool | str | None:
        return validate_ollama_think(value)


class OllamaShowRequest(BaseModel):
    model: str = Field(validation_alias=AliasChoices("model", "name"))

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _prefer_legacy_name_when_model_is_blank(cls, data: Any) -> Any:
        return prefer_legacy_name_when_model_is_blank(data)


class OllamaPullRequest(BaseModel):
    model: str = Field(validation_alias=AliasChoices("model", "name"))
    stream: bool | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _prefer_legacy_name_when_model_is_blank(cls, data: Any) -> Any:
        return prefer_legacy_name_when_model_is_blank(data)
