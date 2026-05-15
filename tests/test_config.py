from __future__ import annotations

from codex_openai_ollama_proxy.core.config import Settings


def test_disable_copilot_adaptations_env_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("DISABLE_COPILOT_ADAPTATIONS", "false")

    settings = Settings.from_sources(cli_args=[], cwd=tmp_path)

    assert settings.disable_copilot_adaptations is False


def test_disable_copilot_adaptations_defaults_to_true(monkeypatch, tmp_path):
    monkeypatch.delenv("DISABLE_COPILOT_ADAPTATIONS", raising=False)

    settings = Settings.from_sources(cli_args=[], cwd=tmp_path)

    assert settings.disable_copilot_adaptations is True


def test_add_default_responses_instructions_env_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ADD_DEFAULT_RESPONSES_INSTRUCTIONS", "true")

    settings = Settings.from_sources(cli_args=[], cwd=tmp_path)

    assert settings.add_default_responses_instructions is True


def test_add_default_responses_instructions_defaults_to_false(monkeypatch, tmp_path):
    monkeypatch.delenv("ADD_DEFAULT_RESPONSES_INSTRUCTIONS", raising=False)

    settings = Settings.from_sources(cli_args=[], cwd=tmp_path)

    assert settings.add_default_responses_instructions is False
