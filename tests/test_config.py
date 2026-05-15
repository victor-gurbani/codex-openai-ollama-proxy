from __future__ import annotations

from codex_openai_ollama_proxy.core.config import Settings


def test_disable_copilot_adaptations_env_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("DISABLE_COPILOT_ADAPTATIONS", "true")

    settings = Settings.from_sources(cli_args=[], cwd=tmp_path)

    assert settings.disable_copilot_adaptations is True


def test_disable_copilot_adaptations_defaults_to_false(monkeypatch, tmp_path):
    monkeypatch.delenv("DISABLE_COPILOT_ADAPTATIONS", raising=False)

    settings = Settings.from_sources(cli_args=[], cwd=tmp_path)

    assert settings.disable_copilot_adaptations is False
