from __future__ import annotations

from pathlib import Path

from codex_openai_ollama_proxy import cli
from codex_openai_ollama_proxy.core.config import Settings


def test_cli_prints_effective_debug_and_copilot_flags(monkeypatch, capsys, tmp_path):
    settings = Settings(
        port=8888,
        auth_path=tmp_path / "auth.json",
        required_client_api_key=None,
        debug=True,
        disable_copilot_adaptations=True,
        project_root=Path.cwd(),
    )

    monkeypatch.setattr(cli.Settings, "from_sources", lambda cli_args: settings)
    monkeypatch.setattr(cli, "create_app", lambda resolved_settings: object())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, host, port: None)

    cli.main([])

    output = capsys.readouterr().out
    assert "DEBUG=on" in output
    assert "DISABLE_COPILOT_ADAPTATIONS=on" in output
