# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Safety contract for clicking local paths rendered by the browser UI."""
from __future__ import annotations

from pathlib import Path

import pytest

import web_ui


def test_reveal_local_file_selects_it_in_finder(monkeypatch, tmp_path):
    target = tmp_path / "report with spaces.pdf"
    target.write_text("safe document", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(web_ui.subprocess, "run", fake_run)
    web_ui._reveal_local_path(str(target))

    assert calls == [(["open", "-R", str(target.resolve())], {"capture_output": True, "timeout": 5})]


def test_reveal_strips_markdown_code_and_bold_wrappers(monkeypatch, tmp_path):
    target = tmp_path / "report.pdf"
    target.write_text("safe document", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        web_ui.subprocess, "run",
        lambda command, **kwargs: (calls.append(command), type("Result", (), {"returncode": 0})())[1],
    )

    web_ui._reveal_local_path(f"`**{target}**`")

    assert calls == [["open", "-R", str(target.resolve())]]


def test_reveal_rejects_missing_or_relative_path(tmp_path):
    with pytest.raises(ValueError, match="absolute path"):
        web_ui._reveal_local_path("relative.txt")
    with pytest.raises(ValueError, match="ไม่พบ"):
        web_ui._reveal_local_path(str(tmp_path / "missing.txt"))


def test_websocket_reveal_path_dispatches_only_after_origin_check(monkeypatch, tmp_path):
    target = tmp_path / "clickable.md"
    target.write_text("safe document", encoding="utf-8")
    revealed = []
    monkeypatch.setattr(web_ui, "_reveal_local_path", lambda path: revealed.append(path))

    from fastapi.testclient import TestClient

    origin = {"origin": f"http://127.0.0.1:{web_ui.config.UI_PORT}"}
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers=origin) as ws:
            ws.receive_json()  # status
            ws.receive_json()  # workspace files
            ws.send_json({"type": "reveal_path", "path": str(target)})
            assert ws.receive_json() == {"type": "path_revealed"}

    assert revealed == [str(target)]


def test_chat_renderer_contains_safe_url_and_path_linkification():
    html = (Path(web_ui.config.PROJECT_ROOT) / "ui" / "chat.html").read_text(encoding="utf-8")
    assert "rel=\"noopener noreferrer\"" in html
    assert "data-local-path" in html
    assert "type:'reveal_path'" in html
    assert "trimLocalPathSuffix" in html
    assert "Word|PDF|Excel|PowerPoint" in html
    assert "<table><thead><tr>" in html
    assert "function handleLocalPathClick(event)" in html
    assert "$('panel-body').addEventListener('click', handleLocalPathClick)" in html
    assert "${renderMarkdown(a)}" in html
