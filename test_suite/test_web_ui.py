# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No-model checks for Lite's local browser UI transport and file boundary."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import agent
import config
import web_ui


def test_server_start_does_not_take_over_an_occupied_port(monkeypatch):
    """A local port may belong to another app; never kill it to start Lite."""
    monkeypatch.setattr(web_ui, "_mlx_server_up", lambda: True)
    with TestClient(web_ui.app) as client:
        payload = client.post(
            "/server/start", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"},
        ).json()
    assert payload["ok"] is False
    assert "ถูกใช้งานอยู่แล้ว" in payload["error"]


def test_server_health_requires_configured_model(monkeypatch):
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return json.dumps({"data": [{"id": config.MODEL_ID}]}).encode("utf-8")

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response()

    monkeypatch.setattr(web_ui, "urlopen", fake_urlopen)
    assert web_ui._mlx_server_up() is True
    assert calls == [(f"{config.MLX_BASE_URL}/models", 0.75)]


def test_server_health_rejects_wrong_model(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return b'{"data":[{"id":"wrong-model"}]}'

    monkeypatch.setattr(web_ui, "urlopen", lambda *args, **kwargs: _Response())
    assert web_ui._mlx_server_up() is False


def test_stop_only_terminates_the_server_started_by_this_web_ui(monkeypatch):
    class _Proc:
        pid = 4321

        def poll(self):
            return None

    terminated = []
    monkeypatch.setattr(web_ui, "_mlx_server_proc", _Proc())
    monkeypatch.setattr(web_ui, "_terminate_pid", terminated.append)
    assert web_ui._stop_managed_mlx_server() is True
    assert terminated == [4321]
    assert web_ui._mlx_server_proc is None


def test_browser_page_and_websocket_initial_state():
    with TestClient(web_ui.app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "ENDEAVOR AGENT LITE" in page.text
        assert client.get("/ui-token").json() == {"token": ""}
        with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"}) as ws:
            status = ws.receive_json()
            files = ws.receive_json()
    assert status["type"] == "status"
    assert status["server_port"] == config.MLX_SERVER_PORT
    assert isinstance(status["online"], bool)
    assert status["history_max_chars"] > 0
    assert any(cmd["name"] == "index" for cmd in status["builtin_cmds"])
    # '/stop' must stay out of the picker's builtin_cmds list: it is a
    # wire-only command consumed by the WS reader and never replied to when
    # idle, so surfacing it as a clickable/typeable slash-command would hang
    # the client's spinner forever with nothing to cancel.
    assert not any(cmd["name"] == "stop" for cmd in status["builtin_cmds"])
    assert files["type"] == "files"
    assert files["root"] == str(config.WORKSPECT_DIR)
    assert files["truncated"] is False


def test_assistant_markdown_renders_inside_one_bubble():
    html = (Path(config.PROJECT_ROOT) / "ui" / "chat.html").read_text(encoding="utf-8")
    assert "function renderMarkdown(text)" in html
    assert "function inlineMarkdown(rawText)" in html
    assert "function parseTableRow(line)" in html
    assert "function isTableDivider(line)" in html
    assert "function stripUnpairedMarkdownMarkers(value)" in html
    assert "return renderMarkdown(text);" in html
    assert ".msg-content > pre" not in html
    assert "LITEMD${slots.length}X" in html
    assert "LITE_MD_" not in html
    assert "liveActivity: null" in html
    assert "function renderPhaseBar()" in html
    assert "function liveActivitySummary(task)" in html
    assert "S.phaseSub = liveActivitySummary(S.liveActivity);" in html


def test_browser_websocket_rejects_cross_origin():
    with TestClient(web_ui.app) as client:
        try:
            with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
                pass
        except Exception:
            return
    raise AssertionError("cross-origin WebSocket should be rejected")


def test_workspace_path_rejects_escape():
    assert web_ui._workspace_path(".") == config.WORKSPECT_DIR.resolve()
    try:
        web_ui._workspace_path("/etc")
    except ValueError as exc:
        assert "workspect" in str(exc)
    else:
        raise AssertionError("outside workspace should be rejected")


def test_browser_websocket_rejects_malformed_and_oversized_requests():
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"}) as ws:
            ws.receive_json(); ws.receive_json()
            ws.send_text("not json")
            assert "JSON" in ws.receive_json()["msg"]
            ws.send_text("x" * (web_ui._MAX_WS_MESSAGE_CHARS + 1))
            assert "ใหญ่เกิน" in ws.receive_json()["msg"]
            ws.send_text("9" * 5_000)
            assert "JSON" in ws.receive_json()["msg"]
            ws.send_json({"type": "query", "content": {"not": "text"}})
            assert "content" in ws.receive_json()["msg"]
            ws.send_bytes(b"not json")
            assert "JSON" in ws.receive_json()["msg"]


def test_websocket_stop_command_cancels_an_in_flight_turn(monkeypatch):
    """The reader task must intercept '/stop' and reach the running turn's
    cancel_event without waiting for _run_query (awaited inline) to return."""
    def fake_run_turn(
        user_input, messages, on_step=None, attachment_image=None,
        cancel_event=None, stream_tokens=False,
    ):
        assert stream_tokens is True
        for _ in range(500):
            if cancel_event is not None and cancel_event.is_set():
                return agent.TurnResult(text="", cancelled=True, aborted=True)
            time.sleep(0.01)
        return agent.TurnResult(text="should have been cancelled long before this")

    monkeypatch.setattr(agent, "run_turn", fake_run_turn)
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"}) as ws:
            ws.receive_json(); ws.receive_json()
            ws.send_json({"type": "query", "content": "long task"})
            assert ws.receive_json()["type"] == "start"
            ws.send_json({"type": "command", "cmd": "/stop"})
            assert ws.receive_json() == {"type": "cancelled"}


def test_websocket_disconnect_cancels_an_in_flight_turn():
    """A closed tab must cancel the worker just like the explicit /stop command."""
    class DisconnectingSocket:
        async def receive(self):
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect()

    session = web_ui.BrowserSession(cancel_event=threading.Event())
    queue: asyncio.Queue[dict] = asyncio.Queue()
    asyncio.run(web_ui._ws_reader(DisconnectingSocket(), session, queue))
    assert session.cancel_event.is_set()
    assert queue.get_nowait() == {"type": "websocket.disconnect"}


def test_websocket_reader_bounds_flooded_inbox_and_cancels_work():
    class FloodingSocket:
        async def receive(self):
            return {
                "type": "websocket.receive",
                "text": json.dumps({"type": "query", "content": "x" * 1_000}),
            }

    session = web_ui.BrowserSession(cancel_event=threading.Event())
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=web_ui._MAX_WS_INBOX_ITEMS)
    asyncio.run(web_ui._ws_reader(FloodingSocket(), session, queue))
    assert session.cancel_event.is_set()
    assert queue.qsize() == 1
    assert queue.get_nowait() == {"type": "lite.inbox_overflow"}


def test_html_query_streams_first_token_before_final_response(monkeypatch):
    class Socket:
        def __init__(self):
            self.events = []

        async def send_json(self, payload):
            self.events.append(payload)

    def fake_run_turn(
        user_input, messages, on_step=None, attachment_image=None,
        cancel_event=None, stream_tokens=False,
    ):
        assert stream_tokens is True
        on_step(event="stream_start")
        on_step(event="token", text="ส")
        on_step(event="token", text="วัสดี")
        on_step(event="stream_end")
        return agent.TurnResult(text="สวัสดี")

    monkeypatch.setattr(agent, "run_turn", fake_run_turn)
    socket = Socket()
    asyncio.run(web_ui._run_query(socket, web_ui.BrowserSession(), "hello"))

    event_types = [event["type"] for event in socket.events]
    first_token = event_types.index("token")
    assert first_token < event_types.index("response")
    assert socket.events[first_token]["text"].startswith("ส")
    assert socket.events[-1]["type"] == "done"


def test_model_launcher_applies_bounded_vision_cache(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(web_ui, "_mlx_server_up", lambda: False)
    monkeypatch.setattr(web_ui, "_mlx_server_proc", None)
    monkeypatch.setattr(web_ui, "_MLX_SERVER_LOG", tmp_path / "mlx.log")
    monkeypatch.setattr(web_ui.subprocess, "Popen", fake_popen)
    assert web_ui._start_mlx_server() is None
    command = captured["command"]
    index = command.index("--vision-cache-size")
    assert command[index + 1] == str(config.MLX_SERVER_VISION_CACHE_SIZE) == "2"


def test_workspace_listing_is_bounded_and_reports_truncation(tmp_path):
    for index in range(web_ui._MAX_BROWSE_ENTRIES + 5):
        (tmp_path / f"file-{index:04d}.txt").touch()
    rows, truncated = web_ui._files_for(tmp_path)
    assert len(rows) == web_ui._MAX_BROWSE_ENTRIES
    assert truncated is True
    assert rows[0]["name"] == "file-0000.txt"


def test_websocket_stop_command_with_no_turn_running_is_a_silent_no_op():
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"}) as ws:
            ws.receive_json(); ws.receive_json()
            ws.send_json({"type": "command", "cmd": "/stop"})
            # Must not surface as an unrecognized command, and must not hang;
            # confirm the connection is still alive with an ordinary request.
            ws.send_json({"type": "command", "cmd": "/clear"})
            assert ws.receive_json()["type"] == "clear_ok"


def test_activity_args_keep_tool_queries_and_redact_sensitive_values():
    assert web_ui._activity_args({"query": "latest SET index news"}) == {
        "query": "latest SET index news"
    }
    assert web_ui._activity_args({"command": "find /tmp -name '*.pdf'"}) == {
        "command": "find /tmp -name '*.pdf'"
    }
    assert web_ui._activity_args({"command": "echo sk-" + "A" * 32}) == {
        "command": "[redacted: sensitive value]"
    }


def test_browser_upload_stores_a_bounded_attachment_inside_workspect(monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui, "_ATTACHMENTS_DIR", tmp_path)
    with TestClient(web_ui.app) as client:
        response = client.post(
            "/upload", content=b"attachment text", headers={
                "content-type": "text/plain", "x-file-name": "notes.txt",
                "origin": f"http://127.0.0.1:{config.UI_PORT}",
            },
        )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["name"] == "notes.txt"
    assert payload["isImage"] is False
    stored = Path(payload["path"])
    assert stored.parent == web_ui._ATTACHMENTS_DIR
    assert stored.read_bytes() == b"attachment text"
    stored.unlink()


def test_state_changing_http_routes_require_same_origin(monkeypatch):
    monkeypatch.setattr(web_ui, "_start_mlx_server", lambda: None)
    monkeypatch.setattr(web_ui, "_stop_managed_mlx_server", lambda: True)
    with TestClient(web_ui.app) as client:
        for path in ("/upload", "/server/start", "/server/stop"):
            assert client.post(path).status_code == 403
        headers = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
        assert client.post("/server/start", headers=headers).json()["ok"] is True
        assert client.post("/server/stop", headers=headers).json()["ok"] is True


def test_attachment_cleanup_expires_old_files_and_bounds_total_size(monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui, "_ATTACHMENTS_DIR", tmp_path)
    monkeypatch.setattr(web_ui, "_ATTACHMENT_MAX_TOTAL_BYTES", 3)
    old = tmp_path / f"{web_ui._MANAGED_ATTACHMENT_PREFIX}old.txt"
    old.write_bytes(b"old")
    os.utime(old, (time.time() - web_ui._ATTACHMENT_MAX_AGE_SECONDS - 1,) * 2)
    first = tmp_path / f"{web_ui._MANAGED_ATTACHMENT_PREFIX}first.txt"
    first.write_bytes(b"aa")
    second = tmp_path / f"{web_ui._MANAGED_ATTACHMENT_PREFIX}second.txt"
    second.write_bytes(b"bb")
    web_ui._cleanup_attachments()
    assert not old.exists()
    assert sum(item.stat().st_size for item in tmp_path.iterdir()) <= 3


def test_attachment_image_path_accepts_only_lite_uploaded_images(tmp_path):
    image = web_ui._ATTACHMENTS_DIR / "attachment-test.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    try:
        assert web_ui._attachment_image_path({"path": str(image), "isImage": True}) == str(image.resolve())
        assert web_ui._attachment_image_path({"path": str(image), "isImage": False}) is None
        assert web_ui._attachment_image_path({"path": "/tmp/not-owned.png", "isImage": True}) is None
    finally:
        image.unlink(missing_ok=True)
