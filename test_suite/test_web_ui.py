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


def test_browser_page_and_websocket_initial_state():
    with TestClient(web_ui.app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "ENDEAVOR LOCAL AGENT LITE" in page.text
        assert client.get("/ui-token").json() == {"token": ""}
        with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"}) as ws:
            status = ws.receive_json()
            files = ws.receive_json()
    assert status["type"] == "status"
    assert status["server_port"] == config.MLX_SERVER_PORT
    assert isinstance(status["online"], bool)
    assert status["history_max_chars"] > 0
    assert status["effort"] == "medium"
    assert status["thinking_budget"] == 256
    assert status["thinking_budgets"] == {"low": 128, "medium": 256, "high": 512}
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
        cancel_event=None, stream_tokens=False, effort=None,
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
        cancel_event=None, stream_tokens=False, effort=None,
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


def test_websocket_effort_persists_across_turns(monkeypatch):
    calls = []

    def fake_run_turn(
        user_input, messages, on_step=None, attachment_image=None,
        cancel_event=None, stream_tokens=False, effort=None,
    ):
        calls.append(effort)
        return agent.TurnResult(text="ok")

    monkeypatch.setattr(agent, "run_turn", fake_run_turn)
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"}) as ws:
            ws.receive_json(); ws.receive_json()
            ws.send_json({"type": "query", "content": "first", "effort": "high"})
            assert ws.receive_json()["type"] == "start"
            assert ws.receive_json()["type"] == "response"
            assert ws.receive_json()["type"] == "ctx_update"
            assert ws.receive_json()["type"] == "done"
            ws.send_json({"type": "query", "content": "second"})
            assert ws.receive_json()["type"] == "start"
            assert ws.receive_json()["type"] == "response"
            assert ws.receive_json()["type"] == "ctx_update"
            assert ws.receive_json()["type"] == "done"

    assert calls == ["high", "high"]


def test_effort_controls_are_present_and_sent_with_queries():
    html = (Path(config.PROJECT_ROOT) / "ui" / "chat.html").read_text(encoding="utf-8")
    assert 'aria-label="Effort"' in html
    assert '<select id="effort-select"' in html
    assert 'label id="effort-label"' not in html
    assert html.index('id="effort-controls"') < html.index('id="send-btn"')
    assert 'height: 34px' in html
    for value in ("low", "medium", "high"):
        assert f'<option value="{value}">' in html
    assert "option.textContent = `Effort: ${name}`" in html
    assert "thinking tokens" not in html
    assert "function setEffort(value)" in html
    assert "effort: S.effort" in html


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

    monkeypatch.setattr(web_ui, "_kill_mlx_server", lambda _port: None)
    monkeypatch.setattr(web_ui, "_MLX_SERVER_LOG", tmp_path / "mlx.log")
    monkeypatch.setattr(web_ui.subprocess, "Popen", fake_popen)
    web_ui._start_mlx_server()
    command = captured["command"]
    index = command.index("--vision-cache-size")
    assert command[index + 1] == str(config.MLX_SERVER_VISION_CACHE_SIZE) == "2"


def test_model_stop_refuses_unmanaged_port_owner(monkeypatch):
    import pytest

    monkeypatch.setattr(web_ui, "_mlx_server_proc", None)
    monkeypatch.setattr(web_ui, "_pids_on_port", lambda _port: [4242])
    monkeypatch.setattr(web_ui, "_process_uid", lambda _pid: os.getuid())
    monkeypatch.setattr(web_ui, "_process_command", lambda _pid: "/usr/bin/python -m unrelated.server --port 8090")
    terminated: list[int] = []
    monkeypatch.setattr(web_ui, "_terminate_pid", terminated.append)

    with pytest.raises(RuntimeError, match="process อื่น"):
        web_ui._kill_mlx_server(config.MLX_SERVER_PORT)
    assert terminated == []


def test_model_probe_requires_the_configured_model(monkeypatch):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self, _limit):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(web_ui, "_mlx_server_up", lambda: True)
    monkeypatch.setattr(
        web_ui.urllib.request, "urlopen",
        lambda *_args, **_kwargs: Response({"data": [{"id": config.MODEL_ID}]}),
    )
    assert web_ui._mlx_server_state() == "ready"

    monkeypatch.setattr(
        web_ui.urllib.request, "urlopen",
        lambda *_args, **_kwargs: Response({"data": [{"id": "another-model"}]}),
    )
    assert web_ui._mlx_server_state() == "wrong_model"


def test_query_worker_exception_returns_error_and_clears_cancel_state(monkeypatch):
    class Socket:
        def __init__(self):
            self.events = []

        async def send_json(self, payload):
            self.events.append(payload)

    def failing_run_turn(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(agent, "run_turn", failing_run_turn)
    session = web_ui.BrowserSession()
    socket = Socket()
    asyncio.run(web_ui._run_query(socket, session, "hello"))

    assert any(event["type"] == "error" for event in socket.events)
    assert session.cancel_event is None


def test_queued_index_can_be_cancelled_before_child_spawn(monkeypatch):
    timeline: list[tuple[str, dict]] = []
    class Socket:
        async def send_json(self, payload):
            timeline.append(("index", payload))

    socket = Socket()
    session = web_ui.BrowserSession()

    async def scenario():
        await web_ui._HEAVY_WORK_LOCK.acquire()
        try:
            task = asyncio.create_task(web_ui._run_index(socket, session))
            for _ in range(20):
                if session.cancel_event is not None:
                    break
                await asyncio.sleep(0.01)
            assert session.cancel_event is not None
            session.cancel_event.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            if web_ui._HEAVY_WORK_LOCK.locked():
                web_ui._HEAVY_WORK_LOCK.release()

    asyncio.run(scenario())
    assert any(payload.get("type") == "cancelled" for _, payload in timeline)
    assert web_ui._active_index_owner is None


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


def test_browser_upload_streams_a_bounded_private_attachment(monkeypatch, tmp_path):
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
    assert stored.stat().st_mode & 0o777 == 0o600
    stored.unlink()


def test_browser_upload_rejects_invalid_size_and_name_without_creating_files(monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui, "_ATTACHMENTS_DIR", tmp_path)
    origin = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
    with TestClient(web_ui.app) as client:
        bad_length = client.post(
            "/upload", content=b"x", headers={**origin, "content-length": "not-a-size", "x-file-name": "x.txt"},
        )
        bad_name = client.post(
            "/upload", content=b"x", headers={**origin, "x-file-name": ("x" * 513) + ".txt"},
        )
    assert bad_length.json()["ok"] is False
    assert bad_name.json()["ok"] is False
    assert list(tmp_path.iterdir()) == []


def test_model_server_log_rotation_is_bounded_and_private(monkeypatch, tmp_path):
    log = tmp_path / "mlx.log"
    log.write_bytes(b"old")
    monkeypatch.setattr(web_ui, "_SERVER_LOG_MAX_BYTES", 2)
    monkeypatch.setattr(web_ui, "_SERVER_LOG_BACKUP_COUNT", 1)
    web_ui._rotate_log_file(log)
    assert not log.exists()
    assert (tmp_path / "mlx.log.1").read_bytes() == b"old"

    with web_ui._open_private_append(log) as handle:
        handle.write(b"new")
    assert log.stat().st_mode & 0o777 == 0o600


def test_state_changing_http_routes_require_same_origin(monkeypatch):
    monkeypatch.setattr(web_ui, "_start_mlx_server", lambda: None)
    monkeypatch.setattr(web_ui, "_kill_mlx_server", lambda _port: None)
    monkeypatch.setattr(web_ui, "_mlx_server_state", lambda: "down")
    with TestClient(web_ui.app) as client:
        for path in ("/upload", "/pdf-to-text/start", "/pdf-to-text/status", "/server/start", "/server/stop"):
            assert client.post(path).status_code == 403
        headers = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
        started = client.post("/server/start", headers=headers).json()
        assert started["ok"] is True
        assert started["server_state"] == "starting"
        stopped = client.post("/server/stop", headers=headers).json()
        assert stopped["ok"] is True
        assert stopped["server_up"] is False


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


def test_attachment_image_path_accepts_only_lite_uploaded_images(monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui, "_ATTACHMENTS_DIR", tmp_path)
    image = web_ui._ATTACHMENTS_DIR / "attachment-test.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    try:
        assert web_ui._attachment_image_path({"path": str(image), "isImage": True}) == str(image.resolve())
        assert web_ui._attachment_image_path({"path": str(image), "isImage": False}) is None
        assert web_ui._attachment_image_path({"path": "/tmp/not-owned.png", "isImage": True}) is None
    finally:
        image.unlink(missing_ok=True)


def test_rag_panel_http_search_and_sync_bypass_llm(monkeypatch):
    """RAG side-panel utilities must never enter agent.run_turn/ReAct."""
    calls = []

    def forbidden_llm(*_args, **_kwargs):
        raise AssertionError("RAG panel must not call agent.run_turn")

    def fake_search(query):
        calls.append(("search", query))
        return "พบ 1 ไฟล์\n- /tmp/manual.pdf\n  page 9: hydraulic pressure"

    def fake_sync():
        calls.append(("sync", None))
        return {
            "raw_scanned": 2, "indexed": 2, "unchanged": 0, "removed": 0,
            "documents": 2, "chunks": 7, "inserted": 1, "updated": 1,
            "sync_errors": 0, "seconds": 0.25,
        }

    monkeypatch.setattr(agent, "run_turn", forbidden_llm)
    monkeypatch.setattr(web_ui.rag_mcp_service, "search", fake_search)
    monkeypatch.setattr(web_ui.rag_mcp_service, "sync_kb", fake_sync)
    headers = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}

    with TestClient(web_ui.app) as client:
        found = client.post("/rag/search", json={"query": "hydraulic"}, headers=headers)
        synced = client.post("/rag/sync", headers=headers)

    assert found.status_code == 200
    assert found.json() == {
        "ok": True,
        "result": "พบ 1 ไฟล์\n- /tmp/manual.pdf\n  page 9: hydraulic pressure",
    }
    assert synced.status_code == 200
    sync_body = synced.json()
    assert sync_body["ok"] is True
    assert sync_body["stats"]["documents"] == 2
    assert "status=ok" in sync_body["summary"]
    assert calls == [("search", "hydraulic"), ("sync", None)]


def test_rag_panel_sync_reports_degraded_result_as_error_without_llm(monkeypatch):
    def forbidden_llm(*_args, **_kwargs):
        raise AssertionError("RAG panel sync must not call agent.run_turn")

    monkeypatch.setattr(agent, "run_turn", forbidden_llm)
    monkeypatch.setattr(
        web_ui.rag_mcp_service,
        "sync_kb",
        lambda: {
            "raw_scanned": 3, "indexed": 2, "unchanged": 0, "removed": 0,
            "documents": 2, "chunks": 7, "inserted": 1, "updated": 1,
            "sync_errors": 1, "seconds": 0.25,
        },
    )
    headers = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}

    with TestClient(web_ui.app) as client:
        response = client.post("/rag/sync", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["stats"]["sync_errors"] == 1
    assert "status=degraded" in body["summary"]
    assert "1" in body["error"]


def test_direct_rag_sync_job_keeps_live_phase_and_percent(monkeypatch):
    stats = {
        "raw_scanned": 1, "total_files": 1, "completed_files": 1,
        "progress_percent": 99.0, "estimated_remaining_seconds": 0.0,
        "indexed": 1, "unchanged": 0, "removed": 0, "documents": 1,
        "chunks": 2, "inserted": 1, "updated": 0, "sync_errors": 0,
        "seconds": 0.01,
    }

    def fake_sync(*, on_progress=None):
        if on_progress:
            on_progress("prepare", None, {"total_files": 0, "completed_files": 0, "progress_percent": 0.0})
            on_progress("result", Path("/tmp/manual.txt"), stats)
        return {**stats, "progress_percent": 100.0}

    monkeypatch.setattr(web_ui.rag_mcp_service, "sync_kb", fake_sync)
    monkeypatch.setattr(web_ui, "_rag_sync_state", {
        "job_id": "sync-test", "status": "queued", "event": "queued", "path": None,
        "stats": {}, "summary": None, "error": None, "message": "queued",
    })

    asyncio.run(web_ui._run_rag_sync_job("sync-test"))
    snapshot = web_ui._rag_sync_snapshot()

    assert snapshot["status"] == "done"
    assert snapshot["stats"]["progress_percent"] == 100.0
    assert "100%" in snapshot["message"]
    assert "status=ok" in snapshot["summary"]


def test_rag_panel_http_requires_same_origin_and_valid_query(monkeypatch):
    monkeypatch.setattr(web_ui.rag_mcp_service, "search", lambda query: query)
    headers = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
    with TestClient(web_ui.app) as client:
        assert client.post("/rag/search", json={"query": "x"}).status_code == 403
        assert client.post("/rag/sync").status_code == 403
        assert client.post("/rag/search", json={"query": 123}, headers=headers).json()["ok"] is False
        assert client.post("/rag/search", content=b"not-json", headers=headers).json()["ok"] is False


def test_pdf_to_text_panel_streams_private_pdf_and_bypasses_chat_llm(monkeypatch, tmp_path):
    """The direct panel owns its PDF pipeline and must not enter agent.run_turn."""
    upload_dir = tmp_path / "uploads"
    output = tmp_path / "converted.txt"
    calls = []

    def forbidden_agent(*_args, **_kwargs):
        raise AssertionError("PDF to Text panel must not call agent.run_turn")

    def fake_convert(source, *, source_name, rewrite_thai, progress=None):
        source = Path(source)
        calls.append((source_name, rewrite_thai, source.read_bytes()))
        if progress:
            progress("extract", 1, 2, {"native": 1, "ocr": 0, "tables": 0})
            progress("rewrite", 2, 2, {"batch": 1, "batches": 1, "pages": 1})
            progress("write", 2, 2, {"native": 1, "ocr": 1, "tables": 0})
        output.write_text(
            "===== PDF PAGE 1/2 =====\nA\n\n===== PDF PAGE 2/2 =====\nB\n",
            encoding="utf-8",
        )
        return {
            "job_id": "inner-job",
            "source_name": source_name,
            "total_pages": 2,
            "output_path": str(output),
            "rewrite_thai": rewrite_thai,
            "stats": {
                "native": 1, "ocr": 1, "tables": 0,
                "rewritten": 1, "rewrite_fallback": 0,
            },
            "warnings": [],
        }

    monkeypatch.setattr(agent, "run_turn", forbidden_agent)
    monkeypatch.setattr(web_ui.pdf_to_text, "convert_pdf", fake_convert)
    monkeypatch.setattr(web_ui, "_PDF_TEXT_UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(web_ui, "_pdf_text_state", None)
    monkeypatch.setattr(web_ui, "_pdf_text_task", None)
    headers = {
        "origin": f"http://127.0.0.1:{config.UI_PORT}",
        "content-type": "application/pdf",
        "x-file-name": "%E0%B8%84%E0%B8%B9%E0%B9%88%E0%B8%A1%E0%B8%B7%E0%B8%AD.pdf",
        "x-rewrite-thai": "1",
    }

    with TestClient(web_ui.app) as client:
        started = client.post(
            "/pdf-to-text/start",
            content=b"%PDF-1.7\nminimal-test-body",
            headers=headers,
        )
        assert started.status_code == 200
        start_body = started.json()
        assert start_body["ok"] is True
        job_id = start_body["job"]["job_id"]

        final = None
        for _ in range(100):
            status = client.post(
                "/pdf-to-text/status",
                json={"job_id": job_id},
                headers={"origin": f"http://127.0.0.1:{config.UI_PORT}"},
            )
            assert status.status_code == 200
            body = status.json()
            assert body["ok"] is True
            final = body["job"]
            if final["status"] in {"done", "error"}:
                break
            time.sleep(0.01)

    assert final is not None
    assert final["status"] == "done"
    assert final["total_pages"] == 2
    assert final["output_path"] == str(output)
    assert final["stats"]["rewritten"] == 1
    assert calls == [("คู่มือ.pdf", True, b"%PDF-1.7\nminimal-test-body")]
    assert list(upload_dir.glob(f"{web_ui._MANAGED_PDF_TEXT_PREFIX}*")) == []


def test_pdf_to_text_start_rejects_non_pdf_and_bad_magic(monkeypatch, tmp_path):
    monkeypatch.setattr(web_ui, "_PDF_TEXT_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(web_ui, "_pdf_text_state", None)
    headers = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
    with TestClient(web_ui.app) as client:
        wrong_name = client.post(
            "/pdf-to-text/start",
            content=b"%PDF-1.7\n",
            headers={**headers, "x-file-name": "notes.txt"},
        )
        bad_magic = client.post(
            "/pdf-to-text/start",
            content=b"not a pdf",
            headers={**headers, "x-file-name": "notes.pdf"},
        )
    assert wrong_name.json()["ok"] is False
    assert bad_magic.json()["ok"] is False
    assert list(tmp_path.iterdir()) == []


def test_browser_page_exposes_direct_pdf_to_text_panel():
    html = (Path(config.PROJECT_ROOT) / "ui" / "chat.html").read_text(encoding="utf-8")
    assert 'data-panel="pdf"' in html
    assert "function renderPdfToText()" in html
    assert "function pdfToTextStart()" in html
    assert "function pollPdfToText(jobId)" in html
    assert "fetch('/pdf-to-text/start'" in html
    assert "fetch('/pdf-to-text/status'" in html
    assert "text layer" in html
    assert "Apple Vision OCR" in html
    assert "SQLite" in html
    assert "no-think" in html
    assert "สูงสุด 4 หน้าต่อ request" in html


def test_browser_page_exposes_deterministic_rag_panel():
    html = (Path(config.PROJECT_ROOT) / "ui" / "chat.html").read_text(encoding="utf-8")
    assert 'data-panel="rag"' in html
    assert "function renderRag()" in html
    assert "function ragSearchDirect()" in html
    assert "function ragSyncDirect()" in html
    assert "fetch('/rag/search'" in html
    assert "fetch('/rag/sync/start'" in html
    assert "fetch('/rag/sync/status'" in html
    assert "ไม่เรียก LLM" in html
    assert "ไม่เพิ่ม chat context" in html
    assert "↻ sync_kb" in html
    assert "↻ /sync_kb" not in html
    assert "onclick=\"runCmd('${cmd}')\">${icon} ${cmd}</button>`;" in html


def test_browser_page_supports_persisted_light_and_dark_themes():
    html = (Path(config.PROJECT_ROOT) / "ui" / "chat.html").read_text(encoding="utf-8")
    assert ':root, html[data-theme="dark"]' in html
    assert 'html[data-theme="light"]' in html
    assert '--bg: #1e1e1e;' in html
    assert '--bg: #f7f7f8;' in html
    assert 'color-scheme: dark' in html
    assert 'color-scheme: light' in html
    assert 'id="theme-toggle"' in html
    assert 'function applyTheme(theme, persist = true)' in html
    assert 'function toggleTheme()' in html
    assert "localStorage.getItem('endeavor-lite-theme')" in html
    assert 'localStorage.setItem(THEME_KEY, next)' in html
    # Preserve the historical UI as the default for users without a preference.
    assert "let theme = 'dark';" in html
    # Key surfaces consume variables instead of hard-coding dark-only colors.
    assert 'background: var(--bg);' in html
    assert 'background: var(--surface);' in html
    assert 'background: var(--user-bg);' in html
    assert 'background: var(--surface-2);' in html
