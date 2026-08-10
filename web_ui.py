# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Local-only WebSocket backend for the copied ENDEAVOR browser UI.

Run with ``python web_ui.py`` then open http://127.0.0.1:8766.  The service
binds only loopback and exposes workspace browsing only under workspect/.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import heapq
import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import unquote
from urllib.request import Request as URLRequest, urlopen
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

import agent
import config
import ui
from helpers.sensitive_paths import contains_secret_content, is_sensitive_path, path_is_or_contains_secret

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_HTML = config.PROJECT_ROOT / "ui" / "chat.html"
_MAX_VIEW_BYTES = 64 * 1024
_MAX_WS_MESSAGE_CHARS = 12_000
_MAX_WS_INBOX_ITEMS = 16
_MAX_BROWSE_ENTRIES = 500
_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
_ATTACHMENT_MAX_AGE_SECONDS = 24 * 60 * 60
_ATTACHMENT_MAX_TOTAL_BYTES = 100 * 1024 * 1024
_MANAGED_ATTACHMENT_PREFIX = "lite-attachment-"
_MAX_CONNECTIONS = 4
_ATTACHMENTS_DIR = config.WORKSPECT_DIR / ".attachments"
_IMAGE_ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic", ".heif", ".tiff", ".tif"}
_ALLOWED_ORIGINS = {
    f"http://127.0.0.1:{config.UI_PORT}",
    f"http://localhost:{config.UI_PORT}",
}
_INDEX_WORKER = config.PROJECT_ROOT / "indexer_worker.py"
# Single process-global gate shared by /index and agent.run_turn (PLAN.md
# §9.2): the old separate _INDEX_LOCK only kept two index runs from
# overlapping each other -- an index job and a model turn from a *different*
# browser tab could still run concurrently, racing the same SQLite file and
# doubling peak memory. Everything heavy now serializes through one lock, in
# FIFO order, so a second tab's request queues deterministically instead of
# silently overlapping.
_HEAVY_WORK_LOCK = asyncio.Lock()
_active_index_process: "asyncio.subprocess.Process | None" = None
# The BrowserSession that started the currently-running index -- "/stop"
# must only ever cancel the index its OWN tab started. Without this, a
# second tab's "/stop" (meant to abandon their own queued chat turn while
# they wait behind someone else's /index) would kill a stranger's running
# index job out from under them.
_active_index_owner: "BrowserSession | None" = None
# Set right before an explicit user-initiated termination (the "/stop"
# handler below) so _run_index_subprocess can tell that apart from the
# child dying on its own (a real crash) -- both surface as "no stats", but
# only one of them should ever be reported to the user as "cancelled".
_index_cancel_requested = False
_MAX_UNPARSED_TAIL_CHARS = 4_000
_CONNECTIONS = asyncio.Semaphore(_MAX_CONNECTIONS)
_ACTIVITY_VALUE_MAX_CHARS = 4_000

# mlx_vlm.server (the model process on config.MLX_SERVER_PORT) is started and
# stopped from the browser's header button.  It is a completely separate
# process from this FastAPI app -- a stale one may already be running from a
# previous session, a manual `python -m mlx_vlm.server` invocation, or a
# crashed prior start, so "kill" always targets the *port*, not a tracked
# Popen handle, or an externally-started server would survive a "stop" click.
_MLX_SERVER_LOCK = threading.Lock()
_MLX_SERVER_LOG = config.LOG_DIR / "mlx_server.log"
_mlx_server_proc: "subprocess.Popen | None" = None


@dataclass
class BrowserSession:
    messages: list[dict] = field(default_factory=list)
    cancel_event: "threading.Event | None" = None


def _origin_is_allowed(origin: str | None) -> bool:
    return origin in _ALLOWED_ORIGINS


def _require_http_same_origin(request: Request) -> None:
    """Reject browser-CSRF attempts against loopback state-changing routes."""
    if not _origin_is_allowed(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="origin ไม่ได้รับอนุญาต")


def _cleanup_attachments(*, keep: Path | None = None) -> None:
    """Bound stale private uploads without touching the active new upload."""
    try:
        entries = [
            item for item in _ATTACHMENTS_DIR.iterdir()
            if item.name.startswith(_MANAGED_ATTACHMENT_PREFIX) and item.is_file() and not item.is_symlink()
        ]
    except OSError:
        return
    now = time.time()
    retained = []
    for item in entries:
        try:
            stat = item.stat()
        except OSError:
            continue
        if item != keep and now - stat.st_mtime > _ATTACHMENT_MAX_AGE_SECONDS:
            try:
                item.unlink()
            except OSError:
                pass
        else:
            retained.append((stat.st_mtime, stat.st_size, item))
    total = sum(size for _, size, _ in retained)
    for _, size, item in sorted(retained):
        if total <= _ATTACHMENT_MAX_TOTAL_BYTES:
            break
        if item == keep:
            continue
        try:
            item.unlink()
            total -= size
        except OSError:
            pass


def _workspace_path(raw: str) -> Path:
    """Resolve a browser file request without allowing directory traversal."""
    root = config.WORKSPECT_DIR.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("เข้าถึงได้เฉพาะ workspect") from exc
    return resolved


def _reveal_local_path(raw: str) -> None:
    """Reveal a safe, existing local path in Finder.

    This is intentionally a WebSocket-only action: the browser UI already
    verifies its loopback Origin before accepting WebSocket commands, so an
    unrelated web page cannot make Finder open paths on the user's machine.
    """
    if not raw or len(raw) > 4_096:
        raise ValueError("path ไม่ถูกต้อง")
    # A response renderer normally removes Markdown wrappers, but accept the
    # common wrappers here too.  That makes a stale browser tab harmless and
    # prevents a closing code tick or bold marker from becoming part of a
    # Finder path.
    raw = raw.strip().strip("`*").rstrip("),.;:!?")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ValueError("เปิดได้เฉพาะ absolute path หรือ ~/path")
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("ไม่พบ path นี้แล้ว") from exc
    if not (path.is_file() or path.is_dir()) or path_is_or_contains_secret(path):
        raise ValueError("ไม่อนุญาตให้เปิด path นี้")

    # ``open -R`` selects a file in Finder; opening a directory directly is
    # more useful than selecting it in its parent.
    command = ["open", "-R", str(path)] if path.is_file() else ["open", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("เปิด Finder ไม่สำเร็จ") from exc
    if result.returncode:
        raise ValueError("เปิด Finder ไม่สำเร็จ")


def _files_for(path: Path) -> tuple[list[dict], bool]:
    if not path.is_dir():
        raise ValueError("path ไม่ใช่โฟลเดอร์")

    def iter_rows():
        for item in path.iterdir():
            # A listing exposes only names; content is rechecked immediately
            # before open. Running the content scanner here used to read up
            # to 2 MiB from every ordinary file just to render a directory.
            if item.name == ".lite_state" or is_sensitive_path(item):
                continue
            try:
                item_type = "dir" if item.is_dir() else "file"
                yield {
                    "name": item.name, "path": str(item), "type": item_type,
                    "size": item.stat().st_size,
                }
            except OSError:
                continue

    rows = heapq.nsmallest(
        _MAX_BROWSE_ENTRIES + 1,
        iter_rows(),
        key=lambda row: (row["type"] != "dir", row["name"].lower()),
    )
    truncated = len(rows) > _MAX_BROWSE_ENTRIES
    return rows[:_MAX_BROWSE_ENTRIES], truncated


def _files_event(path: Path) -> dict:
    rows, truncated = _files_for(path)
    return {
        "type": "files", "files": rows, "path": str(path),
        "root": str(config.WORKSPECT_DIR), "truncated": truncated,
        "limit": _MAX_BROWSE_ENTRIES,
    }


def _history_pairs(messages: list[dict]) -> list[dict]:
    pairs = []
    for i in range(0, len(messages) - 1, 2):
        q, answer = messages[i : i + 2]
        if q.get("role") == "user" and answer.get("role") == "assistant":
            pairs.append({"q": q.get("content", ""), "a": answer.get("content", "")})
    return pairs


def _status() -> dict:
    return {
        "type": "status", "model": config.MODEL_ID, "server_up": _mlx_server_up(),
        "server_port": config.MLX_SERVER_PORT,
        "online": _internet_available(),
        "history_chars": 0, "history_max_chars": agent.MAX_HISTORY_CHARS,
        "skill": "", "skills": [],
        "builtin_cmds": [
            {"name": "index", "desc": "สร้างหรืออัปเดต RAG file index"},
            {"name": "clear", "desc": "ล้างบทสนทนาปัจจุบัน"},
            {"name": "history", "desc": "ดูประวัติใน session นี้"},
        ],
    }


def _internet_available() -> bool:
    """Small, bounded route check for the UI's Internet/Offline badge."""
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.75):
            return True
    except OSError:
        return False


def _mlx_server_up() -> bool:
    """Return true only when the configured OpenAI endpoint is ready.

    A TCP listener can be an unrelated process or an MLX server that is still
    loading its model.  `/v1/models` verifies both protocol readiness and the
    model selected in Lite's effective configuration.
    """
    try:
        headers = {"Authorization": f"Bearer {config.MLX_API_KEY}"} if config.MLX_API_KEY else {}
        request = URLRequest(
            f"{config.MLX_BASE_URL.rstrip('/')}/models",
            headers=headers,
        )
        with urlopen(request, timeout=0.75) as response:
            payload = json.loads(response.read(64 * 1024).decode("utf-8"))
        models = payload.get("data", [])
        return isinstance(models, list) and any(
            isinstance(item, dict) and item.get("id") == config.MODEL_ID
            for item in models
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _terminate_pid(pid: int) -> None:
    """SIGTERM then, after a grace period, SIGKILL a single pid.

    SIGTERM first so mlx_vlm.server can release its GPU/unified-memory
    allocation cleanly; escalate to SIGKILL only for stragglers so the RAM is
    reliably back before the caller (start or stop) proceeds.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # probe: still alive?
        except OSError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _stop_managed_mlx_server() -> bool:
    """Stop only the model server launched by this Web UI process.

    A local port can belong to another application.  The UI must never kill
    an untracked process merely because it happens to listen on the configured
    MLX port.
    """
    global _mlx_server_proc
    if _mlx_server_proc is None or _mlx_server_proc.poll() is not None:
        _mlx_server_proc = None
        return False
    _terminate_pid(_mlx_server_proc.pid)
    _mlx_server_proc = None
    return True


def _start_mlx_server() -> str | None:
    """Launch a fresh managed server, without taking over an occupied port."""
    global _mlx_server_proc
    if _mlx_server_proc is not None and _mlx_server_proc.poll() is None:
        return "Model server นี้ถูกเปิดโดย Web UI อยู่แล้ว"
    if _mlx_server_up():
        return (
            f"Port {config.MLX_SERVER_PORT} ถูกใช้งานอยู่แล้ว; "
            "Web UI จะไม่หยุด process ที่ไม่ได้เริ่มเอง"
        )
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_MLX_SERVER_LOG, "ab") as log_fh:
        _mlx_server_proc = subprocess.Popen(
            [
                str(config.MLX_SERVER_PYTHON), "-m", "mlx_vlm.server",
                "--model", config.MODEL_ID,
                "--host", config.MLX_SERVER_HOST,
                "--port", str(config.MLX_SERVER_PORT),
                "--api-key", config.MLX_API_KEY,
                "--vision-cache-size", str(config.MLX_SERVER_VISION_CACHE_SIZE),
                "--prefill-step-size", str(config.MLX_SERVER_PREFILL_STEP_SIZE),
            ],
            cwd=str(config.PROJECT_ROOT),
            stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return None


def _activity_args(args: object) -> dict[str, object]:
    """Return tool arguments safe to expose in the browser activity panel.

    The panel is useful precisely because it shows the actual query or shell
    command.  Keep that observability without bypassing the UI's credential
    boundary: hide credential-shaped values and protected paths before an
    argument ever crosses the WebSocket.
    """
    if not isinstance(args, dict):
        return {}

    safe: dict[str, object] = {}
    for key, value in args.items():
        text = str(value)
        if contains_secret_content(text):
            safe[str(key)] = "[redacted: sensitive value]"
        elif str(key).lower() in {"path", "cwd"} and path_is_or_contains_secret(text):
            safe[str(key)] = "[redacted: sensitive path]"
        else:
            safe[str(key)] = text[:_ACTIVITY_VALUE_MAX_CHARS]
    return safe


def _attachment_image_path(raw: object) -> str | None:
    """Accept only an uploaded image path owned by Lite's attachment store."""
    if not isinstance(raw, dict) or raw.get("isImage") is not True:
        return None
    candidate = raw.get("path")
    if not isinstance(candidate, str):
        return None
    try:
        path = Path(candidate).resolve(strict=True)
        path.relative_to(_ATTACHMENTS_DIR.resolve())
    except (OSError, ValueError):
        return None
    return str(path) if path.is_file() and path.suffix.lower() in _IMAGE_ATTACHMENT_EXTENSIONS else None


async def _run_query(websocket: WebSocket, session: BrowserSession, text: str, attachment_image: str | None = None) -> None:
    loop = asyncio.get_running_loop()
    await websocket.send_json({"type": "start"})
    token_parts: list[str] = []
    token_chars = 0
    last_token_flush = 0.0

    def send(payload: dict) -> None:
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
            future.result(timeout=5)
        except Exception:
            if future is not None:
                future.cancel()

    def flush_tokens() -> None:
        nonlocal token_chars, last_token_flush
        if not token_parts:
            return
        text = "".join(token_parts)
        token_parts.clear()
        token_chars = 0
        send({"type": "token", "text": text})
        last_token_flush = time.monotonic()

    def on_step(**event) -> None:
        nonlocal token_chars, last_token_flush
        if event["event"] == "tool_start":
            name = event["name"]
            args = _activity_args(event.get("args"))
            send({"type": "tool", "name": name, "args": args, "detail": "กำลังทำงาน…"})
            send({"type": "progress", "msg": f"{name}: กำลังทำงาน…"})
        elif event["event"] == "tool_done":
            send({"type": "tool", "name": event["name"], "detail": "เสร็จแล้ว" if event["ok"] else "ไม่สำเร็จ"})
        elif event["event"] == "stream_start":
            flush_tokens()
            send({"type": "synthesis_start"})
        elif event["event"] == "token":
            piece = event.get("text", "")
            if not piece:
                return
            token_parts.append(piece)
            token_chars += len(piece)
            now = time.monotonic()
            # First token is immediate. Afterwards batch at <=25fps to keep
            # markdown rendering and cross-thread WebSocket sends lightweight.
            if last_token_flush == 0.0 or now - last_token_flush >= 0.04 or token_chars >= 1024:
                flush_tokens()
        elif event["event"] == "stream_end":
            flush_tokens()
        elif event["event"] == "discard_stream":
            token_parts.clear()
            token_chars = 0
            send({"type": "discard_stream"})

    cancel_event = threading.Event()
    session.cancel_event = cancel_event
    if _HEAVY_WORK_LOCK.locked():
        await websocket.send_json({"type": "progress", "msg": "กำลังสร้าง RAG index อยู่ — คำถามนี้เข้าคิวรอ…"})
    try:
        async with _HEAVY_WORK_LOCK:
            result = await asyncio.to_thread(
                agent.run_turn, text, session.messages, on_step, attachment_image, cancel_event, True,
            )
    finally:
        session.cancel_event = None

    if result.cancelled:
        await websocket.send_json({"type": "cancelled"})
        return
    flush_tokens()
    await websocket.send_json({"type": "response", "content": result.text})
    await websocket.send_json({"type": "ctx_update", "chars": sum(len(m.get("content", "")) for m in session.messages), "max_chars": agent.MAX_HISTORY_CHARS})
    await websocket.send_json({"type": "done"})


async def _run_index_subprocess(websocket: WebSocket) -> tuple[dict | None, bool, str]:
    """Run indexer_worker.py as a short-lived child process, relaying its
    bounded JSONL progress to the browser as it arrives.

    Returns (stats, crashed, detail):
    - stats is the final dict on a normal completion.
    - stats is None either because the run was deliberately cancelled
      (_index_cancel_requested, set by the "/stop" handler) or because the
      child died on its own (crashed=True) -- these must not be conflated,
      or a genuine failure gets silently reported to the user as if they'd
      clicked stop themselves.
    - detail carries a bounded tail of any stray (non-JSON) output for the
      crashed case only, to give the user/log something to act on.
    """
    global _active_index_process, _index_cancel_requested
    _index_cancel_requested = False
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(_INDEX_WORKER),
        stdout=asyncio.subprocess.PIPE,
        # Merged into stdout rather than its own PIPE: nothing here drains
        # stderr separately, and an unread pipe that fills up would block
        # the child's write and deadlock this whole relay loop. Any stray
        # non-JSON stderr line is tolerated below (kept as diagnostic
        # detail, never fatal to the relay loop itself).
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(config.PROJECT_ROOT),
    )
    _active_index_process = process
    stats: dict | None = None
    unparsed_tail: list[str] = []
    try:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    unparsed_tail.append(text)
                    del unparsed_tail[:-20]  # bounded: keep only the most recent lines
                continue
            if event.get("event") == "file":
                path = Path(event["path"]) if event.get("path") else None
                await websocket.send_json({"type": "progress", "msg": ui.index_progress_text(path, event.get("stats", {}))})
            elif event.get("event") == "stale":
                path = Path(event["path"]) if event.get("path") else None
                await websocket.send_json({"type": "progress", "msg": ui.index_stale_progress_text(path, event.get("stats", {}))})
            elif event.get("event") == "done":
                stats = event.get("stats")
        await process.wait()
    finally:
        cancel_requested = _index_cancel_requested
        if process.returncode is None:
            # Disconnect or /stop terminated us while the child was still
            # running -- stop it too rather than leaving an orphaned scan.
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        _active_index_process = None

    crashed = stats is None and not cancel_requested
    detail = ("\n".join(unparsed_tail))[-_MAX_UNPARSED_TAIL_CHARS:] if crashed else ""
    return stats, crashed, detail


async def _run_index(websocket: WebSocket, session: "BrowserSession") -> None:
    global _active_index_owner
    if _active_index_process is not None:
        await websocket.send_json({"type": "error", "msg": "กำลังสร้าง RAG index อยู่แล้วในอีกหน้าต่างหนึ่ง"})
        return
    if _HEAVY_WORK_LOCK.locked():
        await websocket.send_json({"type": "progress", "msg": "กำลังตอบคำถามอยู่ — index รอคิว…"})
    # Not "start": that event is the LLM-query flow's signal to reset the
    # spinner to a hardcoded 'thinking…' label and open a streaming assistant
    # bubble -- neither applies here and would clobber the
    # "กำลังสร้าง RAG index…" label already showing. "busy" only disables
    # the input while the index subprocess runs.
    await websocket.send_json({"type": "busy"})

    async with _HEAVY_WORK_LOCK:
        _active_index_owner = session
        try:
            stats, crashed, detail = await _run_index_subprocess(websocket)
        finally:
            _active_index_owner = None

    if crashed:
        msg = "สร้าง RAG index ล้มเหลว (โปรเซสหยุดกะทันหัน)" + (f": {detail}" if detail else "")
        await websocket.send_json({"type": "error", "msg": msg})
        return
    if stats is None:
        await websocket.send_json({"type": "cancelled"})
        return
    summary = (f"✓ สร้าง RAG index เสร็จ: {stats['scanned']:,} scanned, "
               f"{stats['inserted']:,} new, {stats['updated']:,} updated, "
               f"{stats['skipped']:,} skipped, {stats['error']:,} errors ({stats['seconds']:.2f}s)")
    await websocket.send_json({"type": "response", "content": summary})
    await websocket.send_json({"type": "done"})


async def _ws_reader(websocket: WebSocket, session: BrowserSession, ws_in_q: "asyncio.Queue[dict]") -> None:
    """Owns websocket.receive() exclusively so a '/stop' command reaches a
    turn while it's still running -- the main loop is otherwise blocked
    awaiting agent.run_turn in a worker thread and can't call receive()
    again until that returns. '/stop' is handled right here and never
    forwarded, so it never needs to send anything itself; this keeps
    websocket.send_json single-sender (only the main loop / _run_query ever
    call it), avoiding interleaved-frame corruption.

    Never cancelled while inside receive() (a cancelled Starlette WS receive
    corrupts its internal state so subsequent reads silently never
    complete) -- this loop only ever exits on its own, via a disconnect.
    """
    while True:
        try:
            incoming = await websocket.receive()
        except WebSocketDisconnect:
            if session.cancel_event is not None:
                session.cancel_event.set()
            while not ws_in_q.empty():
                ws_in_q.get_nowait()
            ws_in_q.put_nowait({"type": "websocket.disconnect"})
            return
        if incoming["type"] == "websocket.disconnect":
            if session.cancel_event is not None:
                session.cancel_event.set()
            while not ws_in_q.empty():
                ws_in_q.get_nowait()
            ws_in_q.put_nowait(incoming)
            return
        if incoming.get("bytes") is None:
            raw = incoming.get("text")
            if isinstance(raw, str) and len(raw) <= _MAX_WS_MESSAGE_CHARS:
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if (
                    isinstance(parsed, dict) and parsed.get("type") == "command"
                    and str(parsed.get("cmd", "")).strip().lower() == "/stop"
                ):
                    ev = session.cancel_event
                    if ev is not None:
                        ev.set()
                    if (
                        _active_index_process is not None
                        and _active_index_process.returncode is None
                        and _active_index_owner is session  # only this tab's own index job
                    ):
                        global _index_cancel_requested
                        _index_cancel_requested = True
                        _active_index_process.terminate()
                    continue
        try:
            ws_in_q.put_nowait(incoming)
        except asyncio.QueueFull:
            # A normal UI disables input while a turn is running. A broken or
            # hostile client can still flood valid frames while the consumer
            # is busy; cancel instead of retaining an unbounded backlog.
            if session.cancel_event is not None:
                session.cancel_event.set()
            while not ws_in_q.empty():
                ws_in_q.get_nowait()
            ws_in_q.put_nowait({"type": "lite.inbox_overflow"})
            return


@app.get("/")
async def chat_page():
    # The renderer is a single static HTML file.  Never reuse an old copy
    # after the local launcher replaces the backend; stale JavaScript would
    # silently discard new WebSocket fields such as tool arguments.
    return FileResponse(_HTML, headers={"Cache-Control": "no-store"})


@app.post("/upload")
async def upload_attachment(request: Request):
    """Copy one browser-selected attachment into Lite's private workspace.

    Browser file pickers deliberately do not expose an absolute local path.
    Keeping a bounded copy under workspect gives the existing read_file and
    read_image tools a normal, safe path without broadening their filesystem
    permissions or serving the uploaded bytes back to the browser.
    """
    _require_http_same_origin(request)
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > _MAX_ATTACHMENT_BYTES):
        return {"ok": False, "error": f"ไฟล์ใหญ่เกิน {_MAX_ATTACHMENT_BYTES // 1024 // 1024}MB"}

    original_name = unquote(request.headers.get("x-file-name", ""))
    name = Path(original_name).name.strip()
    if not name or name in {".", ".."}:
        return {"ok": False, "error": "ชื่อไฟล์ไม่ถูกต้อง"}

    data = await request.body()
    if not data:
        return {"ok": False, "error": "ไฟล์ว่างเปล่า"}
    if len(data) > _MAX_ATTACHMENT_BYTES:
        return {"ok": False, "error": f"ไฟล์ใหญ่เกิน {_MAX_ATTACHMENT_BYTES // 1024 // 1024}MB"}

    try:
        _ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
        _ATTACHMENTS_DIR.chmod(0o700)
        _cleanup_attachments()
        suffix = Path(name).suffix[:16]
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=_MANAGED_ATTACHMENT_PREFIX, suffix=suffix,
            dir=_ATTACHMENTS_DIR, delete=False,
        ) as handle:
            handle.write(data)
            stored_path = Path(handle.name)
        _cleanup_attachments(keep=stored_path)
    except OSError:
        return {"ok": False, "error": "บันทึกไฟล์แนบไม่ได้"}

    return {
        "ok": True, "path": str(stored_path), "name": name,
        "isImage": stored_path.suffix.lower() in _IMAGE_ATTACHMENT_EXTENSIONS,
    }


@app.get("/ui-token")
async def ui_token():
    """Compatibility endpoint for the copied UI; loopback needs no token."""
    return {"token": ""}


@app.get("/server/status")
async def server_status():
    return {"server_up": await asyncio.to_thread(_mlx_server_up)}


@app.post("/server/start")
async def server_start(request: Request):
    """Launch a managed mlx_vlm.server when its port is available."""
    _require_http_same_origin(request)
    if _MLX_SERVER_LOCK.locked():
        return {"ok": False, "error": "กำลังเปิด/ปิด server อยู่ — รอสักครู่"}
    def _do() -> str | None:
        with _MLX_SERVER_LOCK:
            return _start_mlx_server()
    error = await asyncio.to_thread(_do)
    return {"ok": error is None, "error": error or ""}


@app.post("/server/stop")
async def server_stop(request: Request):
    """Stop the model process launched by this Web UI, if one exists."""
    _require_http_same_origin(request)
    if _MLX_SERVER_LOCK.locked():
        return {"ok": False, "error": "กำลังเปิด/ปิด server อยู่ — รอสักครู่"}
    def _do() -> None:
        with _MLX_SERVER_LOCK:
            _stop_managed_mlx_server()
    await asyncio.to_thread(_do)
    return {"ok": True, "server_up": await asyncio.to_thread(_mlx_server_up)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Browsers allow a web page from another origin to initiate a WebSocket to
    # localhost.  Reject it before accept so a hostile site cannot drive Lite's
    # filesystem-capable agent through the user's browser.
    origin = websocket.headers.get("origin")
    if not _origin_is_allowed(origin):
        await websocket.close(code=1008)
        return
    if _CONNECTIONS.locked():
        await websocket.close(code=1013)
        return
    await _CONNECTIONS.acquire()
    reader_task: "asyncio.Task | None" = None
    try:
        await websocket.accept()
        session = BrowserSession()
        await websocket.send_json(_status())
        await websocket.send_json(_files_event(config.WORKSPECT_DIR))
        ws_in_q: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=_MAX_WS_INBOX_ITEMS)
        reader_task = asyncio.create_task(_ws_reader(websocket, session, ws_in_q))
        while True:
            incoming = await ws_in_q.get()
            if incoming["type"] == "websocket.disconnect":
                return
            if incoming["type"] == "lite.inbox_overflow":
                await websocket.send_json({"type": "error", "msg": "มีคำขอค้างมากเกินไป กรุณาเชื่อมต่อใหม่"})
                return
            if incoming.get("bytes") is not None:
                await websocket.send_json({"type": "error", "msg": "รองรับเฉพาะข้อความ JSON"})
                continue
            raw = incoming.get("text")
            if not isinstance(raw, str):
                await websocket.send_json({"type": "error", "msg": "รองรับเฉพาะข้อความ JSON"})
                continue
            if len(raw) > _MAX_WS_MESSAGE_CHARS:
                await websocket.send_json({"type": "error", "msg": "คำขอใหญ่เกินขีดจำกัด"})
                continue
            try:
                request = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                await websocket.send_json({"type": "error", "msg": "รูปแบบ JSON ไม่ถูกต้อง"})
                continue
            if not isinstance(request, dict):
                await websocket.send_json({"type": "error", "msg": "คำขอต้องเป็น JSON object"})
                continue
            kind = request.get("type")
            if not isinstance(kind, str):
                await websocket.send_json({"type": "error", "msg": "type ต้องเป็นข้อความ"})
                continue
            if kind == "query":
                content = request.get("content", "")
                if not isinstance(content, str):
                    await websocket.send_json({"type": "error", "msg": "content ต้องเป็นข้อความ"})
                    continue
                text = content.strip()
                if text:
                    await _run_query(websocket, session, text, _attachment_image_path(request.get("attachment")))
            elif kind == "command":
                command_raw = request.get("cmd", "")
                if not isinstance(command_raw, str):
                    await websocket.send_json({"type": "error", "msg": "cmd ต้องเป็นข้อความ"})
                    continue
                command = command_raw.lower()
                if command == "/index":
                    await _run_index(websocket, session)
                elif command == "/clear":
                    session.messages.clear()
                    await websocket.send_json({"type": "clear_ok"})
                elif command == "/history":
                    pairs = _history_pairs(session.messages)
                    await websocket.send_json({"type": "history_list", "pairs": pairs, "total": len(pairs)})
                    await websocket.send_json({"type": "memory_ok", "turns": len(pairs), "total": sum(len(str(p)) for p in pairs), "topics": "Session history"})
                else:
                    await websocket.send_json({"type": "error", "msg": f"ไม่รู้จักคำสั่ง: {command}"})
            elif kind == "get_history":
                pairs = _history_pairs(session.messages)
                await websocket.send_json({"type": "history_list", "pairs": pairs, "total": len(pairs)})
            elif kind == "get_files":
                try:
                    raw_path = request.get("path", str(config.WORKSPECT_DIR))
                    if not isinstance(raw_path, str):
                        raise ValueError("path ต้องเป็นข้อความ")
                    path = _workspace_path(raw_path)
                    await websocket.send_json(_files_event(path))
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "msg": str(exc)})
            elif kind == "open_file":
                try:
                    raw_path = request.get("path", "")
                    if not isinstance(raw_path, str):
                        raise ValueError("path ต้องเป็นข้อความ")
                    path = _workspace_path(raw_path)
                    if not path.is_file() or path_is_or_contains_secret(path):
                        raise ValueError("ไม่อนุญาตให้อ่านไฟล์นี้")
                    with path.open("rb") as fh:
                        data = fh.read(_MAX_VIEW_BYTES + 1)
                    if len(data) > _MAX_VIEW_BYTES:
                        raise ValueError(f"ดูได้เฉพาะไฟล์ไม่เกิน {_MAX_VIEW_BYTES // 1024}KB")
                    if b"\0" not in data[:4096] and contains_secret_content(data.decode("utf-8", errors="replace")):
                        raise ValueError("ไม่อนุญาตให้แสดงไฟล์ที่มี credential หรือ token")
                    mime, _ = mimetypes.guess_type(path.name)
                    if mime and mime.startswith("image/"):
                        encoded = base64.b64encode(data).decode("ascii")
                        await websocket.send_json({"type": "file_image", "path": str(path), "data_url": f"data:{mime};base64,{encoded}"})
                    else:
                        await websocket.send_json({"type": "file_content", "path": str(path), "content": data.decode("utf-8", errors="replace")})
                except (OSError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "msg": str(exc)})
            elif kind == "reveal_path":
                raw_path = request.get("path", "")
                if not isinstance(raw_path, str):
                    await websocket.send_json({"type": "error", "msg": "path ต้องเป็นข้อความ"})
                    continue
                try:
                    await asyncio.to_thread(_reveal_local_path, raw_path)
                    await websocket.send_json({"type": "path_revealed"})
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "msg": str(exc)})
            else:
                await websocket.send_json({"type": "error", "msg": "คำขอไม่รองรับ"})
    except WebSocketDisconnect:
        return
    finally:
        if reader_task is not None:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
        _CONNECTIONS.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.UI_HOST, port=config.UI_PORT, ws_max_size=_MAX_WS_MESSAGE_CHARS)
