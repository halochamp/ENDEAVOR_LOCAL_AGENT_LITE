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
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import unquote
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

import agent
import config
import ui
import rag_mcp_service
import pdf_to_text
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
_MAX_CONNECTIONS = 4
# Uploads are runtime-private state. Keep them beside the per-machine SQLite
# and OCR snapshots, not in the iCloud-synced user workspace.
_ATTACHMENTS_DIR = config.DATA_DIR / "attachments"
_MANAGED_ATTACHMENT_PREFIX = "lite-attachment-"
_IMAGE_ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic", ".heif", ".tiff", ".tif"}
_ALLOWED_ORIGINS = {
    f"http://127.0.0.1:{config.UI_PORT}",
    f"http://localhost:{config.UI_PORT}",
}
_INDEX_WORKER = config.PROJECT_ROOT / "indexer_worker.py"
_INDEX_MAX_RUNTIME_SECONDS = config.INDEX_TIMEOUT_SECONDS
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
_RAG_SYNC_STATE_LOCK = threading.RLock()
_rag_sync_state: dict | None = None
_rag_sync_task: "asyncio.Task | None" = None
_rag_sync_counter = 0
_PDF_TEXT_STATE_LOCK = threading.RLock()
_pdf_text_state: dict | None = None
_pdf_text_task: "asyncio.Task | None" = None
_pdf_text_counter = 0
_PDF_TEXT_UPLOAD_DIR = config.DATA_DIR / "pdf_to_text" / "uploads"
_MANAGED_PDF_TEXT_PREFIX = "lite-pdf-to-text-"


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
_SERVER_LOG_MAX_BYTES = 5 * 1024 * 1024
_SERVER_LOG_BACKUP_COUNT = 3
_mlx_server_proc: "subprocess.Popen | None" = None
_MODEL_PROBE_TIMEOUT_SECONDS = 1.0


class _AttachmentTooLarge(Exception):
    pass


@dataclass
class BrowserSession:
    messages: list[dict] = field(default_factory=list)
    cancel_event: "threading.Event | None" = None
    effort: str = config.DEFAULT_THINKING_EFFORT


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
            # before open.  Running the content scanner here used to read up
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


def _status(session: BrowserSession | None = None) -> dict:
    effort = config.normalize_thinking_effort(
        session.effort if session is not None else config.DEFAULT_THINKING_EFFORT
    )
    server_state = _mlx_server_state()
    return {
        "type": "status", "model": config.MODEL_ID, "server_up": server_state == "ready",
        "server_state": server_state,
        "server_port": config.MLX_SERVER_PORT,
        "online": _internet_available(),
        "history_chars": 0, "history_max_chars": agent.MAX_HISTORY_CHARS,
        "skill": "", "skills": [],
        "effort": effort,
        "thinking_budget": config.thinking_budget_for_effort(effort),
        "thinking_budgets": dict(config.THINKING_BUDGET_PRESETS),
        "builtin_cmds": [
            {"name": "index", "desc": "สร้างหรืออัปเดต RAG file index"},
            {"name": "sync_kb", "desc": "stream rag_document/ documents -> searchable SQLite chunks"},
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
    try:
        with socket.create_connection((config.MLX_SERVER_HOST, config.MLX_SERVER_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _mlx_server_state() -> str:
    """Return readiness, not merely TCP-listener state.

    mlx_vlm can have a live socket while its model is still loading. Calling
    the OpenAI-compatible model listing also catches a stale server serving a
    different model, which otherwise looks healthy until the first user turn.
    """
    if not _mlx_server_up():
        return "down"
    host = f"[{config.MLX_SERVER_HOST}]" if ":" in config.MLX_SERVER_HOST else config.MLX_SERVER_HOST
    url = f"http://{host}:{config.MLX_SERVER_PORT}/v1/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {config.MLX_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_MODEL_PROBE_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return "starting"
            payload = json.loads(response.read(128 * 1024).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return "auth_error" if exc.code in {401, 403} else "starting"
    except (OSError, TimeoutError, UnicodeError, json.JSONDecodeError):
        return "starting"

    model_ids = {
        str(item.get("id")) for item in (payload.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    }
    if config.MODEL_ID in model_ids:
        return "ready"
    return "wrong_model" if model_ids else "starting"


async def _acquire_heavy_work(cancel_event: threading.Event) -> bool:
    """Acquire the global heavy-work gate while honoring browser cancellation."""
    while True:
        if cancel_event.is_set():
            return False
        # Poll instead of leaving a timed-out waiter on the lock. Besides
        # making cancellation prompt, this keeps the test/CLI call path safe
        # when an asyncio lock is reused after a short-lived event loop closes.
        # The check and acquire have no await between them, so another task in
        # this loop cannot take the unlocked gate in the middle.
        if not _HEAVY_WORK_LOCK.locked():
            await _HEAVY_WORK_LOCK.acquire()
            if cancel_event.is_set():
                _HEAVY_WORK_LOCK.release()
                return False
            return True
        await asyncio.sleep(0.05)


def _pids_on_port(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(pid) for pid in result.stdout.split() if pid.isdigit()]


def _process_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def _process_uid(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "uid="],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _option_value(tokens: list[str], option: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(option + "="):
            return token[len(option) + 1:]
    return None


def _is_managed_mlx_pid(pid: int, port: int) -> bool:
    """Only recognize our exact local model command before sending a signal."""
    if _mlx_server_proc is not None and pid == _mlx_server_proc.pid:
        return True
    if _process_uid(pid) != os.getuid():
        return False
    try:
        tokens = shlex.split(_process_command(pid))
    except ValueError:
        return False
    try:
        module_index = tokens.index("-m")
    except ValueError:
        return False
    return (
        module_index + 1 < len(tokens)
        and tokens[module_index + 1] == "mlx_vlm.server"
        and _option_value(tokens, "--port") == str(port)
        and _option_value(tokens, "--host") == config.MLX_SERVER_HOST
        and _option_value(tokens, "--model") == config.MODEL_ID
    )


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


def _rotate_log_file(path: Path) -> None:
    """Bound a service log before a browser-triggered server restart."""
    try:
        if path.stat().st_size <= _SERVER_LOG_MAX_BYTES:
            return
    except OSError:
        return
    for index in range(_SERVER_LOG_BACKUP_COUNT, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        target = path.with_name(f"{path.name}.{index + 1}")
        try:
            if index == _SERVER_LOG_BACKUP_COUNT:
                source.unlink(missing_ok=True)
            elif source.exists():
                os.replace(source, target)
        except OSError:
            pass
    try:
        os.replace(path, path.with_name(f"{path.name}.1"))
    except OSError:
        pass


def _open_private_append(path: Path):
    """Open a log with restrictive permissions even when umask is permissive."""
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
    except OSError:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "ab")


def _kill_mlx_server(port: int) -> None:
    """Free Lite's model server without killing an unrelated port owner.

    The tracked pid catches a server still loading before it binds. A stale
    listener from a previous Lite session is accepted only when its uid and
    command line match this exact model/host/port. An unrelated service is
    left untouched and reported to the caller.
    """
    global _mlx_server_proc
    pids = set(_pids_on_port(port))
    if _mlx_server_proc is not None and _mlx_server_proc.poll() is None:
        pids.add(_mlx_server_proc.pid)
    unmanaged = [pid for pid in pids if not _is_managed_mlx_pid(pid, port)]
    if unmanaged:
        raise RuntimeError(
            f"ไม่หยุด process อื่นที่ใช้พอร์ต model {port} (pid {unmanaged[0]})"
        )
    for pid in pids:
        _terminate_pid(pid)
    _mlx_server_proc = None


def _start_mlx_server() -> None:
    """Stop a stale Lite server, then launch a fresh server.

    Runs under _MLX_SERVER_LOCK (via the /server/start route) so two rapid
    clicks can't race and spawn two servers on the same port. Launched
    detached (start_new_session) so it outlives this request/app process,
    matching how the user runs it manually per walking_test.md.
    """
    global _mlx_server_proc
    python = Path(config.MLX_SERVER_PYTHON).expanduser()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"ไม่พบ Lite Python ที่ใช้งานได้: {python}")
    _kill_mlx_server(config.MLX_SERVER_PORT)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        config.LOG_DIR.chmod(0o700)
    except OSError:
        pass
    try:
        _rotate_log_file(_MLX_SERVER_LOG)
        with _open_private_append(_MLX_SERVER_LOG) as log_fh:
            _mlx_server_proc = subprocess.Popen(
                [
                    str(python), "-m", "mlx_vlm.server",
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
    except OSError as exc:
        _mlx_server_proc = None
        raise RuntimeError(f"เปิด Lite model server ไม่สำเร็จ: {type(exc).__name__}") from exc


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
    cancel_event = threading.Event()
    session.cancel_event = cancel_event
    token_parts: list[str] = []
    token_chars = 0
    last_token_flush = 0.0

    def send(payload: dict) -> None:
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
            future.result(timeout=5)
        except Exception:
            cancel_event.set()
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
            # First token is immediate.  Afterwards batch at <=25fps to keep
            # markdown rendering and cross-thread WebSocket sends lightweight.
            if last_token_flush == 0.0 or now - last_token_flush >= 0.04 or token_chars >= 1024:
                flush_tokens()
        elif event["event"] == "stream_end":
            flush_tokens()
        elif event["event"] == "discard_stream":
            token_parts.clear()
            token_chars = 0
            send({"type": "discard_stream"})

    try:
        await websocket.send_json({"type": "start"})
        if _HEAVY_WORK_LOCK.locked():
            await websocket.send_json({"type": "progress", "msg": "กำลังสร้าง RAG index อยู่ — คำถามนี้เข้าคิวรอ…"})
        if not await _acquire_heavy_work(cancel_event):
            await websocket.send_json({"type": "cancelled"})
            return
        try:
            result = await asyncio.to_thread(
                agent.run_turn, text, session.messages, on_step, attachment_image, cancel_event, True,
                effort=session.effort,
            )
        finally:
            _HEAVY_WORK_LOCK.release()

        if result.cancelled:
            await websocket.send_json({"type": "cancelled"})
            return
        flush_tokens()
        await websocket.send_json({"type": "response", "content": result.text})
        await websocket.send_json({"type": "ctx_update", "chars": sum(len(m.get("content", "")) for m in session.messages), "max_chars": agent.MAX_HISTORY_CHARS})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        cancel_event.set()
    except Exception as exc:
        cancel_event.set()
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "msg": f"การทำงานของ Agent Lite ล้มเหลว: {type(exc).__name__}"})
    finally:
        if session.cancel_event is cancel_event:
            session.cancel_event = None


async def _run_index_subprocess(
    websocket: WebSocket, *, sync_kb: bool = False,
    cancel_event: threading.Event | None = None,
) -> tuple[dict | None, bool, str]:
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
    command = [sys.executable, str(_INDEX_WORKER)]
    if sync_kb:
        command.append("--sync-kb")
    child_env = {
        key: value for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "PYTHONNOUSERSITE", "TMPDIR"}
        or (key.startswith("LITE_") and key != "LITE_MLX_API_KEY")
    }
    data_dir = Path(config.DATA_DIR).expanduser().resolve(strict=False)
    if config._state_path_allowed(data_dir):
        child_env["LITE_DATA_DIR"] = str(data_dir)
    file_index_db = Path(config.FILE_INDEX_DB).expanduser().resolve(strict=False)
    if config._state_path_allowed(file_index_db):
        child_env["LITE_FILE_INDEX_DB"] = str(file_index_db)
    roots: list[str] = []
    for root in getattr(config, "INDEX_ROOTS", ()):
        try:
            roots.append(str(Path(root).expanduser().resolve(strict=False)))
        except (OSError, RuntimeError, TypeError):
            continue
    child_env["LITE_INDEX_ROOTS"] = ",".join(dict.fromkeys(roots))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        # Merged into stdout rather than its own PIPE: nothing here drains
        # stderr separately, and an unread pipe that fills up would block
        # the child's write and deadlock this whole relay loop. Any stray
        # non-JSON stderr line is tolerated below (kept as diagnostic
        # detail, never fatal to the relay loop itself).
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(config.PROJECT_ROOT),
        env=child_env,
    )
    _active_index_process = process
    stats: dict | None = None
    unparsed_tail: list[str] = []
    read_task: asyncio.Task | None = None
    cancel_sent = False
    send_progress = True
    timed_out = False
    started_at = time.monotonic()
    try:
        assert process.stdout is not None
        read_task = asyncio.create_task(process.stdout.readline())
        while True:
            done, _ = await asyncio.wait({read_task}, timeout=0.25)
            if not done:
                if (
                    time.monotonic() - started_at >= _INDEX_MAX_RUNTIME_SECONDS
                    and not cancel_sent
                ):
                    timed_out = True
                    cancel_sent = True
                    send_progress = False
                    unparsed_tail.append(
                        f"index worker exceeded {_INDEX_MAX_RUNTIME_SECONDS}s"
                    )
                    if process.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            process.terminate()
                if cancel_event is not None and cancel_event.is_set() and not cancel_sent:
                    _index_cancel_requested = True
                    cancel_sent = True
                    process.terminate()
                    send_progress = False
                continue
            line = read_task.result()
            read_task = None
            if not line:
                break
            if cancel_event is not None and cancel_event.is_set() and not cancel_sent:
                _index_cancel_requested = True
                cancel_sent = True
                process.terminate()
                send_progress = False
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    unparsed_tail.append(text)
                    del unparsed_tail[:-20]  # bounded: keep only the most recent lines
                event = None
            if event is None:
                pass
            elif not isinstance(event, dict):
                unparsed_tail.append("worker emitted non-object JSON")
                del unparsed_tail[:-20]
            elif not send_progress:
                pass
            elif sync_kb and event.get("event") in {"prepare", "prepared", "file", "result", "cleanup", "done"}:
                path = Path(event["path"]) if event.get("path") else None
                if event.get("event") == "done":
                    stats = event.get("stats")
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "msg": ui.sync_kb_progress_text(event.get("event", ""), path, event.get("stats", {})),
                    })
                except Exception:
                    if cancel_event is not None:
                        cancel_event.set()
                    _index_cancel_requested = True
                    cancel_sent = True
                    send_progress = False
                    if process.returncode is None:
                        process.terminate()
            elif event.get("event") == "file":
                path = Path(event["path"]) if event.get("path") else None
                try:
                    await websocket.send_json({"type": "progress", "msg": ui.index_progress_text(path, event.get("stats", {}))})
                except Exception:
                    if cancel_event is not None:
                        cancel_event.set()
                    _index_cancel_requested = True
                    cancel_sent = True
                    send_progress = False
                    if process.returncode is None:
                        process.terminate()
            elif event.get("event") == "stale":
                path = Path(event["path"]) if event.get("path") else None
                try:
                    await websocket.send_json({"type": "progress", "msg": ui.index_stale_progress_text(path, event.get("stats", {}))})
                except Exception:
                    if cancel_event is not None:
                        cancel_event.set()
                    _index_cancel_requested = True
                    cancel_sent = True
                    send_progress = False
                    if process.returncode is None:
                        process.terminate()
            elif event.get("event") == "done":
                stats = event.get("stats")
            read_task = asyncio.create_task(process.stdout.readline())
        await process.wait()
    finally:
        cancel_requested = _index_cancel_requested or bool(cancel_event and cancel_event.is_set())
        if read_task is not None and not read_task.done():
            read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await read_task
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

    crashed = (timed_out or stats is None or process.returncode not in (None, 0)) and not cancel_requested
    detail = ("\n".join(unparsed_tail))[-_MAX_UNPARSED_TAIL_CHARS:] if crashed else ""
    return stats, crashed, detail


async def _run_index(
    websocket: WebSocket, session: "BrowserSession", *, sync_kb: bool = False,
) -> None:
    global _active_index_owner
    # Reserve the index slot before the first await. Checking only
    # _active_index_process here allowed two requests to pass while the first
    # one was still waiting for create_subprocess_exec; the second then ran
    # after the first completed instead of being rejected.
    if _active_index_owner is not None or _active_index_process is not None:
        await websocket.send_json({"type": "error", "msg": "กำลังสร้าง RAG index อยู่แล้วในอีกหน้าต่างหนึ่ง"})
        return
    cancel_event = threading.Event()
    session.cancel_event = cancel_event
    _active_index_owner = session
    try:
        if _HEAVY_WORK_LOCK.locked():
            await websocket.send_json({"type": "progress", "msg": "กำลังตอบคำถามอยู่ — index รอคิว…"})
        # Not "start": that event is the LLM-query flow's signal to reset the
        # spinner to a hardcoded 'thinking…' label and open a streaming assistant
        # bubble -- neither applies here and would clobber the
        # "กำลังสร้าง RAG index…" label already showing. "busy" only disables
        # the input while the index subprocess runs.
        await websocket.send_json({"type": "busy"})
        if not await _acquire_heavy_work(cancel_event):
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "cancelled"})
            return
        try:
            if sync_kb:
                stats, crashed, detail = await _run_index_subprocess(
                    websocket, sync_kb=True, cancel_event=cancel_event,
                )
            else:
                stats, crashed, detail = await _run_index_subprocess(
                    websocket, cancel_event=cancel_event,
                )
        finally:
            _HEAVY_WORK_LOCK.release()

        if crashed:
            label = "sync Knowledge Base" if sync_kb else "สร้าง RAG index"
            msg = f"{label} ล้มเหลว (โปรเซสหยุดกะทันหัน)" + (f": {detail}" if detail else "")
            await websocket.send_json({"type": "error", "msg": msg})
            return
        if stats is None:
            await websocket.send_json({"type": "cancelled"})
            return
        if sync_kb:
            summary = (
                f"✓ sync_kb เสร็จ: source {stats.get('raw_scanned', 0):,} ไฟล์, "
                f"index {stats.get('indexed', 0):,}, ไม่เปลี่ยน {stats.get('unchanged', 0):,}, "
                f"ลบ {stats.get('removed', 0):,}, documents {stats.get('documents', 0):,}, "
                f"chunks {stats.get('chunks', 0):,}; SQLite {stats.get('inserted', 0):,} new, "
                f"{stats.get('updated', 0):,} updated, errors {stats.get('sync_errors', 0):,} "
                f"({stats.get('seconds', 0):.2f}s)"
            )
        else:
            summary = (f"✓ สร้าง RAG index เสร็จ: {stats['scanned']:,} scanned, "
                       f"{stats['inserted']:,} new, {stats['updated']:,} updated, "
                       f"{stats['skipped']:,} skipped, {stats['error']:,} errors ({stats['seconds']:.2f}s)")
        await websocket.send_json({"type": "response", "content": summary})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        cancel_event.set()
    except Exception:
        cancel_event.set()
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "msg": "การสร้าง index ล้มเหลว กรุณาดู log แล้วลองใหม่"})
    finally:
        if _active_index_owner is session:
            _active_index_owner = None
        if session.cancel_event is cancel_event:
            session.cancel_event = None


def _rag_sync_snapshot() -> dict | None:
    with _RAG_SYNC_STATE_LOCK:
        if _rag_sync_state is None:
            return None
        snapshot = dict(_rag_sync_state)
        snapshot["stats"] = dict(_rag_sync_state.get("stats") or {})
        return snapshot


def _update_rag_sync_state(job_id: str, **updates) -> None:
    with _RAG_SYNC_STATE_LOCK:
        if _rag_sync_state is None or _rag_sync_state.get("job_id") != job_id:
            return
        _rag_sync_state.update(updates)
        _rag_sync_state["updated_at"] = time.time()


async def _run_rag_sync_job(job_id: str) -> None:
    """Run a direct-panel sync while retaining a bounded live status snapshot."""
    try:
        _update_rag_sync_state(
            job_id,
            status="queued",
            event="queued",
            message="กำลังรอคิว sync Knowledge Base…",
        )

        def on_progress(event: str, path: Path | None, stats: dict) -> None:
            _update_rag_sync_state(
                job_id,
                status="running",
                event=event,
                path=str(path) if path is not None else None,
                stats=dict(stats),
                message=ui.sync_kb_progress_text(event, path, stats),
            )

        async with _HEAVY_WORK_LOCK:
            _update_rag_sync_state(
                job_id,
                status="running",
                event="start",
                message="กำลังเริ่ม sync Knowledge Base…",
            )
            stats = await asyncio.to_thread(
                rag_mcp_service.sync_kb,
                on_progress=on_progress,
            )
        summary = rag_mcp_service.format_sync_result(stats)
        errors = int(stats.get("sync_errors", 0) or 0)
        _update_rag_sync_state(
            job_id,
            status="error" if errors else "done",
            event="done",
            path=None,
            stats=dict(stats),
            summary=summary,
            error=(f"sync Knowledge Base มีข้อผิดพลาด {errors} รายการ" if errors else None),
            message=(
                f"sync เสร็จแต่มีข้อผิดพลาด {errors} รายการ · ประมาณ 100%"
                if errors else "sync Knowledge Base เสร็จ · ประมาณ 100%"
            ),
        )
    except asyncio.CancelledError:
        _update_rag_sync_state(
            job_id,
            status="error",
            event="cancelled",
            error="sync Knowledge Base ถูกยกเลิก",
            message="sync Knowledge Base ถูกยกเลิก",
        )
        raise
    except Exception:
        _update_rag_sync_state(
            job_id,
            status="error",
            event="error",
            error="sync Knowledge Base ไม่สำเร็จ",
            message="sync Knowledge Base ไม่สำเร็จ",
        )


def _pdf_text_snapshot() -> dict | None:
    with _PDF_TEXT_STATE_LOCK:
        if _pdf_text_state is None:
            return None
        snapshot = dict(_pdf_text_state)
        snapshot["stats"] = dict(_pdf_text_state.get("stats") or {})
        snapshot["warnings"] = list(_pdf_text_state.get("warnings") or [])
        return snapshot


def _update_pdf_text_state(job_id: str, **updates) -> None:
    with _PDF_TEXT_STATE_LOCK:
        if _pdf_text_state is None or _pdf_text_state.get("job_id") != job_id:
            return
        _pdf_text_state.update(updates)
        _pdf_text_state["updated_at"] = time.time()


def _pdf_text_progress_message(event: str, page: int, total: int, stats: dict) -> str:
    if event == "extract":
        return (
            f"กำลังอ่าน PDF หน้า {page:,}/{total:,} · native {int(stats.get('native', 0)):,} · "
            f"OCR {int(stats.get('ocr', 0)):,} · table {int(stats.get('tables', 0)):,}"
        )
    if event == "rewrite":
        return (
            "กำลังเกลาภาษาไทยจาก OCR แบบ no-think · "
            f"batch {int(stats.get('batch', 0)):,}/{int(stats.get('batches', 0)):,} · "
            f"{int(stats.get('pages', 0)):,} หน้าใน request นี้"
        )
    if event == "write":
        return "กำลังเขียนไฟล์ข้อความตามลำดับหน้าจริงของ PDF…"
    return "กำลังแปลง PDF เป็นข้อความ…"


async def _run_pdf_text_job(
    job_id: str,
    source_path: Path,
    source_name: str,
    rewrite_thai: bool,
) -> None:
    """Run one direct PDF conversion without adding anything to chat history."""
    try:
        _update_pdf_text_state(
            job_id,
            status="queued",
            event="queued",
            message="กำลังรอคิว PDF to Text…",
        )

        def on_progress(event: str, page: int, total: int, stats: dict) -> None:
            _update_pdf_text_state(
                job_id,
                status="running",
                event=event,
                page=int(page),
                total_pages=int(total),
                stats=dict(stats),
                message=_pdf_text_progress_message(event, page, total, stats),
            )

        async with _HEAVY_WORK_LOCK:
            _update_pdf_text_state(
                job_id,
                status="running",
                event="start",
                message="กำลังเริ่ม PDF to Text…",
            )
            result = await asyncio.to_thread(
                pdf_to_text.convert_pdf,
                source_path,
                source_name=source_name,
                rewrite_thai=rewrite_thai,
                progress=on_progress,
            )

        warnings = list(result.get("warnings") or [])
        stats = dict(result.get("stats") or {})
        _update_pdf_text_state(
            job_id,
            status="done",
            event="done",
            page=int(result.get("total_pages") or 0),
            total_pages=int(result.get("total_pages") or 0),
            output_path=str(result.get("output_path") or ""),
            stats=stats,
            warnings=warnings,
            error=None,
            message=(
                "PDF to Text เสร็จ · ใช้ OCR/LLM เฉพาะหน้าที่จำเป็น"
                if not warnings else
                "PDF to Text เสร็จ · บางหน้าใช้ OCR เดิมแทน LLM cleanup"
            ),
        )
    except asyncio.CancelledError:
        _update_pdf_text_state(
            job_id,
            status="error",
            event="cancelled",
            error="PDF to Text ถูกยกเลิก",
            message="PDF to Text ถูกยกเลิก",
        )
        raise
    except Exception as exc:
        _update_pdf_text_state(
            job_id,
            status="error",
            event="error",
            error=f"PDF to Text ไม่สำเร็จ: {type(exc).__name__}",
            message="PDF to Text ไม่สำเร็จ",
        )
    finally:
        with contextlib.suppress(OSError):
            source_path.unlink()


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
            # A normal UI disables input while a turn is running.  A broken
            # or hostile client can still flood valid 12KB frames while the
            # consumer is busy, so cancel that work and close this endpoint
            # deterministically instead of retaining an unbounded backlog.
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
    Keeping a bounded copy in the per-machine private state directory gives the
    existing read_file and read_image tools a normal, safe path without
    broadening their filesystem permissions or serving the uploaded bytes back
    to the browser. Stream the body so a missing Content-Length cannot turn a
    local client mistake into an unbounded allocation.
    """
    _require_http_same_origin(request)
    length = request.headers.get("content-length")
    if length:
        try:
            declared_length = int(length)
        except ValueError:
            return {"ok": False, "error": "ขนาดไฟล์ไม่ถูกต้อง"}
        if declared_length < 0 or declared_length > _MAX_ATTACHMENT_BYTES:
            return {"ok": False, "error": f"ไฟล์ใหญ่เกิน {_MAX_ATTACHMENT_BYTES // 1024 // 1024}MB"}

    original_name = unquote(request.headers.get("x-file-name", ""))
    name = Path(original_name).name.strip()
    if not name or name in {".", ".."} or len(name) > 512 or "\x00" in name:
        return {"ok": False, "error": "ชื่อไฟล์ไม่ถูกต้อง"}

    stored_path: Path | None = None
    try:
        _ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
        _ATTACHMENTS_DIR.chmod(0o700)
        _cleanup_attachments()
        suffix = Path(name).suffix[:16]
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=_MANAGED_ATTACHMENT_PREFIX, suffix=suffix,
            dir=_ATTACHMENTS_DIR, delete=False,
        ) as handle:
            stored_path = Path(handle.name)
            total = 0
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_ATTACHMENT_BYTES:
                    raise _AttachmentTooLarge
                handle.write(chunk)
            if total == 0:
                raise ValueError("empty attachment")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o600)
        _cleanup_attachments(keep=stored_path)
    except _AttachmentTooLarge:
        if stored_path is not None:
            with contextlib.suppress(OSError):
                stored_path.unlink()
        return {"ok": False, "error": f"ไฟล์ใหญ่เกิน {_MAX_ATTACHMENT_BYTES // 1024 // 1024}MB"}
    except Exception:
        if stored_path is not None:
            with contextlib.suppress(OSError):
                stored_path.unlink()
        return {"ok": False, "error": "บันทึกไฟล์แนบไม่ได้"}

    return {
        "ok": True, "path": str(stored_path), "name": name,
        "isImage": stored_path.suffix.lower() in _IMAGE_ATTACHMENT_EXTENSIONS,
    }


@app.get("/ui-token")
async def ui_token():
    """Compatibility endpoint for the copied UI; loopback needs no token."""
    return {"token": ""}


def _cleanup_pdf_text_uploads(*, keep: Path | None = None) -> None:
    """Delete only stale private PDF-to-text source snapshots."""
    try:
        entries = [
            item for item in _PDF_TEXT_UPLOAD_DIR.iterdir()
            if item.name.startswith(_MANAGED_PDF_TEXT_PREFIX)
            and item.is_file()
            and not item.is_symlink()
        ]
    except OSError:
        return
    cutoff = time.time() - config.PDF_TO_TEXT_CACHE_MAX_AGE_SECONDS
    for item in entries:
        if item == keep:
            continue
        try:
            if item.stat().st_mtime <= cutoff:
                item.unlink()
        except OSError:
            pass


@app.post("/pdf-to-text/start")
async def pdf_to_text_start_http(request: Request):
    """Stream one selected PDF into private state and start direct conversion."""
    global _pdf_text_state, _pdf_text_task, _pdf_text_counter
    _require_http_same_origin(request)

    original_name = unquote(request.headers.get("x-file-name", ""))
    name = Path(original_name).name.strip()
    if (
        not name
        or name in {".", ".."}
        or len(name) > 512
        or "\x00" in name
        or Path(name).suffix.casefold() != ".pdf"
    ):
        return {"ok": False, "error": "กรุณาเลือกไฟล์ PDF ที่ชื่อถูกต้อง"}

    length = request.headers.get("content-length")
    if length:
        try:
            declared_length = int(length)
        except ValueError:
            return {"ok": False, "error": "ขนาด PDF ไม่ถูกต้อง"}
        if declared_length < 0 or declared_length > config.PDF_TO_TEXT_MAX_BYTES:
            return {
                "ok": False,
                "error": f"PDF ใหญ่เกิน {config.PDF_TO_TEXT_MAX_BYTES // 1024 // 1024}MB",
            }

    rewrite_thai = request.headers.get("x-rewrite-thai", "1").strip().casefold() in {
        "1", "true", "yes", "on",
    }
    with _PDF_TEXT_STATE_LOCK:
        if _pdf_text_state and _pdf_text_state.get("status") in {
            "uploading", "queued", "running",
        }:
            return {"ok": True, "already_running": True, "job": _pdf_text_snapshot()}
        _pdf_text_counter += 1
        job_id = f"pdf-{int(time.time() * 1000)}-{_pdf_text_counter}"
        _pdf_text_state = {
            "job_id": job_id,
            "status": "uploading",
            "event": "uploading",
            "source_name": name,
            "rewrite_thai": rewrite_thai,
            "page": 0,
            "total_pages": 0,
            "stats": {},
            "warnings": [],
            "output_path": None,
            "error": None,
            "message": "กำลังรับ PDF…",
            "updated_at": time.time(),
        }

    stored_path: Path | None = None
    try:
        _PDF_TEXT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        _PDF_TEXT_UPLOAD_DIR.chmod(0o700)
        _cleanup_pdf_text_uploads()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=_MANAGED_PDF_TEXT_PREFIX,
            suffix=".pdf",
            dir=_PDF_TEXT_UPLOAD_DIR,
            delete=False,
        ) as handle:
            stored_path = Path(handle.name)
            total = 0
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > config.PDF_TO_TEXT_MAX_BYTES:
                    raise _AttachmentTooLarge
                handle.write(chunk)
            if total == 0:
                raise ValueError("empty pdf")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o600)
        with stored_path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("not a pdf")
        _cleanup_pdf_text_uploads(keep=stored_path)
    except _AttachmentTooLarge:
        if stored_path is not None:
            with contextlib.suppress(OSError):
                stored_path.unlink()
        _update_pdf_text_state(
            job_id,
            status="error",
            event="error",
            error=f"PDF ใหญ่เกิน {config.PDF_TO_TEXT_MAX_BYTES // 1024 // 1024}MB",
            message="รับ PDF ไม่สำเร็จ",
        )
        return {"ok": False, "error": f"PDF ใหญ่เกิน {config.PDF_TO_TEXT_MAX_BYTES // 1024 // 1024}MB"}
    except Exception:
        if stored_path is not None:
            with contextlib.suppress(OSError):
                stored_path.unlink()
        _update_pdf_text_state(
            job_id,
            status="error",
            event="error",
            error="ไฟล์ที่เลือกไม่ใช่ PDF ที่อ่านได้",
            message="รับ PDF ไม่สำเร็จ",
        )
        return {"ok": False, "error": "ไฟล์ที่เลือกไม่ใช่ PDF ที่อ่านได้"}

    _update_pdf_text_state(
        job_id,
        status="queued",
        event="queued",
        message="รับ PDF แล้ว · กำลังรอคิวแปลง…",
    )
    _pdf_text_task = asyncio.create_task(
        _run_pdf_text_job(job_id, stored_path, name, rewrite_thai)
    )
    return {"ok": True, "already_running": False, "job": _pdf_text_snapshot()}


@app.post("/pdf-to-text/status")
async def pdf_to_text_status_http(request: Request):
    """Return the latest bounded progress snapshot for the PDF panel."""
    _require_http_same_origin(request)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    job_id = payload.get("job_id", "").strip() if isinstance(payload, dict) else ""
    snapshot = _pdf_text_snapshot()
    if not snapshot or (job_id and snapshot.get("job_id") != job_id):
        return {"ok": False, "error": "ไม่พบงาน PDF to Text นี้"}
    return {"ok": True, "job": snapshot}


@app.post("/rag/search")
async def rag_search_http(request: Request):
    """Deterministic KB search for the RAG side panel; never invokes the LLM."""
    _require_http_same_origin(request)
    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "error": "JSON ไม่ถูกต้อง"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "request ต้องเป็น JSON object"}
    query = payload.get("query")
    if not isinstance(query, str):
        return {"ok": False, "error": "query ต้องเป็นข้อความ"}
    try:
        result = await asyncio.to_thread(rag_mcp_service.search, query)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception:
        return {"ok": False, "error": "ค้นหา Knowledge Base ไม่สำเร็จ"}
    return {"ok": True, "result": result}


@app.post("/rag/sync")
async def rag_sync_http(request: Request):
    """Sync knowledge-root documents through the shared RAG service, no LLM/ReAct."""
    _require_http_same_origin(request)
    # Keep KB/OCR mutation serialized with model/index heavy work exactly like
    # the existing /sync_kb WebSocket command.
    async with _HEAVY_WORK_LOCK:
        try:
            stats = await asyncio.to_thread(rag_mcp_service.sync_kb)
        except Exception:
            return {"ok": False, "error": "sync Knowledge Base ไม่สำเร็จ"}
    summary = rag_mcp_service.format_sync_result(stats)
    sync_errors = int(stats.get("sync_errors", 0) or 0)
    if sync_errors:
        return {
            "ok": False,
            "error": f"sync Knowledge Base มีข้อผิดพลาด {sync_errors} รายการ",
            "stats": stats,
            "summary": summary,
        }
    return {"ok": True, "stats": stats, "summary": summary}


@app.post("/rag/sync/start")
async def rag_sync_start_http(request: Request):
    """Start a direct-panel sync and return a job id for live polling."""
    global _rag_sync_state, _rag_sync_task, _rag_sync_counter
    _require_http_same_origin(request)
    with _RAG_SYNC_STATE_LOCK:
        if _rag_sync_state and _rag_sync_state.get("status") in {"queued", "running"}:
            return {"ok": True, "already_running": True, "job": _rag_sync_snapshot()}
        _rag_sync_counter += 1
        job_id = f"sync-{int(time.time() * 1000)}-{_rag_sync_counter}"
        _rag_sync_state = {
            "job_id": job_id,
            "status": "queued",
            "event": "queued",
            "path": None,
            "stats": {},
            "summary": None,
            "error": None,
            "message": "กำลังเตรียม sync Knowledge Base…",
            "updated_at": time.time(),
        }
    _rag_sync_task = asyncio.create_task(_run_rag_sync_job(job_id))
    return {"ok": True, "already_running": False, "job": _rag_sync_snapshot()}


@app.post("/rag/sync/status")
async def rag_sync_status_http(request: Request):
    """Return the latest bounded progress snapshot for a direct-panel sync."""
    _require_http_same_origin(request)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    job_id = payload.get("job_id", "").strip() if isinstance(payload, dict) else ""
    snapshot = _rag_sync_snapshot()
    if not snapshot or (job_id and snapshot.get("job_id") != job_id):
        return {"ok": False, "error": "ไม่พบงาน sync นี้"}
    return {"ok": True, "job": snapshot}


@app.get("/server/status")
async def server_status():
    state = await asyncio.to_thread(_mlx_server_state)
    return {
        "server_up": state == "ready",
        "server_state": state,
        "model": config.MODEL_ID,
        "server_port": config.MLX_SERVER_PORT,
    }


@app.post("/server/start")
async def server_start(request: Request):
    """Start the configured local mlx_vlm.server if its port is safe.

    Returns immediately after spawning (model load can take a while) --
    the browser polls /server/status to learn when it's actually up.
    """
    _require_http_same_origin(request)
    if _MLX_SERVER_LOCK.locked():
        return {"ok": False, "error": "กำลังเปิด/ปิด server อยู่ — รอสักครู่"}
    def _do() -> None:
        with _MLX_SERVER_LOCK:
            _start_mlx_server()
    try:
        await asyncio.to_thread(_do)
    except (RuntimeError, OSError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "server_state": "starting", "model": config.MODEL_ID}


@app.post("/server/stop")
async def server_stop(request: Request):
    """Stop Lite's model server without touching an unrelated port owner."""
    _require_http_same_origin(request)
    if _MLX_SERVER_LOCK.locked():
        return {"ok": False, "error": "กำลังเปิด/ปิด server อยู่ — รอสักครู่"}
    def _do() -> None:
        with _MLX_SERVER_LOCK:
            _kill_mlx_server(config.MLX_SERVER_PORT)
    try:
        await asyncio.to_thread(_do)
    except (RuntimeError, OSError) as exc:
        return {"ok": False, "error": str(exc)}
    state = await asyncio.to_thread(_mlx_server_state)
    return {"ok": True, "server_up": state == "ready", "server_state": state}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Browsers allow a web page from another origin to initiate a WebSocket to
    # localhost.  Reject it before accept so a hostile site cannot drive Lite's
    # filesystem-capable agent through the user's browser.
    origin = websocket.headers.get("origin")
    if origin not in _ALLOWED_ORIGINS:
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
        await websocket.send_json(_status(session))
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
                requested_effort = request.get("effort")
                if isinstance(requested_effort, str):
                    session.effort = config.normalize_thinking_effort(requested_effort)
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
                elif command == "/sync_kb":
                    await _run_index(websocket, session, sync_kb=True)
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
