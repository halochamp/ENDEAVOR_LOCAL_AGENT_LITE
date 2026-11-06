"""save_output.py — spill full tool output to SSD before truncating for context.

Keeping the model's context small keeps its KV cache small, which is the
main lever on RAM for an in-process model (see PLAN.md §3). Nothing here
is ever loaded back into RAM automatically — the saved path is just a
pointer the user (or a future read_file call) can follow.
"""
from __future__ import annotations
import hashlib
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import config

_OUT_DIR = config.LOG_DIR / "tool_outputs"
_MAX_OUTPUT_TOTAL_BYTES = 50 * 1024 * 1024
_MAX_OUTPUT_AGE_SECONDS = 7 * 24 * 60 * 60
_MAX_OUTPUT_FILES = 256
_OUTPUT_LOCK = threading.Lock()
_SAFE_TOOL_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _cleanup_outputs(*, keep: Path | None = None) -> None:
    try:
        entries = [
            path for path in _OUT_DIR.iterdir()
            if path.is_file() and not path.is_symlink() and path.name != (keep.name if keep else None)
        ]
    except OSError:
        return

    now = time.time()
    retained: list[tuple[float, int, Path]] = []
    for path in entries:
        try:
            stat = path.stat()
        except OSError:
            continue
        if now - stat.st_mtime > _MAX_OUTPUT_AGE_SECONDS:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        retained.append((stat.st_mtime, stat.st_size, path))

    # Keep both a byte ceiling and a file-count ceiling. Oldest files are the
    # least useful after a bounded model turn has already consumed its preview.
    total = sum(size for _, size, _ in retained)
    for _, size, path in sorted(retained):
        if total <= _MAX_OUTPUT_TOTAL_BYTES and len(retained) <= _MAX_OUTPUT_FILES:
            break
        try:
            path.unlink()
            total -= size
            retained.remove((_, size, path))
        except OSError:
            pass


def save_full_output(text: str, tool_name: str) -> str | None:
    """Atomically save bounded diagnostic output with best-effort retention.

    A full-output pointer is useful, but a full disk must not turn an otherwise
    successful read-only command into an agent failure. Callers can fall back
    to their already-bounded preview when this returns ``None``.
    """
    try:
        with _OUTPUT_LOCK:
            _OUT_DIR.mkdir(parents=True, exist_ok=True)
            _OUT_DIR.chmod(0o700)
            _cleanup_outputs()
            h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:10]
            safe_name = _SAFE_TOOL_NAME.sub("_", str(tool_name)).strip("._") or "tool"
            path = _OUT_DIR / f"{safe_name}_{time.time_ns()}_{h}.txt"
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", errors="replace", dir=_OUT_DIR,
                prefix=f".{safe_name}-", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
            path.chmod(0o600)
            _cleanup_outputs(keep=path)
            return str(path)
    except OSError:
        try:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return None
