"""tools/write_file.py — trimmed from ENDEAVOR_LOCAL_AGENT_MAX_VLM/tools/write_file.py.

Dropped: langchain @tool decorator, chunked progress-reporting write (256KB
chunks — not needed at the sizes this agent writes), plan_write()'s
copy-on-write "sibling .edited file" behavior for pre-existing outside files.
Kept: atomic write via temp-file + os.replace, explicit overwrite gate.

Writes are restricted to config.BASH_ALLOWED_ROOT, which is the project's
workspect/ directory — the agent cannot alter its own source tree or any
other part of the user's home directory.
"""
from __future__ import annotations
import os
from pathlib import Path

import config

SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        # See tools/bash.py's 2026-07-23 note — reverted alongside it.
        "description": (
            "Write text content to a file on disk, given its full path. Fails if the "
            "file already exists unless overwrite=true is passed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~-relative path to write"},
                "content": {"type": "string", "description": "Full file content to write"},
                "overwrite": {"type": "string", "description": "'true' to replace an existing file, else omit"},
            },
            "required": ["path", "content"],
        },
    },
}


def run(args: dict[str, str]) -> str:
    raw_path = (args.get("path") or "").strip()
    content = args.get("content", "")
    overwrite = (args.get("overwrite") or "").strip().lower() == "true"

    if not raw_path:
        return "[error] path is required"

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return f"[error] ต้องเป็น absolute path (หรือขึ้นต้นด้วย ~) ได้ path: '{raw_path}'"

    allowed = config.BASH_ALLOWED_ROOT.resolve()
    try:
        path.resolve().relative_to(allowed)
    except ValueError:
        return f"[error] เขียนได้เฉพาะใต้ {allowed} เท่านั้น (ได้ path: {path})"

    if path.exists() and not overwrite:
        return f"[error] file exists: {path} ({path.stat().st_size} bytes) — ใส่ overwrite='true' ถ้าต้องการเขียนทับ"

    tmp = None
    try:
        prev_size = path.stat().st_size if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = content.encode("utf-8")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        n_lines = content.count("\n") + 1 if content else 0
        if prev_size is not None:
            return f"overwrote {path}: {len(content)} chars ({n_lines} lines, previous file was {prev_size} bytes)"
        return f"written {len(content)} chars ({n_lines} lines) to {path}"
    except Exception as e:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        return f"[error] write_file failed: {e}"


# LangChain entry point — see tools/bash.py's 2026-07-23 note.
from langchain_core.tools import tool as _lc_tool


@_lc_tool
def write_file(path: str, content: str, overwrite: str = "") -> str:
    """Write text content to a file on disk, given its full path. Fails if the
    file already exists unless overwrite=true is passed."""
    from helpers import turn_guard
    args = {"path": path, "content": content, "overwrite": overwrite}
    blocked = turn_guard.check("write_file", args)
    if blocked:
        return blocked
    result = run(args)
    turn_guard.record("write_file", args, result)
    return result
