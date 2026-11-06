"""Trimmed generic MCP client forked from Agent MAX VLM for Agent Lite.

Lite keeps only list/call against developer-provisioned servers. Dynamic
add/remove registration is intentionally omitted so the 2B model cannot widen
its own capability surface. Local stdio servers run through sandbox-exec with
write access only to explicit developer-provisioned paths.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Coroutine, TypeVar

from langchain_core.tools import tool

import config
from helpers import search_scope, turn_guard
from tools.bash import _SENSITIVE_SANDBOX_DIRS

_T = TypeVar("_T")
_SANDBOX_EXEC = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"

MCP_LIST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mcp_list_tools",
        "description": "List tools on Agent Lite's configured MCP server, or inspect one exact tool schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "tool_name": {"type": "string"},
            },
            "required": ["server"],
        },
    },
}

MCP_CALL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mcp_call_tool",
        "description": "Call a tool on Agent Lite's configured MCP server with JSON-object arguments.",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "tool_name": {"type": "string"},
                "arguments_json": {"type": "string"},
            },
            "required": ["server", "tool_name"],
        },
    },
}


def _inside(path: str, root: str) -> bool:
    try:
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root)
        return os.path.commonpath((real_path, real_root)) == real_root
    except ValueError:
        return False


def _known_servers() -> str:
    return ", ".join(sorted(config.MCP_SERVERS)) or "(none configured)"


def _cap(text: str) -> str:
    if len(text) <= config.MCP_MAX_CHARS:
        return text
    marker = f"\n...[truncated at {config.MCP_MAX_CHARS} chars]"
    return text[: max(0, config.MCP_MAX_CHARS - len(marker))] + marker


def _compact_tool_list(tools) -> str:
    rows = [
        (str(item.name), " ".join((item.description or "(no description)").split()))
        for item in tools
    ]
    if not rows:
        return "(no tools exposed)"
    text = "\n".join(f"{name}: {description}" for name, description in rows)
    return _cap(text)


def _normalise_server_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("MCP server configuration must be an object")
    transport = str(cfg.get("transport") or "").strip()
    if transport != "stdio":
        raise ValueError("Agent Lite currently permits only local stdio MCP servers")

    command = os.path.realpath(os.path.expanduser(str(cfg.get("command") or "")))
    if not os.path.isabs(command) or not os.path.isfile(command) or not os.access(command, os.X_OK):
        raise ValueError("stdio MCP command must be an existing absolute executable")

    args = cfg.get("args") or []
    if not isinstance(args, list) or not all(isinstance(item, str) and "\x00" not in item for item in args):
        raise ValueError("stdio MCP args must be a list of strings")

    workspace = os.path.realpath(os.path.expanduser(str(config.MCP_WORKSPACE)))
    cwd = os.path.realpath(os.path.expanduser(str(cfg.get("cwd") or workspace)))
    if not _inside(cwd, workspace):
        raise ValueError("stdio MCP cwd must stay inside Lite MCP_WORKSPACE")
    # DATA_DIR/MCP_WORKSPACE are private runtime state, not user content.  A
    # prior test, cleanup, or interrupted run may remove them after config.py
    # imported; recreate only this already-validated in-workspace directory so
    # a healthy provisioned MCP server can start again.
    try:
        Path(cwd).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError("stdio MCP cwd is not available") from exc
    if not os.path.isdir(cwd):
        raise ValueError("stdio MCP cwd is not available")
    return {"transport": "stdio", "command": command, "args": args, "cwd": cwd}


def _sandbox_write_paths(server: str, cfg: dict) -> tuple[str, ...]:
    provisioned = config.MCP_SERVERS.get(server)
    if provisioned is None or cfg is not provisioned:
        return ()
    raw = provisioned.get("sandbox_write_paths") or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("sandbox_write_paths must be a list")
    paths: list[str] = []
    for value in raw:
        path = os.path.realpath(os.path.expanduser(str(value)))
        if not os.path.isabs(path):
            raise ValueError("sandbox_write_paths must contain existing absolute directories")
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError("sandbox_write_paths must contain usable absolute directories") from exc
        if not os.path.isdir(path):
            raise ValueError("sandbox_write_paths must contain usable absolute directories")
        paths.append(path)
    return tuple(paths)


def _build_mcp_sandbox_profile(extra_write_paths: tuple[str, ...]) -> str:
    home = os.path.realpath(os.path.expanduser("~"))
    denied_reads = " ".join(f'(subpath "{path}")' for path in sorted(_SENSITIVE_SANDBOX_DIRS(home)))
    allowed = [os.path.realpath(str(config.MCP_WORKSPACE)), *extra_write_paths]
    allowed_writes = " ".join(f'(subpath "{path}")' for path in dict.fromkeys(allowed))
    return f'''(version 1)
(allow default)
(deny file-write*)
(deny file-read* {denied_reads})
(deny network*)
(allow file-write* {allowed_writes} (subpath "/private/tmp"))
'''


def _server_environment() -> dict[str, str]:
    """Pass the parent's bounded RAG context into the stdio child.

    The MCP server imports its own ``config`` module, so a parent-side
    ``config.DATA_DIR``/``FILE_INDEX_DB``/``INDEX_ROOTS`` override would
    otherwise vanish at the process boundary. Only the already-resolved,
    locally-allowed paths are forwarded; the child still applies config.py's
    boundary checks before using them. Search roots are forwarded as explicit
    scope so result-path validation agrees with the index selected by the
    parent. Do not pass the complete parent environment: a stdio child has no
    reason to inherit unrelated API credentials or cloud configuration.
    """
    passthrough = {
        "HOME", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "PYTHONNOUSERSITE",
        "TMPDIR", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    }
    environment = {
        key: value for key, value in os.environ.items()
        if key in passthrough or (key.startswith("LITE_") and key != "LITE_MLX_API_KEY")
    }
    # MCP-RagDoc is agent-agnostic. Translate Lite's trusted, already-resolved
    # runtime paths into the generic server contract instead of making the MCP
    # implementation import or understand Agent Lite configuration.

    data_dir = Path(config.DATA_DIR).expanduser().resolve(strict=False)
    if config._state_path_allowed(data_dir):
        environment["LITE_DATA_DIR"] = str(data_dir)
        environment["RAGDOC_DATA_DIR"] = str(data_dir)
    else:
        environment.pop("LITE_DATA_DIR", None)
        environment.pop("RAGDOC_DATA_DIR", None)

    index_db = Path(config.FILE_INDEX_DB).expanduser().resolve(strict=False)
    if not config._state_path_allowed(index_db):
        # Do not let a monkeypatch or a hostile runtime setting widen the
        # child beyond Lite's private/local state. The child falls back to its
        # normal DATA_DIR index when the path is outside the boundary.
        environment.pop("LITE_FILE_INDEX_DB", None)
        environment.pop("RAGDOC_DB", None)
    else:
        environment["LITE_FILE_INDEX_DB"] = str(index_db)
        environment["RAGDOC_DB"] = str(index_db)

    roots = []
    for root in getattr(config, "INDEX_ROOTS", ()):
        try:
            roots.append(str(Path(root).expanduser().resolve(strict=False)))
        except (OSError, RuntimeError, TypeError):
            continue
    environment["LITE_INDEX_ROOTS"] = ",".join(dict.fromkeys(roots))
    try:
        environment["RAGDOC_KNOWLEDGE_DIR"] = str(
            Path(config.KNOWLEDGE_DIR).expanduser().resolve(strict=False)
        )
    except (OSError, RuntimeError, TypeError):
        environment.pop("RAGDOC_KNOWLEDGE_DIR", None)
    environment["RAGDOC_SEMANTIC"] = (
        "1" if bool(getattr(config, "SEMANTIC_ENABLED", True)) else "0"
    )
    model_dir = os.getenv("LITE_SEMANTIC_MODEL_DIR", "").strip()
    if model_dir:
        environment["RAGDOC_SEMANTIC_MODEL_DIR"] = model_dir
    return environment


def _run_async(factory: Callable[[], Coroutine[object, object, _T]]) -> _T:
    async def _bounded() -> _T:
        return await asyncio.wait_for(factory(), timeout=config.MCP_TIMEOUT)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_bounded())

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lite-mcp") as pool:
        future = pool.submit(lambda: asyncio.run(_bounded()))
        try:
            return future.result(timeout=config.MCP_TIMEOUT + 5)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"MCP operation exceeded {config.MCP_TIMEOUT}s") from exc


@asynccontextmanager
async def _open_session(cfg: dict, *, sandbox_write_paths: tuple[str, ...] = ()):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    cfg = _normalise_server_config(cfg)
    profile = _build_mcp_sandbox_profile(sandbox_write_paths)
    # Use sandbox-exec's inline profile form instead of a temporary .sb file.
    # This avoids cleanup races and leaves no runtime artifact behind.
    params = StdioServerParameters(
        command=_SANDBOX_EXEC,
        args=["-p", profile, cfg["command"], *cfg["args"]],
        env=_server_environment(),
        cwd=cfg["cwd"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _list_tools_async(cfg: dict, *, tool_name: str = "", sandbox_write_paths: tuple[str, ...] = ()) -> str:
    async with _open_session(cfg, sandbox_write_paths=sandbox_write_paths) as session:
        result = await session.list_tools()
    if tool_name:
        for item in result.tools:
            if item.name == tool_name:
                return json.dumps({
                    "name": item.name,
                    "description": item.description or "",
                    "inputSchema": item.inputSchema,
                }, ensure_ascii=False)
        return f"[error] MCP tool '{tool_name}' not found"
    return _compact_tool_list(result.tools)


async def _call_tool_async(cfg: dict, tool_name: str, arguments: dict, *, sandbox_write_paths: tuple[str, ...] = ()) -> str:
    async with _open_session(cfg, sandbox_write_paths=sandbox_write_paths) as session:
        result = await session.call_tool(tool_name, arguments)
    parts = [content.text for content in result.content if getattr(content, "type", None) == "text"]
    body = "\n".join(parts) if parts else "(tool returned no text content)"
    return f"[error] {tool_name} returned an error: {body}" if result.isError else body


def run_list(args: dict[str, str]) -> str:
    server = (args.get("server") or "").strip()
    tool_name = (args.get("tool_name") or "").strip()
    cfg = config.MCP_SERVERS.get(server)
    if cfg is None:
        return f"[error] no MCP server named '{server}' configured — known servers: {_known_servers()}"
    try:
        return _cap(_run_async(lambda: _list_tools_async(
            cfg, tool_name=tool_name, sandbox_write_paths=_sandbox_write_paths(server, cfg)
        )))
    except Exception as exc:
        return f"[error] mcp_list_tools failed for '{server}' ({type(exc).__name__})"


def run_call(args: dict[str, str]) -> str:
    server = (args.get("server") or "").strip()
    tool_name = (args.get("tool_name") or "").strip()
    arguments_json = args.get("arguments_json") or "{}"
    cfg = config.MCP_SERVERS.get(server)
    if cfg is None:
        return f"[error] no MCP server named '{server}' configured — known servers: {_known_servers()}"
    if not tool_name:
        return "[error] tool_name is required"
    if len(arguments_json) > 12_000:
        return "[error] arguments_json is too large"
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return f"[error] invalid arguments_json: {exc}"
    if not isinstance(arguments, dict):
        return "[error] arguments_json must be a JSON object"
    try:
        return _cap(_run_async(lambda: _call_tool_async(
            cfg, tool_name, arguments, sandbox_write_paths=_sandbox_write_paths(server, cfg)
        )))
    except Exception as exc:
        return f"[error] mcp_call_tool failed for '{server}.{tool_name}' ({type(exc).__name__})"


@tool
def mcp_list_tools(server: str, tool_name: str = "") -> str:
    """List tools on a configured MCP server or inspect one exact schema.

    Agent Lite has one built-in server: ``lite-rag``. You normally do not need
    discovery for its known tools (rag_search, rag_manage, rag_status); use this
    tool when checking capability/schema or diagnosing an MCP error.
    """
    args = {"server": server, "tool_name": tool_name}
    blocked = turn_guard.check("mcp_list_tools", args)
    if blocked:
        return blocked
    result = run_list(args)
    turn_guard.record("mcp_list_tools", args, result)
    return result


@tool
def mcp_call_tool(
    server: str = "lite-rag",
    tool_name: str = "rag_search",
    arguments_json: str = "{}",
) -> str:
    """Call a tool on Agent Lite's configured MCP server.

    For local Knowledge Base use ``server='lite-rag'``. Known tools:
    ``rag_search`` with ``arguments_json='{"query":"keyword"}'``;
    ``rag_manage`` with ``action=sync|cancel`` starts/cancels background refresh;
    ``rag_status`` with ``view=job|health`` polls progress or readiness.
    """
    args = {"server": server, "tool_name": tool_name, "arguments_json": arguments_json}
    if (
        server == "lite-rag"
        and tool_name == "rag_search"
        and search_scope.is_outside_knowledge_base_query(turn_guard.user_query())
    ):
        return (
            "[blocked] คำขอนี้ระบุให้ค้นนอก Knowledge Base/RAG "
            "จึงต้องใช้ bash กับ Desktop/Documents แทน rag_search"
        )
    if (
        server == "lite-rag"
        and tool_name == "rag_search"
        and search_scope.is_knowledge_base_explanation_query(turn_guard.user_query())
    ):
        return (
            "[blocked] คำถามนี้ถามความหมายหรือการทำงานของ RAG "
            "ไม่ใช่การค้นใน Knowledge Base; ตอบจากความรู้ทั่วไปโดยไม่เรียก rag_search"
        )
    blocked = turn_guard.check("mcp_call_tool", args)
    if blocked:
        return blocked
    result = run_call(args)
    turn_guard.record("mcp_call_tool", args, result)
    return result
