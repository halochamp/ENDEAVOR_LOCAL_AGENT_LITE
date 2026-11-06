"""agent.py — turn orchestration around the LangGraph react agent (react.py).

Architecture switch (2026-07-23, user request — see config.py's module
docstring): the ReAct loop itself is now LangGraph's own (react.py's
create_react_agent), not a hand-rolled while-loop. This module's job
shrank to what LangGraph doesn't provide out of the box:

  1. Cross-turn history compaction — same design as before the switch:
     `messages` (the persisted list the caller owns, e.g. ui_cli.py's
     session variable) holds one compact (user, assistant) pair per turn —
     a tool-call log line per call (name+args+success/fail, never the tool's
     actual output) followed by the final answer — sliding-windowed to
     MAX_HISTORY_TURNS/MAX_HISTORY_CHARS. The full LangGraph run (with real
     tool outputs) happens on a throwaway `working` message list per turn,
     never persisted.
  2. Per-turn guardrail reset (helpers/turn_guard.reset()) — repeat-call/quota
     checks live inside each tool wrapper. This module additionally hard-stops
     only after bash/web_search received one exact-duplicate wrapper hint and
     still emitted the same call again; changed arguments remain recoverable.
  3. A step budget via LangGraph's own recursion_limit config, and a crash
     safety net around .invoke() (server down, timeout, GraphRecursionError).
"""
from __future__ import annotations
import json
import functools
import os
import re
import shlex
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import config
from helpers import turn_guard
from helpers.date_inject import inject_datetime_context
from helpers.guardrails import has_leaked_tool_syntax, is_boilerplate_refusal, is_degenerate_repetition
from helpers import search_scope
from helpers.sensitive_paths import contains_secret_content, is_sensitive_path
from react import build_react_agent
from tools import bash as _bash_tool
from tools import local_search as _local_search_tool
from tools import mcp_client as _mcp_tool
from tools import read_image as _read_image_tool
from tools import web_search as _web_search_tool

_LOG_PATH = config.LOG_DIR / "session.jsonl"
_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3
_LOG_VALUE_MAX_CHARS = 8_000
_LOG_LOCK = threading.Lock()
_LOG_PATH_KEYS = {"path", "cwd", "file", "filename", "directory", "root"}

MAX_HISTORY_TURNS = 5
MAX_HISTORY_CHARS = 20_000

# Each ReAct round is 2 LangGraph steps (agent -> tools); +2 for the final
# answer step. Mirrors the old hand-rolled StepBudget's MAX_TOOL_STEPS.
_RECURSION_LIMIT = config.MAX_TOOL_STEPS * 2 + 2

_react_agents: dict[tuple[bool, int], object] = {}
_TURN_LOCK = threading.RLock()
_HINT_RETRY_TOOLS = frozenset({"bash", "web_search"})


@dataclass
class _ToolCallAttemptState:
    seen: set[tuple[str, str]] = field(default_factory=set)
    last_key: tuple[str, str] | None = None
    streak: int = 0


def _tool_call_attempt(
    state: _ToolCallAttemptState, name: str, args: dict,
) -> tuple[int, bool]:
    """Register a model-emitted call and decide whether outer hard-stop is due.

    bash/web_search use a consecutive streak: the second identical emission is
    allowed through so the wrapper can return its actionable hint; the third
    identical emission hard-stops. Any changed tool/args resets that streak.
    Other tools retain the historical second-identical-call stop anywhere in
    the turn.
    """
    key_args = (
        _read_image_tool.progressive_guard_args(args)
        if name == "read_image"
        else args
    )
    key = (name, json.dumps(key_args, sort_keys=True, ensure_ascii=False, default=str))
    if name in _HINT_RETRY_TOOLS:
        state.streak = state.streak + 1 if state.last_key == key else 1
        state.last_key = key
        state.seen.add(key)
        return state.streak, state.streak >= 3

    duplicate = key in state.seen
    state.seen.add(key)
    state.last_key = key
    state.streak = 1
    return (2 if duplicate else 1), duplicate


def _serialized_turn(fn):
    """Keep process-global tool guards and progressive image state turn-scoped."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _TURN_LOCK:
            user_input = args[0] if args else kwargs.get("user_input", "")
            token = search_scope.set_request_query(user_input)
            stale = _read_image_tool.begin_image_turn()
            if stale:
                _log({"event": "vision_stale_cleanup", "count": stale})
            _read_image_tool.purge_orphaned_pending_images()
            try:
                return fn(*args, **kwargs)
            finally:
                _read_image_tool.end_image_turn()
                search_scope.reset_request_query(token)
    return wrapped


def _get_react_agent(streaming: bool = False, thinking_budget: int | None = None):
    """Return a cached graph for this transport/thinking-budget pair.

    Tool selection, argument construction, and interpreting a tool result all
    benefit from the same bounded reasoning budget as a direct answer.  The UI
    may choose a bounded effort profile per session; cache one graph per
    profile so changing it affects the next turn without rebuilding repeatedly.
    """
    budget = config.THINKING_BUDGET if thinking_budget is None else int(thinking_budget)
    budget = max(0, min(4096, budget))
    key = (bool(streaming), budget)
    if key not in _react_agents:
        build_args = {"streaming": streaming}
        if thinking_budget is not None:
            build_args["extra_body"] = {"thinking_budget": budget}
        _react_agents[key] = build_react_agent(**build_args)
    return _react_agents[key]


def _response_has_tool_calls(response) -> bool:
    try:
        message = response.generations[0][0].message
        return bool(
            getattr(message, "tool_calls", None)
            or (getattr(message, "additional_kwargs", None) or {}).get("tool_calls")
        )
    except (AttributeError, IndexError, TypeError):
        return False


class _TurnCancelled(BaseException):
    """Interrupt a streaming model call as soon as the browser cancels it."""


class _TokenStreamCallback(BaseCallbackHandler):
    """Relay provisional model tokens through ``on_step`` for the HTML UI.

    A model call that ends in tool calls is deliberation, not the answer.  Its
    provisional text is discarded, matching MAX_VLM's browser behavior; the
    next model call then resumes in the same assistant bubble.
    """

    # BaseException escapes LangChain's ``except Exception`` callback wrapper
    # without producing a misleading callback-error log line.
    raise_error = False

    def __init__(self, on_step, cancel_event=None):
        super().__init__()
        self._on_step = on_step
        self._cancel_event = cancel_event
        self._token_counts: dict[str, int] = {}

    @staticmethod
    def _run_key(run_id) -> str:
        return str(run_id) if run_id is not None else "default"

    def on_chat_model_start(self, _serialized, _messages, *, run_id, **_kwargs):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise _TurnCancelled("turn cancelled by user")
        self._on_step(event="stream_start")
        self._token_counts[self._run_key(run_id)] = 0

    def on_llm_new_token(self, token: str, *, run_id=None, **_kwargs) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise _TurnCancelled("turn cancelled by user")
        if not token:
            return
        key = self._run_key(run_id)
        self._token_counts[key] = self._token_counts.get(key, 0) + 1
        self._on_step(event="token", text=token)

    def on_llm_end(self, response, *, run_id, **_kwargs) -> None:
        key = self._run_key(run_id)
        token_count = self._token_counts.pop(key, 0)
        if token_count and _response_has_tool_calls(response):
            self._on_step(event="discard_stream")
        else:
            self._on_step(event="stream_end")

    def on_llm_error(self, _error, *, run_id, **_kwargs) -> None:
        if self._token_counts.pop(self._run_key(run_id), 0):
            self._on_step(event="discard_stream")


def _compact_turn(user_text: str, tool_log: list[str], final_text: str) -> list[dict]:
    parts = []
    if tool_log:
        parts.append("[tool log]\n" + "\n".join(tool_log))
    parts.append(final_text)
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "\n\n".join(parts)},
    ]


def trim_history(messages: list[dict]) -> None:
    pairs = [messages[i : i + 2] for i in range(0, len(messages), 2)]
    pairs = pairs[-MAX_HISTORY_TURNS:]

    def _total_chars(ps: list[list[dict]]) -> int:
        return sum(len(m.get("content", "") or "") for pair in ps for m in pair)

    while len(pairs) > 1 and _total_chars(pairs) > MAX_HISTORY_CHARS:
        pairs.pop(0)

    messages[:] = [m for pair in pairs for m in pair]


def _to_lc_messages(messages: list[dict]) -> list:
    """Convert the persisted compact history (plain dicts) into LangChain
    message objects for a fresh `working` list passed to the react agent."""
    lc = []
    for m in messages:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    return lc


_ABSOLUTE_USER_PATH_RE = re.compile(r"/(?:Users|Volumes|private|tmp)/[^\s`\]>,)]+")
_PATH_LIKE_QUERY_TOKEN_RE = re.compile(
    r"(?:~|/(?:Users|Volumes|private|tmp|var|etc|Library|System|opt|home))/[^\s`\]>,)]+"
)
_WEB_REQUEST_RE = re.compile(
    r"(?:ค้นหา(?:ว่า)?|ค้นข้อมูล|หาข้อมูล|แนบลิงก์|แหล่งที่มา|ล่าสุด|ตอนนี้|ปัจจุบัน|ข่าว|\bsearch\b|\blook\s+up\b|\bnow\b|\blatest\b|\bcurrent\b|\bsources?\b)",
    re.IGNORECASE,
)
_WEB_SEARCH_FOLLOWUP_RE = re.compile(
    r"^\s*(?:(?:คุณ|ช่วย|กรุณา)\s*)?(?:ไป\s*)?(?:ค้นหา|search(?:\s+it)?)"
    r"\s*(?:ข้อมูล\s*)?(?:ก่อน\s*)?(?:ตอบ|answer)(?:\s*(?:สิ|ด้วย|นะ|หน่อย))?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_LOCAL_FILE_REQUEST_RE = re.compile(
    r"(?:ไฟล์|เอกสาร|ในเครื่อง|บนเครื่อง|โปรเจกต์|โฟลเดอร์|\bpath\b|/Users/|/Volumes/|\bfile\b|\bdocument\b)",
    re.IGNORECASE,
)
_WEB_FACET_SPLIT_RE = re.compile(
    r"\s*(?:และ|(?<!เกี่ยว)กับ|/|,)\s*",
    re.IGNORECASE,
)


def _should_force_web_search(query: str) -> bool:
    """Catch a missed web route without affecting local file requests."""
    return (
        bool(_WEB_REQUEST_RE.search(query or ""))
        and not _LOCAL_FILE_REQUEST_RE.search(query or "")
        and not search_scope.is_knowledge_base_query(query)
        and not search_scope.is_knowledge_base_explanation_query(query)
    )


def _is_web_search_followup(query: str) -> bool:
    """Recognize a bare request to research the immediately prior topic."""
    return bool(_WEB_SEARCH_FOLLOWUP_RE.fullmatch((query or "").strip()))


def _web_search_subject(user_input: str, messages: list[dict]) -> str:
    """Use the prior substantive question for a bare "search first" follow-up."""
    if not _is_web_search_followup(user_input):
        return user_input.strip()
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        candidate = str(message.get("content") or "").strip()
        if candidate and not _is_web_search_followup(candidate):
            return candidate
    return user_input.strip()


def _focused_web_queries(subject: str) -> str | list[str]:
    """Split a current broad topic into bounded, literal search facets.

    This is query shaping, not answer generation: it only preserves subjects
    already present in the user's text and adds the current calendar year when
    they explicitly asked about the present. It avoids asking one result page
    to match two unrelated subjects such as war risk and investment outlook.
    """
    value = (subject or "").strip()
    if not value:
        return value
    current = bool(re.search(r"(?:ตอนนี้|ปัจจุบัน|ล่าสุด|\bnow\b|\bcurrent\b|\blatest\b)", value, re.IGNORECASE))
    year = str(date.today().year)
    if re.search(r"สงคราม|ความขัดแย้ง", value, re.IGNORECASE) and re.search(r"ลงทุน|ตลาด(?:การเงิน)?", value, re.IGNORECASE):
        return [
            f"ความขัดแย้งระหว่างประเทศ {year}",
            f"การลงทุนโลก {year}",
        ]
    if not current and not _WEB_FACET_SPLIT_RE.search(value):
        return value

    simplified = re.sub(
        r"(?:บรรยากาศ|สถานการณ์|ภาพรวม|ตอนนี้|ปัจจุบัน|ล่าสุด|เป็นอย่างไร|มีอะไรบ้าง|ช่วยค้นหา(?:ว่า)?|ค้นหา(?:ว่า)?)",
        " ", value, flags=re.IGNORECASE,
    )
    facets = []
    for part in _WEB_FACET_SPLIT_RE.split(simplified):
        facet = part.strip(" ,.?!")
        if not facet:
            continue
        # "เกี่ยวกับ" is a preposition ("about"), not the conjunction "กับ".
        # When a real later conjunction creates multiple facets, remove only
        # this leading relation marker from the affected facet.
        facet = re.sub(r"^เกี่ยวกับ\s*", "", facet, flags=re.IGNORECASE).strip()
        if facet:
            facets.append(facet)
    if not facets:
        facets = [value]
    if current and not re.search(r"\b(?:20|25)\d{2}\b", value):
        facets = [f"{facet} {year}" for facet in facets]
    unique = list(dict.fromkeys(facets))[:4]
    return unique[0] if len(unique) == 1 else unique


def _web_recovery_query(query: str) -> str | None:
    """Choose one code-owned, materially different retry for empty web evidence."""
    official_variant = getattr(_web_search_tool, "_official_site_variant", None)
    if callable(official_variant):
        candidate = official_variant(query)
        if candidate and candidate.strip() != query.strip():
            return candidate.strip()
    return None


def _has_out_of_scope_path(text: str) -> bool:
    if not search_scope.is_project_scoped_query():
        return False
    root = config.PROJECT_ROOT.resolve()
    for raw_path in _ABSOLUTE_USER_PATH_RE.findall(text or ""):
        path = Path(raw_path.rstrip(".,;:，。"))
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError:
            return True
    return False


def _query_requests_sensitive_path(query: str) -> bool:
    """Recognize a sensitive path request before a model can soften the result."""
    for raw_path in _PATH_LIKE_QUERY_TOKEN_RE.findall(query or ""):
        path = raw_path.rstrip(".,;:，。!?！？")
        if is_sensitive_path(path):
            return True
    return False


def _logical_tool_name(name: str, args: dict | None = None) -> str:
    """Map the fixed Lite RAG MCP call back to its semantic tool name.

    Transport remains ``mcp_call_tool`` in LangGraph/Activity events; this map
    exists only so the established RAG grounding and recovery invariants keep
    working after moving RAG behind MCP.
    """
    if name == "mcp_call_tool" and isinstance(args, dict):
        if args.get("server") == "lite-rag":
            tool_name = str(args.get("tool_name") or "").strip()
            if tool_name == "rag_search":
                return "rag_search"
            raw = args.get("arguments_json") or "{}"
            try:
                payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if tool_name == "rag_manage":
                action = str(payload.get("action") or "").strip().casefold()
                if action == "sync":
                    return "sync_kb"
                if action == "cancel":
                    return "cancel_sync"
            if tool_name == "rag_status":
                view = str(payload.get("view") or "health").strip().casefold()
                if view == "health":
                    return "rag_health"
                if view == "job":
                    return "sync_status"
    return name


def _normalize_mcp_tool_args(tool_name: str, args: dict) -> dict:
    """Repair the one safe shorthand emitted by the small model.

    Qwen occasionally emits ``mcp_call_tool(arguments_json=...)`` after a
    discovery call, omitting the two constant routing fields.  The only
    provisioned Lite MCP server is the local RAG server, so filling these
    defaults is bounded and lets the actual wrapper validate/execute the
    query instead of turning a parser omission into a duplicate-call abort.
    Explicit server/tool values are never overwritten.
    """
    if tool_name != "mcp_call_tool" or not isinstance(args, dict):
        return args
    normalized = dict(args)
    if not str(normalized.get("server") or "").strip():
        normalized["server"] = "lite-rag"
    if not str(normalized.get("tool_name") or "").strip():
        normalized["tool_name"] = "rag_search"
    return normalized


def _ground_project_search_answer(final_text: str, last_tool_name: str | None, last_tool_result: str | None) -> str:
    """Do not display a sibling-project path hallucinated after a local search."""
    if last_tool_name not in {"rag_search", "bash", "local_search"} or not last_tool_result:
        return final_text
    if not _has_out_of_scope_path(final_text):
        return final_text
    result = _strip_tool_reminder(last_tool_result)
    if _has_out_of_scope_path(result):
        return "ไม่พบผลลัพธ์ที่อยู่ในโปรเจกต์ปัจจุบันจากคำค้นนี้ครับ"
    heading = "ผลการค้นหาใน Desktop/Documents:" if last_tool_name == "local_search" else "ผลการค้นหาในโปรเจกต์ปัจจุบัน:"
    return f"{heading}\n\n{result}"


def _is_project_no_result_answer(text: str) -> bool:
    """Recognize a small-model refusal that contradicts local evidence."""
    normalized = (text or "").casefold()
    return bool(re.search(
        r"(?:ไม่พบ|ไม่เจอ|ไม่มีผล|ไม่มีไฟล์|ไม่พบข้อมูล|no results?|not found|couldn't find|cannot find)",
        normalized,
    ))


def _has_useful_project_rag_evidence(text: str | None) -> bool:
    """Whether a project RAG result can safely outlive a later model loop."""
    if not text or text.startswith(("[error]", "ไม่พบไฟล์", "ไม่พบผลลัพธ์")):
        return False
    return "/" in text and not _is_project_no_result_answer(text)


_RAG_ABSOLUTE_PATH_LINE_RE = re.compile(
    r"^\s*-\s+((?:/(?:Users|Volumes|private|tmp|var|etc|Library|System|opt|home))/[^\r\n]+?)\s*$",
    re.MULTILINE,
)


def _restore_known_local_paths(final_text: str, rag_result: str | None) -> str:
    """Restore absolute links when the model shortens verified RAG paths.

    Small models often turn ``- /absolute/path/file.pdf`` into a relative
    bullet or a bare filename while summarizing. They can also preserve a
    relative directory prefix, so both forms are canonicalized. Only names
    that occur exactly once in the successful RAG result are expanded, so
    this cannot invent a directory or resolve an ambiguous filename.
    """
    if not final_text or not rag_result:
        return final_text
    paths = [match.group(1).rstrip(".,;:，。") for match in _RAG_ABSOLUTE_PATH_LINE_RE.finditer(rag_result)]
    by_name: dict[str, list[str]] = {}
    by_normalized_name: dict[str, list[str]] = {}
    for path in paths:
        name = Path(path).name
        by_name.setdefault(name.casefold(), []).append(path)
        normalized_name = re.sub(r"[^\w]+", "", name.casefold(), flags=re.UNICODE)
        by_normalized_name.setdefault(normalized_name, []).append(path)
    restored = final_text
    for name, candidates in by_name.items():
        if len(candidates) != 1:
            continue
        absolute = candidates[0]
        basename = Path(absolute).name
        restored = re.sub(
            rf"(?<![/\w]){re.escape(basename)}",
            absolute,
            restored,
            flags=re.IGNORECASE,
        )
    # A model may keep the RAG-relative directory, e.g.
    # ``เอกสารเก่า Syllabus/เอกสารเก่า/Syllabus 402.xlsx``. Replace that whole
    # line-level path, rather than only the basename (which would leave a
    # broken relative prefix before the absolute link).
    label_re = re.compile(
        r"\s+\((?:Word|PDF|Excel|PowerPoint|Text|Image|Spreadsheet|Presentation)\)\s*$",
        flags=re.IGNORECASE,
    )
    for line in restored.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        line_ending = line[len(line_body):]
        if re.search(r"/(?:Users|Volumes|private|tmp|var|etc|Library|System|opt|home)/", line_body):
            continue
        content_match = re.match(r"^(\s*(?:[-*+]\s+)?)", line_body)
        content_start = content_match.end() if content_match else 0
        content = line_body[content_start:].strip()
        label_match = label_re.search(content)
        label = label_match.group(0) if label_match else ""
        path_text = content[:label_match.start()] if label_match else content
        path_text = path_text.strip().strip("`*").rstrip(".,;:，。")
        if "/" not in path_text:
            continue
        candidate_name = Path(path_text).name
        normalized_name = re.sub(r"[^\w]+", "", candidate_name.casefold(), flags=re.UNICODE)
        candidates = by_normalized_name.get(normalized_name, [])
        if len(candidates) != 1:
            continue
        canonical = candidates[0]
        replacement = line[:content_start] + canonical + label + line_ending
        restored = restored.replace(line, replacement, 1)
    # Models sometimes alter punctuation in a filename while copying a
    # result (for example ``ฝป..pdf`` instead of the indexed ``ฝป.pdf``).
    # For absolute path lines, repair only a unique filename whose letters
    # and digits match after removing punctuation; this keeps the directory
    # and the extension grounded in the successful RAG result.
    for line in restored.splitlines(keepends=True):
        match = re.search(r"/(?:Users|Volumes|private|tmp|var|etc|Library|System|opt|home)/", line)
        if not match:
            continue
        tail = line[match.start():].rstrip("\r\n")
        label = ""
        label_match = re.search(
            r"\s+\((?:Word|PDF|Excel|PowerPoint|Text|Image|Spreadsheet|Presentation)\)\s*$",
            tail,
            flags=re.IGNORECASE,
        )
        if label_match:
            label = tail[label_match.start():]
            tail = tail[:label_match.start()]
        candidate_name = Path(tail.rstrip(".,;:，。")).name
        normalized_name = re.sub(r"[^\w]+", "", candidate_name.casefold(), flags=re.UNICODE)
        candidates = by_normalized_name.get(normalized_name, [])
        if len(candidates) != 1:
            continue
        canonical = candidates[0]
        if tail == canonical:
            continue
        line_body = line.rstrip("\r\n")
        line_ending = line[len(line_body):]
        replacement = line[:match.start()] + canonical + label + line_ending
        restored = restored.replace(line, replacement, 1)
    return restored


def _normalize_write_tool_args(tool_name: str, args: dict, user_query: str = "") -> dict:
    """Resolve a user-friendly relative workspect path before write_file.

    The tool itself remains strict and accepts only absolute paths; this
    orchestration adapter turns the common ``workspect/foo.txt`` request into
    the exact permitted project path without broadening the write boundary.
    """
    if tool_name != "write_file" or not isinstance(args.get("path"), str):
        return args
    raw_path = args["path"].strip()
    if raw_path and not raw_path.startswith("~"):
        candidate = Path(raw_path).expanduser()
        relative = candidate
        if candidate.is_absolute():
            try:
                relative = candidate.resolve(strict=False).relative_to(config.PROJECT_ROOT.resolve())
            except ValueError:
                return args
        else:
            # Small models sometimes drop the leading slash from an absolute
            # path (for example ``/tmp/x`` becomes ``tmp/x``).  Recover the
            # path from the user's request so the strict write tool can reject
            # it instead of silently redirecting it under workspect.  The
            # same rule preserves legitimate absolute paths inside the
            # project; it is not a filename-specific exception.
            raw_name = candidate.name
            for requested in _ABSOLUTE_USER_PATH_RE.findall(user_query or ""):
                requested_path = Path(requested.rstrip(".,;:，。"))
                if requested_path.name == raw_name:
                    args["path"] = str(requested_path)
                    return args
        # Accept both ``workspect/foo.txt`` (the user-facing project-relative
        # form) and ``foo.txt`` (the small model's shortened form), but never
        # resolve a relative write against the source-tree root.
        if relative.parts[:1] == (config.BASH_ALLOWED_ROOT.name,):
            relative = Path(*relative.parts[1:])
        args["path"] = str(config.BASH_ALLOWED_ROOT / relative)
    return args


def _user_explicitly_allows_overwrite(query: str) -> bool:
    return bool(re.search(
        r"(?:เขียนทับ|replace\s+(?:the\s+)?file|overwrite\s*(?:=|:)?\s*(?:true|yes|อนุญาต))",
        query or "", re.IGNORECASE,
    )) and not bool(re.search(r"(?:ห้าม|ไม่|not|don't|do not)\s*(?:เขียนทับ|overwrite|replace)", query or "", re.IGNORECASE))


def _apply_first_write_overwrite_gate(args: dict, user_query: str, gate_tripped: bool) -> tuple[dict, bool]:
    """Require the tool's safe default before accepting model self-recovery."""
    if (
        args.get("overwrite", "").strip().lower() == "true"
        and not gate_tripped
        and not _user_explicitly_allows_overwrite(user_query)
    ):
        args["overwrite"] = ""
        return args, True
    return args, gate_tripped


_SOURCE_URL_RE = re.compile(r"\[source:\s*(https?://[^\]]+)\]")
_SOURCE_FILE_RE = re.compile(
    r"^\s*\[source:\s*(/[^\]\r\n]+?)\s*\]\s*$", re.MULTILINE,
)
_RAG_FILE_LINE_RE = re.compile(r"^\s*-\s+ไฟล์:\s*(/[^\r\n]+?)\s*$")
_RAG_LOCATION_LINE_RE = re.compile(r"^\s*หน้า/ตำแหน่ง:\s*(.+?)\s*$")
_PERCENT_CLAIM_RE = re.compile(
    r"(?<![\w.])\d+(?:[.,]\d+)?\s*(?:%|เปอร์เซ็นต์|ร้อยละ)", re.IGNORECASE,
)


def _source_urls(web_search_results: list[str]) -> list[str]:
    urls: list[str] = []
    for result in web_search_results:
        for url in _SOURCE_URL_RE.findall(result):
            if url not in urls:
                urls.append(url)
    return urls


def _source_file_refs(rag_search_results: list[str]) -> list[tuple[str, str]]:
    """Extract verified Knowledge Base files and their stored locations.

    ``tools/rag_search.py`` emits the same ``[source: ...]`` marker shape as
    web search, but the source is an absolute local path.  The immediately
    preceding file/location fields are retained so page/line provenance is
    not lost when the model summarizes the result.
    """
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for result in rag_search_results:
        current_path: str | None = None
        current_location = ""
        for line in (result or "").splitlines():
            file_match = _RAG_FILE_LINE_RE.match(line)
            if file_match:
                current_path = file_match.group(1).strip()
                current_location = ""
                continue
            location_match = _RAG_LOCATION_LINE_RE.match(line)
            if location_match:
                current_location = location_match.group(1).strip()
                continue
            marker = _SOURCE_FILE_RE.match(line)
            if not marker:
                continue
            path = marker.group(1).strip()
            if current_path != path:
                current_location = ""
            ref = (path, current_location)
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _usable_web_results(web_search_results: list[str]) -> list[str]:
    """Return source-bearing evidence, excluding retry/error instructions."""
    return [
        result for result in web_search_results
        if result
        and not result.lstrip().startswith(("[error]", "[blocked]", "[needs_retry]"))
        and _source_urls([result])
    ]


def _best_web_evidence(web_search_results: list[str]) -> str | None:
    """Choose one complete result for deterministic completion recovery.

    A model can spend its whole turn issuing changing web queries and then
    return the final ``[needs_retry]`` instruction. Prefer a vetted result
    with an authoritative source when one exists; otherwise preserve the
    first usable result rather than exposing the last failed retry.
    """
    usable = _usable_web_results(web_search_results)
    if not usable:
        return None
    authoritative = getattr(_web_search_tool, "_is_authoritative_url", None)
    if callable(authoritative):
        for result in usable:
            if any(authoritative(url) for url in _source_urls([result])):
                return result
    return usable[0]


def _unsupported_web_percentages(final_text: str, web_search_results: list[str]) -> list[str]:
    """Find model-written percentages absent from all vetted snippets."""
    evidence = "\n".join(web_search_results)

    def normalized(value: str) -> str:
        return re.sub(r"\s+", "", value).replace(",", ".").casefold()

    supported = {normalized(value) for value in _PERCENT_CLAIM_RE.findall(evidence)}
    return [
        value for value in _PERCENT_CLAIM_RE.findall(final_text)
        if normalized(value) not in supported
    ]


def _remove_model_copied_urls(text: str, urls: list[str]) -> str:
    """Keep search URLs in the deterministic footer, never model prose."""
    for url in urls:
        escaped = re.escape(url)
        # Preserve a model-written Markdown label while removing its link.
        text = re.sub(rf"\[([^\]]+)\]\(\s*{escaped}\s*\)", r"\1", text)
        text = re.sub(rf"<?{escaped}>?", "", text)
    # URL removal can leave trailing whitespace or empty bullets.
    text = re.sub(r"(?m)^\s*[-*]\s*$\n?", "", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _append_source_citations(final_text: str, web_search_results: list[str]) -> str:
    """Deterministically attach source URLs from this turn's web_search
    calls, rather than relying on the model to remember to cite them.

    2026-07-24 (user request): a prompt rule + a re-ask nudge (tried and
    removed) both left citation compliance probabilistic -- the model
    sometimes wrote a complete, correct summary and simply omitted the
    link. tools/web_search.py already tags every result with a machine-
    parseable "[source: URL]" marker (see its docstring); extracting and
    appending those here makes the citation 100% reliable and lets the
    model's job shrink to just summarizing, matching the training guide's
    rule: enforce deterministically in code when the behavior must always
    be correct, not merely usually correct.
    """
    urls = _source_urls(web_search_results)
    if not urls:
        return final_text
    final_text = _remove_model_copied_urls(final_text, urls)
    footer = "\n\nแหล่งที่มา:\n" + "\n".join(f"- {u}" for u in urls)
    return final_text + footer


def _append_rag_source_citations(final_text: str, rag_search_results: list[str]) -> str:
    """Deterministically attach Knowledge Base source files to the answer.

    RAG results are local files rather than web URLs, so preserve both the
    absolute path and the page/line location stored on the SQLite chunk.  The
    model may copy the marker into its prose; remove that copy first so the
    footer remains the single source-of-truth presentation.
    """
    refs = _source_file_refs(rag_search_results)
    if not refs:
        return final_text
    final_text = _SOURCE_FILE_RE.sub("", final_text or "")
    final_text = re.sub(r"(?m)^\s*[-*]\s*$\n?", "", final_text)
    final_text = re.sub(r"[ \t]+\n", "\n", final_text).strip()
    lines = []
    for path, location in refs:
        suffix = f" — {location}" if location else ""
        lines.append(f"- {path}{suffix}")
    return final_text + "\n\nแหล่งที่มา (Knowledge Base):\n" + "\n".join(lines)


_TOOL_REMINDER_RE = re.compile(r"\n\n\[(?:bash usage notes|reminder)\].*", re.DOTALL)
_NUMBER_RE = re.compile(r"(?<![\w])\d[\d,]*(?:\.\d+)?")
_PAREN_RE = re.compile(r"\s*\(([^()]*)\)")
_IMAGE_TEXT_REQUEST_RE = re.compile(r"(?:ข้อความ|ตัวอักษร|อ่านรูป|อ่านภาพ|\b(?:text|ocr)\b)", re.IGNORECASE)
_OCR_RESULT_RE = re.compile(r"\[OCR\] ข้อความในภาพ:\n(.+?)(?=\n\[(?:QR|หมายเหตุ)\]|\Z)", re.DOTALL)
_FULL_TOOL_OUTPUT_RE = re.compile(
    r"\[full output:\s*(\d+)\s+lines;\s*saved:\s*([^;\]]+);\s*preview below is incomplete\]"
)

# Lite has no remember/edit tool: this guard is deliberately limited to
# explicit file creation/saving that must culminate in write_file.
_WRITE_ACTION_RE = re.compile(
    r"(?:บันทึก|เซฟ|เขียน|สร้าง|ทำ|save|write|create).{0,80}"
    r"(?:ลง(?:ใน)?|เป็น|to)?\s*(?:ไฟล์|file|[\w./~-]+\.(?:md|txt|csv|json|yaml|yml|py|js|ts|html|css))",
    re.IGNORECASE,
)
_WRITE_RETRY_PROMPT = (
    "[WRITE-FILE RETRY]\n"
    "คำสั่งล่าสุดให้สร้างหรือบันทึกไฟล์ แต่รอบก่อนยังไม่มี write_file tool call; action ยังไม่เสร็จ. "
    "ต้องเรียก write_file(path, content) ตอนนี้ ห้ามตอบเพียงว่าเขียนแล้ว. "
    "ตัวอย่าง: write_file(path=\"report.md\", content=\"...\")\n\n"
    "Original user request: {query}"
)

_KNOWN_LOCAL_IMAGE_PATH_RE = re.compile(
    r"/(?:Users|Volumes|private|var|tmp)/[^\n]+\.(?:png|jpe?g|webp|gif|bmp|tiff?|heic|heif)\b",
    re.IGNORECASE,
)
_LOCAL_DOCUMENT_FILENAME_RE = re.compile(
    r"[^\s\"'“”‘’<>`]+\.(?:md|markdown|txt|text|rtf|csv|tsv|pdf|docx?|xlsx?|pptx?|odt|ods|odp|"
    r"jpe?g|png|webp|gif|bmp|tiff?|heic|heif)\b",
    re.IGNORECASE,
)
_LOCAL_CONTENT_QUERY_RE = re.compile(
    r"(?:เกี่ยวกับ|พูดถึง|กล่าวถึง|มีคำว่า|คำว่า)\s*"
    r"(?P<term>[^\n.!?！？。，、]+?)"
    r"(?=\s*(?:แล้ว|และ|ให้|หน่อย|บ้าง|อะไร|เท่าไร|เท่าไหร่|อยู่ที่ไหน|อยู่ไหน|$))",
    re.IGNORECASE,
)


def _write_action_required(query: str) -> bool:
    return bool(_WRITE_ACTION_RE.search(query))


def _requested_image_path(query: str) -> str | None:
    """Return a user-supplied absolute image path for deterministic routing.

    A 2B model occasionally interpreted the global RAG-before-bash rule as
    RAG-before-every-tool and searched for an image that was already named.
    This route is unambiguous and cheaper to enforce before the first model
    call: read_image both returns OCR and queues the pixels for the VLM.
    """
    match = _KNOWN_LOCAL_IMAGE_PATH_RE.search(query or "")
    return match.group(0) if match else None


def _local_document_search_query(query: str) -> str | None:
    """Extract an explicit filename or content phrase for local recovery.

    This is intentionally narrower than a general natural-language router:
    it activates only when the user supplied a supported filename or a
    clearly delimited phrase such as ``พูดถึงค่าปรับ``.  The model remains
    responsible for ordinary semantic keyword selection; this fallback exists
    to finish an obvious local lookup when it diverts to a blocked web call or
    emits no RAG call.
    """
    for match in _LOCAL_DOCUMENT_FILENAME_RE.findall(query or ""):
        candidate = match.rstrip(".,;:!?！？，。、")
        if "/" not in candidate and not candidate.startswith("~"):
            return candidate
    for match in _LOCAL_CONTENT_QUERY_RE.finditer(query or ""):
        candidate = re.sub(r"\s+", " ", match.group("term")).strip(" .,;:!?！？，。、")
        if len(candidate) >= 2:
            return candidate
    return None


def _rag_search_query(query: str) -> str | None:
    """Extract a bounded content query after an explicit RAG/KB scope marker."""
    candidate = _local_document_search_query(query)
    if candidate:
        return candidate
    value = query or ""
    value = re.sub(
        r"(?:ช่วย\s*)?(?:ค้นหา|ค้น|หา|ดึงข้อมูล|ดึง|เปิด|อ่าน|สรุป|แสดง|อ้างอิง|ตอบ)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:ใน|จาก|ภายใน|ผ่าน|อ้างอิงจาก)\s*"
        r"(?:rag_document/(?:raw|md)|\brag_document\b|\b(?:rag|kb)\b|"
        r"knowledge\s*base|knowledge/(?:raw|md)|ฐานความรู้|คลังความรู้)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:rag_document/(?:raw|md)|knowledge/(?:raw|md)|"
        r"\b(?:rag|kb|rag_document)\b|knowledge\s*base|ฐานความรู้|คลังความรู้)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:ที่มี(?:เนื้อหา)?|เนื้อหา|เกี่ยวกับ|เรื่อง|หัวข้อ)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" .,;:!?！？，。、")
    return value or None


def _outside_rag_search_query(query: str) -> str | None:
    """Extract a short content/filename query for an explicit bash route."""
    candidate = _local_document_search_query(query)
    if candidate:
        return candidate
    value = query or ""
    value = re.sub(
        r"(?:ช่วย\s*)?(?:ค้นหา|ค้น|หา|ดึงข้อมูล|ดึง|เปิด|อ่าน|สรุป|แสดง|อ้างอิง|ตอบ)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:นอก|นอกจาก|ไม่ใช้|ไม่ต้องใช้|ไม่ค้นใน|ไม่ค้นจาก|นอกขอบเขต)\s*"
        r"(?:rag_document/(?:raw|md)|\brag_document\b|\b(?:rag|kb)\b|"
        r"knowledge\s*base|knowledge/(?:raw|md)|ฐานความรู้|คลังความรู้)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:rag_document/(?:raw|md)|knowledge/(?:raw|md)|"
        r"\b(?:rag|kb|rag_document)\b|knowledge\s*base|ฐานความรู้|คลังความรู้)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:ใน|จาก|ภายใน|ผ่าน|ที่มี(?:เนื้อหา)?|เนื้อหา|เกี่ยวกับ|เรื่อง|หัวข้อ)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" .,;:!?！？，。、")
    return value or None


def _bounded_bash_search_command(term: str | None, raw_roots) -> str | None:
    """Build one read-only filename/content search inside explicit local roots."""
    if not term:
        return None
    roots = []
    for raw_root in raw_roots:
        try:
            root = Path(raw_root).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if root.is_dir() and not is_sensitive_path(root) and root not in roots:
            roots.append(root)
    if not roots:
        return None
    quoted_roots = " ".join(shlex.quote(str(root)) for root in roots)
    if _LOCAL_DOCUMENT_FILENAME_RE.fullmatch(term):
        return f"find {quoted_roots} -type f -name {shlex.quote(term)}"
    return f"grep -r -l -F {shlex.quote(term)} {quoted_roots}"


def _outside_rag_bash_command(query: str) -> str | None:
    """Build one bounded, read-only bash search over Desktop/Documents."""
    return _bounded_bash_search_command(
        _outside_rag_search_query(query),
        getattr(config, "LOCAL_SEARCH_ROOTS", ()),
    )


def _rag_fallback_bash_command(query: str, *, project_scope: bool = False) -> str | None:
    """Search the original local scope when the optional RAG MCP is unavailable."""
    term = (
        _local_document_search_query(query) or _rag_search_query(query)
        if project_scope
        else _rag_search_query(query) or _local_document_search_query(query)
    )
    roots = (config.PROJECT_ROOT,) if project_scope else (config.KNOWLEDGE_DIR,)
    return _bounded_bash_search_command(term, roots)


def _rag_mcp_unavailable(result: str | None) -> bool:
    """Recognize transport/capability failures that justify a scoped bash fallback."""
    if not result or not result.startswith(("[error]", "[blocked]")):
        return False
    normalized = result.casefold()
    return any(marker in normalized for marker in (
        "no mcp server named",
        "mcp_call_tool failed",
        "tool 'rag_search' not found",
        "tool not found",
        "unavailable",
        "connection",
        "connecterror",
        "filenotfounderror",
        "no such file",
    ))


def _strip_tool_reminder(text: str) -> str:
    """tools/bash.py and tools/rag_search.py append a model-facing first-
    call-this-turn reminder (routing/retry guidance) to their own return
    value -- meant for the model's next decision, not for display when that
    raw result is surfaced directly to the user (see BUG-004 fallback)."""
    return _TOOL_REMINDER_RE.sub("", text)


def _strip_unsupported_local_numbers(final_text: str, evidence: list[str], query: str) -> str:
    """Remove numeric details invented while summarizing local evidence.

    The 2B model was observed copying a Thai date correctly and then adding
    a false year conversion, and adding a date to an image that contained no
    date.  Prompt wording did not change this across repeated live runs.
    Preserve every number present in a successful local tool result; remove
    only unsupported numeric asides/lines when the user did not request a
    calculation or conversion.
    """
    if not evidence or re.search(r"(?:แปลง|คำนวณ|calculate|convert)", query, re.IGNORECASE):
        return final_text
    supported = {token.replace(",", "") for token in _NUMBER_RE.findall("\n".join(evidence))}
    if not supported:
        return final_text

    def keep_or_drop_aside(match: re.Match) -> str:
        numbers = {token.replace(",", "") for token in _NUMBER_RE.findall(match.group(1))}
        return "" if numbers - supported else match.group(0)

    cleaned = _PAREN_RE.sub(keep_or_drop_aside, final_text)
    kept_lines: list[str] = []
    for line in cleaned.splitlines():
        # Ignore Markdown/list ordinals when deciding whether a factual line
        # contains evidence-backed numbers.
        factual = re.sub(r"^\s*(?:[-*]\s*)?\d+[.)]\s+", "", line)
        numbers = {token.replace(",", "") for token in _NUMBER_RE.findall(factual)}
        if numbers and not (numbers & supported):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


_LOCAL_FACT_REQUEST_RE = re.compile(
    r"(?:เท่าไร|เท่าไหร่|จำนวนเท่าใด|(?<!เ)กี่(?:บาท|คน|ไฟล์|วัน|ครั้ง)?(?!ย)|"
    r"ยอด(?:เงิน|ชำระ)?|ราคา|วันไหน|เมื่อไร|เมื่อไหร่|how much|how many|when)",
    re.IGNORECASE,
)


def _ground_local_fact_answer(final_text: str, rag_result: str | None, query: str) -> str:
    """Preserve a requested numeric fact the model dropped from RAG evidence."""
    if not final_text or not rag_result or not _LOCAL_FACT_REQUEST_RE.search(query or ""):
        return final_text
    evidence_numbers = []
    seen_numbers: set[str] = set()
    for raw in _NUMBER_RE.findall(rag_result):
        normalized = raw.replace(",", "")
        if normalized not in seen_numbers:
            seen_numbers.add(normalized)
            evidence_numbers.append(raw)
    answer_numbers = {raw.replace(",", "") for raw in _NUMBER_RE.findall(final_text)}
    missing = [raw for raw in evidence_numbers if raw.replace(",", "") not in answer_numbers]
    if not missing:
        return final_text

    # Prefer actual snippet lines over result-count/byte-size headers. The raw
    # RAG response is already bounded and source-grounded, so this is a
    # deterministic recovery of the fact rather than a new summary.
    evidence_lines = []
    for line in rag_result.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("พบ ", "[", "- /")):
            continue
        line_numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(stripped)}
        if line_numbers.intersection({n.replace(",", "") for n in missing}):
            evidence_lines.append(stripped)
    if not evidence_lines:
        return final_text
    return (
        f"{final_text.rstrip()}\n\n"
        "รายละเอียดตัวเลขจากข้อความในเอกสาร:\n"
        + "\n".join(f"- {line}" for line in evidence_lines[:3])
    )


def _ground_image_text_answer(final_text: str, image_result: str, query: str) -> str:
    """Use deterministic OCR for an explicit image-text request.

    Visual description remains the VLM's job.  When the user specifically
    asks what text is present, the local OCR result is already the grounded
    answer and avoids a small model dropping or replacing it during prose
    synthesis.
    """
    if not _IMAGE_TEXT_REQUEST_RE.search(query or ""):
        return final_text
    match = _OCR_RESULT_RE.search(image_result or "")
    if not match:
        return final_text
    ocr = match.group(1).strip()
    if not ocr or ocr.startswith("ไม่พบข้อความ"):
        return final_text
    return f"ข้อความที่อ่านได้จากภาพ:\n\n{ocr}"


def _fallback_summary(
    tool_log: list[str],
    last_tool_name: str | None = None,
    last_tool_result: str | None = None,
) -> str:
    """Deterministic Thai stand-in for a degenerate boilerplate final answer
    (see helpers.guardrails.is_boilerplate_refusal) -- built from the tool
    log's own success/fail record rather than trusting the model to retry the
    summary correctly a second time.  If a tool already returned usable
    evidence, expose that verified result instead of replacing a partially
    successful turn with a content-free apology."""
    if not tool_log:
        return "ขออภัย ไม่สามารถสรุปคำตอบได้ในตอนนี้ กรุณาถามใหม่อีกครั้ง"
    if all(line.endswith("ล้มเหลว") for line in tool_log):
        return "ลองค้นหา/ตรวจสอบแล้วแต่ไม่พบข้อมูลที่ต้องการ กรุณาระบุคำค้นหรือชื่อไฟล์ให้ชัดเจนขึ้น"
    if last_tool_result and not last_tool_result.startswith(("[error]", "[blocked]")):
        success_count = sum("-> สำเร็จ" in line for line in tool_log)
        failure_count = len(tool_log) - success_count
        tool_label = last_tool_name or "เครื่องมือ"
        return (
            f"ตรวจสอบด้วย {tool_label} แล้ว แต่โมเดลสรุปผลไม่ทัน "
            f"(สำเร็จ {success_count} ครั้ง, ล้มเหลว {failure_count} ครั้ง) "
            "จึงแสดงผลที่ยืนยันได้จากเครื่องมือ:\n\n"
            f"{_strip_tool_reminder(last_tool_result)}"
        )
    return "ได้ข้อมูลจากเครื่องมือค้นหาแล้ว แต่สรุปคำตอบไม่สำเร็จ กรุณาถามอีกครั้งหรือระบุคำถามให้ชัดเจนขึ้น"


def _redact_log_value(value, *, key: str = ""):
    """Keep diagnostics useful without persisting credential-shaped values."""
    if isinstance(value, dict):
        return {str(name): _redact_log_value(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_log_value(item, key=key) for item in value]
    text = str(value)
    if contains_secret_content(text):
        return "[redacted: sensitive value]"
    if key.casefold() in _LOG_PATH_KEYS and is_sensitive_path(text):
        return "[redacted: sensitive path]"
    return text[:_LOG_VALUE_MAX_CHARS]


def _rotate_session_log() -> None:
    try:
        if _LOG_PATH.stat().st_size <= _LOG_MAX_BYTES:
            return
    except OSError:
        return
    for index in range(_LOG_BACKUP_COUNT, 0, -1):
        source = _LOG_PATH.with_name(f"{_LOG_PATH.name}.{index}")
        target = _LOG_PATH.with_name(f"{_LOG_PATH.name}.{index + 1}")
        try:
            if index == _LOG_BACKUP_COUNT:
                source.unlink(missing_ok=True)
            elif source.exists():
                os.replace(source, target)
        except OSError:
            pass
    try:
        os.replace(_LOG_PATH, _LOG_PATH.with_name(f"{_LOG_PATH.name}.1"))
    except OSError:
        pass


def _log(event: dict) -> None:
    safe_event = _redact_log_value(event)
    safe_event["ts"] = time.time()
    try:
        with _LOG_LOCK:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _rotate_session_log()
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_event, ensure_ascii=False, default=str) + "\n")
            _LOG_PATH.chmod(0o600)
    except Exception:
        pass


@dataclass
class TurnResult:
    text: str
    tool_calls: int = 0
    aborted: bool = False
    cancelled: bool = False
    web_search_results: list[str] | None = None


@_serialized_turn
def run_turn(
    user_input: str,
    messages: list[dict],
    on_step=None,
    attachment_image: str | None = None,
    cancel_event=None,
    stream_tokens: bool = False,
    effort: str | None = None,
) -> TurnResult:
    """`messages` is the persisted, compacted cross-turn history — mutated
    in place with exactly one new compact (user, assistant) pair per call.

    `on_step`, if given, is called synchronously as the graph executes (via
    `.stream(..., stream_mode="updates")` — restored 2026-07-23, user
    request to bring back MAX_VLM's live "what is the agent doing right
    now" status, which the switch to create_react_agent's opaque internal
    loop had dropped versus the old hand-rolled while-loop's per-step hook):
      on_step(event="tool_start", name=str, args=dict)  — model just decided
        to call `name` (fires the moment the "agent" graph node emits an
        AIMessage with tool_calls, before the "tools" node actually runs it)
      on_step(event="tool_done", name=str, ok=bool)      — the "tools" node
        finished executing and returned a ToolMessage

    ``stream_tokens`` is deliberately opt-in.  The HTML UI enables it; CLI
    and programmatic callers keep the original single-response behavior.
    """
    if on_step is None:
        on_step = lambda **_: None

    # Progressive image lifecycle is owned by _serialized_turn(): stale
    # snapshots are drained before this body and all current-turn views are
    # released in its finally block, including graph exceptions/cancellation.
    injected_input = inject_datetime_context(user_input)
    if attachment_image:
        # Browser attachments are an explicit user instruction, not a routing
        # hint for the model to optionally ignore.  Read and queue the image
        # before the graph's first model call so react.dynamic_prompt injects
        # it into that call as an image_url block.
        on_step(event="tool_start", name="read_image", args={"path": attachment_image})
        attachment_result = _read_image_tool.run({"path": attachment_image})
        attachment_ok = not attachment_result.startswith(("[error]", "[blocked]"))
        on_step(event="tool_done", name="read_image", ok=attachment_ok)
        injected_input += f"\n\n[ไฟล์รูปที่ผู้ใช้แนบ — ระบบอ่านภาพแล้ว]\n{attachment_result}"
    turn_guard.reset(user_input)
    local_document_request = search_scope.is_local_document_query(user_input)
    rag_request = search_scope.is_knowledge_base_query(user_input)
    outside_rag_request = search_scope.is_outside_knowledge_base_query(user_input)
    web_request = _web_search_subject(user_input, messages)
    force_web_search = (
        _should_force_web_search(user_input)
        or _is_web_search_followup(user_input)
    ) and not local_document_request and not rag_request and not outside_rag_request
    forced_web_query = _focused_web_queries(web_request)
    code_owned_web_query = isinstance(forced_web_query, list) or _is_web_search_followup(user_input)
    working = _to_lc_messages(messages)
    working.append(HumanMessage(content=injected_input))

    direct_image_path = None if attachment_image else _requested_image_path(user_input)
    if direct_image_path:
        image_args = {"path": direct_image_path}
        on_step(event="tool_start", name="read_image", args=image_args)
        image_result = _read_image_tool.read_image.invoke(image_args)
        image_ok = not image_result.startswith(("[error]", "[blocked]"))
        on_step(event="tool_done", name="read_image", ok=image_ok)
        image_call_id = "direct-image-policy"
        working.extend([
            AIMessage(content="", tool_calls=[{
                "name": "read_image", "args": image_args,
                "id": image_call_id, "type": "tool_call",
            }]),
            ToolMessage(
                content=image_result, tool_call_id=image_call_id, name="read_image",
            ),
        ])

    # Keep the user's request as the model's input and let the agent choose
    # both the first tool and its concise query argument. A code-generated
    # rag_search call would expose the whole natural-language request as the
    # query and would make phrase-specific intent extraction a hidden second
    # router. The system prompt now makes the argument contract explicit.

    # mlx_vlm's Qwen3.5 tool-call parser is reliable on streaming responses;
    # the non-stream endpoint can return legacy <tool_call> XML as plain text
    # and repeat it until max_tokens without producing a structured call. Keep
    # transport streaming enabled for every caller, while stream_tokens still
    # controls whether UI token events are exposed to the caller.
    effort = config.normalize_thinking_effort(effort)
    thinking_budget = config.thinking_budget_for_effort(effort)
    agent = _get_react_agent(streaming=True, thinking_budget=thinking_budget)
    run_config = {"recursion_limit": _RECURSION_LIMIT}
    if stream_tokens:
        run_config["callbacks"] = [_TokenStreamCallback(on_step, cancel_event)]

    try:
        final_msg = None
        web_search_results: list[str] = []
        rag_search_results: list[str] = []
        local_evidence_results: list[str] = []
        if direct_image_path and image_ok:
            local_evidence_results.append(image_result)
        last_tool_name: str | None = (
            "read_image" if direct_image_path and image_ok else None
        )
        last_tool_result: str | None = (
            image_result if direct_image_path and image_ok else None
        )
        first_rag_result: str | None = None
        first_local_search_result: str | None = None
        local_search_results: list[str] = []
        forced_web_result: str | None = None
        write_overwrite_gate_tripped = False
        tool_call_attempts = _ToolCallAttemptState()
        logical_tool_by_call_id: dict[str, str] = {}
        rag_search_called = False
        rag_mcp_unavailable = False
        rag_fallback_bash_called = False
        rag_fallback_scope: str | None = None
        if direct_image_path:
            _tool_call_attempt(tool_call_attempts, "read_image", {"path": direct_image_path})
        # Explicit RAG/Knowledge Base scope is a hard boundary.  Open the
        # deterministic MCP search before the model's first synthesis call and
        # preload its ToolMessage, so the model cannot divert this request to
        # web_search or answer from memory before seeing the local evidence.
        if rag_request:
            rag_query = _rag_search_query(user_input)
            if rag_query:
                rag_args = {
                    "server": "lite-rag",
                    "tool_name": "rag_search",
                    "arguments_json": json.dumps(
                        {"query": rag_query}, ensure_ascii=False,
                    ),
                }
                rag_search_called = True
                on_step(event="tool_start", name="mcp_call_tool", args=rag_args)
                rag_result = _mcp_tool.mcp_call_tool.invoke(rag_args)
                rag_ok = not rag_result.startswith(("[error]", "[blocked]"))
                on_step(event="tool_done", name="mcp_call_tool", ok=rag_ok)
                if rag_ok:
                    rag_search_results.append(rag_result)
                    first_rag_result = rag_result
                    last_tool_name, last_tool_result = "rag_search", rag_result
                    local_evidence_results.append(rag_result)
                    rag_call_id = "policy-rag-search"
                    working.extend([
                        AIMessage(content="", tool_calls=[{
                            "name": "mcp_call_tool", "args": rag_args,
                            "id": rag_call_id, "type": "tool_call",
                        }]),
                        ToolMessage(
                            content=rag_result,
                            tool_call_id=rag_call_id,
                            name="mcp_call_tool",
                        ),
                    ])
                else:
                    rag_mcp_unavailable = _rag_mcp_unavailable(rag_result)
                    if rag_mcp_unavailable:
                        bash_command = _rag_fallback_bash_command(user_input)
                        if bash_command:
                            bash_args = {"command": bash_command}
                            on_step(event="tool_start", name="bash", args=bash_args)
                            bash_result = _bash_tool.bash.invoke(bash_args)
                            bash_ok = not bash_result.startswith(("[error]", "[blocked]"))
                            on_step(event="tool_done", name="bash", ok=bash_ok)
                            rag_fallback_bash_called = True
                            rag_fallback_scope = "knowledge"
                            if bash_ok:
                                first_local_search_result = bash_result
                                local_search_results.append(bash_result)
                                last_tool_name, last_tool_result = "bash", bash_result
                                local_evidence_results.append(bash_result)
                                bash_call_id = "policy-rag-unavailable-bash-search"
                                working.extend([
                                    AIMessage(content="", tool_calls=[{
                                        "name": "bash", "args": bash_args,
                                        "id": bash_call_id, "type": "tool_call",
                                    }]),
                                    ToolMessage(
                                        content=bash_result,
                                        tool_call_id=bash_call_id,
                                        name="bash",
                                    ),
                                ])
        # Explicitly asking to search outside the local Knowledge Base is a
        # hard bash boundary.  Run one deterministic, read-only search over
        # the configured Desktop/Documents roots before the model's first
        # synthesis call, so it cannot divert the request to RAG or web search.
        if outside_rag_request:
            bash_command = _outside_rag_bash_command(user_input)
            if bash_command:
                bash_args = {"command": bash_command}
                on_step(event="tool_start", name="bash", args=bash_args)
                bash_result = _bash_tool.bash.invoke(bash_args)
                bash_ok = not bash_result.startswith(("[error]", "[blocked]"))
                on_step(event="tool_done", name="bash", ok=bash_ok)
                if bash_ok:
                    first_local_search_result = bash_result
                    local_search_results.append(bash_result)
                    last_tool_name, last_tool_result = "bash", bash_result
                    local_evidence_results.append(bash_result)
                    bash_call_id = "policy-outside-rag-bash-search"
                    working.extend([
                        AIMessage(content="", tool_calls=[{
                            "name": "bash", "args": bash_args,
                            "id": bash_call_id, "type": "tool_call",
                        }]),
                        ToolMessage(
                            content=bash_result,
                            tool_call_id=bash_call_id,
                            name="bash",
                        ),
                    ])
        duplicate_tool_call: dict | None = None
        cancelled = False
        for update in agent.stream(
            {"messages": working},
            config=run_config,
            stream_mode="updates",
        ):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            stop_for_duplicate = False
            for node_name, node_output in update.items():
                for m in node_output.get("messages", []):
                    if node_name == "agent":
                        tool_calls = getattr(m, "tool_calls", None) or []
                        for tc in tool_calls:
                            args = dict(tc.get("args", {}))
                            args = _normalize_write_tool_args(tc["name"], args, user_input)
                            args = _normalize_mcp_tool_args(tc["name"], args)
                            args, write_overwrite_gate_tripped = _apply_first_write_overwrite_gate(
                                args, user_input, write_overwrite_gate_tripped,
                            )
                            if tc["name"] == "bash" and isinstance(args.get("command"), str):
                                args["command"] = _bash_tool.normalize_command_for_query(
                                    args["command"], user_input,
                                )
                            tc["args"] = args
                            logical_name = _logical_tool_name(tc["name"], args)
                            if logical_name == "rag_search":
                                rag_search_called = True
                            call_id = str(tc.get("id") or "")
                            if call_id:
                                logical_tool_by_call_id[call_id] = logical_name
                            attempt_count, should_stop_duplicate = _tool_call_attempt(
                                tool_call_attempts, tc["name"], args,
                            )
                            if should_stop_duplicate:
                                duplicate_tool_call = {
                                    "name": tc["name"], "logical_name": logical_name, "args": args,
                                }
                                _log({"event": "duplicate_tool_loop_stopped", **duplicate_tool_call})
                                stop_for_duplicate = True
                                break
                            if attempt_count == 2 and tc["name"] in _HINT_RETRY_TOOLS:
                                _log({
                                    "event": "duplicate_tool_hint_passed_to_wrapper",
                                    "name": tc["name"], "logical_name": logical_name, "args": args,
                                })
                            on_step(event="tool_start", name=tc["name"], args=args)
                        if not tool_calls:
                            final_msg = m
                    elif node_name == "tools":
                        # LangGraph may complete parallel tool calls out of
                        # issue order. ToolMessage already carries the true
                        # name and execution status, so never infer either
                        # from a FIFO queue of prior agent calls.
                        raw_name = getattr(m, "name", None) or "?"
                        call_id = str(getattr(m, "tool_call_id", "") or "")
                        name = logical_tool_by_call_id.get(call_id, raw_name)
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        if name == "web_search" and not content.startswith("[blocked]"):
                            web_search_results.append(content)
                        status = getattr(m, "status", None)
                        ok = status != "error" and not (
                            content.startswith("[error]") or content.startswith("[blocked]")
                        )
                        if name == "write_file" and not ok and not _user_explicitly_allows_overwrite(user_input):
                            write_overwrite_gate_tripped = True
                        on_step(event="tool_done", name=raw_name, ok=ok)
                        if ok:
                            last_tool_name, last_tool_result = name, content
                            if name == "rag_search":
                                rag_search_called = True
                                rag_search_results.append(content)
                                if first_rag_result is None:
                                    first_rag_result = content
                            if name == "local_search":
                                local_search_results.append(content)
                                if first_local_search_result is None:
                                    first_local_search_result = content
                            if name in {"bash", "rag_search", "local_search", "read_file", "read_image"}:
                                local_evidence_results.append(content)
                    if stop_for_duplicate:
                        break
                if stop_for_duplicate:
                    break
            if stop_for_duplicate:
                break

        # Local document requests are a hard routing boundary.  A small model
        # can still choose web_search despite the prompt, especially when the
        # requested filename is Thai.  The web wrapper blocks that call; keep
        # any such response out of the web-evidence path and make one bounded,
        # exact-filename/content-phrase recovery so an obvious local lookup
        # can finish even when Qwen emits no tool call or chooses the KB for a
        # generic user-file request. Explicit KB/project requests stay on RAG;
        # generic requests use the deterministic Desktop/Documents helper.
        if local_document_request and not outside_rag_request:
            web_search_results.clear()
            local_search_query = _local_document_search_query(user_input)
            knowledge_scope = (
                search_scope.is_knowledge_base_query(user_input)
                or search_scope.is_project_scoped_query(user_input)
            )
            if local_search_query and not _PATH_LIKE_QUERY_TOKEN_RE.search(user_input) and not local_search_results:
                # Keep the established RAG-first behavior. A generic local
                # request gets one KB probe for backwards compatibility, then
                # falls through to Desktop/Documents only when that probe is
                # absent or not useful. Explicit KB/project requests stop at
                # the RAG boundary and never broaden their scope.
                if not rag_search_called and not _has_useful_project_rag_evidence(first_rag_result):
                    rag_args = {
                        "server": "lite-rag",
                        "tool_name": "rag_search",
                        "arguments_json": json.dumps(
                            {"query": local_search_query}, ensure_ascii=False,
                        ),
                    }
                    on_step(event="tool_start", name="mcp_call_tool", args=rag_args)
                    rag_result = _mcp_tool.mcp_call_tool.invoke(rag_args)
                    rag_ok = not rag_result.startswith(("[error]", "[blocked]"))
                    on_step(event="tool_done", name="mcp_call_tool", ok=rag_ok)
                    if rag_ok:
                        last_tool_name, last_tool_result = "rag_search", rag_result
                        rag_search_results.append(rag_result)
                        first_rag_result = rag_result
                        local_evidence_results.append(rag_result)
                    else:
                        rag_mcp_unavailable = rag_mcp_unavailable or _rag_mcp_unavailable(rag_result)
                if (
                    rag_mcp_unavailable
                    and not _has_useful_project_rag_evidence(first_rag_result)
                    and not local_search_results
                    and not rag_fallback_bash_called
                ):
                    project_scope = search_scope.is_project_scoped_query(user_input)
                    bash_command = (
                        _rag_fallback_bash_command(user_input, project_scope=True)
                        if project_scope
                        else _outside_rag_bash_command(user_input)
                    )
                    if bash_command:
                        bash_args = {"command": bash_command}
                        on_step(event="tool_start", name="bash", args=bash_args)
                        bash_result = _bash_tool.bash.invoke(bash_args)
                        bash_ok = not bash_result.startswith(("[error]", "[blocked]"))
                        on_step(event="tool_done", name="bash", ok=bash_ok)
                        rag_fallback_bash_called = True
                        rag_fallback_scope = "project" if project_scope else "local"
                        local_search_results.append(bash_result)
                        if bash_ok:
                            last_tool_name, last_tool_result = "bash", bash_result
                            first_local_search_result = bash_result
                            local_evidence_results.append(bash_result)
                if (
                    not knowledge_scope
                    and not rag_mcp_unavailable
                    and not _has_useful_project_rag_evidence(first_rag_result)
                    and not local_search_results
                ):
                    local_args = {"query": local_search_query}
                    on_step(event="tool_start", name="local_search", args=local_args)
                    local_result = _local_search_tool.local_search.invoke(local_args)
                    local_ok = not local_result.startswith(("[error]", "[blocked]"))
                    on_step(event="tool_done", name="local_search", ok=local_ok)
                    local_search_results.append(local_result)
                    if local_ok:
                        last_tool_name, last_tool_result = "local_search", local_result
                        first_local_search_result = local_result
                        local_evidence_results.append(local_result)

        if rag_request:
            web_search_results.clear()
        if outside_rag_request:
            web_search_results.clear()
            rag_search_results.clear()

        # A current/web request must not silently become a model-only answer.
        # If the small model skipped web_search entirely, issue one guarded
        # call using the original request; tools/web_search owns query
        # normalization and source filtering. The final answer is grounded in
        # that result rather than in the model's unsupported prior prose.
        if (
            not cancelled
            and force_web_search
            and (code_owned_web_query or not _usable_web_results(web_search_results))
        ):
            on_step(event="tool_start", name="web_search", args={"query": forced_web_query})
            web_result = _web_search_tool.web_search.invoke({"query": forced_web_query})
            web_ok = not web_result.startswith(("[error]", "[blocked]"))
            on_step(event="tool_done", name="web_search", ok=web_ok)
            if web_ok:
                web_search_results.append(web_result)
                last_tool_name, last_tool_result = "web_search", web_result
                forced_web_result = web_result

        # A search engine can return only [needs_retry] messages
        # even after the model has spent its retry budget.  Make one bounded,
        # code-owned refinement (official-source variant when available) so a
        # transient model routing failure does not turn a current-information
        # request into a false "no sources" answer.  Local requests are
        # excluded above and are handled by the RAG recovery path instead.
        if (
            not cancelled
            and not local_document_request
            and force_web_search
            and web_search_results
            and not _source_urls(web_search_results)
        ):
            recovery_query = _web_recovery_query(web_request)
            if recovery_query:
                on_step(event="tool_start", name="web_search", args={"query": recovery_query})
                recovery_result = _web_search_tool.web_search.invoke({"query": recovery_query})
                recovery_ok = not recovery_result.startswith(("[error]", "[blocked]"))
                on_step(event="tool_done", name="web_search", ok=recovery_ok)
                if recovery_ok and _source_urls([recovery_result]):
                    web_search_results = [recovery_result]
                    last_tool_name, last_tool_result = "web_search", recovery_result
                    forced_web_result = recovery_result

        duplicate_recovered = bool(
            duplicate_tool_call is not None
            and last_tool_name == (
                duplicate_tool_call.get("logical_name") or duplicate_tool_call.get("name")
            )
            and last_tool_name in {"bash", "rag_search", "local_search", "web_search"}
            and last_tool_result
        )
        if duplicate_tool_call is not None:
            if duplicate_recovered:
                # The tool already succeeded; a small model asking for the
                # identical call again is a completion failure, not a failed
                # user operation. Stop the loop and surface the proven result.
                final_text = f"พบผลลัพธ์จากการค้นหาดังนี้ครับ:\n\n{_strip_tool_reminder(last_tool_result or '')}"
                if duplicate_tool_call.get("name") == "bash":
                    final_text += (
                        "\n\n[hint] bash command เดิมถูกบล็อกแล้ว; "
                        "ถ้าจำเป็นต้องลองใหม่ ให้เปลี่ยน command หรือ scope อย่างชัดเจน"
                    )
                _log({"event": "duplicate_tool_completion_recovered", **duplicate_tool_call})
            else:
                duplicate_name = duplicate_tool_call.get("logical_name") or duplicate_tool_call.get("name")
                retry_subject = "command" if duplicate_name == "bash" else "query" if duplicate_name in {"web_search", "rag_search"} else "argument"
                final_text = (
                    "หยุดการทำงานเพื่อป้องกัน tool loop: โมเดลเรียก tool เดิมด้วย "
                    f"{retry_subject} เดิมซ้ำติดต่อกัน จึงถูกบล็อก "
                    f"กรุณาเปลี่ยน {retry_subject} จริงก่อนลองใหม่"
                )
        else:
            final_text = "" if cancelled else (
                final_msg.content if final_msg is not None and isinstance(final_msg.content, str)
                else (str(final_msg.content) if final_msg is not None else "")
            )

        # Long histories can make the small model skip an otherwise-bound
        # write_file call and answer as if the file had been saved.  Retry the
        # ReAct graph once only when its completed first pass emitted no actual
        # write_file wrapper call.  The retry instruction is working-context
        # only; compacted cross-turn history keeps the original user request.
        if (
            not cancelled
            and duplicate_tool_call is None
            and _write_action_required(user_input)
            and not any(line.startswith("- write_file(") for line in turn_guard.get_tool_log())
        ):
            _log({"event": "write_action_retry", "query": user_input[:300]})
            first_final_msg = final_msg
            retry_emitted_write_call = False
            retry_working = list(working)
            if first_final_msg is not None:
                retry_working.append(first_final_msg)
            retry_working.append(HumanMessage(content=_WRITE_RETRY_PROMPT.format(query=user_input)))
            final_msg = None
            if stream_tokens:
                on_step(event="discard_stream")
            for update in agent.stream(
                {"messages": retry_working},
                config=run_config,
                stream_mode="updates",
            ):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                for node_name, node_output in update.items():
                    for m in node_output.get("messages", []):
                        if node_name == "agent":
                            tool_calls = getattr(m, "tool_calls", None) or []
                            for tc in tool_calls:
                                retry_args = _normalize_write_tool_args(
                                    tc["name"], dict(tc.get("args", {})), user_input,
                                )
                                retry_args, write_overwrite_gate_tripped = _apply_first_write_overwrite_gate(
                                    retry_args, user_input, write_overwrite_gate_tripped,
                                )
                                tc["args"] = retry_args
                                retry_logical_name = _logical_tool_name(tc["name"], retry_args)
                                retry_call_id = str(tc.get("id") or "")
                                if retry_call_id:
                                    logical_tool_by_call_id[retry_call_id] = retry_logical_name
                                if tc["name"] == "write_file":
                                    retry_emitted_write_call = True
                                on_step(event="tool_start", name=tc["name"], args=retry_args)
                            if not tool_calls:
                                final_msg = m
                        elif node_name == "tools":
                            raw_name = getattr(m, "name", None) or "?"
                            call_id = str(getattr(m, "tool_call_id", "") or "")
                            name = logical_tool_by_call_id.get(call_id, raw_name)
                            content = m.content if isinstance(m.content, str) else str(m.content)
                            if name == "web_search" and not content.startswith("[blocked]"):
                                web_search_results.append(content)
                            status = getattr(m, "status", None)
                            ok = status != "error" and not (
                                content.startswith("[error]") or content.startswith("[blocked]")
                            )
                            on_step(event="tool_done", name=raw_name, ok=ok)
                            if ok:
                                last_tool_name, last_tool_result = name, content
                                if name == "rag_search" and first_rag_result is None:
                                    first_rag_result = content
                                if name == "rag_search":
                                    rag_search_results.append(content)
                                if name == "local_search":
                                    local_search_results.append(content)
                                    if first_local_search_result is None:
                                        first_local_search_result = content
                                if name in {"bash", "rag_search", "local_search", "read_file", "read_image"}:
                                    local_evidence_results.append(content)
            if not retry_emitted_write_call:
                # The retry produced no actual write_file call, so the first
                # answer remains the only meaningful fallback for the caller.
                _log({"event": "write_action_retry_failed", "query": user_input[:300]})
                final_msg = first_final_msg
            final_text = "" if cancelled else (
                final_msg.content if final_msg is not None and isinstance(final_msg.content, str)
                else (str(final_msg.content) if final_msg is not None else "")
            )
        aborted = cancelled or (duplicate_tool_call is not None and not duplicate_recovered)
    except _TurnCancelled:
        final_text = ""
        aborted = True
        cancelled = True
    except GraphRecursionError:
        final_text = "ทำงานครบจำนวนขั้นตอนสูงสุดแล้ว หยุดและสรุปเท่าที่ทำได้"
        aborted = True
        cancelled = False
        _log({"event": "recursion_limit_hit", "tool_calls": turn_guard.tool_call_count()})
    except Exception as e:
        # Defense in depth: server down/timeout/any uncaught error must not
        # crash the whole turn / CLI process — see ui_cli.py's own top-level
        # try/except for the second layer of this same protection.
        final_text = f"[error] เกิดข้อผิดพลาดระหว่างประมวลผล: {e}"
        aborted = True
        cancelled = False
        _log({"event": "invoke_crash", "error": str(e)})

    # Exception paths do not create a completed tools-node update, but keep
    # the result shape stable for the terminal UI and programmatic callers.
    web_search_results = locals().get("web_search_results", [])
    rag_search_results = locals().get("rag_search_results", [])
    last_tool_name = locals().get("last_tool_name")
    last_tool_result = locals().get("last_tool_result")
    tool_log = turn_guard.get_tool_log()
    tool_calls = turn_guard.tool_call_count()
    if not cancelled:
        # A 2B model occasionally never resolves a real tool call and instead
        # leaks its own tool-call XML as plain text, sometimes spiraling into
        # runaway repetition with fabricated content -- catch this before any
        # other check, since the text is garbage regardless of tool_calls.
        if has_leaked_tool_syntax(final_text) or is_degenerate_repetition(final_text):
            _log({"event": "degenerate_output_caught", "text": final_text[:500], "tool_calls": tool_calls})
            # Live-observed 2026-07-25/26 (/walk T02, BUG-004): this model is
            # specifically prone to degenerating while trying to prose-
            # synthesize a multi-file bash/rag_search list -- when exactly one
            # such call succeeded this turn, present its own already-correct
            # raw result directly instead of a content-free apology (same
            # "100% determinism -> code, not model synthesis" principle as
            # _append_source_citations).
            if tool_calls == 1 and last_tool_name in ("bash", "rag_search", "local_search") and last_tool_result:
                final_text = f"พบผลลัพธ์จากการค้นหาดังนี้ครับ:\n\n{_strip_tool_reminder(last_tool_result)}"
            else:
                final_text = "ขออภัยครับ การประมวลผลคำตอบไม่สำเร็จ กรุณาถามคำถามเดิมอีกครั้ง"
            aborted = True

        # A 2B model occasionally gives up with a canned English refusal
        # instead of following its own instruction to summarize in Thai --
        # replace it deterministically rather than showing it to the user.
        elif tool_calls and is_boilerplate_refusal(final_text):
            _log({"event": "boilerplate_refusal_caught", "text": final_text, "tools": tool_log})
            web_fallback = _best_web_evidence(web_search_results)
            if force_web_search and web_fallback:
                # Do not surface the last [needs_retry] result after a useful
                # source already survived the web quality gates.
                web_search_results = [web_fallback]
                final_text = (
                    "ผลการค้นหาจากเว็บ:\n\n"
                    f"{_strip_tool_reminder(web_fallback)}"
                )
                aborted = False
            else:
                final_text = _fallback_summary(tool_log, last_tool_name, last_tool_result)

        # The model may produce a [needs_retry] sentence after a useful web
        # result, even when it did not trigger the English-refusal guard above.
        web_fallback = _best_web_evidence(web_search_results)
        if (
            force_web_search
            and web_fallback
            and final_text.lstrip().startswith("[needs_retry]")
        ):
            web_search_results = [web_fallback]
            final_text = f"ผลการค้นหาจากเว็บ:\n\n{_strip_tool_reminder(web_fallback)}"
            aborted = False

        # A preview of a large find/grep result is never the complete list.
        # Qwen2B live-observed the first 19 visible paths and claimed there
        # were exactly 19 although the spill contained 3,894. Code owns the
        # exact line count and full-output pointer, so use those directly.
        if last_tool_name == "bash" and last_tool_result:
            full_output = _FULL_TOOL_OUTPUT_RE.search(last_tool_result)
            if full_output:
                line_count, saved_path = full_output.groups()
                final_text = (
                    f"พบผลลัพธ์ทั้งหมด {line_count} บรรทัด รายการเต็มถูกบันทึกไว้ที่:\n"
                    f"{saved_path.strip()}\n\n"
                    "คำตอบนี้ไม่แสดง preview เป็นรายการทั้งหมด เพื่อป้องกันการนับหรือตัดรายการผิด"
                )

        source_urls = _source_urls(web_search_results)
        if web_search_results and not source_urls:
            # No code-vetted evidence means the model must not fill the gap
            # from memory or from its own failed retry query.
            final_text = (
                "ไม่พบแหล่งข้อมูลบนเว็บที่เกี่ยวข้องและตรวจสอบได้เพียงพอในรอบนี้ "
                "จึงไม่สรุปข้อเท็จจริงหรือแนบลิงก์ที่อาจไม่ตรงคำถาม กรุณาลองระบุหัวข้อ "
                "พื้นที่ หรือช่วงเวลาให้เจาะจงขึ้น"
            )
            aborted = True
        else:
            unsupported_percentages = _unsupported_web_percentages(final_text, web_search_results)
            if source_urls and unsupported_percentages:
                _log({
                    "event": "unsupported_web_percentage_caught",
                    "claims": unsupported_percentages,
                })
                final_text = (
                    "พบแหล่งข้อมูลที่เกี่ยวข้อง แต่ข้อความจากแหล่งค้นหาในรอบนี้ยังไม่ยืนยันตัวเลขที่แน่นอน "
                    "จึงไม่แสดงค่าที่โมเดลอาจคาดเดา กรุณาตรวจค่าล่าสุดจากลิงก์ต้นฉบับด้านล่าง"
                )
                aborted = True
            final_text = _append_source_citations(final_text, web_search_results)

        if not web_search_results:
            # A successful project content fallback is already a complete,
            # grounded locator result. If the small model loops while trying
            # to summarize it, surface that result instead of the loop guard
            # message; this is the same deterministic completion principle as
            # the large-output bash fallback below.
            local_result = first_local_search_result or first_rag_result
            if (
                (
                    local_document_request
                    or rag_request
                    or outside_rag_request
                    or search_scope.is_project_scoped_query()
                )
                and local_result
                and _has_useful_project_rag_evidence(local_result)
                and (
                    aborted
                    or (
                        local_result.startswith("[ค้นจากเนื้อหาในโปรเจกต์ปัจจุบัน:")
                        and _is_project_no_result_answer(final_text)
                    )
                    or (local_document_request and _is_project_no_result_answer(final_text))
                    or (local_document_request and not final_text.strip())
                    or (rag_request and _is_project_no_result_answer(final_text))
                    or (rag_request and not final_text.strip())
                    or (outside_rag_request and _is_project_no_result_answer(final_text))
                    or (outside_rag_request and not final_text.strip())
                )
            ):
                if outside_rag_request:
                    heading = "ผลการค้นหานอก Knowledge Base ด้วย bash:"
                elif rag_fallback_scope == "knowledge":
                    heading = "ผลการค้นหาใน Knowledge Base ด้วย bash:"
                elif rag_fallback_scope == "project":
                    heading = "ผลการค้นหาในโปรเจกต์ปัจจุบันด้วย bash:"
                elif rag_fallback_scope == "local":
                    heading = "ผลการค้นหาใน Desktop/Documents ด้วย bash:"
                else:
                    heading = (
                        "ผลการค้นหาใน Desktop/Documents:"
                        if first_local_search_result
                        else (
                            "ผลการค้นหาใน Knowledge Base:"
                            if rag_request
                            else "ผลการค้นหาในโปรเจกต์ปัจจุบัน:"
                        )
                    )
                final_text = f"{heading}\n\n{_strip_tool_reminder(local_result)}"
                aborted = False
            final_text = _ground_project_search_answer(
                final_text, last_tool_name, last_tool_result,
            )
            final_text = _ground_local_fact_answer(
                final_text, local_result, user_input,
            )
            final_text = _restore_known_local_paths(final_text, local_result)
            final_text = _strip_unsupported_local_numbers(
                final_text, locals().get("local_evidence_results", []), user_input,
            )
        elif forced_web_result and _source_urls([forced_web_result]):
            final_text = f"ผลการค้นหาจากเว็บ:\n\n{_strip_tool_reminder(forced_web_result)}"
            aborted = False
            if direct_image_path and image_ok:
                final_text = _ground_image_text_answer(final_text, image_result, user_input)

        # Like web citations, Knowledge Base provenance is code-owned.  Only
        # attach RAG sources on a local/project turn; a mistaken RAG call in a
        # current-information turn must not dilute the web-source contract.
        if (
            rag_search_results
            and not _write_action_required(user_input)
            and (local_document_request or not force_web_search)
        ):
            final_text = _append_rag_source_citations(final_text, rag_search_results)

        # Defense in depth: a tool or a vision-model response must not place a
        # credential-shaped string into the UI, history, or session log.
        denied_user_command = _bash_tool._is_denied(user_input)
        if denied_user_command:
            final_text = f"[blocked] คำสั่งนี้ถูกบล็อกเพื่อความปลอดภัย: {denied_user_command}"
            aborted = True
        elif _query_requests_sensitive_path(user_input):
            final_text = "[blocked] ไม่อนุญาตให้อ่านไฟล์หรือโฟลเดอร์ที่อาจมีรหัสผ่าน, token, key หรือ session"
            aborted = True
        elif contains_secret_content(final_text):
            final_text = "[blocked] คำตอบมีข้อมูลที่มีรูปแบบคล้าย credential หรือ token จึงไม่แสดงผล"
            aborted = True
    # Keep a private, structured turn audit.  The older log retained only a
    # count, which made a failed tool route impossible to diagnose later.
    _log({
        "event": "final", "text": final_text, "tool_calls": tool_calls,
        "tools": tool_log, "aborted": aborted, "cancelled": cancelled,
    })

    # A cancelled turn produced no answer worth remembering across turns.
    if not cancelled:
        messages.extend(_compact_turn(injected_input, tool_log, final_text))
        trim_history(messages)

    return TurnResult(text=final_text, tool_calls=tool_calls, aborted=aborted, cancelled=cancelled,
                      web_search_results=web_search_results)
