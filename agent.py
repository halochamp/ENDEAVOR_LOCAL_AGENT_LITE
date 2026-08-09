# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
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
  2. Per-turn guardrail reset (helpers/turn_guard.reset()) — the actual
     repeat-call/quota checks now live inside each tool wrapper
     (tools/*.py), not here; this just resets that shared state before each
     turn and reads back its tool-call log afterward.
  3. A step budget via LangGraph's own recursion_limit config, and a crash
     safety net around .invoke() (server down, timeout, GraphRecursionError).
"""
from __future__ import annotations
import json
import functools
import re
import shlex
import threading
import time
from dataclasses import dataclass
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
from tools import rag_search as _rag_search_tool
from tools import read_image as _read_image_tool
from tools import web_search as _web_search_tool

_LOG_PATH = config.LOG_DIR / "session.jsonl"

MAX_HISTORY_TURNS = 5
MAX_HISTORY_CHARS = 20_000

# Each ReAct round is 2 LangGraph steps (agent -> tools); +2 for the final
# answer step. Mirrors the old hand-rolled StepBudget's MAX_TOOL_STEPS.
_RECURSION_LIMIT = config.MAX_TOOL_STEPS * 2 + 2

_react_agents: dict[bool, object] = {}
_TURN_LOCK = threading.RLock()


def _serialized_turn(fn):
    """Keep process-global tool guard and pending-image state turn-scoped."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _TURN_LOCK:
            user_input = args[0] if args else kwargs.get("user_input", "")
            token = search_scope.set_request_query(user_input)
            try:
                return fn(*args, **kwargs)
            finally:
                search_scope.reset_request_query(token)
    return wrapped


def _get_react_agent(streaming: bool = False):
    """Return the cached graph without overriding the global thinking mode.

    Tool selection, argument construction, and interpreting a tool result all
    benefit from the same bounded reasoning budget as a direct answer.  Keep
    the only graph split that is actually required: streaming versus regular
    transport.  ``build_llm`` owns the single thinking policy for both.
    """
    if streaming not in _react_agents:
        _react_agents[streaming] = build_react_agent(streaming=streaming)
    return _react_agents[streaming]


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
    r"(?:ค้นหา(?:ว่า)?|ค้นข้อมูล|หาข้อมูล|แนบลิงก์|แหล่งที่มา|ล่าสุด|ข่าว|\bsearch\b|\blook\s+up\b|\blatest\b|\bsources?\b)",
    re.IGNORECASE,
)
_LOCAL_FILE_REQUEST_RE = re.compile(
    r"(?:ไฟล์|เอกสาร|ในเครื่อง|บนเครื่อง|โปรเจกต์|โฟลเดอร์|path|/Users/|/Volumes/|\bfile\b|\bdocument\b)",
    re.IGNORECASE,
)


def _should_force_web_search(query: str) -> bool:
    """Catch a missed web route without affecting local file requests."""
    return bool(_WEB_REQUEST_RE.search(query or "")) and not _LOCAL_FILE_REQUEST_RE.search(query or "")


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


def _ground_project_search_answer(final_text: str, last_tool_name: str | None, last_tool_result: str | None) -> str:
    """Do not display a sibling-project path hallucinated after a local search."""
    if last_tool_name not in {"rag_search", "bash"} or not last_tool_result:
        return final_text
    if not _has_out_of_scope_path(final_text):
        return final_text
    result = _strip_tool_reminder(last_tool_result)
    if _has_out_of_scope_path(result):
        return "ไม่พบผลลัพธ์ที่อยู่ในโปรเจกต์ปัจจุบันจากคำค้นนี้ครับ"
    return f"ผลการค้นหาในโปรเจกต์ปัจจุบัน:\n\n{result}"


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


def _fallback_summary(tool_log: list[str]) -> str:
    """Deterministic Thai stand-in for a degenerate boilerplate final answer
    (see helpers.guardrails.is_boilerplate_refusal) -- built from the tool
    log's own success/fail record rather than trusting the model to retry
    the summary correctly a second time."""
    if not tool_log:
        return "ขออภัย ไม่สามารถสรุปคำตอบได้ในตอนนี้ กรุณาถามใหม่อีกครั้ง"
    if all(line.endswith("ล้มเหลว") for line in tool_log):
        return "ลองค้นหา/ตรวจสอบแล้วแต่ไม่พบข้อมูลที่ต้องการ กรุณาระบุคำค้นหรือชื่อไฟล์ให้ชัดเจนขึ้น"
    return "ได้ข้อมูลจากเครื่องมือค้นหาแล้ว แต่สรุปคำตอบไม่สำเร็จ กรุณาถามอีกครั้งหรือระบุคำถามให้ชัดเจนขึ้น"


def _log(event: dict) -> None:
    event["ts"] = time.time()
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
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

    # A queued image belongs only to the graph turn that produced it.  If a
    # prior graph crashed between its tools and agent nodes, never attach that
    # old image to this user's new question.
    _read_image_tool.discard_pending_images()
    _read_image_tool.purge_orphaned_pending_images()
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

    agent = _get_react_agent(streaming=stream_tokens)
    run_config = {"recursion_limit": _RECURSION_LIMIT}
    if stream_tokens:
        run_config["callbacks"] = [_TokenStreamCallback(on_step, cancel_event)]

    try:
        final_msg = None
        web_search_results: list[str] = []
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
        first_rag_query: str | None = None
        bash_seen = False
        forced_whole_machine_result: str | None = None
        forced_web_result: str | None = None
        write_overwrite_gate_tripped = False
        issued_tool_calls: set[tuple[str, str]] = set()
        if direct_image_path:
            issued_tool_calls.add((
                "read_image",
                json.dumps({"path": direct_image_path}, sort_keys=True, ensure_ascii=False),
            ))
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
                            args, write_overwrite_gate_tripped = _apply_first_write_overwrite_gate(
                                args, user_input, write_overwrite_gate_tripped,
                            )
                            if tc["name"] == "rag_search" and first_rag_query is None:
                                first_rag_query = str(args.get("query") or "").strip()
                            if tc["name"] == "bash" and isinstance(args.get("command"), str):
                                args["command"] = _bash_tool.normalize_command_for_query(
                                    args["command"], user_input,
                                )
                            tc["args"] = args
                            key = (
                                tc["name"],
                                json.dumps(args, sort_keys=True, ensure_ascii=False, default=str),
                            )
                            if key in issued_tool_calls:
                                duplicate_tool_call = {"name": tc["name"], "args": args}
                                _log({"event": "duplicate_tool_loop_stopped", **duplicate_tool_call})
                                stop_for_duplicate = True
                                break
                            issued_tool_calls.add(key)
                            on_step(event="tool_start", name=tc["name"], args=args)
                        if not tool_calls:
                            final_msg = m
                    elif node_name == "tools":
                        # LangGraph may complete parallel tool calls out of
                        # issue order. ToolMessage already carries the true
                        # name and execution status, so never infer either
                        # from a FIFO queue of prior agent calls.
                        name = getattr(m, "name", None) or "?"
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        if name == "bash":
                            bash_seen = True
                        if name == "web_search":
                            web_search_results.append(content)
                        status = getattr(m, "status", None)
                        ok = status != "error" and not (
                            content.startswith("[error]") or content.startswith("[blocked]")
                        )
                        if name == "write_file" and not ok and not _user_explicitly_allows_overwrite(user_input):
                            write_overwrite_gate_tripped = True
                        on_step(event="tool_done", name=name, ok=ok)
                        if ok:
                            last_tool_name, last_tool_result = name, content
                            if name == "rag_search" and first_rag_result is None:
                                first_rag_result = content
                            if name in {"bash", "rag_search", "read_file", "read_image"}:
                                local_evidence_results.append(content)
                    if stop_for_duplicate:
                        break
                if stop_for_duplicate:
                    break
            if stop_for_duplicate:
                break

        # A whole-machine request needs a completeness pass even when the
        # global RAG index already returned a few hits. The model selected the
        # concise first query; the code only supplies the required Spotlight
        # backend and records it like a normal tool call. This prevents a
        # partial index result from being presented as the whole machine.
        if (
            not cancelled
            and search_scope.is_whole_machine_query(user_input)
            and first_rag_query
            and not bash_seen
        ):
            whole_machine_command = f"mdfind {shlex.quote(first_rag_query)}"
            on_step(event="tool_start", name="bash", args={"command": whole_machine_command})
            whole_machine_result = _bash_tool.bash.invoke({"command": whole_machine_command})
            whole_machine_ok = not whole_machine_result.startswith(("[error]", "[blocked]"))
            on_step(event="tool_done", name="bash", ok=whole_machine_ok)
            if whole_machine_ok:
                last_tool_name, last_tool_result = "bash", whole_machine_result
                local_evidence_results.append(whole_machine_result)
                forced_whole_machine_result = whole_machine_result

        # A current/web request must not silently become a model-only answer.
        # If the small model skipped web_search entirely, issue one guarded
        # call using the original request; tools/web_search owns query
        # normalization and source filtering. The final answer is grounded in
        # that result rather than in the model's unsupported prior prose.
        if not cancelled and _should_force_web_search(user_input) and not web_search_results:
            on_step(event="tool_start", name="web_search", args={"query": user_input})
            web_result = _web_search_tool.web_search.invoke({"query": user_input})
            web_ok = not web_result.startswith(("[error]", "[blocked]"))
            on_step(event="tool_done", name="web_search", ok=web_ok)
            if web_ok:
                web_search_results.append(web_result)
                last_tool_name, last_tool_result = "web_search", web_result
                forced_web_result = web_result

        duplicate_recovered = bool(
            duplicate_tool_call is not None
            and last_tool_name == duplicate_tool_call.get("name")
            and last_tool_name in {"bash", "rag_search"}
            and last_tool_result
        )
        if duplicate_tool_call is not None:
            if duplicate_recovered:
                # The tool already succeeded; a small model asking for the
                # identical call again is a completion failure, not a failed
                # user operation. Stop the loop and surface the proven result.
                final_text = f"พบผลลัพธ์จากการค้นหาดังนี้ครับ:\n\n{_strip_tool_reminder(last_tool_result or '')}"
                _log({"event": "duplicate_tool_completion_recovered", **duplicate_tool_call})
            else:
                final_text = (
                    "หยุดการทำงานเพื่อป้องกัน tool loop: โมเดลพยายามเรียกคำสั่งเดิมซ้ำโดยไม่เปลี่ยนวิธี "
                    "กรุณาปรับคำค้นหรือขอบเขตแล้วลองใหม่"
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
                                if tc["name"] == "write_file":
                                    retry_emitted_write_call = True
                                on_step(event="tool_start", name=tc["name"], args=retry_args)
                            if not tool_calls:
                                final_msg = m
                        elif node_name == "tools":
                            name = getattr(m, "name", None) or "?"
                            content = m.content if isinstance(m.content, str) else str(m.content)
                            if name == "web_search":
                                web_search_results.append(content)
                            status = getattr(m, "status", None)
                            ok = status != "error" and not (
                                content.startswith("[error]") or content.startswith("[blocked]")
                            )
                            on_step(event="tool_done", name=name, ok=ok)
                            if ok:
                                last_tool_name, last_tool_result = name, content
                                if name in {"bash", "rag_search", "read_file", "read_image"}:
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
            if tool_calls == 1 and last_tool_name in ("bash", "rag_search") and last_tool_result:
                final_text = f"พบผลลัพธ์จากการค้นหาดังนี้ครับ:\n\n{_strip_tool_reminder(last_tool_result)}"
            else:
                final_text = "ขออภัยครับ การประมวลผลคำตอบไม่สำเร็จ กรุณาถามคำถามเดิมอีกครั้ง"
            aborted = True

        # A 2B model occasionally gives up with a canned English refusal
        # instead of following its own instruction to summarize in Thai --
        # replace it deterministically rather than showing it to the user.
        elif tool_calls and is_boilerplate_refusal(final_text):
            _log({"event": "boilerplate_refusal_caught", "text": final_text, "tools": tool_log})
            final_text = _fallback_summary(tool_log)

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
            if forced_whole_machine_result:
                final_text = f"ผลการค้นหาทั้งเครื่อง:\n\n{_strip_tool_reminder(forced_whole_machine_result)}"
                aborted = False
            # A successful project content fallback is already a complete,
            # grounded locator result. If the small model loops while trying
            # to summarize it, surface that result instead of the loop guard
            # message; this is the same deterministic completion principle as
            # the large-output bash fallback below.
            if (
                search_scope.is_project_scoped_query()
                and first_rag_result
                and _has_useful_project_rag_evidence(first_rag_result)
                and (
                    aborted
                    or (
                        first_rag_result.startswith("[ค้นจากเนื้อหาในโปรเจกต์ปัจจุบัน:")
                        and _is_project_no_result_answer(final_text)
                    )
                )
            ):
                final_text = f"ผลการค้นหาในโปรเจกต์ปัจจุบัน:\n\n{_strip_tool_reminder(first_rag_result)}"
                aborted = False
            final_text = _ground_project_search_answer(
                final_text, last_tool_name, last_tool_result,
            )
            final_text = _restore_known_local_paths(final_text, first_rag_result)
            final_text = _strip_unsupported_local_numbers(
                final_text, locals().get("local_evidence_results", []), user_input,
            )
        elif forced_web_result:
            final_text = f"ผลการค้นหาจากเว็บ:\n\n{_strip_tool_reminder(forced_web_result)}"
            aborted = False
            if direct_image_path and image_ok:
                final_text = _ground_image_text_answer(final_text, image_result, user_input)

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

    # Normal direct-VLM injection consumes snapshots itself.  This is the
    # exception/finally layer for a graph that stopped before its next model
    # call or had no HumanMessage to attach to.
    _read_image_tool.discard_pending_images()
    return TurnResult(text=final_text, tool_calls=tool_calls, aborted=aborted, cancelled=cancelled,
                      web_search_results=web_search_results)
