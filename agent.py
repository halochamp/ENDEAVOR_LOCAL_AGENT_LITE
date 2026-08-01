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
import threading
import time
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import config
from helpers import turn_guard
from helpers.date_inject import inject_datetime_context
from helpers.guardrails import has_leaked_tool_syntax, is_boilerplate_refusal, is_degenerate_repetition
from helpers.sensitive_paths import contains_secret_content
from react import build_react_agent
from tools import bash as _bash_tool
from tools import rag_search as _rag_search_tool
from tools import read_image as _read_image_tool

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
            return fn(*args, **kwargs)
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

_EXPLICIT_BASH_RE = re.compile(r"(?:ใช้|รัน|ด้วย|ผ่าน)\s*(?:คำสั่ง\s*)?(?:bash|shell)|\b(?:bash|shell)\s+(?:first|ก่อน)", re.IGNORECASE)
_EXPLICIT_WEB_RE = re.compile(r"(?:เว็บ|เว็บไซต์|อินเทอร์เน็ต|ออนไลน์|\bweb\b|\binternet\b|\bwebsite\b|\burl\b|ข่าว|\bnews\b)", re.IGNORECASE)
_LOCAL_DOCUMENT_CONTEXT_RE = re.compile(
    r"(?:ไฟล์|เอกสาร|โฟลเดอร์|ในเครื่อง|ทั้งเครื่อง|ทั้งคอม|"
    r"\b(?:files?|documents?|folders?|downloads?|desktop|icloud|pdf|word|excel|powerpoint|spreadsheet)\b|"
    r"\.(?:pdf|docx?|xlsx?|pptx?|txt|md|csv)\b)",
    re.IGNORECASE,
)
_LOCAL_DOCUMENT_ACTION_RE = re.compile(
    r"(?:หา|ค้น|อยู่ไหน|ที่ไหน|กี่|นับ|จำนวน|รายการ|ลิสต์|อ่าน|เปิด|ดู|สรุป|"
    r"\b(?:find|search|locate|where|list|count|read|open|show|summari[sz]e)\b)",
    re.IGNORECASE,
)
_KNOWN_LOCAL_FILE_PATH_RE = re.compile(
    r"/(?:Users|Volumes|private|tmp)/[^\n]+\.(?:pdf|docx?|xlsx?|pptx?|txt|md|csv)\b",
    re.IGNORECASE,
)


def _write_action_required(query: str) -> bool:
    return bool(_WRITE_ACTION_RE.search(query))


def _requires_rag_first(query: str) -> bool:
    """Recognize local document discovery without memorizing eval phrases.

    This only establishes the first cheap local lookup. The model still
    decides whether the result is sufficient or a complete read-only bash
    listing/count is required. Explicit bash, explicit web, writes, and reads
    of an already known absolute file path keep their direct routes.
    """
    return bool(
        query
        and not _EXPLICIT_BASH_RE.search(query)
        and not _EXPLICIT_WEB_RE.search(query)
        and not _write_action_required(query)
        and not _KNOWN_LOCAL_FILE_PATH_RE.search(query)
        and _LOCAL_DOCUMENT_CONTEXT_RE.search(query)
        and _LOCAL_DOCUMENT_ACTION_RE.search(query)
    )


def _strip_tool_reminder(text: str) -> str:
    """tools/bash.py and tools/rag_search.py append a model-facing first-
    call-this-turn reminder (routing/retry guidance) to their own return
    value -- meant for the model's next decision, not for display when that
    raw result is surfaced directly to the user (see BUG-004 fallback)."""
    return _TOOL_REMINDER_RE.sub("", text)


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

    # Model-free first step for local document discovery. Repeated live runs
    # showed that a 2B model can route the same local PDF request to the web
    # despite correct examples. Starting from a valid tool-call/result pair
    # enforces RAG-first, saves one model round, and leaves the next decision
    # (answer, read_file, or a complete bash list/count) to ReAct.
    rag_prefill_result: str | None = None
    rag_prefill_ok = False
    rag_prefill_args: dict[str, str] | None = None
    if _requires_rag_first(user_input):
        rag_prefill_args = {"query": user_input}
        on_step(event="tool_start", name="rag_search", args=rag_prefill_args)
        rag_prefill_result = _rag_search_tool.rag_search.invoke(rag_prefill_args)
        rag_prefill_ok = not rag_prefill_result.startswith(("[error]", "[blocked]"))
        on_step(event="tool_done", name="rag_search", ok=rag_prefill_ok)
        call_id = "rag-first-policy"
        working.extend([
            AIMessage(content="", tool_calls=[{
                "name": "rag_search", "args": rag_prefill_args,
                "id": call_id, "type": "tool_call",
            }]),
            ToolMessage(content=rag_prefill_result, tool_call_id=call_id, name="rag_search"),
        ])

    agent = _get_react_agent(streaming=stream_tokens)
    run_config = {"recursion_limit": _RECURSION_LIMIT}
    if stream_tokens:
        run_config["callbacks"] = [_TokenStreamCallback(on_step, cancel_event)]

    try:
        final_msg = None
        web_search_results: list[str] = []
        last_tool_name: str | None = "rag_search" if rag_prefill_ok else None
        last_tool_result: str | None = rag_prefill_result if rag_prefill_ok else None
        issued_tool_calls: set[tuple[str, str]] = set()
        if rag_prefill_args is not None:
            issued_tool_calls.add((
                "rag_search",
                json.dumps(rag_prefill_args, sort_keys=True, ensure_ascii=False, default=str),
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
                            on_step(event="tool_start", name=tc["name"], args=tc.get("args", {}))
                        if not tool_calls:
                            final_msg = m
                    elif node_name == "tools":
                        # LangGraph may complete parallel tool calls out of
                        # issue order. ToolMessage already carries the true
                        # name and execution status, so never infer either
                        # from a FIFO queue of prior agent calls.
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
                    if stop_for_duplicate:
                        break
                if stop_for_duplicate:
                    break
            if stop_for_duplicate:
                break

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
                                if tc["name"] == "write_file":
                                    retry_emitted_write_call = True
                                on_step(event="tool_start", name=tc["name"], args=tc.get("args", {}))
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

        # Defense in depth: a tool or a vision-model response must not place a
        # credential-shaped string into the UI, history, or session log.
        if contains_secret_content(final_text):
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
