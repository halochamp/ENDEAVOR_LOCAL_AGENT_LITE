# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""tool_parser.py — parse and validate Qwen3.5's <tool_call> XML format.

Qwen3.5's chat template (see the model's chat_template.jinja) expects tool
calls shaped like:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Bangkok
    </parameter>
    </function>
    </tool_call>

Only the first tool call in a response is parsed — the lite agent's loop
executes one tool per step and feeds the result back, so a second call in
the same message (if the model ever emits one) is intentionally ignored
rather than silently executed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re

_FUNCTION_RE = re.compile(r"<function=([\w\-.]+)>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([\w\-.]+)>\n?(.*?)\n?</parameter>", re.S)


@dataclass
class ToolCall:
    name: str
    args: dict[str, str] = field(default_factory=dict)
    raw: str = ""


def parse_tool_call(text: str) -> ToolCall | None:
    """Return the first tool call found in `text`, or None if there isn't one."""
    if "<tool_call>" not in text:
        return None
    m = _FUNCTION_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    body = m.group(2)
    args = {k: v.strip() for k, v in _PARAM_RE.findall(body)}
    return ToolCall(name=name, args=args, raw=m.group(0))


def strip_tool_call(text: str) -> str:
    """Return `text` with the <tool_call>...</tool_call> block removed.

    Used to salvage any natural-language reasoning the model wrote before
    the call, per the template's own instruction that reasoning may only
    precede (never follow) a function call.
    """
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()


# Code-level intercept (training_guide.md L11/L36 pattern: prompt-only tool
# selection is unreliable on a small model — fix at code, not by adding more
# prompt rules). Live-observed 2026-07-23: asked to search file content, the
# 2B model called rag_search but used bash's parameter name "command"
# instead of rag_search's "query" — it borrowed the shape of a
# recently-relevant tool. Remap known synonyms before validating, so a
# plausible-but-wrong param name doesn't hard-fail a call that's otherwise
# correct.
_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "rag_search": {"command": "query", "search": "query", "keyword": "query", "keywords": "query"},
    "web_search": {"command": "query", "search": "query", "keyword": "query"},
    "read_file": {"file": "path", "filepath": "path", "file_path": "path"},
    "write_file": {"file": "path", "filepath": "path", "file_path": "path", "text": "content", "data": "content"},
    "read_image": {"file": "path", "filepath": "path", "image": "path", "image_path": "path"},
    "bash": {"cmd": "command", "query": "command"},
}


def apply_param_aliases(call: ToolCall, schemas: dict[str, dict]) -> ToolCall:
    aliases = _PARAM_ALIASES.get(call.name)
    schema = schemas.get(call.name)
    if not aliases or not schema:
        return call
    required = set(schema.get("parameters", {}).get("required", []))
    missing = required - set(call.args)
    if not missing:
        return call
    new_args = dict(call.args)
    for wrong_name, correct_name in aliases.items():
        if correct_name in missing and wrong_name in new_args and correct_name not in new_args:
            new_args[correct_name] = new_args.pop(wrong_name)
    return ToolCall(name=call.name, args=new_args, raw=call.raw)


def validate_tool_call(call: ToolCall, schemas: dict[str, dict]) -> str | None:
    """Check `call` against the tool JSON-schema registry.

    Returns None if valid, else a short Thai error string suitable for
    feeding straight back to the model as a <tool_response>.
    """
    schema = schemas.get(call.name)
    if schema is None:
        known = ", ".join(sorted(schemas)) or "(ไม่มี)"
        return f"ไม่รู้จัก tool ชื่อ '{call.name}' — tool ที่มีคือ: {known}"

    params = schema.get("parameters", {})
    required = params.get("required", [])
    missing = [p for p in required if p not in call.args or not call.args[p]]
    if missing:
        return (
            f"tool '{call.name}' ขาด parameter ที่จำเป็น: {', '.join(missing)} — "
            f"ลองเรียกใหม่พร้อมใส่ parameter ให้ครบ"
        )
    return None
