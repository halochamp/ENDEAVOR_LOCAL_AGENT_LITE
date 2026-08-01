# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""react.py — ReAct agent builder, copied-and-trimmed from
ENDEAVOR_LOCAL_AGENT_MAX_VLM/react.py (2026-07-23, architecture switch — see
config.py's module docstring).

Kept from MAX_VLM's pattern: create_react_agent(llm, tools, prompt=dynamic_prompt)
with a dynamic_prompt callable that (a) keeps the system message byte-identical
every turn (prompt-cache friendly) and (b) injects any pending image from
tools.read_image into the last HumanMessage right before each model call —
this is what lets the model "see" an image in the same turn it was read,
mirroring MAX_VLM's direct-vision fork behavior (that project's CLAUDE.md
vision-subsystem table).

Dropped from MAX_VLM's react.py: the ~44KB SYSTEM prompt (KB routing, plan/
create_plan, investment framework, 40-tool shortlist — built for a 35B model
with a large context budget and a much bigger toolset). The lite agent's
system prompt lives in prompts.py — short, example-driven, tuned for a 2B
model per this project's own training-round findings (see prompts.py's
docstring and PLAN.md).
"""
from __future__ import annotations
import base64
from pathlib import Path

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

import config
from llm import build_llm
from tools import ALL_TOOLS
from tools import read_image as _read_image_tool
import prompts

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".tiff": "image/tiff", ".heic": "image/heic", ".heif": "image/heic",
}


def _image_to_data_uri(path: str) -> str | None:
    """mlx_vlm.server's OpenAI-compatible endpoint expects image_url content
    as a base64 data URI (see mlx_vlm/prompt_utils.py) — read_image only
    queues bare local paths, so this is the conversion point, done here
    (not in tools/read_image.py) since it's specific to the server-based
    transport, not the tool's own logic."""
    pending = _read_image_tool.read_pending_snapshot_bytes(path)
    if pending is not None:
        p, data = pending
    else:
        p = _read_image_tool.safe_image_path(path)
        data = _read_image_tool.read_safe_image_bytes(path)
    try:
        if p is None or data is None:
            return None
        mime = _MIME_BY_EXT.get(p.suffix.lower(), "image/png")
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except ValueError:
        pass
    finally:
        # These are private, per-turn snapshots.  Even a corrupt/oversize
        # snapshot must not accumulate if the VLM transport rejects it.
        try:
            candidate = Path(path).resolve()
            candidate.relative_to(config.DATA_DIR / "pending_images")
            candidate.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
    return None


def _inject_pending_images(msgs: list) -> list:
    """Attach queued images to the latest user message, or preserve them.

    Kept as a pure-ish helper so the server transport path is testable without
    starting a model server.  The queue is process-global, so agent.run_turn's
    turn lock serializes callers while this function consumes it.
    """
    try:
        pending_paths = _read_image_tool.pop_pending_images()
    except Exception:
        pending_paths = []
    if not pending_paths:
        return msgs

    # Do not consume snapshots merely to discover that this graph invocation
    # has no HumanMessage.  They remain valid for the next model call.
    user_index = next((i for i in range(len(msgs) - 1, -1, -1)
                       if isinstance(msgs[i], HumanMessage)), None)
    if user_index is None:
        _read_image_tool._PENDING_IMAGES.extend(pending_paths)
        return msgs

    img_blocks = []
    for p in pending_paths:
        uri = _image_to_data_uri(p)
        if uri:
            img_blocks.append({"type": "image_url", "image_url": {"url": uri}})
    if img_blocks:
        m = msgs[user_index]
        base = m.content if isinstance(m.content, list) else [{"type": "text", "text": m.content}]
        msgs[user_index] = HumanMessage(content=base + img_blocks, id=getattr(m, "id", None))
        return msgs

    # Invalid/oversize private snapshots were cleaned by _image_to_data_uri.
    # Do not requeue stale paths indefinitely.
    return msgs


def build_react_agent(tools=None, system: str | None = None, checkpointer=None, **llm_overrides):
    """Build the lite agent's ReAct graph. `tools` overrides ALL_TOOLS;
    `system` overrides prompts.SYSTEM_PROMPT."""
    llm_client = build_llm(**llm_overrides)
    active_tools = tools if tools is not None else ALL_TOOLS
    base_system = system if system is not None else prompts.SYSTEM_PROMPT

    def dynamic_prompt(state: dict) -> list:
        msgs = list(state.get("messages", []))
        msgs = _inject_pending_images(msgs)
        return [SystemMessage(content=base_system)] + msgs

    return create_react_agent(llm_client, active_tools, prompt=dynamic_prompt, checkpointer=checkpointer)
