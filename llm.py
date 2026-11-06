"""llm.py — ChatOpenAI factory pointed at mlx_vlm.server.

Copied-and-trimmed from ENDEAVOR_LOCAL_AGENT_MAX_VLM/llm.py (2026-07-23,
architecture switch to a real server — see config.py's module docstring for
why). Lite keeps a smaller bounded thinking budget for its 2B model and the
same repetition guard used by MAX_VLM.
"""
from __future__ import annotations
import httpx
from langchain_openai import ChatOpenAI

import config


def build_llm(**overrides) -> ChatOpenAI:
    """Build a ChatOpenAI client for mlx_vlm.server (OpenAI-compatible).

    overrides: any param can be overridden (e.g. temperature for a specific
    call site). extra_body is merged (not replaced) with the default so an
    override doesn't accidentally drop repetition_penalty.
    """
    override_extra_body = overrides.pop("extra_body", {})
    # Newer langchain-openai versions rename ``max_tokens`` to
    # ``max_completion_tokens``.  mlx_vlm.server currently accepts the former
    # only, silently falling back to its much larger server default otherwise.
    # OpenAI's client merges ``extra_body`` into the wire JSON, so keep the
    # legacy field there until mlx_vlm supports the newer alias.
    wire_max_tokens = overrides.pop("max_tokens", config.MAX_TOKENS)
    wire_max_tokens = overrides.pop("max_completion_tokens", wire_max_tokens)
    params = dict(
        base_url=config.MLX_BASE_URL,
        api_key=config.MLX_API_KEY,
        model=config.MODEL_ID,
        temperature=config.TEMPERATURE,
        streaming=False,
        # Bounds the gap between response chunks — without this, a stalled
        # server response blocks the calling thread indefinitely.
        timeout=httpx.Timeout(10.0, read=90.0, write=10.0, pool=10.0),
        extra_body={
            "enable_thinking": config.ENABLE_THINKING,
            "thinking_budget": config.THINKING_BUDGET,
            "repetition_penalty": config.REPETITION_PENALTY,
            "max_tokens": wire_max_tokens,
            **override_extra_body,
        },
    )
    params.update(overrides)
    return ChatOpenAI(**params)
