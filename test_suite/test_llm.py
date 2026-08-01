# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Regression tests for the OpenAI-compatible MLX client boundary."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

import config
from llm import build_llm


def _payload(llm):
    return llm._get_request_payload([HumanMessage(content="hello")])


def test_max_tokens_uses_mlx_vlm_wire_field():
    payload = _payload(build_llm())

    assert "max_completion_tokens" not in payload
    assert payload["extra_body"]["max_tokens"] == config.MAX_TOKENS
    assert payload["extra_body"]["enable_thinking"] is config.ENABLE_THINKING
    assert payload["extra_body"]["thinking_budget"] == config.THINKING_BUDGET == 256
    assert payload["extra_body"]["max_tokens"] == config.MAX_TOKENS == 768


def test_max_tokens_override_uses_mlx_vlm_wire_field():
    payload = _payload(build_llm(max_tokens=73))

    assert "max_completion_tokens" not in payload
    assert payload["extra_body"]["max_tokens"] == 73


def test_thinking_can_be_disabled_for_a_specific_call():
    payload = _payload(build_llm(extra_body={"enable_thinking": False}))

    assert payload["extra_body"]["enable_thinking"] is False
    assert payload["extra_body"]["thinking_budget"] == config.THINKING_BUDGET
