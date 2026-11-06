"""tools/__init__.py — tool registry.

Each tools/<name>.py module exports:
  SCHEMA: dict   — OpenAI-style function schema (still used by SCHEMAS below,
                   for hook-based validation — see agent.py's pre/post hooks)
  run(args: dict[str, str]) -> str        — the actual (tested) tool logic
  <name>(...): BaseTool (via @tool)       — LangChain entry point wrapping run(),
                   bound to the react agent via ALL_TOOLS (2026-07-23, architecture
                   switch to create_react_agent — see config.py's module docstring)
"""
from __future__ import annotations

from tools import bash, read_file, write_file, web_search, rag_search, read_image, local_search, mcp_client

_MODULES = {
    "bash": bash,
    "read_file": read_file,
    "write_file": write_file,
    "web_search": web_search,
    "local_search": local_search,
    "read_image": read_image,
}

# LangChain tool objects (bound to create_react_agent) — name matches the
# module's own @tool-decorated function.
ALL_TOOLS = [
    bash.bash, read_file.read_file, write_file.write_file, web_search.web_search,
    local_search.local_search, read_image.read_image,
    mcp_client.mcp_list_tools, mcp_client.mcp_call_tool,
]

TOOLS: list[dict] = [
    *(m.SCHEMA for m in _MODULES.values()),
    mcp_client.MCP_LIST_SCHEMA,
    mcp_client.MCP_CALL_SCHEMA,
]
SCHEMAS: dict[str, dict] = {name: m.SCHEMA["function"] for name, m in _MODULES.items()}
SCHEMAS.update({
    "mcp_list_tools": mcp_client.MCP_LIST_SCHEMA["function"],
    "mcp_call_tool": mcp_client.MCP_CALL_SCHEMA["function"],
})
REGISTRY: dict[str, callable] = {name: m.run for name, m in _MODULES.items()}
REGISTRY.update({
    "mcp_list_tools": mcp_client.run_list,
    "mcp_call_tool": mcp_client.run_call,
})
