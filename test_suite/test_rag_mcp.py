"""Focused tests for Agent Lite's RAG MCP service and client boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_agent_registry_exposes_mcp_not_native_rag():
    import tools

    names = [tool.name for tool in tools.ALL_TOOLS]
    assert "mcp_list_tools" in names
    assert "mcp_call_tool" in names
    assert "rag_search" not in names
    assert [schema["function"]["name"] for schema in tools.TOOLS] == names


def test_lite_rag_server_is_fixed_developer_provisioned():
    import config

    assert set(config.MCP_SERVERS) == {"lite-rag"}
    cfg = config.MCP_SERVERS["lite-rag"]
    assert cfg["transport"] == "stdio"
    assert Path(cfg["args"][0]).resolve() == (config.RAG_CODE_DIR / "rag_server.py").resolve()
    assert Path(cfg["cwd"]).resolve() == config.MCP_WORKSPACE.resolve()
    write_paths = {Path(p).resolve() for p in cfg["sandbox_write_paths"]}
    assert write_paths == {config.DATA_DIR.resolve()}
    assert config.KNOWLEDGE_DIR.resolve() not in write_paths


def test_mcp_sandbox_denies_sensitive_reads_and_allows_only_provisioned_writes():
    import config
    from tools import mcp_client

    cfg = config.MCP_SERVERS["lite-rag"]
    profile = mcp_client._build_mcp_sandbox_profile(
        mcp_client._sandbox_write_paths("lite-rag", cfg)
    )
    assert "(deny network*)" in profile
    assert "(deny file-write*)" in profile
    assert str(config.DATA_DIR.resolve()) in profile
    assert str(config.KNOWLEDGE_DIR.resolve()) not in profile
    assert str((Path.home() / ".ssh").resolve()) in profile


def test_mcp_recreates_missing_provisioned_state_directories(tmp_path, monkeypatch):
    import config
    from tools import mcp_client

    workspace = tmp_path / "mcp_workspace"
    state = tmp_path / "state"
    cfg = {
        "transport": "stdio",
        "command": str(config.MCP_PYTHON),
        "args": [],
        "cwd": str(workspace),
        "sandbox_write_paths": [str(state)],
    }
    monkeypatch.setattr(config, "MCP_WORKSPACE", workspace)
    monkeypatch.setitem(config.MCP_SERVERS, "lite-rag", cfg)

    normalized = mcp_client._normalise_server_config(cfg)
    paths = mcp_client._sandbox_write_paths("lite-rag", cfg)

    assert normalized["cwd"] == str(workspace.resolve())
    assert workspace.is_dir()
    assert paths == (str(state.resolve()),)
    assert state.is_dir()


def test_mcp_client_stdio_round_trip_without_outer_sandbox(tmp_path, monkeypatch):
    """Exercise official MCP SDK transport while bypassing only nested sandbox-exec."""
    import config
    from helpers import turn_guard
    from tools import mcp_client

    server = tmp_path / "fake_mcp.py"
    server.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp=FastMCP('fake')\n"
        "@mcp.tool()\n"
        "def ping(text: str='ok') -> str:\n"
        "    return text.upper()\n"
        "if __name__ == '__main__': mcp.run(transport='stdio')\n",
        encoding="utf-8",
    )
    wrapper = tmp_path / "sandbox-wrapper"
    wrapper.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-f\" ] || [ \"$1\" = \"-p\" ]; then shift 2; fi\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    fake_cfg = {
        "transport": "stdio",
        "command": str(config.MCP_PYTHON),
        "args": [str(server)],
        "cwd": str(config.MCP_WORKSPACE),
        "sandbox_write_paths": [],
    }
    monkeypatch.setitem(config.MCP_SERVERS, "fake", fake_cfg)
    monkeypatch.setattr(mcp_client, "_SANDBOX_EXEC", str(wrapper))
    turn_guard.reset("mcp transport test")

    listed = mcp_client.mcp_list_tools.invoke({"server": "fake"})
    called = mcp_client.mcp_call_tool.invoke({
        "server": "fake", "tool_name": "ping", "arguments_json": json.dumps({"text": "hello"}),
    })
    assert "ping" in listed
    assert called == "HELLO"


def test_actual_lite_rag_server_lists_three_tools_without_outer_sandbox(tmp_path, monkeypatch):
    import config
    from helpers import turn_guard
    from tools import mcp_client

    wrapper = tmp_path / "sandbox-wrapper"
    wrapper.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-f\" ] || [ \"$1\" = \"-p\" ]; then shift 2; fi\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    monkeypatch.setattr(mcp_client, "_SANDBOX_EXEC", str(wrapper))
    turn_guard.reset("list lite rag")
    listed = mcp_client.mcp_list_tools.invoke({"server": "lite-rag"})
    health = mcp_client.mcp_call_tool.invoke({
        "server": "lite-rag", "tool_name": "rag_status",
        "arguments_json": json.dumps({"view": "health"}),
    })
    assert not listed.startswith("[error]")
    assert {"rag_search", "rag_manage", "rag_status"}.issubset(set(line.split(":", 1)[0] for line in listed.splitlines()))
    assert health.startswith("status=")
    assert "documents=" in health
    assert "chunks=" in health


def test_mcp_child_inherits_parent_rag_index_context(monkeypatch):
    import config
    import rag_mcp_service
    from helpers import turn_guard
    from tools import mcp_client

    # Keep both source and state inside Lite's explicit workspect boundary. The
    # Lite adapter must translate these trusted parent paths into generic
    # RAGDOC_* child settings; MCP-RagDoc itself knows nothing about Lite.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="lite-mcp-context-", dir=config.WORKSPECT_DIR) as name:
        root = Path(name)
        knowledge = root / "knowledge"
        knowledge.mkdir()
        source = knowledge / "renewal.txt"
        source.write_text("policy renewal date 30 September 2569 premium 12500", encoding="utf-8")
        db = root / "live-index.sqlite3"
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
        monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", knowledge)
        monkeypatch.setattr(config, "RAW_DIR", knowledge)
        monkeypatch.setattr(config, "DATA_DIR", root)
        monkeypatch.setattr(config, "FILE_INDEX_DB", db)
        monkeypatch.setattr(config, "SEMANTIC_ENABLED", False)
        rag_mcp_service.sync_kb()
        wrapper = root / "sandbox-wrapper"
        wrapper.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"-f\" ] || [ \"$1\" = \"-p\" ]; then shift 2; fi\n"
            "exec \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        monkeypatch.setattr(mcp_client, "_SANDBOX_EXEC", str(wrapper))
        turn_guard.reset("mcp child context")

        result = mcp_client.mcp_call_tool.invoke({
            "server": "lite-rag",
            "tool_name": "rag_search",
            "arguments_json": json.dumps({"query": "renewal.txt"}),
        })

        assert str(source.resolve()) in result
        assert "renewal" in result.casefold()


def test_service_search_and_sync_share_backend_contract(tmp_path, monkeypatch):
    import config
    import kb_sync
    import rag_mcp_service

    knowledge = tmp_path / "knowledge"
    raw = knowledge  # Compatibility fixture name; sources live at knowledge/ root.
    md = knowledge / "md"
    knowledge.mkdir(parents=True)
    md.mkdir()
    data = tmp_path / "state"
    data.mkdir()
    db = data / "file_index.sqlite3"

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", raw)
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", md)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "MD_DIR", md)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "INDEX_ROOTS", [md])
    monkeypatch.setattr(config, "KB_CHUNK_TARGET_CHARS", 240)
    monkeypatch.setattr(config, "KB_CHUNK_OVERLAP_CHARS", 30)

    source = raw / "thai.txt"
    source.write_text("ระบบน้ำมันเครื่องยนต์ทำงานปกติ hydraulic marker", encoding="utf-8")
    stats = rag_mcp_service.sync_kb()
    assert stats["inserted"] == 1
    result = rag_mcp_service.search("น้ำมันเครื่องยนต์")
    assert str(source.resolve()) in result
    assert "hydraulic" in rag_mcp_service.search("hydraulic").casefold()
    health = rag_mcp_service.health()
    assert "status=healthy" in health
    assert "documents=1" in health
    assert "chunks=" in health


def test_sync_result_is_compact_and_reports_degraded_errors():
    import rag_mcp_service

    text = rag_mcp_service.format_sync_result({
        "raw_scanned": 3, "indexed": 2, "unchanged": 1, "documents": 2,
        "chunks": 9, "inserted": 1, "updated": 1, "removed": 0,
        "sync_errors": 1, "seconds": 1.234,
    })
    assert "status=degraded" in text
    assert "sync_errors=1" in text
    assert "seconds=1.23" in text


def test_indexer_worker_sync_flag_routes_through_mcp_service(monkeypatch, capsys):
    import indexer_worker
    import rag_mcp_service

    stats = {
        "raw_scanned": 1, "indexed": 1, "unchanged": 0, "removed": 0,
        "documents": 1, "chunks": 2, "inserted": 1, "updated": 0,
        "sync_errors": 0, "seconds": 0.01,
    }
    calls = []

    def fake_sync(*, on_progress=None):
        calls.append("sync")
        if on_progress is not None:
            on_progress("done", None, stats)
        return stats

    monkeypatch.setattr(rag_mcp_service, "sync_kb", fake_sync)
    monkeypatch.setattr(sys, "argv", ["indexer_worker.py", "--sync-kb"])
    indexer_worker.main()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert calls == ["sync"]
    assert lines == [{"event": "done", "stats": stats}]


def test_mcp_rag_duplicate_recovery_uses_logical_tool_name(monkeypatch):
    import agent

    class Graph:
        def stream(self, _input, *, config, stream_mode):
            args = {
                "server": "lite-rag",
                "tool_name": "rag_search",
                "arguments_json": json.dumps({"query": "syllabus"}),
            }
            first = {"name": "mcp_call_tool", "id": "rag-1", "args": dict(args)}
            second = {"name": "mcp_call_tool", "id": "rag-2", "args": dict(args)}
            yield {"agent": {"messages": [type("M", (), {"tool_calls": [first]})()]}}
            yield {"tools": {"messages": [type("M", (), {
                "name": "mcp_call_tool", "tool_call_id": "rag-1", "status": "success",
                "content": "พบ 1 ไฟล์\n- /project/syllabus.pdf",
            })()]}}
            yield {"agent": {"messages": [type("M", (), {"tool_calls": [second]})()]}}
            raise AssertionError("duplicate MCP RAG call must stop before another execution")

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    monkeypatch.setattr(agent, "_should_force_web_search", lambda _query: False)
    result = agent.run_turn("ค้นหา syllabus ใน knowledge", [])
    assert result.aborted is False
    assert "/project/syllabus.pdf" in result.text


def test_grouped_rag_mcp_actions_keep_logical_names():
    import agent

    assert agent._logical_tool_name("mcp_call_tool", {
        "server": "lite-rag", "tool_name": "rag_manage",
        "arguments_json": json.dumps({"action": "sync"}),
    }) == "sync_kb"
    assert agent._logical_tool_name("mcp_call_tool", {
        "server": "lite-rag", "tool_name": "rag_manage",
        "arguments_json": json.dumps({"action": "cancel", "job_id": "sync-abc"}),
    }) == "cancel_sync"
    assert agent._logical_tool_name("mcp_call_tool", {
        "server": "lite-rag", "tool_name": "rag_status",
        "arguments_json": json.dumps({"view": "job", "job_id": "sync-abc"}),
    }) == "sync_status"
    assert agent._logical_tool_name("mcp_call_tool", {
        "server": "lite-rag", "tool_name": "rag_status",
        "arguments_json": json.dumps({"view": "health"}),
    }) == "rag_health"


def test_actual_lite_rag_stdio_sync_search_health_on_isolated_kb(tmp_path, monkeypatch):
    """Full Pipe-C-style route: MCP client -> stdio -> FastMCP -> service -> SQLite."""
    import config
    from helpers import turn_guard
    from tools import mcp_client

    knowledge = tmp_path / "knowledge"
    state = tmp_path / "state"
    knowledge.mkdir(parents=True)
    state.mkdir()
    source = knowledge / "manual.txt"
    source.write_text("ระบบน้ำมันเครื่องยนต์ hydraulic MCP marker", encoding="utf-8")
    server = config.RAG_CODE_DIR / "rag_server.py"
    wrapper = tmp_path / "sandbox-wrapper"
    wrapper.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-f\" ] || [ \"$1\" = \"-p\" ]; then shift 2; fi\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    cfg = {
        "transport": "stdio",
        "command": str(config.MCP_PYTHON),
        "args": [
            str(server),
            "--knowledge-dir", str(knowledge),
            "--data-dir", str(state),
            "--db", str(state / "file_index.sqlite3"),
            "--semantic", "off",
        ],
        "cwd": str(config.MCP_WORKSPACE),
        "sandbox_write_paths": [],
    }
    monkeypatch.setitem(config.MCP_SERVERS, "isolated-rag", cfg)
    monkeypatch.setattr(mcp_client, "_SANDBOX_EXEC", str(wrapper))
    turn_guard.reset("isolated MCP RAG E2E")

    sync_result = mcp_client.mcp_call_tool.invoke({
        "server": "isolated-rag", "tool_name": "rag_manage",
        "arguments_json": json.dumps({"action": "sync"}),
    })
    assert "status=started" in sync_result
    job_id = next(
        line.split("=", 1)[1]
        for line in sync_result.splitlines()
        if line.startswith("job_id=")
    )
    final_status = ""
    for _ in range(100):
        final_status = mcp_client.mcp_call_tool.invoke({
            "server": "isolated-rag",
            "tool_name": "rag_status",
            "arguments_json": json.dumps({"view": "job", "job_id": job_id}),
        })
        if "status=done" in final_status or "status=error" in final_status or "status=cancelled" in final_status:
            break
        time.sleep(0.05)

    assert "status=done" in final_status
    assert "inserted=1" in final_status
    search_result = mcp_client.mcp_call_tool.invoke({
        "server": "isolated-rag", "tool_name": "rag_search",
        "arguments_json": json.dumps({"query": "น้ำมันเครื่องยนต์"}, ensure_ascii=False),
    })
    health_result = mcp_client.mcp_call_tool.invoke({
        "server": "isolated-rag", "tool_name": "rag_status",
        "arguments_json": json.dumps({"view": "health"}),
    })

    assert str(source.resolve()) in search_result
    assert "status=healthy" in health_result
    assert "documents=1" in health_result
    assert "chunks=" in health_result


def test_legacy_database_error_fails_closed_with_sync_hint(tmp_path, monkeypatch):
    import config
    import sqlite3
    from tools import rag_search

    db = tmp_path / "legacy.sqlite3"
    db.write_bytes(b"exists")
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)
    monkeypatch.setattr(rag_search.config, "FILE_INDEX_DB", db)
    monkeypatch.setattr(
        rag_search,
        "_fallback_filename_search",
        lambda *_args, **_kwargs: "ไม่พบไฟล์ที่ชื่อตรงกับ query",
    )

    class EmptyCursor:
        def __iter__(self):
            return iter(())

        def fetchone(self):
            return None

    class BrokenLegacyConnection:
        def execute(self, sql, _params=()):
            if "sqlite_master" in sql or "PRAGMA table_info" in sql:
                return EmptyCursor()
            raise sqlite3.DatabaseError("database disk image is malformed")

        def close(self):
            return None

    monkeypatch.setattr(rag_search.sqlite3, "connect", lambda *_args, **_kwargs: BrokenLegacyConnection())
    result = rag_search.run({"query": "hydraulic"})
    assert result.startswith("[error] Knowledge Base index เสียหายหรืออ่านไม่ได้")
    assert "rag_manage(action='sync')" in result


def test_health_rejects_partial_v3_schema_missing_primary_content(tmp_path, monkeypatch):
    import config
    import rag_mcp_service
    import sqlite3

    knowledge = tmp_path / "knowledge"
    raw = knowledge
    knowledge.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    db = state / "file_index.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE kb_documents(path TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
            "path UNINDEXED, chunk_index UNINDEXED, name, location UNINDEXED, "
            "search_text, tokenize='trigram')"
        )
        conn.commit()

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", raw)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "DATA_DIR", state)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)

    result = rag_mcp_service.health()
    assert result.startswith("status=needs_sync")
    assert "reason=kb_schema_old" in result


def test_health_ignores_inert_zero_length_journal(tmp_path, monkeypatch):
    import config
    import rag_mcp_service
    import sqlite3

    knowledge = tmp_path / "knowledge"
    raw = knowledge
    knowledge.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    db = state / "file_index.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE kb_documents(path TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
            "path UNINDEXED, chunk_index UNINDEXED, name, location UNINDEXED, "
            "content UNINDEXED, search_text, tokenize='trigram')"
        )
        conn.commit()
    Path(str(db) + "-journal").write_bytes(b"")

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", raw)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "DATA_DIR", state)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)

    result = rag_mcp_service.health()
    assert result.startswith("status=healthy")
    assert Path(str(db) + "-journal").read_bytes() == b""


def test_health_sidecar_is_observation_only(tmp_path, monkeypatch):
    import config
    import rag_mcp_service

    knowledge = tmp_path / "knowledge"
    raw = knowledge
    knowledge.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    db = state / "file_index.sqlite3"
    db.write_bytes(b"not-opened-because-sidecar-is-authoritative")
    Path(str(db) + "-journal").write_bytes(b"pending" + (b"x" * 506))

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", raw)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "DATA_DIR", state)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)

    result = rag_mcp_service.health()
    assert result == (
        "status=degraded\nsource_files=0\ndocuments=unknown\nchunks=unknown\n"
        "reason=sqlite_recovery_pending"
    )
    assert Path(str(db) + "-journal").read_bytes() == b"pending" + (b"x" * 506)


def test_mcp_client_rejects_unknown_server_invalid_json_and_unknown_tool(tmp_path, monkeypatch):
    import config
    from helpers import turn_guard
    from tools import mcp_client

    turn_guard.reset("invalid MCP inputs")
    assert mcp_client.run_call({
        "server": "missing", "tool_name": "x", "arguments_json": "{}",
    }).startswith("[error] no MCP server")
    assert "invalid arguments_json" in mcp_client.run_call({
        "server": "lite-rag", "tool_name": "rag_search", "arguments_json": "{bad",
    })

    wrapper = tmp_path / "sandbox-wrapper"
    wrapper.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"-f\" ] || [ \"$1\" = \"-p\" ]; then shift 2; fi\nexec \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    monkeypatch.setattr(mcp_client, "_SANDBOX_EXEC", str(wrapper))
    unknown = mcp_client.run_call({
        "server": "lite-rag", "tool_name": "does_not_exist", "arguments_json": "{}",
    })
    assert unknown.startswith("[error]")
