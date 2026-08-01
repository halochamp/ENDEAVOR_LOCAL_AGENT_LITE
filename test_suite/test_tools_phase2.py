# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No model needed — direct tool.run() tests for bash/read_file/write_file."""
import sys, os, shlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import subprocess
import tempfile
from tools import bash, read_file, write_file
from helpers import turn_guard


def test_lite_state_stays_within_workspect():
    config.DATA_DIR.resolve().relative_to(config.WORKSPECT_DIR.resolve())
    config.LOG_DIR.resolve().relative_to(config.WORKSPECT_DIR.resolve())


def test_bash_safe_command():
    assert bash.run({"command": "echo hello"}).strip() == "hello"


def test_bash_allows_read_only_wc_count_pipeline(tmp_path):
    (tmp_path / "one.pdf").write_text("one")
    (tmp_path / "two.txt").write_text("two")

    out = bash.run({"command": f"find {tmp_path} -type f -iname '*.pdf' | wc -l"})

    assert out.strip() == "1"


def test_bash_allows_read_only_sort_pipeline(tmp_path):
    (tmp_path / "z.pdf").write_text("")
    (tmp_path / "a.pdf").write_text("")

    out = bash.run({"command": f"find {tmp_path} -type f | sort"})

    assert out.splitlines() == sorted(out.splitlines())


def test_large_bash_output_starts_with_exact_count_and_full_path(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TOOL_OUTPUT_TRUNCATE_CHARS", 20)
    for index in range(12):
        (tmp_path / f"{index:02d}_document.pdf").write_text("")

    out = bash.run({"command": f"find {tmp_path} -type f -iname '*.pdf'"})

    assert out.startswith("[full output: ")
    assert "; saved: " in out
    assert "preview below is incomplete" in out


def test_find_alignment_preserves_requested_document_kind():
    assert bash._align_find_to_document_kind(
        "find /Users/example/Documents -type f | wc -l",
        "มีเอกสาร PDF กี่ไฟล์ใน Documents",
    ) == "find /Users/example/Documents -type f -iname '*.pdf' | wc -l"
    assert bash._align_find_to_document_kind(
        "find /Users/example/Documents -type f -name '*.docx' | wc -l",
        "นับไฟล์ Excel ใน Documents",
    ) == "find /Users/example/Documents -type f -iname '*.xlsx' | wc -l"


def test_whole_machine_alignment_replaces_project_grep_with_mdfind():
    command = bash.normalize_command_for_query(
        'grep -r -l "เลขที่กรมธรรม์" /Users/example/project',
        "ค้นหาทั้งเครื่องที่มีคำว่า เลขที่กรมธรรม์",
    )
    assert command == "mdfind 'เลขที่กรมธรรม์'"


def test_listing_alignment_removes_accidental_count_stage():
    assert bash.normalize_command_for_query(
        "find /Users/example/Downloads -type f | sort | wc -l",
        "ช่วยใช้ bash ดูรายการเอกสารใน Downloads",
    ) == "find /Users/example/Downloads -type f | sort"
    assert bash.normalize_command_for_query(
        "find /Users/example/Documents -type f | wc -l",
        "นับว่ามีกี่ไฟล์ใน Documents",
    ).endswith("| wc -l")


def test_turn_guard_keeps_original_user_query_for_tool_fidelity():
    turn_guard.reset("นับไฟล์ PDF")
    assert turn_guard.user_query() == "นับไฟล์ PDF"


def test_bash_denies_rm_rf():
    out = bash.run({"command": "rm -rf /"})
    assert out.startswith("[blocked]")


def test_bash_denies_sudo():
    out = bash.run({"command": "sudo ls /"})
    assert out.startswith("[blocked]")


def test_bash_denies_fork_bomb():
    out = bash.run({"command": ":(){ :|:& };:"})
    assert out.startswith("[blocked]")


def test_bash_denies_all_write_syntax_and_unknown_executables():
    for command in (
        "echo nope > /tmp/nope.txt", "touch /tmp/nope.txt", "python -c 'print(1)'",
        "sed -i '' 's/a/b/' file.txt", "sed 'w output.txt' file.txt", "sort -o output.txt file.txt",
        "env rm -rf /", "find . -delete", "find . -fprint0 output.txt", "curl https://example.com",
        "/tmp/ls", "git branch new-branch", "git commit -m nope", "rg --pre /tmp/helper needle .", "hostname changed-name",
        "cat /dev/zero", "git grep --textconv needle", "git grep -O'cat /etc/passwd' needle",
        "git branch --unset-upstream",
    ):
        assert bash.run({"command": command}).startswith("[blocked]"), command


def test_bash_sandbox_blocks_write_bypass(tmp_path):
    target = tmp_path / "bypass.txt"
    # `tee` is rejected before launch, while the OS sandbox remains the
    # independent boundary for a parser mistake or future safe-command bug.
    assert bash.run({"command": f"echo nope | tee {target}"}).startswith("[blocked]")
    assert not target.exists()


def test_bash_sandbox_enforces_no_write_even_if_parser_is_bypassed(tmp_path):
    target = tmp_path / "kernel_boundary.txt"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as profile:
        profile.write(bash._build_readonly_sandbox_profile())
        profile_path = profile.name
    try:
        result = subprocess.run(
            ["sandbox-exec", "-f", profile_path, "bash", "-c", f"touch {target}"],
            capture_output=True, text=True,
        )
    finally:
        os.unlink(profile_path)
    assert result.returncode != 0
    assert not target.exists()


def test_bash_allows_non_sensitive_file_read_outside_workspect(tmp_path):
    outside = tmp_path / "readable.txt"
    outside.write_text("outside workspace")
    out = bash.run({"command": f"cat {outside}"})
    assert out == "outside workspace"
    assert read_file.run({"path": str(outside)}) == "outside workspace"


def test_bash_blocks_sensitive_path_and_secret_shaped_output(tmp_path):
    secret_path = tmp_path / ".env"
    secret_path.write_text("API_KEY=abcdefghijklmnop")
    assert bash.run({"command": f"cat {secret_path}"}).startswith("[blocked]")
    ordinary = tmp_path / "ordinary.txt"
    ordinary.write_text("token=abcdefghijklmnop")
    assert bash.run({"command": f"cat {ordinary}"}).startswith("[blocked]")


def test_secret_detector_catches_common_bare_token_formats():
    from helpers.sensitive_paths import contains_secret_content
    assert contains_secret_content("sk-" + "A" * 32)
    assert contains_secret_content("xoxb-123456789012-" + "A" * 24)
    assert contains_secret_content("AIza" + "A" * 35)
    assert contains_secret_content("ghs_" + "A" * 32)
    assert contains_secret_content("ghr_" + "A" * 32)
    assert contains_secret_content("ghu_" + "A" * 32)
    assert contains_secret_content("glpat-" + "A" * 32)
    assert contains_secret_content("npm_" + "A" * 32)
    assert contains_secret_content("sk-\n" + "A" * 32)


def test_bash_allows_git_read_only_subcommands():
    out = bash.run({"command": f"git -C {shlex.quote(str(config.PROJECT_ROOT))} status --short"})
    assert not out.startswith("[blocked]")


def test_bash_allows_rg_files_for_local_file_discovery():
    out = bash.run({"command": f"rg --files {shlex.quote(str(config.WORKSPECT_DIR))}"})
    assert not out.startswith("[blocked]")
    assert "command not found" not in out


def test_bash_missing_command():
    assert bash.run({"command": ""}).startswith("[error]")


# ── ported from ENDEAVOR_LOCAL_AGENT_MAX/tools/bash.py, 2026-07-23 ─────────
def test_bash_mdfind_allowed_in_principle():
    out = bash.run({"command": f"mdfind -onlyin {config.PROJECT_ROOT} -name PLAN.md"})
    assert not out.startswith("[blocked]")
    # Not asserting a match is found: this Mac's Spotlight index for this
    # volume is in an "unknown indexing state" (`mdutil -s`) -- confirmed
    # live 2026-07-23 that even long-existing files return no mdfind hits
    # regardless of sandboxing, so a real result can't be relied on in CI.
    # rag_search remains the primary/reliable file-search path; mdfind is a
    # supplementary allow-listed command, not load-bearing.


def test_bash_mdfind_live_blocked():
    out = bash.run({"command": "mdfind -live foo"})
    assert out.startswith("[blocked]")


def test_bash_mdfind_onlyin_sensitive_path_blocked():
    out = bash.run({"command": "mdfind -onlyin ~/.ssh -name id_rsa"})
    assert out.startswith("[blocked]")


def test_bash_classify_error_unit():
    # The strict allow-list makes a genuine exit-127 ("command not found")
    # nearly unreachable through run() (we only ever exec a validated,
    # installed executable) -- test _classify_bash_error directly instead of
    # trying to provoke one through the full sandboxed pipeline.
    assert "command not found" in bash._classify_bash_error(127, "")
    assert "permission denied" in bash._classify_bash_error(126, "")
    assert bash._classify_bash_error(0, "anything") is None
    assert bash._classify_bash_error(None, "anything") is None
    assert bash._classify_bash_error(1, "Permission denied") is not None


def test_bash_classify_error_never_fires_on_success():
    out = bash.run({"command": "echo ok"})
    assert "[hint]" not in out


def test_read_file_missing_path_arg():
    assert read_file.run({"path": ""}).startswith("[error]")


def test_read_file_relative_path_rejected():
    out = read_file.run({"path": "relative/path.txt"})
    assert out.startswith("[error]")


def test_read_file_nonexistent():
    out = read_file.run({"path": "/nonexistent_dir_xyz/nope.txt"})
    assert "ไม่พบไฟล์" in out


def test_read_file_directory_given():
    out = read_file.run({"path": str(config.PROJECT_ROOT)})
    assert "โฟลเดอร์" in out


def test_read_file_line_range():
    out = read_file.run({
        "path": str(config.PROJECT_ROOT / "README.md"),
        "start_line": "1",
        "end_line": "1",
    })
    assert out.startswith("[บรรทัด 1-1")


def test_read_file_blocks_sensitive_name_anywhere(tmp_path):
    secret = tmp_path / ".env.production"
    secret.write_text("API_KEY=nope")
    assert read_file.run({"path": str(secret)}).startswith("[blocked]")


def test_write_file_outside_allowed_root_blocked():
    out = write_file.run({"path": "/etc/passwd", "content": "x"})
    assert out.startswith("[error]")


def test_write_file_home_outside_lite_workspace_blocked():
    target = os.path.expanduser("~/endeavor_lite_must_not_write_here.txt")
    out = write_file.run({"path": target, "content": "x"})
    assert out.startswith("[error]")
    assert not os.path.exists(target)


def test_write_file_lite_source_tree_outside_workspect_blocked():
    target = config.PROJECT_ROOT / "source_tree_must_not_be_written.txt"
    out = write_file.run({"path": str(target), "content": "x"})
    assert out.startswith("[error]")
    assert not target.exists()


def test_write_file_relative_path_rejected():
    out = write_file.run({"path": "relative.txt", "content": "x"})
    assert out.startswith("[error]")


def test_write_file_roundtrip_and_overwrite_gate(tmp_path_factory=None):
    target = config.BASH_ALLOWED_ROOT / "_scratch_write_roundtrip.txt"
    try:
        r1 = write_file.run({"path": str(target), "content": "hello"})
        assert r1.startswith("written")
        r2 = write_file.run({"path": str(target), "content": "should not land"})
        assert r2.startswith("[error]")
        assert target.read_text() == "hello"
        r3 = write_file.run({"path": str(target), "content": "replaced", "overwrite": "true"})
        assert r3.startswith("overwrote")
        assert target.read_text() == "replaced"
    finally:
        target.unlink(missing_ok=True)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
