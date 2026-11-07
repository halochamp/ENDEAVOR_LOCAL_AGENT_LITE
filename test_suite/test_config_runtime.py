"""Production configuration boundaries for the per-Mac Lite runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _probe(extra: dict[str, str | None]) -> dict:
    env = os.environ.copy()
    for key in (
        "LITE_DATA_DIR", "LITE_FILE_INDEX_DB", "LITE_SERVER_HOST", "LITE_SERVER_PORT",
        "LITE_UI_PORT", "LITE_MLX_BASE_URL", "LITE_MLX_API_KEY", "LITE_MAX_TOKENS",
    ):
        env.pop(key, None)
    for key, value in extra.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    code = (
        "import json, sys; sys.path.insert(0, '.'); import config; "
        "print(json.dumps({'data': str(config.DATA_DIR), 'default': str(config.DEFAULT_DATA_DIR), "
        "'file': str(config.FILE_INDEX_DB), 'host': config.MLX_SERVER_HOST, "
        "'port': config.MLX_SERVER_PORT, 'ui': config.UI_PORT, 'base': config.MLX_BASE_URL, "
        "'tokens': config.MAX_TOKENS}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT,
        env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_default_state_is_per_machine_and_unsafe_override_is_ignored(tmp_path):
    default = _probe({})
    assert default["data"] == default["default"]
    # An install that predates the rename keeps its old directory; otherwise the new name.
    assert default["data"].endswith((
        "Library/Application Support/ENDEAVOR_LOCAL_AGENT_LITE",
        "Library/Application Support/ENDEAVOR_AGENT_LITE",
    ))

    unsafe = _probe({"LITE_DATA_DIR": str(tmp_path / "outside")})
    assert unsafe["data"] == unsafe["default"]
    assert unsafe["file"].startswith(unsafe["default"])


def test_state_dir_uses_new_name_but_keeps_pre_rename_data(tmp_path):
    sys.path.insert(0, str(PROJECT_ROOT))
    import config

    new_name = config.STATE_DIR_NAME
    old_name = config.PRE_RENAME_STATE_DIR_NAME
    assert (new_name, old_name) == ("ENDEAVOR_LOCAL_AGENT_LITE", "ENDEAVOR_AGENT_LITE")

    # Fresh machine: neither exists -> new name.
    assert config._choose_default_data_dir(tmp_path) == (tmp_path / new_name).resolve()

    # Existing pre-rename data, no new directory -> keep using the old one, move nothing.
    (tmp_path / old_name).mkdir()
    (tmp_path / old_name / "file_index.sqlite3").write_bytes(b"existing index")
    assert config._choose_default_data_dir(tmp_path) == (tmp_path / old_name).resolve()
    assert not (tmp_path / new_name).exists()
    assert (tmp_path / old_name / "file_index.sqlite3").read_bytes() == b"existing index"

    # Once the new directory exists it wins, and the old one is left untouched.
    (tmp_path / new_name).mkdir()
    assert config._choose_default_data_dir(tmp_path) == (tmp_path / new_name).resolve()
    assert (tmp_path / old_name / "file_index.sqlite3").exists()


def test_both_state_names_are_accepted_override_targets():
    import config

    for name in (config.STATE_DIR_NAME, config.PRE_RENAME_STATE_DIR_NAME):
        assert config._state_path_allowed(config.STATE_PARENT_DIR / name / "file_index.sqlite3")
    assert not config._state_path_allowed(config.STATE_PARENT_DIR / "SomethingElse" / "x")


def test_legacy_workspect_override_remains_explicitly_compatible():
    legacy = PROJECT_ROOT / "workspect" / "_test_config_runtime"
    try:
        result = _probe({"LITE_DATA_DIR": str(legacy)})
        assert result["data"] == str(legacy.resolve())
    finally:
        shutil.rmtree(legacy, ignore_errors=True)


def test_network_and_numeric_configuration_fail_closed_to_safe_bounds():
    result = _probe({
        "LITE_SERVER_HOST": "0.0.0.0",
        "LITE_SERVER_PORT": "not-a-port",
        "LITE_UI_PORT": "8766",
        "LITE_MLX_BASE_URL": "https://example.invalid/v1",
        "LITE_MAX_TOKENS": "not-a-number",
    })
    assert result["host"] == "127.0.0.1"
    assert result["port"] == 8090
    assert result["ui"] != result["port"]
    assert result["base"].startswith("http://127.0.0.1:8090/v1")
    assert result["tokens"] == 768


def test_model_base_url_rejects_nested_proxy_paths_and_controlled_keys():
    nested = _probe({"LITE_MLX_BASE_URL": "http://127.0.0.1:8090/proxy/v1?debug=1"})
    assert nested["base"] == "http://127.0.0.1:8090/v1"

    # The probe does not print secrets; importing must still normalize the
    # malformed key without raising a header-construction error later.
    _probe({"LITE_MLX_API_KEY": "bad\nkey"})
