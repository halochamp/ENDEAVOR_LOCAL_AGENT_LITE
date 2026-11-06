#!/usr/bin/env bash
# Install ENDEAVOR AGENT LITE into a project-local Python 3.11 virtual env.
#
# Usage:
#   bash install_library/install.sh
#
# The script installs dependencies only. The model is downloaded by
# mlx_vlm.server on first launch, so this installer never starts a server or
# consumes model RAM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "=== ENDEAVOR AGENT LITE installer ==="
echo

echo "[1/5] Checking platform"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[error] ENDEAVOR AGENT LITE requires macOS because MLX uses Apple Metal."
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "[error] Apple Silicon is required; detected: $(uname -m)"
  exit 1
fi
echo "macOS Apple Silicon: OK"

echo
echo "[2/5] Checking Python 3.11"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Could not find $PYTHON_BIN. Install Python 3.11, or set PYTHON_BIN to its full path."
  exit 1
fi
if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
then
  echo "[error] $PYTHON_BIN is not Python 3.11: $($PYTHON_BIN --version)"
  exit 1
fi
echo "Using: $($PYTHON_BIN --version)"

echo
echo "[3/5] Building the Apple Vision OCR helper from source"
SWIFTC="$(xcrun --find swiftc 2>/dev/null || true)"
if [[ -z "$SWIFTC" ]]; then
  echo "[warning] Swift compiler not found; skipping optional Apple Vision OCR helper."
  echo "          Install Xcode Command Line Tools and rerun if image OCR is needed:"
  echo "          xcode-select --install"
elif "$SWIFTC" "$PROJECT_DIR/helpers/_vision_ocr.swift" -o "$PROJECT_DIR/helpers/vision_ocr_macos"; then
  chmod 755 "$PROJECT_DIR/helpers/vision_ocr_macos"
  mkdir -p "$PROJECT_DIR/MCP-RagDoc/helpers"
  install -m 755 "$PROJECT_DIR/helpers/vision_ocr_macos" "$PROJECT_DIR/MCP-RagDoc/helpers/vision_ocr_macos"
  echo "Built: helpers/vision_ocr_macos"
  echo "Provisioned: MCP-RagDoc/helpers/vision_ocr_macos"
else
  echo "[warning] Could not build optional Apple Vision OCR helper; continuing without it."
  echo "          Install/fix Xcode Command Line Tools and rerun if image OCR is needed."
fi

echo
echo "[4/5] Creating project virtual environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "Created: $VENV_DIR"
else
  echo "Reusing: $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --require-hashes -r "$PROJECT_DIR/requirements.txt"

echo
echo "[5/5] Installation complete"
cat <<EOF

Next steps:

  Terminal 1 — start the local model server:
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    python -m mlx_vlm.server \\
      --model "\${LITE_MODEL:-mlx-community/Qwen3.5-2B-OptiQ-4bit}" \\
      --port "\${LITE_SERVER_PORT:-8090}" --host "\${LITE_SERVER_HOST:-127.0.0.1}" \\
      --api-key "\${LITE_MLX_API_KEY:-x}" \\
      --prefill-step-size "\${LITE_PREFILL_STEP_SIZE:-512}"

  Terminal 2 — start the CLI agent:
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    python ui_cli.py

  Optional local Web UI:
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    python web_ui.py
    # Then open http://127.0.0.1:8766

The model download starts only on the first mlx_vlm.server launch.
EOF
