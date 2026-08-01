# Contributing

## Development setup

Use macOS on Apple Silicon with Python 3.11. Run the installer, which builds
the Apple Vision OCR helper locally and creates `.venv`:

```bash
bash install_library/install.sh
source .venv/bin/activate
python -m pytest
```

Do not commit `.venv/`, `workspect/`, local indexes, logs, screenshots, or
real personal documents.

## Change expectations

- Keep the Web UI bound to `127.0.0.1`.
- Treat file paths, credentials, tokens, and personal documents as sensitive.
- Add or update regression coverage for changed guardrails and tool behavior.
- Run the deterministic test suite before opening a pull request. Live-model
  tests require a local MLX server and should be called out separately.
