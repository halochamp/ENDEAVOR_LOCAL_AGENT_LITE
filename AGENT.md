# ENDEAVOR_AGENT_LITE — Agent Overview

This is the quick operating map for an agent working directly in this public repository.

Read order:

1. [`CLAUDE.md`](CLAUDE.md) — hard constraints.
2. **This file** — project mental model and quick workflow.
3. [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) — full execution procedure.
4. [`README.md`](README.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`SECURITY.md`](SECURITY.md) — product, contributor, and security contracts.

## Core model

ENDEAVOR_AGENT_LITE is a local-first Thai-capable AI agent designed for small Apple Silicon Macs. The implementation intentionally uses deterministic helpers and guardrails to compensate for a small local model.

The main invariants are:

- local model and Web UI stay local by default;
- writes stay under `workspect/`;
- sensitive paths/content stay protected;
- heavy ML dependencies load lazily;
- low-memory behavior is part of correctness;
- tool routing and guardrails are verified deterministically where possible; and
- live-model behavior is tested separately from deterministic code.

## Start every task

Before editing:

1. inspect Git status;
2. read the relevant README/CONTRIBUTING/SECURITY section;
3. identify whether the change affects runtime, tools, indexing, Web UI, model behavior, security, or resource usage;
4. state the smallest verifiable invariant; and
5. prefer a deterministic test before a live-model test.

## Security boundaries

Do not weaken these casually:

- Web UI stays on `127.0.0.1`;
- file writes remain confined to `workspect/`;
- path traversal/symlink and sensitive-path checks remain fail-closed;
- credentials/tokens/private keys/session data/personal documents are not exposed or committed;
- external web access happens only through the intended web tools.

## Resource boundary

Lite is not merely a smaller copy of a large agent. Resource behavior is a product invariant.

When changing imports, prompt construction, model generation, image/document handling, indexing, or context management:

- preserve lazy import behavior;
- avoid holding large intermediates in RAM when disk-backed behavior exists;
- do not change generation/chunking behavior without checking routing stability;
- distinguish deterministic memory/code changes from model-quality changes.

## Testing rule

Default deterministic regression command:

```bash
python -m pytest test_suite -q
```

Live-model tests such as agent/routing probes require the intended local model/runtime and must be reported separately. Do not substitute a different model and call the result production verification.

## Binary/source evidence

Review only source that is actually present. If a behavior depends on a compiled artifact without corresponding source, do not claim an internal audit of that binary; constrain conclusions to its observed interface/behavior.

## Git/release hygiene

Before commit/push:

- inspect status and diff;
- stage only intended files;
- exclude `.venv/`, `workspect/`, indexes, logs, screenshots, caches, model artifacts, and personal documents;
- never commit secrets or machine-specific absolute paths.

Full workflow: [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md).

**Mental model:** small local model + deterministic guardrails + bounded resources + local verification.
