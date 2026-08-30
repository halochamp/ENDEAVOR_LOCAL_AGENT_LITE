# ENDEAVOR_AGENT_LITE — Agent Procedure

This is the complete repository-agent workflow for the standalone public project.

Document roles:

- `AGENTS.md` — discovery entry point.
- `CLAUDE.md` — hard constraints.
- `AGENT.md` — quick overview.
- **This file** — full execution procedure.
- `README.md` / `CONTRIBUTING.md` / `SECURITY.md` — authoritative product, contributor, and security documentation.

## 1. Start-of-task procedure

Before changing code:

1. Read `CLAUDE.md` and `AGENT.md`.
2. Inspect `git status` and preserve unrelated work.
3. Read the relevant README/CONTRIBUTING/SECURITY section.
4. Classify the change: tool/guardrail, model/generation, indexing/search, document/image handling, Web UI/API, packaging/install, or docs.
5. State one primary invariant and the smallest verification that can falsify the proposed change.
6. Prefer deterministic evidence before using the live local model.

## 2. Architecture boundary

This project is intentionally optimized for a small local model and constrained memory. Do not mechanically port architecture from a larger agent.

Preserve:

- the project's own ReAct/tool-parser flow;
- deterministic helper logic that compensates for model limitations;
- local-only model/UI assumptions;
- workspace confinement;
- lazy loading of heavy dependencies;
- disk-backed/index-backed behavior where designed;
- bounded context and output handling.

Do not add a second cloud LLM or silently route private local content to a cloud model.

## 3. Security procedure

For any path/file/tool/UI change, re-check:

1. Writes remain under the documented `workspect/` boundary.
2. Canonicalization/symlink/traversal cannot escape the intended scope.
3. Sensitive paths and credential-like content remain blocked.
4. Web UI remains bound to `127.0.0.1` unless the security model is explicitly redesigned.
5. Web access sends only the data required by the intended web operation.
6. Logs/errors do not expose secrets or private document content unnecessarily.
7. `.venv/`, workspect state, indexes, logs, screenshots, caches, and real personal documents remain out of Git.

Security regressions require deterministic tests.

## 4. Model and routing procedure

The small production model is part of the product behavior.

When changing prompts, generation parameters, context trimming, chunking, tool schemas, parser behavior, or routing helpers:

1. Identify whether the change can alter first-tool selection or response synthesis.
2. Run deterministic parser/guardrail tests first.
3. When live-model verification is required, use the project's intended production model/runtime rather than a smaller/faster substitute.
4. Run repeated probes for behavior known to be stochastic or margin-sensitive.
5. Compare against an unchanged baseline when changing generation/chunking behavior.
6. Report deterministic and live-model results separately.

Do not call a live-model regression a code regression until deterministic causes are ruled out.

## 5. Resource-efficiency procedure

Low-memory operation is an invariant.

For changes involving imports, OCR/VLM, document parsing, indexing, context, or generation:

- keep heavy imports lazy where the current design is lazy;
- avoid loading model/index/document state earlier than required;
- release or stream large intermediates when practical;
- avoid broad in-memory copies when an existing disk-backed path exists;
- measure/observe memory behavior for changes likely to alter peak RAM;
- do not trade a large permanent RAM increase for a minor convenience without documenting it.

## 6. Tool and filesystem changes

When modifying a tool:

1. Read its caller, validator, and output consumer—not only the tool body.
2. Preserve structured errors and bounded outputs.
3. Verify invalid input is rejected before side effects.
4. Verify path checks occur on canonicalized paths where relevant.
5. Add/update tests for success, refusal, and boundary behavior.
6. Check that model-facing tool descriptions still match runtime behavior.

For document/image readers, test representative small and large inputs without using real private documents.

## 7. Index/search changes

For indexer/search work:

- preserve incremental/state semantics described by the project;
- avoid indexing protected/sensitive locations;
- handle stale/deleted/moved files deterministically;
- keep query/output sizes bounded;
- add regression cases for filenames, contents, Unicode/Thai text, and stale state when relevant;
- do not use a live model to hide retrieval bugs.

## 8. Web UI/API changes

For Web UI or server changes:

- preserve localhost binding;
- preserve authentication/guardrails if present in the current path;
- keep file/reveal operations within documented boundaries;
- avoid sending local file contents to third-party resources by default;
- test server/UI behavior deterministically where possible;
- distinguish browser rendering checks from backend correctness tests.

## 9. Testing procedure

Install/use the project environment documented in `CONTRIBUTING.md`.

Standard deterministic suite:

```bash
python -m pytest test_suite -q
```

`python -m pytest` is also valid under the configured project test paths.

Run targeted tests first for fast feedback, then the full deterministic suite before declaring a code change complete.

Live-model tests are separate. Name which model/runtime was used and which scenarios were exercised. If the required runtime is unavailable, say so rather than substituting another model silently.

## 10. Evidence and compiled artifacts

Only claim code-level verification for source actually available in the repository.

If a referenced behavior is implemented by a compiled `.so`/binary without source:

- do not claim to have inspected its internals;
- do not reverse-engineer or patch the binary as the default workflow;
- reason from documented and observed interfaces;
- state the evidence limitation explicitly.

## 11. Documentation changes

Keep roles separated:

- `AGENT.md` = concise operational overview;
- this file = detailed agent procedure;
- `CLAUDE.md` = hard rules;
- README = user/product documentation;
- CONTRIBUTING = contributor setup/expectations;
- SECURITY = public security boundary/reporting.

Update cross-links when names/headings change. Do not copy private parent-repository workflows into this public repo.

## 12. Git and release hygiene

Before commit/push:

1. inspect status and diff;
2. stage only intended files;
3. confirm no `.env`, tokens, credentials, absolute personal paths, private documents, screenshots, logs, indexes, workspace state, or model artifacts are staged;
4. use a focused commit message;
5. run applicable deterministic tests;
6. never force-push unless explicitly requested and appropriate.

## 13. Completion criteria

A code task is complete when applicable items hold:

- requested behavior is implemented;
- deterministic regression coverage exists for changed boundaries;
- full deterministic suite passes;
- required live-model verification is either completed with the production runtime or explicitly reported unavailable;
- local-first/security/resource invariants remain intact;
- final diff contains only intended changes;
- verification claims match the evidence actually obtained.

**Decision rule:** deterministic code first, production-model evidence when necessary, never use model behavior to excuse a broken guardrail.
