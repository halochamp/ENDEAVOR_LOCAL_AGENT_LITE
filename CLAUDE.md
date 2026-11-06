# ENDEAVOR_AGENT_LITE project rules

- This public repository is self-contained; do not depend on a private parent repository.
- Read [`AGENT.md`](AGENT.md) and [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) before substantial work.
- Preserve the Lite design goal: useful local agent behavior on small Apple Silicon Macs through deterministic code and bounded resource use.
- Keep the Web UI and local model services bound to `127.0.0.1` unless the product security model is deliberately redesigned.
- Keep agent-created user files confined to the documented `workspect/` workspace; private runtime state may use the documented per-user Application Support directory. Preserve path/credential/personal-document guardrails.
- Treat credentials, tokens, private keys, session data, absolute personal paths, screenshots, and real personal documents as sensitive; do not commit them.
- Do not replace the production model or model-routing behavior merely to make tests faster. Separate deterministic tests from live-model tests.
- Preserve lazy loading and low-memory behavior; avoid importing heavy ML dependencies before they are needed.
- Add/update deterministic regression coverage for changed guardrails, tool behavior, indexing, routing helpers, or UI/API behavior.
- Standard deterministic suite: `python -m pytest test_suite -q` (or `python -m pytest` under the project environment).
- Call out live-model tests separately; do not report them as deterministic coverage.
