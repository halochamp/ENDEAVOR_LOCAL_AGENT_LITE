# Security policy

ENDEAVOR LOCAL AGENT LITE works with local files and launches local processes. Do
not report suspected vulnerabilities in a public issue.

Send a private report to **champoomwat@gmail.com** with:

- a clear description and affected version/commit;
- reproducible steps or a minimal proof of concept;
- the potential impact; and
- any suggested mitigation, if available.

Please do not include real credentials, personal files, or destructive payloads
in the report. We will acknowledge a valid report, investigate it privately,
and coordinate disclosure after a fix is available.

## Security boundaries

The Web UI is intentionally bound to `127.0.0.1`; do not expose it to a LAN
or the public internet. The agent blocks common sensitive paths and confines
user-facing file creation to `workspect/`. Runtime indexes/logs live in the
per-user Application Support directory, and the bundled MCP-RagDoc receives
write access only to that private runtime state—not to source documents. These
are defense-in-depth controls, not a guarantee that every untrusted file is safe.
