# Security policy

ENDEAVOR AGENT LITE works with local files and launches local processes. Do
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
file creation to `workspect/`, but these are defense-in-depth controls—not a
guarantee that a local agent can safely process every untrusted file.
