# Bug Report — ENDEAVOR_LOCAL_AGENT_LITE

## 2026-07-28 — Web UI disconnect cancellation, localhost CSRF, attachment retention

### Root cause

`_ws_reader()` reported a WebSocket disconnect to the main loop but did not set
the active turn's `cancel_event`, so `agent.run_turn()` continued in its worker
thread after the browser tab closed. State-changing loopback POST routes also
lacked the WebSocket's same-origin check. Finally, private attachment copies
were retained indefinitely.

### Fix

- Set `session.cancel_event` in both disconnect paths.
- Require the known Lite origin on `/upload`, `/server/start`, and `/server/stop`.
- Expire new Lite-managed attachment files after 24 hours and cap their retained
  bytes at 100 MB, running cleanup before and after a new upload is created.
  Older pre-existing attachments are deliberately untouched.

### Validation

Targeted `test_web_ui.py` and `test_web_ui_reveal_path.py` cover disconnect
cancellation, HTTP origin rejection/acceptance, cleanup, workspace boundaries,
and reveal-path routing.

## 2026-08-01 — Core performance, tool routing, web relevance, and large-output stability

### Summary

The 2B agent was structurally safe but still had stochastic tool-routing and
post-tool completion failures: a local document request could jump to web
search or skip RAG, safe Bash pipelines could be rejected and retried, and a
large successful file listing could be truncated and misreported as complete.
The fix moves deterministic policy and evidence ownership into small code
guards while leaving semantic answering to the model. It also completes the
bounded-RAM/streaming work tracked by `AUDIT-LITE-PERF-20260801`,
`AUDIT-LITE-PERF-20260801-B`, and `AUDIT-LITE-TOKEN-CAP-20260801`.

### Symptom

- Repeated live trajectory runs routed `มีเอกสาร PDF กี่ไฟล์ใน Documents`
  through `web_search` three times in 2/3 rounds, then reached the right Bash
  count only after exhausting the web tool budget.
- `find ... | sort` was rejected because `sort` was absent from
  `_READ_ONLY_COMMANDS`, causing an unnecessary second Bash call.
- A 1.67 MB PDF listing exceeded the old 64 KiB capture cap. After the cap was
  raised, only a 2,000-character preview reached the model and the model
  claimed that the 19 visible paths were the complete result (the real count
  was 3,893 in the final verification run).
- The model sometimes repeated an identical successful Bash call. The loop
  guard stopped it correctly but marked the whole user operation aborted.
- Web search could accept topic-adjacent pages and the model could render a URL
  from memory instead of a source actually accepted by code.
- The HTML UI originally delivered only the final response, so users saw no
  text until generation completed.

### Root cause

1. `prompts.SYSTEM_PROMPT` carried the RAG-first rule, but the first tool choice
   still depended on one stochastic 2B inference. Examples improved the mean
   result but could not make a hard routing invariant.
2. `tools/bash.py::_READ_ONLY_COMMANDS` and the prompt advertised slightly
   different capabilities (`wc` and later `sort`). Command arguments were not
   reconciled with the original list/count, document-kind, and whole-machine
   scope after a multi-step RAG fallback.
3. `tools/bash.py::run()` discarded output above 64 KiB, then later exposed a
   preview without machine-readable total-count/full-output metadata. The 2B
   model treated the visible preview as the complete set.
4. `agent.run_turn()` treated every exact duplicate as a failed turn even when
   the first identical search had already succeeded and provided enough
   deterministic evidence to finish.
5. Web relevance and citation correctness were delegated too far downstream:
   lexical similarity alone could retain intent-mismatched results, while URL
   presentation remained model-controlled.
6. Several independent resource paths had no tight bound: WebSocket inbox
   backlog, browser message DOM, document extraction intermediates, index
   subprocess concurrency, and the default MLX vision cache. These do not share
   one cause but all amplify RAM/latency risk on an 8 GB machine.

### Why it produced the symptom

The model received a long system prompt plus six tool schemas on every routing
turn. A wrong first tool added another full inference and another network/tool
round; three wrong web attempts therefore looked like a slow machine even when
the server was healthy. Large Bash output followed the same pattern: the tool
had succeeded, but incomplete evidence sent back to the model triggered either
a fabricated count or an identical retry. Prompt-only fixes could reduce but
not eliminate these paths because the invariant was evaluated anew by the 2B
model on each turn.

### Fix

- `agent._requires_rag_first()` recognizes the general class of local document
  discovery. `run_turn()` performs the cheap local RAG lookup first and starts
  LangGraph from a valid `AIMessage`/`ToolMessage` pair. Explicit Bash, explicit
  web, known absolute file reads, and writes keep their direct routes. This
  removes one model routing call rather than adding another abstraction.
- `tools/bash.py::normalize_command_for_query()` preserves requested document
  kind, whole-machine scope, and list-vs-count intent. `sort` is allowed only as
  stdout processing; `-o`, `--output`, and compressor execution remain blocked.
- Bash capture is bounded at 2 MiB. Oversized context is spilled under private
  Lite state and begins with `[full output: N lines; saved: ...]`.
  `agent.run_turn()` renders that exact count/path in code.
- An exact duplicate after a successful Bash/RAG search stops immediately and
  completes from the first result without setting `aborted`; failed duplicates
  still abort.
- `tools/web_search.py` uses fixed DuckDuckGo/Bing retrieval order, intent and
  freshness guards, bounded retries, and one batched non-thinking YES/NO
  semantic rejection call over at most eight candidates. `agent.py` strips
  model-copied URLs and appends only code-vetted source URLs.
- HTML WebSocket transport now relays first-token events only for the browser,
  batches later token paints to at most 25 fps, and supports discard/cancel.
- Document extraction/read/index paths stream bounded units; WebSocket queues,
  index concurrency, suggestions, history/DOM, tool output, and vision cache
  all have explicit bounds. The MLX launcher uses vision cache size 2.
- `llm.build_llm()` sends the legacy `max_tokens=768` field on the actual MLX
  wire payload and keeps thinking enabled with budget 256 for production tool
  turns.

### How it was found

The debugging loop used three layers: LLM-free unit tests, repeated direct
tool-routing probes, and full ReAct trajectory tests against the resident MLX
server. A one-shot trajectory initially passed 5/6, while three repeats exposed
the `sort`, web-routing, list/count, and successful-duplicate flakes. Direct
Bash output measurement falsified RAM pressure: the 1.67 MB listing completed
in 0.68 seconds with about 90 MB client RSS. Alternating think/no-think calls
also showed that simple routing latency was reasoning-token cost, not hardware
stalling.

### Why it slipped through

The original tests asserted the first tool or a single happy-path execution,
not the complete multi-tool trajectory over repeated model runs. Bash tests
used short outputs and paths without spaces. Web tests checked for returned
results but not intent-specific rejection and code ownership of the displayed
URL. These gaps let prompt-dependent flakes and post-success completion errors
survive despite a green deterministic suite.

### Validation

- Development suite: **252/252 passed** in 40.86 s; maximum RSS 546,734,080
  bytes, peak footprint 399,557,928 bytes, **0 swaps**.
- Live local-search routing probe after web/prompt training: **87/87 votes**.
- Repeated full local-document trajectory: **18/18 passed** (six scenarios ×
  three), with exact tool sequences, valid Bash commands, successful final
  tool state, and no aborted turn.
- A late four-line document list/count self-check was detected after public
  sync. The required post-training regression also passed: **29/29 routing
  cases** and **6/6 full ReAct trajectories** on the final prompt bytes.
- A subsequent no-tool example change was checked with four contrastive unseen
  prompts: two document-organization opinions stayed no-tool, while document
  discovery still chose RAG and explicit Bash still chose Bash (**4/4**).
- Live large PDF listing: one successful Bash call; code reported 3,893 lines
  and the private full-output path instead of the preview count.
- Web relevance: unseen Thai/English holdouts retained relevant results and
  rejected intent mismatches; the end-to-end Thai policy-rate query completed
  with one web tool call and only a vetted supporting URL.
- HTML live transport: first token at 1.276 s, final response at 1.381 s, with
  token events preceding `response`/`done`.
- Public project: **252 passed, 2 expected skips** (optional unbundled macOS OCR
  helper), 46.58 s; maximum RSS 547,766,272 bytes, peak footprint 399,721,696
  bytes, **0 swaps**.
- Graphify: 895 nodes, 1,691 edges, 39 communities, 98% extracted edges, and no
  import cycle. `run_turn()` remains the expected orchestration hub.
- Model process on the Main Mac stayed near 3.0 GB RSS during live testing. The
  actual 8 GB secondary Mac has not been exercised in this session; the
  measured client/model bounds are the acceptance evidence available here.

### Action items / follow-ups

- Run the same `test_bash_document_live.py --repeat 3` and a short image test on
  the target 8 GB Mac after installing its optional OCR helper. Owner: deployer.
- Keep the public OCR tests skipped when the optional Apple Vision helper is
  absent; the runtime intentionally fails closed instead of sending an
  unscanned image to the VLM.
