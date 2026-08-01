# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Live HTML-WebSocket first-token check; requires UI and model servers."""
from __future__ import annotations

import json
import time

from websockets.sync.client import connect


def main() -> None:
    started = time.perf_counter()
    streamed: list[str] = []
    first_token_at: float | None = None
    final = ""
    types: list[str] = []
    with connect(
        "ws://127.0.0.1:8766/ws",
        origin="http://127.0.0.1:8766",
        open_timeout=5,
        close_timeout=2,
    ) as socket:
        socket.recv(timeout=5)  # status
        socket.recv(timeout=5)  # files
        socket.send(json.dumps({"type": "query", "content": "อธิบายดอกเบี้ยทบต้นสั้นๆ"}))
        while True:
            event = json.loads(socket.recv(timeout=30))
            types.append(event["type"])
            if event["type"] == "token":
                if first_token_at is None:
                    first_token_at = time.perf_counter() - started
                streamed.append(event.get("text", ""))
            elif event["type"] == "discard_stream":
                streamed.clear()
            elif event["type"] == "response":
                final = event["content"]
            elif event["type"] == "done":
                break

    assert first_token_at is not None, types
    assert types.index("token") < types.index("response"), types
    assert "".join(streamed) == final, ("".join(streamed), final)
    print(f"PASS first_token={first_token_at:.3f}s chars={len(final)} events={types}")


if __name__ == "__main__":
    main()
