# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""vision_ocr.py — copied near-verbatim from
ENDEAVOR_LOCAL_AGENT_MAX_VLM/tools/_ocr.py (already lean, ~85 lines, nothing
to trim). Uses the bundled arm64 helper binary; Lite never invokes a compiler at
runtime, so OCR cannot create compiler cache files outside workspect. Only
text extraction and the Vision bounding boxes needed for lightweight table
reconstruction are exposed.
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_BIN = Path(__file__).resolve().parent / "vision_ocr_macos"


def _ensure_binary() -> bool:
    return _BIN.is_file() and _BIN.stat().st_mode & 0o111 != 0


def read_layout(image_path: str) -> list[dict]:
    """Apple Vision OCR lines with normalized Vision bounding boxes.

    An unavailable helper, subprocess timeout, malformed helper response, or
    an explicit helper error is a *detector failure*, not the same thing as a
    valid image containing no text.  Callers that could expose the image to a
    VLM must fail closed, so those cases raise ``RuntimeError``.
    """
    if not image_path or not _ensure_binary():
        raise RuntimeError("Apple Vision OCR helper is unavailable")
    try:
        out = subprocess.run([str(_BIN), image_path], capture_output=True, text=True, timeout=15)
        raw = (out.stdout or "").strip()
        if out.returncode != 0 or raw.startswith("ERR"):
            raise RuntimeError("Apple Vision OCR helper failed")
        if not raw:
            return []
        boxes = []
        for line in raw.split("\n"):
            parts = line.split("\t", 4)
            if len(parts) != 5 or not parts[4].strip():
                raise RuntimeError("Apple Vision OCR helper returned malformed output")
            try:
                x, y, w, h = (float(parts[i]) for i in range(4))
            except ValueError:
                raise RuntimeError("Apple Vision OCR helper returned malformed output")
            boxes.append({"text": parts[4].strip(), "x": x, "y": y, "w": w, "h": h})
        return boxes
    except Exception as e:
        log.debug(f"[vision_ocr] read error: {e}")
        raise RuntimeError("Apple Vision OCR failed") from e


def read_text(image_path: str) -> list[str]:
    """Plain-text view of :func:`read_layout`."""
    return [box["text"] for box in read_layout(image_path)]
