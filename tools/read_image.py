# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""tools/read_image.py — trimmed from ENDEAVOR_LOCAL_AGENT_MAX_VLM/tools/read_image.py.

MAX_VLM's version (54KB) handles table reconstruction from OCR boxes, zoom/
crop regions, QR decoding, HEIC conversion, EXIF rotation, and (in the MAX/
API forks, not MAX_VLM) a separate small-VLM HTTP call. This version keeps
only what the lite agent's two use cases need:

  1. OCR (Apple Vision, LLM-free, via helpers/vision_ocr.py) — always run,
     since it's fast and gives exact text even when the model would guess.
  2. Direct vision — Qwen3.5-2B-OptiQ-4bit is itself a VLM (same
     direct-vision pattern as MAX_VLM's fork, see that project's CLAUDE.md
     vision-subsystem table), so this tool queues the image path rather than
     calling a separate model. agent.py must pop_pending_images() and pass
     it as the `image=` kwarg on the NEXT llm.generate() call — mirrors
     MAX_VLM's dynamic_prompt() pending-image injection.
"""
from __future__ import annotations
import io
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import config
from helpers.guardrails import truncate_output
from helpers.sensitive_paths import contains_secret_content, path_is_or_contains_secret
from helpers.vision_ocr import read_layout as _ocr_read_layout

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_image",
        # See tools/bash.py's 2026-07-23 note — reverted alongside it.
        "description": (
            "Read an image file: extracts any text in it (OCR) and queues the image "
            "for the vision model to see and describe in its next reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the image file"},
                "region": {"type": "string", "description": "full (default), center, top, bottom, left, right, or a corner"},
                "zoom": {"type": "string", "description": "1-4; use >1 only with a named region to inspect small detail"},
            },
            "required": ["path"],
        },
    },
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff", ".gif", ".webp"}
_PENDING_IMAGES: list[str] = []
VLM_MAX_SIDE = 1024
MAX_SOURCE_IMAGE_PIXELS = 50_000_000
MAX_HEIC_CONVERTED_BYTES = 32 * 1024 * 1024
PENDING_SNAPSHOT_MAX_AGE_SECONDS = 60 * 60
_ZOOM_MAX = 4.0
_REGION_ANCHORS = {
    "center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0),
    "left": (0.0, 0.5), "right": (1.0, 0.5),
    "top-left": (0.0, 0.0), "top-right": (1.0, 0.0),
    "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0),
}


def safe_image_path(raw_path: str) -> Path | None:
    """Return an eligible image path after the shared secret/size policy."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or not path.is_file():
        return None
    try:
        if path.suffix.lower() not in _IMAGE_EXTS or path.stat().st_size > config.READ_IMAGE_MAX_BYTES:
            return None
    except OSError:
        return None
    if path_is_or_contains_secret(path):
        return None
    return path


def read_safe_image_bytes(raw_path: str) -> bytes | None:
    """Revalidate and read at most the configured image limit plus one byte."""
    path = safe_image_path(raw_path)
    if path is None:
        return None
    try:
        with path.open("rb") as fh:
            data = fh.read(config.READ_IMAGE_MAX_BYTES + 1)
    except OSError:
        return None
    return data if len(data) <= config.READ_IMAGE_MAX_BYTES else None


def read_pending_snapshot_bytes(raw_path: str) -> tuple[Path, bytes] | None:
    """Read a Lite-created VLM snapshot, never an arbitrary state file.

    Normal user reads must continue to reject ``.lite_state``.  This narrowly
    scoped transport exception is for files that this module wrote into the
    fixed pending-image directory in the current process.
    """
    try:
        path = Path(raw_path).resolve()
        path.relative_to(config.DATA_DIR / "pending_images")
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            return None
        if path.stat().st_size > config.READ_IMAGE_MAX_BYTES:
            return None
        with path.open("rb") as fh:
            data = fh.read(config.READ_IMAGE_MAX_BYTES + 1)
        return (path, data) if len(data) <= config.READ_IMAGE_MAX_BYTES else None
    except (OSError, ValueError):
        return None


def _ocr_bounded_copy(path: Path, data: bytes) -> list[dict] | None:
    """OCR a checked byte snapshot, never the mutable source path itself."""
    temp_path: Path | None = None
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            prefix="ocr-image-", suffix=path.suffix, dir=config.DATA_DIR, delete=False
        ) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)
        temp_path.chmod(0o600)
        return _ocr_read_layout(str(temp_path))
    except Exception:
        return None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _queue_bounded_snapshot(path: Path, data: bytes, suffix: str | None = None) -> Path | None:
    """Persist the exact OCR-scanned bytes for the next VLM call."""
    try:
        pending_dir = config.DATA_DIR / "pending_images"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_dir.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            prefix="pending-image-", suffix=suffix or path.suffix, dir=pending_dir, delete=False
        ) as tmp:
            tmp.write(data)
            snapshot = Path(tmp.name)
        snapshot.chmod(0o600)
        return snapshot
    except OSError:
        return None


def _prepare_normalized_image(path: Path, data: bytes, region: str, zoom: float):
    """Phase 1 (PLAN.md §9.3): EXIF-upright + native crop only. Returns (PIL
    Image the caller must close, full-resolution PNG bytes for OCR/QR), or
    None on any failure. Tiling/VLM-frame encoding is a separate phase
    (_encode_tiles) so a caller can run OCR/QR -- and bail out on a security
    block -- before ever touching per-tile memory."""
    try:
        from PIL import Image, ImageOps
        source_data = _convert_heic_bytes_if_needed(path, data)
        with Image.open(io.BytesIO(source_data)) as source:
            if source.width * source.height > MAX_SOURCE_IMAGE_PIXELS:
                return None
            image = ImageOps.exif_transpose(source).convert("RGB")
        if region != "full":
            width, height = image.size
            crop_w, crop_h = max(1, round(width / zoom)), max(1, round(height / zoom))
            cx, cy = _REGION_ANCHORS[region]
            left = min(max(0, round(cx * width - crop_w / 2)), max(0, width - crop_w))
            top = min(max(0, round(cy * height - crop_h / 2)), max(0, height - crop_h))
            cropped = image.crop((left, top, left + crop_w, top + crop_h))
            image.close()
            image = cropped
        ocr_buffer = io.BytesIO()
        image.save(ocr_buffer, format="PNG", optimize=True)
        return image, ocr_buffer.getvalue()
    except Exception:
        return None


def _encode_tiles(image) -> list[bytes]:
    """Phase 2: crop, downscale, and PNG-encode one tile at a time.

    The previous implementation built every tile's full-size crop up front
    (via _tile_image) before resizing any of them, so a wide/tall screenshot
    held N full-resolution tile buffers simultaneously alongside the source
    image. Tile boundaries are cheap coordinates (_tile_boxes) -- only one
    tile's actual pixel buffer is ever alive at a time here.
    """
    from PIL import Image
    encoded: list[bytes] = []
    for box in _tile_boxes(*image.size):
        tile = image.crop(box)
        try:
            tile.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            tile.save(output, format="PNG", optimize=True)
            encoded.append(output.getvalue())
        finally:
            tile.close()
    return encoded


def _normalized_frames(path: Path, data: bytes, region: str, zoom: float) -> tuple[list[bytes], bytes] | None:
    """Combined convenience wrapper over _prepare_normalized_image +
    _encode_tiles, kept for callers/tests that just want both outputs from
    one call with the original (pre-§9.3) signature and behavior. The
    production path in run() below calls the two phases separately so it
    can run OCR/QR -- and release full-resolution buffers -- before tiling.
    """
    prepared = _prepare_normalized_image(path, data, region, zoom)
    if prepared is None:
        return None
    image, ocr_data = prepared
    try:
        frames = _encode_tiles(image)
    finally:
        image.close()
    return frames, ocr_data


def _convert_heic_bytes_if_needed(path: Path, data: bytes) -> bytes:
    """Use macOS sips only when Pillow cannot open an iPhone HEIC/HEIF image.

    ``sips`` needs filenames, so its two short-lived files are deliberately
    created below Lite's private ``workspect/.lite_state`` directory.  Normal
    PNG/JPEG paths stay entirely in memory.
    """
    if path.suffix.lower() not in {".heic", ".heif"}:
        return data
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)):
            return data
    except Exception:
        pass
    source: Path | None = None
    output: Path | None = None
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.chmod(0o700)
        with tempfile.NamedTemporaryFile(prefix="heic-source-", suffix=path.suffix,
                                         dir=config.DATA_DIR, delete=False) as tmp:
            tmp.write(data)
            source = Path(tmp.name)
        source.chmod(0o600)
        with tempfile.NamedTemporaryFile(prefix="heic-png-", suffix=".png",
                                         dir=config.DATA_DIR, delete=False) as tmp:
            output = Path(tmp.name)
        output.chmod(0o600)
        probe = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(source)],
                               capture_output=True, text=True, timeout=10)
        dimensions = [int(value) for value in re.findall(r"(?:pixelWidth|pixelHeight):\s*(\d+)",
                                                          probe.stdout or "")]
        if probe.returncode != 0 or len(dimensions) != 2 or dimensions[0] * dimensions[1] > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError("HEIC dimensions exceed safety limit")
        result = subprocess.run(["sips", "-s", "format", "png", str(source), "--out", str(output)],
                                capture_output=True, timeout=20)
        if result.returncode != 0 or output.stat().st_size == 0 or output.stat().st_size > MAX_HEIC_CONVERTED_BYTES:
            raise ValueError("HEIC conversion failed")
        return output.read_bytes()
    finally:
        for temporary in (source, output):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Same tile geometry as before (MAX_VLM behavior, kept byte-identical),
    but returns crop coordinates instead of materialized image copies --
    _encode_tiles crops, downscales, and encodes one of these at a time."""
    long_side, short_side = max(width, height), max(1, min(width, height))
    ratio = long_side / short_side
    if ratio <= 2.5:
        return [(0, 0, width, height)]
    count = max(2, min(3, math.ceil(ratio / 2)))
    vertical = height >= width
    axis = height if vertical else width
    overlap = 0.08
    tile_len = axis / (1 + (count - 1) * (1 - overlap))
    step = tile_len * (1 - overlap)
    boxes = []
    for index in range(count):
        start = round(index * step)
        end = axis if index == count - 1 else round(start + tile_len)
        if index == count - 1:
            start = max(0, end - round(tile_len))
        if vertical:
            boxes.append((0, start, width, end))
        else:
            boxes.append((start, 0, end, height))
    return boxes


def _column_intervals(boxes: list[dict]) -> list[tuple[float, float]]:
    """Merge OCR x-intervals into stable table columns (MAX_VLM algorithm)."""
    intervals = sorted((box["x"], box["x"] + box["w"])
                       for box in boxes if box["w"] <= 0.4)
    merged: list[list[float]] = []
    for left, right in intervals:
        if merged and left <= merged[-1][1] + 0.03:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(left, right) for left, right in merged]


def _assign_column(box: dict, columns: list[tuple[float, float]]) -> int:
    left, right = box["x"], box["x"] + box["w"]
    overlaps = [min(right, hi) - max(left, lo) for lo, hi in columns]
    best = max(range(len(columns)), key=lambda index: overlaps[index])
    if overlaps[best] > 0:
        return best
    center = left + box["w"] / 2
    return min(range(len(columns)), key=lambda index: abs((columns[index][0] + columns[index][1]) / 2 - center))


def _reconstruct_table(boxes: list[dict]) -> str | None:
    """Small port of MAX_VLM's OCR grid reconstruction, with conservative gates."""
    if len(boxes) < 4:
        return None
    ordered = sorted(boxes, key=lambda box: -(box["y"] + box["h"] / 2))
    median_h = sorted(box["h"] for box in boxes)[len(boxes) // 2]
    threshold = max(median_h * 0.6, 0.005)
    rows: list[list[dict]] = [[ordered[0]]]
    for box in ordered[1:]:
        center = box["y"] + box["h"] / 2
        current = sum(item["y"] + item["h"] / 2 for item in rows[-1]) / len(rows[-1])
        if current - center <= threshold:
            rows[-1].append(box)
        else:
            rows.append([box])
    columns = _column_intervals(boxes)
    if len(rows) < 2 or len(columns) < 2:
        return None
    grid = [[""] * len(columns) for _ in rows]
    for row_index, row in enumerate(rows):
        for box in row:
            column = _assign_column(box, columns)
            grid[row_index][column] = (grid[row_index][column] + " " + box["text"]).strip()
    filled = sum(bool(cell) for row in grid for cell in row)
    multi_rows = sum(sum(bool(cell) for cell in row) >= 2 for row in grid)
    if filled < len(rows) * len(columns) * 0.6 or multi_rows < max(2, len(rows) // 2):
        return None
    # Two side-by-side prose blocks are not tables.  Keep grids containing
    # numbers, but reject uniformly long prose cells just as MAX_VLM does.
    flat = [cell for row in grid for cell in row if cell]
    numeric = sum(any(char.isdigit() for char in cell) for cell in flat)
    if flat and numeric / len(flat) < 0.15:
        medians = []
        for index in range(len(columns)):
            lengths = sorted(len(row[index]) for row in grid if row[index])
            if lengths:
                medians.append(lengths[len(lengths) // 2])
        if medians and min(medians) > 80:
            return None
    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |"
    return "\n".join([render(grid[0]), "| " + " | ".join("---" for _ in columns) + " |", *[render(row) for row in grid[1:]]])


def _decode_qr(png_data: bytes) -> list[str] | None:
    try:
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("QR detector could not decode image")
        detector = cv2.QRCodeDetector()
        try:
            ok, values, _points, _straight = detector.detectAndDecodeMulti(image)
            if ok:
                # A detected-but-undecodable QR is not safe to hand to the
                # VLM: it could hold a credential the detector failed to read.
                if any(not value for value in values):
                    return None
                return list(dict.fromkeys(values))
        except Exception:
            return None
        value, _points, _straight = detector.detectAndDecode(image)
        return [value] if value else []
    except Exception:
        return None


def pop_pending_images() -> list[str]:
    """Called by agent.py right before the next llm.generate() — returns and
    clears any images queued by read_image calls this turn."""
    global _PENDING_IMAGES
    imgs, _PENDING_IMAGES = _PENDING_IMAGES, []
    return imgs


def discard_pending_images() -> None:
    """Clear abandoned per-turn VLM snapshots without touching user files."""
    for raw_path in pop_pending_images():
        try:
            path = Path(raw_path).resolve()
            path.relative_to(config.DATA_DIR / "pending_images")
            path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def purge_orphaned_pending_images() -> None:
    """Remove only aged Lite snapshots; never race a second live process."""
    pending_dir = config.DATA_DIR / "pending_images"
    try:
        cutoff = __import__("time").time() - PENDING_SNAPSHOT_MAX_AGE_SECONDS
        for path in pending_dir.iterdir():
            if (path.is_file() and path.name.startswith("pending-image-")
                    and path.stat().st_mtime <= cutoff):
                path.unlink(missing_ok=True)
    except OSError:
        pass


def run(args: dict[str, str]) -> str:
    raw_path = (args.get("path") or "").strip()
    if not raw_path:
        return "[error] path is required"

    region = (args.get("region") or "full").strip().lower()
    try:
        zoom = max(1.0, min(float(args.get("zoom") or 1), _ZOOM_MAX))
    except (TypeError, ValueError):
        zoom = 1.0
    if region != "full" and region not in _REGION_ANCHORS:
        return f"[error] region ต้องเป็น full หรือ: {', '.join(_REGION_ANCHORS)}"
    if zoom > 1 and region == "full":
        return "[error] zoom ต้องระบุ region เช่น top-right หรือ center"

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return f"[error] ต้องเป็น absolute path (หรือขึ้นต้นด้วย ~) ได้ path: '{raw_path}'"
    if not path.exists():
        return f"[error] ไม่พบไฟล์: {path}"
    if path.suffix.lower() not in _IMAGE_EXTS:
        return f"[error] '{path.suffix}' ไม่ใช่นามสกุลรูปภาพที่รองรับ ({', '.join(sorted(_IMAGE_EXTS))})"
    if path_is_or_contains_secret(path):
        return "[blocked] ไม่อนุญาตให้อ่านรูปจาก path ที่อาจมี credential, token, key หรือ session"
    try:
        if path.stat().st_size > config.READ_IMAGE_MAX_BYTES:
            return f"[error] รูปใหญ่เกินขีดจำกัด {config.READ_IMAGE_MAX_BYTES // (1024 * 1024)}MB"
    except OSError as e:
        return f"[error] อ่านข้อมูลรูปไม่ได้: {e}"

    # Snapshot before OCR so a source file replaced after stat() cannot make
    # the Vision subprocess read an unbounded file.
    image_bytes = read_safe_image_bytes(str(path))
    if image_bytes is None:
        return "[blocked] รูปเปลี่ยนแปลงหรือใหญ่เกินขีดจำกัดระหว่างอ่าน"
    # PLAN.md §9.3: normalize/crop, then OCR+QR from the resulting bounded
    # snapshot, THEN release the full-resolution image -- only after all of
    # that (and only if nothing gets blocked) do we spend memory tiling and
    # encoding VLM frames. The previous single-call _normalized_frames did
    # normalize+tile+encode all up front, before OCR/QR ever ran.
    prepared = _prepare_normalized_image(path, image_bytes, region, zoom)
    if prepared is None:
        return "[error] ไม่สามารถ normalize ภาพเพื่ออ่านได้"
    image, ocr_data = prepared
    try:
        ocr_boxes = _ocr_bounded_copy(path.with_suffix(".png"), ocr_data)
        if ocr_boxes is None:
            return "[blocked] OCR ตรวจสอบภาพไม่สำเร็จ จึงไม่ส่งภาพต่อให้ vision model"
        ocr_lines = [box["text"] for box in ocr_boxes]
        ocr_text = "\n".join(ocr_lines)
        if contains_secret_content(ocr_text):
            return "[blocked] OCR พบข้อความที่มีรูปแบบคล้าย credential หรือ token"
        # Decode and classify QR before creating any model snapshot.  A QR can
        # itself contain an access token even when OCR sees no text at all.
        qr_payloads = _decode_qr(ocr_data)
        if qr_payloads is None:
            return "[blocked] QR detector ตรวจสอบภาพไม่สำเร็จ จึงไม่ส่งภาพต่อให้ vision model"
        if any(contains_secret_content(payload) for payload in qr_payloads):
            return "[blocked] QR พบข้อมูลที่มีรูปแบบคล้าย credential หรือ token"
        # Nothing blocked -- only now spend memory tiling/encoding VLM frames.
        vlm_frames = _encode_tiles(image)
    finally:
        image.close()
    # Queue frames derived from the exact OCR-scanned bytes, not the mutable path.
    snapshots = [_queue_bounded_snapshot(path, frame, suffix=".png") for frame in vlm_frames]
    if not snapshots or any(snapshot is None for snapshot in snapshots):
        for snapshot in snapshots:
            if snapshot is not None:
                snapshot.unlink(missing_ok=True)
        return "[error] ไม่สามารถเตรียมรูปสำหรับ vision model ได้"
    _PENDING_IMAGES.extend(str(snapshot) for snapshot in snapshots if snapshot is not None)

    parts = []
    table = _reconstruct_table(ocr_boxes) if ocr_boxes else None
    if table:
        parts.append(f"[TABLE — reconstructed from image OCR]\n{table}")
    elif ocr_lines:
        if len(ocr_text) > config.TOOL_OUTPUT_TRUNCATE_CHARS:
            ocr_text = truncate_output(ocr_text, config.TOOL_OUTPUT_TRUNCATE_CHARS)
        parts.append(f"[OCR] ข้อความในภาพ:\n{ocr_text}")
    else:
        parts.append("[OCR] ไม่พบข้อความในภาพ (หรือ OCR ใช้งานไม่ได้)")
    if qr_payloads:
        parts.append("[QR]\n" + "\n".join(qr_payloads))
    detail = f" region={region} x{zoom:g}" if region != "full" else ""
    tile_note = f" ({len(snapshots)} tiles)" if len(snapshots) > 1 else ""
    parts.append(f"[หมายเหตุ] ภาพ{detail}{tile_note} ถูกส่งเข้า context ของ vision model แล้ว — บรรยาย/ตอบคำถามจากสิ่งที่เห็นในภาพในคำตอบถัดไปได้เลย")
    # Table reconstruction and a QR payload bypass the plain-OCR truncation
    # above, so cap the final envelope too.
    return truncate_output("\n".join(parts), config.TOOL_OUTPUT_TRUNCATE_CHARS)


# LangChain entry point — see tools/bash.py's 2026-07-23 note. pop_pending_images()
# still returns bare file paths (unchanged) — react.py's dynamic_prompt (the
# server-based equivalent of MAX_VLM's own dynamic_prompt) base64-encodes them
# into image_url data URIs when injecting, since mlx_vlm.server's OpenAI-
# compatible endpoint expects "data:image/...;base64,..." (see prompt_utils.py),
# not a bare local path.
# PLAN.md §9.5 docstring-split pattern (2026-07-25) applied here: the
# follow-up zoom guidance only matters AFTER seeing a full-image read's
# result (it governs whether a SECOND call is needed, not the first call's
# construction), so it's deferred into the first full-image call's own
# return value. See tools/bash.py for the original design/verification
# writeup and helpers.turn_guard.call_count().
_ZOOM_FOLLOWUP_NOTE = (
    "\n\n[reminder] Only call read_image AGAIN with a specific region (top, "
    "bottom, left, right, center, or a corner) and zoom=2-4 if this text is "
    "too small/unclear to answer the question. zoom requires a real named "
    "region, never region=\"full\"."
)

from langchain_core.tools import tool as _lc_tool


@_lc_tool
def read_image(path: str, region: str = "full", zoom: str = "1") -> str:
    """Read an image file: extracts any text in it (OCR) and queues the image
    for the vision model to see and describe in its next reply.

    path must be an absolute path. Default to region="full", zoom="1" unless
    the user names a specific area up front.
    ✅ "Read this image" -> read_image(path="/abs/path.png") -- no zoom needed
    ✅ "Read the top-right corner of /Users/me/bill.jpg" -> read_image(path="/Users/me/bill.jpg", region="top-right", zoom="2")"""
    from helpers import turn_guard
    args = {"path": path, "region": region, "zoom": zoom}
    blocked = turn_guard.check("read_image", args)
    if blocked:
        return blocked
    first_full_call_this_turn = region == "full" and turn_guard.call_count("read_image") == 1
    result = run(args)
    turn_guard.record("read_image", args, result)
    if first_full_call_this_turn:
        result += _ZOOM_FOLLOWUP_NOTE
    return result
