"""tools/read_image.py — Lite-compatible fork of MAX_VLM's progressive read_image.

The Lite fork keeps MAX_VLM's important vision semantics while preserving Lite's
smaller/saner runtime boundary:
- local absolute image paths only (no URL/screen expansion)
- overview pixels first; OCR/table/QR/detail sensors are explicit follow-ups
- batch overview/OCR for at most four images, with per-image OCR selection
- native-resolution find/zoom plus chart/slide structural crops
- transient image snapshots stay active for the whole outer ReAct turn
- hidden OCR/QR DLP preflight still runs before any pixels reach the local VLM

The hidden DLP preflight is a Lite safety invariant. Its text is not exposed on
an overview read; visible OCR remains an explicit detail="text" sensor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import io
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Annotated

from pydantic import Field

import config
from helpers.guardrails import truncate_output
from helpers.sensitive_paths import contains_secret_content, path_is_or_contains_secret
from helpers.thai_text_normalize import repair_thai_orthographic_order
from helpers.vision_ocr import read_layout as _ocr_read_layout

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_image",
        "description": (
            "Progressive direct-vision reader. First read a source with detail='overview' "
            "so the model sees original pixels. Then use detail='text' for OCR/table/QR, "
            "detail='chart'/'slide' for structural crops, find='exact text', or named "
            "region+zoom. path may be one absolute image path or a batch of 1-4 paths. "
            "Batch mode is overview-only and can select OCR with ocr=true or 1-based "
            "positions such as ocr=[1,3]; for new images, originals are attached first "
            "and requested OCR is deferred to a later batch call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                    ],
                    "description": "Absolute local image path, or 1-4 distinct absolute paths",
                },
                "region": {
                    "type": "string",
                    "description": "full (default), center, top, bottom, left, right, or a corner",
                },
                "zoom": {
                    "oneOf": [{"type": "number"}, {"type": "string"}],
                    "description": "1-4; zoom >1 requires a named region",
                },
                "find": {
                    "type": "string",
                    "description": "Exact/substring text to locate with OCR and crop at native resolution",
                },
                "detail": {
                    "type": "string",
                    "description": "overview (required first), text, chart, or slide",
                },
                "ocr": {
                    "oneOf": [
                        {"type": "boolean"},
                        {"type": "array", "items": {"type": "integer"}, "maxItems": 4},
                        {"type": "string"},
                    ],
                    "description": "Batch only: false, true, or 1-based image positions, e.g. [1,3]",
                },
            },
            "required": ["path"],
        },
    },
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff", ".gif", ".webp"}
_DETAIL_MODES = {"overview", "text", "chart", "slide"}
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
# User decision 2026-09-09: Lite batch is deliberately smaller than MAX_VLM.
_READ_IMAGE_BATCH_MAX = 4
_OCR_BATCH_MAX = 4
# Structural follow-ups can add four crops to one source, so keep the same
# bounded outer-turn attachment capacity while limiting a single batch to four.
_IMAGE_TURN_MAX = 10
_READ_ATTEMPT_MAX = 6
_ZOOM_ATTEMPT_MAX = 6

_PENDING_IMAGES: list[str] = []
_ACTIVE_TURN_IMAGES: list[str] = []
_OVERVIEW_SEEN: set[str] = set()
_READ_SEEN: dict[tuple[str, str], str] = {}
_READ_ATTEMPTS: dict[str, int] = {}
_ZOOM_SEEN: set[tuple[str, str, float]] = set()
_ZOOM_ATTEMPTS = [0]
_IMAGE_TURN_COUNT = [0]
_READ_IMAGE_CALL_LOCK = threading.RLock()


def _serialize_read_image(fn):
    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _READ_IMAGE_CALL_LOCK:
            return fn(*args, **kwargs)
    return _locked


def _private_snapshot(path: Path) -> bool:
    try:
        path.resolve().relative_to((config.DATA_DIR / "pending_images").resolve())
        return path.name.startswith("pending-image-")
    except (OSError, ValueError):
        return False


def _delete_snapshot(raw_path: str) -> None:
    try:
        path = Path(raw_path)
        if _private_snapshot(path):
            path.unlink(missing_ok=True)
    except OSError:
        pass


def reset_read_guards() -> None:
    """Reset progressive/loop guards at the real outer-turn boundary."""
    _OVERVIEW_SEEN.clear()
    _READ_SEEN.clear()
    _READ_ATTEMPTS.clear()
    _ZOOM_SEEN.clear()
    _ZOOM_ATTEMPTS[0] = 0
    _IMAGE_TURN_COUNT[0] = 0


def begin_image_turn() -> int:
    """Drop any stale transient snapshots and start a fresh vision turn."""
    stale = len(_PENDING_IMAGES) + len(_ACTIVE_TURN_IMAGES)
    for raw in [*_PENDING_IMAGES, *_ACTIVE_TURN_IMAGES]:
        _delete_snapshot(raw)
    _PENDING_IMAGES.clear()
    _ACTIVE_TURN_IMAGES.clear()
    reset_read_guards()
    return stale


def active_turn_images() -> list[str]:
    """Promote new images and keep all current-turn pixels for every ReAct step."""
    if _PENDING_IMAGES:
        _ACTIVE_TURN_IMAGES.extend(_PENDING_IMAGES)
        _PENDING_IMAGES.clear()
    return list(_ACTIVE_TURN_IMAGES)


def end_image_turn() -> None:
    """Release all private snapshots owned by the completed outer turn."""
    for raw in [*_PENDING_IMAGES, *_ACTIVE_TURN_IMAGES]:
        _delete_snapshot(raw)
    _PENDING_IMAGES.clear()
    _ACTIVE_TURN_IMAGES.clear()
    reset_read_guards()


def pop_pending_images() -> list[str]:
    """Compatibility drain; normal runtime uses active_turn_images()."""
    images = list(_PENDING_IMAGES)
    _PENDING_IMAGES.clear()
    return images


def discard_pending_images() -> None:
    """Delete unpromoted snapshots only; kept for older callers/tests."""
    for raw in pop_pending_images():
        _delete_snapshot(raw)


def purge_orphaned_pending_images() -> None:
    """Delete only aged Lite-created snapshots, never arbitrary state files."""
    pending_dir = config.DATA_DIR / "pending_images"
    try:
        cutoff = time.time() - PENDING_SNAPSHOT_MAX_AGE_SECONDS
        for path in pending_dir.iterdir():
            if (
                path.is_file()
                and path.name.startswith("pending-image-")
                and path.stat().st_mtime <= cutoff
            ):
                path.unlink(missing_ok=True)
    except OSError:
        pass


def _image_turn_slots() -> int:
    return max(0, _IMAGE_TURN_MAX - len(_PENDING_IMAGES) - len(_ACTIVE_TURN_IMAGES))


def safe_image_path(raw_path: str) -> Path | None:
    """Return a local image after Lite's path/extension/size/DLP boundary."""
    try:
        path = Path(raw_path).expanduser()
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            return None
        if path.suffix.lower() not in _IMAGE_EXTS:
            return None
        if path.stat().st_size > config.READ_IMAGE_MAX_BYTES:
            return None
        if path_is_or_contains_secret(path):
            return None
        return path
    except OSError:
        return None


def read_safe_image_bytes(raw_path: str) -> bytes | None:
    path = safe_image_path(raw_path)
    if path is None:
        return None
    try:
        with path.open("rb") as handle:
            data = handle.read(config.READ_IMAGE_MAX_BYTES + 1)
    except OSError:
        return None
    return data if len(data) <= config.READ_IMAGE_MAX_BYTES else None


def read_pending_snapshot_bytes(raw_path: str) -> tuple[Path, bytes] | None:
    """Read only Lite-owned current-turn snapshots for react.py transport."""
    try:
        path = Path(raw_path).resolve()
        if not _private_snapshot(path) or not path.is_file():
            return None
        if path.stat().st_size > config.READ_IMAGE_MAX_BYTES:
            return None
        with path.open("rb") as handle:
            data = handle.read(config.READ_IMAGE_MAX_BYTES + 1)
        return (path, data) if len(data) <= config.READ_IMAGE_MAX_BYTES else None
    except (OSError, ValueError):
        return None


def _queue_bounded_snapshot(path: Path, data: bytes, suffix: str = ".png") -> Path | None:
    try:
        pending_dir = config.DATA_DIR / "pending_images"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_dir.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            prefix="pending-image-", suffix=suffix, dir=pending_dir, delete=False
        ) as tmp:
            tmp.write(data)
            snapshot = Path(tmp.name)
        snapshot.chmod(0o600)
        return snapshot
    except OSError:
        return None


def _queue_encoded_bundle(path: Path, frames: list[bytes]) -> bool:
    """Queue one source's views atomically within the outer-turn image budget."""
    if not frames or len(frames) > _image_turn_slots():
        return False
    created: list[Path] = []
    for frame in frames:
        snapshot = _queue_bounded_snapshot(path, frame)
        if snapshot is None:
            for item in created:
                item.unlink(missing_ok=True)
            return False
        created.append(snapshot)
    _PENDING_IMAGES.extend(str(item) for item in created)
    return True


def _vision_budget_notice(needed: int) -> str:
    used = _IMAGE_TURN_MAX - _image_turn_slots()
    return (
        "[DIRECT VISION DEFERRED — "
        f"turn นี้มีภาพ {used}/{_IMAGE_TURN_MAX} views แล้ว และชุดนี้ต้องใช้ {needed} view(s). "
        "ตอบเฉพาะจากภาพที่เห็นแล้ว หรือให้ผู้ใช้ส่งภาพนี้ใหม่ใน turn ถัดไป]"
    )


def _convert_heic_bytes_if_needed(path: Path, data: bytes) -> bytes:
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
        with tempfile.NamedTemporaryFile(
            prefix="heic-source-", suffix=path.suffix, dir=config.DATA_DIR, delete=False
        ) as tmp:
            tmp.write(data)
            source = Path(tmp.name)
        source.chmod(0o600)
        with tempfile.NamedTemporaryFile(
            prefix="heic-png-", suffix=".png", dir=config.DATA_DIR, delete=False
        ) as tmp:
            output = Path(tmp.name)
        output.chmod(0o600)
        probe = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(source)],
            capture_output=True, text=True, timeout=10,
        )
        dimensions = [
            int(value)
            for value in re.findall(
                r"(?:pixelWidth|pixelHeight):\s*(\d+)", probe.stdout or ""
            )
        ]
        if (
            probe.returncode != 0
            or len(dimensions) != 2
            or dimensions[0] * dimensions[1] > MAX_SOURCE_IMAGE_PIXELS
        ):
            raise ValueError("HEIC dimensions exceed safety limit")
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(source), "--out", str(output)],
            capture_output=True, timeout=20,
        )
        if (
            result.returncode != 0
            or output.stat().st_size == 0
            or output.stat().st_size > MAX_HEIC_CONVERTED_BYTES
        ):
            raise ValueError("HEIC conversion failed")
        return output.read_bytes()
    finally:
        for temporary in (source, output):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _prepare_normalized_image(path: Path, data: bytes, region: str, zoom: float):
    """EXIF-upright image plus full-resolution PNG bytes for deterministic sensors."""
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
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return image, buffer.getvalue()
    except Exception:
        return None


def _encode_one(image) -> bytes:
    """Aspect-preserving, downscale-only PNG for one direct-vision view."""
    from PIL import Image
    view = image.copy()
    try:
        view.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        view.save(output, format="PNG", optimize=True)
        return output.getvalue()
    finally:
        view.close()


def _tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Compatibility helper retained for Lite's bounded-RAM tiling regression."""
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
        boxes.append((0, start, width, end) if vertical else (start, 0, end, height))
    return boxes


def _encode_tiles(image) -> list[bytes]:
    """Compatibility helper: materialize/encode only one long-image tile at a time."""
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
    prepared = _prepare_normalized_image(path, data, region, zoom)
    if prepared is None:
        return None
    image, ocr_data = prepared
    try:
        return _encode_tiles(image), ocr_data
    finally:
        image.close()


def _ocr_bounded_copy(path: Path, data: bytes) -> list[dict] | None:
    """OCR a private checked copy, never the mutable user path."""
    temp_path: Path | None = None
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            prefix="ocr-image-", suffix=".png", dir=config.DATA_DIR, delete=False
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


def _enhance_png_bytes(png_data: bytes) -> bytes | None:
    """MAX_VLM-style one-shot CLAHE OCR retry without retaining native buffers."""
    try:
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        h, w = enhanced.shape[:2]
        if max(h, w) < 1200:
            enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", enhanced)
        return buf.tobytes() if ok else None
    except Exception:
        return None


def _repair_boxes(boxes: list[dict]) -> list[dict]:
    for box in boxes:
        box["text"] = repair_thai_orthographic_order(str(box.get("text", "")))
    return boxes


def _decode_qr(png_data: bytes) -> list[str] | None:
    """Fail closed when a detected QR cannot be decoded completely."""
    try:
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        detector = cv2.QRCodeDetector()
        try:
            ok, values, _points, _straight = detector.detectAndDecodeMulti(image)
            if ok:
                if any(not value for value in values):
                    return None
                return list(dict.fromkeys(values))
        except Exception:
            return None
        value, _points, _straight = detector.detectAndDecode(image)
        return [value] if value else []
    except Exception:
        return None


def _sensor_scan(path: Path, png_data: bytes, *, retry_empty: bool) -> tuple[list[dict] | None, list[str] | None, bool]:
    boxes = _ocr_bounded_copy(path, png_data)
    enhanced = False
    if boxes == [] and retry_empty:
        retry = _enhance_png_bytes(png_data)
        if retry:
            retried = _ocr_bounded_copy(path, retry)
            if retried:
                boxes = retried
                enhanced = True
    if boxes is not None:
        boxes = _repair_boxes(boxes)
    qr = _decode_qr(png_data)
    return boxes, qr, enhanced


def _sensor_security_error(boxes: list[dict] | None, qr: list[str] | None) -> str:
    if boxes is None:
        return "[blocked] OCR ตรวจสอบภาพไม่สำเร็จ จึงไม่ส่งภาพต่อให้ vision model"
    text = "\n".join(str(box.get("text", "")) for box in boxes)
    if contains_secret_content(text):
        return "[blocked] OCR พบข้อความที่มีรูปแบบคล้าย credential หรือ token"
    if qr is None:
        return "[blocked] QR detector ตรวจสอบภาพไม่สำเร็จ จึงไม่ส่งภาพต่อให้ vision model"
    if any(contains_secret_content(payload) for payload in qr):
        return "[blocked] QR พบข้อมูลที่มีรูปแบบคล้าย credential หรือ token"
    return ""


def _column_intervals(boxes: list[dict], min_gap: float = 0.025) -> list[tuple[float, float]]:
    spans = sorted(
        ((box["x"], box["x"] + box["w"]) for box in boxes if box["w"] <= 0.4),
        key=lambda value: value[0],
    )
    if not spans:
        return []
    columns: list[list[float]] = [list(spans[0])]
    for left, right in spans[1:]:
        if left - columns[-1][1] <= min_gap:
            columns[-1][1] = max(columns[-1][1], right)
        else:
            columns.append([left, right])
    return [(left, right) for left, right in columns]


def _assign_column(box: dict, columns: list[tuple[float, float]]) -> int:
    left, right = box["x"], box["x"] + box["w"]
    overlaps = [min(right, hi) - max(left, lo) for lo, hi in columns]
    best = max(range(len(columns)), key=lambda index: overlaps[index])
    if overlaps[best] > 0:
        return best
    center = left + box["w"] / 2
    return min(
        range(len(columns)),
        key=lambda index: abs((columns[index][0] + columns[index][1]) / 2 - center),
    )


def _reconstruct_table(boxes: list[dict]) -> str | None:
    """MAX_VLM conservative OCR grid reconstruction."""
    if len(boxes) < 4:
        return None
    ordered = sorted(boxes, key=lambda box: -(box["y"] + box["h"] / 2))
    heights = sorted(box["h"] for box in boxes)
    median_h = heights[len(heights) // 2]
    threshold = max(median_h * 0.6, 0.005)
    rows: list[list[dict]] = [[ordered[0]]]
    for box in ordered[1:]:
        center = box["y"] + box["h"] / 2
        row_center = sum(item["y"] + item["h"] / 2 for item in rows[-1]) / len(rows[-1])
        if row_center - center <= threshold:
            rows[-1].append(box)
        else:
            rows.append([box])
    columns = _column_intervals(boxes)
    if len(rows) < 2 or len(columns) < 2:
        return None
    grid = [[""] * len(columns) for _ in rows]
    for row_index, row in enumerate(rows):
        for box in sorted(row, key=lambda item: item["x"]):
            column = _assign_column(box, columns)
            grid[row_index][column] = (grid[row_index][column] + " " + box["text"]).strip()
    multi_rows = sum(sum(bool(cell) for cell in row) >= 2 for row in grid)
    if multi_rows < max(2, len(grid) * 0.5):
        return None
    filled = sum(bool(cell) for row in grid for cell in row)
    if filled < len(grid) * len(columns) * 0.6:
        return None
    flat = [cell for row in grid for cell in row if cell]
    numeric = sum(any(char.isdigit() for char in cell) for cell in flat)
    if flat and numeric / len(flat) < 0.15:
        medians = []
        for index in range(len(columns)):
            lengths = sorted(len(row[index]) for row in grid if row[index])
            if lengths:
                medians.append(lengths[len(lengths) // 2])
        if medians and min(medians) > 14:
            return None

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |"

    return "\n".join([
        render(grid[0]),
        "| " + " | ".join("---" for _ in columns) + " |",
        *[render(row) for row in grid[1:]],
    ])


def _locate_find_box(boxes: list[dict], find: str) -> dict | None:
    needle = find.casefold()
    matches = [
        box for box in boxes
        if needle in str(box.get("text", "")).casefold()
    ]
    return min(matches, key=lambda box: box["h"]) if matches else None


def _crop_to_find_image(image, box: dict):
    width, height = image.size
    top = 1.0 - (box["y"] + box["h"])
    bottom = 1.0 - box["y"]
    left = box["x"]
    right = box["x"] + box["w"]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    win_w = max(right - left, 0.25)
    win_h = max(bottom - top, 0.25)
    x0, x1 = cx - win_w / 2, cx + win_w / 2
    y0, y1 = cy - win_h / 2, cy + win_h / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > 1:
        x0 -= x1 - 1
        x1 = 1
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > 1:
        y0 -= y1 - 1
        y1 = 1
    bounds = (
        max(0, round(max(0.0, x0) * width)),
        max(0, round(max(0.0, y0) * height)),
        min(width, round(min(1.0, x1) * width)),
        min(height, round(min(1.0, y1) * height)),
    )
    return image.crop(bounds)


def _png_bytes(image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _chart_detail_images(image) -> list[tuple[str, object]]:
    """MAX_VLM generic four-view chart detail geometry using PIL crops."""
    width, height = image.size

    def crop(x0: float, y0: float, x1: float, y1: float):
        return image.crop((
            max(0, round(x0 * width)), max(0, round(y0 * height)),
            min(width, round(x1 * width)), min(height, round(y1 * height)),
        ))

    if width >= height:
        return [
            ("left panel/detail", crop(0.03, 0.10, 0.55, 0.91)),
            ("right panel/detail", crop(0.47, 0.10, 0.99, 0.91)),
            ("bottom-left legend/source", crop(0.03, 0.82, 0.56, 0.995)),
            ("bottom-right legend/source", crop(0.44, 0.82, 0.99, 0.995)),
        ]
    return [
        ("top panel/detail", crop(0.03, 0.05, 0.97, 0.54)),
        ("bottom panel/detail", crop(0.03, 0.43, 0.97, 0.91)),
        ("bottom-left legend/source", crop(0.03, 0.82, 0.56, 0.995)),
        ("bottom-right legend/source", crop(0.44, 0.82, 0.99, 0.995)),
    ]


def _grid_detail_images(image) -> list[tuple[str, object]]:
    width, height = image.size

    def crop(x0: float, y0: float, x1: float, y1: float):
        return image.crop((
            max(0, round(x0 * width)), max(0, round(y0 * height)),
            min(width, round(x1 * width)), min(height, round(y1 * height)),
        ))

    return [
        ("left table columns", crop(0.02, 0.08, 0.40, 0.88)),
        ("center table columns", crop(0.31, 0.08, 0.70, 0.88)),
        ("right/latest table columns", crop(0.61, 0.08, 0.995, 0.88)),
        ("source/definitions footer", crop(0.03, 0.82, 0.995, 0.995)),
    ]


def _encode_detail_bundle(items: list[tuple[str, object]]) -> tuple[list[str], list[bytes]]:
    labels: list[str] = []
    frames: list[bytes] = []
    for label, image in items:
        try:
            labels.append(label)
            frames.append(_encode_one(image))
        finally:
            image.close()
    return labels, frames


def _ocr_label_assist(boxes: list[dict], limit: int = 60) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for box in sorted(boxes, key=lambda item: (-(item["y"] + item["h"] / 2), item["x"])):
        text = " ".join(str(box.get("text", "")).split())
        if not text or any(char.isdigit() for char in text) or text in seen:
            continue
        seen.add(text)
        labels.append(text[:120])
        if len(labels) >= limit:
            break
    return "\n".join(labels)


def _format_text_sensor(boxes: list[dict], qr: list[str], enhanced: bool) -> str:
    ocr_text = "\n".join(str(box.get("text", "")) for box in boxes).strip()
    table = _reconstruct_table(boxes) if ocr_text else None
    sections: list[str] = []
    if table:
        sections.append(
            "[OCR ASSIST — reconstructed table candidate; pixels are primary]\n" + table
        )
    elif ocr_text:
        tag = (
            "[OCR ASSIST — enhanced transcription; pixels are primary]"
            if enhanced
            else "[OCR ASSIST — transcription hint; pixels are primary]"
        )
        sections.append(f"{tag}\n{ocr_text}")
    else:
        sections.append("[OCR ASSIST — no text found; original image remains visible]")
    if qr:
        sections.append("[QR]\n" + "\n".join(qr))
    return truncate_output("\n\n".join(sections), config.TOOL_OUTPUT_TRUNCATE_CHARS)


def _coerce_ocr_selector(ocr):
    if isinstance(ocr, str):
        text = ocr.strip()
        if text.casefold() == "true":
            return True
        if text.casefold() in {"false", ""}:
            return False
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, list):
                return decoded
    return ocr


def _normalise_batch_ocr(ocr, count: int) -> tuple[set[int], str]:
    ocr = _coerce_ocr_selector(ocr)
    if ocr is False or ocr is None:
        return set(), ""
    if ocr is True:
        return set(range(count)), ""
    if not isinstance(ocr, list):
        return set(), "[error] ocr ต้องเป็น false, true หรือ list หมายเลขภาพแบบ 1-based"
    if len(ocr) > _OCR_BATCH_MAX:
        return set(), f"[error] OCR เลือกได้สูงสุด {_OCR_BATCH_MAX} ภาพ"
    selected: set[int] = set()
    for raw in ocr:
        if isinstance(raw, bool) or not isinstance(raw, int):
            return set(), "[error] หมายเลขภาพ OCR ต้องเป็น integer"
        if raw < 1 or raw > count:
            return set(), f"[error] หมายเลขภาพ OCR {raw} อยู่นอกช่วง 1-{count}"
        selected.add(raw - 1)
    return selected, ""


def _resolve_source(src: str) -> tuple[Path | None, bytes | None, str]:
    if not isinstance(src, str) or not src.strip():
        return None, None, "[error] path ของภาพต้องเป็น string ที่ไม่ว่าง"
    raw = src.strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return None, None, f"[error] ต้องเป็น absolute path (หรือขึ้นต้นด้วย ~) ได้ path: '{raw}'"
    if not path.exists():
        return None, None, f"[error] ไม่พบไฟล์: {path}"
    if path.suffix.lower() not in _IMAGE_EXTS:
        return None, None, f"[error] '{path.suffix}' ไม่ใช่นามสกุลรูปภาพที่รองรับ"
    if path_is_or_contains_secret(path):
        return None, None, "[blocked] ไม่อนุญาตให้อ่านรูปจาก path ที่อาจมี credential, token, key หรือ session"
    data = read_safe_image_bytes(str(path))
    if data is None:
        return None, None, "[blocked] รูปเปลี่ยนแปลงหรือใหญ่เกินขีดจำกัดระหว่างอ่าน"
    return path, data, ""


def _prepare_batch_source(src: str, *, overview: bool, visible_ocr: bool) -> dict:
    path, data, error = _resolve_source(src)
    if error:
        return {"error": error}
    assert path is not None and data is not None
    prepared = _prepare_normalized_image(path, data, "full", 1.0)
    if prepared is None:
        return {"error": "[error] ไม่สามารถ normalize ภาพเพื่ออ่านได้"}
    image, png_data = prepared
    try:
        boxes, qr, enhanced = _sensor_scan(path, png_data, retry_empty=visible_ocr)
        security = _sensor_security_error(boxes, qr)
        if security:
            return {"error": security}
        result: dict = {"path": path}
        if overview:
            result["frame"] = _encode_one(image)
        if visible_ocr:
            result["ocr"] = _format_text_sensor(boxes or [], qr or [], enhanced)
        return result
    finally:
        image.close()


def _read_image_batch(paths: list[str], ocr) -> str:
    if not paths:
        return "[error] batch read_image ว่าง"
    if len(paths) > _READ_IMAGE_BATCH_MAX:
        return f"[error] read_image ของ Lite รับ batch ได้สูงสุด {_READ_IMAGE_BATCH_MAX} ภาพ"

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            return "[error] ทุก path ใน batch ต้องเป็น string ที่ไม่ว่าง"
        src = raw.strip()
        if src in seen:
            return f"[error] มีภาพซ้ำใน batch: {src}"
        seen.add(src)
        cleaned.append(src)

    selected, error = _normalise_batch_ocr(ocr, len(cleaned))
    if error:
        return error

    unseen = [src for src in cleaned if src not in _OVERVIEW_SEEN]
    if len(unseen) > _image_turn_slots():
        return _vision_budget_notice(len(unseen))

    specs: list[tuple[int, str, bool, bool, bool]] = []
    results: list[str | None] = [None] * len(cleaned)
    for index, src in enumerate(cleaned):
        need_overview = src not in _OVERVIEW_SEEN
        wants_ocr = index in selected
        # Progressive contract: visible OCR is legal only after an overview is active.
        visible_ocr = wants_ocr and not need_overview
        deferred_ocr = wants_ocr and need_overview
        if not need_overview and not visible_ocr:
            results[index] = "[original image already active in direct-vision context]"
            continue
        specs.append((index, src, need_overview, visible_ocr, deferred_ocr))

    worker_results: dict[int, dict] = {}
    if specs:
        with ThreadPoolExecutor(max_workers=min(_READ_IMAGE_BATCH_MAX, len(specs))) as pool:
            futures = [
                pool.submit(
                    _prepare_batch_source,
                    src,
                    overview=need_overview,
                    visible_ocr=visible_ocr,
                )
                for _index, src, need_overview, visible_ocr, _deferred in specs
            ]
            for spec, future in zip(specs, futures):
                worker_results[spec[0]] = future.result()

    for index, src, need_overview, visible_ocr, deferred_ocr in specs:
        worker = worker_results[index]
        if worker.get("error"):
            results[index] = str(worker["error"])
            continue
        _READ_ATTEMPTS[src] = _READ_ATTEMPTS.get(src, 0) + 1
        pieces: list[str] = []
        if need_overview:
            frame = worker.get("frame")
            queued = isinstance(frame, (bytes, bytearray)) and _queue_encoded_bundle(
                worker["path"], [bytes(frame)]
            )
            if queued:
                _OVERVIEW_SEEN.add(src)
                _IMAGE_TURN_COUNT[0] += 1
                overview = "[original image queued for direct vision — inspect pixels first]"
                _READ_SEEN[(src, "overview")] = overview
                pieces.append(overview)
            else:
                pieces.append(_vision_budget_notice(1))
        if visible_ocr:
            sensor = str(worker.get("ocr") or "[OCR ASSIST — no result]")
            _READ_SEEN[(src, "text")] = sensor
            pieces.append(sensor)
        if deferred_ocr:
            pieces.append(
                "[OCR deferred — original pixels were attached first. "
                "After inspecting them, call the same batch again with the desired ocr selection.]"
            )
        results[index] = "\n".join(pieces)

    return truncate_output(
        "\n\n".join(
            f"[image:{index + 1} {src}]\n{results[index] or '[error] no result'}"
            for index, src in enumerate(cleaned)
        ),
        max(config.TOOL_OUTPUT_TRUNCATE_CHARS, 4000),
    )


def _serve_overview(src: str, deferred: str = "") -> str:
    read_sig = (src, "overview")
    if read_sig in _READ_SEEN:
        return (
            "[อ่านซ้ำ] overview ของภาพนี้ถูกส่งเข้า context แล้ว — "
            "ใช้ detail='text'/'chart'/'slide', find หรือ region+zoom ถ้าต้องการรายละเอียดเพิ่ม"
        )
    if _READ_ATTEMPTS.get(src, 0) >= _READ_ATTEMPT_MAX:
        return "[error] อ่านภาพนี้ครบจำนวนสูงสุดใน turn แล้ว"

    path, data, error = _resolve_source(src)
    if error:
        return error
    assert path is not None and data is not None
    prepared = _prepare_normalized_image(path, data, "full", 1.0)
    if prepared is None:
        return "[error] ไม่สามารถ normalize ภาพเพื่ออ่านได้"
    image, png_data = prepared
    try:
        # Lite-only hidden DLP preflight. No OCR text is exposed here.
        boxes, qr, _enhanced = _sensor_scan(path, png_data, retry_empty=False)
        security = _sensor_security_error(boxes, qr)
        if security:
            return security
        frame = _encode_one(image)
    finally:
        image.close()

    _READ_ATTEMPTS[src] = _READ_ATTEMPTS.get(src, 0) + 1
    if not _queue_encoded_bundle(path, [frame]):
        return _vision_budget_notice(1)
    _OVERVIEW_SEEN.add(src)
    _IMAGE_TURN_COUNT[0] += 1
    label = (
        f"[ภาพที่ {_IMAGE_TURN_COUNT[0]}: {path.name}]\n"
        if _IMAGE_TURN_COUNT[0] >= 2 else ""
    )
    result = (
        f"{label}[original image queued for agent direct vision — "
        "inspect pixels first; visible OCR/table/QR sensor not run]"
    )
    if deferred:
        result += (
            f"\n[requested sensor deferred: {deferred}. "
            "Inspect the original now; if still needed call read_image on the SAME path again.]"
        )
    _READ_SEEN[read_sig] = result
    return result


def _serve_text(src: str) -> str:
    if (src, "text") in _READ_SEEN:
        return "[อ่านซ้ำ] OCR/text ของภาพนี้ถูกอ่านแล้วใน turn นี้ — ใช้ผลเดิม ห้ามเรียกซ้ำ"
    path, data, error = _resolve_source(src)
    if error:
        return error
    assert path is not None and data is not None
    prepared = _prepare_normalized_image(path, data, "full", 1.0)
    if prepared is None:
        return "[error] ไม่สามารถ normalize ภาพเพื่ออ่านได้"
    image, png_data = prepared
    image.close()
    boxes, qr, enhanced = _sensor_scan(path, png_data, retry_empty=True)
    security = _sensor_security_error(boxes, qr)
    if security:
        return security
    result = _format_text_sensor(boxes or [], qr or [], enhanced)
    _READ_ATTEMPTS[src] = _READ_ATTEMPTS.get(src, 0) + 1
    _READ_SEEN[(src, "text")] = result
    return result


def _serve_find(src: str, find: str) -> str:
    _ZOOM_ATTEMPTS[0] += 1
    if _ZOOM_ATTEMPTS[0] > _ZOOM_ATTEMPT_MAX:
        return "[หยุดซูม] ใช้ find/zoom หลายครั้งแล้ว — ตอบจากหลักฐานที่มี"
    signature = (src, f"find:{find.casefold()}", 0.0)
    if signature in _ZOOM_SEEN:
        return f"[ซูมซ้ำ] เคยหา '{find}' ในภาพนี้แล้ว — ใช้ crop เดิมหรือเปลี่ยนคำค้น"

    path, data, error = _resolve_source(src)
    if error:
        return error
    assert path is not None and data is not None
    prepared = _prepare_normalized_image(path, data, "full", 1.0)
    if prepared is None:
        return "[error] ไม่สามารถ normalize ภาพเพื่ออ่านได้"
    image, png_data = prepared
    try:
        boxes, qr, _enhanced = _sensor_scan(path, png_data, retry_empty=True)
        security = _sensor_security_error(boxes, qr)
        if security:
            return security
        match = _locate_find_box(boxes or [], find)
        if match is None:
            return f"[error] find: ไม่พบ '{find}' ใน OCR — ลองคำอื่นหรือใช้ region+zoom"
        crop = _crop_to_find_image(image, match)
    finally:
        image.close()

    try:
        crop_png = _png_bytes(crop)
        crop_boxes, crop_qr, enhanced = _sensor_scan(path, crop_png, retry_empty=True)
        security = _sensor_security_error(crop_boxes, crop_qr)
        if security:
            return security
        frame = _encode_one(crop)
    finally:
        crop.close()

    if not _queue_encoded_bundle(path, [frame]):
        return _vision_budget_notice(1)
    _ZOOM_SEEN.add(signature)
    crop_text = "\n".join(str(box.get("text", "")) for box in (crop_boxes or [])).strip()
    tag = "[zoomed OCR — find=\"" + find + "\"]"
    result = f"{tag}\n{crop_text}" if crop_text else f"{tag}\n(no OCR text; crop attached for direct vision)"
    if enhanced:
        result += "\n[OCR enhanced retry used]"
    return truncate_output(result, config.TOOL_OUTPUT_TRUNCATE_CHARS)


def _serve_zoom(src: str, region: str, zoom: float) -> str:
    _ZOOM_ATTEMPTS[0] += 1
    if _ZOOM_ATTEMPTS[0] > _ZOOM_ATTEMPT_MAX:
        return "[หยุดซูม] ใช้ find/zoom หลายครั้งแล้ว — ตอบจากหลักฐานที่มี"
    signature = (src, region, zoom)
    if signature in _ZOOM_SEEN:
        return f"[ซูมซ้ำ] เคยซูม region='{region}' x{zoom:g} แล้ว — เปลี่ยนบริเวณหรือใช้ผลเดิม"

    path, data, error = _resolve_source(src)
    if error:
        return error
    assert path is not None and data is not None
    prepared = _prepare_normalized_image(path, data, region, zoom)
    if prepared is None:
        return "[error] ไม่สามารถ normalize/crop ภาพเพื่ออ่านได้"
    image, png_data = prepared
    try:
        boxes, qr, enhanced = _sensor_scan(path, png_data, retry_empty=True)
        security = _sensor_security_error(boxes, qr)
        if security:
            return security
        frame = _encode_one(image)
    finally:
        image.close()
    if not _queue_encoded_bundle(path, [frame]):
        return _vision_budget_notice(1)
    _ZOOM_SEEN.add(signature)
    ocr_text = "\n".join(str(box.get("text", "")) for box in (boxes or [])).strip()
    result = f"[zoomed OCR — region={region} x{zoom:g}]\n{ocr_text or '(no OCR text; crop attached for direct vision)'}"
    if enhanced:
        result += "\n[OCR enhanced retry used]"
    return truncate_output(result, config.TOOL_OUTPUT_TRUNCATE_CHARS)


def _serve_structural(src: str, detail: str) -> str:
    read_sig = (src, detail)
    if read_sig in _READ_SEEN:
        return f"[อ่านซ้ำ] detail='{detail}' ของภาพนี้ถูกสร้างแล้วใน turn นี้ — ใช้ภาพเดิม"
    path, data, error = _resolve_source(src)
    if error:
        return error
    assert path is not None and data is not None
    prepared = _prepare_normalized_image(path, data, "full", 1.0)
    if prepared is None:
        return "[error] ไม่สามารถ normalize ภาพเพื่ออ่านได้"
    image, png_data = prepared
    try:
        boxes, qr, _enhanced = _sensor_scan(path, png_data, retry_empty=True)
        security = _sensor_security_error(boxes, qr)
        if security:
            return security
        items = _grid_detail_images(image) if detail == "slide" and len(boxes or []) >= 60 else _chart_detail_images(image)
        labels, frames = _encode_detail_bundle(items)
    finally:
        image.close()

    if not _queue_encoded_bundle(path, frames):
        return _vision_budget_notice(len(frames))
    label_text = _ocr_label_assist(boxes or [])
    heading = "MARKET/SLIDE DETAIL PACK" if detail == "slide" else "CHART DETAIL PACK"
    result = (
        f"[{heading} — {len(frames)} progressive detail views attached]\n"
        f"crop order: {', '.join(f'{index + 1}={label}' for index, label in enumerate(labels))}\n"
        "VISION-FIRST: compare these crops with the original overview. OCR below is spelling-only; "
        "read numbers, signs, ranks, axes, cells and relationships from pixels.\n"
        + (f"\n[OCR label assist]\n{label_text}" if label_text else "")
    )
    _READ_ATTEMPTS[src] = _READ_ATTEMPTS.get(src, 0) + 1
    _READ_SEEN[read_sig] = result
    return truncate_output(result, config.TOOL_OUTPUT_TRUNCATE_CHARS)


def progressive_guard_args(args: dict) -> dict:
    """Include progressive state in duplicate-call identity without changing tool args.

    A MAX-style sensor request may be intentionally emitted twice with byte-identical
    model arguments: the first unseen call is downgraded to overview, while the
    second (after pixels became visible) performs the requested OCR/find/zoom/detail.
    Lite's generic repeat guard would otherwise stop that valid second phase.
    """
    guarded = dict(args)
    raw = guarded.get("path")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, list):
                raw = decoded
    sources = raw if isinstance(raw, list) else [raw]
    cleaned = [str(source).strip() for source in sources if isinstance(source, str)]
    guarded["_vision_phase"] = (
        "sensor" if cleaned and all(source in _OVERVIEW_SEEN for source in cleaned)
        else "overview"
    )
    return guarded


def _read_image_impl(path, region: str = "full", zoom=1.0, find: str = "", detail: str = "overview", ocr=False) -> str:
    if isinstance(path, str):
        text = path.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, list):
                path = decoded
    ocr = _coerce_ocr_selector(ocr)

    if isinstance(path, list):
        if region != "full" or str(zoom) not in {"1", "1.0"} or find or detail != "overview":
            return (
                "[error] batch read_image รองรับ overview + OCR selection เท่านั้น; "
                "find/zoom/chart/slide/text ให้เรียกทีละภาพ"
            )
        return _read_image_batch(path, ocr)

    if not isinstance(path, str) or not path.strip():
        return "[error] path is required"
    if ocr not in (False, None, []):
        return "[error] ocr selector ใช้เฉพาะ batch; ภาพเดียวให้ใช้ detail='text'"

    src = path.strip()
    detail = (detail or "overview").strip().lower()
    if detail not in _DETAIL_MODES:
        return f"[error] detail ต้องเป็น: {', '.join(sorted(_DETAIL_MODES))}"
    region = (region or "full").strip().lower()
    if region != "full" and region not in _REGION_ANCHORS:
        return f"[error] region ต้องเป็น full หรือ: {', '.join(_REGION_ANCHORS)}"
    try:
        zoom_value = max(1.0, min(float(zoom or 1), _ZOOM_MAX))
    except (TypeError, ValueError):
        zoom_value = 1.0
    find = (find or "").strip()
    do_find = bool(find)
    do_zoom = region != "full" or zoom_value > 1.0
    if do_zoom and region == "full":
        return "[error] zoom ต้องระบุ region เช่น top-right หรือ center"

    if src not in _OVERVIEW_SEEN and (do_find or do_zoom or detail != "overview"):
        if do_find:
            deferred = f'find="{find}"'
        elif do_zoom:
            deferred = f'region="{region}", zoom={zoom_value:g}'
        else:
            deferred = f'detail="{detail}"'
        return _serve_overview(src, deferred=deferred)

    if do_find:
        return _serve_find(src, find)
    if do_zoom:
        return _serve_zoom(src, region, zoom_value)
    if detail == "text":
        return _serve_text(src)
    if detail in {"chart", "slide"}:
        return _serve_structural(src, detail)
    return _serve_overview(src)


def run(args: dict) -> str:
    """Direct entry used by attachment flow/tests; same serialized transaction."""
    with _READ_IMAGE_CALL_LOCK:
        return _read_image_impl(
            args.get("path"),
            region=args.get("region") or "full",
            zoom=args.get("zoom") or 1,
            find=args.get("find") or "",
            detail=args.get("detail") or "overview",
            ocr=args.get("ocr", False),
        )


from langchain_core.tools import tool as _lc_tool

_ImageBatch = Annotated[list[str], Field(min_length=1, max_length=_READ_IMAGE_BATCH_MAX)]
_OcrSelection = Annotated[list[int], Field(max_length=_OCR_BATCH_MAX)]


@_lc_tool
@_serialize_read_image
def read_image(
    path: str | _ImageBatch,
    region: str = "full",
    zoom: float = 1.0,
    find: str = "",
    detail: str = "overview",
    ocr: bool | _OcrSelection | str = False,
) -> str:
    """Read one local image or batch-read up to four images with progressive vision.

    REQUIRED FIRST STEP for a new image: detail="overview". This attaches only the
    original pixels for visible evidence (Lite still performs a hidden OCR/QR DLP
    safety preflight). After the model has inspected those pixels, call the SAME
    image again with:
    - detail="text" for OCR + table reconstruction + QR;
    - detail="chart" or detail="slide" for four native structural crops;
    - find="exact text" to locate and crop around OCR text;
    - region=<named area>, zoom=2-4 for a native-resolution crop.

    Batch mode accepts path=[img1,...,img4] only. It preserves order and processes
    source-local work concurrently. ocr=False attaches originals only; ocr=True or
    ocr=[1,3] requests OCR for all/selected 1-based positions. Progressive vision
    still wins: OCR requested for a NEW image is deferred until a later call, after
    its original overview has been visible to the model.

    Paths must be absolute local files. Lite intentionally does not inherit MAX_VLM's
    URL/screen source expansion.
    """
    from helpers import turn_guard

    args = {
        "path": path, "region": region, "zoom": zoom, "find": find,
        "detail": detail, "ocr": ocr,
    }
    guard_args = progressive_guard_args(args)
    blocked = turn_guard.check("read_image", guard_args)
    if blocked:
        return blocked
    result = _read_image_impl(path, region=region, zoom=zoom, find=find, detail=detail, ocr=ocr)
    turn_guard.record("read_image", guard_args, result)
    return result
