# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""OCR-only tests for tools/read_image.py — no model load needed (that path
was verified live separately, see PLAN.md Phase 4 checkpoint notes)."""
import io
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def require_ocr_helper():
    from helpers import vision_ocr
    if not vision_ocr._ensure_binary():
        pytest.skip("optional macOS OCR helper is not bundled in the public source tree")


@pytest.fixture(scope="module")
def test_image_path(tmp_path_factory):
    from PIL import Image, ImageDraw, ImageFont

    tmp_dir = tmp_path_factory.mktemp("read_image_test")
    path = tmp_dir / "sample.png"
    img = Image.new("RGB", (500, 200), color="white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Hello Agent Lite", fill="black")
    thai_font_path = "/System/Library/Fonts/Supplemental/Thonburi.ttc"
    if os.path.exists(thai_font_path):
        d.text((20, 60), "ทดสอบภาษาไทย 123", fill="black", font=ImageFont.truetype(thai_font_path, 24))
    img.save(path)
    return str(path)


def test_missing_path_arg():
    from tools import read_image
    assert read_image.run({"path": ""}).startswith("[error]")


def test_nonexistent_file():
    from tools import read_image
    out = read_image.run({"path": "/tmp/does_not_exist_xyz.png"})
    assert out.startswith("[error]")


def test_sensitive_image_path_is_blocked(tmp_path):
    from tools import read_image
    p = tmp_path / ".env.png"
    p.write_bytes(b"not-an-image")
    assert read_image.run({"path": str(p)}).startswith("[blocked]")


def test_read_image_blocks_secret_shaped_ocr(monkeypatch, test_image_path):
    from tools import read_image
    monkeypatch.setattr(read_image, "_ocr_read_layout", lambda _: [
        {"text": "sk-" + "A" * 32, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.05}
    ])
    assert read_image.run({"path": test_image_path}).startswith("[blocked]")


def test_read_image_fails_closed_when_ocr_detector_fails(monkeypatch, test_image_path):
    from tools import read_image
    read_image._PENDING_IMAGES.clear()
    monkeypatch.setattr(read_image, "_ocr_bounded_copy", lambda *_: None)
    assert read_image.run({"path": test_image_path}).startswith("[blocked]")
    assert read_image._PENDING_IMAGES == []


def test_vision_ocr_rejects_partially_malformed_helper_output(monkeypatch):
    from types import SimpleNamespace
    from helpers import vision_ocr
    monkeypatch.setattr(vision_ocr, "_ensure_binary", lambda: True)
    monkeypatch.setattr(vision_ocr.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="0\t0\t1\t1\tvalid\nmalformed", stderr=""
    ))
    with pytest.raises(RuntimeError):
        vision_ocr.read_layout("/tmp/image.png")


def test_unsupported_extension(tmp_path):
    from tools import read_image
    p = tmp_path / "notanimage.txt"
    p.write_text("hi")
    out = read_image.run({"path": str(p)})
    assert out.startswith("[error]")


def test_ocr_extracts_english_and_thai(test_image_path, require_ocr_helper):
    from tools import read_image
    read_image._PENDING_IMAGES.clear()
    out = read_image.run({"path": test_image_path})
    assert "Hello Agent Lite" in out
    assert "ทดสอบภาษาไทย" in out


def test_image_queued_and_popped(test_image_path, require_ocr_helper):
    from tools import read_image
    import config
    read_image._PENDING_IMAGES.clear()
    read_image.run({"path": test_image_path})
    queued = read_image._PENDING_IMAGES
    assert len(queued) == 1
    assert queued[0].startswith(str(config.DATA_DIR / "pending_images"))
    popped = read_image.pop_pending_images()
    assert popped == queued
    for pending in popped:
        Path(pending).unlink(missing_ok=True)
    assert read_image._PENDING_IMAGES == []


def test_normalize_preserves_exif_upright_crop_and_vlm_limit(tmp_path):
    from PIL import Image
    from tools import read_image
    path = tmp_path / "portrait.jpg"
    image = Image.new("RGB", (3000, 1800), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # 90 degrees CW: becomes an upright 1800x3000 image.
    image.save(path, exif=exif)

    frames, ocr_data = read_image._normalized_frames(path, path.read_bytes(), "top", 2)
    assert frames
    with Image.open(__import__("io").BytesIO(ocr_data)) as cropped:
        assert cropped.size == (900, 1500)
    for frame in frames:
        with Image.open(__import__("io").BytesIO(frame)) as result:
            assert max(result.size) <= read_image.VLM_MAX_SIDE


def test_table_reconstruction_keeps_grid_but_rejects_prose_columns():
    from tools import read_image
    table = [
        {"text": "Name", "x": .10, "y": .80, "w": .12, "h": .05},
        {"text": "Total", "x": .60, "y": .80, "w": .12, "h": .05},
        {"text": "Rice", "x": .10, "y": .60, "w": .12, "h": .05},
        {"text": "42", "x": .60, "y": .60, "w": .08, "h": .05},
    ]
    result = read_image._reconstruct_table(table)
    assert result is not None and "| Name | Total |" in result and "| Rice | 42 |" in result

    prose = [dict(box, text="ข้อความยาว " * 20) for box in table]
    assert read_image._reconstruct_table(prose) is None


def test_zoom_requires_named_region(test_image_path):
    from tools import read_image
    assert read_image.run({"path": test_image_path, "zoom": "2"}).startswith("[error]")


def test_read_image_blocks_secret_shaped_qr(monkeypatch, test_image_path):
    from tools import read_image
    read_image._PENDING_IMAGES.clear()
    monkeypatch.setattr(read_image, "_decode_qr", lambda _: ["sk-" + "B" * 32])
    assert read_image.run({"path": test_image_path}).startswith("[blocked]")
    assert read_image._PENDING_IMAGES == []


def test_read_image_fails_closed_when_qr_detector_fails(monkeypatch, test_image_path):
    from tools import read_image
    read_image._PENDING_IMAGES.clear()
    monkeypatch.setattr(read_image, "_decode_qr", lambda _: None)
    assert read_image.run({"path": test_image_path}).startswith("[blocked]")
    assert read_image._PENDING_IMAGES == []


def test_qr_partial_multicode_decode_fails_closed(monkeypatch):
    import cv2
    from tools import read_image

    class _Detector:
        def detectAndDecodeMulti(self, _image):
            return True, ("", "https://example.test/benign"), None, None

    monkeypatch.setattr(cv2, "imdecode", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cv2, "QRCodeDetector", _Detector)
    assert read_image._decode_qr(b"synthetic") is None


def test_qr_output_respects_tool_output_cap(monkeypatch, test_image_path):
    import config
    from tools import read_image
    monkeypatch.setattr(read_image, "_decode_qr", lambda _: ["https://example.test/" + "x" * 5000])
    output = read_image.run({"path": test_image_path})
    assert len(output) <= config.TOOL_OUTPUT_TRUNCATE_CHARS + 40
    read_image.discard_pending_images()


def test_purge_orphaned_pending_images_removes_only_aged_agent_snapshots():
    import os
    import time
    import config
    from tools import read_image
    pending = config.DATA_DIR / "pending_images"
    pending.mkdir(parents=True, exist_ok=True)
    orphan = pending / "pending-image-orphan-test.png"
    fresh = pending / "pending-image-fresh-test.png"
    unrelated = pending / "keep-me.txt"
    orphan.write_bytes(b"snapshot")
    fresh.write_bytes(b"current snapshot")
    unrelated.write_text("not a snapshot")
    old = time.time() - read_image.PENDING_SNAPSHOT_MAX_AGE_SECONDS - 1
    os.utime(orphan, (old, old))
    read_image.purge_orphaned_pending_images()
    assert not orphan.exists()
    assert fresh.exists()
    assert unrelated.exists()
    fresh.unlink()
    unrelated.unlink()


# ---------------------------------------------------------------------------
# PLAN.md §9.3: multi-tile geometry + bounded transient memory during tiling
# ---------------------------------------------------------------------------

def _reference_tile_boxes(width, height):
    """The pre-§9.3 tiling formula, kept only as an independent reference to
    check _tile_boxes' geometry hasn't drifted -- not the code path under
    test."""
    import math as _math
    long_side, short_side = max(width, height), max(1, min(width, height))
    ratio = long_side / short_side
    if ratio <= 2.5:
        return [(0, 0, width, height)]
    count = max(2, min(3, _math.ceil(ratio / 2)))
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


def test_tile_boxes_matches_reference_geometry_for_extreme_aspect_ratios():
    from tools import read_image
    for width, height in [(3000, 200), (200, 3000), (12000, 1000), (1000, 12000), (1000, 1000)]:
        assert read_image._tile_boxes(width, height) == _reference_tile_boxes(width, height)


def test_extreme_aspect_ratio_image_produces_multiple_tiles_within_vlm_limit(tmp_path):
    from PIL import Image
    from tools import read_image

    path = tmp_path / "long_screenshot.jpg"
    image = Image.new("RGB", (6000, 400), "white")
    image.save(path)

    frames, ocr_data = read_image._normalized_frames(path, path.read_bytes(), "full", 1)
    assert len(frames) >= 2, "a 15:1 aspect ratio image should be split into multiple tiles"
    for frame in frames:
        with Image.open(io.BytesIO(frame)) as result:
            assert max(result.size) <= read_image.VLM_MAX_SIDE
    with Image.open(io.BytesIO(ocr_data)) as full:
        assert full.size == (6000, 400)  # OCR/QR snapshot stays full-resolution, untiled


def test_encode_tiles_never_holds_more_than_one_tile_at_a_time_in_memory(tmp_path):
    """Direct proof (via a subprocess RSS comparison) that PLAN.md §9.3's fix
    -- crop/resize/encode one tile at a time -- actually reduces peak memory
    versus the old approach of materializing every tile's full-size crop
    before resizing any of them."""
    import resource
    import subprocess
    import sys
    import textwrap

    project_root = str(Path(__file__).resolve().parents[1])
    # 45000x1000 = 45MP (near the 50MP ceiling), ratio 45:1 -> 3 tiles, each
    # a large fraction of the source's pixel count before being downscaled.
    width, height = 45_000, 1_000

    baseline_script = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {project_root!r})
        import tools.read_image  # noqa: F401 -- import cost only
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    old_approach_script = textwrap.dedent(f"""
        import io, math, resource, sys
        sys.path.insert(0, {project_root!r})
        from PIL import Image
        image = Image.new("RGB", ({width}, {height}), "white")
        width, height = image.size
        long_side, short_side = max(width, height), max(1, min(width, height))
        ratio = long_side / short_side
        count = max(2, min(3, math.ceil(ratio / 2)))
        overlap = 0.08
        tile_len = width / (1 + (count - 1) * (1 - overlap))
        step = tile_len * (1 - overlap)
        # Old behavior: build every tile's full-size crop BEFORE resizing any.
        tiles = []
        for index in range(count):
            start = round(index * step)
            end = width if index == count - 1 else round(start + tile_len)
            if index == count - 1:
                start = max(0, end - round(tile_len))
            tiles.append(image.crop((start, 0, end, height)))
        encoded = []
        for tile in tiles:
            tile.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            tile.save(buf, format="PNG", optimize=True)
            encoded.append(buf.getvalue())
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    new_approach_script = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {project_root!r})
        from PIL import Image
        from tools import read_image
        image = Image.new("RGB", ({width}, {height}), "white")
        encoded = read_image._encode_tiles(image)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)

    def peak_mb(script: str) -> float:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return float(result.stdout.strip().splitlines()[-1]) / (1024 * 1024)

    baseline_mb = peak_mb(baseline_script)
    old_mb = peak_mb(old_approach_script)
    new_mb = peak_mb(new_approach_script)
    old_delta = old_mb - baseline_mb
    new_delta = new_mb - baseline_mb
    print(f"\n[PLAN.md 9.3 RAM] baseline={baseline_mb:.1f}MB old-tiling delta={old_delta:.1f}MB new-tiling delta={new_delta:.1f}MB ({width}x{height})")

    # A small margin, not a bare `<`: this is an RSS high-water-mark delta,
    # which can vary a little run to run (allocator/OS memory pressure) --
    # observed stable at ~70MB across repeated local runs, so 20MB is a
    # comfortable floor without being so loose it stops meaning anything.
    assert new_delta < old_delta - 20, (
        f"one-tile-at-a-time tiling did not meaningfully reduce peak memory: "
        f"old delta {old_delta:.1f}MB, new delta {new_delta:.1f}MB"
    )


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
