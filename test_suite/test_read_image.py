"""OCR-only tests for tools/read_image.py — no model load needed (that path
was verified live separately, see PLAN.md Phase 4 checkpoint notes)."""
import io
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def isolated_read_image_state(tmp_path, monkeypatch):
    """Keep tests out of the production App Support state and reset turn guards."""
    import config
    from tools import read_image

    data_dir = tmp_path / "lite-state"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    read_image.begin_image_turn()
    yield
    read_image.end_image_turn()


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


def test_vision_ocr_parses_confidence_and_candidates(monkeypatch):
    from types import SimpleNamespace
    from helpers import vision_ocr
    monkeypatch.setattr(vision_ocr, "_ensure_binary", lambda: True)
    monkeypatch.setattr(vision_ocr.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0,
        stdout="0.1\t0.2\t0.3\t0.4\t0.625000\tน้ามัน\tน้ำมัน\tนำ้มัน",
        stderr="",
    ))
    result = vision_ocr.read_layout("/tmp/image.png")
    assert result[0]["text"] == "น้ามัน"
    assert result[0]["confidence"] == pytest.approx(0.625)
    assert result[0]["candidates"] == ["น้ามัน", "น้ำมัน", "นำ้มัน"]


def test_vision_ocr_accepts_legacy_five_field_helper_output(monkeypatch):
    from types import SimpleNamespace
    from helpers import vision_ocr
    monkeypatch.setattr(vision_ocr, "_ensure_binary", lambda: True)
    monkeypatch.setattr(vision_ocr.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="0\t0\t1\t1\tlegacy text", stderr=""
    ))
    result = vision_ocr.read_layout("/tmp/image.png")
    assert result[0]["text"] == "legacy text"
    assert result[0]["confidence"] is None
    assert result[0]["candidates"] == ["legacy text"]


def test_vision_ocr_rejects_out_of_range_confidence(monkeypatch):
    from types import SimpleNamespace
    from helpers import vision_ocr
    monkeypatch.setattr(vision_ocr, "_ensure_binary", lambda: True)
    monkeypatch.setattr(vision_ocr.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0,
        stdout="0\t0\t1\t1\t1.500000\ttext\talt\t",
        stderr="",
    ))
    with pytest.raises(RuntimeError):
        vision_ocr.read_layout("/tmp/image.png")


def test_vision_ocr_rejects_partially_malformed_helper_output(monkeypatch):
    from types import SimpleNamespace
    from helpers import vision_ocr
    monkeypatch.setattr(vision_ocr, "_ensure_binary", lambda: True)
    monkeypatch.setattr(vision_ocr.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="0\t0\t1\t1\tvalid\nmalformed", stderr=""
    ))
    with pytest.raises(RuntimeError):
        vision_ocr.read_layout("/tmp/image.png")


def test_read_image_repairs_only_deterministic_thai_mark_order(monkeypatch, test_image_path):
    from tools import read_image
    monkeypatch.setattr(read_image, "_ocr_bounded_copy", lambda *_args, **_kwargs: [{
        "text": "นำ้ น้า",
        "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.1,
        "confidence": 0.7,
        "candidates": ["นำ้ น้า", "น้ำ น้ำ"],
    }])
    monkeypatch.setattr(read_image, "_decode_qr", lambda _data: [])

    overview = read_image.run({"path": test_image_path})
    assert "original image queued" in overview
    result = read_image.run({"path": test_image_path, "detail": "text"})
    assert "น้ำ น้า" in result
    assert "น้ำ น้ำ" not in result


def test_unsupported_extension(tmp_path):
    from tools import read_image
    p = tmp_path / "notanimage.txt"
    p.write_text("hi")
    out = read_image.run({"path": str(p)})
    assert out.startswith("[error]")


def test_ocr_extracts_english_and_thai(test_image_path):
    from tools import read_image
    overview = read_image.run({"path": test_image_path})
    assert "original image queued" in overview
    out = read_image.run({"path": test_image_path, "detail": "text"})
    assert "Hello Agent Lite" in out
    assert "ทดสอบภาษาไทย" in out


def test_image_queued_promoted_and_kept_for_whole_turn(test_image_path):
    from tools import read_image
    import config

    read_image.run({"path": test_image_path})
    queued = list(read_image._PENDING_IMAGES)
    assert len(queued) == 1
    assert queued[0].startswith(str(config.DATA_DIR / "pending_images"))

    active = read_image.active_turn_images()
    assert read_image._PENDING_IMAGES == []
    assert active == queued
    assert Path(active[0]).exists()
    assert read_image.active_turn_images() == active


def test_batch_accepts_four_rejects_five_and_preserves_order(tmp_path, monkeypatch):
    from PIL import Image
    from tools import read_image

    paths = []
    for index in range(5):
        path = tmp_path / f"img-{index + 1}.png"
        Image.new("RGB", (32, 24), "white").save(path)
        paths.append(str(path))

    monkeypatch.setattr(
        read_image,
        "_sensor_scan",
        lambda _path, _png, retry_empty: ([], [], False),
    )
    result = read_image.run({"path": paths[:4]})
    assert [f"[image:{i} {paths[i - 1]}]" for i in range(1, 5)] == [
        line for line in result.splitlines() if line.startswith("[image:")
    ]
    assert len(read_image._PENDING_IMAGES) == 4

    read_image.begin_image_turn()
    rejected = read_image.run({"path": paths})
    assert rejected.startswith("[error]")
    assert "สูงสุด 4" in rejected


def test_batch_selected_ocr_runs_only_after_overview(tmp_path, monkeypatch):
    from PIL import Image
    from tools import read_image

    paths = []
    for index in range(4):
        path = tmp_path / f"ocr-{index + 1}.png"
        Image.new("RGB", (32, 24), "white").save(path)
        paths.append(str(path))

    calls = []
    def fake_prepare(src, *, overview, visible_ocr):
        calls.append((src, overview, visible_ocr))
        return {
            "path": Path(src),
            "frame": b"frame" if overview else None,
            "ocr": f"OCR:{Path(src).name}" if visible_ocr else None,
        }

    monkeypatch.setattr(read_image, "_prepare_batch_source", fake_prepare)
    monkeypatch.setattr(read_image, "_queue_encoded_bundle", lambda _path, frames: True)

    first = read_image.run({"path": paths, "ocr": [1, 3]})
    assert "OCR deferred" in first
    assert all(not visible_ocr for _src, _overview, visible_ocr in calls)
    assert read_image._OVERVIEW_SEEN == set(paths)

    calls.clear()
    second = read_image.run({"path": paths, "ocr": [1, 3]})
    assert "OCR:ocr-1.png" in second
    assert "OCR:ocr-3.png" in second
    assert "OCR:ocr-2.png" not in second
    assert "OCR:ocr-4.png" not in second
    selected = [Path(src).name for src, _overview, visible_ocr in calls if visible_ocr]
    assert selected == ["ocr-1.png", "ocr-3.png"]


def test_batch_work_overlaps_with_at_most_four_workers(tmp_path, monkeypatch):
    import threading
    import time
    from PIL import Image
    from tools import read_image

    paths = []
    for index in range(4):
        path = tmp_path / f"parallel-{index + 1}.png"
        Image.new("RGB", (16, 16), "white").save(path)
        paths.append(str(path))

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_prepare(src, *, overview, visible_ocr):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return {"path": Path(src), "frame": b"frame"}

    monkeypatch.setattr(read_image, "_prepare_batch_source", fake_prepare)
    monkeypatch.setattr(read_image, "_queue_encoded_bundle", lambda _path, frames: True)
    result = read_image.run({"path": paths})

    assert not result.startswith("[error]")
    assert 2 <= state["peak"] <= 4


def test_langchain_schema_exposes_four_image_batch():
    from tools import read_image

    schema = read_image.read_image.args_schema.model_json_schema()
    path_schema = schema["properties"]["path"]
    rendered = str(path_schema)
    assert "array" in rendered
    assert "4" in rendered
    assert "ocr" in schema["properties"]
    assert "detail" in schema["properties"]
    assert "find" in schema["properties"]


def test_progressive_duplicate_guard_allows_same_sensor_after_overview(monkeypatch, test_image_path):
    from helpers import turn_guard
    from tools import read_image

    boxes = [{"text": "ยอดรวม 42", "x": .1, "y": .7, "w": .2, "h": .05}]
    monkeypatch.setattr(
        read_image,
        "_sensor_scan",
        lambda _path, _png, retry_empty: (boxes, [], False),
    )
    turn_guard.reset("อ่านข้อความ")
    args = {"path": test_image_path, "detail": "text"}

    first = read_image.read_image.invoke(args)
    assert "requested sensor deferred" in first
    second = read_image.read_image.invoke(args)
    assert "ยอดรวม 42" in second
    assert "เรียกซ้ำ" not in second
    assert not second.startswith("[error]")


def test_outer_duplicate_guard_tracks_progressive_phase(test_image_path):
    import agent
    from tools import read_image

    state = agent._ToolCallAttemptState()
    args = {"path": test_image_path, "detail": "text"}
    attempt, stop = agent._tool_call_attempt(state, "read_image", args)
    assert (attempt, stop) == (1, False)

    read_image._OVERVIEW_SEEN.add(test_image_path)
    attempt, stop = agent._tool_call_attempt(state, "read_image", args)
    assert (attempt, stop) == (1, False)


def test_progressive_find_and_chart_defer_until_overview(monkeypatch, test_image_path):
    from tools import read_image

    monkeypatch.setattr(
        read_image,
        "_sensor_scan",
        lambda _path, _png, retry_empty: ([], [], False),
    )
    deferred_find = read_image.run({"path": test_image_path, "find": "ยอดรวม"})
    assert "requested sensor deferred" in deferred_find
    assert 'find="ยอดรวม"' in deferred_find

    read_image.begin_image_turn()
    deferred_chart = read_image.run({"path": test_image_path, "detail": "chart"})
    assert "requested sensor deferred" in deferred_chart
    assert 'detail="chart"' in deferred_chart


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
    read_image.run({"path": test_image_path})
    output = read_image.run({"path": test_image_path, "detail": "text"})
    assert len(output) <= config.TOOL_OUTPUT_TRUNCATE_CHARS + 40


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
