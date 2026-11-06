"""test_agent_live.py — Phase 5 live verification (PLAN.md §7 Phase 5 table).

Needs the real model loaded and (for scenario 3) network access. Not a
pytest-asserts suite — model output is non-deterministic enough that hard
asserts on wording would be flaky. Instead this runs each scenario, prints
the full transcript, and scores it against the pass criterion so a human
(or the checkpoint log) can see exactly why something failed.

Run directly: `python test_suite/test_agent_live.py`
Pass bar (PLAN.md): >= 5/6 scenarios, peak RAM across the whole session < 4GB.
"""
from __future__ import annotations
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess

import agent
import config
import indexer

RESULTS: list[tuple[str, bool, str]] = []
PEAK_MEM_MB = 0.0
LIVE_IMAGE = Path(tempfile.gettempdir()) / "agent_live_test_image.png"
LIVE_CORPUS: tempfile.TemporaryDirectory[str] | None = None


def _create_image_fixture() -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (700, 220), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 70), "Invoice Total: 4500 THB", fill="black", font=font)
    image.save(LIVE_IMAGE)


def _track_peak():
    # Architecture switch 2026-07-23 (config.py's module docstring): the
    # model runs in a separate mlx_vlm.server process now, not this client
    # — mx.get_peak_memory() here would always read 0 (no local mlx
    # inference happens in the client at all). Read the SERVER process's
    # real memory footprint instead via `footprint` (ps/RSS undercounts
    # Metal/GPU-resident memory on this machine — see PLAN.md's earlier
    # in-process RAM investigation for why `footprint`/`top`, not `ps`, is
    # the trustworthy source here).
    global PEAK_MEM_MB
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-tiTCP:{config.MLX_SERVER_PORT}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        pid = out.splitlines()[0] if out else None
        if not pid:
            return
        fp = subprocess.run(
            ["footprint", pid], capture_output=True, text=True, timeout=10
        ).stdout
        for line in fp.splitlines():
            if "Footprint:" in line:
                mb = float(line.split("Footprint:")[1].split("MB")[0].strip())
                PEAK_MEM_MB = max(PEAK_MEM_MB, mb)
                break
    except Exception:
        pass


def scenario(name: str, user_input: str, check) -> None:
    print(f"\n{'='*70}\nSCENARIO: {name}\nUSER: {user_input}\n{'='*70}")
    messages: list[dict] = []
    result = agent.run_turn(user_input, messages)
    _track_peak()
    print(f"\n[FINAL] {result.text}")
    print(f"[tool_calls={result.tool_calls} aborted={result.aborted}]")
    ok, reason = check(result, messages)
    print(f"[{'PASS' if ok else 'FAIL'}] {reason}")
    RESULTS.append((name, ok, reason))


def main() -> None:
    global LIVE_CORPUS
    _create_image_fixture()
    LIVE_CORPUS = tempfile.TemporaryDirectory(prefix="lite-live-docs-", dir=config.WORKSPECT_DIR)
    corpus = Path(LIVE_CORPUS.name)
    (corpus / "หนังสือแจ้งต่ออายุประกัน.txt").write_text(
        "กรมธรรม์ต่ออายุวันที่ 30 กันยายน 2569 เบี้ยประกัน 12,500 บาท",
        encoding="utf-8",
    )
    (corpus / "เงื่อนไขสัญญาเช่า.txt").write_text(
        "ผู้เช่าต้องแจ้งล่วงหน้า 30 วัน ค่าปรับยกเลิกสัญญา 8,000 บาท",
        encoding="utf-8",
    )
    real_index = config.FILE_INDEX_DB
    real_index_roots = config.INDEX_ROOTS
    config.FILE_INDEX_DB = corpus / "live-index.sqlite3"
    # The live path now crosses the MCP stdio process boundary.  Keep the
    # fixture's source scope explicit so the child can validate returned paths
    # against the same corpus as the parent indexer.
    config.INDEX_ROOTS = [corpus]
    indexer.build_index([corpus], verbose=False)
    try:
        scenario(
            "1. find document by name",
            "หนังสือแจ้งต่ออายุประกัน.txt อยู่ที่ไหน แล้วสรุปวันต่ออายุและเบี้ยประกัน",
            lambda r, m: (
                all(term in r.text for term in ("หนังสือแจ้งต่ออายุประกัน.txt", "30 กันยายน 2569", "12,500"))
                and "2534" not in r.text
                and r.tool_calls > 0,
                "ต้องค้นเอกสารและคัดวัน/เบี้ยตรงต้นฉบับโดยไม่แปลงปีเอง",
            ),
        )
        scenario(
            "2. find document by content",
            "เอกสารไหนพูดถึงค่าปรับยกเลิกสัญญา และค่าปรับเท่าไร",
            lambda r, m: (
                "8,000" in r.text.replace("8000", "8,000") and r.tool_calls > 0,
                "ต้องค้นเนื้อหาเอกสารและรายงานค่าปรับ 8,000 บาท",
            ),
        )
        scenario(
            "3. web search + citation",
            "ช่วยค้นหาอัตราดอกเบี้ยนโยบายไทยล่าสุด พร้อมแนบแหล่งที่มา",
            lambda r, m: (
                ("http://" in r.text or "https://" in r.text) and r.tool_calls > 0,
                "คำตอบสุดท้ายต้องมี URL ต้นฉบับ",
            ),
        )
        scenario(
            "4. chit-chat, no tool",
            "สวัสดีครับ",
            lambda r, m: (r.tool_calls == 0, "ไม่ควรเรียก tool สำหรับคำทักทาย"),
        )
        scenario(
            "5. tool failure, no infinite loop",
            "อ่านไฟล์ /tmp/nonexistent_absolutely_not_real_path_xyz_ghost.txt ให้หน่อย",
            lambda r, m: (
                not r.aborted or "หลายวิธี" in r.text or "ขั้นตอนสูงสุด" in r.text,
                "ต้องจบ turn อย่างมีเหตุผล ไม่ค้างหรือวนซ้ำ",
            ),
        )
        scenario(
            "6. read image",
            f"อ่านรูปนี้ให้หน่อย {LIVE_IMAGE} มีข้อความอะไรบ้าง",
            lambda r, m: (
                r.tool_calls > 0 and any(term in r.text for term in ("4500", "THB", "ใบเสร็จ", "Invoice")),
                "ควรเรียก read_image แล้วรายงานข้อความในภาพ",
            ),
        )
    finally:
        config.FILE_INDEX_DB = real_index
        config.INDEX_ROOTS = real_index_roots

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, reason in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {reason}")
    print(f"\n{passed}/{len(RESULTS)} passed (bar: >=5/6)")
    print(f"mlx_vlm.server peak footprint: {PEAK_MEM_MB:.0f} MB ({PEAK_MEM_MB/1024:.3f} GB) "
          f"— standing server process, not a per-session in-process budget anymore")
    LIVE_IMAGE.unlink(missing_ok=True)
    LIVE_CORPUS.cleanup()
    LIVE_CORPUS = None


if __name__ == "__main__":
    main()
