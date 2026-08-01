# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
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

RESULTS: list[tuple[str, bool, str]] = []
PEAK_MEM_MB = 0.0
LIVE_IMAGE = Path(tempfile.gettempdir()) / "agent_live_test_image.png"


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
    _create_image_fixture()
    # Scenario 1: find a file by name
    scenario(
        "1. find file by name",
        "PLAN.md อยู่ที่ไหนในโปรเจกต์ ENDEAVOR_AGENT_LITE (บอก path เต็ม) แล้วสรุปว่าไฟล์นี้พูดถึงอะไร",
        lambda r, m: (
            "PLAN.md" in r.text and r.tool_calls > 0,
            "ต้องมี PLAN.md ในคำตอบและมีการเรียก tool อย่างน้อย 1 ครั้ง",
        ),
    )

    # Scenario 2: find a file by content
    # 2026-07-24: content search now routes to bash grep (prompts.py rule 3)
    # since the RAG index only stores 250 chars/file -- "ในโปรเจกต์นี้" scopes
    # the grep to PROJECT_ROOT regardless of how large the real machine-wide
    # index has grown, so this should still reliably find indexer.py.
    scenario(
        "2. find file by content",
        "มีไฟล์ไหนในโปรเจกต์นี้พูดถึง trigram tokenizer บ้าง",
        lambda r, m: (
            ("indexer.py" in r.text or "indexer" in r.text.lower()) and r.tool_calls > 0,
            "ควรเจอ indexer.py ผ่าน bash grep (scoped to PROJECT_ROOT)",
        ),
    )

    # Scenario 3: web search with citation
    scenario(
        "3. web search + citation",
        "ช่วยค้นหาว่า Python 3.13 มีฟีเจอร์ใหม่อะไรบ้าง พร้อมแนบลิงก์ที่มา",
        lambda r, m: (
            ("http://" in r.text or "https://" in r.text) and r.tool_calls > 0,
            "คำตอบสุดท้ายต้องมี URL ต้นฉบับ",
        ),
    )

    # Scenario 4: chit-chat, no tool needed
    scenario(
        "4. chit-chat, no tool",
        "สวัสดีครับ วันนี้เป็นไงบ้าง",
        lambda r, m: (
            r.tool_calls == 0,
            "ไม่ควรเรียก tool เลยสำหรับคำทักทาย",
        ),
    )

    # Scenario 5: tool failure, must not loop forever
    scenario(
        "5. tool failure, no infinite loop",
        "อ่านไฟล์ /nonexistent_absolutely_not_real_path_xyz/ghost.txt ให้หน่อย",
        lambda r, m: (
            not r.aborted or "หลายวิธี" in r.text or "ขั้นตอนสูงสุด" in r.text,
            "ต้องจบ turn อย่างมีเหตุผล ไม่ค้าง/ชน step budget โดยไม่มีคำอธิบาย",
        ),
    )

    # Scenario 6: read image
    scenario(
        "6. read image",
        f"อ่านรูปนี้ให้หน่อย {LIVE_IMAGE} มีข้อความอะไรบ้าง",
        lambda r, m: (
            r.tool_calls > 0 and ("4500" in r.text or "THB" in r.text or "ใบเสร็จ" in r.text or "Invoice" in r.text),
            "ควรเรียก read_image แล้วเห็น/รายงานข้อความในภาพ (OCR หรือ vision)",
        ),
    )

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, reason in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {reason}")
    print(f"\n{passed}/{len(RESULTS)} passed (bar: >=5/6)")
    print(f"mlx_vlm.server peak footprint: {PEAK_MEM_MB:.0f} MB ({PEAK_MEM_MB/1024:.3f} GB) "
          f"— standing server process, not a per-session in-process budget anymore")
    LIVE_IMAGE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
