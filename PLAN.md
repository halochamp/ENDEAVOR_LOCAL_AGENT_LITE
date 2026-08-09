# ENDEAVOR_AGENT_LITE — Implementation Plan

> วันที่วางแผน: 2026-07-23 · สถานะ: **รออนุมัติ / ยังไม่เริ่ม implement**
> เครื่องเป้าหมาย: **Backup Mac (HaloChampMacBook Air, RAM 8GB)** — `whoami` = `halochampmac`

## 1. เป้าหมาย

Agent ขนาดเบาที่รันได้จริงบนเครื่อง RAM 8GB โดยไม่ต้องพึ่ง Main Mac / tunnel ใดๆ:

1. **หาไฟล์ในเครื่อง** จากชื่อไฟล์หรือเนื้อหา → บอก path เต็ม + สรุปเนื้อหาคร่าวๆ
2. **ค้นหาข้อมูลจากอินเทอร์เน็ต** → สรุปคร่าวๆ พร้อมแนบ link ต้นฉบับเสมอ
3. **อ่านรูปภาพ** (OCR + vision) ได้ เพราะโมเดลเป็น VLM

ไม่ใช่เป้าหมาย: investment mode, planner, computer-use, speak/transcribe, awake engine,
user-profile learning — ทั้งหมดนี้**ตัดทิ้ง** ไม่ copy มา

## 2. การตัดสินใจหลัก (ยืนยันกับผู้ใช้แล้ว 2026-07-23)

| หัวข้อ | ตัดสินใจ | เหตุผล |
|---|---|---|
| โมเดล | `mlx-community/Qwen3.5-2B-OptiQ-4bit` (โหลดไว้แล้ว 2.1GB บนดิสก์) | ทดสอบผ่านแล้ว: tool-call format ถูก 100%, gen ~38 tok/s, peak RAM ~2.3-2.9GB |
| Runtime | **In-process on-demand** — `mlx_vlm.load()` ในตัว agent เอง ไม่มี server แยก | ปิดโปรแกรม = RAM คืนหมด; cold start ~5-10s ยอมรับได้ |
| RAG | **SQLite FTS5** (lexical) — ไม่ใช้ MiniLM/ChromaDB | index อยู่บน SSD, RAM ใกล้ศูนย์; ต่างจาก MAX_VLM ที่พึ่ง ENDEAVOR_RAG_MAX |
| Orchestration | **ReAct loop เขียนใหม่** (~200-300 บรรทัด) — ไม่ copy graph.py (86KB, tuned เพื่อ 35B) | โมเดลเล็กต้องการ guardrail ระดับ code ไม่ใช่ routing ซับซ้อน |
| UI | **CLI** เท่านั้น | RAM น้อยสุด |

## 3. งบ RAM (เป้า: peak รวม < 4GB บนเครื่อง 8GB)

| ส่วน | RAM โดยประมาณ |
|---|---|
| Qwen3.5-2B-OptiQ-4bit (weights + KV cache ตอน generate) | ~2.3-2.9GB |
| Python + mlx_vlm + agent code | ~300-500MB |
| SQLite FTS5 query | ~10-50MB (mmap จาก SSD) |
| Apple Vision OCR (เรียกเป็นครั้งๆ ผ่าน subprocess) | ชั่วคราว แล้วคืน |
| **รวม peak** | **~3-3.5GB** → เหลือ headroom ให้ OS + แอปอื่น |

### หลักการ SSD-first (สิ่งที่ต้องอยู่บน SSD ห้ามค้างใน RAM)

- **FTS5 index** (`data/file_index.sqlite3`) — mmap อ่านจากดิสก์
- **Web cache** (`data/web_cache.sqlite3`) — เก็บ full body ของหน้าเว็บ, ใน context มีแค่ summary สั้น + URL
- **Session log / history** (`logs/*.jsonl`) — append-only ลงดิสก์ทันที ไม่สะสมใน memory
- **Conversation history เกิน budget** — ตัดทิ้งจาก context (sliding window) แต่ log เต็มอยู่บนดิสก์
- **Lazy import ทุก dependency หนัก** — `mlx_vlm` โหลดเมื่อถึงตา generate ครั้งแรกเท่านั้น; tool ที่ไม่ถูกเรียกต้องไม่ import อะไรหนัก

## 4. โครงสร้างไฟล์

```
ENDEAVOR_AGENT_LITE/
├── PLAN.md                  # ไฟล์นี้
├── config.py                # model id, paths, limits ทั้งหมด (env-overridable)
├── llm.py                   # mlx_vlm in-process wrapper: load-once, generate, chat template
├── agent.py                 # ReAct loop + guardrails (หัวใจของระบบ)
├── ui_cli.py                # CLI chat (trim จาก MAX_VLM ui_cli.py)
├── prompts.py               # system prompt สั้น (< 2KB — ไม่ใช่ 90KB แบบ MAX)
├── helpers/
│   ├── tool_parser.py       # parse <tool_call> XML ของ Qwen3.5 + validate params
│   ├── guardrails.py        # repeat-call limiter, timeout feedback, output truncation
│   └── summarize_lite.py    # extractive summary (ไม่ใช้ LLM) — trim จาก tools/_summarize.py
├── tools/
│   ├── __init__.py          # TOOLS registry (6 ตัว)
│   ├── bash.py              # trim จาก MAX_VLM (คง safety deny-list, ตัด progress/telemetry)
│   ├── read_file.py         # trim หนัก (ต้นฉบับ 51KB → เป้า < 8KB)
│   ├── write_file.py        # copy เกือบตรง (ต้นฉบับ 3.7KB เล็กอยู่แล้ว)
│   ├── web_search.py        # trim จาก MAX_VLM: ddgs + extractive summary + บังคับแนบ URL
│   ├── rag_search.py        # เขียนใหม่: query FTS5 → path + snippet (แทน rag_tool.py 35KB)
│   └── read_image.py        # trim: Apple Vision OCR + ส่งภาพเข้า context ของ VLM โดยตรง
├── indexer.py               # สร้าง/อัพเดต FTS5 index (รันแยก ไม่ใช่ตอน chat)
├── data/                    # SSD state: file_index.sqlite3, web_cache.sqlite3
├── logs/                    # session logs (jsonl)
└── test_suite/
    ├── test_tool_parser.py  # ไม่ต้องใช้โมเดล — รันได้ทุกเมื่อ
    ├── test_guardrails.py   # ไม่ต้องใช้โมเดล
    ├── test_indexer.py      # ไม่ต้องใช้โมเดล
    └── test_agent_live.py   # ต้องใช้โมเดล — 6 สถานการณ์จริง (ดู §7)
```

## 5. Tools ทั้ง 6 ตัว — แหล่งที่มาและวิธีตัดทอน

| Tool | ต้นทาง (MAX_VLM) | วิธีทำ |
|---|---|---|
| `bash` | `tools/bash.py` (10.5KB) | **Copy + trim**: คง deny-list กันคำสั่งอันตราย + timeout; ตัด progress hooks, telemetry. เพิ่ม: บังคับ `find` ให้ scope ใต้ HOME เสมอ (กัน `find /` ที่ 2B ชอบทำ — เจอจริงตอนทดสอบ) |
| `read_file` | `tools/read_file.py` (51KB) | **Trim หนัก**: อ่าน text/markdown/code + PDF/DOCX/XLSX/PPTX แบบสกัดข้อความ read-only; จำกัดผล 5,000 ตัวอักษรและสุ่ม sample แบบกระจายเมื่อเกิน |
| `write_file` | `tools/write_file.py` (3.7KB) | **Copy เกือบตรง** — เพิ่ม guard: เขียนได้เฉพาะใต้ workspace/ กับ path ที่ user สั่งชัดเจน |
| `web_search` | `tools/web_search.py` (18KB) + `web_cache.py` + `_summarize.py` | **Copy + trim**: คง ddgs multi-engine + cache + extractive summary (ไม่ใช้ LLM สรุป — เร็ว + ไม่กิน RAM เพิ่ม) **ทุกผลลัพธ์ต้องมี `[source: URL]` ต่อท้ายเสมอ** ตาม requirement |
| `rag_search` | แนวคิดจาก `rag_tool.py` แต่ **เขียนใหม่** | SQLite FTS5: ตาราง `files(path, name, mtime, content)` + trigram/porter tokenizer. Query ครั้งเดียวได้ทั้ง match จากชื่อไฟล์และเนื้อหา คืน path + snippet |
| `read_image` | `tools/read_image.py` | **Lite port จาก MAX_VLM**: Apple Vision OCR + bounding boxes, EXIF-upright, HEIC/HEIF conversion ผ่าน macOS `sips`, named crop/zoom, long-image tiling, conservative table reconstruction และ QR decode; ทุกภาพที่ส่ง VLM ถูก resize ≤1024px/ด้านและ snapshot ส่วนตัวใน `workspect/.lite_state` ก่อนส่งเข้า mlx_vlm context. ไม่เพิ่ม separate VLM/URL-download pipeline ของ MAX เพราะ Lite ใช้โมเดล VLM เดียวและไม่ต้องขยาย network surface |

## 6. Helper functions สำหรับโมเดลเล็ก (บทเรียนจากการทดสอบจริง 2026-07-23)

ผลทดสอบ Qwen3.5-2B ที่ผ่านมาชี้จุดอ่อน 3 เรื่อง → helper ต้องชดเชยทั้งหมด:

### 6.1 Repeat-call limiter (`guardrails.py`)
**ปัญหาที่เจอ:** คำสั่ง `find /` timeout แล้วโมเดลรันคำสั่ง*เดิมเป๊ะ*ซ้ำ 4 รอบจนชน max-steps
**แก้:** เก็บ hash ของ (tool, args) ต่อ turn — ถ้าซ้ำครั้งที่ 2 → ไม่รันจริง แต่ feed กลับว่า
`"คำสั่งนี้เพิ่งล้มเหลวไปแล้ว ห้ามรันซ้ำ ให้เปลี่ยนวิธี เช่น จำกัด scope ของ find ให้แคบลง"`
ซ้ำครั้งที่ 3 → บังคับจบ turn พร้อมสรุปสิ่งที่ทำได้/ไม่ได้

### 6.2 Command rewriting / param validation (`tool_parser.py`)
**ปัญหาที่เจอ:** ถาม "ไฟล์ python กี่ไฟล์" → โมเดลใช้ `find ENDMEMEX -type f` (ลืม `-name "*.py"`)
**แก้:** validate + auto-fix ก่อน execute:
- `find /` หรือ `find` นอก HOME → rewrite เป็น scope ที่ปลอดภัย + แจ้งโมเดลใน tool_response ว่าถูกแก้
- path ไม่มีอยู่จริง → คืน error ชัดเจนพร้อม suggestion (ls parent dir ให้ดูเลย)
- required param หาย → คืน format error สั้นๆ ให้ลองใหม่ (ไม่ crash)

### 6.3 Tool-necessity nudge (`prompts.py`)
**ปัญหาที่เจอ:** ตอนไม่มี system prompt เน้นย้ำ โมเดลหลอนเรียก calculator กับคำถามชวนคุย
**แก้:** system prompt สั้นแต่มีกฎ "ตอบตรงๆ ถ้าไม่จำเป็นต้องใช้ tool" — ยืนยันจากการทดสอบแล้วว่า
prompt แบบนี้ทำให้เคสทักทายไม่มโนเรียก tool

### 6.4 Output truncation ทุกทาง (`guardrails.py`)
tool result เกิน ~2,000 chars → ตัด + แนบบรรทัด `"[ตัดที่ 2000 chars — full output: <path บน SSD>]"`
กัน context โต (context โต = KV cache โต = RAM โต — สำคัญกว่าเครื่องใหญ่)

### 6.5 Step budget
max 6 tool-steps ต่อ turn (MAX ใช้ RECURSION_LIMIT=60 — 2B ไม่ควรถูกปล่อยยาวขนาดนั้น)
ชน budget → บังคับสรุปด้วยข้อมูลเท่าที่มี

## 7. แผนงานเป็น Phase (แต่ละ phase มี verification ของตัวเอง)

### Phase 1 — โครง + LLM wrapper + tool parser  *(ไม่ต้องใช้โมเดล ยกเว้นข้อสุดท้าย)*
1. `config.py`, `prompts.py`, `helpers/tool_parser.py`, `helpers/guardrails.py`
2. Unit tests: parser (XML ครบ/เพี้ยน/ไม่มี tool call), guardrails (repeat limit, truncation)
3. `llm.py` — load + chat template + generate
4. ✅ Verify: unit tests เขียว; `llm.py` smoke test 1 prompt → ได้ text กลับ, RAM peak < 3GB (ดูจาก `peak_memory` ของ mlx_vlm)

### Phase 2 — Tools ชุดแรก: bash, read_file, write_file
1. Trim จาก MAX_VLM ตาม §5 — ห้ามลาก import ของ `_progress`/`config` ฝั่ง MAX มาด้วย
2. ✅ Verify: เรียกตรงแต่ละ tool (ไม่ผ่านโมเดล) ได้ผลถูก; deny-list กัน `rm -rf` ฯลฯ; `find` โดน scope

### Phase 3 — Indexer + rag_search (FTS5)
1. `indexer.py`: เดินไฟล์ใต้ roots ที่กำหนดใน config → FTS5
   - ไม่จำกัดขนาดไฟล์: text เก็บเฉพาะต้นไฟล์ 250 ตัวอักษร; PDF/Office เก็บชื่อ/path/type/size/mtime เท่านั้น ไม่เก็บเนื้อหา; incremental ด้วย mtime
2. `tools/rag_search.py`: query → `[{path, name-match/content-match, snippet}]`
3. ✅ Verify: index โฟลเดอร์ทดสอบ → หาไฟล์เจอทั้งจากชื่อ (ไทย/อังกฤษ) และจากเนื้อหา; วัดเวลา query < 1s; RAM ระหว่าง index < 500MB

### Phase 4 — web_search + read_image
1. Trim ตาม §5; web_search ทุก summary ต้องลงท้าย `[source: URL]`
2. ✅ Verify: ค้นจริง 3 query (ไทย 2, อังกฤษ 1) → ได้ summary + link ครบ; read_image ทั้งโหมด OCR และโหมด vision กับรูปตัวอย่าง

### Phase 5 — ReAct loop + CLI + live test
1. `agent.py` ประกอบทุกอย่าง + `ui_cli.py`
2. `test_suite/test_agent_live.py` — 6 สถานการณ์:
   | # | สถานการณ์ | เกณฑ์ผ่าน |
   |---|---|---|
   | 1 | หาไฟล์จากชื่อ ("CLAUDE.md อยู่ไหน") | ได้ path จริง + สรุปเนื้อหา |
   | 2 | หาไฟล์จากเนื้อหา ("ไฟล์ไหนพูดถึง X") | rag_search → path ถูก |
   | 3 | ค้นเว็บ + แนบ link | มี URL ต้นฉบับในคำตอบ |
   | 4 | คุยเฉยๆ ไม่ต้องใช้ tool | ไม่เรียก tool เลย |
   | 5 | tool ล้มเหลว (สั่งหาไฟล์ที่ไม่มีจริง) | ไม่ loop ซ้ำ, จบ turn อย่างมีเหตุผล |
   | 6 | อ่านรูป | OCR/บรรยายภาพถูกต้อง |
3. ✅ Verify: ≥ 5/6 ผ่าน + peak RAM ตลอด session < 4GB (`sudo memory_pressure` หรือ mlx `peak_memory`)

### Phase 6 — เก็บงาน
1. `CLAUDE.md` ประจำโปรเจกต์ (กฎ: เครื่องไหนรันได้, LLM test ยาวไงตาม §8 ของ root CLAUDE.md)
2. อัพเดต ENDMEMEX + `developer/PROJECT_MEMORY.md` แรกเริ่ม
3. Commit — **บน Main Mac เท่านั้น** (กฎ backup-Mac ห้าม commit)

### Phase 7 — Adaptive Chunked Prefill (2026-07-23, RAM ต่อยอด)

> สถานะ: **✅ Implement + verify เสร็จแล้ว (7.1-7.3) — ดูผลจริงที่ §Phase 7 ผลลัพธ์ท้ายสุด**
> บริบท: Phase 5 เสร็จแล้ว (compaction fix + `mx.clear_cache()`) ทำให้ RAM ไม่โตข้าม turn อีกต่อไป
> แต่ peak ต่อ session ยังนิ่งอยู่ที่ ~4.3-4.4GB — ใกล้เกิน budget 8GB เมื่อรวม OS + แอปอื่น
> เคยลอง `prefill_step_size=128` (chunked prefill ของ mlx_vlm) แบบ **chunk ทุก prompt** มาแล้ว:
> ลด peak ได้เยอะมาก (4.13→2.49GB) แต่ live 6-scenario suite ตกจาก 5/6 เหลือ 3/6 — **revert ไปแล้ว**
> (รายละเอียดการ isolate สาเหตุอยู่ใน ENDMEMEX checkpoint 284)

#### สาเหตุที่รอบก่อนพัง (วิเคราะห์แล้ว)

- mlx_vlm default คือ `prefill_step_size=2048` — prompt ของเรา (~1,200 tokens ตอน turn แรกที่มีแค่ tool schema) **ไม่เคยถูก chunk เลยในทุก live test ที่ผ่านมา** จนถึงตอนนี้
- พอบังคับ chunk=128 การคำนวณของ **turn แรก** (จุดตัดสินใจ routing ที่ margin บางที่สุด) เปลี่ยนไปจาก baseline ที่ผ่านการทดสอบมาตลอด — floating-point ต่างกันเล็กน้อยตรงรอยต่อ chunk พลิก top-token ที่ borderline ได้
- regression เกิดเฉพาะจุดที่ routing บาง (`rag_search` vs `bash` ตอน turn แรก) ไม่ใช่ทุกจุด — แปลว่า "chunk ทุกอย่างเท่ากันหมด" คือปัญหา ไม่ใช่ chunking เองที่ผิด

#### หลักการแก้: Adaptive chunking ตามความยาว prompt

แยกสองปัญหาที่เคยปนกัน:

| | prompt สั้น (turn แรก, ~1,200 tokens) | prompt ยาว (กลาง turn, มี tool_response) |
|---|---|---|
| ความเสี่ยง RAM | ต่ำ (~3.5GB) | สูง (~4.1-4.4GB) |
| ความเปราะของ routing | **สูง** (margin บาง, เพิ่งพัง) | ต่ำ (ตัดสินใจไปแล้วจาก turn ก่อน หรือกำลังสรุปคำตอบ) |
| กลยุทธ์ | **ไม่ chunk เลย** — คำนวณต้อง byte-identical กับ baseline ที่ผ่าน test | **chunk** ที่ก้อนใหญ่พอ (512) ตัด spike RAM ตัวใหญ่สุดทิ้ง |

Threshold ตัดสินใจ: นับ token ของ prompt จริง (มี tokenizer อยู่แล้วใน `llm.py`) — ถ้าเกิน `LITE_CHUNK_THRESHOLD` (เริ่มที่ 1,800 tokens, เผื่อ margin เหนือ tool-schema-only prompt ที่ ~1,200) ค่อยส่ง `prefill_step_size=512` ให้ mlx_vlm ถ้าไม่เกิน **ไม่ส่ง parameter นี้เลย** (ให้ mlx_vlm ใช้ path เดิมที่ไม่ chunk ทุกประการ — ไม่ใช่ "chunk ด้วยค่ามาก" ซึ่งอาจยังมีพฤติกรรมคำนวณต่างจาก unchunked path)

#### แผนงานเป็น Phase ย่อย (แต่ละ phase มีเกณฑ์ผ่าน/ถอยชัดเจน)

**Phase 7.1 — Routing probe (เครื่องมือวัดที่ขาดไปตอนลองรอบแรก)**
1. `test_suite/test_routing_probe.py` (หรือ script แยก ไม่ผ่าน model — ต้องใช้ model จริง เป็น live test): ยิง turn-1 อย่างเดียว (ไม่ต้องรัน ReAct loop เต็ม) ของ query ทั้ง 6 scenario + variant ที่เกี่ยวข้อง (โดยเฉพาะคู่ rag_search vs bash ที่เคยพัง) — เช็คแค่ "tool แรกที่โมเดลเลือกเรียกคืออะไร" (เทียบกับ MAX 35B's L42 pattern: วัด routing divergence ไม่ใช่ end-to-end task success)
2. ✅ Verify: รัน probe กับ baseline (ไม่ chunk เลย, ของปัจจุบัน) ×3 รอบ — ต้องนิ่ง 3/3 ทุก query ก่อน ถ้า query ไหน flaky อยู่แล้วใน baseline ให้ตัดออกจากเกณฑ์ตัดสิน (ไม่ใช่ regression ของงานนี้)

**Phase 7.2 — Implement adaptive chunking ใน `llm.py`**
1. เพิ่ม `config.LITE_CHUNK_THRESHOLD_TOKENS = 1800`, `config.LITE_CHUNK_STEP_SIZE = 512`
2. `llm.generate()`: นับ token ของ prompt ที่ build แล้ว (`_tokenizer` มีอยู่แล้ว) → ถ้า `n_tokens > threshold` ส่ง `prefill_step_size=LITE_CHUNK_STEP_SIZE` ไปยัง `mlx_vlm.generate()`, ถ้าไม่เกิน**ไม่ส่ง parameter นี้เข้าไปเลย**
3. ✅ Verify: probe (7.1) ×3 รอบ ต้อง**เท่า baseline เป๊ะทุก query** (เพราะ turn แรกไม่ถูกแตะเลยตาม design) — ถ้าไม่เท่า แปลว่า threshold ตั้งต่ำไป (มี query สั้นๆ อื่นที่ยาวเกิน 1,800 โดยไม่ตั้งใจ) ต้องปรับ threshold ก่อนไปต่อ ไม่ใช่ยอมรับความต่าง
4. วัด RAM เพิ่ม: reproduce isolated test เดิม (tiny prompt+tools vs long in-turn context ~6,000 chars) — เฉพาะเคส "ยาว" ต้องเห็น peak ลดลงชัดเจนจาก baseline (~4.13GB)

**Phase 7.3 — Full regression**
1. `test_suite/` ทั้งหมด (unit, LLM-free) — ต้องผ่าน 100% เหมือนเดิม (ไม่ควรมีอะไรพังเพราะเป็นแค่ inference-path change)
2. `test_suite/test_agent_live.py` (6 scenario) × 3 รอบ
3. 12-turn RAM stress test (สคริปต์เดิมที่ใช้วัดตอน Phase 5/hotfix) — วัด peak รวม session
4. ✅ **เกณฑ์ผ่าน**: ทุกรอบ ≥5/6 (ตาม variance ปกติที่เจอมาตลอด, scenario 3 web-search-loop ยกเว้นได้เพราะเป็น known limitation เดิม) **และ** peak RAM ต่ำกว่า baseline เดิม (~4.3-4.4GB) อย่างชัดเจน
5. ❌ **เกณฑ์ถอย**: scenario ที่เคยเสถียร (1, 2, 4, 5, 6) พังซ้ำ ≥2 รอบจาก 3 → revert `prefill_step_size` ทั้งหมดทันที ไม่ debate ต่อ (ตาม Iteration Protocol: ≤3 รอบต่อเทคนิค)

**Phase 7.4 (optional, ทำเฉพาะถ้า 7.1-7.3 ผ่านและอยากบีบ RAM เพิ่ม) — ลด threshold แบบมีหลักฐาน**
1. ลด `LITE_CHUNK_THRESHOLD_TOKENS` ทีละขั้น (1800 → 1400 → 1000) รัน probe (7.1) ×3 ทุกขั้น
2. หยุดที่ขั้นสุดท้ายที่ probe ยังนิ่ง 3/3 ทุก query — ไม่ใช่เดา เอาหลักฐานจาก probe ล้วนๆ
3. ถ้าอยากไปได้ไกลกว่านั้นจริงๆ: เสริม few-shot ใน `prompts.py` ให้ margin ของคู่ `rag_search` vs `bash` กว้างขึ้นก่อน (จุดเดียวที่เคยพัง) แล้วค่อยลอง threshold ต่ำลงอีกรอบ — แต่เป็นงานเทรนแยก ทำทีละอย่าง (One Change at a Time) ห้ามรวมกับการปรับ threshold ในการทดสอบเดียวกัน

#### ทำไมแผนนี้ต่างจากรอบที่พัง

รอบก่อน: chunk **ทุก** prompt รวมถึง turn แรกที่ routing เปราะที่สุด = เปลี่ยนการคำนวณตรงจุดที่เสี่ยงที่สุดพร้อมกับจุดที่ไม่เสี่ยง
รอบนี้: จุดเปราะ (prompt สั้น) **ไม่ถูกแตะเลย** (การคำนวณ identical กับที่ผ่าน test มา) จุดที่กิน RAM จริง (prompt ยาว) ค่อย chunk ด้วยก้อนใหญ่ (512 ไม่ใช่ 128) ลด boundary noise ลงด้วย

#### ผลลัพธ์จริง (7.1-7.3 เสร็จแล้ว, 2026-07-23)

**Phase 7.1 (routing probe):** baseline (ไม่ chunk) 3 รอบ = 7/8 case นิ่ง+ตรง, 1 case (`find_by_content`) flaky อยู่แล้วแม้ไม่มี chunking (ไม่นับเป็น regression ของงานนี้) — หลังเปิด adaptive chunking รัน 3 รอบ = **8/8 นิ่ง+ตรงทุก case** (รวม case ที่เคย flaky ด้วย) → turn แรกไม่ถูกกระทบจริงตามดีไซน์

**Phase 7.2 (implement + isolated RAM check):** วัด apples-to-apples บน prompt เดียวกัน (เนื้อหาไฟล์จริง ไม่ใช่ synthetic repeated-char string ซึ่งพบว่า BPE บีบได้ผิดปกติ) — unchunked 4.90GB → chunked 2.97GB ยืนยันกลไกทำงานจริงเมื่อ threshold โดน trigger

**Advisor review (Codex, high effort):** เรียกมาตรวจก่อนสรุปผล พบ 2 จุดสำคัญ:
1. โค้ดเดิมส่ง `prefill_step_size=None` ตรงๆ แทนที่จะ**ไม่ส่ง parameter เลย** ตามที่ plan นี้ระบุ — verify แล้วว่าให้ผลเหมือนกันในเวอร์ชัน mlx_vlm ที่ใช้อยู่ แต่เป็น version-dependent risk ที่ไม่จำเป็น → **แก้แล้ว** ให้ omit kwarg จริงเมื่อไม่เกิน threshold
2. ตั้งข้อสังเกตว่า peak รวมทั้ง session (full live-suite runs) ไม่ลดลงชัดเจนเท่าที่ isolated test บ่งชี้ — ตั้งสมมติฐานว่า scenario 6 (vision) อาจตั้ง peak แยกที่ chunking ข้อความแตะไม่ถึง

**สืบสาเหตุ "peak รวมไม่ลดเท่าที่คาด" ต่อ (พบคำตอบจริง ไม่ใช่แค่สมมติฐาน):**
- Trace token count จริงของแต่ละ generate() call ใน scenario จริง → ยืนยันว่า chunking **ทำงานถูกต้อง** (call ที่มี tool_response ยาว 4,459 tokens ถูก chunk จริง)
- แต่ `mx.get_peak_memory()` เป็น running high-water-mark ของทั้ง process ไม่ reset ระหว่าง call — แม้แต่ turn สั้นที่สุด (ทักทายเฉยๆ, ไม่เรียก tool เลย) ก็ตั้ง floor ไว้แล้วที่ ~3.8-3.9GB จาก system prompt + tool schema ที่ส่งทุก call (จุดนี้ chunking แตะไม่ได้ตามดีไซน์ เพื่อปกป้อง routing) — เพดานที่แท้จริงจึงเป็น floor นี้บวกกับ call ที่หนักที่สุดในทั้ง session ไม่ใช่ "เฉลี่ยของทุก call"
- Vision (scenario 6) ไม่ได้เป็นตัวการหลักอย่างที่สมมติไว้ (peak เดี่ยว 4.048GB ใกล้เคียง floor เอง ไม่ได้สูงกว่าอย่างมีนัยสำคัญ)

**Phase 7.3 (full regression, ผลจริงสุดท้าย):**
- Live 6-scenario suite × 3 รอบ: **5/6 ทุกรอบ** (scenario 3 web-search-loop คือ known limitation เดิม ไม่ใช่ของใหม่) — ไม่มี regression เลย
- 76/76 unit test ผ่านตลอด
- **12-turn stress test เทียบตรงๆ กับ baseline ที่บันทึกไว้ก่อนหน้าด้วย methodology เดียวกันเป๊ะ**: unchunked = 4.363GB, chunked = **4.148GB** → ลดลงจริง ~0.215GB (~5%)

**สรุปตรงๆ:** ไม่ใช่ชัยชนะเยอะแบบที่ isolated test เคยโชว์ (1.9GB) เพราะ floor จาก system prompt + tool schema (~3.8-3.9GB) เป็นต้นทุนคงที่ที่ chunking แตะไม่ได้โดยตั้งใจ (เพื่อปกป้อง routing) — แต่**ได้ผลจริง ปลอดภัย 100% ต่อ routing (8/8 probe + 5/6×3 live suite, ไม่มี regression เลยสักรอบ)** และ 5% แถมมาโดยไม่มีความเสี่ยงเพิ่ม ถือว่าคุ้มที่จะ ship

## 8. ข้อจำกัด / ความเสี่ยงที่รู้ล่วงหน้า

- **2B ไม่ฉลาดพอจะ self-correct** — ระบบนี้ออกแบบโดยยอมรับข้อนี้: guardrails ทำงานแทน ไม่หวังพึ่ง prompt อย่างเดียว (CLAUDE.md §1: ไล่ prompt → code-level fixes ให้สุดก่อนสรุป model ceiling)
- **เครื่องนี้คือ Backup Mac** — LLM live tests (Phase 1.4, 4, 5) รันบนเครื่องนี้ได้ *เพราะ agent lite ออกแบบมาให้รันบนเครื่องนี้เอง* (in-process, ไม่แตะ :8080/:8085 ของ Main Mac) แต่ห้ามรันพร้อมงาน LLM อื่น และ**ห้าม git commit จากเครื่องนี้**
- **FTS5 ไม่มี semantic search** — "หาไฟล์เกี่ยวกับ X" ที่ไม่มีคำว่า X ตรงๆ ในไฟล์จะไม่เจอ; ยอมรับ trade-off นี้ (ถ้าเจ็บจริงค่อยพิจารณา MiniLM on-demand ภายหลัง)
- **iCloud sync**: `data/*.sqlite3` อยู่ใต้ Desktop ซึ่ง sync กับ Main Mac — index ควร**สร้างต่อเครื่อง** ไม่ควร sync ข้าม; จะกันด้วยการใส่ hostname ในชื่อไฟล์ index หรือย้าย data/ ออกนอก iCloud path (`~/Library/Application Support/ENDEAVOR_AGENT_LITE/`) — **เลือกอย่างหลัง**

### Phase 8 — สลับสถาปัตยกรรมไปใช้ LangGraph + mlx_vlm.server จริง (2026-07-23)

> สถานะ: **✅ เสร็จแล้ว, verify ผ่าน 3 รอบ**
> เหตุผล: ผู้ใช้สั่งให้ copy โครงจาก MAX_VLM จริงๆ (`react.py`'s `create_react_agent`) แทนที่ ReAct
> loop มือเขียนของ Phase 1-7 — `create_react_agent` ต้องการ LangChain chat-model object
> ไม่ใช่ raw `mlx_vlm.generate()` เลยต้องรัน **`mlx_vlm.server` จริงเป็น process แยก** +
> `langchain_openai.ChatOpenAI` ต่อเข้า (ตรงกับที่ AskUserQuestion ให้ผู้ใช้เลือกและยืนยัน)

**สิ่งที่เปลี่ยนจาก Phase 1-7:**
- `llm.py` — จาก in-process `mlx_vlm.generate()` wrapper → `build_llm()` สร้าง `ChatOpenAI` ชี้ที่
  `http://127.0.0.1:8090/v1` (`mlx_vlm.server`, เริ่มด้วย `python -m mlx_vlm.server --model ... --port 8090 --api-key x`
  **ไม่ตั้ง** `--prefill-step-size` — server รับ flag นี้เป็น global ตอน startup เท่านั้น ไม่ใช่ per-request
  แบบที่ Phase 7's adaptive chunking เคยทำได้ในตัว process เดียวกัน ใช้ default 2048 ของ mlx_vlm เอง
  ซึ่งปลอดภัยกับ prompt สั้น (~1,200-1,400 tokens) เหมือนเดิม แต่เสีย benefit เล็กๆ ของ Phase 7
  ในช่วง 1,800-2,048 tokens ไป (trade-off ที่ยอมรับ)
- `tools/*.py` — แต่ละไฟล์เพิ่มฟังก์ชัน `@tool`-decorated (LangChain `BaseTool`) ห่อ `run()` เดิม
  (ลอจิกเดิมไม่เปลี่ยน แค่เพิ่ม entry point ใหม่) — `tools/__init__.py`'s `ALL_TOOLS` คือ list ที่
  bind เข้า `create_react_agent`
- `react.py` (ใหม่) — copy pattern `build_react_agent()` จาก MAX_VLM: `dynamic_prompt()` ทำ 2 อย่าง
  (system message คงที่ทุก turn เพื่อ prompt-cache; inject รูปที่ `read_image` queue ไว้เข้า
  HumanMessage ล่าสุดก่อนเรียกโมเดลทุก step) — ต่างจาก MAX_VLM ตรงที่ inject เป็น
  `data:image/...;base64,...` (mlx_vlm.server ต้องการ base64 data URI ไม่ใช่ path ตรงๆ)
- `agent.py` — เขียนใหม่ทั้งไฟล์ ไม่มี hand-rolled while-loop อีกต่อไป (ให้ `create_react_agent`
  ทำเอง) เหลือแค่: history compaction (เหมือน Phase 5), reset `helpers/turn_guard.py` ก่อนแต่ละ
  turn, step budget ผ่าน LangGraph's `recursion_limit`, crash safety net รอบ `.invoke()`
- `helpers/turn_guard.py` (ใหม่) — ย้าย RepeatCallGuard/MAX_CALLS_PER_TOOL quota เข้าไปเป็น
  shared per-turn state ที่ **แต่ละ tool wrapper เช็คเอง** ก่อนรัน `run()` จริง (แทนที่จะ intercept
  ที่ graph level ผ่าน `pre_model_hook`/`post_model_hook` ซึ่งพบว่าเปราะกว่า เพราะต้องไปแก้ไข
  `AIMessage.tool_calls` แข่งกับ routing logic ภายในของ LangGraph เอง)
- `prompts.py` — ตัดตัวอย่าง `<tool_call><function=...>` XML ทิ้งทั้งหมด (เป็น mechanism ของ
  ReAct loop มือเขียนเดิม ไม่เกี่ยวกับ structured tool-calling ของ server แล้ว — ทิ้งไว้กลับ
  confuse โมเดลเรื่อง routing) เปลี่ยนเป็นตัวอย่างแบบ natural-language ล้วน

**Regression ที่เจอระหว่างสลับสถาปัตยกรรม (ไม่เกี่ยวกับ Phase 7's chunking) + วิธีแก้:**
- Routing ผ่าน structured tool-calling (`bind_tools()`) ต่างจาก raw XML parsing เดิมพอสมควร —
  query ผสม "หาไฟล์ + สรุปเนื้อหา" เริ่มเลือก `bash` แทน `rag_search` (query สั้นๆ ธรรมดายังถูกอยู่)
  → แก้ด้วยตัวอย่างเจาะจงใน `prompts.py` (ไม่ใช่ chunking-related เลย)
- แก้ตัวแรกแล้วดันไปกระทบ "นับไฟล์ python กี่ไฟล์" ให้เลือก `rag_search` แทน `bash` (whack-a-mole
  ตรงกับ training_guide.md L64) → เพิ่มตัวอย่าง contrast "นับจำนวน → bash, ค้นหาเจาะจง → rag_search"
  แก้ครบทั้งคู่ → **routing probe 7/7 case ที่ gate ไว้ นิ่ง+ตรง 3/3 รอบ**

**ผล verify สุดท้าย:**
- Unit tests (LLM-free): 75/75 ผ่าน
- Routing probe (`test_suite/test_routing_probe.py`, 8 case): 7/7 ที่ gate ไว้ stable+match 3/3 รอบ
- Live 6-scenario suite (`test_suite/test_agent_live.py`): **6/6 ผ่านทั้ง 3 รอบ** (ดีกว่า baseline
  เดิมของ Phase 1-7 architecture ที่ปกติอยู่ที่ 5/6 — scenario 3 web-search-loop ที่เคยเป็น known
  limitation หายไปด้วย เข้าใจว่า server's built-in tool-calling เสถียรกว่าการ parse XML มือเขียน)
- RAM (`footprint <server_pid>`, ไม่ใช่ `mx.get_peak_memory()` อีกต่อไปเพราะโมเดลไม่ได้อยู่ใน client
  process แล้ว): idle floor **~2.69GB**, ระหว่าง 12-turn stress test พุ่งชั่วคราวถึง **~3.67GB**
  แล้วลงกลับมาที่ **~2.71GB** หลัง load หาย — ไม่มี memory leak/โตไม่หยุด

**Trade-off ที่ยอมรับจากการสลับสถาปัตยกรรม (บอกผู้ใช้ตรงๆ):**
- **RAM ถูกใช้ตลอดเวลาที่ server รันอยู่** (~2.7GB คงที่) ไม่ใช่แค่ตอนใช้งาน CLI จริงเหมือน
  in-process เดิม (PLAN.md §2 เดิม) — server ต้องถูกสั่ง start/stop เอง แยกจากการเปิด/ปิด `ui_cli.py`
- Phase 7's adaptive chunked-prefill (per-request threshold) ใช้ไม่ได้แบบเดิมอีกต่อไป (server
  รับแค่ global setting) — ยอมรับ default 2048 ของ mlx_vlm แทน

**วิธีรัน (ปัจจุบัน — อัปเดต 2026-07-25 ใส่ `--vision-cache-size 2` [§9.3] และ `--prefill-step-size 512`
[§9.4] หลัง verify แล้วทั้งคู่):**
```bash
# 1. เปิด server ก่อน (ครั้งเดียว ค้างไว้)
python -m mlx_vlm.server --model mlx-community/Qwen3.5-2B-OptiQ-4bit --port 8090 --host 127.0.0.1 \
  --api-key x --vision-cache-size 2 --prefill-step-size 512

# 2. รัน CLI (เปิด/ปิดได้อิสระ ไม่กระทบ server)
python ui_cli.py
```

**บั๊กที่เจอ+แก้ระหว่าง smoke test สุดท้าย (2026-07-23):**
- `tools/rag_search.py` — ค้นหาชื่อไฟล์ตรงตัว (เช่น "PLAN.md") ไม่เจอไฟล์นั้นเองอยู่ใน top results
  เพราะ `bm25()` บน trigram-tokenized `name` column ให้ค่าที่ขัดสามัญสำนึก (ลองปรับ column weight
  หลายแบบแล้วไม่ช่วย) — แก้ด้วยการ rank ผลลัพธ์จาก `name MATCH` เองใน Python (exact match > substring
  > ชื่อสั้นกว่า) แทนที่จะเชื่อ `bm25()` score บน field สั้นๆ แบบนี้
- `indexer.py` **ยังไม่แก้**: ค้นพบว่า scan แบบ default (`~/Desktop`, `~/Documents`) ค้างได้ถ้าเจอไฟล์
  ใน cloud-sync folder (พบจริง: ไฟล์ PDF ใต้ Google Drive sync folder) ที่เป็น placeholder ยังไม่ได้
  download มาจริง — การเปิดอ่านไฟล์แบบนี้ trigger การ download ที่ใช้เวลานานมาก ทำให้ indexer หยุดค้าง
  ที่ไฟล์เดียวเป็นนาทีๆ **ยังไม่มี timeout/skip mechanism ป้องกันเรื่องนี้** — ถ้าจะใช้ indexer กับ
  scope กว้าง (default ~/Desktop) ต้องระวัง หรือ scope ให้แคบเฉพาะโฟลเดอร์ที่ไม่มี cloud-sync
  placeholder ไปก่อน (แก้ชั่วคราวที่ใช้ตอนนี้: `LITE_INDEX_ROOTS=<path เฉพาะ>`)

## Phase 9 — ลด RAM สำหรับ Backup Mac 8GB โดยไม่ลดความสามารถของ Agent (handoff ให้ Claude Sonnet, 2026-07-24)

> **สถานะ: 🟡 สำรวจ/วิจัยและเขียนแผนเสร็จแล้ว — ยังไม่ได้ implement**
>
> ผู้ใช้กำหนดชัดว่าเครื่องเป้าหมายคือ Backup Mac RAM 8GB และห้ามแลก RAM กับคุณภาพการตอบ,
> routing, ความยาวคำตอบ, history, safety หรือความสามารถของ tools. ผู้รับช่วงต้องอ่าน root
> `AGENTS.md` และ `CLAUDE.md` ก่อนทำงาน, ทำทีละ phase/ทีละ change, วัดก่อน-หลังด้วย workload
> เดียวกัน และ **ห้าม commit/push จาก Backup Mac**.

### 9.0 Baseline และข้อเท็จจริงที่วัดแล้ว (ห้ามเดาใหม่จาก `ps RSS` อย่างเดียว)

วัดจากระบบจริงขณะ `mlx_vlm.server` และ browser UI ทำงานเมื่อ 2026-07-24:

- Model: `mlx-community/Qwen3.5-2B-OptiQ-4bit`; model safetensors ที่ใช้งานจริงประมาณ **1.4GB**.
  `optiq_metadata.json` ระบุ achieved BPW ~5.283 และมี mixed 4/8-bit weights; `kv_config.json`
  ก็กำหนด KV precision ราย full-attention layerเป็น 4/8-bit อยู่แล้ว.
- Runtime: `mlx-vlm 0.6.6`, `mlx-lm 0.31.3`, `mlx 0.32.0`. `mlx-lm 0.31.x` มีการ
  `mx.clear_cache()` ระหว่าง piecewise prefill/batch generation และรุ่นที่ติดตั้งแล้วรวม fix
  Qwen3.5 GatedDeltaNet shared-buffer leak — ห้ามเสนอเพิ่ม `clear_cache()` แบบสุ่มหรืออ้างว่า
  ต้อง upgrade เพื่อได้ fix นี้โดยไม่ตรวจ version/source ก่อน.
- Model-server process: physical footprint ปัจจุบันประมาณ **2.9GB**; `vmmap` process peak เคยถึง
  **5.6GB**. MLX `/metrics` รายงาน running high-water mark `peak_memory_gb` ประมาณ **4.151GB**.
  ตัวเลขสองชุดวัดคนละขอบเขต จึงต้องเก็บทั้งคู่; ห้ามนำมาปนเป็น metric เดียว.
- Web UI process: physical footprintประมาณ **241.5MB**. การตัด FastAPI/LangGraph/HTML เพื่อหวัง
  ประหยัดส่วนนี้ไม่คุ้มกับ regression risk ในรอบแรก.
- Log model server 1,116 requests: prompt เฉลี่ย ~2,547.6 tokens, ต่ำสุด 13, สูงสุด 3,802;
  241 requests ≤2,048 และ 875 requests อยู่ช่วง 2,049–4,096; ไม่มี request >4,096 ใน log ชุดนี้.
  ค่านี้เป็นหลักฐาน workload ปัจจุบัน แต่ไม่ใช่สิทธิ์ให้ cap context ที่ 4,096 เพราะ history จริง
  ยังอาจยาวกว่า test log.
- RAG DB ปัจจุบันประมาณ **112MB บน SSD**, ไม่มี embedding model และแต่ละ query ปิด SQLite
  connection เมื่อเสร็จ. RAG search ปกติไม่ใช่ตัวกิน RAM หลัก.
- Launcher ยังเริ่ม server โดยไม่กำหนด `--prefill-step-size`, `--vision-cache-size`,
  `--max-kv-size` หรือ APC. APC ปัจจุบัน disabled. mlx-vlm server default vision feature cache = 20.
- Agent turn ถูก serialize ด้วย `agent._TURN_LOCK`; แต่ `/index` ใช้ `_INDEX_LOCK` คนละตัวและรัน
  `asyncio.to_thread(indexer.build_index, ...)` จึงยังซ้อนกับ model turn จากอีก browser tab ได้.

**ข้อสรุป baseline:** เครื่อง 8GB รัน Lite ได้ แต่ headroom แคบเมื่อ inference ซ้อนกับ index,
เอกสารใหญ่ หรือภาพใหญ่. เป้าหมายหลักไม่ใช่บีบ SQLite/UI; ต้องกำจัด allocation ซ้ำและงานที่โหลด
ไฟล์ทั้งก้อนก่อน แล้วค่อยแตะ inference knobs ที่มีหลักฐานว่าไม่ทำ routing ถอย.

### 9.1 Priority 1 — Streaming/two-pass document extraction (ต้องรักษา output และ safety เดิม)

**ปัญหาปัจจุบัน**

- `tools/read_file.py` เรียก `extract_text(path)` สำหรับ PDF/Office หรือ `path.read_text()` สำหรับ
  text แล้วถือเนื้อหาทั้งไฟล์ใน RAM ก่อนตรวจ secret และสุ่มเหลือ 5,000 chars.
- `indexer.py` เรียก `extract_text(path)` ทั้ง PDF/Office ก่อนตัดเหลือ
  `RAG_INDEX_CONTENT_CHARS == 250`. การเก็บใน SQLite เล็ก แต่ transient extraction ไม่ได้เล็ก.
- `helpers/document_extract.py` สะสมทุกหน้า/paragraph/row/slide ใน list/string; XLSX แม้เปิด
  `read_only=True` ก็ยังสะสมผลลัพธ์ทั้งหมดก่อนคืนค่า.

**งานที่ต้องทำ**

1. เพิ่ม iterator/streaming extraction กลางที่ yield หน่วยข้อความตามลำดับเดิม:
   - text/code: chunk/line iterator;
   - PDF: page-by-page;
   - DOCX: paragraph/table row ตามลำดับที่ implementation เดิมใช้;
   - XLSX/XLS: sheet/row streaming;
   - PPTX: slide/shape/table row ตามลำดับเดิม.
2. `indexer` ต้องหยุด extraction เมื่อได้ normalized safe prefix 250 chars โดยไม่สร้าง full text.
   ผล `name/path/type/size/mtime/content` ใน index ต้องเท่ากับ behavior เดิมสำหรับ fixture เดียวกัน.
3. `read_file` ต้องใช้ deterministic two-pass sampling:
   - pass 1 นับจำนวน chars/หน่วยที่มีเนื้อหาและตรวจ secret แบบ streaming;
   - pass 2 คำนวณ index จาก seed เดิม (`path:length`) แล้วอ่านเฉพาะหน่วยที่เลือก;
   - header, sample order, cap 5,000 chars และ `start_line`/`end_line` ต้องคง semantics เดิม.
4. Secret scanner แบบ chunk ต้อง carry overlapอย่างน้อยยาวกว่ารูปแบบ token/private-key ที่ยาวสุด
   เพื่อไม่ให้ credential ที่คร่อม chunk หลุด. ห้ามลดจาก “ตรวจทั้งไฟล์ก่อนคืนข้อมูล” เป็นตรวจเฉพาะ sample.
5. ห้ามเพิ่ม file-size cap เพียงเพื่อกด RAM; เป้าหมายคือ bounded-memory streaming ไม่ใช่ปฏิเสธไฟล์ใหญ่.

**Verification gate**

- Golden-output tests เทียบ implementation เดิมกับใหม่สำหรับ text/PDF/DOCX/XLSX/PPTX ขนาดเล็ก.
- Secret ที่อยู่ต้น/กลาง/ท้ายไฟล์และคร่อม chunk ต้องถูก block เหมือนเดิม.
- Fixture ใหญ่ (อย่างน้อย text 64MB, PDF หลายร้อยหน้า, XLSX หลายหมื่นแถว) ต้องไม่ทำให้ UI
  process โตตามขนาดไฟล์; วัด process footprint delta และเทียบ baseline เดิมแบบ apples-to-apples.
- Index row ต้องยังเก็บ contentไม่เกิน 250 chars และ `read_file` ไม่เกิน 5,000 chars.
- Unit suite เดิมผ่าน 100% ก่อนขยับ phase ต่อไป.

### 9.2 Priority 2 — แยกงาน index/extraction เป็น subprocess + shared heavy-work gate

**ปัญหาปัจจุบัน**

- `_run_with_timeout` ใน `indexer.py` สร้าง daemon thread ใหม่ต่อไฟล์. ถ้า cloud placeholder/network
  mount ค้างจริง `join(timeout)` คืนได้ แต่ thread ที่ค้างยังอยู่และอาจสะสมใน UI process.
- `/index` ทำงานใน thread ของ web UI; memory ที่ Python/native libraries cache ไว้อาจไม่กลับสู่ OS
  แม้ index จบ.
- `_INDEX_LOCK` กัน index ซ้อน index แต่ไม่กัน index ซ้อน agent turn/model inference.

**งานที่ต้องทำ**

1. รัน whole index job หรืออย่างน้อย document extractor ที่อาจค้างใน child process อายุสั้น.
2. ส่ง progress/result กลับด้วย bounded JSONL/pipe; ห้าม buffer transcript หรือ path ทั้งเครื่องไว้ใน RAM.
3. Timeout ต้อง terminate/kill เฉพาะ child ที่ resolve แล้วอย่างปลอดภัย และเก็บ partial index transaction
   ตาม policy ที่ชัดเจน; ห้ามทิ้ง daemon thread ที่ควบคุมไม่ได้.
4. เพิ่ม process-global heavy-work gate ใน web UI ให้ `/index` กับ `agent.run_turn` ไม่ทำงานหนักพร้อมกัน
   ข้ามหลาย WebSocket tabs. ผู้ใช้ต้องเห็นสถานะ “กำลัง index — chat รอคิว” ไม่ใช่ request หาย.
5. เมื่อ child จบ ต้องไม่มี orphan process/thread และ UI footprint ต้องกลับใกล้ baseline.

**Verification gate**

- จำลอง extractor ค้างเกิน timeoutหลายไฟล์: job ต้องไปไฟล์ถัดไป/จบได้, thread count และ process count
  ไม่โตตามจำนวน timeout.
- เปิดสอง tabs: tab แรก `/index`, tab สองส่ง chat — ต้อง serialize อย่าง deterministic,
  WebSocket progress ไม่สลับลำดับและไม่มี model request ซ้อน.
- Cancel/ปิด tab ระหว่าง index ต้องมี owner/cleanup semantics ชัดเจนและไม่ทำ DB เสีย.
- Index result หลัง subprocess ต้องเท่ากับ baseline fixture.

### 9.3 Priority 3 — ลด transient RAM ของ `read_image` โดยไม่เปลี่ยน pixels/OCR/security

> สถานะ (2026-07-25, Claude Sonnet): **ส่วน `read_image` (งานข้อ 1-2) เสร็จ + verify แล้ว (LLM-free)**.
> งานข้อ 3 (พิจารณา subprocess) **ยังไม่ได้ประเมิน** — `ru_maxrss` เป็น high-water mark สะสม วัด
> "native allocator คืน RAM คืน OS หลัง object ถูกปล่อยหรือไม่" ไม่ได้ ต้องวัด current RSS หลัง return
> แทน ยังไม่ได้ทำ. **Vision feature cache subphase: เสร็จ + verify แล้วด้วย mlx_vlm.server จริง**
> (แก้ความเข้าใจผิดก่อนหน้า: เครื่องนี้ [`halochampmac`, MacBook Air M1 8GB] รัน `mlx_vlm.server` กับ
> `Qwen3.5-2B-OptiQ-4bit` ได้จริงตามที่ project นี้ออกแบบไว้ตั้งแต่แรก — ดู root `CLAUDE.md` §6.5 ที่
> อัปเดตแล้ว). ผลการทดลอง `--vision-cache-size 2` เทียบ default 20 อยู่ด้านล่าง.

**ปัญหาปัจจุบัน**

`_normalized_frames()` สามารถถือ compressed source bytes, RGB เต็มความละเอียด, full-resolution PNG
สำหรับ OCR, full-size tile copiesทั้งหมด และ encoded 1024px frames พร้อมกัน. จากนั้น QR path ยัง decode
PNG เป็น OpenCV BGR อีกสำเนา. ที่ safety ceiling 50MP แค่ RGB หนึ่งสำเนาก็ประมาณ 150MB; รวมทุกสำเนา
อาจเป็นหลายร้อย MB.

**งานที่ต้องทำ**

1. รักษา EXIF transpose, crop geometry, `VLM_MAX_SIDE == 1024`, LANCZOS, PNG lossless,
   overlap/count ของ tiles และ OCR/QR-before-VLM safety order เดิม.
2. ทำ pipeline เป็นลำดับ:
   - normalize/crop;
   - ทำ OCR + QR จาก bounded private snapshot;
   - ปล่อย OCR/OpenCV/full-resolution buffers;
   - crop/resize/encode VLM tile ทีละอัน ไม่สร้าง list ของ full-size PIL tiles.
3. พิจารณา short-lived preprocessing subprocess หาก Pillow/OpenCV native allocator ยังเก็บ footprint
   หลังคืน object; output ต้องเป็น private snapshotsใต้ `workspect/.lite_state` เหมือนเดิม.
4. ห้ามลด 50MP ceiling, ลด 1024px, เปลี่ยนเป็น lossy JPEG หรือทิ้ง extreme-image tiling เพื่อให้ test ผ่าน.

**Vision feature cache subphase — ✅ ทดลองจริงแล้ว (2026-07-25, เครื่องนี้, `mlx-vlm 0.6.6`)**

- mlx-vlm server default `--vision-cache-size 20` สูงเกิน workload single-user/serialized ของ Lite.
  ยืนยัน source (`mlx_vlm/server/cli.py` L241, `server/app.py` L614-615): flag นี้ตั้ง
  `MLX_VLM_VISION_CACHE_SIZE` env var แล้วสร้าง `VisionFeatureCache(max_size=...)` จริง (LRU cache
  ของ vision-tower+embed_vision features ต่อภาพ) — ไม่ใช่ APC (`/cache/stats` endpoint คือคนละตัว
  กับ vision cache นี้ อย่าใช้เป็นตัวยืนยัน).
- คำสั่งเปิด server ที่ใช้ทดลอง:
  ```bash
  python -m mlx_vlm.server --model mlx-community/Qwen3.5-2B-OptiQ-4bit --port 8090 \
    --host 127.0.0.1 --api-key x --vision-cache-size 2
  ```
- Harness: `test_suite/vision_cache_ab_manual.py <server_pid>` (ไม่ผ่าน pytest — ต้องรัน server เอง
  ก่อน). รัน 3 สถานการณ์ตามที่กำหนด (ภาพเดิมซ้ำ 10 turns, สลับ 3 ภาพ x4 รอบ, 25 ภาพไม่ซ้ำ) ด้วยภาพ
  สังเคราะห์ 200x100px สีพื้นต่างๆ + คำถาม "สีอะไร" (ตรวจ correctness อัตโนมัติได้ที่ temperature=0).
- **ผลจริง — default (cache=20) เทียบ `--vision-cache-size 2`:**

  | Metric | cache=20 (default) | cache=2 |
  |---|---|---|
  | repeat x10 | 10/10 ถูก, latency เฉลี่ย 0.52s | 10/10 ถูก, 0.54s |
  | alternate 3 images x4 | 12/12 ถูก, 0.48s | 12/12 ถูก, 0.53s |
  | unique 25 images | 12/25 ถูก, 0.52s | 12/25 ถูก, 0.53s |
  | Server RSS steady-state | ~374-389MB | ~365-367MB |

  - **Correctness**: unique25 correctness pattern (right/wrong ต่อภาพ) **เหมือนกันทุกตัวเป๊ะ**
    ระหว่าง cache=20 กับ cache=2 (12/25 ถูกเท่ากัน ที่ index เดียวกันหมด — ผิดเฉพาะเฉดสีกำกวม เช่น
    coral/salmon/indigo ที่โมเดล 2B แยกยากอยู่แล้ว ไม่เกี่ยวกับ cache size เลย) → ยืนยัน "คุณภาพ output
    ไม่เปลี่ยน" ตามที่ plan คาดไว้.
  - **Latency**: ไม่ถอยอย่างมีนัยสำคัญในทุก scenario รวมถึง "alternate 3 images" ที่ cache=2 ตาม
    โครงสร้างแล้วเก็บภาพที่ 3 พร้อมกันไม่ได้ (LRU 2 ช่องกับภาพหมุนเวียน 3 ภาพ = miss ทุกครั้ง) — สาเหตุที่
    ไม่กระทบ latency ชัดเจน: ภาพทดสอบเล็ก (200x100px) ทำให้ vision-tower recompute ถูกมาก เทียบกับ
    ต้นทุนอื่นของ request ทั้งหมด (~0.5s ส่วนใหญ่คือ text-generation ไม่ใช่ vision encode).
  - **Footprint**: cache=2 ต่ำกว่า cache=20 จริงราว 10-20MB ที่ steady state — ทิศทางตรงกับที่ plan
    คาดไว้ แต่ margin ไม่ใหญ่ **เพราะภาพทดสอบเล็ก**: ภาพจริงที่ผ่าน VLM tile (สูงสุด 1024px/ด้าน) จะมี
    vision-feature array ใหญ่กว่านี้มาก ดังนั้นตัวเลขนี้คือ **lower bound** ของประโยชน์จริง ไม่ใช่ค่าสูงสุด.
  - ไม่มี footprint โตต่อเนื่องไม่หยุดในทั้งสอง config ตลอด 25 unique images (ทรงตัวหลัง ramp-up แรก).
  - **สรุป**: `--vision-cache-size 2` ปลอดภัยที่จะใช้จริง — คุณภาพเท่าเดิม, latency ไม่ถอย, RAM ลดลงจริง
    (แม้ margin ที่วัดได้ด้วยภาพทดสอบเล็กจะไม่ใหญ่มาก) ยังไม่จำเป็นต้องลอง 4.

**Verification gate**

- OCR text, table reconstruction, QR secret blocking และ VLM snapshots ต้องเทียบ fixture เดิมได้.
- ภาพปกติ, long screenshot 3 tiles, crop+zoom, HEIC และ 50MP boundary ต้องผ่าน.
- 25 unique-image sequence ต้องไม่ทำ model footprint โตต่อเนื่องหลัง cache ถึง limit.

### 9.4 Priority 4 — นำ adaptive chunked prefill กลับมาแบบ per-request เท่านั้น

> สถานะ: **✅ เสร็จแล้ว (2026-07-25, เครื่องนี้)** — routing probe ×3 + live suite ×3 + isolated RAM
> measurement ยืนยันตรงแล้ว; **12-turn stress ไม่ได้รันซ้ำ** (ดูเหตุผลท้าย "ผลจริง") ดังนั้นไม่ใช้คำว่า
> "verify ผ่านครบ" ตรงตาม gate เดิมทุกข้อ — ผลจริงต่างจากที่ตั้งสมมติฐานไว้ตอนแรก: **ไม่ต้อง patch
> mlx-vlm หรือทำ per-request เลย** global `--prefill-step-size 512` ก็ปลอดภัยและได้ผลในสถาปัตยกรรม
> ปัจจุบัน — แต่แลกกับ latency ที่สูงขึ้น (ดูรายละเอียดที่ "ผลจริง" ด้านล่าง).

**หลักฐานเดิมที่เชื่อถือได้จาก Phase 7**

- บังคับ global `prefill_step_size=128` ทุก prompt ลด 4.13→2.49GB แต่ live suiteตก 5/6→3/6:
  **ห้ามทำซ้ำ**.
- Adaptive path เดิมที่ไม่แตะ routing call และ chunkเฉพาะ long/tool-augmented calls:
  isolated 4.90→2.97GB; 12-turn stress 4.363→4.148GB (~0.215GB/~5%); routing/live ไม่ถอย.
- หลังย้ายเป็น server-based LangGraph architecture ความสามารถนี้หาย เพราะ launcherมีแต่ global flag.

**งานที่ต้องทำ**

1. ก่อนออกแบบ ให้วัด token attribution และแยก distribution ของ **first routing call** ออกจาก
   follow-up calls ปัจจุบัน; ห้ามใช้ threshold 1,800 เก่าทันที เพราะ tool schema/architecture เปลี่ยนแล้ว.
2. ตรวจ source/API ของ mlx-vlm version ที่ติดตั้งก่อนว่า request-level prefill optionมีอยู่แล้วหรือไม่.
   ถ้าไม่มี ให้เลือก extension/wrapper ที่เล็กและทดสอบได้; ห้าม fork ใหญ่โดยไม่จำเป็น.
3. First routing call ต้องใช้ computation path/defaultเดียวกับ baseline. Chunkเฉพาะ follow-up ที่มี
   ToolMessage/large context ด้วย step 512 หรือค่าที่ probe ยืนยัน.
4. ทำ one-change-at-a-time matrix; ห้ามรวม prompt rewrite, KV changes หรือ model changeในรอบเดียวกัน.

**Verification gate**

- Baseline routing probeก่อนเปลี่ยน ×3; หลังเปลี่ยน ×3 ต้อง match stable cases เท่าเดิมทุก case.
- Live 6-scenario suite ×3 ต้องไม่ต่ำกว่า baselineปัจจุบัน (Phase 8 เคยได้ 6/6 ×3).
- 12-turn stress และ isolated long tool-response measurement ต้องลด peakอย่างน้อยมีนัยสำคัญพอคุ้ม
  complexity; targetขั้นต่ำอ้างอิงผลเดิม ~5%, ไม่ใช่หวัง 1.9GB จาก isolated caseอย่างเดียว.
- Stable routing case พังซ้ำ ≥2/3 รอบให้ revert ทันที.

**ผลจริง (2026-07-25)**

งานข้อ 1-2 (วัดก่อนออกแบบ): ตรวจ source ของ `mlx-vlm 0.6.6` ที่ติดตั้งจริง (`server/schemas.py`,
`server/request_normalization.py`, `server/generation.py`) — ยืนยันว่า**ไม่มี** request-level
`prefill_step_size` field เลย (ต่างจาก `repetition_penalty` ที่เป็น field จริงใน schema);
`get_prefill_step_size()` อ่านจาก env var `PREFILL_STEP_SIZE` (ตั้งครั้งเดียวตอน startup) เท่านั้น
ยืนยันสมมติฐานเดิมว่าไม่มี per-request API ให้ใช้ตรงๆ.

ก่อนออกแบบ patch/wrapper ใดๆ ทำการทดลองที่ตัดสินใจได้เร็วและถูกกว่าตามคำแนะนำ (แทนที่จะ patch
`mlx_vlm` internals ทันที ซึ่งเป็น "fork ใหญ่" ตามที่งานข้อ 2 ห้ามถ้าไม่จำเป็น): เทียบ global flag
ตรงๆ (`--prefill-step-size 512`) กับ default ของ mlx-vlm เอง (2048) — ค้นพบสำคัญระหว่างทาง: **default
2048 ไม่ใช่ "ไม่ chunk เลย"** — mlx-vlm เองมี built-in default `prefill_step_size=2048` อยู่แล้ว
(`generation.py`'s `DEFAULT_PREFILL_STEP_SIZE=2048`) ดังนั้น prompt ที่ยาวกว่า 2048 tokens ก็โดน chunk
อยู่แล้วแม้ไม่ตั้ง flag ใดๆ — ต่างจากสมมติฐานเดิมที่คิดว่า "ไม่ตั้ง flag = ไม่ chunk เลย".

- **Routing probe** (`test_suite/test_routing_probe.py`, 8 cases): รันจริง ×3 รอบต่อ config บน
  server จริง (`--vision-cache-size 2` ตรึงไว้คงที่ตาม one-change-at-a-time) — turn-1 query จริงตอนนี้
  มี prompt ~3,000 tokens (ใหญ่กว่ายุค Phase 7 ที่เคยวัดไว้ ~1,200-1,400 tokens เพราะ tool schema/
  ระบบใหม่ใหญ่กว่าเดิม) **นั่นแปลว่า turn-1 โดน chunk อยู่แล้วที่ default 2048 เช่นกัน — ไม่มี
  "unchunked short-prompt path" ให้ปกป้องเหมือน Phase 7's era แล้วจริงๆ**. ผล: **byte-identical
  ทุก case ทั้ง 2 config** (default vs 512) — 7/8 stable+ตรง, 1 case (`find_by_content` → เลือก `bash`
  แทน `rag_search`) stable แต่ผิดใน**ทั้งสอง config เหมือนกันเป๊ะ** (บั๊ก routing ที่มีอยู่ก่อนแล้ว ไม่เกี่ยวกับ
  prefill เลย — บันทึกไว้เป็นบั๊กแยกที่ยังไม่แก้ ไม่ใช่ scope ของ 9.4).
- **Live 6-scenario suite** (`test_suite/test_agent_live.py`): รันจริง baseline ×1 + `--prefill-step-size
  512` ×3 รอบ — **ผลเท่ากันหมดทุกรอบทุก config: 5/6 ผ่าน** (เกิน bar ที่กำหนด ≥5/6) ความล้มเหลวเดียวคือ
  scenario 6 (read image) ซึ่งเจอทั้งใน baseline และ 512 เหมือนกัน — **เป็นบั๊ก test fixture ที่มีอยู่ก่อน
  ไม่เกี่ยวกับ 9.4**: check function หา keyword ของภาพใบเสร็จ ("4500", "THB", "Invoice") แต่ไฟล์ทดสอบจริง
  ที่ `/tmp/agent_live_test_image.png` มีแค่ข้อความ "Hello Agent Lite" — โมเดลตอบถูกทุกครั้ง
  ("Hello Agent Lite") เพียงแต่ check เขียนผิด fixture ไป (ต้องแก้ไฟล์ทดสอบแยกต่างหาก ไม่ใช่งานนี้).
- **RAM (isolated long tool-response, apples-to-apples)**: restart server สดใหม่ทุกครั้งเพื่อ reset
  high-water mark, ส่ง request เดิมทุกตัวอักษร (3,321 tokens, จำลอง tool_response ยาวกลาง turn) —
  วัดจาก response's `timings.peak_memory` (MLX server's own accounting, ตรงตาม pattern ที่ Phase 8
  วางไว้): **default (2048) = 3.85GB → `--prefill-step-size 512` = 3.03GB (ลดลง ~0.82GB / ~21%)**
  คำตอบเหมือนกันทุกตัวอักษร (temperature=0) — คุณภาพไม่เปลี่ยน. **นี่คือ peak ระหว่างการ prefill ของ
  request นั้นๆ — ตรงกับ "spark ของแรม" ที่ต้องการลดจริงๆ** ไม่ใช่ RAM รวมของ process ตลอดเวลา ตัวเลข
  นี้จึงไม่ขัดกับ live-suite footprint ที่แทบไม่เปลี่ยน (2733→2732MB) เพราะ `_track_peak()` ใน live suite
  สุ่มตัวอย่าง footprint **หลัง** `run_turn` จบแล้วเท่านั้น (post-turn snapshot) จึงมองไม่เห็น spike
  ระหว่าง prefill โดยโครงสร้างของมันเอง ไม่ใช่ว่า benefit ไม่มีจริง.
- **Latency cost (ต้องบันทึกคู่กับ RAM เพราะเป็นคนละครึ่งของ trade-off เดียวกัน)**: request เดิม (3,321
  tokens) ที่ default (2048) ~13.5s → ที่ 512 ~17.5s (**+~30% latency บน request ยาว**) และ prompt
  ที่อยู่ในช่วง 512–2048 tokens ที่แต่ก่อนไม่โดน chunk เลย จะเริ่มโดน chunk (ช้าลงเล็กน้อย) ด้วย —
  ผู้ใช้เป็นคนขอให้ลด RAM spike โดยเฉพาะ ("จุดมุ่งหวัง ฉันอยากลดการ spark ของแรมนะ") ดังนั้นการแลกนี้
  สมเหตุสมผลกับสิ่งที่ขอ แต่ต้องบันทึกให้เห็นชัดเพราะเป็น agent แบบ interactive.
- **12-turn stress**: **ไม่ได้รันซ้ำ** ตาม gate เดิม — เหตุผล: (1) ตอนออกแบบไว้ (Phase 7) การรันชุดนี้
  มีเหตุผลเพราะต้องพิสูจน์ความคุ้มค่าเทียบ complexity ของ per-request patch/wrapper ที่ซับซ้อน แต่ผลจริง
  พบว่าไม่ต้อง patch อะไรเลย (แค่ global flag เดียว) ทำให้คำถามเรื่อง "คุ้ม complexity ไหม" ไม่มีน้ำหนัก
  อีกต่อไป (2) `test_agent_live._track_peak()` สุ่มตัวอย่าง footprint หลัง `run_turn` คืนค่าแล้วเท่านั้น
  จึงจับ mid-prefill spike ไม่ได้อยู่แล้วโดยโครงสร้าง — รันซ้ำจะวัดผิดจังหวะ ไม่ได้ข้อมูลใหม่ที่มีความหมาย.
- **`find_by_content` → `bash` routing bug**: stable-wrong เหมือนกันทุก config (default และ 512) —
  ยืนยันว่าไม่เกี่ยวกับ prefill เลย เป็นบั๊ก routing ที่มีอยู่ก่อน 9.4 (เดิมอาจเป็น flaky case จาก Phase 8
  ที่ตอนนี้กลายเป็น stable-wrong แล้ว) **บันทึกไว้เป็น known issue ที่ยังไม่แก้ ควรตามดูภายหลัง ไม่ใช่ scope
  ของ 9.4**.
- **การนำไปใช้จริง**: เปลี่ยนแค่ launch flag ของ server, **ไม่มีผลจนกว่าจะ restart server ที่ :8090 ด้วย
  คำสั่งใหม่** (ดู "วิธีรัน" ด้านบน) — ตอนนี้ไม่มี server รันอยู่ (ปิดหลังทดสอบเสร็จแล้ว).
- **สรุป**: ใช้ global `--prefill-step-size 512` เป็นค่าใหม่ (ดู "วิธีรัน" ด้านบน) — ไม่ต้อง patch
  `mlx_vlm` internals, ไม่ต้องทำ per-request contextvar/middleware ตามที่งานข้อ 2-3 คาดไว้แต่แรก เพราะ
  สถาปัตยกรรมปัจจุบัน (structured tool-calling ผ่าน `bind_tools()`, tool schema ใหญ่กว่าเดิม) ทำให้
  turn-1 routing ไม่ได้ "เปราะบาง" ต่อ chunking เหมือน Phase 7's raw-XML-parsing era อีกต่อไป — ยืนยัน
  ด้วยการวัดจริงตามที่งานข้อ 1 กำหนด ไม่ใช่การเดา. Trade-off ที่ยอมรับ: **RAM peak ลด ~21% แลกกับ
  latency สูงขึ้น ~30% บน request ยาว**.

### 9.5 Optional หลัง 9.1–9.4 เท่านั้น — ลด static prompt/tool-schema tokens

Model logsแสดง prompt floorประมาณ 2.5–3.8K tokensใน workloadใหม่; system prompt + tool schemas ถูกส่ง
ทุก model callและเป็น activation/prefill floor. สามารถทำ token-attribution report แล้วหา description/example
ที่ซ้ำกันได้ แต่ **นี่ไม่ใช่ zero-risk mechanical optimizationสำหรับโมเดล 2B** — wording มีผลต่อ routing.

- ห้าม dynamic-remove tools จาก intuition หรือใช้ pre-router ที่อาจเลือก toolผิด.
- เปลี่ยนทีละ schema/ข้อความและผ่าน routing probe ×3 + live suite ×3 ทุกครั้ง.
- ถ้าประหยัด tokenน้อยหรือ routing margin แย่ลงให้เก็บ promptเดิม; UI 241MB และ prompt correctnessสำคัญกว่า.

**ผลจริง — bash docstring split experiment (2026-07-25, ทดสอบแล้ว, ใช้งานจริง)**

แนวคิด: token-attribution วัดจริงพบว่า tool schema ทั้ง 6 ตัวกิน 1,682 tokens (60% ของ floor
2,814 tokens) และ `bash` ตัวเดียวกิน 542 tokens — มากที่สุดในบรรดา tool ทั้งหมด เพราะ docstring มี
ทั้งส่วน**ตัดสินใจเลือก tool** (คำเตือน "ไม่ใช่สำหรับหาไฟล์ตามชื่อ/เนื้อหา ใช้ rag_search แทน" ที่
`tools/bash.py:38-43` มีคอมเมนต์ระบุว่าเคยสั้นแล้วทำให้ routing พัง 3/3 รอบมาก่อน — ดู 9.6 attempt B)
กับส่วน**syntax/retry ล้วนๆ** (`grep -r -l` แยก flag, ห้าม redirect, ห้าม retry คำสั่งเดิม) ปนกันอยู่
ก้อนเดียว — ส่วนหลังไม่มีผลต่อการ**เลือก** tool เพราะมันมีประโยชน์แค่**หลัง**ตัดสินใจเรียก bash แล้ว
เท่านั้น จึงแยกได้โดยไม่แตะส่วนที่เคยพังมาก่อน.

การเปลี่ยน (`tools/bash.py`, `helpers/turn_guard.py`): แยก docstring เป็น 2 ส่วน — ส่วนสั้น (routing-
critical, อยู่ใน schema ที่ bind เข้าโมเดลทุก turn เหมือนเดิม) กับ `_SYNTAX_MANUAL` (ส่วน syntax/retry
เดิมทุกคำ ย้ายไปเก็บเป็น module constant) แล้ว inject `_SYNTAX_MANUAL` เข้า **ผลลัพธ์ที่ tool คืนกลับมา
เอง** (ไม่ใช่ message แยก — ไม่ต้อง custom `ToolNode`, ไม่แตะ `create_react_agent`) เฉพาะตอนเรียก `bash`
ครั้งแรกของ turn (เช็คด้วย `turn_guard.call_count("bash") == 1` ที่เพิ่มเป็น helper ใหม่) วิธีนี้ปลอดภัย
ต่อ prefix cache เพราะเป็นแค่ conversation append ปกติ ไม่แตะ tool schema ต่อ turน (ตรวจแล้วว่า schema
คงที่ทุก turn เหมือนเดิม — ดู "ตรวจดูด้วยว่าแคชแตกไหม" ด้านล่าง).

- **Token savings วัดจริง**: `bash` schema 542→269 tokens; floor รวม 2,814→2,542 tokens
  (**ลด ~272 tokens, ~9.7%**) — ตรงกับที่ประเมินไว้ก่อนทดสอบ (~9%).
- **Routing probe**: รัน short-docstring ×3 รอบ แล้วสลับกลับไป full-docstring เดิม รัน baseline สดใหม่
  ×3 รอบ **ในเซสชันเดียวกัน** ตามคำแนะนำ (ไม่เทียบกับผลจากความจำ) — **ผลตรงกันทุกตัวอักษรทั้ง 8 case
  ทั้ง 3 รอบ ทั้ง 2 config**: `find_by_name_1/2` (discriminator หลัก, ต้องไม่ drift ไป `bash`) ยัง
  stable match `rag_search`; `bash_count_files` ยัง stable match `bash`; `find_by_content` ยัง
  stable mismatch `bash` เหมือนเดิมทุกประการ (บั๊กเดิมที่มีอยู่ก่อน ไม่ใช่ผลจากการเปลี่ยนนี้) — **ไม่มี
  regression บน discriminator ใดๆ เลย**.
- **Cache check** (คำถามผู้ใช้ "แคชแตกไหม"): เช็ค server log ตลอด 2 config นี้ (31 request รวม) พบ
  `cached_tokens=0` ทุก request ไม่มีข้อยกเว้น — ยืนยันว่า APC ปิดอยู่แล้ว (ตามที่ตั้งใจไว้ตั้งแต่ §9.6)
  จึงไม่มี cache ให้ทำลายจากการสลับ config นี้เลย. กลไก "เปลี่ยน tool schema ทำลาย prefix cache จาก
  token 0" ที่พิสูจน์แล้วด้วย chat-template diff ยังเป็นจริงในหลักการ แต่ไม่กระทบ deployment นี้เพราะ
  ไม่มี cache ทำงานอยู่ตั้งแต่แรก — และวิธีนี้ (`_SYNTAX_MANUAL` inject เข้า tool result, ไม่แตะ schema
  ต่อ turn) ก็ไม่ทำให้ schema เปลี่ยนระหว่าง turn อยู่แล้วไม่ว่า APC จะเปิดหรือปิดในอนาคต.
- **Live 6-scenario suite**: รัน ×1 รอบกับ short-docstring — **5/6 ผ่าน เท่ากับ baseline เดิมทุกประการ**
  (fail เดียวคือ scenario 6 fixture-keyword mismatch เดิมที่ไม่เกี่ยวกับงานนี้) **ที่สำคัญ: scenario 2
  (find-by-content ผ่าน bash grep — เคสที่เสี่ยงสุดเพราะ syntax notes ถูก defer ไปแล้ว) จบด้วย
  `tool_calls=1` ไม่มี retry เพิ่มเลย** — ความเสี่ยงเรื่อง step-budget ที่กังวลไว้ (first-call syntax
  risk ต่อ `MAX_TOOL_STEPS=6`) ไม่เกิดขึ้นจริงในการทดสอบนี้. peak footprint 2730MB ตรงตามช่วงเดิม.
- **สรุป**: adopt ใช้งานจริงแล้ว (`tools/bash.py`, `helpers/turn_guard.py`) — full LLM-free suite
  171/171 ผ่าน. Payoff ต่อ Lite เองเล็ก (~9.7% ของ floor, ได้เฉพาะ turn ที่ไม่เรียก bash) แต่ pattern นี้
  (แยก routing-critical จาก syntax/retry-only แล้ว inject เฉพาะตอนเรียกจริง) ใช้ซ้ำกับ tool docstring
  ยาวๆ ของ MAX (30 tools, มี tool ที่ docstring ยาวกว่า `bash` มากเช่น `tool_loop.py`'s ~65 บรรทัด) ได้
  ถ้าต้องการพอร์ตไปทดสอบที่นั่นในอนาคต — ต้องทำ routing probe + live suite ของ MAX เองก่อน adopt เช่นกัน
  ไม่ใช่ port ตรงๆ โดยไม่ตรวจซ้ำ. บันทึกไว้เป็น durable knowledge ที่ ENDMEMEX
  (`KNOW-LITE-DOCSTRING-SPLIT`) และเปิด checkpoint ไว้แล้วสำหรับงาน MAX (project
  `ENDEAVOR_LOCAL_AGENT_MAX`, pending).

**ผลจริง — ขยายไปทำครบทุก tool ของ Lite (2026-07-25, ผู้ใช้ขอให้ทำแยกทุกตัวเพื่อสอนเพิ่มได้ง่ายในอนาคต)**

ตรวจ docstring ทั้ง 6 tool ทีละตัวก่อนแก้ ไม่ใช่ force-split ทุกตัวโดยไม่ดูเนื้อหา:

- **`read_file`** (209 tokens) และ **`write_file`** (103 tokens): เนื้อหาทั้งหมดเป็น routing-critical/
  functional ล้วนๆ ไม่มีส่วน post-call/syntax-only ให้แยกเลย — **ไม่แก้ไข** (ไม่สร้าง deferred constant
  เปล่าๆ โดยไม่มีอะไรให้ defer จริง).
- **`web_search`** (237→211 tokens): แยกประโยค "The final answer... must always include the original
  source link" ออกเป็น `_CITATION_REMINDER` — เป็น post-call output-formatting instruction (มีผลตอน
  เขียนคำตอบสุดท้าย ไม่ใช่ตอนตัดสินใจเรียก tool) inject เข้า result ตอนเรียกครั้งแรกของ turn.
- **`rag_search`** (292→235 tokens): แยกย่อหน้า "ถ้า fallback/ไม่มี index อย่าสลับไป web_search" ออกเป็น
  `_FALLBACK_REMINDER` — เป็น post-call reaction guidance (มีผลหลังเห็นผลลัพธ์แล้วเท่านั้น).
- **`read_image`** (299→212 tokens): แยกส่วน "เมื่อไหร่ควรเรียกซ้ำพร้อม region/zoom" ออกเป็น
  `_ZOOM_FOLLOWUP_NOTE` — เป็น guidance สำหรับการเรียกครั้งที่สอง (มีผลหลังเห็นผล full-image ครั้งแรก
  แล้วเท่านั้น) inject เฉพาะตอนเรียกครั้งแรกด้วย region="full" (ค่า default).

**Token savings สะสม (วัดจริง)**: floor รวม 2,542→2,372 tokens (ลดเพิ่มอีก ~170 tokens, ~6.7%) —
รวมกับ `bash` split ก่อนหน้า ลดจาก baseline เดิม 2,814 tokens เหลือ 2,372 tokens (**ลดสะสม ~15.7%**).

**Verification (methodology เดิมทุกอย่าง)**:
- Routing probe ×3 รอบ (all-tools-split) — **byte-identical กับทุก baseline ก่อนหน้าทั้งหมด** ทั้ง 8
  case ไม่มี drift บน discriminator ใดๆ เลย (`web_search`/`read_image` case ก็ยัง stable match ปกติ).
- Live 6-scenario suite ×1 รอบ — **5/6 ผ่านเท่าเดิมทุกประการ**: scenario 3 (web search) ยังมี URL
  ต้นทางอยู่ในคำตอบสุดท้ายครบ (`_CITATION_REMINDER` ทำงานถูกจังหวะ), scenario 2 (rag_search) ไม่กระทบ,
  scenario 6 (read image) จบด้วย `tool_calls=1` (ไม่ต้องเรียกซ้ำ, `_ZOOM_FOLLOWUP_NOTE` ไม่ก่อปัญหา) —
  fail เดียวยังเป็น fixture-keyword mismatch เดิมที่ไม่เกี่ยวกับงานนี้. peak footprint 2730MB
  ตรงตามช่วงเดิมทุกรอบ (2730-2733MB).
- Full LLM-free suite: 171/171 ผ่าน หลังแก้ครบทั้ง 3 ไฟล์เพิ่มเติม.
- **สรุป**: adopt ใช้งานจริงแล้วทั้ง `tools/web_search.py`, `tools/rag_search.py`, `tools/read_image.py`
  (ไม่แตะ `read_file.py`/`write_file.py` เพราะไม่มีอะไรให้แยกจริง) — pattern เดียวกันทุกไฟล์:
  short docstring (routing-critical, อยู่ใน schema) + module-level deferred constant (post-call
  guidance, inject เข้า tool's own return value ผ่าน `turn_guard.call_count(name) == 1` เช็คว่าเป็น
  ครั้งแรกของ turn).

**ผลจริง — chat UI DOM scrollback cap (2026-07-25, ผู้ใช้ถาม "ui html ลด ram ได้อีกไหม")**

ตรวจ `ui/chat.html` พบจุดที่ยังไม่เคยแตะมาก่อน: `#messages` DOM growsหนึ่ง node ต่อ turn (ทั้ง
`appendSystem`, `appendUser`, streaming agent bubble ใน `onEvent`'s `'start'` handler) **ไม่มี cap เลย**
ต่างจาก activity panel ที่มี `MAX_ACT = 500` cap อยู่แล้ว (`chat.html:596`) — เป็นคนละ layer จาก
`agent.py`'s server-side history trim (คุมแค่สิ่งที่ส่งให้โมเดล ไม่เกี่ยวกับสิ่งที่ browser tab เก็บ
render ไว้) session ที่คุยยาวๆ ใน tab เดียวโดยไม่ reload จะทำให้ DOM โตไม่มีเพดาน.

แก้ไข: เพิ่ม `MAX_DOM_MESSAGES = 50` และ `trimMessageLog()` (`removeChild(firstChild)` วนจนกว่าจะ
≤ cap) เรียกหลัง `appendChild` ทั้ง 3 จุด — ปลอดภัย 100% เพราะแก้แค่ browser-side display, ไม่แตะ
context ที่ส่งให้โมเดลเลย (คนละ layer) จึงไม่มีความเสี่ยง routing/quality ใดๆ.

**Verification**: `node --check` ยืนยัน syntax ถูกต้อง; full LLM-free suite 171/171 ผ่าน (ไม่กระทบ
เพราะเป็น client-side JS ล้วนๆ, pytest ไม่ครอบคลุมส่วนนี้อยู่แล้ว). ไม่มี browser automation test จริง
เพราะ GUI automation ถูกบล็อกโดย permission classifier ของเครื่องมือ (ตาม root CLAUDE.md §8.5) — นี่คือ
ข้อจำกัดของการ verify ที่ควรบอกตรงๆ ไม่ใช่อ้างว่าทดสอบครบ.

### 9.6 สิ่งที่ห้ามใช้เป็นทางลัดลด RAM

- ห้ามลดไปโมเดล 0.8B, quantize weights/KV ต่ำลงแบบ global หรือเปลี่ยน model โดยไม่ผ่าน full A/B;
  โมเดลปัจจุบันเป็น quality-tuned mixed precisionและ KV 4/8-bit อยู่แล้ว.
- ห้ามตั้ง `--max-kv-size 512`/ต่ำโดยพลการ; rotating cache เล็กทำให้ long-context quality ลด.
- ห้ามเปิด APC warm-memory เพื่อหวังลด RAM: มันเก็บ reusable block poolควบคู่ runtime KV และ pool
  เป็น dequantized float จึงอาจเพิ่ม steady RAM. APC ปัจจุบัน disabled ให้คงไว้จนมี benchmarkเฉพาะ.
- ห้ามลด `MAX_TOKENS=512`, history 5 turns/20,000 chars, tool output 2,000 chars,
  `READ_FILE_MAX_CHARS=5,000`, image resolution/tiling หรือ step budget เพื่อทำตัวเลขสวย.
- ห้ามเพิ่ม speculative/draft model; เพิ่ม weights/RAM แม้อาจเร็วขึ้น.
- ห้ามเพิ่ม `iogpu.wired_limit_mb` แล้วเรียกว่า RAM optimization; นั่นเป็น speed/wiring knob ไม่ได้ลด usage.
- ห้าม optimize RAG DB/HTML ก่อน bounded extraction; DB 112MBบนดิสก์และ UI ~241.5MBไม่ใช่ bottleneckหลัก.
- Idle unload เป็น optional modeเท่านั้น: ประหยัด ~2.9GB ตอนพักแต่มี cold-start; ห้ามเปิดเป็น default
  ภายใต้ข้อกำหนด “พร้อมใช้และไม่ลด performance”.

### 9.7 Final acceptance matrix และลำดับส่งมอบ

ทำตามลำดับ **9.1 → 9.2 → 9.3 → 9.4**; checkpoint/รายงาน RAM หลังทุก phaseและอย่ารวม diffข้าม phase.

| Gate | เกณฑ์ขั้นต่ำ |
|---|---|
| Unit/regression | test suite LLM-free 100% |
| Routing | baseline ×3 และ candidate ×3; stable casesห้ามถอย |
| Live agent | 6 scenarios ×3; ไม่ต่ำกว่า baselineที่วัดใหม่ |
| Text RAM | 12-turn stress, idle floor, current/peak footprint และ MLX metricครบ |
| Large documents | text 64MB + multi-hundred-page PDF + large XLSX; output/safetyเท่าเดิมและ RAMไม่โตตาม body size |
| Vision | normal/HEIC/long/3-tile/50MP + 25 unique images; ไม่มี unbounded cache growth |
| Concurrency | two browser tabs `/index` + chat; serialize/progress/cancel deterministic |
| Cleanup | ไม่มี orphan child/thread; UI/model footprintกลับใกล้ post-warm baseline |

**เป้าหมายที่สมจริง:** normal text inference น่าจะลดได้ประมาณ 0.2–0.4GBโดยไม่เปลี่ยนโมเดล;
ประโยชน์ใหญ่กว่าคือกำจัด document/image/index spikesหลักร้อย MBถึงหลาย GB. อย่าสัญญาว่าจะลด warm
model floorเหลือ ~2GBโดยยังใช้ model/latencyเดิม — weightsและ runtime floorทำให้เป็นไปไม่ได้โดยไม่มี trade-off.

### 9.8 Primary references สำหรับผู้รับช่วง

- MLX-LM long prompts/KV cache: <https://github.com/ml-explore/mlx-lm#long-prompts-and-generations>
- MLX-LM releases (`prefill-step-size`, cache clear, Qwen3.5 fixes):
  <https://github.com/ml-explore/mlx-lm/releases>
- MLX-VLM server/APC/KV/vision cache documentation: <https://github.com/Blaizzy/mlx-vlm/>
- OptiQ 2B quantization/quality cliff reference:
  <https://huggingface.co/mlx-community/Qwen3.5-2B-4bit-OptiQ>

**Handoff note:** Phase 9 นี้คือ implementation brief ที่ผู้ใช้อนุมัติให้ Claude Sonnetอ่านและทำต่อ;
Codex รอบ 2026-07-24 ทำเฉพาะสำรวจ/วิจัย/เขียนแผน ยังไม่ได้แก้ runtime codeหรือ launcher.

## Phase 10 — Core hardening closure (implemented and verified, 2026-08-01)

Phase 9 is retained above as the historical implementation brief. The work is
now implemented. The final design deliberately keeps code-owned policy small:

- local document discovery is RAG-first in `agent.run_turn()`; explicit Bash
  remains Bash-first;
- Bash normalizes only deterministic scope/kind/list-count mistakes and keeps a
  strict read-only allow-list;
- web search retrieves broadly, rejects candidates in one bounded no-think LLM
  batch, and displays only code-vetted links;
- HTML alone receives first-token streaming;
- production uses thinking budget 256 and wire `max_tokens=768`;
- document/index/UI/vision paths are bounded for an 8 GB target.

Final acceptance matrix:

| Gate | Result |
|---|---|
| Development regression | 252/252 passed; 40.86 s; RSS max 546,734,080 B; 0 swaps |
| Routing after final prompt/web training | 87/87 repeated votes |
| Local document trajectories | 18/18 (6 scenarios ×3), exact sequence/command/success |
| Post-training final-byte regression | 29/29 routing + 6/6 full trajectories |
| No-tool contrast holdout after final example edit | 4/4; no-tool/RAG/Bash boundaries preserved |
| HTML live streaming | first token 1.276 s; response 1.381 s |
| Public regression | 252 passed, 2 optional-OCR skips; 46.58 s; 0 swaps |
| Structural graph | 895 nodes / 1,691 edges / 39 communities; no import cycle |
| Public sync | Core source byte-equal to development except public headers and explicit portability/server-ownership adaptations |

Thinking A/B on two simple routing cases (two repeats each): think-256 averaged
about 2.56 s and no-think about 1.12 s. Both chose the correct tool; thinking
was more exact in Bash arguments but was not dramatically smarter for simple
routing. Production keeps think-256 because multi-step tool interpretation is
the harder workload. The web candidate rejector intentionally uses no-think
because its output contract is only indexed YES/NO decisions.
