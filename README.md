# ENDEAVOR LOCAL AGENT LITE

> เดิมชื่อ ENDEAVOR AGENT LITE (repository เดิม `halochamp/ENDEAVOR_AGENT_LITE`) — เปลี่ยนชื่อให้สอดคล้องกับ ENDEAVOR LOCAL AGENT TH ลิงก์เก่าบน GitHub จะส่งต่อมาที่นี่อัตโนมัติ

## ผู้ช่วย AI ภาษาไทยที่ทำงานบน Mac ของคุณเอง

ENDEAVOR LOCAL AGENT LITE ช่วยให้ Mac Apple Silicon เครื่องเล็ก รวมถึงรุ่นที่มี RAM 8 GB มี AI agent สำหรับงานกับไฟล์ เอกสาร รูปภาพ และข้อมูลปัจจุบันจากเว็บ โดยไม่ต้องส่งไฟล์ในเครื่องไปประมวลผลกับ LLM บนคลาวด์ และไม่มีค่า token รายเดือนสำหรับโมเดล

คุณพิมพ์คำสั่งภาษาไทยตามปกติ แล้ว agent ช่วยค้นหาเอกสารที่หาไม่เจอ อ่านและสรุป PDF/Word/Excel ตรวจรูปภาพ ค้นเว็บพร้อมลิงก์อ้างอิง หรือสร้างไฟล์ผลลัพธ์ในพื้นที่ทำงานที่ควบคุมไว้ได้

> **เหมาะเมื่อคุณต้องการความเป็นส่วนตัวและความคล่องตัว มากกว่าการใช้โมเดลขนาดใหญ่** — LITE ใช้โมเดลขนาดเล็กแบบ local จึงเหมาะกับงานประจำวันและคำสั่งที่ชัดเจน ไม่ได้ออกแบบมาแทนโมเดลใหญ่สำหรับงานวิเคราะห์หลายชั้นหรือวิจัยขนาดใหญ่

### คุณจะได้อะไร

- **แยกการค้นเอกสารออกเป็นสองขอบเขตชัดเจน** — Knowledge Base ใช้ MCP-RagDoc แบบ deterministic; ไฟล์ทั่วไปบน Desktop/Documents ใช้ local search หรือ `bash` แบบ read-only ตาม scope ที่ผู้ใช้ระบุ
- **อ่านและสรุปเอกสารได้ในที่เดียว** — รองรับข้อความ, PDF, Word และ Excel; อ่านรูปด้วย OCR ของ macOS และใช้ vision model เมื่อจำเป็น
- **ถามข้อมูลปัจจุบันพร้อมแหล่งที่มา** — ค้นเว็บและแนบ URL ที่ใช้ประกอบคำตอบโดยอัตโนมัติ
- **ทำงานกับไฟล์อย่างมีขอบเขต** — สร้างไฟล์ได้เฉพาะใน `workspect/`; การอ่านและคำสั่ง shell มีชั้นป้องกันข้อมูลอ่อนไหว
- **ใช้ได้ทั้งหน้าต่าง Terminal และเบราว์เซอร์** — เลือก CLI ที่เบา หรือ Web UI แบบ local ที่จัดการเปิด–ปิดโมเดลได้
- **ข้อมูลสำคัญอยู่บนเครื่องคุณ** — โมเดล, Knowledge Base, SQLite index และ log ทำงานบน Mac ของคุณ; การเชื่อมต่อออกอินเทอร์เน็ตเกิดขึ้นเมื่อสั่งค้นเว็บเท่านั้น

---

## ใช้แก้ปัญหาอะไรได้บ้าง

| เมื่อคุณเจอปัญหานี้ | LITE ช่วยอย่างไร |
|---|---|
| “จำไม่ได้ว่าไฟล์อยู่ที่ไหน” | ถ้าเป็น Knowledge Base จะค้นผ่าน MCP-RagDoc; ถ้าเป็นไฟล์ทั่วไปจะค้นเฉพาะ Desktop/Documents แล้วคืน path ที่ตรวจได้ |
| “มีเอกสารหลายหน้า แต่อยากรู้ใจความก่อน” | อ่าน PDF, Word, Excel หรือ text แล้วสรุปเป็นภาษาไทย โดยจำกัดข้อมูลต่อรอบเพื่อให้เครื่องเล็กยังทำงานไหว |
| “ในรูป/สกรีนช็อตมีอะไรบ้าง” | ใช้ Apple Vision OCR อ่านข้อความ รวมถึงจัดเตรียมภาพให้ vision model วิเคราะห์เมื่อจำเป็น |
| “ต้องหาข้อมูลล่าสุดและตรวจที่มา” | ค้นเว็บ ดึงผลลัพธ์ที่ใช้ได้ และแนบแหล่งอ้างอิงท้ายคำตอบ |
| “อยากให้ช่วยทำไฟล์ แต่ไม่อยากเสี่ยงเขียนทับเครื่อง” | สร้างหรือเขียนทับไฟล์ได้เฉพาะใต้ `workspect/` ซึ่งเป็น workspace ที่กำหนดไว้ |

ตัวอย่างคำสั่ง:

```text
หาไฟล์ที่เกี่ยวกับ authentication ในโปรเจกต์นี้ให้หน่อย
อ่านไฟล์ report.pdf แล้วสรุปประเด็นสำคัญเป็น bullet
ในรูป screenshot.png มีข้อความหรือรหัสผิดพลาดอะไรบ้าง
ช่วยค้นหาว่า Python รุ่นล่าสุดมีอะไรใหม่ พร้อมแนบลิงก์
สร้างไฟล์สรุป.md ใน workspect จากข้อมูลนี้
```

## เหมาะกับใคร และควรคาดหวังอะไร

เหมาะกับผู้ใช้ macOS บน Apple Silicon ที่ต้องการผู้ช่วยภาษาไทยสำหรับงานส่วนตัวหรืองานเอกสารบนเครื่อง เช่น นักพัฒนา นักเรียน นักวิจัย หรือคนที่อยากค้นและสรุปข้อมูลโดยให้ไฟล์อยู่ในเครื่องเป็นหลัก

LITE เลือกใช้โมเดล `mlx-community/Qwen3.5-2B-OptiQ-4bit` เป็นค่าเริ่มต้นเพื่อให้ใช้งานบนเครื่อง RAM 8 GB ได้จริง โดยงานทั่วไปตั้งเป้าใช้หน่วยความจำรวมราว 3–4 GB ขึ้นกับข้อมูลที่เปิดและสภาพแวดล้อมของเครื่อง ผลลัพธ์จากโมเดลขนาดเล็กอาจแปรผันได้ จึงควรใช้คำสั่งที่ระบุเป้าหมายและขอบเขตให้ชัด และตรวจทานคำตอบสำคัญเสมอ

## เริ่มใช้งานแบบเร็ว

### 1. ติดตั้ง

```bash
# หลัง clone จาก URL ของ repository ที่เผยแพร่แล้ว
cd ENDEAVOR_LOCAL_AGENT_LITE
bash install_library/install.sh
```

ตัวติดตั้งจะสร้าง `.venv` ด้วย Python 3.11, ติดตั้ง dependencies ที่ lock พร้อม hash และพยายาม build Apple Vision OCR helper จาก Swift source ให้ทั้ง Agent Lite และ MCP-RagDoc ที่ bundle มากับ repo โดยยังไม่เริ่มโหลดโมเดล

### 2. เลือกวิธีเปิดใช้งาน

**CLI** — เปิด model server ใน Terminal หนึ่ง:

```bash
source .venv/bin/activate
python -m mlx_vlm.server \
  --model mlx-community/Qwen3.5-2B-OptiQ-4bit \
  --host 127.0.0.1 --port 8090 --api-key x \
  --prefill-step-size 512
```

แล้วเปิดอีก Terminal:

```bash
source .venv/bin/activate
python ui_cli.py
```

วางเอกสาร Knowledge Base ไว้ใต้ `rag_document/` แล้วใช้ `/sync_kb` เพื่อสร้าง/อัปเดตดัชนี หรือใช้ `/exit` และ `/quit` เพื่อออกจาก CLI (`/index` ยังมีไว้เป็น compatibility path สำหรับ workflow เดิม)

**Web UI** — เปิดด้วย:

```bash
source .venv/bin/activate
python web_ui.py
```

จากนั้นไปที่ [http://127.0.0.1:8766](http://127.0.0.1:8766) แล้วกดเปิด model server จากหน้าเว็บได้ ตัว Web UI เปิดรับเฉพาะเครื่องตัวเอง (`127.0.0.1`) เพื่อไม่เผย filesystem tools ให้เครื่องอื่นในเครือข่าย

## ความสามารถหลัก

| ความสามารถ | สิ่งที่ทำได้ |
|---|---|
| Knowledge Base (`lite-rag`) | bundle `MCP-RagDoc` แบบ standalone ภายใน repo; ค้นชื่อ/เนื้อหาด้วย FTS5/BM25 + OCR fuzzy + optional semantic retrieval และมี `rag_manage` (sync/cancel) กับ `rag_status` (job/health) ผ่าน MCP |
| ค้นไฟล์ทั่วไป (`local_search`) | ค้นแบบ read-only เฉพาะ Desktop และ Documents ตาม path ของ account ที่กำลังรัน |
| อ่านเอกสาร (`read_file`) | อ่าน text/code/PDF/Word/Excel และ batch ได้สูงสุด 4 ไฟล์ต่อ call พร้อม filter แยกต่อไฟล์ |
| PDF to Text | pipeline แยกสำหรับแปลง PDF เป็นข้อความ โดยใช้ native text/OCR ตามชนิดหน้าและเก็บผลแบบ local |
| อ่านรูป (`read_image`) | Apple Vision OCR + progressive VLM analysis; รองรับหลายภาพแบบ bounded สูงสุด 4 ภาพต่อ call ใน Lite |
| สร้างไฟล์ (`write_file`) | เขียนไฟล์ผู้ใช้เฉพาะใต้ `workspect/` และต้องยืนยันก่อนเขียนทับ |
| ค้นเว็บ (`web_search`) | ค้นข้อมูลปัจจุบันได้หลาย query แบบ bounded และคืน URL ของแหล่งข้อมูล |
| MCP client | `mcp_list_tools` / `mcp_call_tool` สำหรับ server ที่ developer provision ไว้; Lite ไม่ให้โมเดล self-register server ใหม่ |
| คำสั่งอ่านอย่างปลอดภัย (`bash`) | allow-list สำหรับค้น/อ่านข้อมูล; ถ้า RAG/MCP ไม่มีหรือ unavailable จะ fallback ไป `bash` โดยรักษา scope เดิม ไม่ขยายไปทั้งเครื่อง |

### Knowledge Base และ RAG ทำงานอย่างไร

Agent Lite แยก RAG ออกจากตัว agent แล้ว: implementation อยู่ใน `MCP-RagDoc/` ซึ่งเป็น server แบบ agent-agnostic และถูก bundle มาใน public repo นี้เพื่อให้ clone เดียวใช้งานได้ครบ โดย Agent Lite ติดต่อผ่าน MCP adapter เท่านั้น

- วาง source documents ใต้ `rag_document/` (รวม nested directories ได้)
- ใช้ `/sync_kb` ใน CLI/Web UI หรือ MCP `rag_manage(action="sync")` เพื่อ rebuild แบบ incremental
- ดัชนีและ runtime state อยู่ใน `~/Library/Application Support/ENDEAVOR_LOCAL_AGENT_LITE/` โดยค่าเริ่มต้น ไม่เขียนกลับ source documents (เครื่องที่เคยติดตั้งไว้ก่อนเปลี่ยนชื่อและมีโฟลเดอร์เดิม `ENDEAVOR_AGENT_LITE/` อยู่แล้ว จะใช้โฟลเดอร์เดิมต่อโดยไม่ย้ายไฟล์)
- `rag_search` ถูกจำกัดอยู่ใน Knowledge Base; การค้น Desktop/Documents ทั่วไปใช้ `local_search`/`bash` คนละ scope
- ถ้า `lite-rag` ไม่มี, `rag_search` หาย หรือ MCP เชื่อมต่อไม่ได้ Agent จะหยุด retry RAG และ fallback ไป `bash` แบบ read-only ภายใน scope เดิม: KB → `rag_document/`, project/repository → repo ปัจจุบัน, local file ทั่วไป → Desktop/Documents
- มีหน้า console ของ MCP-RagDoc แยกที่ `MCP-RagDoc/main.py` (loopback only) สำหรับ Search / Sync / Progress / Health

`/index` ยังเก็บไว้เพื่อ compatibility กับ workflow เก่า แต่เส้นทางหลักสำหรับ Knowledge Base คือ `/sync_kb` + MCP-RagDoc

## ความเป็นส่วนตัวและความปลอดภัย

LITE ให้ความสำคัญกับการใช้งานไฟล์บนเครื่อง แต่ไม่ได้หมายความว่าระบบมีสิทธิ์เข้าถึงทุกอย่างโดยไม่มีข้อจำกัด

- Local model server และ Web UI bind กับ `127.0.0.1` เท่านั้น
- ไม่ใช้ LLM API บนคลาวด์: โมเดลทำงานผ่าน MLX บน Mac ของคุณ
- SQLite index, cache และ log อยู่ใต้ `~/Library/Application Support/ENDEAVOR_LOCAL_AGENT_LITE/` โดยค่าเริ่มต้น; user-facing files ที่ agent สร้างยังถูกจำกัดไว้ใต้ `workspect/`
- `MCP-RagDoc` อ่าน source documents แบบ read-only และได้รับสิทธิ์เขียนเฉพาะ private runtime state ที่กำหนดไว้
- การเขียนไฟล์ผู้ใช้ถูกจำกัดที่ `workspect/` และป้องกัน path traversal กับ symlink escape
- path และเนื้อหาที่เข้าลักษณะ credential, token, private key หรือ session ถูกบล็อก/ไม่ส่งกลับเข้าบทสนทนา
- `bash` ใช้ได้เฉพาะงานอ่านและค้นหาที่ผ่าน allow-list ไม่ใช่ shell สำหรับรันคำสั่งอิสระ
- การค้นเว็บส่งออกเฉพาะคำค้นและ URL ที่จำเป็นไปยังบริการค้นหา; ไฟล์ในเครื่องไม่ถูกส่งให้ hosted LLM

ยังควรตรวจคำตอบและแหล่งที่มาก่อนใช้ตัดสินใจเรื่องสำคัญ และไม่ควรใส่ข้อมูลลับลงในคำสั่งหรือไฟล์ workspace โดยไม่จำเป็น

## ความต้องการของระบบ

- macOS บน Apple Silicon
- Python 3.11
- หน่วยความจำเพียงพอสำหรับโมเดลที่เลือก (ค่าเริ่มต้นคือโมเดล quantized ขนาด 2B)
- อินเทอร์เน็ตเฉพาะเมื่อใช้ `web_search`

โปรเจกต์ออกแบบและทดสอบกับ `mlx-community/Qwen3.5-2B-OptiQ-4bit` หากเปลี่ยนโมเดล ควรทดสอบการเรียก tools และความปลอดภัยอีกครั้ง เพราะโมเดลแต่ละตัวอาจทำตาม structured tool calling ได้ไม่เท่ากัน

โมเดลนี้ถูกดาวน์โหลดแยกต่างหากเมื่อเปิด server ครั้งแรก ไม่ได้ถูกเก็บใน repository และมี license Apache-2.0 ตาม [model card](https://huggingface.co/mlx-community/Qwen3.5-2B-OptiQ-4bit) ของผู้เผยแพร่โมเดล

## การตั้งค่า

ค่าหลักเปลี่ยนผ่าน environment variables ได้โดยไม่ต้องแก้ source code:

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `LITE_MODEL` | `mlx-community/Qwen3.5-2B-OptiQ-4bit` | model ID สำหรับ MLX server |
| `LITE_MLX_BASE_URL` | `http://127.0.0.1:8090/v1` | URL ของ MLX server ในเครื่อง |
| `LITE_INDEX_ROOTS` | `rag_document/md` | compatibility roots สำหรับ `/index`; `/sync_kb` ใช้ `rag_document/` ทั้ง tree |
| `LITE_DATA_DIR` | `~/Library/Application Support/ENDEAVOR_LOCAL_AGENT_LITE` (หรือโฟลเดอร์เดิม `ENDEAVOR_AGENT_LITE` ถ้ามีอยู่แล้วและยังไม่มีชื่อใหม่) | private runtime state; override รับเฉพาะขอบเขตที่ config อนุญาต |
| `LITE_MLX_PYTHON` | Python ที่กำลังรัน Agent Lite | interpreter สำหรับเปิด local MLX server |
| `LITE_MAX_TOOL_STEPS` | `10` | จำนวนรอบเรียก tool สูงสุดต่อคำถาม |
| `LITE_UI_PORT` | `8766` | port ของ Web UI |

## โครงสร้างการทำงานโดยย่อ

```mermaid
flowchart LR
    U[ผู้ใช้] --> I[CLI หรือ Web UI]
    I --> A[ReAct agent / LangGraph]
    A --> M[MLX local model]
    A --> G[Guardrails]
    G --> T[Tools]
    T --> R[MCP-RagDoc]
    R --> K[rag_document + SQLite]
    T --> L[Desktop/Documents local search]
    T --> D[read_file / read_image / PDF to Text]
    T --> W[web_search]
    T --> A
```

Agent จะเลือกตอบโดยตรงหรือเรียก tool ตามงานที่ถาม Guardrails จำกัดจำนวนรอบ ป้องกันการเรียกซ้ำ จำกัดขนาดผลลัพธ์ และตรวจทั้ง path กับข้อมูลที่อาจเป็นความลับ ก่อนสรุปคำตอบกลับเป็นภาษาไทย

## การทดสอบ

รัน regression suite แบบ deterministic:

```bash
python -m pytest test_suite -q
```

บางชุดเป็น integration หรือ live-model test ซึ่งต้องเปิด MLX server และมีเงื่อนไขตามที่ระบุไว้ในไฟล์ test นั้น

สำหรับการพัฒนา ใช้ `requirements-dev.txt`; dependency ทั้งหมดถูก resolve ใน `uv.lock` และมี hash ใน requirements files เพื่อให้ติดตั้ง release เดิมซ้ำได้

## การรายงานช่องโหว่และการร่วมพัฒนา

โปรดอ่าน [SECURITY.md](SECURITY.md) เพื่อรายงานช่องโหว่เป็นการส่วนตัว และ [CONTRIBUTING.md](CONTRIBUTING.md) สำหรับการตั้งค่า development กับกติกาการส่ง pull request

## License

MIT License + Commons Clause — see [LICENSE](LICENSE).

ใช้ส่วนตัวและเพื่อการเรียนรู้ได้อย่างอิสระ
ห้ามนำไปใช้เชิงพาณิชย์โดยไม่ได้รับอนุญาตจากผู้พัฒนา

## ผู้พัฒนา

Poomwat Jarussri

GitHub: [halochamp](https://github.com/halochamp)

Email: champoomwat@gmail.com
