import agent
import config
from prompts import SYSTEM_PROMPT


def test_prompt_teaches_bash_fallback_when_rag_is_unavailable():
    assert "ถ้า runtime นี้ไม่มี RAG/MCP tool" in SYSTEM_PROMPT
    assert "ให้หยุดลอง RAG ซ้ำแล้ว fallback ไป `bash`" in SYSTEM_PROMPT
    assert "คำขอที่ระบุ Knowledge Base/RAG/KB/rag_document/knowledge ให้ bash ค้นเฉพาะ" in SYSTEM_PROMPT
    assert "คำขอที่ระบุ project/repository ให้ bash ค้นเฉพาะ" in SYSTEM_PROMPT
    assert "ห้ามใช้ web_search แทนข้อมูล local" in SYSTEM_PROMPT


def test_project_scoped_rag_fallback_uses_project_root():
    command = agent._rag_fallback_bash_command(
        "ค้นหาไฟล์ในโปรเจกต์ที่มีเนื้อหาเกี่ยวกับ landing",
        project_scope=True,
    )
    assert command is not None
    assert str(config.PROJECT_ROOT) in command
    assert "landing" in command
