# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""ui_cli.py — terminal chat loop, using the ui.py visual layer ported from
ENDEAVOR_LOCAL_AGENT_MAX_VLM/ui_cli.py. CLI remains the UI *choice* from
PLAN.md §2 — this only changes how the terminal looks, not the agent logic.

Architecture note (2026-07-23, PLAN.md §Phase 8): the model runs via a
separate mlx_vlm.server process. LangGraph streams each graph-node update,
and agent.run_turn() exposes it through a small on_step callback so this CLI
can keep MAX_VLM's live two-line tool-status spinner.
"""
from __future__ import annotations
import sys

import agent
import config
import indexer
import ui
from tools import TOOLS


def main() -> None:
    ui.print_header(config.MODEL_ID, len(TOOLS))
    messages: list[dict] = []
    while True:
        try:
            user_input = input(f" {ui.C_ORANGE}{ui.BOLD}>{ui.R}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nลาก่อนครับ")
            return
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("ลาก่อนครับ")
            return
        if user_input.lower() == "/index":
            ui.print_user_prompt(user_input)
            with ui.Spinner("กำลังสร้าง RAG index…") as spinner:
                spinner.update_sub("กำลังเตรียมรายการไฟล์…")

                def on_index_progress(event: str, path, stats: dict[str, int]) -> None:
                    if event in {"file", "result"}:
                        spinner.update_sub(ui.index_progress_text(path, stats))
                    elif event == "root":
                        spinner.update_sub(f"กำลังสแกน {path}…")

                try:
                    stats = indexer.build_index(on_progress=on_index_progress)
                except KeyboardInterrupt:
                    spinner.live_print(lambda: print("\n[stopped] ยกเลิกการสร้าง index แล้ว", file=sys.stderr))
                    continue
                except Exception as e:
                    spinner.live_print(lambda: print(f"\n[error] สร้าง index ไม่สำเร็จ: {e}", file=sys.stderr))
                    continue
            ui.print_index_summary(stats)
            ui.print_divider()
            continue

        with ui.Spinner() as spinner:
            try:
                result = agent.run_turn(user_input, messages, on_step=ui.make_on_step(spinner))
            except Exception as e:
                # agent.run_turn() already has its own internal safety net
                # (server-down/timeout/GraphRecursionError all degrade to a
                # normal TurnResult), but this is the second layer in case
                # something outside that (e.g. a bug in agent.py itself)
                # still raises — `messages` is only ever mutated atomically
                # on a successful turn, so a crash here leaves it untouched.
                spinner.live_print(lambda: print(f"\n[error] เกิดข้อผิดพลาดระหว่างประมวลผล: {e}\n", file=sys.stderr))
                continue

        ui.print_agent_response(result.text)
        refs = []
        for search_result in result.web_search_results or []:
            refs.extend(ui.extract_refs_from_search_result(search_result))
        unique_refs = list(dict.fromkeys(refs))
        ui.print_web_refs(unique_refs)
        ui.print_history_bar(messages)
        ui.print_divider()


if __name__ == "__main__":
    main()
