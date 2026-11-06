"""PLAN.md §9.3 vision-cache-size A/B harness (manual, requires a live
mlx_vlm.server -- not collected by pytest, no `test_` prefix).

Usage: start the server yourself with the config you want to measure, find
its PID, then run this script against it.

    python -m mlx_vlm.server --model mlx-community/Qwen3.5-2B-OptiQ-4bit \\
        --port 8090 --host 127.0.0.1 --api-key x [--vision-cache-size N]
    python test_suite/vision_cache_ab_manual.py <server_pid>

Runs the 3 scenarios PLAN.md §9.3 asks for (same image x10, alternating 3
images x4 cycles, 25 unique images) and reports per-request latency,
correctness (a deliberately simple color-naming task, checkable without a
human), and server RSS at checkpoints along the way. Run once against the
default cache size and once with --vision-cache-size 2, then diff the JSON
output -- see PLAN.md §9.3 for the last recorded comparison (2026-07-25).
"""
import base64
import io
import json
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "mlx-community/Qwen3.5-2B-OptiQ-4bit"
COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta",
    "brown", "pink", "gray", "black", "navy", "teal", "maroon", "olive",
    "coral", "salmon", "indigo", "violet", "turquoise", "gold", "silver", "beige", "crimson",
]


def make_image_b64(color: str, seed: int) -> str:
    img = Image.new("RGB", (200, 100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([20 + seed % 5, 20, 80 + seed % 5, 80], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(b64: str, timeout=40):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is the rectangle in this image? Answer with exactly one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "max_tokens": 8,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - t0
    answer = data["choices"][0]["message"]["content"].strip().lower()
    return answer, elapsed


def server_rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024 if out.stdout.strip() else -1.0


def run_suite(server_pid: int):
    results = {"repeat": [], "alternate": [], "unique25": []}
    footprints = {"start": server_rss_mb(server_pid)}

    # Scenario 1: same image, 10 turns
    img = make_image_b64(COLORS[0], 0)
    for _ in range(10):
        answer, elapsed = ask(img)
        results["repeat"].append({"correct": COLORS[0] in answer, "latency": round(elapsed, 2)})
    footprints["after_repeat"] = server_rss_mb(server_pid)

    # Scenario 2: alternate between 3 images, 4 cycles (12 requests)
    imgs3 = [make_image_b64(c, i) for i, c in enumerate(COLORS[:3])]
    for _cycle in range(4):
        for idx, color in enumerate(COLORS[:3]):
            answer, elapsed = ask(imgs3[idx])
            results["alternate"].append({"correct": color in answer, "latency": round(elapsed, 2)})
    footprints["after_alternate"] = server_rss_mb(server_pid)

    # Scenario 3: 25 unique images
    for i, color in enumerate(COLORS):
        b64 = make_image_b64(color, i + 100)
        answer, elapsed = ask(b64)
        results["unique25"].append({"correct": color in answer, "latency": round(elapsed, 2)})
        if i in (4, 9, 14, 19, 24):
            footprints[f"after_unique_{i + 1}"] = server_rss_mb(server_pid)

    return results, footprints


if __name__ == "__main__":
    pid = int(sys.argv[1])
    suite_results, suite_footprints = run_suite(pid)
    print(json.dumps({"results": suite_results, "footprints": suite_footprints}, indent=2))
