"""Shared two-pass deterministic sampling for read_file's bounded-RAM path.

The selection algorithm here is lifted verbatim from tools/read_file.py's
original (pre-streaming) ``_sample_for_coverage``: same seed formula
(``sha256(f"{path}:{total_chars}")``), same bucketed-random index selection,
same header/budget accounting. Only the calling convention changed -- a
caller supplies ``unit_count``/``total_chars`` from its own cheap streaming
pass instead of a fully materialized string, so picking a handful of samples
never requires holding the whole source in memory.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path


def select_sample_indices(unit_count: int, seed_key: str) -> set[int]:
    """Pick the same scattered unit indices the original sampler chose,
    given only a unit count and the exact seed string it used."""
    count = min(18, unit_count)
    rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16))
    selected: set[int] = {0, unit_count - 1}
    for bucket in range(1, max(1, count - 1)):
        start = bucket * unit_count // count
        end = max(start, (bucket + 1) * unit_count // count - 1)
        selected.add(rng.randint(start, end))
    return selected


def build_sample_output(
    path: Path,
    total_chars: int,
    unit_count: int,
    limit: int,
    picked: dict[int, str],
) -> str:
    """Render the same ``"[sample N/M] ..."`` body the original sampler
    produced, given only the already-selected (index -> text) entries."""
    header = f"[{path} — {total_chars:,} chars total; sampled across file, max {limit:,} chars]\n\n"
    budget = max(200, limit - len(header))
    body: list[str] = []
    used = 0
    for index in sorted(picked):
        entry = f"[sample {index + 1}/{unit_count}] {picked[index]}"
        if used + len(entry) + 1 > budget:
            continue
        body.append(entry)
        used += len(entry) + 1
    return header + "\n".join(body)
