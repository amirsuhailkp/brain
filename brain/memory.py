"""
Minimal memory backend + importance scoring.

Deliberately simple for Phase 1: append-only JSONL, filtered by tag on read.
No embeddings, no vector store, no consolidation job yet — those are Phase 4
(Learning) and should only be added once we can show *this* isn't enough.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .interfaces import MemoryBackend
from .models import Experience

IMPORTANCE_THRESHOLD = 0.35  # experiences scoring below this are not stored


def score_importance(
    *,
    novelty: float,
    generality: float,
    surprise: float,
    goal_impact: float,
) -> float:
    """
    Combine the factors the spec calls out (novelty, generality, surprise,
    goal impact) into a single 0..1 score. Weighted average, generality
    weighted highest since it's what makes something worth cross-project
    reuse in Brain Memory specifically.
    """
    weights = {"novelty": 0.2, "generality": 0.35, "surprise": 0.2, "goal_impact": 0.25}
    raw = (
        novelty * weights["novelty"]
        + generality * weights["generality"]
        + surprise * weights["surprise"]
        + goal_impact * weights["goal_impact"]
    )
    return max(0.0, min(1.0, raw))


class JsonlMemory(MemoryBackend):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def store(self, experience: Experience) -> None:
        if experience.importance < IMPORTANCE_THRESHOLD:
            return  # Principle 4: memory stores useful experience, not everything
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(experience)) + "\n")

    def query(
        self,
        tags: list[str] | None = None,
        limit: int = 10,
        require_all_tags: bool = False,
    ) -> list[Experience]:
        """Retrieve experiences matching the given tags.

        Phase 19d: `require_all_tags=False` (the default) preserves the
        original tag-OR behaviour — any experience sharing at least one
        tag is returned. This is what every existing call site expects.

        `require_all_tags=True` switches to tag-AND: only experiences that
        have ALL of the requested tags are returned. This is dramatically
        more precise when the caller knows exactly what combination they
        need (e.g. planning.py wanting experiences tagged BOTH "probe" AND
        "goal_met", not every probe ever run plus every goal-met run).

        The default stays False so every existing call site is unaffected.
        Callers that want AND semantics must opt in explicitly."""
        if not self.path.exists():
            return []
        records: list[Experience] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if tags:
                    exp_tags = set(d.get("tags", []))
                    req_tags = set(tags)
                    if require_all_tags:
                        if not req_tags.issubset(exp_tags):
                            continue
                    else:
                        if not (req_tags & exp_tags):
                            continue
                records.append(Experience(**d))
        records.sort(key=lambda e: e.timestamp, reverse=True)
        return records[:limit]
