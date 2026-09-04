"""
Toy env #1: guess a hidden number in a range using higher/lower feedback.

Purpose: a minimal environment to prove the Brain's cognitive cycle works
end to end with ZERO domain-specific code in brain/. This env only knows
about numbers and ranges — nothing about cybersecurity, code, or research.
"""
from __future__ import annotations

import random

from brain.interfaces import ProjectAdapter
from brain.models import Action, WorldModel


class GuessNumberAdapter(ProjectAdapter):
    project_id = "toy-guess-number"

    def __init__(self, low: int = 1, high: int = 100, seed: int | None = None):
        rng = random.Random(seed)
        self.target = rng.randint(low, high)
        self.low_bound = low
        self.high_bound = high
        self.guesses = 0
        self.solved = False

    def perceive(self) -> dict:
        return {
            "low_bound": self.low_bound,
            "high_bound": self.high_bound,
            "guesses_made": self.guesses,
            "solved": self.solved,
        }

    def available_actions(self, world_model: WorldModel) -> list[str]:
        if world_model.facts.get("solved"):
            return ["stop"]
        return ["guess", "stop"]

    def validate(self, action: Action, world_model: WorldModel):
        if action.kind == "guess":
            n = action.params.get("number")
            if not isinstance(n, int):
                return False, "params.number must be an int"
            lo, hi = world_model.facts["low_bound"], world_model.facts["high_bound"]
            if not (lo <= n <= hi):
                return False, f"number {n} outside current bounds [{lo},{hi}]"
        return True, "ok"

    def execute(self, action: Action):
        if action.kind == "stop":
            return True, "stopped"
        n = action.params["number"]
        self.guesses += 1
        if n == self.target:
            self.solved = True
            return True, "correct"
        elif n < self.target:
            self.low_bound = max(self.low_bound, n + 1)
            return True, "higher"
        else:
            self.high_bound = min(self.high_bound, n - 1)
            return True, "lower"

    def is_goal_met(self, world_model: WorldModel) -> bool:
        return bool(world_model.facts.get("solved"))
