"""
Toy env #2: guess a hidden 4-letter word from a fixed vocabulary, given
per-letter feedback (Wordle-style: correct-position / wrong-position / absent).

Deliberately a different *shape* of problem than guess_number.py (discrete
combinatorial search + partial feedback vs numeric bisection) so that
reusing the same Brain core across both is a real test of domain
independence, not a coincidence of two near-identical toys.
"""
from __future__ import annotations

import random

from brain.interfaces import ProjectAdapter
from brain.models import Action, WorldModel

VOCAB = ["LOCK", "CODE", "GOAL", "PLAN", "WORD", "TEST", "BOLD", "MIND"]


class WordLockAdapter(ProjectAdapter):
    project_id = "toy-word-lock"

    def __init__(self, seed: int | None = None):
        rng = random.Random(seed)
        self.target = rng.choice(VOCAB)
        self.attempts: list[dict] = []
        self.solved = False

    def perceive(self) -> dict:
        return {
            "vocab": VOCAB,
            "attempts_made": len(self.attempts),
            "feedback_history": self.attempts,
            "solved": self.solved,
        }

    def available_actions(self, world_model: WorldModel) -> list[str]:
        if world_model.facts.get("solved"):
            return ["stop"]
        return ["guess", "stop"]

    def validate(self, action: Action, world_model: WorldModel):
        if action.kind == "guess":
            w = action.params.get("word")
            if w not in VOCAB:
                return False, f"'{w}' is not in the allowed vocabulary"
        return True, "ok"

    def execute(self, action: Action):
        if action.kind == "stop":
            return True, "stopped"
        guess = action.params["word"]
        feedback = self._score(guess)
        self.attempts.append({"guess": guess, "feedback": feedback})
        if guess == self.target:
            self.solved = True
        return True, feedback

    def is_goal_met(self, world_model: WorldModel) -> bool:
        return bool(world_model.facts.get("solved"))

    def _score(self, guess: str) -> str:
        return "".join(
            "G" if g == t else ("Y" if g in self.target else "-")
            for g, t in zip(guess, self.target)
        )
