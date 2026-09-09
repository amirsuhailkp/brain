"""
The three seams the Brain talks through. Everything domain-specific lives
on the other side of these interfaces — the Brain core never imports a
concrete project adapter or a concrete LLM client.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import Action, WorldModel


class LLMInterface(ABC):
    """A reasoning resource, not the Brain. It proposes; it does not decide.

    Contract: given a prompt and an expected output shape, return a dict
    matching that shape. The Brain is responsible for validating what comes
    back — the LLM is not trusted to self-police.
    """

    @abstractmethod
    def propose(self, prompt: str, schema_hint: str) -> dict[str, Any]:
        """Return structured output (already-parsed dict). Raise on
        unrecoverable failure; the Brain decides what to do with a bad call,
        the LLM adapter does not silently guess."""
        raise NotImplementedError


class ProjectAdapter(ABC):
    """The environment. Owns all domain knowledge, tools, and validation
    rules. The Brain calls these methods and never reaches around them."""

    project_id: str = "unnamed-project"

    @abstractmethod
    def perceive(self) -> dict[str, Any]:
        """Return a flat dict of current-world facts to merge into WorldModel."""
        raise NotImplementedError

    @abstractmethod
    def available_actions(self, world_model: WorldModel) -> list[str]:
        """Return the *kinds* of action currently legal, given current state.
        Used to constrain what the LLM is even allowed to propose.

        Don't include "stop" here — Brain.run() adds it automatically to
        every step's allowed kinds and short-circuits it before validate()/
        execute() are ever called, so it never reaches this adapter at all.
        Just return your real, dispatchable action kinds."""
        raise NotImplementedError

    @abstractmethod
    def validate(self, action: Action, world_model: WorldModel) -> tuple[bool, str]:
        """Return (is_valid, reason). Called before every execution —
        the Brain never trusts an LLM-proposed action blindly.

        action.kind will never be "stop" here — Brain.run() intercepts it
        before this is called."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, action: Action) -> tuple[bool, Any]:
        """Run the action for real (or in the toy env, simulate it).
        Return (success, raw_result).

        action.kind will never be "stop" here — Brain.run() intercepts it
        before this is called."""
        raise NotImplementedError

    def is_goal_met(self, world_model: WorldModel) -> bool:
        """Optional: project-specific goal check. Default: never auto-stops
        (Brain relies on step budget / explicit stop action)."""
        return False

    def project_memory_context(self, query: str) -> list[str]:
        """Optional: return relevant project-memory snippets as strings for
        prompt context. Default: none."""
        return []

    def principle_tags(self, world_model: WorldModel) -> list[str]:
        """Optional (Phase 8): return tags describing the CURRENT situation,
        in the same vocabulary the project uses when tagging its own
        Experiences (see extraction.py call sites). Used to retrieve
        cross-project Principles relevant to what's happening right now —
        the project is the only thing that knows what "relevant" means for
        its own domain, same reasoning as project_memory_context above.

        Default: empty, meaning "retrieve nothing" — a project that never
        overrides this simply doesn't get Principle-seeded priors, which is
        exactly today's behavior for every existing adapter (GuessNumber,
        WordLock, etc.) with zero code changes required on their part."""
        return []

    def deterministic_propose(
        self, state, allowed_kinds: list[str]
    ) -> tuple[list[dict], list] | None:
        """Optional fast path (Phase 7): let the project try to generate
        hypotheses/candidate actions from its OWN domain knowledge (rule
        libraries, playbooks, prior confirmed patterns) BEFORE paying for
        an LLM call. Same return shape as PlanningEngine.reason_and_plan():
        (raw_hypothesis_proposals, candidate_actions).

        Return None (the default) to always use the LLM — this is what
        every project gets unless it overrides this method, so existing
        adapters (GuessNumberAdapter, WordLockAdapter, etc.) are completely
        unaffected. Return an empty ([], []) to explicitly say "nothing
        deterministic to propose this step, still don't call the LLM" —
        distinct from None, which always falls through to the LLM.

        This is intentionally read-only and side-effect-free: it must not
        execute anything or mutate world state, only look things up. Real
        execution still only ever happens through .execute(), after the
        Brain (not this method) has chosen among the candidates."""
        return None


class MemoryBackend(ABC):
    """Storage for Experience records. Two instances exist in a running
    Brain: one for Brain Memory (project_id=None), one for Project Memory."""

    @abstractmethod
    def store(self, experience) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, tags: list[str] | None = None, limit: int = 10, require_all_tags: bool = False) -> list:
        raise NotImplementedError