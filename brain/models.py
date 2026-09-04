"""
Domain-neutral data models for the Universal Brain.

Nothing in this file may reference a specific domain (no "SQLi", "endpoint",
"repo", etc). If a field name only makes sense for one project type, it does
not belong here — it belongs in that project's adapter payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4


def _id() -> str:
    return uuid4().hex[:12]


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    EXECUTED = "executed"
    FAILED = "failed"
    REJECTED = "rejected"  # rejected by Brain before execution (validation failure)


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class Goal:
    """What the connected project wants the Brain to accomplish right now."""
    description: str
    success_criteria: str = ""
    constraints: list[str] = field(default_factory=list)
    id: str = field(default_factory=_id)


@dataclass
class WorldModel:
    """
    The Brain's current belief about the state of the environment.
    A plain, growable fact store — the Brain does not know what the facts
    *mean* domain-wise, only that they exist and can be reasoned over.
    """
    facts: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time)

    def update(self, new_facts: dict[str, Any]) -> None:
        self.facts.update(new_facts)
        self.last_updated = time()

    def snapshot(self) -> dict[str, Any]:
        return dict(self.facts)


@dataclass
class Action:
    """A candidate or executed action. `kind` and `params` are opaque to the
    Brain — only the ProjectAdapter knows how to run them."""
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    predicted_outcome: str = ""   # exploratory actions: what you expect to see, no hypothesis attached
    predicted_if_true: str = ""   # counterfactual: expected observation if tests_hypothesis is TRUE
    predicted_if_false: str = ""  # counterfactual: expected observation if tests_hypothesis is FALSE
    tests_hypothesis: str | None = None  # Hypothesis.id (or unresolved statement text) this action probes
    cost: float = 1.0  # relative resource/time cost; adapters may set higher for expensive actions
    id: str = field(default_factory=_id)
    status: ActionStatus = ActionStatus.PROPOSED


@dataclass
class Hypothesis:
    """A candidate explanation the Brain is entertaining. Confidence is
    updated ONLY by deterministic code (hypotheses.update_from_observation),
    never directly overwritten by an LLM call — the LLM proposes new
    hypotheses and initial confidence; belief revision from evidence is the
    Brain's job, not the LLM's (Principle: LLM is a component, not the
    Brain)."""
    statement: str
    confidence: float = 0.5  # 0..1
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    supporting_evidence: list[str] = field(default_factory=list)  # observation/action ids
    contradicting_evidence: list[str] = field(default_factory=list)
    id: str = field(default_factory=_id)
    created_step: int = 0
    updated_step: int = 0
    # Phase 6: every confidence value this hypothesis has ever held, in
    # order (opening prior first). Distinct from supporting/contradicting
    # evidence counts - this is the TRAJECTORY, which is what lets
    # reasoning_quality.py detect flip-flopping (repeated direction
    # reversals) as opposed to a clean monotonic march toward a verdict.
    confidence_history: list[float] = field(default_factory=list)


@dataclass
class UncertaintyState:
    """Deterministic, code-computed summary of how settled the Brain's
    current beliefs are. Not an LLM output — see uncertainty.py."""
    overall: float = 1.0  # 0 = fully resolved, 1 = maximally uncertain
    open_questions: list[str] = field(default_factory=list)


@dataclass
class Observation:
    """Raw result of executing an Action."""
    action_id: str
    result: Any
    success: bool
    timestamp: float = field(default_factory=time)
    notes: str = ""


@dataclass
class Decision:
    """Record of a choice the Brain made, for inspectability (Principle 13)."""
    chosen_action: Action
    alternatives_considered: list[Action]
    rationale: str
    id: str = field(default_factory=_id)
    timestamp: float = field(default_factory=time)


@dataclass
class WorkingState:
    """
    Everything alive for the current task only. This is NOT a memory tier —
    it is a plain object that lives for one Brain.run() call and is discarded
    or compressed into Experience at the end. (See design-notes.md: we
    deliberately did not build a third memory subsystem for this.)
    """
    goal: Goal
    world_model: WorldModel = field(default_factory=WorldModel)
    actions_taken: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    uncertainty: UncertaintyState = field(default_factory=UncertaintyState)
    uncertainty_history: list[float] = field(default_factory=list)
    active_strategy: str = "balanced"
    strategy_switches: int = 0
    strategies_tried: list[str] = field(default_factory=list)  # explicit history, not log-string parsing
    strategy_log: list[str] = field(default_factory=list)  # human-readable switch rationale, inspectable
    escalations: int = 0  # Phase 6: times meta.should_escalate() fired this run
    quality_report: Any = None  # Phase 6: ReasoningQualityReport, set after run() completes
    step: int = 0
    stopped: bool = False
    stop_reason: str = ""
    # Phase 8: which active cross-project Principles were actually surfaced
    # this run (inspectable, Principle 13) and which Hypothesis a Principle
    # directly seeded (hypothesis_id -> principle_id), so a later confirm/
    # reject outcome can be credited back to the Principle that predicted it
    # (see PrincipleStore.record_outcome, wired from core.py).
    principles_considered: list[str] = field(default_factory=list)
    principle_seeds: dict[str, str] = field(default_factory=dict)
    # Phase 12: a semantically-merged proposal's own statement text ->
    # the id of the existing ACTIVE Hypothesis it was folded into, instead
    # of becoming a rival object splitting future evidence (see
    # HypothesisEngine._merge_into). Inspectable (Principle 13), same
    # pattern as principle_seeds above.
    hypothesis_merges: dict[str, str] = field(default_factory=dict)
    # Phase 13: a hypothesis id that WOULD have been confirmed by its own
    # evidence -> the id of the already-CONFIRMED hypothesis it directly
    # contradicts, blocking that confirmation instead of letting Brain
    # hold two mutually-exclusive "confirmed" beliefs at once (see
    # HypothesisEngine._find_confirmed_contradiction). Cleared once the
    # blocking hypothesis is no longer CONFIRMED (core.py).
    contradictions: dict[str, str] = field(default_factory=dict)
    # Phase 14: hypothesis id -> id of an ACTIVE hypothesis it would
    # contradict if both were confirmed (a live, unresolved fork, not yet
    # a correctness violation like `contradictions` above). Used by
    # DecisionEngine to prioritize the action that would help resolve the
    # fork (see contradiction.find_active_tensions).
    active_tensions: dict[str, str] = field(default_factory=dict)


@dataclass
class Experience:
    """A compressed, evaluated record worth remembering. This is the unit
    stored in either BrainMemory (general) or ProjectMemory (project-scoped)."""
    summary: str
    lesson: str
    importance: float  # 0..1, see memory.importance for scoring
    tags: list[str] = field(default_factory=list)
    project_id: str | None = None  # None => general/cross-project (Brain Memory)
    id: str = field(default_factory=_id)
    timestamp: float = field(default_factory=time)


@dataclass
class Principle:
    """A higher-level generalization consolidated from multiple similar
    Experience records (Phase 4: Memory Consolidation). Not written during
    a run — produced by a separate, periodic consolidation pass over
    accumulated Experience history. See consolidation.py."""
    statement: str
    supporting_experience_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5  # how consistently the source experiences agreed
    project_id: str | None = None
    id: str = field(default_factory=_id)
    timestamp: float = field(default_factory=time)
    # Phase 6: set when a LATER consolidation run produces a Principle
    # covering substantially the same evidence, so the older one is
    # superseded rather than left indistinguishable from a still-current
    # one. None means "still in force." Old records are never deleted
    # (Principle 13: inspectable) - PrincipleStore.active() is what
    # actually filters these out for a caller that only wants current
    # guidance.
    superseded_by: str | None = None
