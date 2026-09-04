"""
Prior-Knowledge Retrieval & Abductive Seeding (Phase 8).

Everything through Phase 7 was REACTIVE: Brain forms hypotheses only after
perceiving the world and calling the planner, and Brain Memory's
consolidated Principles (consolidation.py) were produced but never read
back by anything in the live loop — a real, verified dead end, not a
hypothetical gap (grep brain/core.py, brain/planning.py,
brain/hypotheses.py for "Principle" before this phase: zero retrieval
call sites).

This module closes that loop, and does the one new thing it enables:

  1. PrincipleRetriever - given the project's OWN description of "what's
     relevant right now" (ProjectAdapter.principle_tags, Phase 8), find
     which accumulated cross-project Principles apply, ranked by tag
     overlap first and Principle confidence second. Domain-neutral: this
     file has no idea what a "tag" means, same as consolidation.py.

  2. seed_hypotheses_from_principles() - turn the single most relevant
     Principles into ACTUAL Hypotheses, before any observation exists.
     This is the concrete mechanism behind "expect where the problem will
     be before you've looked" (any domain, not just security) - what was
     previously only possible via a lucky first LLM guess is now informed
     by every past project's accumulated, outcome-tested experience.

  Seeded hypotheses open at a DELIBERATELY damped confidence relative to
  the source Principle - a Principle earned its confidence by being
  consistently true ACROSS past situations; it hasn't yet been tested
  IN THIS ONE. Treating it as equivalent evidence would let old data
  silently outweigh this run's own observations. Damping keeps Principle
  6 (plausible, not confirmed) in force for priors exactly as it already
  is for LLM-proposed hypotheses.
"""
from __future__ import annotations

from .consolidation import PrincipleStore
from .models import Hypothesis, Principle, WorkingState
from .similarity import LexicalOverlapScorer, SimilarityScorer

# A seeded hypothesis is a prior, not evidence. Capping well below
# CONFIRM_THRESHOLD (hypotheses.py, 0.9) means a Principle alone can never
# masquerade as a confirmed belief - it still has to survive real
# observations in THIS run, same gate every LLM-proposed hypothesis passes
# through.
MAX_SEEDED_CONFIDENCE = 0.6
MAX_PRINCIPLES_PER_RUN = 5


class PrincipleRetriever:
    """Reads active Principles from a PrincipleStore and ranks them against
    the current situation's tags. Pure retrieval - never mutates a
    Principle's confidence (see PrincipleStore.record_outcome for the
    write side, Phase 9).

    `scorer` (Phase 11): defaults to LexicalOverlapScorer, which reproduces
    the exact word-overlap-against-statement-text matching this class
    shipped with originally - existing behavior, unchanged, if the caller
    doesn't pass anything. Pass a TfidfCosineScorer or LLMSemanticScorer
    (similarity.py) to let retrieval find Principles that describe the
    same tactic in different WORDS than the project's own tags use -
    exact string/word matching can never do that by construction."""

    def __init__(self, store: PrincipleStore, scorer: SimilarityScorer | None = None):
        self.store = store
        self.scorer = scorer or LexicalOverlapScorer()

    def relevant(
        self, query_tags: list[str], limit: int = MAX_PRINCIPLES_PER_RUN
    ) -> list[Principle]:
        if not query_tags:
            return []  # nothing to match against - project didn't opt in, or genuinely no context yet

        scored: list[tuple[float, float, Principle]] = []
        for p in self.store.active():
            overlap = self.scorer.score_tags(query_tags, p.statement)
            if overlap <= 0.0:
                continue
            scored.append((overlap, p.confidence, p))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [p for _, _, p in scored[:limit]]


def seed_hypotheses_from_principles(
    state: WorkingState, principles: list[Principle]
) -> None:
    """Only ever called once per run, and only when state.hypotheses is
    still empty (core.py enforces both) - this is priors filling a genuine
    void, never overriding or out-voting anything the LLM or an
    observation has already produced. Recording state.principle_seeds is
    what lets a later confirm/reject be credited back to the Principle
    (Phase 9)."""
    existing = {h.statement.strip().lower() for h in state.hypotheses}
    for p in principles:
        statement = p.statement.strip()
        if not statement or statement.lower() in existing:
            continue
        confidence = min(MAX_SEEDED_CONFIDENCE, p.confidence)
        hyp = Hypothesis(
            statement=statement,
            confidence=confidence,
            created_step=state.step,
            confidence_history=[confidence],
        )
        state.hypotheses.append(hyp)
        state.principle_seeds[hyp.id] = p.id
        existing.add(statement.lower())
