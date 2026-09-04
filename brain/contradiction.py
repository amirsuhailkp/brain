"""
Contradiction Detection (Phase 13).

Everything through Phase 12 checks whether two hypotheses are the SAME
claim (similarity.py, used for Principle retrieval and hypothesis
merging). Nothing checks the opposite question: whether two hypotheses
are OPPOSITE claims that cannot both be true. Read through hypotheses.py
and verification.py before writing this — confirmed there is no such
check anywhere. Concretely, before this phase, Brain could hold "the
target is >= 50" and "the target is < 30" as two ACTIVE hypotheses
simultaneously, and if evidence happened to push BOTH past
CONFIRM_THRESHOLD independently (each tested by a different action,
neither test aware of the other), both would become CONFIRMED — an
internally inconsistent belief state that nothing would ever catch or
flag.

Why this can't reuse SimilarityScorer: word-overlap based scorers would
actively get this WRONG in the common case. "the target is >= 50" and
"the target is < 50" share almost every word — a Jaccard or TF-IDF scorer
would call them highly SIMILAR, when they're actually mutually exclusive.
Detecting logical negation/exclusion from text requires understanding
what the words mean, not just whether they overlap, so — unlike
similarity, which had a legitimate (if weaker) lexical approximation —
there's no honest deterministic substitute to offer here as a cheaper
first rung. LLMContradictionScorer is presented as the only
implementation for exactly that reason, not because a cheaper option was
skipped out of laziness.

Deliberately narrow scope: this phase only prevents the sharpest,
unambiguous failure mode — a hypothesis about to be CONFIRMED that
directly contradicts one ALREADY CONFIRMED. Detecting contradictions
between two merely-ACTIVE hypotheses (competing-but-still-unproven
theories, which are often legitimate to hold simultaneously while
evidence accumulates) is explicitly out of scope here — flagging every
ACTIVE-vs-ACTIVE tension would be noisy and, unlike the CONFIRMED case,
doesn't represent an actual internal-consistency violation Brain has
already committed to. A future phase could use the same scorer to surface
those as a *prioritization* signal (test the discriminating action first)
rather than a correctness gate, but that's a different, more speculative
feature than the one built here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .interfaces import LLMInterface
from .models import Hypothesis, HypothesisStatus

# Phase 14: hard cap on pairwise checks per step. Hypothesis counts in
# practice are small (a handful), but this is an O(n^2) LLM-call pattern
# by construction (every ACTIVE hypothesis checked against every other),
# and unlike Phase 13's check (which only runs at the rare moment a
# confirmation is about to happen), this one is proposed to run EVERY
# step. The cap is a blunt but honest safeguard against that cost
# compounding on a run with unusually many simultaneous open hypotheses -
# it means large hypothesis sets get partial coverage rather than an
# unbounded bill, which is the right trade-off for a prioritization
# signal (best-effort by nature) rather than the correctness gate in
# Phase 13 (which must not silently skip a check).
ACTIVE_TENSION_MAX_PAIRS = 10


class ContradictionScorer(ABC):
    """True means "these cannot both be true." False is the safe default
    on any doubt or failure — same reasoning as SimilarityScorer's
    fail-to-zero default: an undetected contradiction leaves today's
    behavior unchanged (no worse than before this phase existed), while a
    false positive would block a legitimately-earned confirmation, which
    is the more damaging mistake to make by accident."""

    @abstractmethod
    def contradicts(self, a: str, b: str) -> bool:
        raise NotImplementedError


class LLMContradictionScorer(ContradictionScorer):
    """Asks the project's own wired LLM whether two statements are
    mutually exclusive. Same seam as LLMSemanticScorer (similarity.py) —
    LLMInterface.propose() — so no new kind of model dependency, and it
    naturally uses whatever provider/model strength the project already
    chose for its own reasoning."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def contradicts(self, a: str, b: str) -> bool:
        a, b = a.strip(), b.strip()
        if not a or not b:
            return False
        prompt = (
            "Two hypotheses from a reasoning system. Judge whether they are MUTUALLY "
            "EXCLUSIVE — i.e. they cannot both be true at the same time (a direct logical "
            "contradiction, not just about different topics or merely unrelated).\n\n"
            f'Hypothesis A: "{a}"\n'
            f'Hypothesis B: "{b}"\n\n'
            'Respond ONLY with JSON: {"contradicts": true or false}.'
        )
        try:
            raw = self.llm.propose(prompt, '{"contradicts": true|false}')
            return bool(raw.get("contradicts", False))
        except Exception:
            return False  # a failed/malformed judgment must never block a real confirmation


def find_active_tensions(
    hypotheses: list[Hypothesis],
    scorer: ContradictionScorer,
    max_checks: int = ACTIVE_TENSION_MAX_PAIRS,
) -> dict[str, str]:
    """Phase 14: the module docstring above calls ACTIVE-vs-ACTIVE
    contradiction detection out of scope for a CORRECTNESS gate (two
    competing-but-unproven theories are often legitimate to hold at
    once) — but that doesn't mean it's not a useful PRIORITIZATION
    signal. If Brain already suspects two of its own active theories are
    mutually exclusive, the action that tests either one is unusually
    valuable to run next: whichever way it comes out, it moves Brain
    closer to resolving a real fork in its own reasoning, not just
    incrementally updating one isolated belief. DecisionEngine (Phase 14
    wiring) uses this to break ties in favor of exactly that kind of
    action.

    Returns hypothesis id -> the id of ONE active hypothesis it tensions
    with (a representative link, not an exhaustive pairing — enough for
    "is this hypothesis part of a live fork", which is all the consumer
    needs). Symmetric: both directions are recorded for any tensioning
    pair found.

    Costs one scorer call per pair checked, same trade-off as any other
    LLMContradictionScorer/LLMSemanticScorer use elsewhere in this
    package - see ACTIVE_TENSION_MAX_PAIRS above for why this is capped
    rather than exhaustive."""
    active = [h for h in hypotheses if h.status == HypothesisStatus.ACTIVE]
    tensions: dict[str, str] = {}
    checks = 0
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            if checks >= max_checks:
                return tensions
            checks += 1
            if scorer.contradicts(a.statement, b.statement):
                tensions[a.id] = b.id
                tensions[b.id] = a.id
    return tensions
