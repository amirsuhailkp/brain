"""
Verification-Time Confirmation Review (Phase 15).

Read through meta.py, verification.py, and hypotheses.py before writing
this — confirmed the exact gap the README had flagged since Phase 5 and
left open through Phase 14: `meta.should_escalate` already detects the
"about to draw a conclusion on thin evidence" moment
(`hypothesis_near_confirmation`) and escalates the NEXT planning step to
the strong model — but the confirmation itself, decided one call earlier
inside `hypotheses.update_from_observation`, is pure arithmetic
(`verification.can_confirm`) and never sees a model at all, strong or
otherwise. Escalation was real, but it was escalating the wrong step: by
the time the strong model gets a say, the hypothesis has already locked
in as CONFIRMED. This phase moves the check to where it actually matters
— the moment of confirmation itself, not the step after.

Same shape as Phase 13's ContradictionScorer, and for the same reason:
"is this evidence actually enough, or does it just happen to clear a
numeric bar" is a judgment call about the CONTENT of the evidence, not
something a deterministic count-and-threshold check can make — that's
exactly why verification.py's own docstring already says "a real
correlated-evidence discount is future work." A reviewer that reads
what the evidence actually says (not just how many pieces there are) is
that future work, scoped narrowly: it only ever looks at hypotheses
already about to pass the deterministic gate, never overrides it in the
other direction (it cannot confirm something can_confirm() said no to),
so this can only make Brain more conservative, never less.

Deliberately narrow, mirroring Phase 13/14's scoping discipline:
  - Only fires on THIN evidence (<= the same NEAR_CONFIRMATION_MAX_EVIDENCE
    bar meta.py already uses to decide "this is a moment worth a stronger
    model's attention") — a hypothesis confirmed on a long, boring trail
    of consistent evidence doesn't need a second opinion; the whole point
    is catching the "confidence math cleared the bar on the bare minimum
    of evidence" case, not adding a tax to every confirmation forever.
  - Fails OPEN (silently approves) on any error or malformed response,
    same principle as LLMContradictionScorer's own fail-to-False default:
    an undetected review failure leaves today's behavior unchanged (no
    worse than before this phase existed); failing CLOSED would mean a
    single bad LLM call could permanently strand an otherwise well-earned
    confirmation in limbo, which is the more damaging mistake.
  - A hold is a "not yet," never a veto: the hypothesis stays ACTIVE, next
    evidence still writes to the same confidence_history, and it is
    reviewed fresh every time it re-clears the gate — nothing about this
    module can permanently block a hypothesis from ever confirming.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .interfaces import LLMInterface
from .models import Hypothesis

# Same bar meta.py's `hypothesis_near_confirmation` signal already uses —
# deliberately the same number, not a separately-tuned one: this module
# exists to act on exactly the situation that signal was already named
# for, not to invent a second, subtly different definition of "thin."
THIN_EVIDENCE_MAX = 2


class ConfirmationReviewer(ABC):
    """True means "the evidence genuinely supports this, let it confirm."
    False means "hold — not yet." See module docstring for why False is
    never the fail-safe default (that's the reviewer's own job to decide);
    only exceptions/malformed output fail open, inside the LLM
    implementation below."""

    @abstractmethod
    def review(self, statement: str, evidence: list[str]) -> tuple[bool, str]:
        """Returns (approved, reason). `reason` is shown to the person
        (state.confirmation_holds) and to the escalated planner
        (planning.py) when approved is False — a bare boolean would tell
        Brain WHAT happened but not WHY, which breaks Principle 13
        (inspectability) right at the moment it matters most."""
        raise NotImplementedError


class LLMConfirmationReviewer(ConfirmationReviewer):
    """Asks the project's own wired LLM to sanity-check a near-confirmation
    against what the evidence actually says, not just how much of it there
    is. Same seam as LLMContradictionScorer/LLMSemanticScorer —
    LLMInterface.propose() — so this rides on whatever model strength the
    caller already chose (in practice, core.py wires this to the STRONG
    model, same one `meta.should_escalate` already reserves for
    high-stakes moments)."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def review(self, statement: str, evidence: list[str]) -> tuple[bool, str]:
        statement = statement.strip()
        if not statement:
            return True, ""  # nothing to review - never block on a data gap
        evidence_text = "; ".join(str(e) for e in evidence) or "(no evidence details recorded)"
        prompt = (
            "A reasoning system is about to mark the following hypothesis as "
            "CONFIRMED, based on the minimum amount of independent evidence its "
            "own rules require. Judge whether the evidence genuinely supports "
            "this specific claim, or whether it's a coincidence, a confound, or "
            "too thin to be sure yet.\n\n"
            f'Hypothesis: "{statement}"\n'
            f"Supporting evidence (observation ids/results): {evidence_text}\n\n"
            'Respond ONLY with JSON: {"approved": true|false, "reason": "one short sentence"}.'
        )
        try:
            raw = self.llm.propose(
                prompt, '{"approved": true|false, "reason": "..."}'
            )
            approved = bool(raw.get("approved", True))
            reason = str(raw.get("reason", "")).strip()
            return approved, reason
        except Exception:
            # A failed/malformed judgment must never strand an otherwise
            # earned confirmation - fail open, same reasoning as
            # LLMContradictionScorer.
            return True, ""


def needs_review(hypothesis: Hypothesis, max_evidence: int = THIN_EVIDENCE_MAX) -> bool:
    """Thin-evidence gate: only hypotheses confirmed on the bare minimum
    of independent evidence get a second opinion. A hypothesis with a
    long, well-established evidence trail skips review entirely — same
    "don't tax the common case" discipline as every other opt-in scorer
    in this package."""
    return len(hypothesis.supporting_evidence) <= max_evidence
