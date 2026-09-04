"""
Meta-Reasoning Engine (Phase 5).

Everything through Phase 4 reasons about the STATE OF THE WORLD (what's
true, how confident to be). This is the first module that reasons about
the STATE OF THE BRAIN'S OWN REASONING PROCESS this run: is the Challenger
having to intervene a lot? Is the calibration tracker showing
overconfidence? Is a hypothesis about to cross the confirmation threshold
on thin evidence? These are process-quality signals, not world-state
signals, and they're used for two things:

  1. Deciding when to escalate to a stronger (slower/costlier) LLM for a
     step, via `should_escalate` — mirrors agent65's actual two-model
     setup (fast model for the hot loop, stronger model for high-stakes
     checks).
  2. Feeding controller.py's stall handling: `should_change_strategy` is
     what actually authorizes core.py to try a different Strategy instead
     of just stopping outright.

Deliberately no LLM call anywhere in this file — meta-reasoning about
whether the LLM's outputs have been trustworthy so far obviously cannot
itself be delegated back to that same LLM without becoming circular.
"""
from __future__ import annotations

from .calibration import CalibrationTracker
from .models import HypothesisStatus, WorkingState

# A hypothesis this close to the confirm threshold, on this few pieces of
# evidence, is exactly the "about to draw a conclusion" moment Principle 6
# cares about most - worth a stronger model's judgment before finalizing.
NEAR_CONFIRMATION_THRESHOLD = 0.85
NEAR_CONFIRMATION_MAX_EVIDENCE = 2

# Phase 8's PrincipleRetriever surfacing zero relevant Principles on the
# very first step means Brain has no accumulated cross-project experience
# to lean on for this kind of situation at all - the one moment where
# spending a stronger model's judgment on framing the problem well is
# worth the most, per-domain, exactly once. Checking only at step 1 (not
# every step) is deliberate: a domain that stays unfamiliar for the whole
# run already has other signals covering it (challenger interventions,
# overconfidence); re-firing this same trigger every step would just be a
# constant cost tax with no new information after the first look.
UNFAMILIAR_DOMAIN_STEP = 1


def compute_signals(
    state: WorkingState,
    calibration_tracker: CalibrationTracker,
    relevant_principle_count: int | None = None,
) -> dict:
    """`relevant_principle_count` (Phase 8): how many active cross-project
    Principles the PrincipleRetriever judged relevant to the CURRENT step's
    situation. None (the default) means "the caller isn't tracking this" —
    distinct from 0 ("tracked, and genuinely none found") — so a core.py
    call site that predates Phase 8 gets the exact same signals dict as
    before, and the new escalation trigger below simply never fires for
    it."""
    challenger_interventions = sum(1 for d in state.decisions if "CHALLENGED" in d.rationale)
    rejected_actions = sum(1 for a in state.actions_taken if a.status.value == "rejected")

    return {
        "challenger_interventions": challenger_interventions,
        "rejected_actions": rejected_actions,
        "brier_score": calibration_tracker.brier_score(),
        "is_overconfident": calibration_tracker.is_overconfident(),
        "hypothesis_near_confirmation": _any_hypothesis_near_confirmation(state),
        "unfamiliar_domain_first_look": (
            relevant_principle_count is not None
            and relevant_principle_count == 0
            and state.step == UNFAMILIAR_DOMAIN_STEP
        ),
        # Phase 13: Brain's own belief state is internally inconsistent
        # (a hypothesis with earned-enough evidence to confirm is being
        # blocked by an already-CONFIRMED contradiction) — this needs
        # stronger reasoning to resolve, not just more of the same cheap
        # planning loop that got it into this state.
        "unresolved_contradiction": bool(state.contradictions),
    }


def should_escalate(signals: dict) -> bool:
    """Escalate to the stronger model this step if the process looks
    unreliable so far, a high-stakes belief is about to be finalized,
    (Phase 8) this is the first step of a run where Brain Memory has
    nothing relevant to offer, or (Phase 13) two of Brain's own
    hypotheses have collided into a direct contradiction — "ask the
    stronger AI for support" is exactly as valuable when Brain's own
    beliefs disagree with each other as when its process looks shaky. Any
    one trigger is enough — these are rare, cheap-to-check events, not a
    weighted score with tuning to get wrong."""
    if signals["hypothesis_near_confirmation"]:
        return True
    if signals["challenger_interventions"] >= 2:
        return True
    if signals["is_overconfident"]:
        return True
    if signals.get("unfamiliar_domain_first_look"):
        return True
    if signals.get("unresolved_contradiction"):
        return True
    return False


def should_change_strategy(state: WorkingState, max_switches: int) -> bool:
    """Authorizes core.py to try a different Strategy instead of stopping
    outright on a detected stall. Bounded by max_switches so this can
    never turn into an infinite loop of strategy-hopping."""
    return state.strategy_switches < max_switches


def _any_hypothesis_near_confirmation(state: WorkingState) -> bool:
    for h in state.hypotheses:
        if h.status != HypothesisStatus.ACTIVE:
            continue
        if (
            h.confidence >= NEAR_CONFIRMATION_THRESHOLD
            and len(h.supporting_evidence) <= NEAR_CONFIRMATION_MAX_EVIDENCE
        ):
            return True
    return False
