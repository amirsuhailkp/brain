"""
Live Quality Feedback (Phase 16).

The README has flagged this gap since Phase 6 — quoting its own words:
"The Phase 6 quality report is read-only mid-run (logged, and used only
at the memory-write step at the end) - it does not yet feed back into
in-run decision-making (e.g. becoming more conservative the moment
quality starts dropping)."

reasoning_quality.build_report() is called exactly once, after the loop
ends — a retrospective pass. Nothing in the live loop ever reads a running
quality picture, which means:

  - A run where quality deteriorates sharply (oscillating hypotheses,
    repeated challenger overrides, overconfidence) keeps proposing and
    deciding at full speed even as its own reasoning process is signalling
    it's becoming unreliable.
  - The live strategy-switch trigger (meta.should_change_strategy) already
    catches STALLS (uncertainty plateau), but says nothing about process
    quality: a Brain that is actively producing hypotheses and taking
    actions while being systematically overconfident and ignoring its own
    self-critique gets no additional signal asking it to slow down and
    look harder before acting.

This module adds a lightweight, step-level quality picture that the live
loop can act on. It deliberately does NOT replicate or replace the full
retrospective ReasoningQualityReport — that remains the authoritative
whole-run assessment, unchanged. This is a narrower, cheaper read that
answers one question: "right now, does the process look shaky enough that
Brain should be more conservative on its next decision?"

CONSERVATIVE MODE: when quality drops below the threshold, the live loop
switches to a higher-bar confirmation threshold for decision scoring AND
signals meta.py to escalate the next planning step. It does NOT switch
strategy (that's the stall handler's job, already working) and does NOT
change hypothesis belief-update math (that's calibration-tracked). The
only behavioral change is:

  - DecisionEngine: the DUPLICATE_PENALTY is increased (less willing to
    burn a step trying something that already failed), and the info_gain
    weight is boosted (demand more expected value before committing to an
    action). This is "quality conservatism" — not stopping, just raising
    the bar on what it's willing to do next.
  - meta.compute_signals: adds a "live_quality_degraded" signal that
    triggers escalation to the strong model, so the plan for what to test
    next gets stronger reasoning behind it when the process looks shaky.

Why deterministic, not LLM-based:
Same reasoning as meta.py and reasoning_quality.py — judging whether the
LLM's outputs have been trustworthy so far cannot be delegated to that
same LLM. Everything here is a count of things already tracked in
WorkingState; no new data is needed.

Conservative mode is deliberately sticky-downward (triggers when the count
first crosses the threshold) but re-evaluated every step — if the run
genuinely stabilizes (e.g. a strategy switch clears the oscillating
hypotheses and challenger stops intervening), the signal clears
automatically and Brain returns to normal operation. It is not a
one-way ratchet.
"""
from __future__ import annotations

from .calibration import CalibrationTracker
from .models import WorkingState
from .reasoning_quality import _count_reversals, OSCILLATION_REVERSAL_THRESHOLD

# How many of the process-quality signals have to be active at once before
# the live loop enters conservative mode. Set at 2 out of 4 (rather than 1)
# deliberately: a SINGLE challenger intervention or a SINGLE oscillating
# hypothesis is normal variation; it's only when multiple signals fire
# together that the process is genuinely looking unreliable. Tuned against
# the same philosophy as ReasoningQualityReport.is_low_quality (which also
# requires >= 2 flags) so both tools agree on "bad" rather than giving
# contradictory verdicts.
CONSERVATIVE_MODE_SIGNAL_THRESHOLD = 2


def evaluate(state: WorkingState, calibration_tracker: CalibrationTracker) -> dict:
    """Lightweight step-level quality picture. Called once per step in the
    live loop, before DecisionEngine.decide() — not a replacement for the
    full retrospective report (which runs once after the loop ends), just
    the running answer to "is the process looking shaky RIGHT NOW?"

    Returns a plain dict of named bool signals rather than a dataclass —
    fast to compute, easy to unit-test, and passes cleanly through to the
    existing meta.compute_signals dict without a new type import chain."""

    # Signal 1: any hypothesis has oscillated enough to count as pathological
    # (same threshold as the retrospective report, so both agree on "bad").
    oscillating_count = sum(
        1 for h in state.hypotheses
        if _count_reversals(h.confidence_history) >= OSCILLATION_REVERSAL_THRESHOLD
    )

    # Signal 2: self-critique is having to intervene more than once —
    # two+ overrides means the first override didn't fix the underlying
    # tendency.
    challenger_interventions = sum(
        1 for d in state.decisions if "CHALLENGED" in d.rationale
    )

    # Signal 3: calibration is already overconfident (same flag as the
    # retrospective report reads — deliberately not a new computation,
    # just the live-available version of the same tracker).
    is_overconfident = calibration_tracker.is_overconfident()

    # Signal 4: repeated action rejection — proposing actions that the
    # project immediately rejects, more than twice, means the plan-propose-
    # validate loop is broken, not just noisy.
    rejected_actions = sum(
        1 for a in state.actions_taken if a.status.value == "rejected"
    )

    signals = {
        "has_oscillating_hypothesis": oscillating_count > 0,
        "repeated_challenger_overrides": challenger_interventions >= 2,
        "is_overconfident": is_overconfident,
        "repeated_action_rejections": rejected_actions >= 3,
    }

    active_count = sum(1 for v in signals.values() if v)
    signals["conservative_mode"] = active_count >= CONSERVATIVE_MODE_SIGNAL_THRESHOLD
    signals["active_signal_count"] = active_count
    return signals
