"""
Reasoning-Quality Evaluation & Self-Monitoring (Phase 6).

Why this is a real gap and not a relabeling of meta.py:

  meta.py's signals (challenger interventions, rejection count, brier
  score, near-confirmation) are computed FRESH every step and consumed
  IMMEDIATELY, in the moment, to decide two things: escalate this step?
  switch strategy on this stall? Once that decision is made, the signal
  is discarded - nothing keeps a running account of how the Brain's
  reasoning actually went across the WHOLE run, and nothing about a run's
  own reasoning is fed into the Experience produced at the end (Principle
  12: learn general principles, not indiscriminately). A run that flailed
  its way to the goal (heavy escalation, hypotheses flip-flopping,
  overconfident) currently produces exactly the same kind of Experience,
  weighted exactly the same way, as a run that reasoned cleanly.

This module closes that gap with one retrospective pass, run once at the
end of Brain.run(), that:

  1. Detects a failure mode none of the existing engines can see:
     hypothesis confidence OSCILLATING (repeated direction reversals)
     instead of moving monotonically toward a verdict. calibration.py
     only ever looks at a single (confidence_at_test, matched) pair per
     test - it has no notion of a hypothesis's shape over time, so a
     hypothesis that bounces 0.5 -> 0.8 -> 0.3 -> 0.75 -> 0.35 looks
     identical to it as one that climbs cleanly 0.5 -> 0.6 -> 0.75 -> 0.9,
     provided the final brier contributions average out similarly.
     Oscillation is itself evidence the evidence-gathering process is
     noisy or the experiments aren't actually distinguishing - worth
     surfacing on its own.
  2. Produces one inspectable ReasoningQualityReport per run (Principle
     13) instead of leaving "was this run's process trustworthy"
     answerable only by re-reading the full decision log by hand.
  3. Feeds back into extraction: core.py downweights and flags the
     episode-summary Experience when quality is poor, so noisy runs
     don't get generalized into Brain Memory with full weight (Principle
     12 applied to the Brain's OWN process, not just world-facts).

Still no LLM call anywhere here, for the same reason as meta.py: judging
whether the LLM's outputs were trustworthy this run can't be delegated
back to that same LLM without becoming circular.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .calibration import CalibrationTracker
from .models import WorkingState

# A hypothesis needs at least this many recorded confidence points before
# "oscillation" is even a meaningful question - two points can only ever
# go up or down once, that's not flip-flopping, that's just an update.
MIN_HISTORY_FOR_OSCILLATION = 3
# This many direction reversals in one hypothesis's confidence trajectory
# counts as pathological flip-flopping rather than ordinary noisy revision.
OSCILLATION_REVERSAL_THRESHOLD = 2


@dataclass
class ReasoningQualityReport:
    """Retrospective, whole-run summary of how trustworthy the Brain's own
    reasoning process was - distinct from what it CONCLUDED about the
    world. Attached to WorkingState.quality_report after run()."""

    brier_score: float | None
    is_overconfident: bool
    challenger_interventions: int
    rejected_actions: int
    strategy_switches: int
    escalations: int
    conservative_mode_steps: int = 0  # Phase 16: how many steps ran in conservative mode
    oscillating_hypotheses: list[str] = field(default_factory=list)  # hypothesis ids
    flags: list[str] = field(default_factory=list)  # human-readable, inspectable issues
    overall_quality: float = 1.0  # 0 = reasoning process looked unreliable, 1 = clean

    @property
    def is_low_quality(self) -> bool:
        return len(self.flags) >= 2 or self.overall_quality < 0.5


def _count_reversals(history: list[float]) -> int:
    """How many times the confidence trajectory changed direction (went
    up then down, or down then up). A clean monotonic climb toward a
    verdict scores 0 regardless of how many steps it took."""
    if len(history) < MIN_HISTORY_FOR_OSCILLATION:
        return 0
    deltas = [b - a for a, b in zip(history, history[1:]) if b != a]
    reversals = 0
    for prev, cur in zip(deltas, deltas[1:]):
        if (prev > 0) != (cur > 0):
            reversals += 1
    return reversals


def _find_oscillating_hypotheses(state: WorkingState) -> list[str]:
    return [
        h.id
        for h in state.hypotheses
        if _count_reversals(h.confidence_history) >= OSCILLATION_REVERSAL_THRESHOLD
    ]


def build_report(state: WorkingState, calibration_tracker: CalibrationTracker) -> ReasoningQualityReport:
    challenger_interventions = sum(1 for d in state.decisions if "CHALLENGED" in d.rationale)
    rejected_actions = sum(1 for a in state.actions_taken if a.status.value == "rejected")
    oscillating = _find_oscillating_hypotheses(state)
    overconfident = calibration_tracker.is_overconfident()
    brier = calibration_tracker.brier_score()

    flags: list[str] = []
    if oscillating:
        flags.append(
            f"{len(oscillating)} hypothesis(es) flip-flopped confidence direction "
            f"{OSCILLATION_REVERSAL_THRESHOLD}+ times instead of converging"
        )
    if overconfident:
        flags.append("stated confidence ran meaningfully higher than observed match rate")
    if challenger_interventions >= 2:
        flags.append(f"self-critique had to override the Decision Engine's pick {challenger_interventions} times")
    if state.strategy_switches > 0:
        flags.append(f"had to abandon the initial strategy {state.strategy_switches} time(s)")
    if rejected_actions >= 3:
        flags.append(f"{rejected_actions} proposed actions were invalid and rejected before execution")
    # Phase 16: if the live quality gate was active for more than a third of
    # the run, that's worth flagging — it means the process was in a
    # degraded state for a sustained period, not just a transient hiccup.
    conservative_mode_steps = getattr(state, "conservative_mode_steps", 0)
    max_steps_estimate = max(state.step, 1)
    if conservative_mode_steps > 0 and conservative_mode_steps / max_steps_estimate >= 0.33:
        flags.append(
            f"live quality gate was active for {conservative_mode_steps}/{state.step} steps "
            f"({conservative_mode_steps / max_steps_estimate:.0%} of the run) — process was "
            f"in conservative mode for a sustained period"
        )

    # Simple additive penalty, not a tuned weighted score - same philosophy
    # as decision.py and meta.py: each penalty is individually explainable,
    # not a black-box number nobody can audit.
    penalty = 0.0
    penalty += 0.15 * len(oscillating)
    penalty += 0.2 if overconfident else 0.0
    penalty += 0.1 * min(challenger_interventions, 3)
    penalty += 0.1 * state.strategy_switches
    penalty += 0.05 * min(rejected_actions, 4)
    overall_quality = max(0.0, 1.0 - penalty)

    return ReasoningQualityReport(
        brier_score=brier,
        is_overconfident=overconfident,
        challenger_interventions=challenger_interventions,
        rejected_actions=rejected_actions,
        strategy_switches=state.strategy_switches,
        escalations=state.escalations,
        conservative_mode_steps=conservative_mode_steps,
        oscillating_hypotheses=oscillating,
        flags=flags,
        overall_quality=round(overall_quality, 3),
    )
