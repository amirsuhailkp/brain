"""
Stop / Continue / Change-Direction Controller (Phase 3, extended Phase 5).

Principle 10: "The Brain must be able to stop." Two independent failure
signatures are checked, because they don't show up the same way in state:

  - is_stalled(): uncertainty hasn't moved over the last several EXECUTED
    steps — the Brain is acting, but not learning anything from it.
  - is_rejection_looping(): the Brain keeps proposing actions the
    ProjectAdapter rejects as invalid, action after action, never even
    reaching an observation. This can't show up in uncertainty_history at
    all (a rejected action never reaches hypotheses.update_from_observation,
    so uncertainty never updates either way) — found via testing
    VagueGuesser against a narrowing-bounds environment: once its fixed
    guess fell outside the current bounds, EVERY subsequent action got
    rejected, and the rejection branch in core.py used to `continue`
    straight past the stall check, so the run silently burned its entire
    step budget spamming an invalid action instead of stopping or
    switching strategy. Needs its own explicit check.

Phase 5: a detected stall/loop no longer means an automatic stop — core.py
tries switching Strategy first (bounded), and only stops for real once
strategies are exhausted. See meta.py / strategy.py.

Phase 18a: third independent failure signature added —
  - is_in_action_rut(): the Brain keeps executing the same action KIND in
    a streak. This is distinct from is_stalled(): a Brain that runs
    "probe" 5 times in a row while hypothesis confidence slowly walks
    toward the threshold passes the stall check (uncertainty IS moving),
    but it's stuck in a rut — it isn't exploring the available action
    space. The rut check catches what the stall check misses: a run where
    uncertainty moves, but only because one action kind is being hammered
    repeatedly instead of any genuine alternative being tried. Correlated
    with low expected information gain on alternatives (if they were more
    valuable, the DecisionEngine would have picked them) but not identical:
    a run that always picks "probe" because probe is genuinely the best
    action isn't in a rut, only one where it keeps winning even after
    repeated observations tell the same story.

    The check is explicitly NOT fired for a streak of "stop" — choosing
    to stop is a deliberate terminal decision, not a rut.
"""
from __future__ import annotations

from .models import ActionStatus, WorkingState

STALL_WINDOW = 4          # look at the last N uncertainty readings
STALL_EPSILON = 0.03      # if they vary by less than this, call it a stall
REJECTION_STREAK_LIMIT = 4  # this many consecutive REJECTED actions in a row counts as stuck
# Phase 18a: a streak of this length using the SAME action kind counts as a
# rut — not long enough to catch deliberate repetition (some actions ARE
# worth running twice), but long enough to flag "the action space isn't
# being explored at all."
ACTION_RUT_STREAK = 5


def is_stalled(state: WorkingState) -> bool:
    history = state.uncertainty_history
    if len(history) < STALL_WINDOW:
        return False
    recent = history[-STALL_WINDOW:]
    return (max(recent) - min(recent)) < STALL_EPSILON


def is_rejection_looping(state: WorkingState) -> bool:
    recent = state.actions_taken[-REJECTION_STREAK_LIMIT:]
    if len(recent) < REJECTION_STREAK_LIMIT:
        return False
    return all(a.status == ActionStatus.REJECTED for a in recent)


def is_in_action_rut(state: WorkingState, streak: int = ACTION_RUT_STREAK) -> bool:
    """True when the last `streak` EXECUTED (not rejected/failed) actions all
    had the same kind. Rejected and failed actions are excluded because:
      - A rejection streak is already caught by is_rejection_looping().
      - Including failed actions would conflate "can't succeed" with "won't
        diversify" — two different problems with two different remedies.
    "stop" is excluded because a stop decision is terminal and correct, not
    a rut, and firing this check on it would cause a spurious strategy switch
    instead of a clean halt."""
    executed = [
        a for a in state.actions_taken
        if a.status == ActionStatus.EXECUTED and a.kind != "stop"
    ]
    if len(executed) < streak:
        return False
    recent = executed[-streak:]
    return len({a.kind for a in recent}) == 1
