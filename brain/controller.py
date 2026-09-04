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
"""
from __future__ import annotations

from .models import ActionStatus, WorkingState

STALL_WINDOW = 4          # look at the last N uncertainty readings
STALL_EPSILON = 0.03      # if they vary by less than this, call it a stall
REJECTION_STREAK_LIMIT = 4  # this many consecutive REJECTED actions in a row counts as stuck


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
