"""
Experience Extraction Engine (Phase 4).

Phases 1-3 only ever produced ONE Experience per Brain.run() — a summary
of the whole episode. That's a reasonable episode-level rollup (kept, see
core.py's _extract_episode_summary), but it can't capture "this one
specific observation was surprising and worth remembering on its own,
even though the run as a whole was unremarkable." This module does that:
called after EVERY observation, decides per-observation whether it's
worth turning into an Experience at all (most aren't — see
memory.IMPORTANCE_THRESHOLD, which still gates what actually gets
written).

This is the concrete mechanism behind the spec's "100 events -> 20
experiences -> 5 principles" pipeline: this file is the 100 -> 20 step
(selective extraction from raw observations). consolidation.py is the
20 -> 5 step (clustering experiences into principles), run separately
and periodically, not inline in the hot loop.
"""
from __future__ import annotations

from .memory import score_importance
from .models import Action, Experience, Hypothesis, Observation, WorkingState


def extract_from_observation(
    state: WorkingState,
    action: Action,
    observation: Observation,
    hyp_before: Hypothesis | None,
    hyp_after: Hypothesis | None,
    project_id: str,
) -> Experience | None:
    """Returns an Experience if this single observation was informative
    enough to remember on its own, else None. Called once per observation,
    immediately after hypotheses.update_from_observation."""

    surprise = _surprise(action, observation, hyp_before)
    novelty = _novelty(hyp_before, hyp_after)
    goal_impact = 0.5 if observation.success else 0.3
    generality = _generality(hyp_after)

    importance = score_importance(
        novelty=novelty, generality=generality, surprise=surprise, goal_impact=goal_impact
    )

    summary = f"Action '{action.kind}' (step {state.step}) -> {observation.result!r}"
    lesson = _lesson(action, observation, hyp_before, hyp_after)

    tags = [action.kind, project_id]
    if hyp_after is not None:
        tags.append(hyp_after.status.value)

    return Experience(
        summary=summary,
        lesson=lesson,
        importance=importance,
        tags=tags,
        project_id=project_id,
    )


def _surprise(action: Action, observation: Observation, hyp_before: Hypothesis | None) -> float:
    """High surprise = the action failed outright, or a hypothesis-testing
    action's prediction didn't hold (a real 'huh, that's not what I
    expected' moment)."""
    if not observation.success:
        return 0.9
    if hyp_before is None:
        return 0.2  # exploratory action succeeding as expected - unremarkable
    actual = str(observation.result).strip().lower()
    if action.predicted_if_false and action.predicted_if_false.strip().lower() in actual:
        return 0.8  # matched the FALSE branch - genuinely surprising if hypothesis looked likely
    return 0.2


def _novelty(hyp_before: Hypothesis | None, hyp_after: Hypothesis | None) -> float:
    """High novelty = a hypothesis's status actually changed (became
    CONFIRMED/REJECTED this step) — that's new, settled information, not
    just an incremental confidence nudge."""
    if hyp_before is None or hyp_after is None:
        return 0.3
    if hyp_before.status != hyp_after.status:
        return 0.9
    if abs(hyp_after.confidence - hyp_before.confidence) > 0.2:
        return 0.5
    return 0.15


def _generality(hyp_after: Hypothesis | None) -> float:
    """Crude proxy, same spirit as Phase 1/2: an event tied to a settled
    (CONFIRMED/REJECTED) hypothesis is more likely to generalize to future
    runs than an in-progress confidence nudge."""
    if hyp_after is not None and hyp_after.status.value != "active":
        return 0.6
    return 0.3


def _lesson(
    action: Action,
    observation: Observation,
    hyp_before: Hypothesis | None,
    hyp_after: Hypothesis | None,
) -> str:
    if not observation.success:
        return (
            f"Action kind '{action.kind}' failed with params {action.params!r}; "
            f"treat as unreliable in similar states."
        )
    if hyp_before is not None and hyp_after is not None and hyp_before.status != hyp_after.status:
        return (
            f"Hypothesis '{hyp_after.statement}' resolved to {hyp_after.status.value} after "
            f"{len(hyp_after.supporting_evidence) + len(hyp_after.contradicting_evidence)} "
            f"observations testing it."
        )
    if hyp_before is not None and action.predicted_if_false:
        actual = str(observation.result).strip().lower()
        if action.predicted_if_false.strip().lower() in actual:
            return (
                f"Prediction for '{hyp_before.statement}' being true did NOT hold; "
                f"evidence pointed the other way."
            )
    return "Routine, expected outcome."
