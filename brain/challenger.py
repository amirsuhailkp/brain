"""
Self-Critique / Challenger (Phase 3).

Deterministic, no LLM call — same design rule as decision.py. Runs AFTER
DecisionEngine picks a chosen_action, and can veto it in favor of the next
best alternative if the choice looks like it's fooling itself. Two checks,
both narrow and explainable rather than a vague "review this critically":

  1. Confirmation-bias check: the chosen action tests an ALREADY-CONFIRMED
     or ALREADY-REJECTED hypothesis. Testing a settled question again
     can't produce new information (Principle 8: prefer informative
     actions) — if an alternative with real information gain exists, use
     it instead.
  2. Non-distinguishing-experiment check: the chosen action claims to test
     a hypothesis but predicted_if_true == predicted_if_false (or both
     empty) — i.e. it can't actually distinguish anything, so calling it
     an "experiment" for that hypothesis is misleading. Same veto logic.

Both checks only fire if a genuinely better alternative was available;
otherwise they just annotate the decision's rationale so the issue is
visible (Principle 13: inspectable), without blocking the Brain when
there's nothing better to do.
"""
from __future__ import annotations

from .information_gain import expected_information_gain
from .models import Decision, HypothesisStatus, WorkingState


def review(decision: Decision, state: WorkingState) -> Decision:
    action = decision.chosen_action
    issue = _find_issue(action, state)
    if issue is None:
        return decision

    better = _best_alternative(decision, state)
    if better is not None:
        new_alternatives = [a for a in [action, *decision.alternatives_considered] if a is not better]
        decision.rationale += (
            f" | CHALLENGED: {issue}; switched to '{better.kind}' "
            f"(higher expected information gain) instead."
        )
        decision.chosen_action = better
        decision.alternatives_considered = new_alternatives
    else:
        decision.rationale += f" | CHALLENGED: {issue}; no better alternative available, proceeding anyway."

    return decision


def _find_issue(action, state: WorkingState) -> str | None:
    if action.tests_hypothesis:
        hyp = next((h for h in state.hypotheses if h.id == action.tests_hypothesis), None)
        if hyp is not None and hyp.status != HypothesisStatus.ACTIVE:
            return f"action targets an already-{hyp.status.value} hypothesis; re-testing it can't be informative"

        if hyp is not None and hyp.status == HypothesisStatus.ACTIVE:
            same_prediction = (
                action.predicted_if_true.strip().lower() == action.predicted_if_false.strip().lower()
            )
            if same_prediction and (action.predicted_if_true or action.predicted_if_false):
                return "predicted_if_true and predicted_if_false are identical; this isn't a real distinguishing experiment"

    return None


def _best_alternative(decision: Decision, state: WorkingState):
    if not decision.alternatives_considered:
        return None
    scored = [
        (expected_information_gain(a, state), a)
        for a in decision.alternatives_considered
        if _find_issue(a, state) is None
    ]
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best_gain, best_action = scored[0]
    current_gain = expected_information_gain(decision.chosen_action, state)
    return best_action if best_gain > current_gain else None
