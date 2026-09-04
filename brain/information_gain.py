"""
Information-Gain Engine (Phase 3).

Replaces the Phase 2 decision.py proxy ("favor actions testing a
hypothesis near 0.5 confidence") with an actual expected-value calculation:

  1. Simulate the SAME confidence-update rule hypotheses.py uses, for both
     possible branches of an action tied to a hypothesis:
       - the "true" branch (observation matches predicted_if_true)
       - the "false" branch (observation matches predicted_if_false)
  2. Compute overall uncertainty in each branch (uncertainty.py, applied to
     a hypothetical hypothesis set with only this one confidence changed).
  3. Weight each branch by its probability under the CURRENT belief
     (P(true) = current confidence, P(false) = 1 - it) and take
     uncertainty_before - expected_uncertainty_after = information gain.

Documented simplification: this doesn't integrate over a joint
distribution across all hypotheses, and doesn't model correlated evidence.
It IS a real expected-value calculation over two explicit, distinguishable
outcomes, not a static "distance from 0.5" heuristic — and it directly
rewards actions with genuinely distinguishing counterfactual predictions
(predicted_if_true != predicted_if_false) over vague ones.

The simulated update in step 1 calls hypotheses.update_confidence() — the
SAME function the real update in hypotheses.update_from_observation()
uses — rather than a separately-maintained copy of the arithmetic. This
was flagged as a duplication risk since Phase 3 and closed in Phase 6's
cleanup pass: previously this file re-implemented the confidence-update
math inline, and the two copies could silently drift apart.
"""
from __future__ import annotations

import copy

from .hypotheses import update_confidence
from .models import Action, Hypothesis, WorkingState
from .uncertainty import compute_uncertainty

EXPLORATION_BASE_GAIN = 0.15  # nominal value for exploratory actions with no tied hypothesis


def expected_information_gain(action: Action, state: WorkingState) -> float:
    if not action.tests_hypothesis:
        return EXPLORATION_BASE_GAIN

    hyp = next((h for h in state.hypotheses if h.id == action.tests_hypothesis), None)
    if hyp is None:
        return EXPLORATION_BASE_GAIN

    if not action.predicted_if_true and not action.predicted_if_false:
        # tied to a hypothesis but no real counterfactual given — can't
        # claim it's specifically informative about this hypothesis.
        return EXPLORATION_BASE_GAIN * 0.5

    if action.predicted_if_true.strip().lower() == action.predicted_if_false.strip().lower():
        # Same predicted string for both branches - no distinguishing
        # power at all for this hypothesis.
        return 0.0

    uncertainty_before = compute_uncertainty(state.hypotheses).overall

    p_true = hyp.confidence
    p_false = 1.0 - hyp.confidence

    uncertainty_if_true = _simulated_uncertainty(state.hypotheses, hyp.id, matched=True)
    uncertainty_if_false = _simulated_uncertainty(state.hypotheses, hyp.id, matched=False)

    expected_uncertainty_after = p_true * uncertainty_if_true + p_false * uncertainty_if_false
    return max(0.0, uncertainty_before - expected_uncertainty_after)


def _simulated_uncertainty(hypotheses: list[Hypothesis], hyp_id: str, matched: bool) -> float:
    """Apply the SAME update rule as hypotheses.update_from_observation
    (via the shared hypotheses.update_confidence() function — no more
    hand-duplicated arithmetic) to a deep copy, then recompute
    uncertainty."""
    sim = copy.deepcopy(hypotheses)
    h = next(x for x in sim if x.id == hyp_id)
    h.confidence = update_confidence(h.confidence, matched)
    return compute_uncertainty(sim).overall
