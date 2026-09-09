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
# Phase 19a: a multiplier on EXPLORATION_BASE_GAIN for exploratory actions
# that touch a world-model key already known to be relevant to at least one
# ACTIVE hypothesis statement. "Relevant" is a simple word-overlap check —
# no embeddings, no LLM — but it's enough to distinguish "read the field
# three of my live hypotheses mention" from "read a field nobody is
# reasoning about." Both are still exploratory (no counterfactual
# predictions), but the first is meaningfully more likely to unblock
# something, and should score higher than a flat 0.15.
EXPLORATION_RELEVANCE_MULTIPLIER = 1.8


def _exploration_gain(action: Action, state: WorkingState) -> float:
    """Phase 19a: weight exploratory actions by how many of their params
    overlap with vocabulary already present in active hypothesis statements.

    The intuition: if three live hypotheses all mention "temperature" and
    the exploratory action reads `params={"field": "temperature"}`, that
    action is meaningfully more likely to return something useful than
    one reading `params={"field": "checksum_b"}` that no hypothesis has
    mentioned. Both are exploratory — neither has a counterfactual — but
    they're not equally valuable.

    Implementation: collect every word from every ACTIVE hypothesis
    statement (lowercase, split on whitespace/punctuation), then check
    how many of the action's params values contain at least one such word.
    If any do, apply the relevance multiplier; otherwise, return the flat
    base gain. This is O(hypotheses * param_values * words) but in
    practice both sides are small (< 10 hypotheses, < 5 params each)."""
    import re
    active_hyps = [h for h in state.hypotheses if h.status.value == "active"]
    if not active_hyps or not action.params:
        return EXPLORATION_BASE_GAIN

    # Vocabulary from all active hypothesis statements
    hyp_words: set[str] = set()
    for h in active_hyps:
        hyp_words.update(w.lower() for w in re.split(r'\W+', h.statement) if len(w) > 2)

    # Words from action param values (stringify everything)
    param_words: set[str] = set()
    for v in action.params.values():
        param_words.update(w.lower() for w in re.split(r'\W+', str(v)) if len(w) > 2)

    if hyp_words & param_words:
        return EXPLORATION_BASE_GAIN * EXPLORATION_RELEVANCE_MULTIPLIER
    return EXPLORATION_BASE_GAIN


def expected_information_gain(action: Action, state: WorkingState) -> float:
    if not action.tests_hypothesis:
        # Phase 19a: exploratory actions are not all equally valuable —
        # one that touches params already mentioned in active hypothesis
        # statements scores higher than one probing something nobody is
        # reasoning about. Both are still exploratory (no counterfactual),
        # but the relevance-weighted score reflects the difference.
        return _exploration_gain(action, state)

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
