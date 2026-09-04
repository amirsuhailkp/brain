"""
Hypothesis Engine (Phase 2/3).

Split responsibility on purpose:
  - integrate_new(): takes LLM-proposed {statement, confidence} pairs and
    merges them into WorkingState.hypotheses. This is the only place an
    LLM's confidence number is trusted, and only for a BRAND NEW hypothesis
    (its opening prior) — never to overwrite an existing one.
  - update_from_observation(): deterministic belief revision after an
    action executes. No LLM call. Confidence moves based on whether the
    observation matched predicted_if_true / predicted_if_false; status only
    flips to CONFIRMED/REJECTED once verification.py's independent-evidence
    gate is satisfied (Principle 6: an observation isn't automatically a
    conclusion), not just from confidence crossing a number.
"""
from __future__ import annotations

from . import verification
from .models import (
    Action,
    Hypothesis,
    HypothesisStatus,
    Observation,
    WorkingState,
)
from .similarity import SimilarityScorer

CONFIRM_THRESHOLD = 0.9
REJECT_THRESHOLD = 0.1

# Phase 12: how similar two differently-worded proposals need to be before
# treating them as the SAME underlying claim rather than two competing
# ones. Deliberately higher than Phase 11's Principle-retrieval bar
# (which only needs "relevant enough to mention as context") — merging
# hypotheses is a stronger claim than surfacing a prior, so it needs more
# confidence that they're really the same thing before evidence gets
# pooled between them.
MERGE_SIMILARITY_THRESHOLD = 0.75


def update_confidence(confidence: float, matched: bool) -> float:
    """THE single confidence-update rule for the whole Brain. Previously
    duplicated: hypotheses.py had this inline, and information_gain.py's
    `_simulated_uncertainty` re-implemented the identical arithmetic in a
    second place to simulate both branches of a hypothesis test before
    the real observation happens. Flagged as a drift risk since Phase 3;
    extracted here so there is exactly one place this math can be wrong
    or changed. Both the REAL update (hypotheses.update_from_observation)
    and the SIMULATED one (information_gain.expected_information_gain)
    now call this function."""
    if matched:
        return min(0.95, confidence + (1 - confidence) * 0.5)
    return max(0.05, confidence * 0.5)


class HypothesisEngine:
    def integrate_new(
        self,
        state: WorkingState,
        proposed: list[dict],
        scorer: SimilarityScorer | None = None,
    ) -> None:
        """`scorer` (Phase 12): None (the default) is the EXACT original
        Phase 2 behavior — only literal exact-text duplicates get folded
        together, everything else becomes its own Hypothesis. This is why
        the existing test suite needed zero changes to keep passing.

        Passing a SimilarityScorer (similarity.py) additionally catches
        the case exact-text matching can't: a proposal that's really
        restating an existing ACTIVE hypothesis in different words gets
        merged into it (see _find_semantic_duplicate) instead of becoming
        a rival hypothesis that silently splits future evidence between
        two objects representing the same real-world claim — which would
        otherwise make BOTH slower to reach CONFIRM_THRESHOLD than either
        deserves. Costs one scorer call per (new proposal, active
        hypothesis) pair when supplied, same cost trade-off as every
        other scorer= call site (priors.py, consolidation.py)."""
        existing_statements = {h.statement.strip().lower() for h in state.hypotheses}
        for p in proposed:
            statement = str(p.get("statement", "")).strip()
            if not statement or statement.lower() in existing_statements:
                continue
            try:
                confidence = float(p.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = min(0.95, max(0.05, confidence))  # never let LLM open at absolute certainty

            if scorer is not None:
                duplicate = self._find_semantic_duplicate(statement, state, scorer)
                if duplicate is not None:
                    self._merge_into(duplicate, statement, confidence, state)
                    existing_statements.add(statement.lower())
                    continue

            state.hypotheses.append(
                Hypothesis(
                    statement=statement,
                    confidence=confidence,
                    created_step=state.step,
                    confidence_history=[confidence],  # opening prior is the first point
                )
            )
            existing_statements.add(statement.lower())

    @staticmethod
    def _find_semantic_duplicate(
        statement: str, state: WorkingState, scorer: SimilarityScorer
    ) -> Hypothesis | None:
        """Only ACTIVE hypotheses are eligible merge targets - deliberately
        excludes CONFIRMED/REJECTED ones. Folding a new proposal into an
        already-terminal hypothesis would let a fresh, untested claim
        silently reopen or reinforce a settled verdict through the back
        door, which is exactly the confirmation-bias pattern
        challenger.py's first check already exists to catch on the
        DECISION side - this is the same principle applied on the belief-
        formation side instead."""
        best: Hypothesis | None = None
        best_score = 0.0
        for h in state.hypotheses:
            if h.status != HypothesisStatus.ACTIVE:
                continue
            score = scorer.score(statement, h.statement)
            if score >= MERGE_SIMILARITY_THRESHOLD and score > best_score:
                best, best_score = h, score
        return best

    @staticmethod
    def _merge_into(
        hyp: Hypothesis, new_statement: str, new_confidence: float, state: WorkingState
    ) -> None:
        """Records the merge for inspectability (Principle 13 - same
        pattern as state.principle_seeds), and only lets the incoming
        proposal move the target's confidence while that target is STILL
        just an untested opening prior (confidence_history length 1,
        meaning no real observation has touched it yet). Once real
        evidence has moved a hypothesis even once, that evidence-tested
        confidence is authoritative and a later duplicate proposal -
        which is just another guess, not new evidence - must not dilute
        it. This is the same "evidence beats a fresh guess" principle
        Phase 9's Principle-feedback design already relies on."""
        state.hypothesis_merges[new_statement] = hyp.id
        if len(hyp.confidence_history) == 1:
            hyp.confidence = round((hyp.confidence + new_confidence) / 2, 3)
            hyp.confidence_history.append(hyp.confidence)

    def update_from_observation(
        self,
        state: WorkingState,
        action: Action,
        observation: Observation,
        contradiction_scorer=None,
    ) -> None:
        if not action.tests_hypothesis:
            return  # exploratory action, not tied to any specific belief
        hyp = next((h for h in state.hypotheses if h.id == action.tests_hypothesis), None)
        if hyp is None or hyp.status != HypothesisStatus.ACTIVE:
            return

        matched = self._outcome_matches_true_branch(action, observation)

        if matched:
            hyp.confidence = update_confidence(hyp.confidence, matched=True)
            hyp.supporting_evidence.append(observation.action_id)
        else:
            hyp.confidence = update_confidence(hyp.confidence, matched=False)
            hyp.contradicting_evidence.append(observation.action_id)

        hyp.updated_step = state.step
        hyp.confidence_history.append(hyp.confidence)

        # Status only flips once the Verification Engine agrees — a single
        # observation crossing the confidence threshold is treated as
        # "plausible", not "confirmed".
        if verification.can_confirm(hyp, CONFIRM_THRESHOLD):
            conflict = None
            if contradiction_scorer is not None:
                conflict = self._find_confirmed_contradiction(hyp, state, contradiction_scorer)
            if conflict is None:
                hyp.status = HypothesisStatus.CONFIRMED
            else:
                # Evidence alone supports confirming this hypothesis, but
                # Brain cannot accept two mutually-exclusive CONFIRMED
                # beliefs at once (Phase 13) — leave it ACTIVE and record
                # the standoff (surfaced in the planning prompt, and
                # escalates the strong model via meta.py) rather than
                # silently letting an inconsistent belief state stand.
                state.contradictions[hyp.id] = conflict.id
        elif verification.can_reject(hyp, REJECT_THRESHOLD):
            hyp.status = HypothesisStatus.REJECTED

        self._clear_stale_contradictions(state)

    @staticmethod
    def _find_confirmed_contradiction(hyp: Hypothesis, state: WorkingState, scorer):
        """Only checked against hypotheses ALREADY CONFIRMED — deliberately
        narrow scope, see contradiction.py's module docstring for why
        ACTIVE-vs-ACTIVE tension is a different (and out of scope) kind of
        signal."""
        for other in state.hypotheses:
            if other.id == hyp.id or other.status != HypothesisStatus.CONFIRMED:
                continue
            if scorer.contradicts(hyp.statement, other.statement):
                return other
        return None

    @staticmethod
    def _clear_stale_contradictions(state: WorkingState) -> None:
        """A recorded standoff is only meaningful while the blocking
        hypothesis is still CONFIRMED - if it later gets superseded (not
        currently possible for a terminal status to change, but kept
        defensive rather than assuming that invariant holds forever) the
        entry would otherwise linger and misleadingly show up in the
        planning prompt."""
        by_id = {h.id: h for h in state.hypotheses}
        stale = [
            hyp_id
            for hyp_id, blocker_id in state.contradictions.items()
            if by_id.get(blocker_id) is None or by_id[blocker_id].status != HypothesisStatus.CONFIRMED
        ]
        for hyp_id in stale:
            del state.contradictions[hyp_id]

    @staticmethod
    def _outcome_matches_true_branch(action: Action, observation: Observation) -> bool:
        """Prefer the explicit counterfactual (predicted_if_true vs
        predicted_if_false) when both are given — that's a real
        distinguishing test. Fall back to the old single-prediction
        heuristic for actions that only set predicted_outcome."""
        actual = str(observation.result).strip().lower()

        if action.predicted_if_true or action.predicted_if_false:
            true_hit = (
                action.predicted_if_true.strip().lower() in actual
                if action.predicted_if_true
                else False
            )
            false_hit = (
                action.predicted_if_false.strip().lower() in actual
                if action.predicted_if_false
                else False
            )
            if true_hit and not false_hit:
                return True
            if false_hit and not true_hit:
                return False
            return bool(observation.success)  # ambiguous - weak fallback signal

        predicted = action.predicted_outcome.strip().lower()
        if not predicted:
            return bool(observation.success)
        return predicted in actual or actual in predicted
