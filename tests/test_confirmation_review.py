import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import meta
from brain.calibration import CalibrationTracker
from brain.confirmation_review import ConfirmationReviewer, LLMConfirmationReviewer, needs_review, THIN_EVIDENCE_MAX
from brain.contradiction import LLMContradictionScorer
from brain.hypotheses import HypothesisEngine, CONFIRM_THRESHOLD
from brain.interfaces import LLMInterface
from brain.models import Action, Goal as GoalModel, Hypothesis, HypothesisStatus as HS, Observation, WorkingState


# --------------------------------------------------------------- LLMConfirmationReviewer


class FixedReviewLLM(LLMInterface):
    def propose(self, prompt, schema_hint):
        if "target is exactly 42" in prompt:
            return {"approved": False, "reason": "only one distinguishing test has run"}
        return {"approved": True, "reason": "consistent, well-tested evidence"}


class MalformedReviewLLM(LLMInterface):
    def propose(self, prompt, schema_hint):
        raise RuntimeError("model unavailable")


def test_llm_confirmation_reviewer_holds_the_flagged_hypothesis():
    reviewer = LLMConfirmationReviewer(FixedReviewLLM())
    approved, reason = reviewer.review("target is exactly 42", ["a1", "a2"])
    assert approved is False
    assert "distinguishing" in reason


def test_llm_confirmation_reviewer_approves_unrelated_claim():
    reviewer = LLMConfirmationReviewer(FixedReviewLLM())
    approved, reason = reviewer.review("target is < 30", ["a1", "a2"])
    assert approved is True


def test_llm_confirmation_reviewer_degrades_safely_on_failure():
    """Fail OPEN: a broken reviewer must never permanently strand an
    otherwise-earned confirmation."""
    reviewer = LLMConfirmationReviewer(MalformedReviewLLM())
    approved, reason = reviewer.review("anything", ["a1"])
    assert approved is True
    assert reason == ""


def test_empty_statement_never_blocked():
    reviewer = LLMConfirmationReviewer(FixedReviewLLM())
    approved, _ = reviewer.review("", ["a1"])
    assert approved is True


def test_needs_review_thin_evidence_gate():
    thin = Hypothesis(statement="x", confidence=0.9, confidence_history=[0.9])
    thin.supporting_evidence = ["a1", "a2"][:THIN_EVIDENCE_MAX]
    assert needs_review(thin) is True

    thick = Hypothesis(statement="x", confidence=0.9, confidence_history=[0.9])
    thick.supporting_evidence = [f"a{i}" for i in range(THIN_EVIDENCE_MAX + 5)]
    assert needs_review(thick) is False


# --------------------------------------------------------------- HypothesisEngine wiring


def _make_action(hyp_id: str) -> Action:
    return Action(
        kind="probe",
        params={},
        tests_hypothesis=hyp_id,
        predicted_if_true="match",
        predicted_if_false="nomatch",
    )


def test_thin_confirmation_is_held_not_confirmed():
    state = WorkingState(goal=GoalModel(description="test"))
    candidate = Hypothesis(statement="target is exactly 42", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    reviewer = LLMConfirmationReviewer(FixedReviewLLM())

    # Exactly enough matching observations to first clear can_confirm() -
    # MIN_CONFIRMATIONS(2) pieces of evidence - which is also exactly
    # THIN_EVIDENCE_MAX, so this is precisely the moment review should
    # fire, before more (repeated) evidence would make it no longer thin.
    for _ in range(2):
        engine.update_from_observation(state, action, obs, confirmation_reviewer=reviewer)

    assert candidate.status == HS.ACTIVE  # held, not confirmed
    assert candidate.confidence >= CONFIRM_THRESHOLD  # evidence really did clear the bar
    assert candidate.id in state.confirmation_holds
    assert "distinguishing" in state.confirmation_holds[candidate.id]


def test_confirmation_proceeds_normally_without_a_reviewer():
    """Backward compatibility: omitting confirmation_reviewer must behave
    exactly like every pre-Phase-15 call site."""
    state = WorkingState(goal=GoalModel(description="test"))
    candidate = Hypothesis(statement="target is < 30", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    for _ in range(6):
        engine.update_from_observation(state, action, obs)  # no reviewer at all

    assert candidate.status == HS.CONFIRMED
    assert state.confirmation_holds == {}


def test_reviewer_approves_when_evidence_genuinely_supports_it():
    state = WorkingState(goal=GoalModel(description="test"))
    candidate = Hypothesis(statement="target is < 30", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    reviewer = LLMConfirmationReviewer(FixedReviewLLM())
    for _ in range(6):
        engine.update_from_observation(state, action, obs, confirmation_reviewer=reviewer)

    assert candidate.status == HS.CONFIRMED
    assert candidate.id not in state.confirmation_holds


def test_held_hypothesis_confirms_once_evidence_is_no_longer_thin():
    """A hold is a 'not yet', never a permanent veto - once enough
    additional evidence accumulates that it's no longer 'thin' by the
    same bar meta.py uses, review is skipped and confirmation proceeds
    on the deterministic gate alone, exactly like any well-evidenced
    hypothesis."""
    state = WorkingState(goal=GoalModel(description="test"))
    candidate = Hypothesis(statement="target is exactly 42", confidence=0.6, confidence_history=[0.6])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    reviewer = LLMConfirmationReviewer(FixedReviewLLM())

    # Drive it past THIN_EVIDENCE_MAX pieces of supporting evidence before
    # it ever gets a chance to cross CONFIRM_THRESHOLD, by starting low.
    for _ in range(THIN_EVIDENCE_MAX + 3):
        engine.update_from_observation(state, action, obs, confirmation_reviewer=reviewer)

    assert candidate.status == HS.CONFIRMED
    assert candidate.id not in state.confirmation_holds


def test_held_confirmation_skips_the_contradiction_check_that_step():
    """A hold means nothing new is being asserted as CONFIRMED this step,
    so the contradiction gate (Phase 13) has nothing to check against yet
    - it must not fire spuriously on a hypothesis that never actually
    reached CONFIRMED."""
    state = WorkingState(goal=GoalModel(description="test"))
    already_confirmed = Hypothesis(statement="unrelated fact", confidence=0.95, confidence_history=[0.5, 0.95])
    already_confirmed.status = HS.CONFIRMED
    state.hypotheses.append(already_confirmed)

    candidate = Hypothesis(statement="target is exactly 42", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)

    class AlwaysContradicts(LLMInterface):
        def propose(self, prompt, schema_hint):
            return {"contradicts": True}

    reviewer = LLMConfirmationReviewer(FixedReviewLLM())
    contradiction_scorer = LLMContradictionScorer(AlwaysContradicts())

    for _ in range(2):  # exactly enough to first clear can_confirm(), same as above
        engine.update_from_observation(
            state, action, obs,
            contradiction_scorer=contradiction_scorer,
            confirmation_reviewer=reviewer,
        )

    assert candidate.status == HS.ACTIVE
    assert candidate.id in state.confirmation_holds
    assert candidate.id not in state.contradictions  # held, never reached the contradiction gate


# --------------------------------------------------------------- meta.py escalation signal


def test_unresolved_confirmation_hold_triggers_escalation():
    state = WorkingState(goal=GoalModel(description="test"))
    state.confirmation_holds["hyp-a"] = "only thin evidence so far"
    signals = meta.compute_signals(state, CalibrationTracker())
    assert signals["unresolved_confirmation_hold"] is True
    assert meta.should_escalate(signals) is True


def test_no_hold_does_not_escalate_on_that_basis():
    state = WorkingState(goal=GoalModel(description="test"))
    signals = meta.compute_signals(state, CalibrationTracker())
    assert signals["unresolved_confirmation_hold"] is False
