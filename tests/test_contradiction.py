import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import Brain, Goal, JsonlMemory, HypothesisStatus, meta
from brain.calibration import CalibrationTracker
from brain.contradiction import ContradictionScorer, LLMContradictionScorer
from brain.hypotheses import HypothesisEngine, CONFIRM_THRESHOLD
from brain.interfaces import LLMInterface, ProjectAdapter
from brain.models import Action, ActionStatus, Goal as GoalModel, Hypothesis, HypothesisStatus as HS, Observation, WorkingState

import tempfile


# --------------------------------------------------------------- LLMContradictionScorer


class FixedContradictionLLM(LLMInterface):
    def propose(self, prompt, schema_hint):
        if "target is >= 50" in prompt and "target is < 30" in prompt:
            return {"contradicts": True}
        return {"contradicts": False}


class MalformedContradictionLLM(LLMInterface):
    def propose(self, prompt, schema_hint):
        raise RuntimeError("model unavailable")


def test_llm_contradiction_scorer_detects_the_flagged_pair():
    scorer = LLMContradictionScorer(FixedContradictionLLM())
    assert scorer.contradicts("target is >= 50", "target is < 30") is True


def test_llm_contradiction_scorer_false_for_unrelated_pair():
    scorer = LLMContradictionScorer(FixedContradictionLLM())
    assert scorer.contradicts("target is >= 50", "the sky is blue") is False


def test_llm_contradiction_scorer_degrades_safely_on_failure():
    scorer = LLMContradictionScorer(MalformedContradictionLLM())
    assert scorer.contradicts("a", "b") is False  # never raises


def test_empty_strings_never_contradict():
    scorer = LLMContradictionScorer(FixedContradictionLLM())
    assert scorer.contradicts("", "something") is False


# --------------------------------------------------------------- HypothesisEngine wiring


def _confirmed_hypothesis(statement: str) -> Hypothesis:
    h = Hypothesis(statement=statement, confidence=0.95)
    h.status = HS.CONFIRMED
    h.confidence_history = [0.5, 0.95]
    return h


def _make_action(hyp_id: str) -> Action:
    return Action(
        kind="probe",
        params={},
        tests_hypothesis=hyp_id,
        predicted_if_true="match",
        predicted_if_false="nomatch",
    )


def test_confirmation_blocked_when_it_contradicts_an_already_confirmed_hypothesis():
    state = WorkingState(goal=GoalModel(description="test"))
    already_confirmed = _confirmed_hypothesis("target is >= 50")
    state.hypotheses.append(already_confirmed)

    candidate = Hypothesis(statement="target is < 30", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)

    scorer = LLMContradictionScorer(FixedContradictionLLM())
    # Drive confidence up past CONFIRM_THRESHOLD via repeated matching observations
    for _ in range(6):
        engine.update_from_observation(state, action, obs, contradiction_scorer=scorer)

    assert candidate.status == HS.ACTIVE  # NOT confirmed, despite evidence supporting it
    assert candidate.confidence >= CONFIRM_THRESHOLD  # evidence really did earn confirmation
    assert state.contradictions[candidate.id] == already_confirmed.id


def test_confirmation_proceeds_normally_without_a_contradiction_scorer():
    """Backward compatibility: omitting contradiction_scorer must behave
    exactly like every pre-Phase-13 call site."""
    state = WorkingState(goal=GoalModel(description="test"))
    candidate = Hypothesis(statement="target is < 30", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    for _ in range(6):
        engine.update_from_observation(state, action, obs)  # no scorer at all

    assert candidate.status == HS.CONFIRMED
    assert state.contradictions == {}


def test_confirmation_proceeds_when_no_contradiction_exists():
    state = WorkingState(goal=GoalModel(description="test"))
    unrelated_confirmed = _confirmed_hypothesis("the sky is blue")
    state.hypotheses.append(unrelated_confirmed)

    candidate = Hypothesis(statement="target is < 30", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    scorer = LLMContradictionScorer(FixedContradictionLLM())
    for _ in range(6):
        engine.update_from_observation(state, action, obs, contradiction_scorer=scorer)

    assert candidate.status == HS.CONFIRMED  # no real contradiction, nothing blocked
    assert state.contradictions == {}


def test_contradiction_never_checked_against_a_merely_active_hypothesis():
    """Scope guard: only ALREADY-CONFIRMED hypotheses can block a
    confirmation. Two ACTIVE hypotheses that would contradict each other
    if both were confirmed must not block one another while both are
    still just active theories."""
    state = WorkingState(goal=GoalModel(description="test"))
    other_active = Hypothesis(statement="target is >= 50", confidence=0.6, confidence_history=[0.6])
    other_active.status = HS.ACTIVE
    state.hypotheses.append(other_active)

    candidate = Hypothesis(statement="target is < 30", confidence=0.85, confidence_history=[0.85])
    state.hypotheses.append(candidate)

    engine = HypothesisEngine()
    action = _make_action(candidate.id)
    obs = Observation(action_id="a1", result="match", success=True)
    scorer = LLMContradictionScorer(FixedContradictionLLM())
    for _ in range(6):
        engine.update_from_observation(state, action, obs, contradiction_scorer=scorer)

    assert candidate.status == HS.CONFIRMED  # other hypothesis is only ACTIVE, doesn't block


# --------------------------------------------------------------- meta.py escalation signal


def test_unresolved_contradiction_triggers_escalation():
    state = WorkingState(goal=GoalModel(description="test"))
    state.contradictions["hyp-a"] = "hyp-b"
    signals = meta.compute_signals(state, CalibrationTracker())
    assert signals["unresolved_contradiction"] is True
    assert meta.should_escalate(signals) is True


def test_no_contradiction_does_not_escalate_on_that_basis():
    state = WorkingState(goal=GoalModel(description="test"))
    signals = meta.compute_signals(state, CalibrationTracker())
    assert signals["unresolved_contradiction"] is False
