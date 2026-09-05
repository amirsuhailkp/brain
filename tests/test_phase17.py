"""
Phase 17 tests.

17a: word-boundary matching in _outcome_matches_true_branch
17b: meaningful lessons for incremental confidence updates
17c: planning prompt includes settled hypotheses and earlier observations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.extraction import _lesson
from brain.hypotheses import HypothesisEngine
from brain.models import (
    Action,
    ActionStatus,
    Goal as GoalModel,
    Hypothesis,
    HypothesisStatus as HS,
    Observation,
    WorkingState,
)
from brain.interfaces import LLMInterface, ProjectAdapter
from brain.planning import PlanningEngine


# ------------------------------------------------------------------ 17a: word-boundary matching


def _action(predicted_if_true: str = "", predicted_if_false: str = "") -> Action:
    return Action(
        kind="probe",
        params={},
        predicted_if_true=predicted_if_true,
        predicted_if_false=predicted_if_false,
    )


def _obs(result: str, success: bool = True) -> Observation:
    return Observation(action_id="a1", result=result, success=success)


def _matches(action: Action, obs: Observation) -> bool:
    return HypothesisEngine._outcome_matches_true_branch(action, obs)


# --- cases that SHOULD match true branch ---

def test_exact_word_matches_true():
    assert _matches(_action("high", "low"), _obs("value is high")) is True


def test_exact_word_low_matches_true():
    assert _matches(_action("low", "high"), _obs("result is low")) is True


def test_multiword_prediction_still_matches():
    assert _matches(_action("between 10 and 50", "outside range"), _obs("reading between 10 and 50 was recorded")) is True


# --- cases that MUST NOT produce false positives ---

def test_short_token_high_does_not_match_highway():
    """'high' predicted_if_true must NOT spuriously match an observation
    where 'high' only appears as a prefix of 'highway'. When neither
    branch matches, it falls back to observation.success — the correct
    documented fallback for an ambiguous result."""
    # 'highway' doesn't contain 'high' as an isolated word, so true_hit=False.
    # 'low' doesn't appear at all, so false_hit=False.
    # Ambiguous -> falls back to success. The key property is that the
    # regex does NOT find 'high' inside 'highway'; the fallback is correct.
    a = _action("high", "low")
    obs_success = _obs("took the highway exit", success=True)
    obs_failure = _obs("took the highway exit", success=False)
    # With ambiguous result, the ONLY signal is observation.success:
    assert _matches(a, obs_success) is True   # ambiguous but succeeded -> True
    assert _matches(a, obs_failure) is False  # ambiguous but failed -> False


def test_short_token_low_does_not_match_below():
    """'below' contains 'low' but not as an isolated word."""
    a = _action("low", "high")
    obs_success = _obs("temperature is below average", success=True)
    obs_failure = _obs("temperature is below average", success=False)
    assert _matches(a, obs_success) is True   # ambiguous fallback
    assert _matches(a, obs_failure) is False


def test_short_token_yes_does_not_match_yesterday():
    """'yesterday' contains 'yes' but not as an isolated word."""
    a = _action("yes", "no")
    obs_success = _obs("yesterday the probe ran", success=True)
    obs_failure = _obs("yesterday the probe ran", success=False)
    assert _matches(a, obs_success) is True   # ambiguous fallback
    assert _matches(a, obs_failure) is False


def test_short_token_no_does_not_match_node():
    """'node' contains 'no' but not as an isolated word."""
    a = _action("no", "yes")
    obs_success = _obs("connected to node 42", success=True)
    obs_failure = _obs("connected to node 42", success=False)
    assert _matches(a, obs_success) is True   # ambiguous fallback
    assert _matches(a, obs_failure) is False


def test_isolated_high_does_match():
    """When 'high' appears as a genuinely isolated word it SHOULD match."""
    assert _matches(_action("high", "low"), _obs("value is high")) is True
    assert _matches(_action("high", "low"), _obs("the reading was high today")) is True


def test_isolated_no_does_match():
    """When 'no' appears isolated it SHOULD match (e.g. 'no errors found')."""
    assert _matches(_action("no", "yes"), _obs("no errors found", success=True)) is True


# --- false-branch matching ---

def test_false_branch_word_boundary():
    assert _matches(_action("confirmed", "denied"), _obs("the request was denied outright")) is False


def test_ambiguous_falls_back_to_success():
    """When both branches match (or neither), fall back to observation.success."""
    a = _action("match", "match")  # identical: ambiguous
    assert _matches(a, _obs("found a match", success=True)) is True
    assert _matches(a, _obs("found a match", success=False)) is False


def test_no_counterfactual_uses_predicted_outcome():
    a = Action(kind="probe", params={}, predicted_outcome="success")
    assert _matches(a, _obs("overall success on this step")) is True


def test_empty_prediction_falls_back_to_success():
    a = Action(kind="probe", params={})
    assert _matches(a, _obs("anything", success=True)) is True
    assert _matches(a, _obs("anything", success=False)) is False


# ------------------------------------------------------------------ 17b: meaningful lessons


def test_lesson_incremental_confidence_increase():
    hyp_before = Hypothesis(statement="target is X", confidence=0.5, confidence_history=[0.5])
    hyp_after = Hypothesis(statement="target is X", confidence=0.75, confidence_history=[0.5, 0.75])
    action = Action(kind="probe", params={})
    obs = Observation(action_id="a1", result="matched", success=True)
    lesson = _lesson(action, obs, hyp_before, hyp_after)
    assert "Routine" not in lesson
    assert "incremental" in lesson
    assert "increased" in lesson
    assert "target is X" in lesson


def test_lesson_incremental_confidence_decrease():
    hyp_before = Hypothesis(statement="target is X", confidence=0.6, confidence_history=[0.6])
    hyp_after = Hypothesis(statement="target is X", confidence=0.3, confidence_history=[0.6, 0.3])
    action = Action(kind="probe", params={})
    obs = Observation(action_id="a1", result="no match", success=True)
    lesson = _lesson(action, obs, hyp_before, hyp_after)
    assert "Routine" not in lesson
    assert "decreased" in lesson


def test_lesson_action_failure_unchanged():
    obs = Observation(action_id="a1", result="error", success=False)
    action = Action(kind="scan", params={"x": 1})
    lesson = _lesson(action, obs, None, None)
    assert "failed" in lesson
    assert "scan" in lesson


def test_lesson_hypothesis_resolved():
    hyp_before = Hypothesis(statement="claim", confidence=0.85, confidence_history=[0.5, 0.85])
    hyp_before.status = HS.ACTIVE
    hyp_after = Hypothesis(statement="claim", confidence=0.95, confidence_history=[0.5, 0.85, 0.95])
    hyp_after.status = HS.CONFIRMED
    action = Action(kind="probe", params={})
    obs = Observation(action_id="a1", result="confirmed", success=True)
    lesson = _lesson(action, obs, hyp_before, hyp_after)
    assert "resolved" in lesson or "confirmed" in lesson


def test_lesson_no_hypothesis_exploratory():
    action = Action(kind="scan", params={})
    obs = Observation(action_id="a1", result="found 3 items", success=True)
    lesson = _lesson(action, obs, None, None)
    assert "Routine" not in lesson
    assert "scan" in lesson


# ------------------------------------------------------------------ 17c: planning context window


class _CaptureLLM(LLMInterface):
    """Records the last prompt text so tests can inspect it."""
    last_prompt: str = ""

    def propose(self, prompt: str, schema_hint: str) -> dict:
        _CaptureLLM.last_prompt = prompt
        return {"hypotheses": [], "candidate_actions": []}


class _MinimalProject(ProjectAdapter):
    @property
    def project_id(self) -> str:
        return "test"

    def perceive(self) -> dict:
        return {}

    def is_goal_met(self, world_model) -> bool:
        return False

    def available_actions(self, world_model) -> list[str]:
        return ["probe"]

    def validate(self, action) -> bool:
        return True

    def execute(self, action) -> "Observation":
        return Observation(action_id=action.id, result="ok", success=True)

    def project_memory_context(self, goal_description: str) -> list[str]:
        return []


def _make_state_with_n_obs(n: int) -> WorkingState:
    state = WorkingState(goal=GoalModel(description="test goal"))
    state.step = n
    for i in range(n):
        state.observations.append(Observation(action_id=f"a{i}", result=f"obs{i}", success=True))
    return state


def test_planning_prompt_includes_earlier_obs_summary_when_more_than_3():
    state = _make_state_with_n_obs(6)  # 6 observations: 3 recent + 3 earlier
    engine = PlanningEngine(_CaptureLLM())
    engine.reason_and_plan(state, _MinimalProject(), ["probe"])
    prompt = _CaptureLLM.last_prompt
    assert "EARLIER OBSERVATIONS" in prompt
    assert "RECENT OBSERVATIONS" in prompt


def test_planning_prompt_no_earlier_section_when_3_or_fewer():
    state = _make_state_with_n_obs(3)
    engine = PlanningEngine(_CaptureLLM())
    engine.reason_and_plan(state, _MinimalProject(), ["probe"])
    assert "EARLIER OBSERVATIONS" not in _CaptureLLM.last_prompt


def test_planning_prompt_shows_settled_hypotheses():
    state = _make_state_with_n_obs(2)
    confirmed = Hypothesis(statement="target is 42", confidence=0.95, confidence_history=[0.5, 0.95])
    confirmed.status = HS.CONFIRMED
    confirmed.supporting_evidence = ["a0", "a1"]
    state.hypotheses.append(confirmed)

    engine = PlanningEngine(_CaptureLLM())
    engine.reason_and_plan(state, _MinimalProject(), ["probe"])
    prompt = _CaptureLLM.last_prompt
    assert "SETTLED HYPOTHESES" in prompt
    assert "target is 42" in prompt
    assert "confirmed" in prompt


def test_planning_prompt_no_settled_section_when_all_active():
    state = _make_state_with_n_obs(2)
    state.hypotheses.append(Hypothesis(statement="active claim", confidence=0.6, confidence_history=[0.6]))
    engine = PlanningEngine(_CaptureLLM())
    engine.reason_and_plan(state, _MinimalProject(), ["probe"])
    assert "SETTLED HYPOTHESES" not in _CaptureLLM.last_prompt


def test_planning_prompt_shows_both_supported_and_contradicting_counts():
    state = _make_state_with_n_obs(2)
    rejected = Hypothesis(statement="wrong hypothesis", confidence=0.08, confidence_history=[0.5, 0.08])
    rejected.status = HS.REJECTED
    rejected.contradicting_evidence = ["a0", "a1"]
    state.hypotheses.append(rejected)
    engine = PlanningEngine(_CaptureLLM())
    engine.reason_and_plan(state, _MinimalProject(), ["probe"])
    prompt = _CaptureLLM.last_prompt
    assert "wrong hypothesis" in prompt
    assert "rejected" in prompt
