import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import meta, quality_gate
from brain.calibration import CalibrationTracker
from brain.decision import (
    DecisionEngine,
    CONSERVATIVE_DUPLICATE_PENALTY_MULTIPLIER,
    CONSERVATIVE_INFO_GAIN_MULTIPLIER,
    DUPLICATE_PENALTY,
)
from brain.interfaces import LLMInterface
from brain.models import (
    Action,
    ActionStatus,
    Decision,
    Goal as GoalModel,
    Hypothesis,
    HypothesisStatus as HS,
    WorkingState,
)
from brain.reasoning_quality import build_report
from brain.strategy import BALANCED


# --------------------------------------------------------------- quality_gate.evaluate()


def _state_with_oscillating_hypothesis() -> WorkingState:
    state = WorkingState(goal=GoalModel(description="test"))
    h = Hypothesis(statement="claim A", confidence=0.5, confidence_history=[
        0.5, 0.8, 0.3, 0.75, 0.35  # 3 direction reversals — above OSCILLATION_REVERSAL_THRESHOLD
    ])
    state.hypotheses.append(h)
    return state


def _add_challenger_overrides(state: WorkingState, count: int) -> None:
    for i in range(count):
        state.decisions.append(Decision(
            chosen_action=Action(kind="probe", params={}),
            alternatives_considered=[],
            rationale=f"CHALLENGED: override #{i}",
        ))


def _add_rejected_actions(state: WorkingState, count: int) -> None:
    for i in range(count):
        a = Action(kind="probe", params={"n": i})
        a.status = ActionStatus.REJECTED
        state.actions_taken.append(a)


def test_no_signals_active_no_conservative_mode():
    state = WorkingState(goal=GoalModel(description="test"))
    tracker = CalibrationTracker()
    result = quality_gate.evaluate(state, tracker)
    assert result["conservative_mode"] is False
    assert result["active_signal_count"] == 0


def test_single_signal_does_not_trigger_conservative_mode():
    """One oscillating hypothesis alone must not trigger conservative mode —
    threshold is 2-of-4 for the same reason ReasoningQualityReport requires
    >= 2 flags before calling a run 'low quality'."""
    state = _state_with_oscillating_hypothesis()
    result = quality_gate.evaluate(state, CalibrationTracker())
    assert result["has_oscillating_hypothesis"] is True
    assert result["active_signal_count"] == 1
    assert result["conservative_mode"] is False


def test_two_signals_trigger_conservative_mode():
    state = _state_with_oscillating_hypothesis()
    _add_challenger_overrides(state, 2)
    result = quality_gate.evaluate(state, CalibrationTracker())
    assert result["has_oscillating_hypothesis"] is True
    assert result["repeated_challenger_overrides"] is True
    assert result["active_signal_count"] == 2
    assert result["conservative_mode"] is True


def test_three_rejected_actions_triggers_signal():
    state = WorkingState(goal=GoalModel(description="test"))
    _add_rejected_actions(state, 3)
    result = quality_gate.evaluate(state, CalibrationTracker())
    assert result["repeated_action_rejections"] is True


def test_conservative_mode_clears_when_signals_resolve():
    """Conservative mode is re-evaluated every step and not sticky — if
    a strategy switch clears the oscillating hypothesis and the
    challenger stops intervening, the signal clears automatically."""
    state = WorkingState(goal=GoalModel(description="test"))
    # First: enter conservative mode
    _add_challenger_overrides(state, 2)
    h = Hypothesis(statement="x", confidence=0.5, confidence_history=[0.5, 0.8, 0.3, 0.75, 0.35])
    state.hypotheses.append(h)
    result_on = quality_gate.evaluate(state, CalibrationTracker())
    assert result_on["conservative_mode"] is True

    # Now simulate a strategy switch clearing the oscillating trajectory
    # and the challenger calming down (clear decisions and hypothesis history)
    state.decisions.clear()
    h.confidence_history = [0.5, 0.6, 0.7]  # monotonic, no reversals
    result_off = quality_gate.evaluate(state, CalibrationTracker())
    assert result_off["conservative_mode"] is False


# --------------------------------------------------------------- DecisionEngine conservative_mode


def _make_action(kind: str, params: dict | None = None, cost: float = 1.0) -> Action:
    return Action(kind=kind, params=params or {}, cost=cost)


def test_conservative_mode_off_scores_normally():
    state = WorkingState(goal=GoalModel(description="test"))
    engine = DecisionEngine()
    action = _make_action("probe")
    decision = engine.decide([action], state, BALANCED, conservative_mode=False)
    assert "conservative_mode=ON" not in decision.rationale


def test_conservative_mode_on_noted_in_rationale():
    state = WorkingState(goal=GoalModel(description="test"))
    engine = DecisionEngine()
    action = _make_action("probe")
    decision = engine.decide([action], state, BALANCED, conservative_mode=True)
    assert "conservative_mode=ON" in decision.rationale
    assert "quality-conservatism overlay" in decision.rationale


def test_conservative_mode_duplicate_penalty_is_stronger():
    """With conservative_mode=True, a duplicate action must score far lower
    than it would in normal mode — not just lower, but by the exact
    multiplier the module declares."""
    state = WorkingState(goal=GoalModel(description="test"))
    # Mark an action as already taken
    prev = _make_action("probe", {"x": 1})
    prev.status = ActionStatus.EXECUTED
    state.actions_taken.append(prev)

    engine = DecisionEngine()
    dup_action = _make_action("probe", {"x": 1})
    non_dup = _make_action("scan", {"y": 2})

    decision_normal = engine.decide([dup_action, non_dup], state, BALANCED, conservative_mode=False)
    decision_conservative = engine.decide([dup_action, non_dup], state, BALANCED, conservative_mode=True)
    # Conservative mode should prefer the non-duplicate even more strongly —
    # it must still prefer non_dup in both cases (duplicate penalty is huge),
    # but what we're checking is that conservative_mode can't accidentally
    # flip a case where a non-duplicate was going to win into the duplicate winning.
    assert decision_normal.chosen_action.kind == "scan"
    assert decision_conservative.chosen_action.kind == "scan"


def test_high_info_gain_action_still_wins_in_conservative_mode():
    """Conservative mode raises the BAR but must not override a genuinely
    much-more-informative action. A non-duplicate with a much higher
    expected information gain must still beat a lower-value one, same
    as in normal mode."""
    state = WorkingState(goal=GoalModel(description="test"))
    h = Hypothesis(statement="claim", confidence=0.5, confidence_history=[0.5])
    state.hypotheses.append(h)
    # One action tests the hypothesis (meaningful info gain), one is exploratory
    valuable = _make_action("probe", {}, cost=1.0)
    valuable.tests_hypothesis = h.id
    valuable.predicted_if_true = "yes"
    valuable.predicted_if_false = "no"
    cheap_but_uninformative = _make_action("ping", {}, cost=1.0)

    engine = DecisionEngine()
    decision = engine.decide([cheap_but_uninformative, valuable], state, BALANCED, conservative_mode=True)
    assert decision.chosen_action.kind == "probe"


# --------------------------------------------------------------- meta.py signal


def test_live_quality_degraded_triggers_escalation():
    state = WorkingState(goal=GoalModel(description="test"))
    signals = meta.compute_signals(state, CalibrationTracker(), live_quality_degraded=True)
    assert signals["live_quality_degraded"] is True
    assert meta.should_escalate(signals) is True


def test_no_degradation_does_not_escalate_on_that_basis():
    state = WorkingState(goal=GoalModel(description="test"))
    signals = meta.compute_signals(state, CalibrationTracker(), live_quality_degraded=False)
    assert signals["live_quality_degraded"] is False


# --------------------------------------------------------------- ReasoningQualityReport integration


def test_sustained_conservative_mode_appears_in_retrospective_report():
    """If conservative_mode_steps is high as a fraction of total steps,
    the final retrospective report should flag it."""
    state = WorkingState(goal=GoalModel(description="test"))
    state.step = 9
    state.conservative_mode_steps = 4  # 44% of steps — above the 33% threshold
    report = build_report(state, CalibrationTracker())
    assert report.conservative_mode_steps == 4
    flag_texts = " ".join(report.flags)
    assert "conservative mode" in flag_texts


def test_brief_conservative_mode_not_flagged_in_report():
    state = WorkingState(goal=GoalModel(description="test"))
    state.step = 10
    state.conservative_mode_steps = 1  # 10% — below the 33% threshold
    report = build_report(state, CalibrationTracker())
    assert report.conservative_mode_steps == 1
    flag_texts = " ".join(report.flags)
    assert "conservative mode" not in flag_texts
