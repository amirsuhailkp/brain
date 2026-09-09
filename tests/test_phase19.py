"""
Phase 19 tests.

19a: world-model-aware exploration scoring (information_gain.py)
19b: duplicate-action check in challenger (challenger.py)
19c: minimum-sample guard in calibration.is_overconfident (calibration.py)
19d: require_all_tags mode in memory.query (memory.py)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.calibration import CalibrationTracker
from brain.challenger import review as challenger_review
from brain.decision import DecisionEngine
from brain.information_gain import (
    EXPLORATION_BASE_GAIN,
    EXPLORATION_RELEVANCE_MULTIPLIER,
    expected_information_gain,
    _exploration_gain,
)
from brain.memory import JsonlMemory
from brain.models import (
    Action,
    ActionStatus,
    Decision,
    Experience,
    Goal as GoalModel,
    Hypothesis,
    HypothesisStatus as HS,
    WorkingState,
)
from brain.strategy import BALANCED


# ================================================================== 19a: exploration scoring


def _state_with_active_hyps(*statements: str) -> WorkingState:
    state = WorkingState(goal=GoalModel(description="test"))
    for s in statements:
        state.hypotheses.append(Hypothesis(statement=s, confidence=0.5, confidence_history=[0.5]))
    return state


def _exploratory(params: dict) -> Action:
    return Action(kind="scan", params=params)  # no tests_hypothesis


def test_exploration_base_gain_with_no_hypotheses():
    state = WorkingState(goal=GoalModel(description="test"))
    action = _exploratory({"field": "temperature"})
    gain = expected_information_gain(action, state)
    assert gain == EXPLORATION_BASE_GAIN


def test_exploration_base_gain_with_no_params():
    state = _state_with_active_hyps("temperature is rising")
    action = _exploratory({})
    gain = expected_information_gain(action, state)
    assert gain == EXPLORATION_BASE_GAIN


def test_exploration_relevant_params_score_higher():
    """Params that overlap with active hypothesis vocabulary score higher
    than the flat base gain."""
    state = _state_with_active_hyps("temperature is rising in the sensor")
    relevant = _exploratory({"field": "temperature"})
    irrelevant = _exploratory({"field": "checksum_xyz_b"})
    gain_relevant = expected_information_gain(relevant, state)
    gain_irrelevant = expected_information_gain(irrelevant, state)
    assert gain_relevant > gain_irrelevant
    assert gain_relevant == EXPLORATION_BASE_GAIN * EXPLORATION_RELEVANCE_MULTIPLIER
    assert gain_irrelevant == EXPLORATION_BASE_GAIN


def test_exploration_partial_word_match_counts():
    """Any word overlap triggers the multiplier — single matching word is enough."""
    state = _state_with_active_hyps("pressure valve is stuck closed")
    action = _exploratory({"sensor": "pressure", "unit": "psi"})
    gain = expected_information_gain(action, state)
    assert gain == EXPLORATION_BASE_GAIN * EXPLORATION_RELEVANCE_MULTIPLIER


def test_exploration_short_words_filtered():
    """Words <= 2 chars are filtered out of both sides — common stop words
    like 'is', 'in', 'to' must not produce spurious hits."""
    state = _state_with_active_hyps("is in to")  # all stop words
    action = _exploratory({"field": "is"})
    gain = expected_information_gain(action, state)
    assert gain == EXPLORATION_BASE_GAIN  # no real overlap after filtering


def test_exploration_confirmed_hypotheses_excluded():
    """Only ACTIVE hypotheses contribute vocabulary — a CONFIRMED hypothesis
    isn't an open question any more and shouldn't inflate exploration scores."""
    state = _state_with_active_hyps()
    confirmed = Hypothesis(statement="temperature is high", confidence=0.95, confidence_history=[0.5, 0.95])
    confirmed.status = HS.CONFIRMED
    state.hypotheses.append(confirmed)
    action = _exploratory({"field": "temperature"})
    gain = expected_information_gain(action, state)
    # No ACTIVE hypotheses -> base gain only
    assert gain == EXPLORATION_BASE_GAIN


def test_hypothesis_tied_action_unaffected():
    """Non-exploratory actions (with tests_hypothesis) must be unaffected by
    the Phase 19a change — they go through the existing counterfactual path."""
    state = _state_with_active_hyps("target is 42")
    hyp = state.hypotheses[0]
    action = Action(
        kind="probe", params={"value": "42"},
        predicted_if_true="match", predicted_if_false="no match",
    )
    action.tests_hypothesis = hyp.id
    gain = expected_information_gain(action, state)
    # This goes through the existing P(true)*U(true) + P(false)*U(false) path
    assert gain >= 0.0  # real positive gain (counterfactual is distinguishing)


# ================================================================== 19b: challenger duplicate check


def _executed_action(kind: str, params: dict | None = None) -> Action:
    a = Action(kind=kind, params=params or {})
    a.status = ActionStatus.EXECUTED
    return a


def _candidate(kind: str, params: dict | None = None) -> Action:
    return Action(kind=kind, params=params or {})


def _decision(chosen: Action, alternatives: list[Action] | None = None) -> Decision:
    return Decision(
        chosen_action=chosen,
        alternatives_considered=alternatives or [],
        rationale="initial",
    )


def test_challenger_flags_duplicate_execution():
    state = WorkingState(goal=GoalModel(description="test"))
    state.actions_taken.append(_executed_action("probe", {"x": 1}))

    dup = _candidate("probe", {"x": 1})
    better = _candidate("scan", {"y": 2})
    dec = _decision(dup, alternatives=[better])
    result = challenger_review(dec, state)

    assert "CHALLENGED" in result.rationale
    assert "already been executed" in result.rationale
    # challenger switched to the non-duplicate alternative
    assert result.chosen_action.kind == "scan"


def test_challenger_does_not_flag_different_params():
    """Same action KIND with different params is not a duplicate."""
    state = WorkingState(goal=GoalModel(description="test"))
    state.actions_taken.append(_executed_action("probe", {"x": 1}))

    different_params = _candidate("probe", {"x": 2})
    dec = _decision(different_params)
    result = challenger_review(dec, state)

    assert "already been executed" not in result.rationale
    assert result.chosen_action.kind == "probe"


def test_challenger_does_not_flag_first_execution():
    """First time an action runs is never flagged as a duplicate."""
    state = WorkingState(goal=GoalModel(description="test"))
    action = _candidate("probe", {"x": 1})
    dec = _decision(action)
    result = challenger_review(dec, state)
    assert "already been executed" not in result.rationale


def test_challenger_only_counts_executed_not_rejected():
    """REJECTED actions don't count as executed for duplicate detection."""
    state = WorkingState(goal=GoalModel(description="test"))
    rejected = Action(kind="probe", params={"x": 1})
    rejected.status = ActionStatus.REJECTED
    state.actions_taken.append(rejected)

    action = _candidate("probe", {"x": 1})
    dec = _decision(action)
    result = challenger_review(dec, state)
    assert "already been executed" not in result.rationale


def test_challenger_duplicate_with_no_better_alternative_annotates_not_blocks():
    """When the only candidate IS the duplicate, challenger annotates the
    rationale but doesn't block (same pattern as other checks)."""
    state = WorkingState(goal=GoalModel(description="test"))
    state.actions_taken.append(_executed_action("probe", {"x": 1}))

    dup = _candidate("probe", {"x": 1})
    dec = _decision(dup, alternatives=[])
    result = challenger_review(dec, state)

    assert "CHALLENGED" in result.rationale
    assert result.chosen_action.kind == "probe"  # kept, no better alternative


def test_challenger_still_catches_existing_issues():
    """Phase 19b must not break the two existing challenger checks."""
    state = WorkingState(goal=GoalModel(description="test"))
    confirmed = Hypothesis(statement="claim", confidence=0.95, confidence_history=[0.5, 0.95])
    confirmed.status = HS.CONFIRMED
    state.hypotheses.append(confirmed)

    stale_action = _candidate("probe", {"z": 9})
    stale_action.tests_hypothesis = confirmed.id

    better = _candidate("scan", {"w": 1})
    dec = _decision(stale_action, alternatives=[better])
    result = challenger_review(dec, state)

    assert "CHALLENGED" in result.rationale
    assert "confirmed" in result.rationale


# ================================================================== 19c: calibration bin guard


def test_is_overconfident_false_with_sparse_data():
    """With only 1 record per bin, no bin meets min_bin_samples=3 and
    is_overconfident must return False rather than being dominated by
    a single data point."""
    tracker = CalibrationTracker()
    tracker.record(0.9, False)  # one very overconfident record
    # Original code would have returned True; Phase 19c returns False
    assert tracker.is_overconfident() is False  # sparse data, no qualifying bins


def test_is_overconfident_true_with_enough_data():
    """With enough records per bin, overconfidence is still detected."""
    tracker = CalibrationTracker()
    for _ in range(5):
        tracker.record(0.9, False)  # high confidence, consistently wrong
    assert tracker.is_overconfident() is True


def test_is_overconfident_false_when_well_calibrated():
    tracker = CalibrationTracker()
    for _ in range(4):
        tracker.record(0.8, True)   # high confidence, mostly right
        tracker.record(0.5, True)
        tracker.record(0.5, False)
    assert tracker.is_overconfident() is False


def test_is_overconfident_empty_tracker():
    assert CalibrationTracker().is_overconfident() is False


def test_is_overconfident_min_bin_samples_configurable():
    """The min_bin_samples threshold is a parameter, not a magic constant."""
    tracker = CalibrationTracker()
    for _ in range(2):
        tracker.record(0.9, False)
    # With min_bin_samples=1, the 2-record bin qualifies
    assert tracker.is_overconfident(min_bin_samples=1) is True
    # With min_bin_samples=3 (default), same data doesn't qualify
    assert tracker.is_overconfident(min_bin_samples=3) is False


# ================================================================== 19d: memory tag-AND query


def _write_temp_memory(experiences: list[Experience]) -> JsonlMemory:
    tmp = tempfile.mktemp(suffix=".jsonl")
    mem = JsonlMemory(tmp)
    for e in experiences:
        mem.store(e)
    return mem


def _exp(tags: list[str], importance: float = 0.8) -> Experience:
    return Experience(
        summary="test summary",
        lesson="test lesson",
        importance=importance,
        tags=tags,
    )


def test_query_or_mode_default_unchanged():
    """Default (require_all_tags=False) preserves original tag-OR behaviour —
    any experience sharing at least one requested tag is returned."""
    mem = _write_temp_memory([
        _exp(["probe", "goal_met"]),  # matches both
        _exp(["probe"]),              # matches "probe"
        _exp(["goal_met"]),           # matches "goal_met"
        _exp(["scan"]),               # matches neither — excluded
    ])
    results = mem.query(tags=["probe", "goal_met"])
    assert len(results) == 3  # scan excluded; the other three each share at least one tag


def test_query_and_mode_requires_all_tags():
    """require_all_tags=True returns only experiences with ALL requested tags."""
    mem = _write_temp_memory([
        _exp(["probe", "goal_met"]),  # has both
        _exp(["scan"]),               # has neither
        _exp(["goal_met"]),           # has only one
        _exp(["probe"]),              # has only one
    ])
    results = mem.query(tags=["probe", "goal_met"], require_all_tags=True)
    assert len(results) == 1
    assert set(results[0].tags) == {"probe", "goal_met"}


def test_query_and_mode_no_match_returns_empty():
    mem = _write_temp_memory([_exp(["probe"]), _exp(["goal_met"])])
    results = mem.query(tags=["probe", "goal_met"], require_all_tags=True)
    assert results == []


def test_query_no_tags_returns_all():
    """No tag filter returns everything (both modes)."""
    mem = _write_temp_memory([_exp(["a"]), _exp(["b"]), _exp(["c"])])
    assert len(mem.query(tags=None)) == 3
    assert len(mem.query(tags=None, require_all_tags=True)) == 3


def test_query_and_mode_single_tag_same_as_or():
    """A single-tag AND query is identical to OR — both just check presence."""
    mem = _write_temp_memory([_exp(["probe", "extra"]), _exp(["scan"])])
    or_results = mem.query(tags=["probe"])
    and_results = mem.query(tags=["probe"], require_all_tags=True)
    assert len(or_results) == len(and_results) == 1


def test_query_limit_respected_in_both_modes():
    mem = _write_temp_memory([_exp(["probe"]) for _ in range(10)])
    assert len(mem.query(tags=["probe"], limit=3)) == 3
    assert len(mem.query(tags=["probe"], limit=3, require_all_tags=True)) == 3
