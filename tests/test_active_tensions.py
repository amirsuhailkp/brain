import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.contradiction import LLMContradictionScorer, find_active_tensions, ACTIVE_TENSION_MAX_PAIRS
from brain.decision import DecisionEngine, CONTRADICTION_RESOLUTION_BONUS
from brain.interfaces import LLMInterface
from brain.models import Action, Goal, Hypothesis, HypothesisStatus, WorkingState


class FixedTensionLLM(LLMInterface):
    def propose(self, prompt, schema_hint):
        if "target is >= 50" in prompt and "target is < 30" in prompt:
            return {"contradicts": True}
        return {"contradicts": False}


def _active(statement: str, confidence: float = 0.5) -> Hypothesis:
    h = Hypothesis(statement=statement, confidence=confidence, confidence_history=[confidence])
    h.status = HypothesisStatus.ACTIVE
    return h


# --------------------------------------------------------------- find_active_tensions


def test_finds_tension_between_two_active_hypotheses():
    a = _active("target is >= 50")
    b = _active("target is < 30")
    scorer = LLMContradictionScorer(FixedTensionLLM())
    tensions = find_active_tensions([a, b], scorer)
    assert tensions[a.id] == b.id
    assert tensions[b.id] == a.id


def test_no_tension_for_compatible_hypotheses():
    a = _active("target is >= 50")
    c = _active("the sky is blue")
    scorer = LLMContradictionScorer(FixedTensionLLM())
    tensions = find_active_tensions([a, c], scorer)
    assert tensions == {}


def test_confirmed_and_rejected_hypotheses_excluded_from_tension_check():
    a = _active("target is >= 50")
    a.status = HypothesisStatus.CONFIRMED
    b = _active("target is < 30")
    scorer = LLMContradictionScorer(FixedTensionLLM())
    tensions = find_active_tensions([a, b], scorer)
    assert tensions == {}  # a is no longer ACTIVE, so the pair is never even checked


def test_respects_max_checks_cap():
    class CountingLLM(LLMInterface):
        def __init__(self):
            self.calls = 0

        def propose(self, prompt, schema_hint):
            self.calls += 1
            return {"contradicts": False}

    # 6 active hypotheses -> 15 possible pairs, well above a small cap
    hyps = [_active(f"statement {i}") for i in range(6)]
    llm = CountingLLM()
    scorer = LLMContradictionScorer(llm)
    find_active_tensions(hyps, scorer, max_checks=3)
    assert llm.calls == 3


# --------------------------------------------------------------- DecisionEngine wiring


def test_active_tension_boosts_action_priority():
    state = WorkingState(goal=Goal(description="test"))
    low_value_untensioned = Action(kind="probe_a", tests_hypothesis="h1", cost=1.0, rationale="r")
    low_value_tensioned = Action(kind="probe_b", tests_hypothesis="h2", cost=1.0, rationale="r")

    engine = DecisionEngine()

    without_tension = engine.decide([low_value_untensioned, low_value_tensioned], state)
    # With identical action shapes and no tension info, tie-break is arbitrary/first — establish baseline
    assert without_tension.chosen_action.kind in ("probe_a", "probe_b")

    with_tension = engine.decide(
        [low_value_untensioned, low_value_tensioned], state, active_tensions={"h2": "h3"}
    )
    assert with_tension.chosen_action.kind == "probe_b"  # the one testing the tensioned hypothesis wins
    assert "resolves a live contradiction" in with_tension.rationale


def test_no_active_tensions_argument_keeps_original_behavior():
    """Backward compatibility: decide() without active_tensions at all
    (every pre-Phase-14 call site) must score identically to before."""
    state = WorkingState(goal=Goal(description="test"))
    action = Action(kind="probe", tests_hypothesis="h1", cost=1.0)
    decision = DecisionEngine().decide([action], state)  # no active_tensions kwarg
    assert "resolves a live contradiction" not in decision.rationale


def test_tension_bonus_does_not_override_a_much_higher_information_gain():
    """The bonus is additive and modest by design — it's a priority nudge,
    not a hard override. An action with dramatically higher expected
    information gain should still win even without a tension."""
    state = WorkingState(goal=Goal(description="test"))
    # No tests_hypothesis at all -> expected_information_gain and the
    # tension check both no-op for it; use cost to create a clear
    # info-gain-per-cost gap that the flat 0.5 bonus can't close.
    high_value = Action(kind="cheap_probe", tests_hypothesis="h9", cost=0.1, rationale="r")
    tensioned_but_costly = Action(kind="costly_probe", tests_hypothesis="h2", cost=50.0, rationale="r")

    decision = DecisionEngine().decide(
        [high_value, tensioned_but_costly], state, active_tensions={"h2": "h3"}
    )
    assert decision.chosen_action.kind == "cheap_probe"
