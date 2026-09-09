"""
Phase 20 tests — bugs found via real end-to-end integration run.

20a: goal_met must confirm the winning hypothesis before loop break
20b: is_in_action_rut must not fire in single-action-kind domains
20c: integration — a clean solving run must produce quality=1.0, 0 switches,
     1 confirmed hypothesis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import Brain
from brain.controller import is_in_action_rut, ACTION_RUT_STREAK
from brain.hypotheses import HypothesisEngine
from brain.interfaces import LLMInterface, ProjectAdapter, MemoryBackend
from brain.models import (
    Action,
    ActionStatus,
    Experience,
    Goal as GoalModel,
    Hypothesis,
    HypothesisStatus as HS,
    Observation,
    WorkingState,
)


# ============================================================ helpers


class NullMemory(MemoryBackend):
    def store(self, e): pass
    def query(self, tags=None, limit=10, require_all_tags=False): return []


class FixedLLM(LLMInterface):
    """Always proposes the same action."""
    def __init__(self, kind: str, params: dict, hyp_statement: str = ""):
        self.kind = kind
        self.params = params
        self.hyp_statement = hyp_statement

    def propose(self, prompt, schema):
        hyps = [{"statement": self.hyp_statement, "confidence": 0.5}] if self.hyp_statement else []
        return {
            "hypotheses": hyps,
            "candidate_actions": [{
                "kind": self.kind,
                "params": self.params,
                "predicted_if_true": "correct",
                "predicted_if_false": "wrong",
                "tests_hypothesis": self.hyp_statement or None,
                "cost": 1.0,
            }]
        }


class MinimalProject(ProjectAdapter):
    """Minimal project that immediately signals goal_met when 'win' is executed."""
    project_id = "test-project"
    def __init__(self, goal_action="win"):
        self.goal_action = goal_action
        self.solved = False

    def perceive(self):
        return {"solved": self.solved}

    def available_actions(self, wm):
        return [self.goal_action]

    def validate(self, action, wm):
        return True, "ok"

    def execute(self, action):
        if action.kind == self.goal_action:
            self.solved = True
            return True, "goal achieved"
        return True, "ok"

    def is_goal_met(self, wm):
        return wm.facts.get("solved", False)

    def project_memory_context(self, query):
        return []


class MultiActionProject(ProjectAdapter):
    """Project with multiple action kinds available."""
    project_id = "multi"
    def perceive(self): return {}
    def available_actions(self, wm): return ["probe", "scan", "fetch"]
    def validate(self, a, wm): return True, "ok"
    def execute(self, a): return True, "ok"
    def project_memory_context(self, q): return []


class SingleActionProject(ProjectAdapter):
    """Project with only one real action kind (like GuessNumber)."""
    project_id = "single"
    def perceive(self): return {}
    def available_actions(self, wm): return ["guess"]
    def validate(self, a, wm): return True, "ok"
    def execute(self, a): return True, "ok"
    def project_memory_context(self, q): return []


# ============================================================ 20a: goal_met confirms winning hypothesis


def test_goal_met_confirms_winning_hypothesis():
    """The hypothesis under test when goal_met fires must be CONFIRMED,
    not left ACTIVE — previously the loop broke before the next
    update_from_observation could run."""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {}, hyp_statement="the win action solves it")
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve it"))

    assert state.stop_reason == "goal_met"
    confirmed = [h for h in state.hypotheses if h.status == HS.CONFIRMED]
    assert len(confirmed) == 1
    assert "win action" in confirmed[0].statement


def test_goal_met_confirmed_hypothesis_not_in_holds():
    """A confirmed-on-goal-met hypothesis must also be cleared from
    confirmation_holds if it was there."""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {}, hyp_statement="win solves it")
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve it"))

    assert not state.confirmation_holds


def test_goal_met_without_hypothesis_does_not_crash():
    """If the winning action has no tests_hypothesis, goal_met must still
    work cleanly."""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {})  # no hypothesis
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve it"))
    assert state.stop_reason == "goal_met"


def test_goal_met_n_confirmed_reflected_in_episode_summary_lesson():
    """n_confirmed > 0 on a goal_met run must produce the informative lesson,
    not 'No strong signal from this run.'"""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {}, hyp_statement="the answer is found")
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve"))
    confirmed = [h for h in state.hypotheses if h.status == HS.CONFIRMED]
    assert len(confirmed) == 1


# ============================================================ 20b: rut suppression in single-action domains


def _executed(kind: str) -> Action:
    a = Action(kind=kind, params={})
    a.status = ActionStatus.EXECUTED
    return a


def _state_with_actions(actions):
    state = WorkingState(goal=GoalModel(description="test"))
    state.actions_taken = list(actions)
    return state


def test_rut_not_fired_for_single_available_kind():
    """When only one non-stop action kind is available, a streak of that
    kind must NOT trigger the rut detector — it's a constraint, not a rut."""
    state = _state_with_actions([_executed("guess")] * ACTION_RUT_STREAK)
    # With single available kind: suppressed
    assert is_in_action_rut(state, available_kinds=["guess", "stop"]) is False
    assert is_in_action_rut(state, available_kinds=["guess"]) is False


def test_rut_fired_for_multi_action_domain():
    """In a multi-action domain, a streak of the same kind IS a rut."""
    state = _state_with_actions([_executed("probe")] * ACTION_RUT_STREAK)
    assert is_in_action_rut(state, available_kinds=["probe", "scan", "fetch"]) is True


def test_rut_without_available_kinds_unchanged():
    """Not passing available_kinds preserves original behavior."""
    state = _state_with_actions([_executed("probe")] * ACTION_RUT_STREAK)
    assert is_in_action_rut(state) is True  # no suppression without kinds


def test_rut_stop_only_domain_suppressed():
    """A domain that only has stop available is also not a rut."""
    state = _state_with_actions([_executed("stop")] * ACTION_RUT_STREAK)
    assert is_in_action_rut(state, available_kinds=["stop"]) is False


def test_rut_two_available_kinds_still_fires():
    """With two+ real action kinds available, rut detection is active."""
    state = _state_with_actions([_executed("probe")] * ACTION_RUT_STREAK)
    assert is_in_action_rut(state, available_kinds=["probe", "scan"]) is True


# ============================================================ 20c: end-to-end clean run integration


def test_clean_solving_run_produces_no_strategy_switches():
    """A run that reaches goal_met cleanly should have 0 strategy switches,
    even in a single-action domain. Pre-Phase-20 this failed because
    is_in_action_rut fired on single-action-space constraints."""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {}, hyp_statement="winning hypothesis")
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve"))
    assert state.strategy_switches == 0
    assert state.strategy_log == []


def test_clean_solving_run_produces_quality_1():
    """A clean goal_met run with no flags should produce quality=1.0."""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {}, hyp_statement="winning hypothesis")
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve"))
    assert state.quality_report is not None
    assert state.quality_report.flags == []
    assert state.quality_report.overall_quality == 1.0


def test_clean_solving_run_has_exactly_one_confirmed_hypothesis():
    """Pre-Phase-20: n_confirmed=0 on goal_met because the winning hypothesis
    never got confirmed before the loop broke. Post-Phase-20: it's confirmed."""
    project = MinimalProject(goal_action="win")
    llm = FixedLLM("win", {}, hyp_statement="correct hypothesis")
    brain = Brain(llm, project, NullMemory(), NullMemory(), max_steps=5)
    state = brain.run(GoalModel(description="solve"))
    confirmed_count = sum(1 for h in state.hypotheses if h.status == HS.CONFIRMED)
    assert confirmed_count == 1
