from __future__ import annotations

from brain.hybrid_planning import HybridPlanningEngine
from brain.interfaces import LLMInterface, ProjectAdapter
from brain.models import Action, Goal, WorkingState


class _FakeLLM(LLMInterface):
    def __init__(self):
        self.calls = 0

    def propose(self, prompt, schema_hint):
        self.calls += 1
        return {"hypotheses": [], "candidate_actions": [
            {"kind": "look", "params": {}, "rationale": "llm fallback"}
        ]}


class _DeterministicHitAdapter(ProjectAdapter):
    project_id = "det-hit"

    def perceive(self):
        return {}

    def available_actions(self, world_model):
        return ["probe"]

    def validate(self, action, world_model):
        return True, ""

    def execute(self, action):
        return True, "ok"

    # Explicit Optional annotation, matching ProjectAdapter.deterministic_
    # propose()'s declared contract — without it, Pylance infers this
    # method's return type from ITS OWN body (a plain tuple), and then
    # flags subclasses that legitimately return None as an incompatible
    # override against that narrower inferred type, not against the
    # actual interface.
    def deterministic_propose(
        self, state, allowed_kinds
    ) -> tuple[list[dict], list[Action]] | None:
        return [], [Action(kind="probe", rationale="from playbook")]


class _DeterministicMissAdapter(_DeterministicHitAdapter):
    project_id = "det-miss"

    def deterministic_propose(
        self, state, allowed_kinds
    ) -> tuple[list[dict], list[Action]] | None:
        return [], []  # tried, found nothing


class _NoDeterministicAdapter(_DeterministicHitAdapter):
    project_id = "no-det"

    def deterministic_propose(
        self, state, allowed_kinds
    ) -> tuple[list[dict], list[Action]] | None:
        return None  # doesn't implement it at all -> base class default


def _state():
    return WorkingState(goal=Goal(description="test"))


def test_uses_deterministic_result_without_calling_llm():
    llm = _FakeLLM()
    engine = HybridPlanningEngine(llm)
    hyps, candidates = engine.reason_and_plan(_state(), _DeterministicHitAdapter(), ["probe"])
    assert llm.calls == 0
    assert len(candidates) == 1
    assert candidates[0].kind == "probe"
    assert candidates[0].rationale == "from playbook"


def test_falls_back_to_llm_when_deterministic_finds_nothing():
    llm = _FakeLLM()
    engine = HybridPlanningEngine(llm)
    hyps, candidates = engine.reason_and_plan(_state(), _DeterministicMissAdapter(), ["look"])
    assert llm.calls == 1
    assert candidates[0].kind == "look"


def test_falls_back_to_llm_when_adapter_has_no_deterministic_hook():
    llm = _FakeLLM()
    engine = HybridPlanningEngine(llm)
    hyps, candidates = engine.reason_and_plan(_state(), _NoDeterministicAdapter(), ["look"])
    assert llm.calls == 1


def test_base_project_adapter_default_hook_returns_none():
    """A pre-Phase-7 adapter that never overrides deterministic_propose()
    gets the interface's own default (None) — behaves exactly as before."""

    class Bare(ProjectAdapter):
        project_id = "bare"

        def perceive(self):
            return {}

        def available_actions(self, world_model):
            return []

        def validate(self, action, world_model):
            return True, ""

        def execute(self, action):
            return True, None

    assert Bare().deterministic_propose(_state(), []) is None