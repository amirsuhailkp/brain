"""Regression test for the bug found via a real agent65 --brain-drive run
on 2026-08-28: an adapter that (correctly, per the interface docstring at
the time) only returns its OWN real action kinds from available_actions()
got every LLM-proposed "stop" rejected as an unregistered tool, burning
real steps instead of ever stopping cleanly. See core.py's run() loop —
"stop" is now a Brain-level built-in the project never needs to know
about.
"""
from __future__ import annotations

from brain.core import Brain
from brain.interfaces import LLMInterface, ProjectAdapter
from brain.models import ActionStatus, Goal, WorldModel


class _FakeLLM(LLMInterface):
    """Always proposes 'stop' — mirrors what a real LLM does once it
    believes there's nothing more to try."""

    def propose(self, prompt, schema_hint):
        return {"hypotheses": [], "candidate_actions": [
            {"kind": "stop", "rationale": "nothing more to try"}
        ]}


class _MinimalAdapter(ProjectAdapter):
    """Deliberately does NOT special-case 'stop' anywhere — exactly what
    Agent65ProjectAdapter looked like before this fix, and exactly what
    the interface docstring now says every adapter is allowed to do."""

    project_id = "minimal"

    def perceive(self):
        return {}

    def available_actions(self, world_model):
        return ["probe"]  # no "stop" — must still work

    def validate(self, action, world_model):
        assert action.kind != "stop", "Brain must never call validate() with a stop action"
        return True, "ok"

    def execute(self, action):
        assert action.kind != "stop", "Brain must never call execute() with a stop action"
        return True, "ok"


def test_llm_proposed_stop_ends_session_without_reaching_adapter():
    brain = Brain(
        llm=_FakeLLM(), project=_MinimalAdapter(),
        brain_memory=_memory(), project_memory=_memory(), max_steps=5,
    )
    state = brain.run(Goal(description="test"))

    assert state.stopped is True
    assert state.stop_reason == "brain_chose_to_stop"
    assert state.step == 1  # stopped on the very first step, not burned steps
    assert state.actions_taken[-1].kind == "stop"
    assert state.actions_taken[-1].status == ActionStatus.EXECUTED  # never REJECTED


def _memory():
    from brain.memory import JsonlMemory
    import tempfile
    return JsonlMemory(tempfile.mktemp(suffix=".jsonl"))


def test_decision_decide_handles_empty_candidates_without_crashing():
    """Regression test — found via a real Brain-drive run where reason_and_
    plan()'s own `except Exception: return [], []` (a deliberate, reasonable
    degrade on ANY llm.propose() failure) fed an empty candidates list into
    decide(), which then crashed with an unhandled IndexError on `scored[0]`.
    That turned any single LLM hiccup into a session-ending crash, directly
    contradicting core.py's own documented Principle 9 ("Brain can change
    its mind, doesn't crash"). decide() must degrade to a clean stop
    instead."""
    from brain.decision import DecisionEngine
    from brain.models import WorkingState

    engine = DecisionEngine()
    state = WorkingState(goal=Goal(description="test"))

    decision = engine.decide([], state)  # must not raise

    assert decision.chosen_action.kind == "stop"
    assert decision.alternatives_considered == []