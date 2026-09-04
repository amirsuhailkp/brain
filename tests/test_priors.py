import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import Brain, Goal, JsonlMemory, HypothesisStatus, meta
from brain.consolidation import PrincipleStore
from brain.interfaces import LLMInterface, ProjectAdapter
from brain.models import Action, Principle, WorkingState, WorldModel, Goal as GoalModel
from brain.priors import PrincipleRetriever, seed_hypotheses_from_principles, MAX_SEEDED_CONFIDENCE
from tests.mock_llm import BisectingGuesser
from toy_envs.guess_number import GuessNumberAdapter


def _store_with(*principles: Principle) -> PrincipleStore:
    d = tempfile.mkdtemp()
    store = PrincipleStore(f"{d}/principles.jsonl")
    for p in principles:
        store.store(p)
    return store


# --------------------------------------------------------------- PrincipleRetriever


def test_retriever_finds_tag_overlapping_principle():
    p_relevant = Principle(statement="auth endpoints often leak via idor", confidence=0.6)
    p_irrelevant = Principle(statement="rate limits get bypassed with header spoofing", confidence=0.9)
    store = _store_with(p_relevant, p_irrelevant)
    retriever = PrincipleRetriever(store)

    results = retriever.relevant(["idor", "auth"])

    assert p_relevant in results
    assert p_irrelevant not in results


def test_retriever_returns_nothing_without_query_tags():
    store = _store_with(Principle(statement="idor everywhere", confidence=0.5))
    retriever = PrincipleRetriever(store)
    assert retriever.relevant([]) == []


def test_retriever_ignores_superseded_principles():
    old = Principle(statement="idor is common on numeric ids", confidence=0.4)
    new = Principle(
        statement="idor is common on numeric ids",
        confidence=0.7,
        supporting_experience_ids=old.supporting_experience_ids,
    )
    old.superseded_by = new.id
    store = _store_with(old, new)
    results = PrincipleRetriever(store).relevant(["idor"])
    assert old not in results


# --------------------------------------------------------------- seeding


def test_seed_hypotheses_from_principles_damps_confidence():
    state = WorkingState(goal=GoalModel(description="test"))
    hot_principle = Principle(statement="target is >= 2", confidence=0.95)

    seed_hypotheses_from_principles(state, [hot_principle])

    assert len(state.hypotheses) == 1
    seeded = state.hypotheses[0]
    assert seeded.statement == "target is >= 2"
    assert seeded.confidence <= MAX_SEEDED_CONFIDENCE  # never inherits full Principle confidence
    assert state.principle_seeds[seeded.id] == hot_principle.id


def test_seed_hypotheses_skips_duplicates_already_present():
    state = WorkingState(goal=GoalModel(description="test"))
    state.hypotheses.append(
        __import__("brain.models", fromlist=["Hypothesis"]).Hypothesis(statement="target is >= 2")
    )
    seed_hypotheses_from_principles(state, [Principle(statement="target is >= 2", confidence=0.9)])
    assert len(state.hypotheses) == 1  # no duplicate added


# --------------------------------------------------------------- meta escalation (Phase 10)


def test_unfamiliar_domain_escalates_only_on_step_one():
    state = WorkingState(goal=GoalModel(description="test"))
    from brain.calibration import CalibrationTracker

    tracker = CalibrationTracker()

    state.step = 1
    signals = meta.compute_signals(state, tracker, relevant_principle_count=0)
    assert signals["unfamiliar_domain_first_look"] is True
    assert meta.should_escalate(signals) is True

    state.step = 2
    signals_later = meta.compute_signals(state, tracker, relevant_principle_count=0)
    assert signals_later["unfamiliar_domain_first_look"] is False


def test_familiar_domain_does_not_escalate():
    state = WorkingState(goal=GoalModel(description="test"))
    from brain.calibration import CalibrationTracker

    state.step = 1
    signals = meta.compute_signals(state, CalibrationTracker(), relevant_principle_count=3)
    assert signals["unfamiliar_domain_first_look"] is False


def test_no_principle_store_never_triggers_new_signal():
    """Backward compatibility: a caller that never passes
    relevant_principle_count (every pre-Phase-8 call site) must get
    identical escalation behavior to before."""
    state = WorkingState(goal=GoalModel(description="test"))
    from brain.calibration import CalibrationTracker

    state.step = 1
    signals = meta.compute_signals(state, CalibrationTracker())
    assert signals["unfamiliar_domain_first_look"] is False
    assert meta.should_escalate(signals) is False


# --------------------------------------------------------------- Principle feedback (Phase 9)


def test_record_outcome_increases_confidence_on_success():
    p = Principle(statement="x", confidence=0.5)
    store = _store_with(p)
    store.record_outcome(p.id, success=True)
    updated = [x for x in store.all() if x.id == p.id][0]
    assert updated.confidence > 0.5


def test_record_outcome_decreases_confidence_on_failure():
    p = Principle(statement="x", confidence=0.5)
    store = _store_with(p)
    store.record_outcome(p.id, success=False)
    updated = [x for x in store.all() if x.id == p.id][0]
    assert updated.confidence < 0.5


# --------------------------------------------------------------- end-to-end integration


class RepeatProbeAdapter(ProjectAdapter):
    """Minimal domain-neutral env, purpose-built for this test: a single
    action kind that can be repeated indefinitely and always confirms
    whichever hypothesis it's pointed at. GuessNumberAdapter's shrinking
    bounds make it a poor fit for testing the confirm/reject FEEDBACK path
    specifically (its own bisection naturally moves on to a fresh
    hypothesis after one test each) — this isolates Phase 8/9's wiring
    from that unrelated toy-env behavior."""

    project_id = "toy-repeat-probe"

    def __init__(self):
        self.probes = 0

    def perceive(self):
        return {"probes": self.probes}

    def available_actions(self, world_model):
        return ["probe"]

    def validate(self, action, world_model):
        return True, "ok"

    def execute(self, action):
        self.probes += 1
        return True, "match"

    def principle_tags(self, world_model):
        return ["target", "range-guess"]


class RepeatProbeLLM(LLMInterface):
    """Always tests the same hypothesis with an action that matches its
    true branch — drives a seeded hypothesis to CONFIRMED within a couple
    of steps."""

    def propose(self, prompt, schema_hint):
        return {
            "hypotheses": [],
            "candidate_actions": [
                {
                    "kind": "probe",
                    "params": {},
                    "rationale": "re-test the seeded prior",
                    "predicted_if_true": "match",
                    "predicted_if_false": "nomatch",
                    "tests_hypothesis": "target is >= 2",
                    "cost": 1.0,
                }
            ],
        }


def test_brain_seeds_and_credits_a_principle_end_to_end():
    principle = Principle(statement="target is >= 2", confidence=0.6, project_id=None)
    store = _store_with(principle)

    project = RepeatProbeAdapter()
    brain_mem_dir = tempfile.mkdtemp()
    brain_mem = JsonlMemory(f"{brain_mem_dir}/brain_memory.jsonl")
    proj_mem = JsonlMemory(f"{brain_mem_dir}/{project.project_id}.jsonl")

    brain = Brain(
        RepeatProbeLLM(), project, brain_mem, proj_mem, max_steps=5, principle_store=store
    )
    state = brain.run(Goal(description="Investigate the target"))

    assert state.principles_considered  # retrieval actually found the Principle
    assert any(
        h.statement == "target is >= 2" for h in state.hypotheses
    )  # seeded, not just LLM-proposed

    seeded_hyp = next(h for h in state.hypotheses if h.statement == "target is >= 2")
    assert seeded_hyp.status == HypothesisStatus.CONFIRMED

    updated = [p for p in store.all() if p.id == principle.id][0]
    assert updated.confidence > 0.6  # feedback loop credited the Principle on real confirmation
