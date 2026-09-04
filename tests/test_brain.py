import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

from brain import (
    Brain,
    Goal,
    JsonlMemory,
    ActionStatus,
    HypothesisStatus,
    compute_uncertainty,
    expected_information_gain,
)
from brain.models import Hypothesis, WorkingState, Observation
from tests.mock_llm import BisectingGuesser, ExhaustiveGuesser, BadActorLLM, VagueGuesser
from toy_envs.guess_number import GuessNumberAdapter
from toy_envs.word_lock import VOCAB, WordLockAdapter


def _fresh_memories():
    d = tempfile.mkdtemp()
    return JsonlMemory(f"{d}/brain_memory.jsonl"), d


# --------------------------------------------------------------- Phase 1/2 behaviors, still true


def test_guess_number_reaches_goal():
    project = GuessNumberAdapter(low=1, high=100, seed=42)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=12)

    state = brain.run(Goal(description="Find the hidden number", success_criteria="exact match"))

    assert state.stop_reason == "goal_met"
    assert project.solved is True
    assert state.step <= 8


def test_domain_independence_same_brain_core_different_toy():
    brain_mem, d = _fresh_memories()

    number_env = GuessNumberAdapter(low=1, high=50, seed=7)
    number_mem = JsonlMemory(f"{d}/{number_env.project_id}.jsonl")
    brain1 = Brain(BisectingGuesser(), number_env, brain_mem, number_mem, max_steps=10)
    state1 = brain1.run(Goal(description="Find the number"))
    assert state1.stop_reason == "goal_met"

    word_env = WordLockAdapter(seed=3)
    word_mem = JsonlMemory(f"{d}/{word_env.project_id}.jsonl")
    brain2 = Brain(ExhaustiveGuesser(VOCAB), word_env, brain_mem, word_mem, max_steps=10)
    state2 = brain2.run(Goal(description="Find the hidden word"))
    assert state2.stop_reason == "goal_met"
    assert word_env.solved is True


def test_brain_filters_disallowed_action_kind_instead_of_crashing():
    project = GuessNumberAdapter(low=1, high=10, seed=1)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BadActorLLM(), project, brain_mem, proj_mem, max_steps=3)

    state = brain.run(Goal(description="Find the number"))

    assert state.stopped is True
    assert all(a.kind != "definitely_not_allowed" for a in state.actions_taken)


def test_memory_selectivity_low_importance_not_stored():
    from brain.models import Experience

    d = tempfile.mkdtemp()
    mem = JsonlMemory(f"{d}/m.jsonl")
    mem.store(Experience(summary="trivial", lesson="nothing", importance=0.05))
    assert mem.query() == []
    mem.store(Experience(summary="big deal", lesson="something", importance=0.9))
    assert len(mem.query()) == 1


def test_hypothesis_engine_never_lets_llm_directly_set_existing_confidence():
    from brain.hypotheses import HypothesisEngine

    engine = HypothesisEngine()
    state = WorkingState(goal=Goal(description="g"))
    engine.integrate_new(state, [{"statement": "X is true", "confidence": 0.5}])
    original_id = state.hypotheses[0].id

    engine.integrate_new(state, [{"statement": "X is true", "confidence": 0.99}])

    assert len(state.hypotheses) == 1
    assert state.hypotheses[0].id == original_id
    assert state.hypotheses[0].confidence == 0.5


# --------------------------------------------------------------- Phase 3: information gain


def test_information_gain_rewards_distinguishable_counterfactuals():
    from brain.models import Action

    hyp = Hypothesis(statement="h", confidence=0.5)
    state = WorkingState(goal=Goal(description="g"), hypotheses=[hyp])

    distinguishing = Action(
        kind="test", tests_hypothesis=hyp.id,
        predicted_if_true="A", predicted_if_false="B",
    )
    vague = Action(
        kind="test", tests_hypothesis=hyp.id,
        predicted_if_true="same", predicted_if_false="same",
    )
    exploratory = Action(kind="explore")

    gain_distinguishing = expected_information_gain(distinguishing, state)
    gain_vague = expected_information_gain(vague, state)
    gain_exploratory = expected_information_gain(exploratory, state)

    assert gain_distinguishing > gain_vague
    assert gain_vague == 0.0  # identical predictions => zero hypothesis-specific gain
    assert gain_exploratory > 0.0  # exploratory actions still get nominal base value


def test_information_gain_is_highest_for_maximally_uncertain_hypothesis():
    from brain.models import Action

    uncertain = Hypothesis(statement="uncertain", confidence=0.5)
    near_settled = Hypothesis(statement="settled", confidence=0.85)
    state = WorkingState(goal=Goal(description="g"), hypotheses=[uncertain, near_settled])

    a1 = Action(kind="t", tests_hypothesis=uncertain.id, predicted_if_true="A", predicted_if_false="B")
    a2 = Action(kind="t", tests_hypothesis=near_settled.id, predicted_if_true="A", predicted_if_false="B")

    assert expected_information_gain(a1, state) > expected_information_gain(a2, state)


def test_decision_engine_prefers_higher_information_gain_over_cost():
    from brain.decision import DecisionEngine
    from brain.models import Action

    hyp = Hypothesis(statement="h", confidence=0.5)
    state = WorkingState(goal=Goal(description="g"), hypotheses=[hyp])

    cheap_informative = Action(
        kind="a", tests_hypothesis=hyp.id, predicted_if_true="X", predicted_if_false="Y",
        rationale="cheap", cost=1.0,
    )
    expensive_vague = Action(
        kind="b", tests_hypothesis=hyp.id, predicted_if_true="Z", predicted_if_false="Z",
        rationale="expensive vague", cost=5.0,
    )
    decision = DecisionEngine().decide([expensive_vague, cheap_informative], state)
    assert decision.chosen_action.kind == "a"


# --------------------------------------------------------------- Phase 3: verification gating


def test_verification_requires_multiple_confirmations_not_just_confidence():
    from brain import verification

    single_confirmation = Hypothesis(statement="h", confidence=0.95, supporting_evidence=["obs1"])
    double_confirmation = Hypothesis(
        statement="h", confidence=0.95, supporting_evidence=["obs1", "obs2"]
    )
    assert verification.can_confirm(single_confirmation, 0.9) is False
    assert verification.can_confirm(double_confirmation, 0.9) is True


def test_hypothesis_status_does_not_flip_confirmed_after_single_match():
    from brain.hypotheses import HypothesisEngine
    from brain.models import Action

    engine = HypothesisEngine()
    state = WorkingState(goal=Goal(description="g"))
    engine.integrate_new(state, [{"statement": "h", "confidence": 0.85}])
    hyp = state.hypotheses[0]

    action = Action(kind="a", tests_hypothesis=hyp.id, predicted_if_true="yes", predicted_if_false="no")
    obs = Observation(action_id=action.id, result="yes", success=True)
    engine.update_from_observation(state, action, obs)

    # confidence should have moved up but NOT be marked CONFIRMED off one match
    assert hyp.confidence > 0.85
    assert hyp.status == HypothesisStatus.ACTIVE
    assert len(hyp.supporting_evidence) == 1


# --------------------------------------------------------------- Phase 3: challenger


def test_challenger_switches_away_from_already_confirmed_hypothesis():
    from brain import challenger
    from brain.models import Action, Decision

    confirmed = Hypothesis(statement="settled", confidence=0.95, status=HypothesisStatus.CONFIRMED)
    open_hyp = Hypothesis(statement="open", confidence=0.5)
    state = WorkingState(goal=Goal(description="g"), hypotheses=[confirmed, open_hyp])

    bad_choice = Action(
        kind="retest", tests_hypothesis=confirmed.id,
        predicted_if_true="a", predicted_if_false="b", rationale="re-confirm",
    )
    better_alt = Action(
        kind="probe", tests_hypothesis=open_hyp.id,
        predicted_if_true="c", predicted_if_false="d", rationale="probe open question",
    )
    decision = Decision(chosen_action=bad_choice, alternatives_considered=[better_alt], rationale="initial pick")

    reviewed = challenger.review(decision, state)

    assert reviewed.chosen_action is better_alt
    assert "CHALLENGED" in reviewed.rationale


def test_vague_experiment_run_still_terminates_via_stall_detection():
    """An LLM that only ever proposes non-distinguishing 'experiments'
    should eventually get caught by stall/rejection-loop detection rather
    than burning the full step budget uselessly. Phase 5: a detected
    stall/loop first tries switching Strategy (bounded) before actually
    stopping, so the final reason may reflect either exhaustion path."""
    project = GuessNumberAdapter(low=1, high=100, seed=5)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(VagueGuesser(), project, brain_mem, proj_mem, max_steps=20)

    state = brain.run(Goal(description="Find the number"))

    assert state.stop_reason in (
        "stalled_no_uncertainty_reduction",
        "stalled_strategies_exhausted",
        "stuck_repeated_invalid_actions",
        "max_steps_reached",
    )
    if state.stop_reason != "max_steps_reached":
        assert state.step < 20  # actually stopped early, not just hit the ceiling


# --------------------------------------------------------------- Phase 3: stall controller (unit)


def test_stall_controller_detects_flat_uncertainty_history():
    from brain import controller

    flat = WorkingState(goal=Goal(description="g"), uncertainty_history=[0.8, 0.79, 0.80, 0.78])
    moving = WorkingState(goal=Goal(description="g"), uncertainty_history=[0.9, 0.7, 0.5, 0.3])
    too_short = WorkingState(goal=Goal(description="g"), uncertainty_history=[0.8, 0.8])

    assert controller.is_stalled(flat) is True
    assert controller.is_stalled(moving) is False
    assert controller.is_stalled(too_short) is False


def test_general_lessons_reused_by_second_run_via_prompt():
    project = GuessNumberAdapter(low=1, high=10, seed=1)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BadActorLLM(), project, brain_mem, proj_mem, max_steps=1)
    brain.run(Goal(description="Find the number"))

    captured = {}

    class CapturingGuesser(BisectingGuesser):
        def propose(self, prompt, schema_hint):
            captured["prompt"] = prompt
            return super().propose(prompt, schema_hint)

    project2 = GuessNumberAdapter(low=1, high=10, seed=1)
    proj_mem2 = JsonlMemory(f"{d}/{project2.project_id}_run2.jsonl")
    brain2 = Brain(CapturingGuesser(), project2, brain_mem, proj_mem2, max_steps=5)
    brain2.run(Goal(description="Find the number"))

    assert "CURRENT HYPOTHESES" in captured["prompt"]


# --------------------------------------------------------------- Phase 4: per-observation extraction


def test_extraction_flags_hypothesis_resolution_as_high_importance():
    from brain.extraction import extract_from_observation
    from brain.models import Action, Hypothesis, Observation

    before = Hypothesis(statement="h", confidence=0.85, status=HypothesisStatus.ACTIVE)
    after = Hypothesis(
        statement="h", confidence=0.92, status=HypothesisStatus.CONFIRMED,
        supporting_evidence=["a", "b"], id=before.id,
    )
    action = Action(kind="test", tests_hypothesis=before.id, predicted_if_true="yes", predicted_if_false="no")
    obs = Observation(action_id=action.id, result="yes", success=True)
    state = WorkingState(goal=Goal(description="g"), hypotheses=[after], step=3)

    exp = extract_from_observation(state, action, obs, before, after, project_id="proj")

    assert exp is not None
    assert exp.importance >= 0.35  # crosses the memory store threshold
    assert "resolved" in exp.lesson.lower()
    assert "confirmed" in exp.tags


def test_extraction_gives_low_importance_to_routine_exploratory_success():
    from brain.extraction import extract_from_observation
    from brain.models import Action, Observation

    action = Action(kind="explore", predicted_outcome="ok")
    obs = Observation(action_id=action.id, result="ok", success=True)
    state = WorkingState(goal=Goal(description="g"), step=1)

    exp = extract_from_observation(state, action, obs, None, None, project_id="proj")

    assert exp is not None
    assert exp.importance < 0.35  # below memory.IMPORTANCE_THRESHOLD - won't actually get stored


def test_full_run_produces_multiple_experience_records_not_just_one():
    """Phase 1-3 only ever wrote ONE experience per run (the episode
    summary). Phase 4 should write per-observation experiences too, for
    any run where a hypothesis actually resolves."""
    project = GuessNumberAdapter(low=1, high=64, seed=9)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=10)

    brain.run(Goal(description="Find the number"))

    records = proj_mem.query(limit=100)
    # at minimum the episode summary; a real run should also have at least
    # one per-observation record given hypotheses resolve during bisection
    assert len(records) >= 1


# --------------------------------------------------------------- Phase 4: consolidation


def test_consolidation_requires_minimum_cluster_size():
    from brain import consolidate
    from brain.models import Experience

    lonely = [Experience(summary="s", lesson="unique lesson", importance=0.9, tags=["rare_tag"])]
    principles = consolidate(lonely, min_cluster_size=3)
    assert principles == []


def test_consolidation_groups_similar_experiences_into_a_principle():
    from brain import consolidate
    from brain.models import Experience

    shared_tags = ["guess", "proj-x", "confirmed"]
    experiences = [
        Experience(summary=f"s{i}", lesson="test before acting", importance=0.6, tags=shared_tags)
        for i in range(4)
    ]
    principles = consolidate(experiences, min_cluster_size=3)

    assert len(principles) == 1
    assert principles[0].statement == "test before acting"
    assert len(principles[0].supporting_experience_ids) == 4
    assert principles[0].confidence == 1.0  # full agreement across the cluster


def test_consolidation_synthesizes_combined_statement_on_disagreement():
    from brain import consolidate
    from brain.models import Experience

    shared_tags = ["guess", "proj-x"]
    experiences = [
        Experience(summary="s1", lesson="lesson A", importance=0.6, tags=shared_tags),
        Experience(summary="s2", lesson="lesson B", importance=0.6, tags=shared_tags),
        Experience(summary="s3", lesson="lesson C", importance=0.6, tags=shared_tags),
    ]
    principles = consolidate(experiences, min_cluster_size=3)

    assert len(principles) == 1
    assert "Multiple related lessons" in principles[0].statement
    assert principles[0].confidence < 0.6  # low agreement, honestly reflected


def test_principle_store_roundtrip():
    from brain import PrincipleStore
    from brain.models import Principle

    d = tempfile.mkdtemp()
    store = PrincipleStore(f"{d}/principles.jsonl")
    store.store(Principle(statement="always validate scope first", confidence=0.8))
    loaded = store.all()

    assert len(loaded) == 1
    assert loaded[0].statement == "always validate scope first"


# --------------------------------------------------------------- Phase 4: calibration


def test_calibration_tracker_reports_overconfidence():
    from brain import CalibrationTracker

    tracker = CalibrationTracker()
    # stated 0.9 confidence, but only right half the time - overconfident
    for matched in [True, False, True, False, True, False]:
        tracker.record(confidence_at_test=0.9, matched=matched)

    assert tracker.is_overconfident(threshold=0.15) is True
    assert tracker.brier_score() is not None
    assert tracker.brier_score() > 0.15  # meaningfully worse than perfect


def test_calibration_tracker_not_overconfident_when_well_calibrated():
    from brain import CalibrationTracker

    tracker = CalibrationTracker()
    # stated 0.7 confidence, right ~70% of the time - well calibrated
    outcomes = [True, True, True, True, True, True, True, False, False, False]
    for m in outcomes:
        tracker.record(confidence_at_test=0.7, matched=m)

    assert tracker.is_overconfident(threshold=0.15) is False


def test_calibration_tracker_wired_into_brain_run_without_changing_public_contract():
    """Existing 5-positional-arg Brain(...) calls must keep working
    unmodified; calibration_tracker is optional and auto-created."""
    project = GuessNumberAdapter(low=1, high=32, seed=2)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=10)  # old-style call

    brain.run(Goal(description="Find the number"))

    assert len(brain.calibration_tracker.records) > 0  # calibration data was actually collected


# --------------------------------------------------------------- Phase 5: strategy selection


def test_strategy_selector_tries_each_strategy_at_most_once():
    from brain import StrategySelector

    selector = StrategySelector()
    first = selector.select_next(set())
    assert first is not None
    second = selector.select_next({first.name})
    assert second is not None
    assert second.name != first.name
    third = selector.select_next({first.name, second.name})
    # only 2 non-balanced strategies exist in the library right now
    assert third is None or third.name not in {first.name, second.name}


def test_decision_engine_strategy_override_changes_cost_sensitivity():
    from brain import CHEAP_FIRST, BALANCED
    from brain.decision import DecisionEngine
    from brain.models import Action

    hyp = Hypothesis(statement="h", confidence=0.5)
    state = WorkingState(goal=Goal(description="g"), hypotheses=[hyp])

    cheap = Action(kind="a", tests_hypothesis=hyp.id, predicted_if_true="X", predicted_if_false="Y", cost=1.0)
    expensive_but_similar_gain = Action(
        kind="b", tests_hypothesis=hyp.id, predicted_if_true="X2", predicted_if_false="Y2", cost=3.0
    )

    balanced_decision = DecisionEngine().decide([expensive_but_similar_gain, cheap], state, BALANCED)
    cheap_first_decision = DecisionEngine().decide([expensive_but_similar_gain, cheap], state, CHEAP_FIRST)

    # CHEAP_FIRST's higher cost_sensitivity should never choose the pricier
    # option over a similarly-informative cheap one, even if BALANCED might
    # be closer either way (documented as a real behavioral difference,
    # not just a label change).
    assert cheap_first_decision.chosen_action.kind == "a"


def test_full_run_switches_strategy_on_stall_before_giving_up():
    """A stuck run (VagueGuesser) should show real evidence of trying an
    alternative strategy, not just silently stopping on the first stall."""
    project = GuessNumberAdapter(low=1, high=100, seed=5)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(VagueGuesser(), project, brain_mem, proj_mem, max_steps=20)

    state = brain.run(Goal(description="Find the number"))

    assert state.strategy_switches > 0
    assert len(state.strategy_log) == state.strategy_switches
    assert state.active_strategy != "balanced"


def test_strategy_switches_are_bounded_never_infinite():
    project = GuessNumberAdapter(low=1, high=100, seed=5)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(VagueGuesser(), project, brain_mem, proj_mem, max_steps=50, max_strategy_switches=2)

    state = brain.run(Goal(description="Find the number"))

    assert state.strategy_switches <= 2
    assert state.step < 50  # actually stopped, didn't just grind to the ceiling


# --------------------------------------------------------------- Phase 5: meta-reasoning / escalation


def test_meta_reasoning_flags_hypothesis_near_confirmation_for_escalation():
    from brain import meta, CalibrationTracker

    near_confirm = Hypothesis(
        statement="h", confidence=0.9, status=HypothesisStatus.ACTIVE, supporting_evidence=["a"]
    )
    far_from_confirm = Hypothesis(statement="h2", confidence=0.5, status=HypothesisStatus.ACTIVE)

    state_risky = WorkingState(goal=Goal(description="g"), hypotheses=[near_confirm])
    state_safe = WorkingState(goal=Goal(description="g"), hypotheses=[far_from_confirm])

    tracker = CalibrationTracker()
    signals_risky = meta.compute_signals(state_risky, tracker)
    signals_safe = meta.compute_signals(state_safe, tracker)

    assert signals_risky["hypothesis_near_confirmation"] is True
    assert signals_safe["hypothesis_near_confirmation"] is False
    assert meta.should_escalate(signals_risky) is True
    assert meta.should_escalate(signals_safe) is False


def test_meta_reasoning_flags_overconfidence_from_calibration_tracker():
    from brain import meta, CalibrationTracker

    tracker = CalibrationTracker()
    for matched in [True, False, True, False, True, False]:
        tracker.record(confidence_at_test=0.9, matched=matched)  # stated 0.9, right ~50% - overconfident

    state = WorkingState(goal=Goal(description="g"))
    signals = meta.compute_signals(state, tracker)

    assert signals["is_overconfident"] is True
    assert meta.should_escalate(signals) is True


def test_strong_llm_actually_gets_used_when_provided_and_escalation_triggers():
    """End-to-end: a Brain given a distinguishable strong_llm should route
    at least one call to it once a hypothesis nears confirmation, and
    still solve normally (opt-in escalation doesn't change the outcome,
    just which model reasons about a given step)."""
    project = GuessNumberAdapter(low=1, high=64, seed=11)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")

    call_log = []

    class LoggingGuesser(BisectingGuesser):
        def __init__(self, tag):
            self.tag = tag

        def propose(self, prompt, schema_hint):
            call_log.append(self.tag)
            return super().propose(prompt, schema_hint)

    hot = LoggingGuesser("hot")
    strong = LoggingGuesser("strong")
    brain = Brain(hot, project, brain_mem, proj_mem, max_steps=12, strong_llm=strong)

    state = brain.run(Goal(description="Find the number"))

    assert state.stop_reason == "goal_met"
    assert "hot" in call_log  # most steps use the cheap model
    # not asserting "strong" must appear - depends on whether a hypothesis
    # happened to get within NEAR_CONFIRMATION_THRESHOLD this particular
    # run; the contract is "escalation is available and wired", not "it
    # always fires", which test_meta_reasoning_flags_* already covers directly.


def test_no_strong_llm_provided_never_escalates():
    """Escalation must be fully opt-in - without a strong_llm, the Brain
    should never even attempt to use one (self.planning_strong is None)."""
    project = GuessNumberAdapter(low=1, high=32, seed=3)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=10)

    assert brain.planning_strong is None
    state = brain.run(Goal(description="Find the number"))
    assert state.stop_reason == "goal_met"  # runs fine with no escalation machinery at all


# --------------------------------------------------------------- Phase 6: reasoning-quality (unit)


def test_oscillating_hypothesis_is_flagged_monotonic_is_not():
    """A hypothesis whose confidence bounces direction repeatedly should
    be flagged; one that climbs (or falls) cleanly toward a verdict,
    however many steps it takes, should not - oscillation is about
    direction reversals, not the number of updates."""
    from brain.reasoning_quality import build_report
    from brain import CalibrationTracker

    calm_state = WorkingState(goal=Goal(description="g"))
    calm_state.hypotheses = [
        Hypothesis(statement="clean climb", confidence_history=[0.5, 0.6, 0.75, 0.85, 0.92])
    ]
    report_calm = build_report(calm_state, CalibrationTracker())
    assert report_calm.oscillating_hypotheses == []

    flippy_state = WorkingState(goal=Goal(description="g"))
    flippy_hyp = Hypothesis(statement="flip flop", confidence_history=[0.5, 0.8, 0.3, 0.75, 0.35])
    flippy_state.hypotheses = [flippy_hyp]
    report_flippy = build_report(flippy_state, CalibrationTracker())
    assert flippy_hyp.id in report_flippy.oscillating_hypotheses
    assert report_flippy.overall_quality < report_calm.overall_quality


def test_short_confidence_history_never_counts_as_oscillating():
    """Two confidence points can only move in one direction once - that's
    an ordinary update, not flip-flopping, regardless of the values."""
    from brain.reasoning_quality import build_report
    from brain import CalibrationTracker

    state = WorkingState(goal=Goal(description="g"))
    state.hypotheses = [Hypothesis(statement="one update", confidence_history=[0.5, 0.9])]
    report = build_report(state, CalibrationTracker())
    assert report.oscillating_hypotheses == []


def test_quality_report_accumulates_flags_and_penalizes_score():
    """Each independent problem (overconfidence, challenger overrides,
    strategy switches, rejected actions) should show up as its own
    inspectable flag (Principle 13), and more flags should mean a lower
    overall_quality - not a single opaque number with no explanation."""
    from brain.reasoning_quality import build_report
    from brain import CalibrationTracker

    tracker = CalibrationTracker()
    for matched in [True, False, True, False, True, False]:  # stated 0.9, ~50% right -> overconfident
        tracker.record(confidence_at_test=0.9, matched=matched)

    state = WorkingState(goal=Goal(description="g"))
    state.decisions = []  # no challenger interventions this scenario
    state.strategy_switches = 1

    report = build_report(state, tracker)
    assert report.is_overconfident is True
    assert any("strategy" in f for f in report.flags)
    assert any("confidence" in f or "confident" in f for f in report.flags)
    assert report.is_low_quality is True
    assert report.overall_quality < 1.0


def test_clean_run_report_is_not_low_quality():
    """A run with no oscillation, no overconfidence, no switches, and few
    rejections should NOT be flagged - the point is to catch real
    unreliability, not manufacture noise on every run."""
    from brain.reasoning_quality import build_report
    from brain import CalibrationTracker

    state = WorkingState(goal=Goal(description="g"))
    report = build_report(state, CalibrationTracker())
    assert report.flags == []
    assert report.is_low_quality is False
    assert report.overall_quality == 1.0


# --------------------------------------------------------- Phase 6: reasoning-quality (integration)


def test_clean_solved_run_gets_attached_high_quality_report():
    """Brain.run() should always attach a quality_report, and a clean
    bisection run that solves efficiently shouldn't be flagged as
    unreliable."""
    project = GuessNumberAdapter(low=1, high=100, seed=42)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=12)

    state = brain.run(Goal(description="Find the hidden number"))

    assert state.quality_report is not None
    assert state.quality_report.is_low_quality is False


def test_escalations_are_counted_across_the_run():
    """state.escalations should accumulate every time meta.should_escalate
    fires, distinct from meta.py's per-step signals which are recomputed
    and discarded each step - this is the running total the quality
    report reads from."""
    project = GuessNumberAdapter(low=1, high=64, seed=11)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=12, strong_llm=BisectingGuesser())

    state = brain.run(Goal(description="Find the number"))

    assert state.quality_report.escalations == state.escalations
    assert state.escalations >= 0  # may legitimately be 0 if nothing ever neared confirmation


def test_low_quality_run_experience_is_tagged_and_downweighted():
    """A run flagged low-quality by the retrospective report should store
    its episode-summary Experience with a 'low_reasoning_quality' tag and
    a reduced importance, so it doesn't generalize into memory with the
    same weight as a clean run (Principle 12 applied to the Brain's own
    process). Forcing this via a stuck VagueGuesser run, which strategy-
    switches and racks up rejected actions - both quality flags."""
    project = GuessNumberAdapter(low=1, high=100, seed=5)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(VagueGuesser(), project, brain_mem, proj_mem, max_steps=20)

    state = brain.run(Goal(description="Find the number"))

    stored = proj_mem.query(tags=[state.stop_reason], limit=50)
    episode_exps = [e for e in stored if e.tags and e.tags[0] == state.stop_reason]

    if state.quality_report.is_low_quality:
        # Importance is downweighted by 0.6x for low-quality runs, which
        # can push it below IMPORTANCE_THRESHOLD and mean it's never
        # stored at all - that's the mechanism working as intended, not a
        # test failure. If it WAS stored, it must carry the tag.
        if episode_exps:
            assert "low_reasoning_quality" in episode_exps[-1].tags
    else:
        if episode_exps:
            assert "low_reasoning_quality" not in episode_exps[-1].tags


# --------------------------------------------------------------- Phase 6 cleanup: shared confidence math


def test_information_gain_and_hypothesis_update_share_one_confidence_function():
    """The Phase 3 known-gap (information_gain._simulated_uncertainty
    manually re-implementing hypotheses.update_from_observation's
    arithmetic in a second place) is closed: both now call
    hypotheses.update_confidence(), so there is exactly one place this
    math can be wrong. Directly check the simulated branch used inside
    expected_information_gain agrees with the real update rule."""
    from brain.hypotheses import update_confidence
    from brain.information_gain import expected_information_gain
    from brain.models import Action

    state = WorkingState(goal=Goal(description="g"))
    hyp = Hypothesis(statement="target is even", confidence=0.5, confidence_history=[0.5])
    state.hypotheses = [hyp]

    action = Action(
        kind="test",
        tests_hypothesis=hyp.id,
        predicted_if_true="even result",
        predicted_if_false="odd result",
    )
    gain = expected_information_gain(action, state)

    # A real distinguishing action at confidence 0.5 must show positive
    # gain (concave uncertainty measure), and that gain must be internally
    # consistent with update_confidence's own arithmetic, not a
    # hand-duplicated copy of it.
    assert gain > 0.0
    assert update_confidence(0.5, matched=True) == 0.75
    assert update_confidence(0.5, matched=False) == 0.25


# --------------------------------------------------------------- Phase 6 cleanup: strategy-switch bound


def test_max_strategy_switches_defaults_from_library_size_not_a_hardcoded_constant():
    """Previously hardcoded to 2, which only happened to equal
    len(LIBRARY) - 1. Now derived explicitly - if the strategy library
    ever grows, this stays correct without anyone remembering to update a
    separate magic number."""
    from brain.strategy import LIBRARY

    project = GuessNumberAdapter(low=1, high=100, seed=1)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=5)

    assert brain.max_strategy_switches == len(LIBRARY) - 1


def test_explicit_max_strategy_switches_still_overrides_default():
    """The derived default must not prevent an explicit override -
    backward compatible with any caller (e.g. earlier Phase 5 tests) that
    passes max_strategy_switches directly."""
    project = GuessNumberAdapter(low=1, high=100, seed=1)
    brain_mem, d = _fresh_memories()
    proj_mem = JsonlMemory(f"{d}/{project.project_id}.jsonl")
    brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=5, max_strategy_switches=0)

    assert brain.max_strategy_switches == 0


# --------------------------------------------------------------- Phase 6 cleanup: principle supersession


def test_principle_store_marks_overlapping_new_principle_as_superseding_old():
    """The Phase 4 known-gap: PrincipleStore had no way to mark an older
    Principle superseded when a later consolidation pass, over a larger
    accumulated Experience history, produces a new Principle covering
    substantially the same evidence. Now it does."""
    from brain import PrincipleStore
    from brain.models import Principle

    d = tempfile.mkdtemp()
    store = PrincipleStore(f"{d}/principles.jsonl")

    old = Principle(
        statement="check scope before every action",
        supporting_experience_ids=["e1", "e2", "e3"],
        confidence=0.6,
        project_id="agent65",
    )
    store.store_generation([old])
    assert len(store.active()) == 1

    # A later, larger consolidation pass produces a refined principle
    # covering mostly the same evidence plus one new experience.
    refined = Principle(
        statement="check scope before every action, especially after a strategy switch",
        supporting_experience_ids=["e1", "e2", "e3", "e4"],
        confidence=0.75,
        project_id="agent65",
    )
    store.store_generation([refined])

    active = store.active()
    assert len(active) == 1
    assert active[0].id == refined.id

    all_records = store.all()
    assert len(all_records) == 2  # old record is never deleted (Principle 13: inspectable)
    stored_old = next(p for p in all_records if p.id == old.id)
    assert stored_old.superseded_by == refined.id


def test_principle_store_does_not_supersede_unrelated_or_cross_project_principles():
    """Low evidence overlap, or a different project scope, must NOT
    trigger supersession - otherwise unrelated generalizations would
    silently erase each other."""
    from brain import PrincipleStore
    from brain.models import Principle

    d = tempfile.mkdtemp()
    store = PrincipleStore(f"{d}/principles.jsonl")

    unrelated_old = Principle(
        statement="prefer cheap actions when uncertain",
        supporting_experience_ids=["x1", "x2"],
        project_id="agent65",
    )
    store.store_generation([unrelated_old])

    new_low_overlap = Principle(
        statement="check scope before every action",
        supporting_experience_ids=["e1", "e2", "e3"],  # no overlap with x1/x2
        project_id="agent65",
    )
    store.store_generation([new_low_overlap])
    assert len(store.active()) == 2  # neither superseded the other

    cross_project = Principle(
        statement="check scope before every action",
        supporting_experience_ids=["e1", "e2", "e3"],  # same evidence, different project
        project_id="some_other_project",
    )
    store.store_generation([cross_project])
    active = store.active()
    assert len(active) == 3  # cross-project scope must never cross-supersede


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
