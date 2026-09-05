"""
The Brain. Phase 5 scope:

    perceive
    -> [meta-reasoning: compute process-quality signals, decide whether
       to escalate this step to a stronger LLM]
    -> reason (hypotheses + candidate actions, incl. counterfactual
       predictions, using the active Strategy's planning directive)
    -> decide (deterministic, scored by real expected information gain,
       weighted by the active Strategy)
    -> challenge (deterministic self-critique, may swap to a better alternative)
    -> validate -> act -> observe
    -> update beliefs (deterministic, gated by the Verification Engine)
    -> record calibration data, extract per-observation experience
    -> update uncertainty, record history
    -> on stall: try switching Strategy (bounded) before actually stopping
    -> extract episode-summary experience

Public contract unchanged since Phase 1: `Brain(llm, project, brain_memory,
project_memory, max_steps).run(goal) -> WorkingState`. Phase 4 added
optional `calibration_tracker`; Phase 5 adds optional `strong_llm` and
`max_strategy_switches` — all with defaults, existing 5-positional-arg
call sites keep working unmodified.

This file must never import a concrete ProjectAdapter or LLM client.
"""
from __future__ import annotations

import copy
import logging

from . import challenger, controller, meta, quality_gate, reasoning_quality
from .calibration import CalibrationTracker
from .consolidation import PrincipleStore
from .contradiction import find_active_tensions
from .decision import DecisionEngine
from .extraction import extract_from_observation
from .hypotheses import HypothesisEngine
from .interfaces import LLMInterface, ProjectAdapter, MemoryBackend
from .memory import score_importance, IMPORTANCE_THRESHOLD
from .models import (
    Action,
    ActionStatus,
    Experience,
    Goal,
    HypothesisStatus,
    Observation,
    WorkingState,
)
from .planning import PlanningEngine
from .priors import PrincipleRetriever, seed_hypotheses_from_principles
from .strategy import BALANCED, LIBRARY as STRATEGY_LIBRARY, StrategySelector
from .uncertainty import compute_uncertainty

logger = logging.getLogger("universal_brain")


class BrainError(Exception):
    pass


class Brain:
    def __init__(
        self,
        llm: LLMInterface,
        project: ProjectAdapter,
        brain_memory: MemoryBackend,
        project_memory: MemoryBackend,
        max_steps: int = 10,
        calibration_tracker: CalibrationTracker | None = None,
        strong_llm: LLMInterface | None = None,
        max_strategy_switches: int | None = None,
        planning_engine_cls: type[PlanningEngine] = PlanningEngine,
        principle_store: PrincipleStore | None = None,
        hypothesis_scorer=None,
        contradiction_scorer=None,
        confirmation_reviewer=None,
    ):
        self.llm = llm
        self.project = project
        self.brain_memory = brain_memory
        self.project_memory = project_memory
        self.max_steps = max_steps
        # Phase 8: optional — a Brain wired up without a PrincipleStore
        # behaves exactly as it did through Phase 7 (no retrieval, no
        # seeding, no feedback). This is how existing call sites (toy env
        # tests, any consumer that hasn't run a consolidation pass yet)
        # stay unaffected.
        self.principle_store = principle_store
        self.principle_retriever = (
            PrincipleRetriever(principle_store) if principle_store is not None else None
        )
        # Phase 12: optional — omitted means integrate_new() keeps its
        # exact original exact-text-only dedup behavior (see hypotheses.py).
        self.hypothesis_scorer = hypothesis_scorer
        # Phase 13: optional — omitted means update_from_observation() can
        # never block a confirmation on a contradiction check (see
        # hypotheses.py).
        self.contradiction_scorer = contradiction_scorer
        # Phase 15: optional — omitted means a hypothesis that clears
        # verification.can_confirm() locks in as CONFIRMED immediately,
        # exactly today's (pre-Phase-15) behavior. When provided, it's
        # given the chance to hold back a confirmation reached on the
        # bare minimum of evidence (see confirmation_review.py /
        # hypotheses.py) — the actual "double-check a near-confirmation
        # before it locks in" verification-time escalation the README
        # flagged as a real, still-open gap through Phase 14.
        self.confirmation_reviewer = confirmation_reviewer
        self.calibration_tracker = calibration_tracker or CalibrationTracker()
        # Was hardcoded to 2, which happened to equal len(STRATEGY_LIBRARY) - 1
        # (the number of non-BALANCED strategies) only by coincidence - the
        # thing that actually makes `_try_change_strategy`'s bounding
        # correct-by-construction is "can't switch to more strategies than
        # exist to switch to." Derived explicitly now so this stays correct
        # if the strategy library ever grows.
        self.max_strategy_switches = (
            max_strategy_switches
            if max_strategy_switches is not None
            else len(STRATEGY_LIBRARY) - 1
        )

        self.planning = planning_engine_cls(llm)
        # Only build a second Planning Engine if a stronger model was
        # actually provided — no cost paid for escalation machinery when
        # nobody's using it. Phase 5's model-escalation is opt-in.
        self.planning_strong = planning_engine_cls(strong_llm) if strong_llm else None

        self.hypotheses = HypothesisEngine()
        self.decision = DecisionEngine()
        self.strategy_selector = StrategySelector()

    # ---------------------------------------------------------------- run

    def run(self, goal: Goal) -> WorkingState:
        state = WorkingState(goal=goal)
        state.world_model.update(self.project.perceive())
        current_strategy = BALANCED

        while not state.stopped and state.step < self.max_steps:
            state.step += 1
            logger.info("step %s: %s", state.step, goal.description)

            if self.project.is_goal_met(state.world_model):
                state.stopped = True
                state.stop_reason = "goal_met"
                break

            allowed_kinds = self.project.available_actions(state.world_model)
            if not allowed_kinds:
                state.stopped = True
                state.stop_reason = "no_actions_available"
                break
            # "stop" is a Brain-level built-in, not a project action kind —
            # always legal to propose, regardless of what the project
            # returns. Without this, "stop" only worked for a
            # ProjectAdapter that happened to reimplement all three of
            # (include it in available_actions, treat it as always-valid
            # in validate, no-op it in execute) correctly, which the
            # interface never actually documented anywhere — a project
            # that forgot any one of those three (e.g. Agent65ProjectAdapter
            # originally did) got "stop" rejected as an unregistered tool,
            # burning real steps instead of ever stopping cleanly.
            allowed_kinds = [*allowed_kinds, "stop"]

            relevant_principles = self._retrieve_relevant_principles(state)

            # Phase 8: fill a genuine void, never override anything — only
            # fires once, only when nothing (LLM, prior step, project) has
            # proposed a single hypothesis yet.
            if state.step == 1 and not state.hypotheses and relevant_principles:
                seed_hypotheses_from_principles(state, relevant_principles)
                state.uncertainty = compute_uncertainty(state.hypotheses)
                state.uncertainty_history.append(state.uncertainty.overall)
                logger.info(
                    "step %s: seeded %d hypothesis(es) from cross-project Principles",
                    state.step, len(state.principle_seeds),
                )

            # Phase 16: lightweight live-quality check before every
            # plan+decide cycle — distinct from the retrospective
            # quality_report built once at the end of run(). Answers one
            # question per step: "is the process looking shaky RIGHT NOW?"
            # so Brain can be more conservative on its next decision without
            # waiting for the post-mortem.
            live_quality = quality_gate.evaluate(state, self.calibration_tracker)
            conservative_mode = live_quality["conservative_mode"]
            if conservative_mode:
                state.conservative_mode_steps += 1
                logger.info(
                    "step %s: live quality degraded (%d signal(s) active), "
                    "entering conservative mode",
                    state.step, live_quality["active_signal_count"],
                )

            signals = meta.compute_signals(
                state,
                self.calibration_tracker,
                relevant_principle_count=(
                    len(relevant_principles) if self.principle_retriever is not None else None
                ),
                live_quality_degraded=conservative_mode,
            )
            planner = self.planning
            if self.planning_strong is not None and meta.should_escalate(signals):
                planner = self.planning_strong
                state.escalations += 1
                logger.info("step %s: escalating to strong LLM (%s)", state.step, signals)

            proposed_hyps, candidates = planner.reason_and_plan(
                state,
                self.project,
                allowed_kinds,
                current_strategy.planning_directive,
                principles=relevant_principles,
            )
            self.hypotheses.integrate_new(state, proposed_hyps, scorer=self.hypothesis_scorer)
            self._reresolve_hypothesis_refs(candidates, state)

            state.uncertainty = compute_uncertainty(state.hypotheses)
            if state.hypotheses:
                # Only track uncertainty history once hypothesis-driven
                # reasoning has actually started. A purely exploratory
                # agent that never forms hypotheses would otherwise sit at
                # a constant 1.0 "no active hypotheses" reading forever,
                # which would look identical to a genuine stall — but it
                # isn't one; there's nothing to have stalled on yet.
                state.uncertainty_history.append(state.uncertainty.overall)

            decision = self.decision.decide(
                candidates, state, current_strategy,
                active_tensions=self._active_tensions(state),
                conservative_mode=conservative_mode,
            )
            decision = challenger.review(decision, state)  # deterministic self-critique
            state.decisions.append(decision)
            action = decision.chosen_action

            if action.kind == "stop":
                # Never routed to the project — "stop" isn't a real tool
                # call, it's the Brain choosing to end the session. No
                # ProjectAdapter should ever see this kind in validate()
                # or execute().
                action.status = ActionStatus.EXECUTED
                state.actions_taken.append(action)
                state.stopped = True
                state.stop_reason = "brain_chose_to_stop"
                break

            valid, reason = self.project.validate(action, state.world_model)
            if not valid:
                action.status = ActionStatus.REJECTED
                state.actions_taken.append(action)
                logger.warning("action rejected: %s", reason)
                if controller.is_rejection_looping(state):
                    switched, current_strategy = self._try_change_strategy(state, current_strategy)
                    if not switched:
                        state.stopped = True
                        state.stop_reason = "stuck_repeated_invalid_actions"
                continue  # Principle 9: Brain can change its mind, doesn't crash

            hyp_before = self._snapshot_hypothesis(state, action.tests_hypothesis)

            success, raw_result = self.project.execute(action)
            action.status = ActionStatus.EXECUTED if success else ActionStatus.FAILED
            state.actions_taken.append(action)

            obs = Observation(action_id=action.id, result=raw_result, success=success)
            state.observations.append(obs)

            self.hypotheses.update_from_observation(
                state,
                action,
                obs,
                contradiction_scorer=self.contradiction_scorer,
                confirmation_reviewer=self.confirmation_reviewer,
            )
            hyp_after = self._find_hypothesis(state, action.tests_hypothesis)

            if hyp_before is not None and hyp_after is not None:
                matched_true_branch = hyp_after.confidence > hyp_before.confidence
                self.calibration_tracker.record(hyp_before.confidence, matched_true_branch)

            self._record_principle_feedback(state, hyp_before, hyp_after)

            self._extract_and_store_observation_experience(state, action, obs, hyp_before, hyp_after)

            state.world_model.update(self.project.perceive())

            if controller.is_stalled(state):
                switched, current_strategy = self._try_change_strategy(state, current_strategy)
                if not switched:
                    state.stopped = True
                    state.stop_reason = "stalled_strategies_exhausted"

        if not state.stopped:
            state.stop_reason = "max_steps_reached"
            state.stopped = True

        # Phase 6: retrospective self-monitoring pass, once, after the loop
        # is done - distinct from meta.compute_signals(), which only ever
        # looks at a single step's snapshot to make an in-the-moment call.
        state.quality_report = reasoning_quality.build_report(state, self.calibration_tracker)
        if state.quality_report.flags:
            logger.info(
                "step %s: reasoning-quality flags: %s", state.step, state.quality_report.flags
            )

        self._extract_episode_summary(state)
        return state

    # ------------------------------------------------------- strategy switching

    def _try_change_strategy(self, state: WorkingState, current_strategy):
        """Phase 3 only ever stopped on a detected stall. Phase 5 tries a
        different Strategy first (bounded by max_strategy_switches) — this
        is the actual 'change direction' half of Stop/Continue/
        Change-Direction that controller.py's docstring flagged as
        deferred. Returns (switched: bool, strategy_to_use)."""
        if not meta.should_change_strategy(state, self.max_strategy_switches):
            return False, current_strategy

        # Every strategy this run has actually been active under is
        # exactly {current_strategy} plus whatever state.strategy_log
        # already switched TO in earlier calls this run. Tracking that as
        # its own list (rather than re-parsing the human-readable log
        # strings) keeps this correct regardless of how the log message
        # is worded.
        tried_names = set(state.strategies_tried) | {current_strategy.name}
        next_strategy = self.strategy_selector.select_next(tried_names)
        if next_strategy is None:
            return False, current_strategy

        state.strategy_switches += 1
        state.strategies_tried.append(next_strategy.name)
        state.strategy_log.append(
            f"switched from '{current_strategy.name}' to '{next_strategy.name}' "
            f"after stall at step {state.step} (uncertainty flat around "
            f"{state.uncertainty.overall:.2f})"
        )
        state.active_strategy = next_strategy.name
        logger.info("step %s: strategy switch -> %s", state.step, next_strategy.name)

        # Give the new strategy a clean window before the stall detector
        # can fire again — otherwise it immediately re-triggers on the
        # same flat history that just caused this switch.
        state.uncertainty_history.clear()

        return True, next_strategy

    # ------------------------------------------------------- priors (Phase 8/9)

    def _retrieve_relevant_principles(self, state: WorkingState) -> list:
        """Returns [] (not None) whether the reason is "no PrincipleStore
        wired up" or "wired up but genuinely nothing relevant" — callers
        that only need the list (planning.py) don't need to care which;
        meta.compute_signals is the one caller that DOES need to tell the
        two apart, which is why it takes an explicit count rather than
        being handed this list directly."""
        if self.principle_retriever is None:
            return []
        query_tags = self.project.principle_tags(state.world_model)
        principles = self.principle_retriever.relevant(query_tags)
        for p in principles:
            if p.id not in state.principles_considered:
                state.principles_considered.append(p.id)
        return principles

    def _record_principle_feedback(self, state: WorkingState, hyp_before, hyp_after) -> None:
        """Phase 9: close the loop the other direction. Only credits a
        Principle when the hypothesis IT seeded reaches a real terminal
        verdict THIS step (status differs before vs after) — never on an
        in-between confidence wobble, and never twice for the same
        verdict, since hyp_before/hyp_after are only ever adjacent steps."""
        if self.principle_store is None or hyp_after is None:
            return
        principle_id = state.principle_seeds.get(hyp_after.id)
        if principle_id is None:
            return  # this hypothesis wasn't seeded by a Principle - nothing to credit
        before_status = hyp_before.status if hyp_before is not None else None
        if hyp_after.status == before_status:
            return  # no verdict change this step
        if hyp_after.status == HypothesisStatus.CONFIRMED:
            self.principle_store.record_outcome(principle_id, success=True)
        elif hyp_after.status == HypothesisStatus.REJECTED:
            self.principle_store.record_outcome(principle_id, success=False)

    def _active_tensions(self, state: WorkingState) -> dict:
        """Phase 14: recomputed fresh each step (see
        contradiction.find_active_tensions's own docstring for the cost
        trade-off and why that's an accepted one here) and cached onto
        state.active_tensions for inspectability (Principle 13) and for
        planning.py to surface in the next prompt — not just consumed
        internally by DecisionEngine."""
        if self.contradiction_scorer is None:
            state.active_tensions = {}
        else:
            state.active_tensions = find_active_tensions(state.hypotheses, self.contradiction_scorer)
        return state.active_tensions

    # ------------------------------------------------------- helpers

    @staticmethod
    def _reresolve_hypothesis_refs(candidates: list[Action], state: WorkingState) -> None:
        """Candidates referencing a hypothesis by statement text that was
        BRAND NEW this step couldn't be resolved to an id in planning.py
        (it didn't exist yet). Now that integrate_new() has run, resolve
        any still-unresolved reference against the fresh hypothesis list."""
        by_statement = {h.statement.strip().lower(): h.id for h in state.hypotheses}
        known_ids = {h.id for h in state.hypotheses}
        for c in candidates:
            if c.tests_hypothesis and c.tests_hypothesis not in known_ids:
                c.tests_hypothesis = by_statement.get(str(c.tests_hypothesis).strip().lower())

    @staticmethod
    def _find_hypothesis(state: WorkingState, hyp_id: str | None):
        if not hyp_id:
            return None
        return next((h for h in state.hypotheses if h.id == hyp_id), None)

    def _snapshot_hypothesis(self, state: WorkingState, hyp_id: str | None):
        """Deep-enough copy (dataclass fields are simple types/lists here)
        of a hypothesis's state BEFORE this step's observation updates it,
        so calibration and per-observation extraction can compare
        before/after without the "before" object mutating out from under
        them (update_from_observation mutates in place)."""
        hyp = self._find_hypothesis(state, hyp_id)
        return copy.deepcopy(hyp) if hyp is not None else None

    # ------------------------------------------------------- experience (Phase 4: per-observation)

    def _extract_and_store_observation_experience(
        self, state: WorkingState, action: Action, obs: Observation, hyp_before, hyp_after
    ) -> None:
        exp = extract_from_observation(
            state, action, obs, hyp_before, hyp_after, self.project.project_id
        )
        if exp is None or exp.importance < IMPORTANCE_THRESHOLD:
            return  # extraction.py always returns something; memory-worthiness is gated here
        self.project_memory.store(exp)
        if "confirmed" in exp.tags or "rejected" in exp.tags:
            # a settled hypothesis outcome is exactly the kind of thing
            # worth generalizing across projects
            general = Experience(
                summary=exp.summary, lesson=exp.lesson, importance=exp.importance,
                tags=[t for t in exp.tags if t != self.project.project_id], project_id=None,
            )
            self.brain_memory.store(general)

    # ------------------------------------------------------- experience (episode summary)

    def _extract_episode_summary(self, state: WorkingState) -> None:
        """Phase-1-era heuristic extraction, hypothesis- and stall-aware.
        Kept as a per-episode rollup ALONGSIDE the per-observation
        extraction above — different granularity, both useful: this one
        captures 'how did the whole run go', the other captures 'what
        specific moment was surprising'."""
        n_actions = len(state.actions_taken)
        n_failed = sum(1 for a in state.actions_taken if a.status == ActionStatus.FAILED)
        n_rejected = sum(1 for a in state.actions_taken if a.status == ActionStatus.REJECTED)
        n_confirmed = sum(1 for h in state.hypotheses if h.status == HypothesisStatus.CONFIRMED)
        n_hyp_rejected = sum(1 for h in state.hypotheses if h.status == HypothesisStatus.REJECTED)
        stalled = state.stop_reason in ("stalled_no_uncertainty_reduction", "stalled_strategies_exhausted")

        surprise = 1.0 if n_failed > 0 else 0.2
        goal_impact = 1.0 if state.stop_reason == "goal_met" else 0.4
        novelty = 0.7 if (n_confirmed or n_hyp_rejected or stalled or state.strategy_switches) else 0.4
        generality = 0.6 if (n_rejected > 0 or n_hyp_rejected > 0 or stalled) else 0.3

        importance = score_importance(
            novelty=novelty, generality=generality, surprise=surprise, goal_impact=goal_impact
        )

        # Phase 6: a run whose own reasoning process looked unreliable
        # (flip-flopping hypotheses, overconfidence, heavy self-critique
        # overrides, forced strategy switches) shouldn't generalize into
        # memory with the same weight as one that reasoned cleanly, even
        # if it happened to reach the goal. This is Principle 12 applied
        # to the Brain's OWN process, not just world-facts.
        quality = state.quality_report
        tags = [state.stop_reason, self.project.project_id]
        if quality is not None and quality.is_low_quality:
            importance = round(importance * 0.6, 3)
            tags.append("low_reasoning_quality")

        summary = (
            f"Goal '{state.goal.description}' finished after {state.step} steps "
            f"({state.stop_reason}); {n_actions} actions ({n_failed} failed, "
            f"{n_rejected} rejected); {len(state.hypotheses)} hypotheses tracked "
            f"({n_confirmed} confirmed, {n_hyp_rejected} rejected)"
            f"{f'; {state.strategy_switches} strategy switch(es)' if state.strategy_switches else ''}."
        )
        lesson = self._derive_lesson(state, n_rejected, n_failed, n_confirmed, n_hyp_rejected, stalled)
        if quality is not None and quality.is_low_quality:
            lesson += (
                f" Note: this run's own reasoning process showed signs of unreliability "
                f"({'; '.join(quality.flags)}) - weight this lesson lower than a clean run's."
            )

        exp = Experience(
            summary=summary,
            lesson=lesson,
            importance=importance,
            tags=tags,
            project_id=self.project.project_id,
        )
        self.project_memory.store(exp)

        if generality >= 0.6:
            general_exp = Experience(
                summary=summary,
                lesson=lesson,
                importance=importance,
                tags=[t for t in tags if t != self.project.project_id],
                project_id=None,
            )
            self.brain_memory.store(general_exp)

    @staticmethod
    def _derive_lesson(
        state: WorkingState,
        n_rejected: int,
        n_failed: int,
        n_confirmed: int,
        n_hyp_rejected: int,
        stalled: bool,
    ) -> str:
        if state.strategy_switches and state.stop_reason == "goal_met":
            return (
                f"The default strategy stalled but switching (ending on "
                f"'{state.active_strategy}') reached the goal — worth trying the "
                f"non-default strategy earlier next time in similar states."
            )
        if stalled:
            return (
                "Uncertainty stopped decreasing despite continued actions, including after "
                "trying alternative strategies — the candidate actions being generated "
                "weren't actually distinguishing between live hypotheses under any approach "
                "tried. May need a genuinely different action space, not just a different weighting."
            )
        if n_hyp_rejected > 0:
            return (
                "One or more hypotheses were actively disproven by evidence rather than "
                "just abandoned — prioritize actions that can falsify a live hypothesis "
                "over ones that only confirm the leading theory."
            )
        if n_rejected > 0:
            return (
                "Proposed actions were sometimes outside the currently valid action space — "
                "check available_actions()/constraints before committing to a plan step."
            )
        if n_failed > 0:
            return "Some executed actions failed; prefer lower-risk/validating actions first."
        if state.stop_reason == "goal_met" and n_confirmed > 0:
            return (
                "Prioritizing the most genuinely undecided hypothesis (via expected "
                "information gain, not just leading-theory confidence) reached the goal "
                "efficiently; keep favoring distinguishing experiments over confirmations."
            )
        if state.stop_reason == "goal_met":
            return "This action sequence reached the goal efficiently; similar sequencing is worth trying again in similar states."
        return "No strong signal from this run."