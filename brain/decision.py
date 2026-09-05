"""
Decision Engine (Phase 2/3/5).

Deliberately contains ZERO LLM calls, still. "The LLM proposes, the Brain
decides" — candidates come in from planning.py, this scores and picks one.

Phase 5 change: scoring weights (info_gain_weight, cost_sensitivity) now
come from the active Strategy (strategy.py) instead of fixed module
constants, so switching strategy after a stall actually changes behavior.
Passing no strategy keeps the exact Phase 2-4 defaults (BALANCED) — fully
backward compatible with existing call sites.
"""
from __future__ import annotations

from .information_gain import expected_information_gain
from .models import Action, Decision, WorkingState
from .strategy import BALANCED, Strategy

DUPLICATE_PENALTY = -1000.0
STOP_PENALTY = -0.5
# Phase 14: additive bonus for an action that tests a hypothesis known to
# be in a live tension with another ACTIVE hypothesis (state.active_tensions,
# populated by contradiction.find_active_tensions). Chosen to be
# meaningful relative to typical (gain/cost)*weight scores without being
# an automatic override — this is a priority nudge, not a hard rule; a
# candidate with a much higher information gain can still legitimately
# win. Additive (not multiplicative) so it can't invert the ranking of
# two low-value actions into something that looks falsely important.
CONTRADICTION_RESOLUTION_BONUS = 0.5
# Phase 16: when quality_gate.evaluate() says conservative_mode is active,
# these two adjustments raise the bar on what actions Brain is willing to
# commit to. The duplicate penalty multiplier makes Brain more reluctant to
# burn steps re-trying things it has already tried (which is a likely cause
# of oscillation); the info_gain multiplier means it demands more expected
# value before acting. Both are multipliers on their normal values, not
# replacements, so the existing scoring logic stays exactly the same path —
# conservative mode is a scaling overlay, not a different scoring algorithm.
CONSERVATIVE_DUPLICATE_PENALTY_MULTIPLIER = 3.0   # 3x reluctance to repeat
CONSERVATIVE_INFO_GAIN_MULTIPLIER = 1.5            # 1.5x demand for expected value


class DecisionEngine:
    def decide(
        self,
        candidates: list[Action],
        state: WorkingState,
        strategy: Strategy = BALANCED,
        active_tensions: dict[str, str] | None = None,
        conservative_mode: bool = False,
    ) -> Decision:
        """`active_tensions` (Phase 14): None/empty (the default) is the
        exact original Phase 2-5 scoring — every pre-Phase-14 call site
        is unaffected. Passing state.active_tensions lets an action that
        would help resolve a live fork between two of Brain's own
        hypotheses outrank an otherwise-similar alternative that
        wouldn't.

        `conservative_mode` (Phase 16): False (the default) is the exact
        original scoring — all pre-Phase-16 call sites are unaffected.
        True applies a quality-conservatism overlay (stricter duplicate
        penalty, higher info_gain demand) when the live quality tracker
        says the process is looking shaky. See quality_gate.py and the
        module-level constants for the exact multipliers and their
        rationale."""
        if not candidates:
            # Real bug found 2026-08-29: reason_and_plan() deliberately
            # swallows any LLM failure and returns ([], []) (see its own
            # `except Exception: return [], []`) — a reasonable degrade on
            # its own. But this method then crashed on `scored[0]` the
            # moment candidates was ever empty, turning ANY single LLM
            # hiccup (timeout, malformed response, network blip) into an
            # unhandled IndexError that killed the whole session. That
            # directly contradicts core.py's own "Principle 9: Brain can
            # change its mind, doesn't crash" comment a few lines below
            # where this Decision gets consumed. A stop action is a real,
            # well-defined outcome core.py already knows how to handle
            # (short-circuited before ever reaching the project) — using
            # it here means "nothing to decide between" ends the session
            # cleanly instead of crashing it.
            return Decision(
                chosen_action=Action(
                    kind="stop",
                    rationale="no candidates to decide between this step "
                              "(planning produced nothing usable)",
                ),
                alternatives_considered=[],
                rationale="no candidates available — stopping rather than crashing",
            )

        scored = [(self._score(c, state, strategy, active_tensions, conservative_mode), c) for c in candidates]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_action = scored[0]

        rationale_parts = [
            f"selected '{best_action.kind}' (score={best_score:.2f}, strategy={strategy.name}"
            + (", conservative_mode=ON" if conservative_mode else "") + ")"
        ]
        if best_action.tests_hypothesis:
            gain = expected_information_gain(best_action, state)
            rationale_parts.append(f"expected information gain={gain:.3f}")
        if active_tensions and best_action.tests_hypothesis in active_tensions:
            rationale_parts.append("prioritized: resolves a live contradiction with another active hypothesis")
        if conservative_mode:
            rationale_parts.append("quality-conservatism overlay active: raised bar on duplicates and info gain demand")
        if len(scored) > 1:
            rationale_parts.append(
                f"over {len(scored) - 1} other candidate(s), next best scored {scored[1][0]:.2f}"
            )
        rationale = "; ".join(rationale_parts)

        return Decision(
            chosen_action=best_action,
            alternatives_considered=[c for _, c in scored[1:]],
            rationale=rationale,
        )

    def _score(
        self,
        action: Action,
        state: WorkingState,
        strategy: Strategy,
        active_tensions: dict[str, str] | None = None,
        conservative_mode: bool = False,
    ) -> float:
        score = 0.0

        dup_penalty = DUPLICATE_PENALTY
        gain_weight = strategy.info_gain_weight
        if conservative_mode:
            dup_penalty *= CONSERVATIVE_DUPLICATE_PENALTY_MULTIPLIER
            gain_weight *= CONSERVATIVE_INFO_GAIN_MULTIPLIER

        if self._is_duplicate(action, state):
            score += dup_penalty

        gain = expected_information_gain(action, state)
        cost = max(action.cost, 0.1) * strategy.cost_sensitivity
        score += (gain / cost) * gain_weight

        if action.kind == "stop":
            score += STOP_PENALTY

        if action.rationale:
            score += 0.05  # trivial tie-break nudge toward justified actions

        if active_tensions and action.tests_hypothesis in active_tensions:
            score += CONTRADICTION_RESOLUTION_BONUS

        return score

    @staticmethod
    def _is_duplicate(action: Action, state: WorkingState) -> bool:
        return any(
            a.kind == action.kind and a.params == action.params for a in state.actions_taken
        )