"""
Hybrid Planning Engine (Phase 7).

PlanningEngine.reason_and_plan() calls the LLM every single step — that's
the documented, intentional Phase 2/3 design (see planning.py's own
docstring), and it's a perfectly sound one: LLM proposes, deterministic
code decides. But that is NOT the same as "the Brain does the real
reasoning; the LLM is only asked when the Brain is stuck" — that pattern
didn't exist anywhere in the codebase before this.

HybridPlanningEngine adds it as a strict wrapper around the existing,
already-validated PlanningEngine, not a replacement for it:

    1. Ask project.deterministic_propose(state, allowed_kinds) first.
    2. If it returns a (hyps, candidates) tuple WITH at least one
       candidate action, use that — zero LLM calls this step.
    3. Otherwise (None, or candidates came back empty) fall through to
       the real PlanningEngine.reason_and_plan() — byte-for-byte the
       original behavior, including its own internal error handling.

Trigger is deliberately conservative: the LLM is only skipped when the
deterministic pass produced an actual usable action, never partially
trusted (e.g. no "use deterministic hypotheses but still ask the LLM for
actions" mixing — that would blur which layer is responsible for what,
which is exactly the ambiguity Principle 13, inspectability, guards
against). A project can be tuned to a more aggressive trigger later, but
only after this conservative version's calibration/quality numbers (see
reasoning_quality.py) show its deterministic proposals are actually as
good as the LLM's — same Principle 14 discipline as everywhere else in
this codebase.

Because step 3 is the complete original behavior, any ProjectAdapter that
doesn't implement deterministic_propose() (every adapter written before
this) gets identical behavior to plain PlanningEngine.
"""
from __future__ import annotations

from .interfaces import ProjectAdapter
from .models import Action, WorkingState
from .planning import PlanningEngine


class HybridPlanningEngine(PlanningEngine):
    def reason_and_plan(
        self,
        state: WorkingState,
        project: ProjectAdapter,
        allowed_kinds: list[str],
        strategy_directive: str = "",
    ) -> tuple[list[dict], list[Action]]:
        deterministic = project.deterministic_propose(state, allowed_kinds)
        if deterministic is not None:
            proposed_hyps, candidates = deterministic
            if candidates:
                return proposed_hyps, candidates
            # Deterministic pass ran but found nothing usable this step —
            # conservative trigger means this IS when we call the LLM,
            # not a reason to stop or to return an empty result.

        return super().reason_and_plan(state, project, allowed_kinds, strategy_directive)