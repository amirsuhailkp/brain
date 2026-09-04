"""
Planning Engine (Phase 2/3).

One LLM call per step produces new hypothesis candidates and candidate
next actions together — they're naturally one reasoning act. What's kept
separate architecturally: this module only prepares the prompt and turns
raw LLM output into typed, validated candidates. It never picks one —
that's decision.py's job, deterministically.

Phase 3 addition: candidates tied to a hypothesis are now asked for
COUNTERFACTUAL predictions (predicted_if_true / predicted_if_false), not
just a single predicted_outcome. That's what lets information_gain.py do
real expected-value scoring instead of a confidence-distance proxy — an
action that predicts the same thing whether the hypothesis is true or
false isn't actually an experiment for that hypothesis.
"""
from __future__ import annotations

import json

from .interfaces import LLMInterface, ProjectAdapter
from .models import Action, Hypothesis, WorkingState

SCHEMA_HINT = (
    '{"hypotheses": [{"statement": "...", "confidence": 0.0-1.0}], '
    '"candidate_actions": [{"kind": "...", "params": {}, "rationale": "...", '
    '"predicted_outcome": "... (for exploratory actions with no tests_hypothesis)", '
    '"predicted_if_true": "... (what you would observe IF the tested hypothesis is true)", '
    '"predicted_if_false": "... (what you would observe IF it is false - must differ from '
    'predicted_if_true, or this is not a real experiment)", '
    '"tests_hypothesis": "... (exact statement text, or null) — this field, like every '
    'other value in this JSON, must be REAL text or a real value you are providing, never '
    'a placeholder token or example syntax echoed back literally", "cost": 1.0}]}'
)


class PlanningEngine:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def reason_and_plan(
        self,
        state: WorkingState,
        project: ProjectAdapter,
        allowed_kinds: list[str],
        strategy_directive: str = "",
        principles: list | None = None,
    ) -> tuple[list[dict], list[Action]]:
        """Returns (raw_hypothesis_proposals, candidate_actions). Candidate
        actions are NOT yet validated against the project — core.py does
        that per-candidate before scoring, since validity can depend on
        params the planner made up.

        `strategy_directive` (Phase 5): an extra instruction injected into
        the prompt from the currently active Strategy (strategy.py),
        empty string by default — fully backward compatible with Phase
        2-4 call sites that never pass it.

        `principles` (Phase 8): active cross-project Principles the
        PrincipleRetriever judged relevant to this step's situation,
        surfaced as context so the LLM's proposals can be informed by
        accumulated experience from OTHER wired projects, not just this
        run's own observations so far. None/empty by default — fully
        backward compatible with every call site that predates Phase 8."""
        prompt = self._build_prompt(state, project, allowed_kinds, strategy_directive, principles)
        try:
            raw = self.llm.propose(prompt, SCHEMA_HINT)
        except Exception:
            return [], []

        proposed_hyps = raw.get("hypotheses", []) or []
        raw_candidates = raw.get("candidate_actions", []) or []

        candidates: list[Action] = []
        for c in raw_candidates:
            kind = c.get("kind")
            if kind not in allowed_kinds:
                continue  # never let a disallowed kind become a candidate at all
            try:
                cost = float(c.get("cost", 1.0))
            except (TypeError, ValueError):
                cost = 1.0
            candidates.append(
                Action(
                    kind=kind,
                    params=c.get("params", {}) or {},
                    rationale=str(c.get("rationale", "")),
                    predicted_outcome=str(c.get("predicted_outcome", "")),
                    predicted_if_true=str(c.get("predicted_if_true", "")),
                    predicted_if_false=str(c.get("predicted_if_false", "")),
                    cost=max(0.1, cost),
                    tests_hypothesis=self._resolve_hypothesis_ref(
                        c.get("tests_hypothesis"), state.hypotheses
                    ),
                )
            )

        if not candidates:
            candidates = [Action(kind="stop", rationale="planner produced no valid candidates")]

        return proposed_hyps, candidates

    @staticmethod
    def _resolve_hypothesis_ref(ref: str | None, existing: list[Hypothesis]) -> str | None:
        """LLM refers to hypotheses by statement text, not id (it doesn't
        know ids for brand-new ones yet). If it already matches an existing
        hypothesis, resolve to that id now. Otherwise keep the raw text —
        core.py re-resolves it to a real id after integrate_new() runs,
        since the hypothesis may be brand new this very step."""
        if not ref:
            return None
        ref_norm = ref.strip().lower()
        for h in existing:
            if h.statement.strip().lower() == ref_norm:
                return h.id
        return ref  # unresolved raw text; core.py resolves after integration

    def _build_prompt(
        self,
        state: WorkingState,
        project: ProjectAdapter,
        allowed_kinds: list[str],
        strategy_directive: str = "",
        principles: list | None = None,
    ) -> str:
        recent_obs = state.observations[-3:]
        obs_text = "\n".join(
            f"- action={o.action_id} success={o.success} result={o.result!r}"
            for o in recent_obs
        ) or "(none yet)"

        hyps_text = "\n".join(
            f"- id={h.id} [{h.status.value}] conf={h.confidence:.2f}: {h.statement}"
            for h in state.hypotheses
        ) or "(none yet — propose some)"

        memory_snippets = project.project_memory_context(state.goal.description)
        memory_text = "\n".join(f"- {m}" for m in memory_snippets) or "(none)"

        strategy_block = f"\nCURRENT STRATEGY DIRECTIVE: {strategy_directive}\n" if strategy_directive else ""

        principles_text = "\n".join(
            f"- (confidence {p.confidence:.2f}, from {len(p.supporting_experience_ids)} past "
            f"experience(s)) {p.statement}"
            for p in (principles or [])
        ) or "(none surfaced this step)"
        principles_block = (
            f"\nCROSS-PROJECT PRINCIPLES (accumulated from OTHER prior runs/projects — treat as "
            f"informed prior evidence, not settled fact; still needs testing here):\n{principles_text}\n"
        )

        contradictions_block = ""
        if state.contradictions:
            by_id = {h.id: h for h in state.hypotheses}
            lines = []
            for blocked_id, blocker_id in state.contradictions.items():
                blocked = by_id.get(blocked_id)
                blocker = by_id.get(blocker_id)
                if blocked and blocker:
                    lines.append(
                        f'- "{blocked.statement}" has enough evidence to confirm, but directly '
                        f'contradicts the already-CONFIRMED "{blocker.statement}". Find a '
                        f"discriminating action that can resolve which one is actually true."
                    )
            if lines:
                contradictions_block = (
                    "\nUNRESOLVED CONTRADICTIONS (Brain's own beliefs are in conflict — "
                    "resolving this should be a priority):\n" + "\n".join(lines) + "\n"
                )

        tensions_block = ""
        if state.active_tensions:
            by_id = {h.id: h for h in state.hypotheses}
            seen: set[tuple[str, str]] = set()
            lines = []
            for a_id, b_id in state.active_tensions.items():
                pair = tuple(sorted((a_id, b_id)))
                if pair in seen:
                    continue
                seen.add(pair)
                a, b = by_id.get(a_id), by_id.get(b_id)
                if a and b:
                    lines.append(f'- "{a.statement}" vs. "{b.statement}" — these cannot both be true')
            if lines:
                tensions_block = (
                    "\nCOMPETING ACTIVE THEORIES (both still unproven, but mutually exclusive — "
                    "prioritize an action that can distinguish between them):\n"
                    + "\n".join(lines) + "\n"
                )

        return f"""You are the reasoning+planning component of a cognitive agent. You
propose possibilities and options; you do NOT decide which action runs —
that is chosen by separate deterministic code after you respond, using
expected information gain, so it matters that your predictions are real
and distinguishable, not vague.
{strategy_block}
GOAL: {state.goal.description}
SUCCESS CRITERIA: {state.goal.success_criteria}
CONSTRAINTS: {', '.join(state.goal.constraints) or '(none)'}

CURRENT WORLD STATE:
{json.dumps(state.world_model.snapshot(), indent=2)}

RECENT OBSERVATIONS:
{obs_text}

CURRENT HYPOTHESES (do not repeat these verbatim, only add genuinely new ones;
do not propose actions testing an already-confirmed or already-rejected one):
{hyps_text}

PROJECT-SPECIFIC CONTEXT:
{memory_text}
{principles_block}
{contradictions_block}
{tensions_block}
ALLOWED ACTION KINDS (candidate_actions[].kind MUST be one of these): {allowed_kinds}

Propose:
1. Any NEW hypotheses worth tracking (leave empty list if nothing new).
2. 1-3 CANDIDATE actions (not just one) — different options, not a ranked
   single choice. For any candidate meant to test a hypothesis (new or
   existing), set tests_hypothesis to that hypothesis's exact statement
   text, and give BOTH predicted_if_true and predicted_if_false — they
   must genuinely differ, otherwise the action can't actually distinguish
   whether the hypothesis is true. For purely exploratory actions not
   tied to any hypothesis, use predicted_outcome instead and leave
   tests_hypothesis null.

Respond ONLY with JSON matching the given schema."""