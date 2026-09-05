# Universal Brain — Phase 1 + 2 + 3 + 4 + 5 + 6

A domain-independent cognitive core:

```
perceive
  -> meta-reasoning: compute process-quality signals, decide escalation
  -> reason (hypotheses + candidate actions, using active Strategy's
     planning directive, optionally on a stronger LLM this step)
  -> decide (deterministic, real expected information gain, weighted by
     the active Strategy)
  -> challenge (deterministic self-critique)
  -> validate -> act -> observe
  -> update beliefs (deterministic, gated by Verification Engine)
  -> record calibration data, extract per-observation experience
  -> update uncertainty -> on stall/rejection-loop: try switching
     Strategy (bounded) before actually stopping
  -> extract episode-summary experience

  -> at end of run: build a retrospective ReasoningQualityReport
     (Phase 6) - flags oscillating hypotheses, overconfidence, forced
     strategy switches, heavy self-critique overrides; downweights and
     tags the episode Experience if the run's own reasoning looked
     unreliable

(separately, periodic/batch, NOT part of the loop above:)
consolidate accumulated Experiences -> Principles
```

Phases 1-6 implemented and tested (44/44). Not yet connected to agent65 —
proven against two structurally different toy environments.

## Why it's smaller than the original spec

Same principle as every prior phase, applied to Phase 5's three asks
(Meta-Reasoning, Strategy Selection, Model Escalation):

- **"Strategy" is 3 fixed, hand-written parameter bundles**
  (`strategy.py`), not a pluggable framework or an LLM-authored policy. A
  strategy IS a `(decision-weight-tweak, planning-prompt-directive)` pair:
  BALANCED (the Phase 2-4 default), CHEAP_FIRST (bias toward cheap fast
  actions), AGGRESSIVE_FALSIFICATION (bias toward disproving the leading
  hypothesis, not confirming it). This is deliberately the smallest thing
  that makes "change direction" real instead of theater, per Phase 3's
  `controller.py` note that flagged this as deferred.
- **Strategy selection is a fixed priority order, not learned.** Same
  "the Brain decides deterministically" rule as every other engine here —
  `StrategySelector.select_next()` just walks a priority list of untried
  strategies. No ML, no scoring model.
- **Model escalation is two `PlanningEngine` instances, not a router
  abstraction.** `Brain(llm, ..., strong_llm=...)` — if `strong_llm` is
  provided, a second `PlanningEngine` wraps it and `core.py` picks between
  them per-step based on `meta.should_escalate()`. No cost paid for
  escalation machinery when nobody's using it (`self.planning_strong` is
  simply `None`). This maps directly onto agent65's actual two-model
  setup (qwen3:4b hot loop / qwen3:8b deep assessment).
- **Meta-reasoning is 4 signals, not a general self-reflection engine**
  (`meta.py`): challenger intervention count, rejected-action count, Brier
  score / overconfidence from the Phase 4 `CalibrationTracker`, and "is
  any hypothesis about to cross the confirmation threshold on thin
  evidence." Escalate if any one fires — not a tuned weighted score,
  since these are meant to be rare, cheap, individually explainable
  triggers.

## A real bug this phase's testing caught

`VagueGuesser` (the Phase 3 mock that always proposes non-distinguishing
"experiments") was re-run against `GuessNumberAdapter` to check the new
stall-to-strategy-switch behavior, and instead of switching strategy it
just silently burned the full 20-step budget. Cause: `VagueGuesser`
always guesses the literal number 50 regardless of the current bounds, so
once the bounds narrowed past 50, `validate()` started rejecting *every*
action — and the rejection branch in `core.py` did `continue` straight
past the stall-check further down the loop body, every single time. A
rejected action never reaches `hypotheses.update_from_observation`, so
`uncertainty_history` never updates either way — this failure mode is
invisible to `is_stalled()`, which only watches uncertainty. The Brain
was stuck proposing an invalid action forever with nothing able to notice.

Fixed with a second, independent check: `controller.is_rejection_looping()`
watches `state.actions_taken` directly for a run of consecutive REJECTED
actions, and the rejection branch now tries a strategy switch (same
bounded mechanism as a stall) before continuing. Verified with a debug
run: before the fix, `VagueGuesser` reached `max_steps_reached` at step 20
having spammed the same rejected guess 19 times; after the fix, it
correctly detects the loop, tries 2 strategy switches, and stops honestly
at step 7 with `stop_reason="stuck_repeated_invalid_actions"`.

A second, smaller issue found while reviewing (not caught by a failing
test, by inspection): `_try_change_strategy`'s tried-strategy tracking
re-parsed the human-readable log message string to reconstruct which
strategies had already been tried, including one dead line that iterated
an empty list and did nothing. It happened to produce correct results
only because `max_strategy_switches` defaults to exactly the number of
non-default strategies in the library — a coincidence, not a guarantee.
Replaced with an explicit `state.strategies_tried: list[str]` field
instead of parsing prose.

## Phase 6 — Reasoning-Quality Evaluation & Self-Monitoring

The spec's own Phase 5 numbering ("reasoning-quality evaluation, strategy
selection, model selection, escalation, self-monitoring") is mostly
already built — strategy selection, model escalation, and the four
meta-reasoning signals came in under what we've been calling Phase 5.
Rather than re-label existing code as "Phase 6" to fill a phase number,
this phase builds only the piece that was genuinely missing: **a
retrospective account of the run's own reasoning process**, as opposed
to the in-the-moment signals `meta.py` already computes and immediately
discards after each step's escalate/switch decision.

Two concrete gaps this closes:

1. **Oscillation detection.** `calibration.py` only ever looks at a
   single `(confidence_at_test, matched)` pair per hypothesis test — it
   has no notion of a hypothesis's *shape* over time. A hypothesis whose
   confidence bounces `0.5 -> 0.8 -> 0.3 -> 0.75 -> 0.35` looked
   identical to calibration as one that climbs cleanly toward a verdict,
   provided the final Brier contributions happened to average out
   similarly. `reasoning_quality.py` tracks each hypothesis's full
   `confidence_history` (new field on `Hypothesis`, seeded with the
   opening prior, appended on every `update_from_observation`) and counts
   direction reversals — 2+ reversals flags real flip-flopping, not
   ordinary noisy revision.
2. **Feeding self-monitoring back into memory.** Before Phase 6, a run
   that flailed its way to the goal (heavy challenger overrides, forced
   strategy switches, overconfidence) produced exactly the same kind of
   episode-summary `Experience`, weighted exactly the same, as a run
   that reasoned cleanly. `core.py` now builds one
   `ReasoningQualityReport` per run (attached as `state.quality_report`)
   and, if it's flagged low-quality, downweights that run's
   episode-summary `Experience` importance by 0.6x and tags it
   `"low_reasoning_quality"` — so noisy runs don't generalize into Brain
   Memory with full weight. This is Principle 12 ("learn general
   principles, not indiscriminately") applied to the Brain's own process,
   not just world-facts.

Verified with a real `VagueGuesser` run (the mock that always proposes a
fixed, eventually-invalid guess): the resulting report showed 7
challenger overrides, 2 forced strategy switches, and 6 rejected actions
— three independent, individually-explainable flags, `overall_quality`
correctly dropping to 0.3, not a synthetic unit-test-only case.

**What Phase 6 deliberately did NOT build:** a general "reasoning quality
score" that gets fed back as a *scalar into decision-making mid-run* (e.g.
"if quality is dropping, become more conservative this step"). That would
be a second, overlapping control mechanism on top of the existing
stall/rejection-loop/strategy-switch machinery, and nothing yet
demonstrates that the two would agree rather than fight each other. The
report is currently read-only within a run (informational + logged) and
only actually changes behavior *after* the run ends, at the memory-write
step — a deliberately narrow, low-risk place to close the gap first.

## Phase 8/9/10 — Prior-Knowledge Retrieval, Principle Trust Feedback, Domain-Familiarity Escalation

Verified before writing any of this: grepped every core file for
"Principle" — `consolidation.py` synthesizes Principles from accumulated
Experience correctly, but nothing in the live loop (`core.py`,
`planning.py`, `hypotheses.py`) ever read one back. Consolidation was a
dead end — Brain Memory accumulated cross-project wisdom that got stored
and never used. This phase closes that loop, in both directions, plus one
new escalation trigger it enables.

1. **PrincipleRetriever (`priors.py`)** — a new, optional `ProjectAdapter`
   hook, `principle_tags(world_model) -> list[str]`, lets a project
   describe "what's relevant right now" in its own vocabulary (same
   pattern as `project_memory_context`). `PrincipleRetriever.relevant()`
   matches those tags against active Principles' statement text (word
   overlap — same "no embeddings yet" honesty as `consolidation.py`'s
   Jaccard-on-tags, not a semantic upgrade) and ranks by overlap then
   confidence. Retrieved Principles are now injected into the planning
   prompt every step (`planning.py`'s new CROSS-PROJECT PRINCIPLES
   section) — the LLM's proposals can be informed by every other wired
   project's accumulated, outcome-tested experience, not just this run's
   own observations so far.

2. **Abductive seeding** — once per run, only when `state.hypotheses` is
   still empty (never overrides an LLM proposal or an observation),
   `seed_hypotheses_from_principles()` turns the top relevant Principles
   directly into Hypotheses, before a single action has executed. This is
   the concrete mechanism for "expect where the problem will be before
   you've looked," generalized to any domain a project defines — for
   agent65 specifically, this is what lets Brain propose "this endpoint
   shape usually means IDOR" from a Principle synthesized out of past
   engagements, before any probe runs. Seeded confidence is deliberately
   damped (`MAX_SEEDED_CONFIDENCE = 0.6`, well under the 0.9 confirm
   threshold) — a Principle earned its confidence across PAST situations;
   it still has to survive this run's own evidence, same Principle-6 gate
   every LLM-proposed hypothesis already goes through.

3. **Principle trust feedback (`PrincipleStore.record_outcome`)** — the
   write side. Before this, a Principle's confidence was frozen forever at
   synthesis time. Now, when a Principle-seeded hypothesis reaches a real
   terminal verdict (CONFIRMED/REJECTED, gated by the existing
   Verification Engine — never a raw single observation), `core.py`
   credits or debits that Principle's confidence via the same
   `update_confidence` arithmetic hypotheses already use. Tactics that
   keep working across different wired projects earn trust over time;
   ones that don't quietly decay — this is what makes Brain Memory
   accumulated wisdom instead of a static lookup table.

4. **Domain-familiarity escalation (`meta.py`)** — a 5th signal alongside
   the existing four: on step 1 only, if `PrincipleRetriever` found zero
   relevant Principles, escalate to the strong LLM once. Checked only at
   step 1 (not every step) deliberately — a domain that stays unfamiliar
   for the whole run is already covered by the other four signals; this
   one exists specifically to spend a stronger model's judgment on
   *framing* a genuinely new kind of situation, exactly once, not as a
   recurring cost.

All four pieces are opt-in and additive: a `Brain(...)` built without a
`principle_store=` argument, or a `ProjectAdapter` that never overrides
`principle_tags`, behaves exactly as it did through Phase 7 — verified by
running the full pre-existing 50-test suite unmodified (still 50/50)
alongside 11 new tests covering retrieval, damped seeding, the escalation
signal's step-1-only behavior, and an end-to-end Brain run that seeds a
Principle-derived hypothesis, drives it to CONFIRMED, and confirms the
originating Principle's confidence actually moved.

## Phase 11 — Pluggable Similarity: from Exact-Word Matching to Meaning

Phase 8's PrincipleRetriever and Phase 4's consolidation clustering both
compared text by exact word/tag overlap. That's honest and fast, but
blind to meaning: "auth endpoint leaks user data via ID swap" and "broken
object-level authorization on numeric IDs" describe the identical tactic
and share almost no words — the old matching silently treated them as
unrelated. Since the entire point of Brain Memory is that a tactic
learned on one wired project should transfer to a DIFFERENTLY-WORDED
situation on another project, that blindness worked against the core
idea, not around the edges of it.

`similarity.py` makes "how similar are two strings" a pluggable seam
(`SimilarityScorer`), three implementations, cheapest-first:

- **`LexicalOverlapScorer`** — the original word/tag-overlap math,
  extracted unchanged. The default everywhere, so every pre-Phase-11 call
  site (including the full existing test suite, still green, 61/61)
  behaves identically unless it explicitly opts into one of the below.
- **`TfidfCosineScorer`** — a real statistical upgrade (term-frequency
  weighted cosine similarity via scikit-learn, optional dependency, zero
  network calls, no model download) over pure Jaccard. Still
  fundamentally lexical — it does NOT close the IDOR/BOLA example above
  (verified by its own test) — but rewards shared distinctive terms
  better than all-or-nothing word counting.
- **`LLMSemanticScorer`** — the actual meaning-based upgrade. Reuses
  `LLMInterface`, the exact seam `PlanningEngine` already calls, to ask
  whichever model a project already has wired in (local Ollama for
  agent65, or anything else) whether two statements describe the same
  underlying tactic. No new dependency, no new network requirement beyond
  what the project already pays for its own reasoning — this is what
  actually bridges the IDOR/BOLA wording gap (verified end-to-end for
  both consolidation clustering and Principle retrieval).

Deliberately not built: a bundled sentence-embedding model. That would
mean downloading pretrained weights at runtime, a new heavy dependency,
and a fixed vocabulary with no way to improve as Brain accumulates more
experience. `LLMSemanticScorer` gets the "understands meaning" benefit
through a seam Brain already has, at whatever model strength a project
already chose for its own reasoning loop — same spirit as Phase 5's
model-escalation design, not a fourth kind of model dependency.

`consolidate()` and `PrincipleRetriever` both take an optional `scorer=`
argument; omitting it is 100% backward compatible. 13 new tests
(`tests/test_similarity.py`) cover all three scorers individually plus
both wiring points end-to-end, including the flagship zero-shared-words
same-tactic case the lexical default can never solve by construction.

## Phase 12 — Semantic Hypothesis Merging (closing a second fragmentation gap)

Same shape of problem as Phase 8's Principle dead-end, found the same
way — read `hypotheses.py` closely rather than assuming. `integrate_new()`
only ever deduped proposals on EXACT (lowercased, stripped) text match.
If the LLM proposes "the endpoint leaks data via ID swap" on step 2 and
"broken object-level authorization on this ID" on step 5 — the same real
claim, worded differently — Brain silently created two separate
Hypothesis objects. Each then accumulated its own confidence
independently from whatever evidence happened to target it, so BOTH took
longer to reach `CONFIRM_THRESHOLD` than the single correctly-pooled
hypothesis would have. Splitting evidence across duplicate beliefs is a
second, distinct instance of the "can't recognize the same idea in
different words" problem Phase 11 fixed for Principle-matching — this
closes it for within-run belief formation too, and reuses the exact same
`SimilarityScorer` seam rather than inventing a new one.

`HypothesisEngine.integrate_new()` takes an optional `scorer=` argument:

- **Omitted (default)** — behaves exactly as before; the full pre-Phase-12
  test suite needed zero changes to keep passing.
- **Provided** (typically `LLMSemanticScorer`, same reasoning as Phase 11
  — TF-IDF still can't bridge the IDOR/BOLA wording gap, so it's not
  useful here either) — a new proposal similar enough
  (`MERGE_SIMILARITY_THRESHOLD = 0.75`, higher than Phase 8's retrieval
  bar since merging is a stronger claim than "worth mentioning as
  context") to an existing **ACTIVE** hypothesis gets folded into it
  instead of becoming a rival.

Two design choices worth being explicit about, both mirroring principles
already established elsewhere in this codebase:

1. **Only ACTIVE hypotheses are eligible merge targets** — never
   CONFIRMED or REJECTED ones. Merging into a settled hypothesis would
   let a fresh, untested proposal reopen or reinforce a verdict through
   the back door — the exact confirmation-bias pattern `challenger.py`
   already guards against on the decision side, applied here to belief
   formation instead.
2. **A merge only moves the target's confidence while it's still an
   untested opening prior** (`confidence_history` length 1). Once a real
   observation has touched a hypothesis even once, that evidence-tested
   confidence is authoritative — a later duplicate proposal is just
   another guess, not new evidence, and must not dilute it. Same
   "evidence beats a fresh guess" reasoning Phase 9's Principle-feedback
   design already relies on.

Wired through `Brain(..., hypothesis_scorer=...)`, fully opt-in — 7 new
tests (`tests/test_hypothesis_merging.py`) cover the backward-compat
default, the merge itself, confidence averaging on an untested prior,
confidence staying untouched once evidence exists, the CONFIRMED/REJECTED
exclusion, and a below-threshold non-merge. 81/81 total tests passing.

## Phase 13 — Contradiction Detection (the opposite question from Phase 11/12)

Phase 11/12 answer "are these two hypotheses the SAME claim." Nothing
answered the opposite question: are two hypotheses OPPOSITE claims that
cannot both be true? Verified by reading `hypotheses.py` and
`verification.py` — no such check existed anywhere. Concretely: before
this phase, Brain could hold "the target is >= 50" and "the target is <
30" as two ACTIVE hypotheses, and if two different actions happened to
push each one past `CONFIRM_THRESHOLD` independently — neither test aware
of the other — BOTH would become CONFIRMED. An internally inconsistent
belief state, silently accepted.

Worth being explicit about why this couldn't just reuse `SimilarityScorer`
(Phase 11): word-overlap scoring gets this exactly backwards. "the target
is >= 50" and "the target is < 50" share nearly every word — Jaccard or
TF-IDF would call them highly *similar*, when they're actually mutually
exclusive. `contradiction.py`'s `LLMContradictionScorer` is the only
implementation offered, and deliberately so — unlike similarity, there's
no honest lexical approximation for logical negation to offer as a
cheaper first rung.

Scope is deliberately narrow: this only blocks the sharpest, unambiguous
failure — a hypothesis that has earned enough evidence to confirm, but
directly contradicts one **already CONFIRMED**. Two merely-ACTIVE
hypotheses that would conflict if both were confirmed are explicitly left
alone (they're often legitimate competing theories still being tested;
flagging every such tension would be noisy and isn't an actual
consistency violation Brain has committed to yet — a documented
limitation, not an oversight, and a reasonable seed for a future
prioritization-focused phase rather than this one's correctness gate).

When a confirmation is blocked, three things happen together: the
hypothesis stays ACTIVE (its evidence-earned confidence is untouched —
only the status flip is blocked, not the belief), `state.contradictions`
records the standoff for inspectability, and two consequences follow
automatically from work already in place — `meta.py`'s 6th escalation
signal fires (Brain's own beliefs disagree; ask the strong model), and
`planning.py` surfaces the standoff directly in the next prompt as an
"UNRESOLVED CONTRADICTIONS" block naming both statements, so the
escalated reasoning actually sees what needs resolving instead of
starting from scratch.

Wired through `Brain(..., contradiction_scorer=...)`, fully opt-in — 10
new tests (`tests/test_contradiction.py`) cover the scorer in isolation,
the confirmation-blocking gate, the backward-compatible no-scorer path,
the "no real contradiction" pass-through, the ACTIVE-vs-ACTIVE scope
guard, and both escalation-signal cases. 91/91 total tests passing.

## Phase 14 — Contradiction-Aware Prioritization (the half of Phase 13 left undone)

Phase 13 deliberately punted on one thing: it only blocks a confirmation
that would collide with an ALREADY-CONFIRMED hypothesis, and explicitly
leaves two merely-ACTIVE, mutually-exclusive hypotheses alone as "often
legitimate to hold simultaneously." True, but leaving them alone
completely wastes a real signal — if Brain already suspects two of its
own open theories can't both be right, the action that tests EITHER one
is unusually valuable to run next, because whichever way it lands, it
resolves a live fork in Brain's own reasoning rather than just
incrementally updating one isolated belief.

`contradiction.find_active_tensions()` runs the same `ContradictionScorer`
from Phase 13 pairwise over currently-ACTIVE hypotheses (capped at
`ACTIVE_TENSION_MAX_PAIRS = 10` pairs per step — an accepted, documented
cost trade-off: this runs every step, unlike Phase 13's check which only
fires at the rare moment of confirmation). `DecisionEngine.decide()` takes
an optional `active_tensions=` dict and adds a modest, additive
`CONTRADICTION_RESOLUTION_BONUS = 0.5` to any candidate action that tests
a hypothesis known to be in a live tension — enough to win a close tie,
not enough to override an action with a genuinely much higher expected
information gain (verified by its own test). `planning.py` also surfaces
these as a "COMPETING ACTIVE THEORIES" prompt block, same pattern as
Phase 13's contradiction block, so an escalated model sees the fork
directly instead of re-deriving it.

Wired automatically whenever `contradiction_scorer=` is already provided
to `Brain(...)` — no separate flag, since Phase 13 and 14 are two
consumers of the exact same detection primitive, just applied to
CONFIRMED-vs-pending (a correctness gate) and ACTIVE-vs-ACTIVE (a
priority signal) respectively. 7 new tests
(`tests/test_active_tensions.py`) cover tension detection, the exclusion
of non-ACTIVE hypotheses, the pair cap, the priority boost itself, the
backward-compatible no-argument path, and confirming the bonus can't
override a much higher information-gain alternative. 98/98 total tests
passing.

## Phase 15 — Verification-Time Confirmation Review (closing the escalation gap the README itself had flagged)

Every prior phase's escalation work (Phase 5's `should_escalate`, Phase 8's
domain-familiarity trigger, Phase 13's contradiction trigger) escalates the
**planning** step — it makes the next reasoning call use the strong model.
But re-reading `meta.py` next to `hypotheses.py` surfaced a real gap none
of that touched: the actual moment a hypothesis flips to CONFIRMED happens
one call earlier, inside `update_from_observation`, and it is pure
arithmetic (`verification.can_confirm` — a confidence number and an
evidence count) that never sees a model at all, strong or otherwise.
`meta.py` already had a signal named exactly for this moment
(`hypothesis_near_confirmation`) and used it to escalate the *next*
planning step — but by then the hypothesis had already locked in. The
escalation was real; it was just escalating the step after the one that
actually mattered. This was the literal gap this README named at the end
of the Phase 6-era notes: *"the strong model should also double-check a
near-confirmation before it locks in."*

**`brain/confirmation_review.py` (new)** — `ConfirmationReviewer` is a
pluggable interface (same shape as Phase 13's `ContradictionScorer`):
`review(statement, evidence) -> (approved, reason)`. `LLMConfirmationReviewer`
asks whatever LLM is wired in (in practice, the strong one) whether the
evidence genuinely supports the specific claim or is coincidental/too
thin — a judgment about the *content* of the evidence, not something a
count-and-threshold check can make, which is exactly why `verification.py`
already flagged "a real correlated-evidence discount is future work."

Wired narrowly, same scoping discipline as Phase 13/14:

- **Thin-evidence gate only** (`needs_review`, `THIN_EVIDENCE_MAX = 2` —
  deliberately the same bar `meta.py`'s own `NEAR_CONFIRMATION_MAX_EVIDENCE`
  already uses, not a separately-tuned number). A hypothesis confirmed on a
  long, boring trail of consistent evidence never pays for a review call.
- **Can only make Brain more conservative, never less** — it only ever
  looks at hypotheses that already passed the deterministic
  `verification.can_confirm` gate; it cannot confirm something the gate
  said no to, only hold back something the gate said yes to.
- **A hold is a "not yet," not a veto.** `hyp.status` stays ACTIVE, the
  hold reason is recorded in `state.confirmation_holds` (inspectable,
  Principle 13) and surfaced in the next planning prompt (a new
  `HELD CONFIRMATIONS` block, same treatment as Phase 13's
  `UNRESOLVED CONTRADICTIONS`), and a new `meta.py` signal
  (`unresolved_confirmation_hold`) escalates the *next* planning step too
  — so the model picking what to test next actually knows a hold exists,
  not just that a review happened somewhere. As soon as evidence is no
  longer thin (more of it accumulates) or a later review approves it, the
  hold clears and confirmation proceeds normally.
- **Fails OPEN, not closed** — an exception or malformed reviewer response
  approves rather than blocks, same reasoning as `LLMContradictionScorer`:
  a bad model call must never permanently strand an otherwise-earned
  confirmation.
- **Runs before the contradiction check, not after** — a held hypothesis
  never actually reaches CONFIRMED this step, so there is nothing new to
  check for contradiction against other CONFIRMED beliefs; Phase 13's gate
  is simply moot for a held confirmation, not bypassed.

Fully opt-in (`Brain(..., confirmation_reviewer=...)`) — omit it and
behavior is byte-for-byte identical to every pre-Phase-15 call site.

One small, honest housekeeping fix alongside this: `brain/contradiction.py`
(`ContradictionScorer`, `LLMContradictionScorer`, `find_active_tensions`)
had never actually been exported from `brain/__init__.py` despite every
other scorer (`similarity.py`'s three implementations) being exported
there — a caller wiring up Phase 13/14 from outside this package had to
reach into the private module path. Fixed alongside Phase 15's own new
exports, not left for a future session to rediscover.

122/122 tests pass (110 prior + 12 new).

**Where things stand overall:** the confidence/decision core (Phases 1-7)
remains untouched. Phases 8-15 have each closed a distinct, *verified* gap
— found by reading the code against its own stated design, never invented
speculatively. Honest next candidates, in rough order of how sure this is
that they matter:

1. **A real end-to-end run** — against a live LLM (freellmapi or
   otherwise) and a real `ProjectAdapter` (agent65, once it's ready).
   Every phase through 15 has been proven in isolation with unit/
   integration tests; nothing has run as one live loop yet, and that's
   the check no amount of reading code can substitute for.
2. **Embedding-based similarity** as a fourth `SimilarityScorer`
   implementation, sitting between the free `TfidfCosineScorer` and the
   per-call-cost `LLMSemanticScorer` — cheaper than an LLM call, more
   robust than word/tag overlap, useful once there's a real corpus of
   Principles/Experiences to cluster (premature before that, per the
   README's own long-standing note on this).
3. Something surfaced by actually using it, once a real project is wired
   in — the same honest limitation every session has repeated: invented
   test cases can prove the mechanism works, but only real usage surfaces
   which of these fifteen phases is actually pulling its weight day to
   day.

## Phase 17 — Three Real Bugs Found by Reading the Code (not speculative improvements)

Phase 17 is three distinct bug fixes identified by reading the actual
implementations against their stated design intent, not by inventing new
capability. All three were silently wrong — no test had caught any of them
because the corrupted outputs were valid Python values and the wrong behavior
was a plausible-but-incorrect result, not a crash.

**17a — Word-boundary matching in `_outcome_matches_true_branch`
(`hypotheses.py`)**

The original implementation used raw substring containment (`predicted_if_true
in actual`) to decide whether an observation matched the "true" or "false"
branch of a hypothesis test. This silently misfires on short prediction
tokens — exactly the vocabulary an LLM naturally produces when asked to predict
outcomes in plain English:

- `"high"` matches observations containing `"highway"`, `"highlight"`,
  `"highly"`
- `"low"` matches `"flow"`, `"below"`, `"allow"`
- `"yes"` matches `"yesterday"`, `"bytes"`

A false positive here means a contradicting observation is recorded as
supporting evidence — or the reverse — corrupting the confidence trajectory and
calibration tracker for that hypothesis silently for the rest of the run, with
no error, because the `matched=True/False` path executes normally either way.

Fixed using negative lookaround regex (`(?<!\w)token(?!\w)`) for short tokens
(< 20 chars), which requires the prediction to appear as a genuinely isolated
token, not a prefix/suffix of a larger word. Long, descriptive predictions
(`"the probe returns a value between 10 and 50"`) stay on the original
substring path — they're specific enough that a substring hit is genuinely
meaningful. When both branches fail to match (ambiguous result), the existing
correct fallback to `observation.success` is preserved unchanged.

**17b — Useless lesson text in `extraction.py`**

The common case in `_lesson()` — any observation where hypothesis confidence
moved but didn't cross a terminal threshold — returned the string `"Routine,
expected outcome."`. This covers the **majority** of all observations (most
steps are incremental, not decisive). A lesson that says nothing gets
extracted, stored, and eventually clustered by `consolidation.py` into a
Principle that also says nothing — wasting the whole memory pipeline's work on
that episode.

Fixed by describing the actual incremental update: which hypothesis moved, in
which direction, by how much, and to what current confidence. Lessons extracted
from incremental steps now cluster usefully with other incremental steps on the
same hypothesis, and carry enough signal that a future consolidation pass can
recognize "this kind of action consistently moved this kind of hypothesis toward
confirmation" as a real learnable pattern.

**17c — Planning prompt blind beyond 3 steps (`planning.py`)**

The planning prompt showed only `state.observations[-3:]` — the last 3
observations. In a run of more than 3 steps, the planner was completely blind
to everything earlier, including:

- Confirmed/rejected hypotheses from early in the run (so it would
  re-propose hypotheses that were already settled, wasting
  `integrate_new()` calls and burning steps re-testing closed questions)
- Actions that resolved an early hypothesis (so it couldn't build on
  "we already know X via action Y" when deciding what angle to test next)

Fixed by adding two blocks to the prompt alongside the existing 3-step
recency window:

1. **`EARLIER OBSERVATIONS`** — a compact summary (action id, success, first
   60 chars of result) of all observations beyond the recency window, so the
   planner knows what happened without paying the token cost of full detail
   on old steps.
2. **`SETTLED HYPOTHESES`** — all terminal (CONFIRMED/REJECTED) hypotheses
   with their evidence counts, explicitly labelled "do NOT re-propose or
   re-test these." Visible regardless of how long ago they were resolved.

The recency window stays exactly as-is — this adds what it was silently
dropping, not replaces it.

146/146 tests pass (123 prior + 23 new), zero regressions.

## Phase 16 — Live Quality Feedback (closing the read-only mid-run gap)

The README has flagged this gap explicitly since Phase 6: *"The Phase 6
quality report is read-only mid-run — it does not yet feed back into
in-run decision-making (e.g. becoming more conservative the moment quality
starts dropping)."* Phases 7–15 never closed it. This phase does.

**What was wrong:** `reasoning_quality.build_report()` runs exactly once,
after `Brain.run()` completes — a retrospective pass. Inside the live loop
nothing ever reads a running quality picture. A run where quality
deteriorates sharply (oscillating hypotheses, repeated challenger overrides,
overconfidence, repeated invalid actions) kept proposing and deciding at
full speed even as its own process was signalling unreliability. The
existing strategy-switch trigger (`meta.should_change_strategy`) catches
stalls — uncertainty plateaus — but says nothing about process quality. A
Brain actively producing hypotheses and taking actions while being
systematically overconfident and ignoring its own self-critique got no
additional signal asking it to slow down.

**`brain/quality_gate.py` (new)** — `evaluate(state, calibration_tracker)`
runs once per step before `DecisionEngine.decide()`. Checks four signals
already tracked in `WorkingState`:

1. Any hypothesis with `OSCILLATION_REVERSAL_THRESHOLD`+ confidence
   reversals (same threshold as the retrospective report, so both tools
   agree on "bad")
2. Challenger overrides ≥ 2 (two+ means the first didn't fix the tendency)
3. `CalibrationTracker.is_overconfident()` (same live tracker the
   retrospective report already reads)
4. Rejected actions ≥ 3 (the plan-propose-validate loop is broken)

**Conservative mode** triggers when ≥ 2-of-4 signals are active — same
"multiple signals together" bar that `ReasoningQualityReport.is_low_quality`
already uses, so both instruments agree on "bad." When active:

- **`DecisionEngine`** applies a quality-conservatism overlay: duplicate
  penalty ×3 (much more reluctant to re-try things that have already
  failed), info-gain weight ×1.5 (demands more expected value before
  committing to an action). Both are multipliers on the existing scoring
  path, not a different algorithm — the scoring logic is unchanged, only
  the bar is raised. Tests verify a high-info-gain action still beats a
  low-value one even in conservative mode.
- **`meta.py`** gets a new `live_quality_degraded` signal that triggers
  escalation to the strong model for the next planning step — so the plan
  for what to test next gets stronger reasoning behind it when the process
  looks shaky.

**Not sticky:** conservative mode is re-evaluated from scratch every step.
If a strategy switch clears the oscillating trajectory and the challenger
calms down, the signal clears automatically and Brain returns to normal
scoring on the very next step.

**Integrated with the retrospective report:** `state.conservative_mode_steps`
tracks how many steps ran in conservative mode. If that count is > 33% of
total steps, `build_report()` flags it — meaning the final retrospective
knows the process was degraded for a sustained portion of the run, not just
a transient hiccup.

**No new Brain constructor parameters needed.** This runs automatically
inside the loop. The retrospective report's `conservative_mode_steps` field
is new (zero by default — backward-compatible with any code that reads the
report).

123/123 tests pass (110 prior + 13 new), zero regressions.

## Installing / connecting from another repo (e.g. agent65)

This is now a real installable package, not a directory that needs
`sys.path` hacking to import - `pyproject.toml` was missing until this
pass, which mattered less while everything ran inside this one repo's
tests but would have been a real blocker for `agent65` (a separate repo)
actually depending on it.

```bash
# from agent65's repo, or any consumer:
pip install -e /path/to/brain
# or, once published: pip install universal-brain
```

```python
from brain import Brain, Goal, JsonlMemory
from brain.interfaces import ProjectAdapter, LLMInterface
```

Only the `brain/` package itself is installed - `toy_envs/`, `tests/`,
and `examples/` stay dev-only, since a real consumer only ever needs the
core package, never the toy environments used to validate it.

## Layout

```
brain/
  models.py          - + strategy/meta fields on WorkingState (active_strategy,
                        strategy_switches, strategies_tried, strategy_log,
                        escalations, quality_report); Hypothesis now tracks
                        confidence_history for oscillation detection
  interfaces.py       - LLMInterface, ProjectAdapter, MemoryBackend
  memory.py           - JSONL memory backend + importance scoring
  hypotheses.py        - HypothesisEngine: LLM proposes, code owns belief revision
  verification.py      - Principle 6 gate: plausible not confirmed
  uncertainty.py        - compute_uncertainty(): pure, concave-in-confidence
  information_gain.py   - expected_information_gain(): real expected-value calc
  planning.py            - PlanningEngine: prompt + candidates, + strategy_directive param
  decision.py             - DecisionEngine: scoring now weighted by active Strategy
  challenger.py            - Self-critique: two narrow deterministic checks
  controller.py             - is_stalled() + is_rejection_looping() (Phase 5)
  strategy.py               - Strategy dataclass, 3-strategy LIBRARY, StrategySelector
  meta.py                   - compute_signals(), should_escalate(), should_change_strategy()
  extraction.py             - Phase 4: per-observation Experience extraction
  consolidation.py          - Phase 4: Experience clustering into Principle
  calibration.py            - Phase 4: CalibrationTracker, Brier score, reliability curve
  reasoning_quality.py      - Phase 6: retrospective ReasoningQualityReport,
                              oscillation detection, is_low_quality gate
  priors.py                 - Phase 8: PrincipleRetriever (relevance-ranked
                              cross-project Principle lookup) +
                              seed_hypotheses_from_principles (damped
                              abductive seeding, once per run)
  similarity.py             - Phase 11: SimilarityScorer interface +
                              LexicalOverlapScorer (default, zero-dep) /
                              TfidfCosineScorer / LLMSemanticScorer
  contradiction.py          - Phase 13/14: ContradictionScorer interface +
                              LLMContradictionScorer + find_active_tensions
  confirmation_review.py    - Phase 15: ConfirmationReviewer interface +
                              LLMConfirmationReviewer + needs_review thin-
                              evidence gate
  quality_gate.py           - Phase 16: evaluate() — lightweight per-step
                              quality tracker; conservative_mode flag fed
                              to DecisionEngine and meta escalation
  core.py                   - Brain: orchestrates the full cycle. No domain imports.
toy_envs/
  guess_number.py     - numeric bisection toy
  word_lock.py        - combinatorial word-guess toy (purely exploratory, no hypotheses)
tests/
  mock_llm.py         - deterministic LLMs incl. one that gets stuck repeating an invalid guess
  test_brain.py       - 44 tests across all six phases + gap-closing cleanup
examples/
  claude_llm_adapter.py - reference adapters for Claude API and local Ollama
  freellmapi_adapter.py - reference adapter for freellmapi (local
                          OpenAI-compatible gateway fronting Groq, Mistral,
                          OpenRouter, Cerebras, Cohere, Google AI Studio,
                          Cloudflare Workers AI, Zhipu AI, Ollama Cloud,
                          HuggingFace Router, ...) — one adapter, freellmapi's
                          own routing chooses the underlying provider
```

## Run the tests

```bash
cd universal-brain
python -m pytest tests/test_brain.py -v
```

## What each Phase 5 piece actually proves

- `test_decision_engine_strategy_override_changes_cost_sensitivity` -
  CHEAP_FIRST's higher cost_sensitivity actually changes which candidate
  wins vs. BALANCED, not just a label on the Decision object.
- `test_full_run_switches_strategy_on_stall_before_giving_up` - a stuck
  real run shows `strategy_switches > 0` and a non-default
  `active_strategy`, not just an immediate stop.
- `test_strategy_switches_are_bounded_never_infinite` -
  `max_strategy_switches` is actually respected; the Brain can't
  strategy-hop forever.
- `test_meta_reasoning_flags_hypothesis_near_confirmation_for_escalation`
  / `..._overconfidence_from_calibration_tracker` - the two most
  important escalation triggers fire on exactly the synthetic cases they
  should and not on their negatives.
- `test_strong_llm_actually_gets_used_when_provided_and_escalation_triggers`
  - confirmed via a live debug run (not just the unit test) that
  escalation actually routes a real planning call to the strong model
  mid-run and the goal still gets reached normally.
- `test_no_strong_llm_provided_never_escalates` - escalation machinery
  costs nothing and does nothing when `strong_llm` isn't provided.

## What each Phase 6 piece actually proves

- `test_oscillating_hypothesis_is_flagged_monotonic_is_not` - a
  flip-flopping confidence trajectory gets flagged and scores lower than
  a clean monotonic climb with the same number of updates - proves this
  is measuring *direction changes*, not just "confidence moved a lot."
- `test_short_confidence_history_never_counts_as_oscillating` - a
  hypothesis with only one update can't be flagged, since a single
  direction change isn't flip-flopping by definition.
- `test_quality_report_accumulates_flags_and_penalizes_score` - each
  independent problem shows up as its own named, inspectable flag
  (Principle 13), and `overall_quality` actually drops as flags
  accumulate rather than being a single opaque number.
- `test_clean_run_report_is_not_low_quality` - a run with nothing wrong
  produces zero flags - the point is catching real unreliability, not
  manufacturing noise on every run.
- `test_clean_solved_run_gets_attached_high_quality_report` /
  `test_escalations_are_counted_across_the_run` - integration-level:
  `Brain.run()` always attaches a `quality_report`, and `escalations`
  correctly accumulates across the whole run rather than being
  recomputed-and-discarded each step like `meta.py`'s signals.
- `test_low_quality_run_experience_is_tagged_and_downweighted` - a real
  stuck `VagueGuesser` run (7 challenger overrides, 2 strategy switches,
  6 rejected actions in a live run, not synthetic) gets its episode
  Experience tagged `low_reasoning_quality` with importance cut to 0.6x -
  confirming the feedback into memory actually fires end-to-end.

## Connecting a real project (e.g. agent65)

Implement `ProjectAdapter`:

- `perceive()` - dict of current facts (recon results, session state, etc.)
- `available_actions(world_model)` - which action *kinds* are legal right now
- `validate(action, world_model)` - catches scope drift, malformed params,
  duplicate actions **before** execution - this is where agent65's existing
  validation logic (hypothesis_engine.py, tools_registry.yaml checks) moves to
- `execute(action)` - dispatches to the Kali SSH runner / diff_requests / etc.
- `is_goal_met(world_model)` (optional) - e.g. confirmed finding recorded
- `project_memory_context(query)` (optional) - pull relevant playbook entries

Nothing about SQLi, IDOR, or Kali should ever appear inside `brain/`.

For model escalation specifically: this is the natural place to plug in
agent65's actual qwen3:4b (hot) / qwen3:8b (strong) split -
`Brain(hot_ollama_llm, project, brain_memory, project_memory, strong_llm=deep_ollama_llm)`.
See `examples/claude_llm_adapter.py`'s `OllamaLLM` for the reference
adapter; point `model` at each respective model name.

## Connecting an LLM: freellmapi (local multi-provider gateway)

If you're running [freellmapi](https://github.com/tashfeenahmed/freellmapi)
locally (its dashboard fronts Groq, Mistral, OpenRouter, Cerebras, Cohere,
Google AI Studio, Cloudflare Workers AI, Zhipu AI, Ollama Cloud,
HuggingFace Router, etc. behind one OpenAI-compatible endpoint with its
own health/fallback routing), `examples/freellmapi_adapter.py` wires it
straight into `LLMInterface` with zero new dependencies (stdlib `urllib`,
same choice `OllamaLLM` above already made):

```python
from freellmapi_adapter import FreeLLMAPILLM

hot    = FreeLLMAPILLM(model="auto:fast",  api_key="freellmapi-your-unified-key")
strong = FreeLLMAPILLM(model="auto:smart", api_key="freellmapi-your-unified-key")

brain = Brain(hot, project, brain_memory, project_memory, strong_llm=strong)
```

`model="auto:fast"` / `"auto:smart"` are freellmapi's own routing
strings, not fixed model names — Brain's hot path and its Phase 5
escalation path each get freellmapi's live health-checked best pick for
that goal (speed vs. intelligence) across whichever of your ten providers
happen to be healthy right now, rather than betting everything on two
model names that might individually go down. `api_key` falls back to the
`FREELLMAPI_KEY` environment variable if omitted. Verified offline
against a mock server (`tests/verify_freellmapi_adapter.py` — plain JSON,
code-fenced JSON, malformed-JSON error contract, 4xx fail-fast, 5xx
retry-then-recover, env-var key fallback; 6/6 passing) since a real
freellmapi instance isn't reachable from this sandbox.

## What's deliberately NOT here yet

**Note:** this section predates Phases 13-15. Two items it originally
listed — cross-hypothesis contradiction detection, and verification-time
escalation on near-confirmations — are no longer open; see the Phase 13/14
and Phase 15 sections above. Left the rest of this list as-is rather than
silently editing history:

- Confidence calibration is still not auto-applied to belief updates
  (unchanged from Phase 4 - still needs real project volume to be
  trustworthy).
- Real semantic (embedding-based) clustering for consolidation - Phase 11
  added a *pluggable* similarity interface (word-overlap / TF-IDF / LLM),
  but a true embedding-vector implementation still doesn't exist; see the
  Phase 15 section's "honest next candidates" above.
- Strategies are hand-written, not learned or LLM-authored - synthesizing
  new strategies from consolidated Principles is a real future candidate,
  but needs real usage data to judge whether it's worthwhile at all.
- The Phase 6 quality report is read-only mid-run (logged, and used only
  at the memory-write step at the end) - it does not yet feed back into
  in-run decision-making (e.g. becoming more conservative the moment
  quality starts dropping). See the Phase 6 section above for why that
  was deliberately deferred rather than built speculatively.
- Benchmark suite vs. a non-Brain baseline agent65.

## Gaps closed in this pass

Three previously-flagged issues were real enough to fix now rather than
defer again:

- **`information_gain._simulated_uncertainty` no longer duplicates
  confidence-update math.** Extracted `hypotheses.update_confidence()` as
  the single source of truth for the belief-update rule; both the real
  update (`hypotheses.update_from_observation`) and the simulated one
  (`information_gain.expected_information_gain`) now call it. Flagged
  since Phase 3 as a drift risk — closed, not just re-flagged.
- **`max_strategy_switches` is derived, not a coincidental hardcoded
  constant.** Was `2`, which happened to equal `len(LIBRARY) - 1` (the
  number of non-BALANCED strategies) by luck rather than by construction.
  Now computed from `len(STRATEGY_LIBRARY) - 1` at `Brain.__init__` time,
  with an explicit override still available. Verified the derived value
  matches the old hardcoded one exactly, so this changes nothing about
  current behavior — only removes the trap for when the library grows.
- **`Principle` supersession**, unresolved since Phase 4: `PrincipleStore`
  now has `store_generation()`, which marks an existing active Principle
  `superseded_by` a new one when they cover substantially the same
  evidence (Jaccard overlap of `supporting_experience_ids` ≥ 0.5, same
  project scope) rather than leaving two consolidation passes over
  growing history silently indistinguishable. Old records are never
  deleted (Principle 13: inspectable) — `active()` is what filters for a
  caller that only wants current guidance. Cross-project principles and
  low-overlap principles are explicitly tested to never cross-supersede.

## Known gaps still open (deliberately not fixed here)

- `REJECTION_STREAK_LIMIT = 4` (controller.py) and the strategy priority
  order in `strategy.py`'s `_SWITCH_PRIORITY` are both hand-picked, not
  calibrated against real outcome data - same category of gap as
  `STALL_WINDOW`/`STALL_EPSILON` flagged in earlier phases. Left alone
  deliberately: tuning these against toy-environment behavior would be
  overfitting to toys, not real signal. This is exactly the kind of
  constant that should be revisited once agent65 is generating volume,
  not before.
- Real semantic (embedding-based) clustering for consolidation - Phase 11
  made this pluggable but the highest-fidelity option remains an LLM call
  per comparison, not a cheap embedding vector. Reasonable upgrade, not an
  architectural gap; adding an embedding dependency before there's a real
  corpus to cluster would be premature.
- Benchmark suite vs. a non-Brain baseline agent65 - still deferred until
  there's a real `ProjectAdapter` to benchmark.
- ~~Cross-hypothesis contradiction detection~~ - built, Phase 13/14.
- ~~Verification-time escalation on near-confirmations~~ - built, Phase 15.
