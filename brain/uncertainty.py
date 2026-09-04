"""
Uncertainty Engine (Phase 2, corrected in Phase 3).

IMPORTANT correctness note (found while building the Information-Gain
Engine): an early version of this used `1 - confidence` as each
hypothesis's uncertainty contribution. That's LINEAR in confidence, and
the belief-update rule in hypotheses.py is a fair/martingale update (its
expected value equals the prior). Combined, that meant
E[uncertainty_after] == uncertainty_before ALWAYS, for every action,
regardless of how distinguishing its predictions were — expected
information gain was silently zero for everything. (Jensen's inequality:
a linear function of a martingale has no expectation gap; you need a
strictly concave function to get one.)

Fixed by using a concave, confidence-symmetric measure instead:
`1 - |2*confidence - 1|` — 0 at confidence=0 or 1 (hypothesis settled
either way), 1 at confidence=0.5 (genuinely undecided). This is the
standard reason info-gain calculations use entropy (also concave) rather
than raw probability; we use this simpler triangular function instead of
true Shannon entropy since these aren't real independent binary events,
documented as a heuristic like the rest of Phase 2/3.

Only ACTIVE hypotheses count. A CONFIRMED or REJECTED hypothesis
contributes nothing further to "how unsettled is the world" — that's also
why testing an already-settled hypothesis correctly scores ~zero
information gain in information_gain.py without any special-case code.
"""
from __future__ import annotations

from .models import Hypothesis, HypothesisStatus, UncertaintyState

COMPETITION_WINDOW = 0.15  # hypotheses within this of the leader count as "competing"
OPEN_QUESTION_THRESHOLD = 0.5  # contribution above this counts as "still genuinely open"


def _contribution(confidence: float) -> float:
    return 1.0 - abs(2.0 * confidence - 1.0)


def compute_uncertainty(hypotheses: list[Hypothesis]) -> UncertaintyState:
    active = [h for h in hypotheses if h.status == HypothesisStatus.ACTIVE]

    if not active:
        return UncertaintyState(overall=1.0, open_questions=["no active hypotheses yet"])

    contributions = sorted(
        ((_contribution(h.confidence), h) for h in active), reverse=True, key=lambda t: t[0]
    )
    top_contribution = contributions[0][0]

    # Overall uncertainty is the MEAN across active hypotheses, not just the
    # top one. Using only the max means a brand-new hypothesis (which
    # always starts near 0.5, i.e. maximally uncertain) resets "overall"
    # back up every time one is proposed, even if earlier hypotheses were
    # genuinely resolved — masking real progress. The mean lets settled
    # hypotheses pull the average down even as new open questions appear,
    # which is what "we're making progress but still have open questions"
    # should look like numerically.
    mean_contribution = sum(c for c, _ in contributions) / len(contributions)

    competing_penalty = 0.0
    if len(contributions) > 1 and (top_contribution - contributions[1][0]) < COMPETITION_WINDOW:
        competing_penalty = 0.1  # more than one hypothesis is genuinely live at once

    overall = min(1.0, max(0.0, mean_contribution + competing_penalty))

    open_questions = [h.statement for c, h in contributions if c >= OPEN_QUESTION_THRESHOLD]
    if not open_questions:
        open_questions = [f"'{contributions[0][1].statement}' not yet confirmed or rejected"]

    return UncertaintyState(overall=overall, open_questions=open_questions)
