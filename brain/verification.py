"""
Verification Engine (Phase 3).

Principle 6: "An observation is not automatically a conclusion." A single
matching observation pushes confidence up, but confidence crossing a
threshold is not by itself proof — it could be one lucky/coincidental
result. This module is the explicit, separately-testable gate that
requires independent corroboration before a hypothesis is allowed to be
marked CONFIRMED or REJECTED, regardless of what the raw confidence number
says.

Kept deliberately simple: count independent supporting/contradicting
observations, don't try to model correlation between them (a real
correlated-evidence discount is future work, noted in the README).
"""
from __future__ import annotations

from .models import Hypothesis, HypothesisStatus

MIN_CONFIRMATIONS = 2  # independent supporting observations required to confirm
MIN_DISCONFIRMATIONS = 2  # independent contradicting observations required to reject


def can_confirm(hypothesis: Hypothesis, confidence_threshold: float) -> bool:
    return (
        hypothesis.confidence >= confidence_threshold
        and len(hypothesis.supporting_evidence) >= MIN_CONFIRMATIONS
    )


def can_reject(hypothesis: Hypothesis, confidence_floor: float) -> bool:
    return (
        hypothesis.confidence <= confidence_floor
        and len(hypothesis.contradicting_evidence) >= MIN_DISCONFIRMATIONS
    )


def verification_note(hypothesis: Hypothesis, confidence_threshold: float, confidence_floor: float) -> str:
    """Human-readable explanation of why a hypothesis is or isn't verified
    yet — for inspectability (Principle 13), not just a bool."""
    if hypothesis.status != HypothesisStatus.ACTIVE:
        return f"already {hypothesis.status.value}"
    if hypothesis.confidence >= confidence_threshold and len(hypothesis.supporting_evidence) < MIN_CONFIRMATIONS:
        return (
            f"confidence ({hypothesis.confidence:.2f}) crossed the threshold on a single "
            f"observation — treated as plausible, not confirmed, until "
            f"{MIN_CONFIRMATIONS - len(hypothesis.supporting_evidence)} more independent "
            f"confirmation(s) arrive"
        )
    return "insufficient evidence either way"
