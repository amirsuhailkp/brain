"""
Confidence Calibration (Phase 4).

A calibrated Brain's "70% confident" hypotheses should turn out true about
70% of the time — not 95%, not 40%. This module doesn't need any external
ground truth to check that: every time a hypothesis-testing action
executes, hypotheses.py already computes whether the observation matched
the TRUE branch. That (confidence_at_test, matched) pair is exactly what a
calibration curve needs, and it's fully observable from inside the Brain's
own run, no oracle required.

Kept as a pure, storage-agnostic recorder + scorer — no LLM, no file I/O
here. core.py owns wiring it into the loop and persisting results.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationRecord:
    confidence_at_test: float  # hypothesis confidence BEFORE this observation
    matched: bool  # did the observation match the predicted TRUE branch


@dataclass
class CalibrationTracker:
    records: list[CalibrationRecord] = field(default_factory=list)

    def record(self, confidence_at_test: float, matched: bool) -> None:
        self.records.append(CalibrationRecord(confidence_at_test, matched))

    def brier_score(self) -> float | None:
        """Mean squared error between stated confidence and outcome (0/1).
        0.0 = perfectly calibrated, 0.25 = no better than always guessing
        50%, 1.0 = perfectly anti-calibrated. None if no data yet."""
        if not self.records:
            return None
        total = sum(
            (r.confidence_at_test - (1.0 if r.matched else 0.0)) ** 2 for r in self.records
        )
        return total / len(self.records)

    def calibration_curve(self, n_bins: int = 5) -> list[dict]:
        """Bin predictions by stated confidence, compare mean predicted
        confidence vs actual match rate per bin. Standard reliability-
        diagram data, for inspectability (Principle 13) rather than a
        single opaque score."""
        if not self.records:
            return []
        bins: list[list[CalibrationRecord]] = [[] for _ in range(n_bins)]
        for r in self.records:
            idx = min(n_bins - 1, int(r.confidence_at_test * n_bins))
            bins[idx].append(r)

        out = []
        for i, b in enumerate(bins):
            if not b:
                continue
            mean_predicted = sum(r.confidence_at_test for r in b) / len(b)
            actual_rate = sum(1 for r in b if r.matched) / len(b)
            out.append(
                {
                    "bin_range": (i / n_bins, (i + 1) / n_bins),
                    "n": len(b),
                    "mean_predicted": round(mean_predicted, 3),
                    "actual_rate": round(actual_rate, 3),
                    "gap": round(actual_rate - mean_predicted, 3),
                }
            )
        return out

    def is_overconfident(self, threshold: float = 0.15) -> bool:
        """True if, on average, stated confidence runs meaningfully higher
        than actual observed match rate — a concrete, checkable version of
        'the Brain thinks it knows more than it does'."""
        curve = self.calibration_curve()
        if not curve:
            return False
        avg_gap = sum(c["gap"] for c in curve) / len(curve)
        return avg_gap < -threshold
