"""
Strategy Selection (Phase 5).

A "strategy" here is deliberately NOT a pluggable framework, a class
hierarchy, or an LLM-authored policy. It's a small, fixed, hand-written
list of named parameter bundles — a strategy IS a
(decision-weight-tweak, planning-prompt-directive) pair, nothing more.
This is the direct continuation of the Phase 3 controller.py decision to
only build "stop", not "change direction", until there was something real
to change TO — this is that something, kept as small as it can be while
still being real:

  - BALANCED: the Phase 2-4 default behavior, unchanged.
  - CHEAP_FIRST: for when the current approach is producing expensive
    dead ends — bias toward cheap, fast actions even at some cost to
    per-action information gain.
  - AGGRESSIVE_FALSIFICATION: for when uncertainty has plateaued — bias
    toward actions that could DISPROVE the leading hypothesis rather than
    ones that only reinforce it. This directly targets confirmation bias,
    the same failure mode challenger.py already checks for, but as a
    standing directive rather than a per-decision veto.

Selection is a plain deterministic rule (try untried strategies in a
fixed priority order), not a learned or LLM-driven choice — consistent
with every other "Brain decides" component in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    info_gain_weight: float = 3.0    # matches decision.py's Phase 2-4 default
    cost_sensitivity: float = 1.0    # multiplier on the cost divisor; higher = more cost-averse
    planning_directive: str = ""     # extra instruction injected into the planning prompt


BALANCED = Strategy(
    name="balanced",
    description="Default: weigh information gain against cost evenly.",
)

CHEAP_FIRST = Strategy(
    name="cheap_first",
    description=(
        "Favor low-cost, fast-turnaround actions over expensive high-value ones. "
        "Use when the current approach is producing costly dead ends."
    ),
    info_gain_weight=2.0,
    cost_sensitivity=2.5,
    planning_directive="Prefer cheap, fast actions over expensive ones, even if slightly less informative.",
)

AGGRESSIVE_FALSIFICATION = Strategy(
    name="aggressive_falsification",
    description=(
        "Actively try to DISPROVE the leading hypothesis rather than confirm it. "
        "Use when confidence has plateaued without new distinguishing information."
    ),
    info_gain_weight=4.0,
    cost_sensitivity=0.7,
    planning_directive=(
        "Propose at least one candidate action specifically designed to DISPROVE the "
        "current leading (highest-confidence) hypothesis, not just confirm it further."
    ),
)

LIBRARY: list[Strategy] = [BALANCED, CHEAP_FIRST, AGGRESSIVE_FALSIFICATION]

# Priority order when picking the NEXT untried strategy after a stall.
# Falsification first: it most directly targets "stuck because every
# candidate only confirms the same leading theory", which is the most
# common cause of a stall given how challenger.py and information_gain.py
# already work. cheap_first second, as a fallback for "stuck because
# every candidate is too expensive/slow to actually execute usefully".
_SWITCH_PRIORITY = ["aggressive_falsification", "cheap_first"]


class StrategySelector:
    def select_next(self, tried_names: set[str]) -> Strategy | None:
        """Returns the next untried strategy to switch to, or None if
        every strategy in the library has already been tried this run."""
        by_name = {s.name: s for s in LIBRARY}
        for name in _SWITCH_PRIORITY:
            if name not in tried_names and name in by_name:
                return by_name[name]
        # fall back to any remaining untried strategy, in library order
        for s in LIBRARY:
            if s.name not in tried_names:
                return s
        return None
