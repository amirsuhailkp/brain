"""
Deterministic mock LLMs so the Brain's cognitive cycle can be unit-tested
without a network call. Return the Phase 3 schema:
{"hypotheses": [...], "candidate_actions": [{..., "predicted_if_true",
"predicted_if_false", "cost"}]}.
"""
from __future__ import annotations

from brain.interfaces import LLMInterface


class BisectingGuesser(LLMInterface):
    """Bisects the currently known bounds for guess_number.py, framing each
    guess as testing an explicit, genuinely distinguishable hypothesis
    ('target >= mid'): if true the env reports 'higher', if false it
    reports 'lower' — real counterfactual predictions, not vague ones."""

    def propose(self, prompt: str, schema_hint: str) -> dict:
        lo = self._extract_int(prompt, '"low_bound":')
        hi = self._extract_int(prompt, '"high_bound":')
        mid = (lo + hi) // 2
        statement = f"target is >= {mid}"

        already_have = statement.lower() in prompt.lower()
        hyps = [] if already_have else [{"statement": statement, "confidence": 0.5}]

        return {
            "hypotheses": hyps,
            "candidate_actions": [
                {
                    "kind": "guess",
                    "params": {"number": mid},
                    "rationale": "bisect the known range",
                    "predicted_if_true": "higher",
                    "predicted_if_false": "lower",
                    "tests_hypothesis": statement,
                    "cost": 1.0,
                }
            ],
        }

    @staticmethod
    def _extract_int(prompt: str, key: str) -> int:
        idx = prompt.index(key) + len(key)
        rest = prompt[idx:].strip()
        num = ""
        for ch in rest:
            if ch.isdigit() or (ch == "-" and not num):
                num += ch
            else:
                break
        return int(num)


class ExhaustiveGuesser(LLMInterface):
    """Tries each word in a fixed vocabulary once, in order — purely
    exploratory (no hypothesis attached), a valid if unintelligent
    strategy for word_lock.py."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.tried: set[str] = set()

    def propose(self, prompt: str, schema_hint: str) -> dict:
        for w in self.vocab:
            if w not in self.tried:
                self.tried.add(w)
                return {
                    "hypotheses": [],
                    "candidate_actions": [
                        {
                            "kind": "guess",
                            "params": {"word": w},
                            "rationale": "next untried word",
                            "predicted_outcome": "GGGG",
                            "tests_hypothesis": None,
                            "cost": 1.0,
                        }
                    ],
                }
        return {
            "hypotheses": [],
            "candidate_actions": [{"kind": "stop", "rationale": "exhausted vocab"}],
        }


class BadActorLLM(LLMInterface):
    """Always proposes an action kind that is NOT in the allowed list —
    used to prove the Brain filters it out instead of blindly executing."""

    def propose(self, prompt: str, schema_hint: str) -> dict:
        return {
            "hypotheses": [],
            "candidate_actions": [
                {
                    "kind": "definitely_not_allowed",
                    "params": {},
                    "rationale": "misbehaving on purpose",
                    "tests_hypothesis": None,
                }
            ],
        }


class VagueGuesser(LLMInterface):
    """Always ties its action to a hypothesis but gives IDENTICAL
    predicted_if_true / predicted_if_false — i.e. proposes fake
    'experiments' with no real distinguishing power. Used to prove the
    Information-Gain Engine scores these near zero and/or the Challenger
    flags them."""

    def __init__(self):
        self.n = 0

    def propose(self, prompt: str, schema_hint: str) -> dict:
        self.n += 1
        statement = "the answer involves number 50"
        hyps = [] if self.n > 1 else [{"statement": statement, "confidence": 0.5}]
        return {
            "hypotheses": hyps,
            "candidate_actions": [
                {
                    "kind": "guess",
                    "params": {"number": 50},
                    "rationale": "vague guess",
                    "predicted_if_true": "some result",
                    "predicted_if_false": "some result",  # identical - not a real experiment
                    "tests_hypothesis": statement,
                    "cost": 1.0,
                }
            ],
        }
