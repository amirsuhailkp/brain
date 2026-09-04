import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.hypotheses import HypothesisEngine, MERGE_SIMILARITY_THRESHOLD
from brain.interfaces import LLMInterface
from brain.models import Goal, HypothesisStatus, WorkingState
from brain.similarity import LLMSemanticScorer

IDOR_A = "auth endpoint leaks user data via ID swap"
IDOR_B = "broken object-level authorization on numeric IDs"


class FixedScoreLLM(LLMInterface):
    """Scores the IDOR_A/IDOR_B pair (in either order) as a semantic
    match, everything else as unrelated — deterministic stand-in so these
    tests don't depend on a live model."""

    def propose(self, prompt, schema_hint):
        if IDOR_A in prompt and IDOR_B in prompt:
            return {"similarity": 0.9}
        return {"similarity": 0.05}


def _state() -> WorkingState:
    return WorkingState(goal=Goal(description="test"))


# --------------------------------------------------------------- backward compatibility


def test_integrate_new_without_scorer_keeps_original_exact_match_only_behavior():
    """The Phase-2 guarantee: two DIFFERENTLY worded proposals both get
    added as separate hypotheses when no scorer is passed — this is what
    every pre-Phase-12 caller (and the full existing test suite) already
    depends on."""
    state = _state()
    engine = HypothesisEngine()
    engine.integrate_new(state, [{"statement": IDOR_A, "confidence": 0.5}])
    engine.integrate_new(state, [{"statement": IDOR_B, "confidence": 0.5}])
    assert len(state.hypotheses) == 2
    assert state.hypothesis_merges == {}


def test_integrate_new_exact_duplicate_still_skipped_without_scorer():
    state = _state()
    engine = HypothesisEngine()
    engine.integrate_new(state, [{"statement": "target is >= 2", "confidence": 0.5}])
    engine.integrate_new(state, [{"statement": "Target is >= 2", "confidence": 0.7}])  # case-only diff
    assert len(state.hypotheses) == 1


# --------------------------------------------------------------- semantic merging (Phase 12)


def test_semantically_equivalent_proposal_merges_instead_of_duplicating():
    state = _state()
    engine = HypothesisEngine()
    scorer = LLMSemanticScorer(FixedScoreLLM())

    engine.integrate_new(state, [{"statement": IDOR_A, "confidence": 0.5}], scorer=scorer)
    engine.integrate_new(state, [{"statement": IDOR_B, "confidence": 0.7}], scorer=scorer)

    assert len(state.hypotheses) == 1  # did NOT create a rival hypothesis
    assert state.hypothesis_merges[IDOR_B] == state.hypotheses[0].id


def test_merge_averages_confidence_while_still_an_untested_prior():
    state = _state()
    engine = HypothesisEngine()
    scorer = LLMSemanticScorer(FixedScoreLLM())

    engine.integrate_new(state, [{"statement": IDOR_A, "confidence": 0.5}], scorer=scorer)
    engine.integrate_new(state, [{"statement": IDOR_B, "confidence": 0.7}], scorer=scorer)

    merged = state.hypotheses[0]
    assert merged.confidence == 0.6  # (0.5 + 0.7) / 2 — still untested, so the average is fair game
    assert merged.confidence_history == [0.5, 0.6]


def test_merge_does_not_dilute_confidence_once_real_evidence_exists():
    """Once a hypothesis has been touched by an actual observation
    (confidence_history length > 1), a later duplicate PROPOSAL — which
    is just another guess, not new evidence — must not move its
    confidence. Evidence-tested confidence is authoritative."""
    state = _state()
    engine = HypothesisEngine()
    scorer = LLMSemanticScorer(FixedScoreLLM())

    engine.integrate_new(state, [{"statement": IDOR_A, "confidence": 0.5}], scorer=scorer)
    hyp = state.hypotheses[0]
    hyp.confidence = 0.85  # simulate a real observation having already moved it
    hyp.confidence_history.append(0.85)

    engine.integrate_new(state, [{"statement": IDOR_B, "confidence": 0.1}], scorer=scorer)

    assert len(state.hypotheses) == 1  # still merged, not duplicated
    assert hyp.confidence == 0.85  # untouched by the low-confidence duplicate proposal
    assert state.hypothesis_merges[IDOR_B] == hyp.id


def test_merge_never_targets_a_confirmed_or_rejected_hypothesis():
    """Merging into a settled hypothesis would let a fresh, untested
    proposal reopen or reinforce a verdict through the back door — the
    same confirmation-bias pattern challenger.py already guards against
    on the decision side."""
    state = _state()
    engine = HypothesisEngine()
    scorer = LLMSemanticScorer(FixedScoreLLM())

    engine.integrate_new(state, [{"statement": IDOR_A, "confidence": 0.9}], scorer=scorer)
    settled = state.hypotheses[0]
    settled.status = HypothesisStatus.CONFIRMED

    engine.integrate_new(state, [{"statement": IDOR_B, "confidence": 0.5}], scorer=scorer)

    assert len(state.hypotheses) == 2  # new one added, NOT merged into the confirmed one
    assert IDOR_B not in state.hypothesis_merges


def test_below_threshold_similarity_does_not_merge():
    class LowScoreLLM(LLMInterface):
        def propose(self, prompt, schema_hint):
            return {"similarity": MERGE_SIMILARITY_THRESHOLD - 0.1}

    state = _state()
    engine = HypothesisEngine()
    scorer = LLMSemanticScorer(LowScoreLLM())

    engine.integrate_new(state, [{"statement": "statement one", "confidence": 0.5}], scorer=scorer)
    engine.integrate_new(state, [{"statement": "statement two", "confidence": 0.5}], scorer=scorer)

    assert len(state.hypotheses) == 2  # not similar enough to merge
