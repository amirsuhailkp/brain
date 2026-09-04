import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.consolidation import consolidate, PrincipleStore
from brain.interfaces import LLMInterface
from brain.models import Experience
from brain.priors import PrincipleRetriever
from brain.similarity import LexicalOverlapScorer, TfidfCosineScorer, LLMSemanticScorer

# The running example from similarity.py's own docstring: same underlying
# tactic, described in completely different vocabulary. Any scorer that
# claims to "understand meaning" has to do something sensible with this
# exact pair, not just a synthetic unit-test fixture.
IDOR_PHRASING_A = "auth endpoint leaks user data via ID swap"
IDOR_PHRASING_B = "broken object-level authorization on numeric IDs"
UNRELATED = "coffee machine descaling schedule for the office kitchen"


# --------------------------------------------------------------- LexicalOverlapScorer


def test_lexical_scorer_scores_zero_for_no_shared_words():
    scorer = LexicalOverlapScorer()
    assert scorer.score(IDOR_PHRASING_A, IDOR_PHRASING_B) == 0.0


def test_lexical_scorer_scores_high_for_near_duplicate_text():
    scorer = LexicalOverlapScorer()
    assert scorer.score("target is >= 2", "target is >= 2") == 1.0


def test_lexical_score_tags_matches_original_priors_behavior():
    scorer = LexicalOverlapScorer()
    # replicates the exact case priors.py's original _tag_overlap covered
    assert scorer.score_tags(["idor", "auth"], "auth endpoints often leak via idor") == 1.0
    assert scorer.score_tags(["idor", "auth"], "rate limits and header spoofing") == 0.0


# --------------------------------------------------------------- TfidfCosineScorer


def test_tfidf_scorer_still_misses_pure_synonym_swap():
    """Documents the honest limitation stated in similarity.py: TF-IDF is
    a real improvement over Jaccard but is still lexical, so the flagship
    zero-shared-words example still scores near zero. This test exists so
    a future change that silently regresses TfidfCosineScorer into
    claiming semantic understanding it doesn't have gets caught, not to
    endorse the limitation."""
    scorer = TfidfCosineScorer()
    assert scorer.score(IDOR_PHRASING_A, IDOR_PHRASING_B) < 0.2


def test_tfidf_scorer_rewards_shared_distinctive_terms():
    scorer = TfidfCosineScorer()
    a = "SSRF vulnerability found in the webhook callback URL parameter"
    b = "SSRF vulnerability found in the image proxy URL parameter"
    unrelated_score = scorer.score(a, UNRELATED)
    related_score = scorer.score(a, b)
    assert related_score > unrelated_score


def test_tfidf_scorer_handles_empty_strings_without_crashing():
    scorer = TfidfCosineScorer()
    assert scorer.score("", "something") == 0.0
    assert scorer.score("", "") == 0.0


# --------------------------------------------------------------- LLMSemanticScorer


class FixedSimilarityLLM(LLMInterface):
    """Deterministic stand-in for a real model: recognizes the one
    semantic-equivalence pair this test suite cares about, scores
    everything else low — enough to prove LLMSemanticScorer plumbs the
    LLM's judgment through correctly without needing a live model call.

    Matches against the prompt's own "Statement A:"/"Statement B:" markers
    specifically (not a bare substring check) — the prompt template itself
    includes an illustrative IDOR example in its instructions, so a naive
    substring search would false-positive on that instructional text."""

    def propose(self, prompt: str, schema_hint: str) -> dict:
        target = f'Statement A: "{IDOR_PHRASING_A}"\nStatement B: "{IDOR_PHRASING_B}"'
        if target in prompt:
            return {"similarity": 0.95}
        return {"similarity": 0.05}


class MalformedLLM(LLMInterface):
    def propose(self, prompt: str, schema_hint: str) -> dict:
        return {"similarity": "not-a-number"}


def test_llm_semantic_scorer_catches_the_synonym_case_tfidf_misses():
    scorer = LLMSemanticScorer(FixedSimilarityLLM())
    assert scorer.score(IDOR_PHRASING_A, IDOR_PHRASING_B) == 0.95


def test_llm_semantic_scorer_low_for_unrelated_text():
    scorer = LLMSemanticScorer(FixedSimilarityLLM())
    assert scorer.score(IDOR_PHRASING_A, UNRELATED) == 0.05


def test_llm_semantic_scorer_degrades_safely_on_malformed_response():
    scorer = LLMSemanticScorer(MalformedLLM())
    assert scorer.score("a", "b") == 0.0  # never raises, never crashes clustering/retrieval


# --------------------------------------------------------------- consolidation.py wiring


def _exp(summary: str, lesson: str) -> Experience:
    return Experience(summary=summary, lesson=lesson, importance=0.9)


def test_consolidate_default_behavior_unchanged_without_a_scorer():
    """The exact Phase-4 tag-Jaccard path must still run when no scorer is
    passed - this is the backward-compatibility guarantee Phase 11 makes."""
    experiences = [
        Experience(summary="s1", lesson="l1", importance=0.9, tags=["idor", "auth"]),
        Experience(summary="s2", lesson="l2", importance=0.9, tags=["idor", "auth"]),
        Experience(summary="s3", lesson="l3", importance=0.9, tags=["idor", "auth"]),
    ]
    principles = consolidate(experiences)  # no scorer= argument at all
    assert len(principles) == 1


def test_consolidate_with_semantic_scorer_clusters_differently_worded_experiences():
    """The actual point of Phase 11: three Experiences describing the same
    tactic in three different wordings, with NO shared tags at all, still
    consolidate into one Principle once a semantic scorer is supplied -
    something the tag-Jaccard default can never do by construction."""
    experiences = [
        _exp("Found IDOR via ID swap", IDOR_PHRASING_A),
        _exp("Confirmed BOLA on numeric id", IDOR_PHRASING_B),
        _exp("Object-level auth bypass via sequential ids", "sequential id enumeration bypasses object-level authorization checks"),
    ]

    class AlwaysSameTacticLLM(LLMInterface):
        def propose(self, prompt, schema_hint):
            return {"similarity": 0.9}  # stand-in: "yes, these three are all the same tactic"

    scorer = LLMSemanticScorer(AlwaysSameTacticLLM())
    principles = consolidate(experiences, min_cluster_size=3, scorer=scorer)
    assert len(principles) == 1
    assert len(principles[0].supporting_experience_ids) == 3


# --------------------------------------------------------------- priors.py wiring


def test_retriever_default_scorer_matches_old_priors_behavior():
    from brain.models import Principle

    d = tempfile.mkdtemp()
    store = PrincipleStore(f"{d}/p.jsonl")
    store.store(Principle(statement="auth endpoints often leak via idor", confidence=0.6))
    retriever = PrincipleRetriever(store)  # no scorer= — must behave exactly as before
    assert len(retriever.relevant(["idor", "auth"])) == 1


def test_retriever_with_llm_scorer_finds_semantically_equivalent_principle():
    """Same scenario as the consolidation test above, but for RETRIEVAL:
    a project tags its situation as 'idor'/'auth', but the only stored
    Principle is worded as 'broken object-level authorization' - the
    lexical default finds nothing; the LLM-backed scorer does."""
    from brain.models import Principle

    d = tempfile.mkdtemp()
    store = PrincipleStore(f"{d}/p.jsonl")
    stored = Principle(statement=IDOR_PHRASING_B, confidence=0.7)
    store.store(stored)

    lexical_retriever = PrincipleRetriever(store)
    assert lexical_retriever.relevant(["idor", "auth"]) == []  # can't bridge the wording gap

    class RecognizesIdorLLM(LLMInterface):
        def propose(self, prompt, schema_hint):
            return {"similarity": 0.9 if "idor" in prompt.lower() else 0.0}

    semantic_retriever = PrincipleRetriever(store, scorer=LLMSemanticScorer(RecognizesIdorLLM()))
    results = semantic_retriever.relevant(["idor", "auth"])
    assert len(results) == 1
    assert results[0].id == stored.id
