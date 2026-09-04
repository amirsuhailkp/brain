"""
Similarity Scoring (Phase 11).

Every place Brain compares two pieces of text for "are these about the
same thing" — consolidation.py clustering Experiences into Principles,
priors.py matching a project's situation tags against Principle text —
did this via literal word/tag overlap (Jaccard, substring match). That's
honest and fast, but it's blind to meaning: "auth endpoint leaks user
data via ID swap" and "broken object-level authorization on numeric IDs"
describe the identical tactic and share almost no words, so the old
matching would silently treat them as unrelated. Since the entire premise
of Brain Memory is that a tactic learned on one wired project should
transfer to a DIFFERENTLY-WORDED but structurally similar situation on
another, that blindness worked against the core idea, not just a minor
edge case.

This module makes "how similar are these two strings" a pluggable seam
(SimilarityScorer), with three implementations, cheapest-first:

  LexicalOverlapScorer  - the exact word/tag-overlap math consolidation.py
                          and priors.py already had, extracted here
                          unchanged. Zero dependencies, deterministic,
                          the DEFAULT everywhere — every existing call
                          site keeps its exact old behavior unless it
                          explicitly opts into one of the scorers below.

  TfidfCosineScorer     - a real statistical upgrade with zero new
                          runtime dependencies beyond scikit-learn (pure
                          local computation, no model download, no
                          network call). Catches partial phrase overlap
                          and term weighting that Jaccard misses, but is
                          still fundamentally lexical — "IDOR" and
                          "broken object-level authorization" still share
                          no terms, so this alone doesn't close the gap
                          in the docstring above. It's a genuine
                          improvement over pure Jaccard, not a semantic
                          one.

  LLMSemanticScorer     - the actual meaning-based upgrade. Reuses
                          Brain's EXISTING LLMInterface seam (the same
                          one PlanningEngine already calls) to ask
                          whichever model a project already has wired in
                          — local Ollama for agent65, or anything else a
                          future project brings — whether two statements
                          describe the same underlying tactic/pattern.
                          No new dependency, no new network requirement
                          beyond what the project already pays for its
                          own reasoning calls, and it's the only one of
                          the three that actually understands synonyms,
                          paraphrase, and cross-domain analogy rather than
                          matching surface text.

Deliberately NOT built: a bundled sentence-embedding model. That would
mean downloading pretrained weights at runtime, a new heavy dependency
(torch or similar), and a fixed vocabulary baked in at model-training
time that has no way to improve as Brain accumulates more experience.
LLMSemanticScorer gets the same "understands meaning" benefit through a
seam Brain already has, using whatever model strength a project already
chose for its own reasoning loop — consistent with Phase 5's model-
escalation design (two PlanningEngine instances, not a router
abstraction) rather than introducing a fourth kind of model dependency.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .interfaces import LLMInterface

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class SimilarityScorer(ABC):
    """0.0 (unrelated) .. 1.0 (same thing). Symmetric: score(a, b) ==
    score(b, a) is expected of every implementation, though not enforced
    here — callers that need it strictly should assert it in their own
    tests, same as any other implementation-detail contract in this
    package."""

    @abstractmethod
    def score(self, a: str, b: str) -> float:
        raise NotImplementedError

    def score_tags(self, query_tags: list[str], target_text: str) -> float:
        """Default: treat the tag list as a space-joined string and defer
        to score(). Scorers for which tag-vs-prose comparison should work
        differently from prose-vs-prose (none currently do) can override
        this directly."""
        if not query_tags:
            return 0.0
        return self.score(" ".join(query_tags), target_text)


class LexicalOverlapScorer(SimilarityScorer):
    """The original Jaccard/word-overlap math from consolidation.py and
    priors.py, extracted unchanged so both modules can default to this
    and see ZERO behavior change from before Phase 11."""

    def score(self, a: str, b: str) -> float:
        wa, wb = _words(a), _words(b)
        union = wa | wb
        if not union:
            return 0.0
        return len(wa & wb) / len(union)

    def score_tags(self, query_tags: list[str], target_text: str) -> float:
        if not query_tags:
            return 0.0
        target_words = _words(target_text)
        hits = sum(1 for t in query_tags if t.lower() in target_words)
        return hits / len(query_tags)


class TfidfCosineScorer(SimilarityScorer):
    """Statistical upgrade over pure word-overlap: term-frequency weighted
    cosine similarity. Still lexical (no synonym understanding), but
    handles partial/weighted overlap better than Jaccard's all-or-nothing
    per-word counting - e.g. two long statements sharing one distinctive
    rare term score higher than sharing one common filler word, which
    Jaccard can't distinguish.

    Fit on-the-fly per comparison (a 2-document corpus) rather than
    holding a persistent fitted vectorizer - simplest correct thing for
    this package's call volume (comparing a handful of Principles per
    step, not a bulk corpus job); a persistent-corpus version is a
    reasonable future upgrade if this becomes a hot path, not required
    now."""

    def __init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as e:
            raise ImportError(
                "TfidfCosineScorer requires scikit-learn: "
                "pip install scikit-learn --break-system-packages"
            ) from e
        self._vectorizer_cls = TfidfVectorizer
        self._cosine_similarity = cosine_similarity

    def score(self, a: str, b: str) -> float:
        a, b = a.strip(), b.strip()
        if not a or not b:
            return 0.0
        try:
            vectorizer = self._vectorizer_cls()
            matrix = vectorizer.fit_transform([a, b])
        except ValueError:
            return 0.0  # e.g. both strings are pure stop-words/empty after tokenizing
        sim = self._cosine_similarity(matrix[0], matrix[1])[0][0]
        return max(0.0, min(1.0, float(sim)))


class LLMSemanticScorer(SimilarityScorer):
    """Asks the project's own wired LLM whether two statements describe
    the same underlying tactic/pattern. This is the scorer that actually
    understands paraphrase and cross-domain analogy - reuses
    LLMInterface.propose(), the exact seam PlanningEngine already calls,
    so no new kind of model dependency is introduced.

    Costs a real model call per comparison - deliberately NOT the default
    anywhere (see LexicalOverlapScorer), and callers doing many pairwise
    comparisons (e.g. consolidation's O(n * clusters) clustering loop)
    should budget for that, same as any other LLM-call-per-item pattern
    already in this codebase (e.g. PlanningEngine's one call per step)."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def score(self, a: str, b: str) -> float:
        a, b = a.strip(), b.strip()
        if not a or not b:
            return 0.0
        prompt = (
            "Two statements from an experience-tracking system. Judge whether they "
            "describe the SAME underlying tactic, pattern, or root cause - even if "
            "worded completely differently, from a different domain, or using different "
            "terminology (e.g. 'auth endpoint leaks data via ID swap' and 'broken "
            "object-level authorization on numeric IDs' are the SAME tactic).\n\n"
            f'Statement A: "{a}"\n'
            f'Statement B: "{b}"\n\n'
            "Respond ONLY with JSON: {\"similarity\": 0.0-1.0} "
            "(1.0 = same tactic, 0.0 = unrelated)."
        )
        try:
            raw = self.llm.propose(prompt, '{"similarity": 0.0-1.0}')
            value = raw.get("similarity")
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0  # a failed/malformed judgment must never crash clustering or retrieval;
                        # treating it as "no evidence of similarity" is the safe default, same
                        # spirit as PlanningEngine.reason_and_plan's own except-and-degrade
