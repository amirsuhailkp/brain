"""
Memory Consolidation Engine (Phase 4).

Deliberately NOT called inside Brain.run() — consolidation is a periodic,
batch operation over accumulated Experience history (e.g. run nightly, or
every N episodes by whatever orchestrates the Brain), not a per-step or
even per-run cost. Mixing it into the hot loop would mean every single run
pays for reclustering the entire memory store, which doesn't scale and
isn't how the spec's "100 events -> 20 experiences -> 5 principles"
pipeline is supposed to work — each stage runs at its own cadence.

Clustering approach is deliberately simple and stated as a limitation:
tag-overlap (Jaccard similarity) instead of semantic/embedding similarity,
since this package has no embedding dependency yet. Two Experiences with
enough shared tags (action kind, hypothesis status, stop_reason, etc.) are
considered "about the same kind of thing" and can be merged into a
Principle if there are enough of them. This will misgroup some
lexically-different-but-semantically-similar experiences and over-group
some coincidentally-tagged ones — a real embedding-based version is a
reasonable, non-architectural upgrade noted in the README, not something
that needs a new engine.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import LLMInterface

from .models import Experience, Principle
from .similarity import SimilarityScorer

MIN_CLUSTER_SIZE = 3          # need at least this many similar experiences to form a principle
JACCARD_THRESHOLD = 0.5       # tag-overlap fraction required to count as "the same cluster"


def consolidate(
    experiences: list[Experience],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    scorer: SimilarityScorer | None = None,
    similarity_threshold: float = JACCARD_THRESHOLD,
    llm: "LLMInterface | None" = None,
) -> list[Principle]:
    """Pure function: takes accumulated Experience records, returns
    Principles for any cluster large enough to generalize from. Does not
    read or write any storage itself — caller decides where Experiences
    come from and where Principles get persisted (see PrincipleStore).

    `scorer` (Phase 11): None (the default) preserves the EXACT original
    behavior — literal tag-set Jaccard overlap, unchanged from Phase 4.
    Passing a SimilarityScorer (similarity.py) switches clustering to
    compare Experience summary+lesson TEXT instead of exact tags, which is
    what lets two differently-worded-but-same-tactic Experiences cluster
    together — something tag matching can never do since it requires the
    literal tag strings to match. Opt-in, not a default change, so every
    pre-Phase-11 call site (including the full existing test suite) sees
    identical output.

    `llm` (Phase 18c): None (the default) preserves the exact original
    synthesis path — deterministic, no LLM cost. When provided, clusters
    with no dominant lesson (agreement < 60%) get an LLM-assisted synthesis
    instead of the old raw-concatenation fallback. Opt-in; every pre-Phase-18c
    call site is unaffected."""
    if scorer is None:
        clusters = _cluster_by_tag_overlap(experiences)
    else:
        clusters = _cluster_by_similarity(experiences, scorer, similarity_threshold)

    principles: list[Principle] = []
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        principles.append(_synthesize_principle(cluster, llm=llm))
    return principles


def _cluster_by_similarity(
    experiences: list[Experience], scorer: SimilarityScorer, threshold: float
) -> list[list[Experience]]:
    """Same greedy single-pass approach as _cluster_by_tag_overlap (join
    the first cluster whose representative is similar enough, else start
    a new one) — only the similarity measure changes, from exact-tag
    Jaccard to scorer.score() over summary+lesson text, so wording
    differences between two Experiences describing the same underlying
    tactic no longer prevent them from clustering."""
    clusters: list[list[Experience]] = []
    for exp in experiences:
        text = f"{exp.summary} {exp.lesson}"
        placed = False
        for cluster in clusters:
            rep_text = f"{cluster[0].summary} {cluster[0].lesson}"
            if scorer.score(text, rep_text) >= threshold:
                cluster.append(exp)
                placed = True
                break
        if not placed:
            clusters.append([exp])
    return clusters


def _cluster_by_tag_overlap(experiences: list[Experience]) -> list[list[Experience]]:
    """Greedy single-pass clustering: for each experience, join the first
    existing cluster whose representative (first member) shares enough
    tags, else start a new cluster. O(n * clusters), fine at this scale;
    a real system with thousands of experiences would want something
    better (and probably embeddings, per the module docstring)."""
    clusters: list[list[Experience]] = []
    for exp in experiences:
        placed = False
        for cluster in clusters:
            if _jaccard(set(exp.tags), set(cluster[0].tags)) >= JACCARD_THRESHOLD:
                cluster.append(exp)
                placed = True
                break
        if not placed:
            clusters.append([exp])
    return clusters


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _synthesize_principle(
    cluster: list[Experience],
    llm: "LLMInterface | None" = None,
) -> Principle:
    """Deterministic synthesis with optional LLM-assisted summarization
    (Phase 18c).

    The original approach:
      - If >= 60% of the cluster agrees on one lesson verbatim → use it.
      - Otherwise → concatenate up to 3 raw lesson strings with " | ".

    The problem with the fallback: the concatenated string is rarely
    coherent enough to be useful as a planning prompt directive. It's
    long, internally inconsistent, and carries no synthesis — it's just
    three lessons glued together. The consolidation docstring itself
    called out "an LLM-assisted summarizer that produces smoother
    natural-language principles is a reasonable upgrade." Now we have the
    LLMInterface seam to do it.

    Wired narrowly, same scoping discipline as every prior opt-in scorer:
      - Only fires when there is no dominant lesson (agreement < 60%) AND
        the cluster has multiple distinct lessons — the case where the
        old concatenation approach was weakest.
      - A 60%+ agreeing cluster is already a clean principle; running an
        LLM over it would add cost with no quality gain.
      - A single-lesson cluster is trivially synthesized; same.
      - Fails gracefully (falls back to the original concatenation) on
        any LLM error or malformed output — a synthesis failure must
        never prevent a consolidation pass from completing.

    `llm` is None by default → behavior identical to every pre-Phase-18c
    call site."""
    lesson_counts: dict[str, int] = defaultdict(int)
    for exp in cluster:
        lesson_counts[exp.lesson] += 1
    ranked = sorted(lesson_counts.items(), key=lambda kv: kv[1], reverse=True)

    dominant_lesson, dominant_count = ranked[0]
    agreement = dominant_count / len(cluster)

    if agreement >= 0.6 or len(ranked) == 1:
        statement = dominant_lesson
    elif llm is not None:
        # Multiple distinct lessons — ask the LLM to synthesize a single
        # coherent principle from what the cluster collectively learned.
        distinct = [lesson for lesson, _ in ranked[:5]]
        prompt = (
            "You are summarizing a set of related lessons learned by an AI "
            "reasoning system across multiple tasks. Synthesize the following "
            "lessons into ONE clear, concise principle (one sentence, under 80 "
            "words) that captures what they have in common and what should be "
            "remembered for future tasks. Do not list them — distill them.\n\n"
            "Lessons:\n"
            + "\n".join(f"- {l}" for l in distinct)
            + '\n\nRespond ONLY with JSON: {"principle": "your one-sentence synthesis"}.'
        )
        try:
            raw = llm.propose(prompt, '{"principle": "..."}')
            statement = str(raw.get("principle", "")).strip()
            if not statement:
                raise ValueError("empty principle from LLM")
        except Exception:
            # Fail gracefully: same concatenation as before
            distinct_short = [lesson for lesson, _ in ranked[:3]]
            statement = "Multiple related lessons observed: " + " | ".join(distinct_short)
    else:
        distinct = [lesson for lesson, _ in ranked[:3]]
        statement = "Multiple related lessons observed: " + " | ".join(distinct)

    project_ids = {exp.project_id for exp in cluster}
    project_id = next(iter(project_ids)) if len(project_ids) == 1 else None

    return Principle(
        statement=statement,
        supporting_experience_ids=[exp.id for exp in cluster],
        confidence=round(agreement, 3),
        project_id=project_id,
    )


SUPERSESSION_OVERLAP_THRESHOLD = 0.5  # shared-evidence fraction to count as "the same generalization, refined"


class PrincipleStore:
    """Minimal JSONL persistence for consolidated Principles — separate
    file from Experience storage (memory.JsonlMemory) since Principles are
    a different unit produced at a different cadence, not just another
    tag on the same stream."""

    def __init__(self, path: str):
        import json as _json
        from pathlib import Path as _Path

        self._json = _json
        self.path = _Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def store(self, principle: Principle) -> None:
        from dataclasses import asdict

        with self.path.open("a") as f:
            f.write(self._json.dumps(asdict(principle)) + "\n")

    def all(self) -> list[Principle]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Principle(**self._json.loads(line)))
        return out

    def active(self) -> list[Principle]:
        """Only Principles still in force - Phase 4 shipped with no way
        to tell an outdated generalization apart from a still-current one
        once a later consolidation run refined it; this is that
        distinction. Old records stay in the file either way
        (Principle 13: inspectable) - this just filters the read side."""
        return [p for p in self.all() if p.superseded_by is None]

    def record_outcome(self, principle_id: str, success: bool) -> None:
        """Phase 9: the write side of Phase 8's retrieval loop. Until now a
        Principle's confidence was frozen forever at synthesis time (the
        agreement ratio of the Experiences that produced it) - nothing ever
        adjusted it after the Principle was actually put to use. That made
        Brain Memory a static lookup table, not something that gets more
        trustworthy (or less) as it proves itself across MORE projects than
        the ones that originally produced it.

        Called from core.py only for a Principle that seeded a Hypothesis
        (state.principle_seeds) once that Hypothesis reaches a terminal
        status (CONFIRMED or REJECTED) - i.e. only on a real, verified
        outcome (Verification Engine already gated that), never on a raw
        single observation. Reuses hypotheses.update_confidence so there is
        still exactly one place this arithmetic can be wrong, per that
        function's own docstring.

        Deliberately does NOT touch supporting_experience_ids or spawn a
        new Principle version - this is confidence drift on an existing
        record, not a new consolidation pass. superseded_by is untouched
        for the same reason: outcome feedback and "a later pass covered
        the same evidence" are different kinds of change."""
        from .hypotheses import update_confidence  # local import: avoids a module-load cycle
                                                     # (hypotheses.py -> verification.py/models.py
                                                     # only; nothing in that chain imports
                                                     # consolidation.py, so this is one-directional)

        principles = self.all()
        changed = False
        for p in principles:
            if p.id != principle_id:
                continue
            p.confidence = round(update_confidence(p.confidence, matched=success), 3)
            changed = True
            break
        if changed:
            self._rewrite(principles)

    def store_generation(self, new_principles: list[Principle]) -> None:
        """Persist a fresh batch from one consolidation pass, marking any
        EXISTING active Principle superseded if a new one covers
        substantially the same evidence (Jaccard overlap of
        supporting_experience_ids >= SUPERSESSION_OVERLAP_THRESHOLD, same
        project scope). This is the actual fix for the Phase 4 gap: two
        consolidation runs over growing Experience history used to
        produce two indistinguishable-looking Principles with no relation
        between them - now the later one explicitly retires the earlier
        one instead of leaving both silently "active"."""
        existing = self.all()
        changed = False

        for new_p in new_principles:
            for old_p in existing:
                if old_p.superseded_by is not None:
                    continue  # already retired by an even earlier pass
                if old_p.project_id != new_p.project_id:
                    continue  # don't cross-supersede across project scopes
                if old_p.id == new_p.id:
                    continue
                overlap = _jaccard(
                    set(old_p.supporting_experience_ids), set(new_p.supporting_experience_ids)
                )
                if overlap >= SUPERSESSION_OVERLAP_THRESHOLD:
                    old_p.superseded_by = new_p.id
                    changed = True

        if changed:
            self._rewrite(existing)  # only rewrite the file if something actually changed
        for new_p in new_principles:
            self.store(new_p)

    def _rewrite(self, principles: list[Principle]) -> None:
        from dataclasses import asdict

        with self.path.open("w") as f:
            for p in principles:
                f.write(self._json.dumps(asdict(p)) + "\n")
