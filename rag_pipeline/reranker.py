"""Context Builder and Reranker — reassesses chunk relevance and selects final context.

Implements Section 11 of Pipeline.md:

  1. Merge chunks from different sources (vector search + graph expansion).
  2. Reranker reassesses relevance of each chunk to the original query.
  3. Select highest-scoring, most diverse chunks for the final context.

Strategy:
  - Primary: MMR (Maximal Marginal Relevance) — balances relevance with diversity.
  - Keyword boost: TF-like overlap between query tokens and chunk content.
  - Optional: Cross-encoder reranking when a suitable model is available.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

import numpy as np

from pipeline_types import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer for Vietnamese keyword extraction
# ---------------------------------------------------------------------------

# Simple word tokenizer: split on whitespace + punctuation, keep words >= 2 chars.
_WORD_RE = re.compile(r"[a-zà-ỹđ]+", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    """Extract lowercase word tokens from Vietnamese text."""
    return {m.group().lower() for m in _WORD_RE.finditer(text) if len(m.group()) >= 2}


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Re-rank retrieval results using MMR diversity + keyword boosting.

    Usage::

        reranker = Reranker(lambda_mmr=0.7, keyword_boost=0.15)
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        lambda_mmr: float = 0.7,
        keyword_boost: float = 0.15,
        diversity_weight: float = 0.15,
    ) -> None:
        """
        Args:
            lambda_mmr: Relevance weight in MMR (1.0 = pure relevance, 0.0 = pure diversity).
            keyword_boost: Weight for keyword-overlap score.
            diversity_weight: Weight for diversity penalty in MMR.
        """
        self.lambda_mmr = lambda_mmr
        self.keyword_boost = keyword_boost
        self.diversity_weight = diversity_weight

    # -- public API ----------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Re-rank candidates using MMR + keyword boosting.

        Args:
            query: The original user query.
            candidates: Chunks from vector search + graph expansion.
            top_k: Number of results to keep after reranking.

        Returns:
            Re-ranked list of RetrievalResult, highest combined score first.
        """
        if not candidates:
            return []

        n = len(candidates)
        if n <= 1:
            return candidates

        # Step 1 — Compute keyword-boosted relevance scores
        query_tokens = _tokenize(query)
        keyword_scores = np.array([
            self._keyword_overlap_score(query_tokens, r.content)
            for r in candidates
        ])
        base_scores = np.array([r.score for r in candidates])

        # Combined relevance: base score + keyword boost
        relevance = base_scores + self.keyword_boost * keyword_scores

        # Step 2 — Build pairwise similarity matrix for diversity
        # Use keyword overlap as a fast proxy for content similarity
        all_tokens = [_tokenize(r.content) for r in candidates]
        similarity = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._jaccard(all_tokens[i], all_tokens[j])
                similarity[i, j] = sim
                similarity[j, i] = sim

        # Step 3 — MMR greedy selection
        selected_indices: list[int] = []
        remaining = set(range(n))

        # First pick: highest combined relevance
        first = int(np.argmax(relevance))
        selected_indices.append(first)
        remaining.remove(first)

        # Subsequent picks: MMR
        while remaining and len(selected_indices) < top_k:
            best_idx = -1
            best_score = -1.0
            for idx in remaining:
                # Diversity penalty: max similarity to any already-selected chunk
                max_sim = max(
                    similarity[idx, sel] for sel in selected_indices
                ) if selected_indices else 0.0

                mmr_score = (
                    self.lambda_mmr * relevance[idx]
                    - self.diversity_weight * max_sim
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        # Step 4 — Build result with updated scores
        reranked: list[RetrievalResult] = []
        for rank, idx in enumerate(selected_indices):
            r = candidates[idx]
            # Recompute final score combining all signals
            final_score = (
                base_scores[idx]
                + self.keyword_boost * keyword_scores[idx]
                - self.diversity_weight * (rank * 0.02)  # small rank penalty
            )
            final_score = max(0.0, min(final_score, 1.0))
            reranked.append(RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                raw_content=r.raw_content,
                article_id=r.article_id,
                unit_type=r.unit_type,
                order=r.order,
                hierarchy_path=r.hierarchy_path,
                score=final_score,
                source=r.source,
                metadata=r.metadata,
            ))

        return reranked

    # -- scoring helpers -----------------------------------------------------

    @staticmethod
    def _keyword_overlap_score(query_tokens: set[str], content: str) -> float:
        """Compute Jaccard-like keyword overlap between query and chunk."""
        content_tokens = _tokenize(content)
        if not query_tokens or not content_tokens:
            return 0.0
        intersection = query_tokens & content_tokens
        return len(intersection) / len(query_tokens)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
