"""Context Builder and Reranker — reassesses chunk relevance and selects final context.

Implements Section 11 of Pipeline.md:

  1. Merge chunks from different sources (vector search + graph expansion).
  2. Reranker reassesses relevance of each chunk to the original query.
  3. Select highest-scoring chunks for the final context.

Strategy:
  - API-based BGE reranker via company endpoint.
  - Falls back to hybrid scores if the API call fails.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from pipeline_types import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Re-rank retrieval results using the company BGE reranker API.

    Usage::

        reranker = Reranker(base_url="https://...", api_key="sk-...")
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "bge-reranker-v2-m3",
        endpoint: str = "/v1/rerank",
        max_candidates: int = 30,
        top_n: int = 30,
        timeout_sec: int = 60,
        blend_weight: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.max_candidates = max_candidates
        self.top_n = top_n
        self.timeout_sec = timeout_sec
        self.blend_weight = blend_weight

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Re-rank candidates using the BGE reranker API.

        Args:
            query: The original user query.
            candidates: Chunks from vector search + BM25 fusion.
            top_k: Number of results to keep after reranking.

        Returns:
            Re-ranked list of RetrievalResult, highest combined score first.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates

        pool = candidates[: self.max_candidates]
        documents = [r.content for r in pool]
        n_docs = len(documents)

        effective_top_n = min(n_docs, max(top_k, self.top_n))

        try:
            api_scores = self._call_api(query, documents, effective_top_n, n_docs)
        except Exception as e:
            logger.warning("BGE reranker API call failed: %s", e)
            return candidates[:top_k]

        if api_scores is None:
            return candidates[:top_k]

        original_scores = {r.chunk_id: r.score for r in pool}
        reranked: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for idx, api_score in api_scores:
            r = pool[idx]
            seen_ids.add(r.chunk_id)
            orig = original_scores.get(r.chunk_id, 0.0)
            blended = self.blend_weight * api_score + (1 - self.blend_weight) * orig
            reranked.append(RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                raw_content=r.raw_content,
                article_id=r.article_id,
                unit_type=r.unit_type,
                order=r.order,
                hierarchy_path=r.hierarchy_path,
                score=blended,
                source=r.source,
                metadata=r.metadata,
            ))

        for r in pool:
            if r.chunk_id not in seen_ids:
                orig = original_scores.get(r.chunk_id, 0.0)
                blended = (1 - self.blend_weight) * orig
                reranked.append(RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    raw_content=r.raw_content,
                    article_id=r.article_id,
                    unit_type=r.unit_type,
                    order=r.order,
                    hierarchy_path=r.hierarchy_path,
                    score=blended,
                    source=r.source,
                    metadata=r.metadata,
                ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    def _call_api(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        n_docs: int = 0,
    ) -> Optional[list[tuple[int, float]]]:
        """Call the BGE reranker API and return list of (index, score)."""
        url = f"{self.base_url}{self.endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()

        if "results" in data:
            results = data["results"]
            if results and isinstance(results[0], dict):
                parsed = [
                    (r["index"], r.get("relevance_score", r.get("score", 0.0)))
                    for r in results
                ]
        elif "scores" in data:
            parsed = [(i, s) for i, s in enumerate(data["scores"])]
        elif "data" in data:
            parsed = [(r["index"], r["score"]) for r in data["data"]]
        else:
            logger.warning("Unexpected reranker response shape: %s", data)
            return None

        valid = [(idx, score) for idx, score in parsed if 0 <= idx < n_docs]
        if len(valid) < len(parsed):
            logger.warning(
                "%d/%d reranker results had out-of-range indices (pool size %d)",
                len(parsed) - len(valid), len(parsed), n_docs,
            )
        return valid
