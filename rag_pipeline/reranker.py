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

import json

import requests

from pipeline_types import RetrievalResult

logger = logging.getLogger(__name__)


LLM_RERANK_PROMPT = """\
Bạn là trợ lý pháp lý người Việt Nam. Nhiệm vụ của bạn là chọn ĐÚNG MỘT đoạn văn bản pháp luật trả lời trực tiếp và chính xác nhất cho câu hỏi của người dùng.

### Câu hỏi:
{query}

### Các ứng viên:
{numbered_docs}

Chỉ trả về một đối tượng JSON hợp lệ với định dạng sau, không thêm bất kỳ văn bản nào khác:
{{"best": <SỐ_THỨ_TỰ_CỦA_ỨNG_VIÊN_ĐƯỢC_CHỌN>}}

Ví dụ: {{"best": 2}}"""


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
        conditional_enabled: bool = False,
        conditional_strategy: str = "top1_top2_gap",
        conditional_top1_top2_gap: float = 0.03,
        conditional_top1_top10_gap: float = 0.08,
        conditional_min_candidates: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.max_candidates = max_candidates
        self.top_n = top_n
        self.timeout_sec = timeout_sec
        self.blend_weight = blend_weight
        self.conditional_enabled = conditional_enabled
        self.conditional_strategy = conditional_strategy
        self.conditional_top1_top2_gap = conditional_top1_top2_gap
        self.conditional_top1_top10_gap = conditional_top1_top10_gap
        self.conditional_min_candidates = conditional_min_candidates
        self.calls_attempted = 0
        self.calls_skipped = 0
        self.calls_failed = 0

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

        if not self.should_rerank(candidates):
            self.calls_skipped += 1
            return candidates[:top_k]

        pool = candidates[: self.max_candidates]
        documents = [r.content for r in pool]
        n_docs = len(documents)

        effective_top_n = min(n_docs, max(top_k, self.top_n))

        try:
            self.calls_attempted += 1
            api_scores = self._call_api(query, documents, effective_top_n, n_docs)
        except Exception as e:
            self.calls_failed += 1
            logger.warning("BGE reranker API call failed: %s", e)
            return candidates[:top_k]

        if api_scores is None:
            self.calls_failed += 1
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

        # Append candidates beyond the rerank pool
        for r in candidates[self.max_candidates :]:
            if r.chunk_id not in seen_ids:
                reranked.append(r)
                seen_ids.add(r.chunk_id)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    def should_rerank(self, candidates: list[RetrievalResult]) -> bool:
        """Return True when the candidate scores look uncertain enough to rerank."""
        if not self.conditional_enabled:
            return True

        if len(candidates) < self.conditional_min_candidates:
            return False

        pool = candidates[: self.max_candidates]
        scores = sorted((r.score for r in pool), reverse=True)
        if len(scores) < self.conditional_min_candidates:
            return False

        if self.conditional_strategy == "top1_top2_gap":
            return (scores[0] - scores[1]) < self.conditional_top1_top2_gap

        if self.conditional_strategy == "top1_top10_gap":
            if len(scores) < 10:
                return True
            return (scores[0] - scores[9]) < self.conditional_top1_top10_gap

        logger.warning(
            "Unknown conditional reranker strategy '%s'; reranking by default",
            self.conditional_strategy,
        )
        return True

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


class LLMReranker:
    """Re-rank retrieval results using an LLM API (Gemma).

    Sends the top-N candidates to the LLM and asks it to select the single
    best candidate. The chosen candidate is moved to rank 1; the rest retain
    their original order within the pool.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gemma-4-31B-it",
        endpoint: str = "/v1/chat/completions",
        max_candidates: int = 3,
        top_n: int = 3,
        temperature: float = 0.0,
        max_output_tokens: int = 48,
        max_candidate_chars: int = 1000,
        blend_weight: float = 1.0,
        timeout_sec: int = 60,
        conditional_enabled: bool = False,
        conditional_strategy: str = "top1_top2_gap",
        conditional_top1_top2_gap: float = 0.03,
        conditional_top1_top10_gap: float = 0.08,
        conditional_min_candidates: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.max_candidates = max_candidates
        self.top_n = top_n
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_candidate_chars = max_candidate_chars
        self.blend_weight = blend_weight
        self.timeout_sec = timeout_sec
        self.conditional_enabled = conditional_enabled
        self.conditional_strategy = conditional_strategy
        self.conditional_top1_top2_gap = conditional_top1_top2_gap
        self.conditional_top1_top10_gap = conditional_top1_top10_gap
        self.conditional_min_candidates = conditional_min_candidates
        self.calls_attempted = 0
        self.calls_skipped = 0
        self.calls_failed = 0

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Re-rank candidates using LLM selection.

        Asks the LLM to pick the single best candidate from the pool.
        The chosen candidate is moved to rank 1; remaining candidates
        keep their original relative order.

        Args:
            query: The original user query.
            candidates: Chunks from vector search + BM25 fusion.
            top_k: Number of results to keep after reranking.

        Returns:
            Re-ranked list of RetrievalResult.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates

        if not self.should_rerank(candidates):
            self.calls_skipped += 1
            return candidates[:top_k]

        pool = candidates[: self.max_candidates]
        n_pool = len(pool)

        if n_pool < 2:
            return candidates[:top_k]

        documents = [r.content[: self.max_candidate_chars] for r in pool]

        try:
            self.calls_attempted += 1
            best_idx = self._call_llm(query, documents, n_pool)
        except Exception as e:
            self.calls_failed += 1
            logger.warning("LLM reranker call failed: %s", e)
            return candidates[:top_k]

        if best_idx is None or best_idx < 0 or best_idx >= n_pool:
            self.calls_failed += 1
            return candidates[:top_k]

        # Move chosen candidate to rank 1, keep rest in pool order
        chosen = pool[best_idx]
        reranked: list[RetrievalResult] = [chosen]
        seen_ids: set[str] = {chosen.chunk_id}

        for r in pool:
            if r.chunk_id not in seen_ids:
                reranked.append(r)
                seen_ids.add(r.chunk_id)

        # Append candidates beyond the pool
        for r in candidates[self.max_candidates :]:
            if r.chunk_id not in seen_ids:
                reranked.append(r)
                seen_ids.add(r.chunk_id)

        return reranked[:top_k]

    def should_rerank(self, candidates: list[RetrievalResult]) -> bool:
        """Return True when the candidate scores look uncertain enough to rerank."""
        if not self.conditional_enabled:
            return True

        if len(candidates) < self.conditional_min_candidates:
            return False

        pool = candidates[: self.max_candidates]
        scores = sorted((r.score for r in pool), reverse=True)
        if len(scores) < self.conditional_min_candidates:
            return False

        if self.conditional_strategy == "top1_top2_gap":
            return (scores[0] - scores[1]) < self.conditional_top1_top2_gap

        if self.conditional_strategy == "top1_top10_gap":
            if len(scores) < 10:
                return True
            return (scores[0] - scores[9]) < self.conditional_top1_top10_gap

        logger.warning(
            "Unknown conditional reranker strategy '%s'; reranking by default",
            self.conditional_strategy,
        )
        return True

    def _call_llm(
        self,
        query: str,
        documents: list[str],
        n_docs: int,
    ) -> int | None:
        """Call the LLM chat API and return the index of the best candidate (0-based)."""
        numbered = "\n".join(
            f"[{i + 1}] {doc}" for i, doc in enumerate(documents)
        )
        prompt = LLM_RERANK_PROMPT.format(query=query, numbered_docs=numbered)

        url = f"{self.base_url}{self.endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()

        raw = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not raw:
            logger.warning("LLM reranker returned empty content")
            return None

        # Try to extract JSON from the response (handle potential markdown fences)
        if "```" in raw:
            raw = raw.split("```")[1] if raw.count("```") >= 2 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM reranker returned invalid JSON: %s", raw[:200])
            return None

        if not isinstance(parsed, dict):
            logger.warning("LLM reranker returned non-dict JSON: %s", parsed)
            return None

        best = parsed.get("best")
        if best is None:
            logger.warning("LLM reranker JSON missing 'best' key: %s", parsed)
            return None

        try:
            idx = int(best) - 1  # Convert 1-based to 0-based
        except (ValueError, TypeError):
            logger.warning("LLM reranker 'best' is not an integer: %s", best)
            return None

        if idx < 0 or idx >= n_docs:
            logger.warning(
                "LLM reranker returned out-of-range index %d (pool size %d)", idx, n_docs
            )
            return None

        return idx
