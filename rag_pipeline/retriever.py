"""Hybrid retrieval combining vector search, BM25 fusion, and metadata filtering.

Implements Sections 10 & 11 of Pipeline.md:

  1. Vector Search — semantic similarity via embeddings.
  2. BM25 Fusion — sparse term matching blended with dense scores.
  3. Metadata Filtering — filter/boost by status, effectiveDate, topic, unitType.
  4. Reranking — optional BGE or LLM reranker.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from bm25_retriever import BM25Retriever
from embedder import Embedder
from pipeline_types import RetrievalResult
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Hybrid retriever for Vietnamese legal chunks.

    Combines dense vector search + BM25 sparse retrieval with
    metadata-based filtering/boosting.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        *,

        top_k: int = 10,
        candidate_pool_size: int = 50,
        similarity_threshold: float = 0.3,
        enable_metadata_filtering: bool = True,
        vector_weight: float = 0.7,
        metadata_boost: float = 0.2,
        prefer_active: bool = True,
        topic_boost: bool = True,
        reranker: "Reranker | LLMReranker | None" = None,
        bm25_retriever: BM25Retriever | None = None,
        bm25_weight: float = 0.5,
        fusion_method: str = "weighted",
        rrf_k: int = 60,
        rrf_vector_weight: float = 1.0,
        rrf_bm25_weight: float = 1.0,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.similarity_threshold = similarity_threshold
        self.enable_metadata_filtering = enable_metadata_filtering
        self.vector_weight = vector_weight
        self.metadata_boost = metadata_boost
        self.prefer_active = prefer_active
        self.topic_boost = topic_boost
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.rrf_vector_weight = rrf_vector_weight
        self.rrf_bm25_weight = rrf_bm25_weight

    # -- public API ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_topic_id: Optional[str] = None,
        query_embedding: "np.ndarray | None" = None,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval for a natural-language query.

        Args:
            query: User query in Vietnamese (raw or normalized).
            top_k: Override the default top-k.
            filter_topic_id: If provided, boost results from this topic.
            query_embedding: Pre-computed embedding vector. If provided,
                skips embedder — used for GPU-offloaded benchmarks.

        Returns:
            Ranked list of RetrievalResult, highest score first.
        """
        k = top_k or self.top_k

        # Stage 1 — Embed and vector search
        if query_embedding is not None:
            q_emb = np.asarray(query_embedding, dtype=np.float32).ravel()
        else:
            q_emb = self.embedder.embed_single(query)
        vector_results = self.vector_store.query(q_emb, top_k=self.candidate_pool_size)

        # Convert distance (cosine 0=identical, 2=opposite) to similarity score
        for r in vector_results:
            r["similarity"] = 1.0 - (r.get("distance", 1.0) / 2.0)

        # Stage 1.5 — BM25 sparse retrieval (parallel)
        if self.bm25_retriever is not None and self.bm25_retriever.is_built():
            bm25_raw = self.bm25_retriever.search(query)
            # Fuse BM25 results with vector results
            vector_results = self._fuse_bm25(vector_results, bm25_raw, self.candidate_pool_size)

        # Stage 2 — Metadata filtering / boosting
        scored = self._apply_metadata_boost(vector_results, filter_topic_id)

        # Stage 3 — Rerank
        if self.reranker is not None:
            scored = self.reranker.rerank(query, scored, top_k=k)
        else:
            # Sort by combined score only when no reranker was used
            scored.sort(key=lambda r: r.score, reverse=True)
        seen_ids: set[str] = set()
        final: list[RetrievalResult] = []
        for r in scored:
            if r.chunk_id not in seen_ids and r.score >= self.similarity_threshold:
                seen_ids.add(r.chunk_id)
                final.append(r)
                if len(final) >= k:
                    break

        return final

    # -- metadata boost ------------------------------------------------------

    def _apply_metadata_boost(
        self,
        results: list[dict],
        filter_topic_id: Optional[str],
    ) -> list[RetrievalResult]:
        """Convert raw vector results to RetrievalResult with metadata boosts."""
        out: list[RetrievalResult] = []
        for r in results:
            meta = r.get("metadata", {})
            score = r["similarity"] * self.vector_weight

            if self.enable_metadata_filtering:
                boost = 0.0

                # Boost ACTIVE documents
                if self.prefer_active and "CÓ_HIỆU_LỰC" in str(meta.get("status", "")):
                    boost += self.metadata_boost * 0.2

                # Boost matching topic
                if self.topic_boost and filter_topic_id:
                    if meta.get("topic_id") == filter_topic_id:
                        boost += self.metadata_boost * 0.5

                # Prefer CLAUSE/POINT-level chunks for specific queries;
                # ARTICLE units still get a small positive signal.
                unit = meta.get("unit_type")
                if unit == "CLAUSE":
                    boost += self.metadata_boost * 0.2
                elif unit == "POINT":
                    boost += self.metadata_boost * 0.1
                elif unit == "ARTICLE":
                    boost += self.metadata_boost * 0.1

                score += boost

            out.append(RetrievalResult(
                chunk_id=r["chunk_id"],
                content=r.get("content", ""),
                raw_content=meta.get("raw_content", r.get("content", "")),
                article_id=meta.get("article_id", ""),
                unit_type=meta.get("unit_type", "ARTICLE"),
                order=meta.get("order", 0),
                hierarchy_path=meta.get("hierarchy_path", ""),
                score=min(score, 1.0),
                source="vector",
                metadata=meta,
            ))
        return out

    # -- BM25 fusion ---------------------------------------------------------

    def _fuse_bm25(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        pool_size: int,
    ) -> list[dict]:
        """Dispatch BM25 fusion based on self.fusion_method."""
        if self.fusion_method == "rrf":
            return self._fuse_bm25_rrf(vector_results, bm25_results, pool_size)
        return self._fuse_bm25_weighted(vector_results, bm25_results, pool_size)

    def _fuse_bm25_weighted(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        pool_size: int,
    ) -> list[dict]:
        """Fuse BM25 sparse scores with vector similarity scores using weighted blend.

        For chunks that appear in both result sets, blend scores
        using weighted average. For chunks only in one set, keep that score
        (slightly penalized). The result list is trimmed to pool_size.
        """
        bm25_by_id: dict[str, float] = {
            r["chunk_id"]: r["bm25_score"] for r in bm25_results
        }

        for vr in vector_results:
            cid = vr.get("chunk_id", "")
            if cid in bm25_by_id:
                vec_sim = vr.get("similarity", 0.0)
                bm25_s = bm25_by_id[cid]
                vr["similarity"] = (
                    (1.0 - self.bm25_weight) * vec_sim
                    + self.bm25_weight * bm25_s
                )

        existing_ids = {r.get("chunk_id", "") for r in vector_results}
        for br in bm25_results:
            if br["chunk_id"] not in existing_ids:
                meta = dict(br.get("metadata", {}))
                meta.setdefault("article_id", br.get("article_id", ""))
                meta.setdefault("unit_type", br.get("unit_type", "ARTICLE"))
                meta.setdefault("order", 0)
                meta.setdefault("hierarchy_path", "")
                vector_results.append({
                    "chunk_id": br["chunk_id"],
                    "content": br["content"],
                    "metadata": meta,
                    "similarity": br["bm25_score"] * self.bm25_weight,
                    "distance": 1.0 - br["bm25_score"],
                })

        vector_results.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
        return vector_results[:pool_size]

    def _fuse_bm25_rrf(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        pool_size: int,
    ) -> list[dict]:
        """Fuse BM25 and vector results using Reciprocal Rank Fusion.

        RRF score = rrf_vector_weight * 1/(rrf_k + rank_v)
                   + rrf_bm25_weight * 1/(rrf_k + rank_b)

        where rank_v is the 1-indexed rank in vector results,
        rank_b is the 1-indexed rank in BM25 results.
        """
        k = self.rrf_k
        w_v = self.rrf_vector_weight
        w_b = self.rrf_bm25_weight

        # Build rank lookups (1-indexed)
        vec_ranks: dict[str, int] = {
            r["chunk_id"]: i + 1 for i, r in enumerate(vector_results)
        }
        bm25_ranks: dict[str, int] = {
            r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)
        }

        all_ids = set(vec_ranks) | set(bm25_ranks)
        fused: list[dict] = []
        for cid in all_ids:
            rrf_score = 0.0
            if cid in vec_ranks:
                rrf_score += w_v / (k + vec_ranks[cid])
            if cid in bm25_ranks:
                rrf_score += w_b / (k + bm25_ranks[cid])

            # Get metadata from whichever source has it
            src = next(
                (r for r in vector_results if r.get("chunk_id") == cid),
                next(
                    (r for r in bm25_results if r.get("chunk_id") == cid),
                    {},
                ),
            )
            meta = dict(src.get("metadata", {}))
            meta.setdefault("article_id", src.get("article_id", ""))
            meta.setdefault("unit_type", src.get("unit_type", "ARTICLE"))
            meta.setdefault("order", 0)
            meta.setdefault("hierarchy_path", "")

            fused.append({
                "chunk_id": cid,
                "content": src.get("content", ""),
                "metadata": meta,
                "similarity": rrf_score,
                "distance": 1.0 - rrf_score,
            })

        fused.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
        return fused[:pool_size]
