"""Hybrid retrieval combining vector search, metadata filtering, and graph expansion.

Implements Sections 10 & 11 of Pipeline.md:

  1. Vector Search — semantic similarity via embeddings.
  2. Metadata Filtering — filter/boost by status, effectiveDate, topic, unitType.
  3. Graph Expansion — follow article references (Chỉ dẫn) to pull in related chunks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from bm25_retriever import BM25Retriever
from data_loader import Article
from embedder import Embedder
from reranker import Reranker
from pipeline_types import RetrievalResult
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Hybrid retriever for Vietnamese legal chunks.

    Combines dense vector search + BM25 sparse retrieval with
    metadata-based filtering/boosting and optional graph expansion
    through article cross-references.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        articles: list[Article],
        *,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        enable_metadata_filtering: bool = True,
        enable_graph_expansion: bool = True,
        expansion_max_articles: int = 5,
        vector_weight: float = 0.7,
        metadata_boost: float = 0.2,
        graph_boost: float = 0.1,
        prefer_active: bool = True,
        topic_boost: bool = True,
        reranker: Reranker | None = None,
        bm25_retriever: BM25Retriever | None = None,
        bm25_weight: float = 0.5,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_metadata_filtering = enable_metadata_filtering
        self.enable_graph_expansion = enable_graph_expansion
        self.expansion_max_articles = expansion_max_articles
        self.vector_weight = vector_weight
        self.metadata_boost = metadata_boost
        self.graph_boost = graph_boost
        self.prefer_active = prefer_active
        self.topic_boost = topic_boost
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever
        self.bm25_weight = bm25_weight

        # Build article lookup by article_id for graph expansion
        self._article_by_id: dict[str, Article] = {
            a.article_id: a for a in articles
        }

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
        vector_results = self.vector_store.query(q_emb, top_k=k * 2)

        # Convert distance (cosine 0=identical, 2=opposite) to similarity score
        for r in vector_results:
            r["similarity"] = 1.0 - (r.get("distance", 1.0) / 2.0)

        # Stage 1.5 — BM25 sparse retrieval (parallel)
        if self.bm25_retriever is not None and self.bm25_retriever.is_built():
            bm25_raw = self.bm25_retriever.search(query)
            # Fuse BM25 results with vector results
            vector_results = self._fuse_bm25(vector_results, bm25_raw, k * 2)

        # Stage 2 — Metadata filtering / boosting
        scored = self._apply_metadata_boost(vector_results, filter_topic_id)

        # Stage 3 — Graph expansion
        if self.enable_graph_expansion:
            graph_results = self._graph_expand(scored[:k], query_embedding)
            scored = self._merge_results(scored, graph_results)

        # Stage 4 — Rerank with cross-attention / MMR / keyword boost
        if self.reranker is not None:
            scored = self.reranker.rerank(query, scored, top_k=k)

        # Sort by combined score, deduplicate, trim
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
                    boost += self.metadata_boost * 0.5

                # Boost matching topic
                if self.topic_boost and filter_topic_id:
                    if meta.get("topic_id") == filter_topic_id:
                        boost += self.metadata_boost * 0.5

                # Prefer ARTICLE-level chunks (more context)
                if meta.get("unit_type") == "ARTICLE":
                    boost += self.metadata_boost * 0.2
                elif meta.get("unit_type") == "CLAUSE":
                    boost += self.metadata_boost * 0.1

                score += boost

            out.append(RetrievalResult(
                chunk_id=r["chunk_id"],
                content=r.get("content", ""),
                raw_content="",  # Will be populated below
                article_id=meta.get("article_id", ""),
                unit_type=meta.get("unit_type", "ARTICLE"),
                order=meta.get("order", 0),
                hierarchy_path=meta.get("hierarchy_path", ""),
                score=min(score, 1.0),
                source="vector",
                metadata=meta,
            ))
        return out

    # -- graph expansion -----------------------------------------------------

    def _graph_expand(
        self,
        top_results: list[RetrievalResult],
        query_embedding: np.ndarray,
    ) -> list[RetrievalResult]:
        """Expand results by following article cross-references.

        For each top result's article, look up its references (Chỉ dẫn),
        find those referenced articles' chunks in the vector store,
        and add them to the candidate pool.
        """
        seen_article_ids: set[str] = set()
        referenced_article_ids: set[str] = set()

        for r in top_results:
            seen_article_ids.add(r.article_id)
            article = self._article_by_id.get(r.article_id)
            if article is None:
                continue
            for ref in article.references:
                ref_id = ref.get("id")
                if ref_id and ref_id not in seen_article_ids:
                    referenced_article_ids.add(ref_id)

        if not referenced_article_ids:
            return []

        # Limit expansion size
        ref_ids = list(referenced_article_ids)[:self.expansion_max_articles]

        expanded: list[RetrievalResult] = []
        for ref_id in ref_ids:
            ref_chunks = self.vector_store.get_by_article(ref_id)
            for rc in ref_chunks:
                meta = rc.get("metadata", {})
                # Score based on graph boost (no vector similarity available)
                expanded.append(RetrievalResult(
                    chunk_id=rc["chunk_id"],
                    content=rc.get("content", ""),
                    raw_content="",
                    article_id=ref_id,
                    unit_type=meta.get("unit_type", "ARTICLE"),
                    order=meta.get("order", 0),
                    hierarchy_path=meta.get("hierarchy_path", ""),
                    score=self.graph_boost,
                    source="graph_expansion",
                    metadata=meta,
                ))

        return expanded

    # -- merge ---------------------------------------------------------------

    @staticmethod
    def _merge_results(
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge vector and graph results, deduplicating by chunk_id."""
        seen: set[str] = set()
        merged: list[RetrievalResult] = []

        for r in vector_results + graph_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                merged.append(r)
            else:
                # If already present (from vector search), boost score
                for existing in merged:
                    if existing.chunk_id == r.chunk_id:
                        existing.score = min(existing.score + 0.05, 1.0)
                        break

        return merged

    # -- BM25 fusion ---------------------------------------------------------

    def _fuse_bm25(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        pool_size: int,
    ) -> list[dict]:
        """Fuse BM25 sparse scores with vector similarity scores.

        Strategy: For chunks that appear in both result sets, blend scores
        using weighted average. For chunks only in one set, keep that score
        (slightly penalized). The result list is trimmed to pool_size.
        """
        # Build lookup of BM25 scores by chunk_id
        bm25_by_id: dict[str, float] = {
            r["chunk_id"]: r["bm25_score"] for r in bm25_results
        }

        # Blend BM25 scores into vector results
        for vr in vector_results:
            cid = vr.get("chunk_id", "")
            if cid in bm25_by_id:
                # Weighted blend: vector dominates, BM25 boosts exact matches
                vec_sim = vr.get("similarity", 0.0)
                bm25_s = bm25_by_id[cid]
                vr["similarity"] = (
                    (1.0 - self.bm25_weight) * vec_sim
                    + self.bm25_weight * bm25_s
                )

        # Add BM25-only results that aren't in vector results
        existing_ids = {r.get("chunk_id", "") for r in vector_results}
        for br in bm25_results:
            if br["chunk_id"] not in existing_ids:
                vector_results.append({
                    "chunk_id": br["chunk_id"],
                    "content": br["content"],
                    "metadata": {
                        "article_id": br.get("article_id", ""),
                        "unit_type": br.get("unit_type", "ARTICLE"),
                        "order": 0,
                        "hierarchy_path": "",
                    },
                    "similarity": br["bm25_score"] * self.bm25_weight,
                    "distance": 1.0 - br["bm25_score"],
                })

        # Sort by blended similarity, trim
        vector_results.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
        return vector_results[:pool_size]
