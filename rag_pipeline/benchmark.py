#!/usr/bin/env python3
"""Recall@10 benchmark for the Vietnamese Legal RAG pipeline.

Indexes the VLQA legal corpus through the chunking pipeline, then evaluates
retrieval quality on the benchmark question set.

Usage:
    python benchmark.py \\
        --corpus "D:/intern/Test/vlqa/legal_corpus.json" \\
        --questions "D:/intern/Test/vlqa/1238_question_map_phap_dien.json" \\
        --config rag_pipeline/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure rag_pipeline is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config
from corpus_loader import load_corpus
from chunker import Chunker, set_tokenizer_model
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from query_rewriter import QueryRewriter
from reranker import Reranker
from bm25_retriever import BM25Retriever

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class Benchmark:
    """Recall@10 benchmark for legal retrieval."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        corpus_path: str,
        questions_path: str,
        max_questions: Optional[int] = None,
    ) -> dict:
        """Run the full benchmark: index → evaluate.

        Returns dict with recall@10 and per-question details.
        """
        # 1. Index the corpus
        print("=" * 60)
        print("VLQA Legal RAG — Recall@10 Benchmark")
        print("=" * 60)

        store, bm25, articles = self._index_corpus(corpus_path)

        # 2. Load questions
        questions = self._load_questions(questions_path, max_questions)

        # 3. Setup retriever
        embedder = Embedder(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
        )
        rewriter = QueryRewriter(
            append_context=self.config.query_rewriter.append_context,
            expand_abbreviations=self.config.query_rewriter.expand_abbreviations,
            context_phrase=self.config.query_rewriter.context_phrase,
        ) if self.config.query_rewriter.enabled else None

        reranker = Reranker(
            lambda_mmr=self.config.reranker.lambda_mmr,
            keyword_boost=self.config.reranker.keyword_boost,
            diversity_weight=self.config.reranker.diversity_weight,
        ) if self.config.reranker.enabled else None

        retriever = Retriever(
            embedder=embedder,
            vector_store=store,
            articles=articles,
            top_k=self.config.retrieval.top_k,
            similarity_threshold=self.config.retrieval.similarity_threshold,
            enable_metadata_filtering=self.config.retrieval.enable_metadata_filtering,
            enable_graph_expansion=False,  # graph expansion not applicable for corpus
            vector_weight=self.config.retrieval.vector_weight,
            metadata_boost=self.config.retrieval.metadata_boost,
            graph_boost=0.0,
            prefer_active=self.config.metadata.prefer_active,
            topic_boost=self.config.metadata.topic_boost,
            reranker=reranker,
            bm25_retriever=bm25,
            bm25_weight=0.3,
        )

        # 4. Evaluate
        print(f"\nEvaluating {len(questions)} questions (Recall@10)…")
        hits = 0
        per_question: list[dict] = []

        for i, q in enumerate(questions):
            query_text = q["question"]
            relevant = set(str(law_id) for law_id in q["relevant_laws"])

            # Rewrite query
            if rewriter is not None:
                query_text = rewriter.rewrite(query_text)

            # Retrieve
            results = retriever.retrieve(query_text, top_k=10)
            retrieved_ids = {r.article_id for r in results}

            # Check hit
            overlap = retrieved_ids & relevant
            hit = len(overlap) > 0
            if hit:
                hits += 1

            per_question.append({
                "qid": q["qid"],
                "question": q["question"][:100],
                "relevant_count": len(relevant),
                "retrieved_count": len(results),
                "hit": hit,
                "overlap": list(overlap)[:5],
            })

            if (i + 1) % 100 == 0:
                current_recall = hits / (i + 1)
                print(f"  [{i + 1}/{len(questions)}] Recall@10: {current_recall:.4f}")

        recall = hits / len(questions) if questions else 0.0
        print(f"\n{'=' * 40}")
        print(f"Recall@10: {recall:.4f}  ({hits}/{len(questions)})")
        print(f"{'=' * 40}")

        return {
            "recall_at_10": recall,
            "total_questions": len(questions),
            "hits": hits,
            "per_question": per_question,
        }

    # -- indexing ------------------------------------------------------------

    def _index_corpus(self, corpus_path: str) -> tuple[VectorStore, Optional[BM25Retriever], list]:
        """Index the VLQA corpus through the pipeline."""

        # 1. Load
        print("\n[1/4] Loading corpus…")
        t0 = time.perf_counter()
        articles = load_corpus(corpus_path)
        t1 = time.perf_counter()
        print(f"  Loaded {len(articles)} articles in {t1 - t0:.1f}s")

        # 2. Chunk
        print("\n[2/4] Chunking…")
        set_tokenizer_model(self.config.embedding.model_name)
        t0 = time.perf_counter()
        chunker = Chunker(
            max_tokens=self.config.chunking.max_tokens,
            min_tokens=self.config.chunking.min_tokens,
            enable_context=self.config.context.enabled,
            max_context_tokens=self.config.context.max_context_tokens,
        )
        chunks = []
        for art in articles:
            chunks.extend(chunker.chunk_article(art))
        t1 = time.perf_counter()
        print(f"  Produced {len(chunks)} chunks in {t1 - t0:.1f}s")

        # 3. Embed
        print("\n[3/4] Embedding…")
        print(f"  Model: {self.config.embedding.model_name}")
        t0 = time.perf_counter()
        embedder = Embedder(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
        )
        embeddings = embedder.embed([ch.content for ch in chunks], show_progress=True)
        t1 = time.perf_counter()
        print(f"  {embeddings.shape[1]}-dim, {len(chunks)} vectors in {t1 - t0:.1f}s")

        # 4. Store
        print("\n[4/4] Storing…")
        t0 = time.perf_counter()
        store = VectorStore(
            path=f"{self.config.vector_store.path}_benchmark",
            collection_name="benchmark_corpus",
        )
        store.clear()
        store.upsert(chunks, embeddings)
        t1 = time.perf_counter()
        print(f"  Stored {store.count()} chunks in {t1 - t0:.1f}s")

        # Build BM25 index
        bm25 = None
        if self.config.bm25.enabled:
            bm25 = BM25Retriever(
                k1=self.config.bm25.k1,
                b=self.config.bm25.b,
                top_k=self.config.bm25.top_k,
            )
            bm25.build_index(chunks)  # type: ignore[arg-type]

        print()
        return store, bm25, articles

    # -- question loading ----------------------------------------------------

    @staticmethod
    def _load_questions(path: str, max_q: Optional[int] = None) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        if max_q:
            questions = questions[:max_q]
        return questions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall@10 benchmark for Vietnamese Legal RAG",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to legal_corpus.json",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to benchmark questions JSON",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "-n", "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for quick tests)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate

    config = load_config(str(config_path))

    benchmark = Benchmark(config)
    result = benchmark.run(
        corpus_path=args.corpus,
        questions_path=args.questions,
        max_questions=args.max_questions,
    )

    # Save detailed results
    out_path = Path("benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
