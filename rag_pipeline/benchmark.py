#!/usr/bin/env python3
"""Recall@10 benchmark using the data/ pipeline.

Indexes articles from data/ (with benchmark IDs from Correct ID.json),
then evaluates retrieval quality against the question set.

Usage:
    python benchmark.py \\
        --config rag_pipeline/config.yaml \\
        --questions 1238_question_map_phap_dien.json \\
        --id-map "Correct ID.json" \\
        -n 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config
from context_generator import LLMContextGenerator
from data_loader import load_articles
from chunker import Chunker, set_tokenizer_model
from embedder import create_embedder
from vector_store import VectorStore
from retriever import Retriever
from query_rewriter import QueryRewriter
from reranker import Reranker
from bm25_retriever import BM25Retriever

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class Benchmark:
    """Recall@10 benchmark — indexes data/ then evaluates retrieval."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        questions_path: str,
        id_map_path: str,
        max_questions: Optional[int] = None,
    ) -> dict:
        """Index articles → evaluate retrieval."""
        print("=" * 60)
        print("VLQA Legal RAG — Recall@10 Benchmark")
        print("=" * 60)

        # 1. Index
        store, bm25, articles = self._index(id_map_path)

        # 2. Load questions
        questions = self._load_questions(questions_path, max_questions)

        # 3. Setup retriever
        embedder = create_embedder(self.config)

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
            enable_graph_expansion=False,
            vector_weight=self.config.retrieval.vector_weight,
            metadata_boost=self.config.retrieval.metadata_boost,
            graph_boost=0.0,
            prefer_active=self.config.metadata.prefer_active,
            topic_boost=self.config.metadata.topic_boost,
            reranker=reranker,
            bm25_retriever=bm25,
            bm25_weight=self.config.bm25.fusion_weight,
        )

        # 4. Evaluate
        print(f"\nEvaluating {len(questions)} questions (Recall@10)…")
        hits = 0
        per_question: list[dict] = []

        for i, q in enumerate(questions):
            query_text = q["question"]
            relevant = set(str(lid) for lid in q["relevant_laws"])

            if rewriter is not None:
                query_text = rewriter.rewrite(query_text)

            results = retriever.retrieve(query_text, top_k=10)

            # Match by benchmark_id (metadata only — never embedded)
            retrieved_ids = {
                r.metadata.get("benchmark_id", "")
                for r in results
            }
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
                print(f"  [{i + 1}/{len(questions)}] Recall@10: {hits / (i + 1):.4f}")

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

    def _index(self, id_map_path: str):
        """Index data/ articles through the full pipeline."""

        # 1. Load with benchmark IDs
        print("\n[1/4] Loading articles…")
        t0 = time.perf_counter()
        articles = load_articles(
            self.config.data.path,
            id_map_path=id_map_path,
        )
        t1 = time.perf_counter()
        matched = sum(1 for a in articles if a.benchmark_id)
        print(f"  Loaded {len(articles)} articles ({matched} with benchmark IDs) in {t1 - t0:.1f}s")

        # 2. Chunk
        print("\n[2/4] Chunking…")
        set_tokenizer_model(self.config.embedding.model_name)
        t0 = time.perf_counter()

        llm_gen = None
        if self.config.context.use_llm:
            llm_gen = LLMContextGenerator(
                url=self.config.context.llm.url,
                api_key=self.config.context.llm.api_key,
                model=self.config.context.llm.model,
                temperature=self.config.context.llm.temperature,
                max_output_tokens=self.config.context.llm.max_output_tokens,
                timeout_sec=self.config.context.llm.timeout_sec,
            )

        chunker = Chunker(
            max_tokens=self.config.chunking.max_tokens,
            min_tokens=self.config.chunking.min_tokens,
            enable_context=self.config.context.enabled,
            max_context_tokens=self.config.context.max_context_tokens,
            llm_context_generator=llm_gen,
        )
        chunks = []
        for art in articles:
            chunks.extend(chunker.chunk_article(art))
        t1 = time.perf_counter()
        print(f"  Produced {len(chunks)} chunks in {t1 - t0:.1f}s")

        # 3. Embed
        print("\n[3/4] Embedding…")
        t0 = time.perf_counter()
        embedder = create_embedder(self.config)
        if self.config.embedding.backend == "api":
            print(f"  Backend: API ({self.config.embedding.api.model})")
        else:
            print(f"  Model: {self.config.embedding.model_name}")
        embeddings = embedder.embed([ch.content for ch in chunks], show_progress=True)
        t1 = time.perf_counter()
        print(f"  {embeddings.shape[1]}-dim, {len(chunks)} vectors in {t1 - t0:.1f}s")

        # 4. Store
        print("\n[4/4] Storing…")
        t0 = time.perf_counter()
        store = VectorStore(
            path=f"{self.config.vector_store.path}_bench",
            collection_name="bench_data",
        )
        store.clear()
        store.upsert(chunks, embeddings)
        t1 = time.perf_counter()
        print(f"  Stored {store.count()} chunks in {t1 - t0:.1f}s")

        # Build BM25
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
        description="Recall@10 benchmark using data/ pipeline",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--questions", required=True, help="Path to benchmark questions JSON")
    parser.add_argument("--id-map", required=True, help="Path to Correct ID.json")
    parser.add_argument("-n", "--max-questions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate
    config = load_config(str(config_path))

    # Resolve paths relative to config
    config_dir = config_path.parent
    if not Path(config.data.path).is_absolute():
        config.data.path = str(config_dir / config.data.path)

    benchmark = Benchmark(config)
    result = benchmark.run(
        questions_path=args.questions,
        id_map_path=args.id_map,
        max_questions=args.max_questions,
    )

    out_path = Path("benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
