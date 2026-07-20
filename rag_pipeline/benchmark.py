#!/usr/bin/env python3
"""Recall@1 / @3 / @10 benchmark using the data/ pipeline.

Indexes articles from data/ (with benchmark IDs from Correct ID.json),
then evaluates retrieval quality against the question set.

Usage:
    python benchmark.py \\
        --config rag_pipeline/config.yaml \\
        --questions 1238_question_map_phap_dien.json \\
        --id-map "Correct ID.json" \\
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
from chunker import chunk_articles, set_tokenizer_model
from embedder import create_embedder
from vector_store import VectorStore
from retriever import Retriever
from query_rewriter import QueryRewriter
from reranker import Reranker, LLMReranker
from bm25_retriever import BM25Retriever

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class Benchmark:
    """Recall@1 / @3 / @10 benchmark — indexes data/ then evaluates retrieval."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        questions_path: str,
        id_map_path: str,
        max_questions: Optional[int] = None,
        force_rechunk: bool = False,
    ) -> dict:
        """Index articles → evaluate retrieval."""
        print("=" * 60)
        print("VLQA Legal RAG — Recall@10 Benchmark")
        print("=" * 60)

        # 1. Index (skips if store exists and not rechunking)
        store, bm25, articles = self._index(id_map_path, force_rechunk)

        # 2. Load questions
        questions = self._load_questions(questions_path, max_questions)

        # 3. Setup retriever
        embedder = create_embedder(self.config)

        rewriter = QueryRewriter(
            append_context=self.config.query_rewriter.append_context,
            expand_abbreviations=self.config.query_rewriter.expand_abbreviations,
            context_phrase=self.config.query_rewriter.context_phrase,
        ) if self.config.query_rewriter.enabled else None

        reranker = None
        if self.config.reranker.enabled:
            if self.config.reranker.mode == "llm_api":
                reranker = LLMReranker(
                    base_url=getattr(self.config, "_api_base_url", ""),
                    api_key=getattr(self.config, "_api_key", ""),
                    model=self.config.reranker.llm.model,
                    endpoint=self.config.reranker.llm.endpoint,
                    max_candidates=self.config.reranker.llm.max_candidates,
                    top_n=self.config.reranker.llm.top_n,
                    temperature=self.config.reranker.llm.temperature,
                    max_output_tokens=self.config.reranker.llm.max_output_tokens,
                    max_candidate_chars=self.config.reranker.llm.max_candidate_chars,
                    blend_weight=self.config.reranker.llm.blend_weight,
                    timeout_sec=self.config.reranker.llm.timeout_sec,
                    conditional_enabled=self.config.reranker.llm.conditional.enabled,
                    conditional_strategy=self.config.reranker.llm.conditional.strategy,
                    conditional_top1_top2_gap=self.config.reranker.llm.conditional.top1_top2_gap,
                    conditional_top1_top10_gap=self.config.reranker.llm.conditional.top1_top10_gap,
                    conditional_min_candidates=self.config.reranker.llm.conditional.min_candidates,
                )
            else:
                reranker = Reranker(
                    base_url=getattr(self.config, "_api_base_url", ""),
                    api_key=self.config.reranker.bge.api_key,
                    model=self.config.reranker.bge.model,
                    endpoint=self.config.reranker.bge.endpoint,
                    max_candidates=self.config.reranker.bge.max_candidates,
                    top_n=self.config.reranker.bge.top_n,
                    timeout_sec=self.config.reranker.bge.timeout_sec,
                    blend_weight=self.config.reranker.bge.blend_weight,
                    conditional_enabled=self.config.reranker.bge.conditional.enabled,
                    conditional_strategy=self.config.reranker.bge.conditional.strategy,
                    conditional_top1_top2_gap=self.config.reranker.bge.conditional.top1_top2_gap,
                    conditional_top1_top10_gap=self.config.reranker.bge.conditional.top1_top10_gap,
                    conditional_min_candidates=self.config.reranker.bge.conditional.min_candidates,
                )

        retriever = Retriever(
            embedder=embedder,
            vector_store=store,
            top_k=self.config.retrieval.top_k,
            candidate_pool_size=self.config.retrieval.candidate_pool_size,
            similarity_threshold=self.config.retrieval.similarity_threshold,
            enable_metadata_filtering=self.config.retrieval.enable_metadata_filtering,
            vector_weight=self.config.retrieval.vector_weight,
            metadata_boost=self.config.retrieval.metadata_boost,
            prefer_active=self.config.metadata.prefer_active,
            topic_boost=self.config.metadata.topic_boost,
            reranker=reranker,
            bm25_retriever=bm25,
            bm25_weight=self.config.bm25.fusion_weight,
            fusion_method=self.config.bm25.fusion_method,
            rrf_k=self.config.bm25.rrf_k,
            rrf_vector_weight=self.config.bm25.rrf_vector_weight,
            rrf_bm25_weight=self.config.bm25.rrf_bm25_weight,
        )

        # 4. Pre-embed all queries (one API batch — fast + retry-safe)
        print(f"\nEmbedding {len(questions)} queries…")
        query_texts = [q["question"] for q in questions]
        if rewriter is not None:
            query_texts = [rewriter.rewrite(t) for t in query_texts]
        query_embeddings = embedder.embed(query_texts, show_progress=True)

        # 5. Evaluate
        print(f"\nEvaluating {len(questions)} questions (Recall@1 / @3 / @10)…")
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0
        per_question: list[dict] = []

        from tqdm import tqdm
        q_iter = tqdm(enumerate(questions), total=len(questions), desc="  Benchmarking", unit="q")
        for i, q in q_iter:
            relevant = set(str(lid) for lid in q["relevant_laws"])

            results = retriever.retrieve(
                query_texts[i], top_k=10,
                query_embedding=query_embeddings[i],
            )

            # Match by benchmark_id (metadata only — never embedded)
            ranked_ids = [
                r.metadata.get("benchmark_id", "")
                for r in results
            ]

            overlap_at_1 = set(ranked_ids[:1]) & relevant
            overlap_at_3 = set(ranked_ids[:3]) & relevant
            overlap_at_10 = set(ranked_ids[:10]) & relevant

            hit_at_1 = len(overlap_at_1) > 0
            hit_at_3 = len(overlap_at_3) > 0
            hit_at_10 = len(overlap_at_10) > 0

            if hit_at_1:
                hits_at_1 += 1
            if hit_at_3:
                hits_at_3 += 1
            if hit_at_10:
                hits_at_10 += 1

            per_question.append({
                "qid": q["qid"],
                "question": q["question"][:100],
                "relevant_count": len(relevant),
                "retrieved_count": len(results),
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_10": hit_at_10,
                "hit": hit_at_10,
                "overlap": list(overlap_at_10)[:5],
            })

            if (i + 1) % 100 == 0:
                print(
                    f"  [{i + 1}/{len(questions)}] "
                    f"R@1: {hits_at_1 / (i + 1):.4f}  "
                    f"R@3: {hits_at_3 / (i + 1):.4f}  "
                    f"R@10: {hits_at_10 / (i + 1):.4f}"
                )

        recall_at_1 = hits_at_1 / len(questions) if questions else 0.0
        recall_at_3 = hits_at_3 / len(questions) if questions else 0.0
        recall_at_10 = hits_at_10 / len(questions) if questions else 0.0
        print(f"\n{'=' * 40}")
        print(f"Recall@1:  {recall_at_1:.4f}  ({hits_at_1}/{len(questions)})")
        print(f"Recall@3:  {recall_at_3:.4f}  ({hits_at_3}/{len(questions)})")
        print(f"Recall@10: {recall_at_10:.4f}  ({hits_at_10}/{len(questions)})")
        if reranker is not None:
            print(f"Reranker calls attempted: {reranker.calls_attempted}")
            print(f"Reranker calls skipped:   {reranker.calls_skipped}")
            print(f"Reranker calls failed:    {reranker.calls_failed}")
        print(f"{'=' * 40}")

        result = {
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_10": recall_at_10,
            "total_questions": len(questions),
            "hits_at_1": hits_at_1,
            "hits_at_3": hits_at_3,
            "hits_at_10": hits_at_10,
            "hits": hits_at_10,
            "per_question": per_question,
        }
        if reranker is not None:
            result.update({
                "reranker_calls_attempted": reranker.calls_attempted,
                "reranker_calls_skipped": reranker.calls_skipped,
                "reranker_calls_failed": reranker.calls_failed,
            })
        return result

    # -- indexing ------------------------------------------------------------

    def _index(self, id_map_path: str, force_rechunk: bool = False):
        """Index data/ articles through the full pipeline.

        If ChromaDB already has data and force_rechunk is False, skips
        chunking + embedding + storing entirely. Only reloads articles
        for the retriever.
        """
        store_path = f"{self.config.vector_store.path}_bench"
        store = VectorStore(path=store_path, collection_name="bench_data")

        if store.count() > 0 and not force_rechunk:
            print("\n[skip] Store already has data — reusing existing index")
            print(f"  {store.count()} chunks in {store_path}")
            articles = load_articles(
                self.config.data.path, id_map_path=id_map_path,
            )
            bm25 = None
            if self.config.bm25.enabled:
                bm25 = BM25Retriever.load(store_path + "/bm25_index.pkl")
                if bm25 is not None and not bm25.has_benchmark_metadata():
                    print("  [rebuild] BM25 index missing metadata — rebuilding from chunks cache…")
                    import pickle as _pickle
                    cache_path = Path("chunks_cache.pkl")
                    if cache_path.exists():
                        with open(cache_path, "rb") as f:
                            chunks = _pickle.load(f)
                    else:
                        from chunker import chunk_articles, set_tokenizer_model
                        set_tokenizer_model(self.config.embedding.model_name)
                        chunks = chunk_articles(
                            articles,
                            max_tokens=self.config.chunking.max_tokens,
                            min_tokens=self.config.chunking.min_tokens,
                            overlap_tokens=self.config.chunking.overlap_tokens,
                            enable_context=self.config.context.enabled,
                            max_context_tokens=self.config.context.max_context_tokens,
                        )
                    bm25 = BM25Retriever(
                        k1=self.config.bm25.k1,
                        b=self.config.bm25.b,
                        top_k=self.config.bm25.top_k,
                    )
                    bm25.build_index(chunks)
                    bm25.save(store_path + "/bm25_index.pkl")
                    print(f"  [rebuild] BM25 index rebuilt ({len(chunks)} chunks)")
            return store, bm25, articles

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
                base_url=getattr(self.config, "_api_base_url", ""),
                api_key=getattr(self.config, "_api_key", ""),
                model=self.config.context.llm.model,
                temperature=self.config.context.llm.temperature,
                max_output_tokens=self.config.context.llm.max_output_tokens,
                timeout_sec=self.config.context.llm.timeout_sec,
            )

        chunks = chunk_articles(
            articles,
            max_tokens=self.config.chunking.max_tokens,
            min_tokens=self.config.chunking.min_tokens,
            overlap_tokens=self.config.chunking.overlap_tokens,
            enable_context=self.config.context.enabled,
            max_context_tokens=self.config.context.max_context_tokens,
            llm_context_generator=llm_gen,
            force_rechunk=force_rechunk,
        )
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
            bm25.save(store_path + "/bm25_index.pkl")

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
    parser.add_argument("--rechunk", action="store_true", help="Force re-chunking (ignore cache)")
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
        force_rechunk=args.rechunk,
    )

    out_path = Path("benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
