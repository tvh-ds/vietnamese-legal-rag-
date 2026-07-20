#!/usr/bin/env python3
"""BM25 parameter grid search tuned for Recall@1.

Usage:
    python rag_pipeline/tune_bm25.py \
        --config rag_pipeline/config.yaml \
        --questions 1238_question_map_phap_dien.json \
        --id-map "Correct ID.json"

Output: bm25_tuning_results.json (per-combo metrics, sorted by R@1)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config
from data_loader import load_articles
from chunker import chunk_articles, set_tokenizer_model
from embedder import create_embedder
from vector_store import VectorStore
from retriever import Retriever
from query_rewriter import QueryRewriter
from bm25_retriever import BM25Retriever

logger = logging.getLogger("tune_bm25")


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Default grid — extend by editing these lists.
_K1_VALUES = [1.2, 1.5, 1.8]
_B_VALUES = [0.75]
_BM25_TOP_K_VALUES = [50]
_FUSION_WEIGHTS = [0.3, 0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# Tuning runner
# ---------------------------------------------------------------------------

class BM25Tuner:
    """Grid search over BM25 parameters, reporting Recall@1 / @3 / @10."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        questions_path: str,
        id_map_path: str,
        max_questions: int | None = None,
        k1_values: list[float] | None = None,
        b_values: list[float] | None = None,
        bm25_top_k_values: list[int] | None = None,
        fusion_weights: list[float] | None = None,
    ) -> list[dict]:
        """Run grid search and return results sorted by Recall@1."""

        k1_values = k1_values or _K1_VALUES
        b_values = b_values or _B_VALUES
        bm25_top_k_values = bm25_top_k_values or _BM25_TOP_K_VALUES
        fusion_weights = fusion_weights or _FUSION_WEIGHTS

        print("=" * 60)
        print("BM25 Parameter Tuning  (optimising for Recall@1)")
        print("=" * 60)

        # 1. Load articles with benchmark IDs
        print("\n[1/5] Loading articles…")
        articles = load_articles(
            self.config.data.path,
            id_map_path=id_map_path,
        )
        matched = sum(1 for a in articles if a.benchmark_id)
        print(f"  Loaded {len(articles)} articles ({matched} with benchmark IDs)")

        # 2. Load / rebuild chunks (work from cache)
        print("\n[2/5] Loading chunks…")
        set_tokenizer_model(self.config.embedding.model_name)
        chunks = chunk_articles(
            articles,
            max_tokens=self.config.chunking.max_tokens,
            min_tokens=self.config.chunking.min_tokens,
            overlap_tokens=self.config.chunking.overlap_tokens,
            enable_context=self.config.context.enabled,
            max_context_tokens=self.config.context.max_context_tokens,
            force_rechunk=False,
        )
        print(f"  {len(chunks)} chunks loaded")

        # 3. Load vector store (existing — no rebuild)
        print("\n[3/5] Loading vector store…")
        store_path = f"{self.config.vector_store.path}_bench"
        store = VectorStore(path=store_path, collection_name="bench_data")
        print(f"  {store.count()} chunks in vector store")

        # 4. Load questions
        print("\n[4/5] Loading questions…")
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        if max_questions:
            questions = questions[:max_questions]
        print(f"  {len(questions)} questions loaded")

        # 5. Setup embedder and pre-embed all queries ONCE
        print("\n[5/5] Pre-embedding queries (one batch)…")
        embedder = create_embedder(self.config)

        rewriter = QueryRewriter(
            append_context=self.config.query_rewriter.append_context,
            expand_abbreviations=self.config.query_rewriter.expand_abbreviations,
            context_phrase=self.config.query_rewriter.context_phrase,
        ) if self.config.query_rewriter.enabled else None

        query_texts = [q["question"] for q in questions]
        if rewriter is not None:
            query_texts = [rewriter.rewrite(t) for t in query_texts]
        query_embeddings = embedder.embed(query_texts, show_progress=True)
        print(f"  {query_embeddings.shape[0]} embeddings, {query_embeddings.shape[1]}-dim")

        # Build base retriever (without BM25 for now)
        base_retriever = Retriever(
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
            reranker=None,
            bm25_retriever=None,
            bm25_weight=0.0,
        )
        base_retriever.reranker = None

        # Grid search
        results: list[dict] = []
        total_combos = (
            len(k1_values) * len(b_values)
            * len(bm25_top_k_values) * len(fusion_weights)
        )
        combo_idx = 0

        for k1 in k1_values:
            for b in b_values:
                for bm25_top_k in bm25_top_k_values:
                    # Build BM25 once per (k1, b, top_k) combo
                    print(f"\n--- BM25: k1={k1}, b={b}, top_k={bm25_top_k} ---")
                    t0 = time.perf_counter()
                    bm25 = BM25Retriever(k1=k1, b=b, top_k=bm25_top_k)
                    bm25.build_index(chunks)
                    build_time = time.perf_counter() - t0
                    print(f"  Build: {build_time:.1f}s")

                    base_retriever.bm25_retriever = bm25

                    for fw in fusion_weights:
                        combo_idx += 1
                        print(f"  [{combo_idx}/{total_combos}] fusion_weight={fw} …", end=" ")

                        base_retriever.bm25_weight = fw

                        hits_at_1 = 0
                        hits_at_3 = 0
                        hits_at_10 = 0

                        for i, q in enumerate(questions):
                            relevant = set(str(lid) for lid in q["relevant_laws"])

                            retrieved = base_retriever.retrieve(
                                query_texts[i],
                                top_k=10,
                                query_embedding=query_embeddings[i],
                            )

                            ranked_ids = [
                                r.metadata.get("benchmark_id", "")
                                for r in retrieved
                            ]

                            if len(set(ranked_ids[:1]) & relevant) > 0:
                                hits_at_1 += 1
                            if len(set(ranked_ids[:3]) & relevant) > 0:
                                hits_at_3 += 1
                            if len(set(ranked_ids[:10]) & relevant) > 0:
                                hits_at_10 += 1

                        n = len(questions) if questions else 1
                        r1 = hits_at_1 / n
                        r3 = hits_at_3 / n
                        r10 = hits_at_10 / n

                        print(
                            f"R@1={r1:.4f}  R@3={r3:.4f}  R@10={r10:.4f}"
                            f"  ({hits_at_1}/{hits_at_3}/{hits_at_10})"
                        )

                        results.append({
                            "k1": k1,
                            "b": b,
                            "bm25_top_k": bm25_top_k,
                            "fusion_weight": fw,
                            "recall_at_1": round(r1, 6),
                            "recall_at_3": round(r3, 6),
                            "recall_at_10": round(r10, 6),
                            "hits_at_1": hits_at_1,
                            "hits_at_3": hits_at_3,
                            "hits_at_10": hits_at_10,
                        })

        # Sort by Recall@1 descending, then R@3, then R@10
        results.sort(
            key=lambda r: (-r["recall_at_1"], -r["recall_at_3"], -r["recall_at_10"])
        )

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 parameter grid search for Recall@1 optimisation",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--questions", required=True, help="Path to benchmark questions JSON")
    parser.add_argument("--id-map", required=True, help="Path to Correct ID.json")
    parser.add_argument("-n", "--max-questions", type=int, default=None)
    parser.add_argument("--output", default="bm25_tuning_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate
    config = load_config(str(config_path))

    # Resolve data path relative to config
    config_dir = config_path.parent
    if not Path(config.data.path).is_absolute():
        config.data.path = str(config_dir / config.data.path)

    tuner = BM25Tuner(config)
    results = tuner.run(
        questions_path=args.questions,
        id_map_path=args.id_map,
        max_questions=args.max_questions,
    )

    # Save
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}")

    # Print leaderboard
    print(f"\n{'=' * 60}")
    print(f"Leaderboard (sorted by Recall@1)")
    print(f"{'=' * 60}")
    print(
        f"  {'k1':<5} {'b':<5} {'bm25_top_k':<12} {'fusion_wt':<10} "
        f"{'R@1':<8} {'R@3':<8} {'R@10':<8} {'hits (1/3/10)':<15}"
    )
    print(f"  {'-'*5} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")
    for r in results:
        hits = f"{r['hits_at_1']}/{r['hits_at_3']}/{r['hits_at_10']}"
        print(
            f"  {r['k1']:<5} {r['b']:<5} {r['bm25_top_k']:<12} {r['fusion_weight']:<10} "
            f"{r['recall_at_1']:<8.4f} {r['recall_at_3']:<8.4f} {r['recall_at_10']:<8.4f} "
            f"{hits:<15}"
        )


if __name__ == "__main__":
    main()