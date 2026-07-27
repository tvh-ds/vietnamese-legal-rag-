#!/usr/bin/env python3
"""Recall@1 / @3 / @10 benchmark using the data/ pipeline.

Indexes articles from data/ (with benchmark IDs from Correct ID.json),
then evaluates retrieval quality against the question set.

Usage:
    python benchmark.py \\
        --config rag_pipeline/config.yaml \\
        --questions benchmarks/input/1238_question_map_phap_dien.json \\
        --id-map "benchmarks/input/Correct ID.json" \\
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config, load_config
from data.data_loader import load_articles
from indexing.chunker import chunk_articles, set_tokenizer_model
from indexing.context_generator import LLMContextGenerator
from indexing.embedder import create_embedder
from indexing.vector_store import VectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

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
        run_start = time.perf_counter()
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
            reranker = Reranker(
                    base_url=getattr(self.config, "_api_base_url", ""),
                    api_key=self.config.reranker.bge.api_key,
                    model=self.config.reranker.bge.model,
                    endpoint=self.config.reranker.bge.endpoint,
                    max_candidates=self.config.reranker.bge.max_candidates,
                    top_n=self.config.reranker.bge.top_n,
                    timeout_sec=self.config.reranker.bge.timeout_sec,
                    blend_weight=self.config.reranker.bge.blend_weight,
                    mode=self.config.reranker.bge.mode,
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
        print(f"\nEvaluating {len(questions)} questions (Recall + Precision @1 / @3 / @10)…")
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0
        precision_sum_at_1 = 0.0
        precision_sum_at_3 = 0.0
        precision_sum_at_10 = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        r_precision_sum = 0.0
        normalized_precision_sum_at_10 = 0.0
        ideal_precision_sum_at_10 = 0.0
        category_counts: dict[str, int] = {}
        pre_mrr_sum = 0.0
        pre_ndcg_sum = 0.0
        pre_norm_prec_sum = 0.0
        candidate_recall_sum = 0.0
        oracle_recall_sum = 0.0
        oracle_norm_prec_sum = 0.0
        reranker_rel_score_sum = 0.0
        reranker_irr_score_sum = 0.0
        reranker_rel_count = 0
        reranker_irr_count = 0
        per_question: list[dict] = []

        from tqdm import tqdm
        q_iter = tqdm(enumerate(questions), total=len(questions), desc="  Benchmarking", unit="q")
        for i, q in q_iter:
            relevant = set(str(lid) for lid in q["relevant_laws"])

            results = retriever.retrieve(
                query_texts[i], top_k=10,
                query_embedding=query_embeddings[i],
            )

            # --- Reranker diagnostics (pre) ---
            pre_candidates = getattr(retriever, '_pre_rerank_candidates', [])
            pre_ranked_ids = [r.metadata.get("benchmark_id", "") for r in pre_candidates]
            pre_hit_at_10 = len(set(pre_ranked_ids[:10]) & relevant) > 0
            pre_first_relevant_rank = self._first_relevant_rank(pre_ranked_ids, relevant)
            pre_mrr = self._calc_mrr(pre_ranked_ids, relevant)
            pre_ndcg = self._calc_ndcg(pre_ranked_ids, relevant)
            pre_norm_prec = self._calc_normalized_precision(pre_ranked_ids, relevant, 10)
            candidate_relevant_ids = set(pre_ranked_ids) & relevant
            candidate_has_relevant = len(candidate_relevant_ids) > 0
            candidate_unique_count = len(candidate_relevant_ids)
            r_count = len(relevant)
            oracle_recall_at_10 = min(candidate_unique_count, 10) / r_count if r_count > 0 else 0.0
            oracle_norm_prec_at_10 = min(candidate_unique_count, 10) / min(r_count, 10) if min(r_count, 10) > 0 else 0.0
            reranker_diag = getattr(retriever.reranker, 'last_diagnostics', None) if retriever.reranker else None
            # --- end diagnostics pre ---

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

            precision_at_1 = len(overlap_at_1) / 1
            precision_at_3 = len(overlap_at_3) / 3
            precision_at_10 = len(overlap_at_10) / 10

            # Ranking metrics (unique relevant laws only)
            mrr = self._calc_mrr(ranked_ids, relevant)
            ndcg = self._calc_ndcg(ranked_ids, relevant)
            r_prec = self._calc_r_precision(ranked_ids, relevant)
            norm_prec = self._calc_normalized_precision(ranked_ids, relevant, 10)
            ideal_prec = min(len(relevant), 10) / 10

            mrr_sum += mrr
            ndcg_sum += ndcg
            r_precision_sum += r_prec
            normalized_precision_sum_at_10 += norm_prec
            ideal_precision_sum_at_10 += ideal_prec

            if hit_at_1:
                hits_at_1 += 1
            if hit_at_3:
                hits_at_3 += 1
            if hit_at_10:
                hits_at_10 += 1
            precision_sum_at_1 += precision_at_1
            precision_sum_at_3 += precision_at_3
            precision_sum_at_10 += precision_at_10

            # --- Reranker diagnostics (categorize + accumulate) ---
            post_first_relevant_rank = self._first_relevant_rank(ranked_ids, relevant)
            if not candidate_has_relevant:
                category = "missing_from_candidates"
            elif reranker_diag and not reranker_diag.get("applied", False):
                cat_reason = reranker_diag.get("reason", "unknown")
                if cat_reason in ("api_exception", "api_null_response"):
                    category = "reranker_failed"
                elif cat_reason == "conditional_skip":
                    category = "reranker_skipped"
                else:
                    category = f"reranker_not_applied_{cat_reason}"
            elif not retriever.reranker:
                category = "no_reranker"
            else:
                if not pre_hit_at_10 and hit_at_10:
                    category = "improved_gain"
                elif pre_hit_at_10 and not hit_at_10:
                    category = "hurt_loss"
                elif pre_hit_at_10 and hit_at_10:
                    if post_first_relevant_rank < pre_first_relevant_rank:
                        category = "improved_rank"
                    elif post_first_relevant_rank > pre_first_relevant_rank:
                        category = "hurt_rank"
                    else:
                        category = "unchanged_hit"
                else:
                    category = "unchanged_miss"

            category_counts[category] = category_counts.get(category, 0) + 1
            pre_mrr_sum += pre_mrr
            pre_ndcg_sum += pre_ndcg
            pre_norm_prec_sum += pre_norm_prec
            candidate_recall_sum += candidate_unique_count / r_count if r_count > 0 else 0.0
            oracle_recall_sum += oracle_recall_at_10
            oracle_norm_prec_sum += oracle_norm_prec_at_10

            # Analyze reranker API scores (relevant vs irrelevant)
            avg_rel_api_score = None
            avg_irr_api_score = None
            if reranker_diag and reranker_diag.get("applied", False):
                score_trace = reranker_diag.get("score_trace", [])
                rel_scores = []
                irr_scores = []
                chunk_to_bid = {c.chunk_id: c.metadata.get("benchmark_id", "") for c in pre_candidates}
                for item in score_trace:
                    bid = chunk_to_bid.get(item["chunk_id"], "")
                    if bid in relevant:
                        rel_scores.append(item["api_score"])
                    else:
                        irr_scores.append(item["api_score"])
                if rel_scores:
                    avg_rel_api_score = sum(rel_scores) / len(rel_scores)
                    reranker_rel_score_sum += avg_rel_api_score
                    reranker_rel_count += 1
                if irr_scores:
                    avg_irr_api_score = sum(irr_scores) / len(irr_scores)
                    reranker_irr_score_sum += avg_irr_api_score
                    reranker_irr_count += 1

            rd_entry = {
                "qid": q["qid"],
                "question": q["question"][:100],
                "num_relevant": r_count,
                "candidate_pool_size": len(pre_candidates),
                "candidate_has_relevant": candidate_has_relevant,
                "candidate_unique_relevant_count": candidate_unique_count,
                "pre_hit_at_10": pre_hit_at_10,
                "post_hit_at_10": hit_at_10,
                "pre_first_relevant_rank": pre_first_relevant_rank,
                "post_first_relevant_rank": post_first_relevant_rank,
                "pre_mrr_at_10": pre_mrr,
                "post_mrr_at_10": mrr,
                "pre_ndcg_at_10": pre_ndcg,
                "post_ndcg_at_10": ndcg,
                "pre_normalized_precision_at_10": pre_norm_prec,
                "post_normalized_precision_at_10": norm_prec,
                "oracle_recall_at_10": oracle_recall_at_10,
                "oracle_normalized_precision_at_10": oracle_norm_prec_at_10,
                "avg_relevant_api_score": avg_rel_api_score,
                "avg_irrelevant_api_score": avg_irr_api_score,
                "reranker_applied": reranker_diag.get("applied", False) if reranker_diag else False,
                "reranker_reason": reranker_diag.get("reason", None) if reranker_diag else None,
                "category": category,
            }
            # --- end diagnostics categorize ---

            # Build full ranked result list for this question
            ranked_list = []
            for rank, r in enumerate(results[:10], 1):
                ranked_list.append({
                    "rank": rank,
                    "chunk_id": r.chunk_id,
                    "article_id": r.article_id,
                    "benchmark_id": r.metadata.get("benchmark_id", ""),
                    "score": r.score,
                    "source": r.source,
                    "unit_type": r.unit_type,
                    "document_id": r.metadata.get("document_id", ""),
                    "document_type": r.metadata.get("document_type", ""),
                    "status": r.metadata.get("status", ""),
                    "topic_name": r.metadata.get("topic_name", ""),
                    "chapter_title": r.metadata.get("chapter_title", ""),
                    "content": r.content,
                    "raw_content": r.raw_content,
                })

            per_question.append({
                "qid": q["qid"],
                "question": q["question"][:100],
                "relevant_count": len(relevant),
                "retrieved_count": len(results),
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_10": hit_at_10,
                "precision_at_1": precision_at_1,
                "precision_at_3": precision_at_3,
                "precision_at_10": precision_at_10,
                "mrr_at_10": mrr,
                "ndcg_at_10": ndcg,
                "r_precision": r_prec,
                "normalized_precision_at_10": norm_prec,
                "ideal_precision_at_10": ideal_prec,
                "hit": hit_at_10,
                "overlap": list(overlap_at_10)[:5],
                "reranker_diagnostics": rd_entry,
                "ranked_results": ranked_list,
            })

            if (i + 1) % 100 == 0:
                print(
                    f"  [{i + 1}/{len(questions)}] "
                    f"R@1: {hits_at_1 / (i + 1):.4f}  "
                    f"R@3: {hits_at_3 / (i + 1):.4f}  "
                    f"R@10: {hits_at_10 / (i + 1):.4f}  "
                    f"P@1: {precision_sum_at_1 / (i + 1):.4f}  "
                    f"P@3: {precision_sum_at_3 / (i + 1):.4f}  "
                    f"P@10: {precision_sum_at_10 / (i + 1):.4f}"
                )

        # Save per-question ranked results
        per_question_path = (
            f"benchmarks/results/per_question_ranked_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(per_question_path, "w", encoding="utf-8") as f:
            for item in per_question:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Per-question ranked results saved to {per_question_path}")

        # Save reranker diagnostics (compact per-query)
        reranker_diag_path = (
            f"benchmarks/results/reranker_diagnostics_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(reranker_diag_path, "w", encoding="utf-8") as f:
            for item in per_question:
                f.write(json.dumps(item.get("reranker_diagnostics", {}), ensure_ascii=False) + "\n")
        print(f"Reranker diagnostics saved to {reranker_diag_path}")

        recall_at_1 = hits_at_1 / len(questions) if questions else 0.0
        recall_at_3 = hits_at_3 / len(questions) if questions else 0.0
        recall_at_10 = hits_at_10 / len(questions) if questions else 0.0
        precision_at_1 = precision_sum_at_1 / len(questions) if questions else 0.0
        precision_at_3 = precision_sum_at_3 / len(questions) if questions else 0.0
        precision_at_10 = precision_sum_at_10 / len(questions) if questions else 0.0
        print(f"\n{'=' * 40}")
        print(f"Recall@1:     {recall_at_1:.4f}  ({hits_at_1}/{len(questions)})")
        print(f"Recall@3:     {recall_at_3:.4f}  ({hits_at_3}/{len(questions)})")
        print(f"Recall@10:    {recall_at_10:.4f}  ({hits_at_10}/{len(questions)})")
        print(f"Precision@1:  {precision_at_1:.4f}")
        print(f"Precision@3:  {precision_at_3:.4f}")
        print(f"Precision@10: {precision_at_10:.4f}")
        mrr = mrr_sum / len(questions) if questions else 0.0
        ndcg = ndcg_sum / len(questions) if questions else 0.0
        r_prec_avg = r_precision_sum / len(questions) if questions else 0.0
        norm_prec = normalized_precision_sum_at_10 / len(questions) if questions else 0.0
        ideal_prec_avg = ideal_precision_sum_at_10 / len(questions) if questions else 0.0
        print(f"MRR@10:       {mrr:.4f}")
        print(f"nDCG@10:      {ndcg:.4f}")
        print(f"R-Precision:  {r_prec_avg:.4f}")
        print(f"Norm Prec@10: {norm_prec:.4f}  (theoretical max: {ideal_prec_avg:.4f})")
        if reranker is not None:
            print(f"Reranker calls attempted: {reranker.calls_attempted}")
            print(f"Reranker calls skipped:   {reranker.calls_skipped}")
            print(f"Reranker calls failed:    {reranker.calls_failed}")
        print(f"{'=' * 40}")

        # --- Reranker diagnostic summary ---
        avg_pre_mrr = pre_mrr_sum / len(questions) if questions else 0.0
        avg_pre_ndcg = pre_ndcg_sum / len(questions) if questions else 0.0
        avg_pre_norm_prec = pre_norm_prec_sum / len(questions) if questions else 0.0
        avg_candidate_recall = candidate_recall_sum / len(questions) if questions else 0.0
        avg_oracle_norm_prec = oracle_norm_prec_sum / len(questions) if questions else 0.0
        avg_oracle_recall = oracle_recall_sum / len(questions) if questions else 0.0
        avg_rel_score = reranker_rel_score_sum / reranker_rel_count if reranker_rel_count > 0 else None
        avg_irr_score = reranker_irr_score_sum / reranker_irr_count if reranker_irr_count > 0 else None

        print(f"Pre-Rerank MRR@10:              {avg_pre_mrr:.4f}")
        print(f"Pre-Rerank nDCG@10:             {avg_pre_ndcg:.4f}")
        print(f"Pre-Rerank Norm Prec@10:        {avg_pre_norm_prec:.4f}")
        print(f"Avg Candidate Recall:           {avg_candidate_recall:.4f}")
        print(f"Oracle Recall@10 (candidate):   {avg_oracle_recall:.4f}")
        print(f"Oracle Norm Prec@10 (candidate):{avg_oracle_norm_prec:.4f}")
        if avg_rel_score is not None and avg_irr_score is not None:
            print(f"Reranker relevant avg API score: {avg_rel_score:.4f}")
            print(f"Reranker irrelevant avg API scr: {avg_irr_score:.4f}")
            print(f"Reranker gap (rel - irr):        {avg_rel_score - avg_irr_score:.4f}")
        print(f"Category breakdown: {dict(sorted(category_counts.items()))}")
        print(f"{'=' * 40}")

        result = {
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_10": recall_at_10,
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_10": precision_at_10,
            "total_questions": len(questions),
            "hits_at_1": hits_at_1,
            "hits_at_3": hits_at_3,
            "hits_at_10": hits_at_10,
            "hits": hits_at_10,
            "precision_sum_at_1": precision_sum_at_1,
            "precision_sum_at_3": precision_sum_at_3,
            "precision_sum_at_10": precision_sum_at_10,
            "mrr_at_10": mrr,
            "ndcg_at_10": ndcg,
            "r_precision": r_prec_avg,
            "normalized_precision_at_10": norm_prec,
            "ideal_precision_at_10": ideal_prec_avg,
            "reranker_diagnostics": {
                "pre_mrr_at_10": avg_pre_mrr,
                "pre_ndcg_at_10": avg_pre_ndcg,
                "pre_normalized_precision_at_10": avg_pre_norm_prec,
                "avg_candidate_recall": avg_candidate_recall,
                "oracle_recall_at_10": avg_oracle_recall,
                "oracle_normalized_precision_at_10": avg_oracle_norm_prec,
                "avg_relevant_api_score": avg_rel_score,
                "avg_irrelevant_api_score": avg_irr_score,
                "reranker_gap": (avg_rel_score - avg_irr_score) if avg_rel_score is not None and avg_irr_score is not None else None,
                "category_counts": dict(sorted(category_counts.items())),
            },
            "per_question": per_question,
        }
        if reranker is not None:
            result.update({
                "reranker_calls_attempted": reranker.calls_attempted,
                "reranker_calls_skipped": reranker.calls_skipped,
                "reranker_calls_failed": reranker.calls_failed,
            })

        total_elapsed_sec = time.perf_counter() - run_start
        print(f"\nTotal elapsed: {total_elapsed_sec:.2f}s")
        print(f"Avg retrieval: {total_elapsed_sec / len(questions) * 1000:.1f} ms/q ({len(questions) / total_elapsed_sec:.1f} q/s)")

        # Build run log
        run_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_questions": len(questions),
            "elapsed_sec": round(total_elapsed_sec, 2),
            "chunk_count": getattr(self, "_chunk_count", None),
            "vector_store_path": f"{self.config.vector_store.path}_bench",
            "metrics": {
                "recall_at_1": round(recall_at_1, 6),
                "recall_at_3": round(recall_at_3, 6),
                "recall_at_10": round(recall_at_10, 6),
                "precision_at_1": round(precision_at_1, 6),
                "precision_at_3": round(precision_at_3, 6),
                "precision_at_10": round(precision_at_10, 6),
                "mrr_at_10": round(mrr, 6),
                "ndcg_at_10": round(ndcg, 6),
                "r_precision": round(r_prec_avg, 6),
                "normalized_precision_at_10": round(norm_prec, 6),
                "ideal_precision_at_10": round(ideal_prec_avg, 6),
                "pre_mrr_at_10": round(avg_pre_mrr, 6),
                "pre_ndcg_at_10": round(avg_pre_ndcg, 6),
                "pre_normalized_precision_at_10": round(avg_pre_norm_prec, 6),
                "avg_candidate_recall": round(avg_candidate_recall, 6),
                "oracle_recall_at_10": round(avg_oracle_recall, 6),
                "oracle_normalized_precision_at_10": round(avg_oracle_norm_prec, 6),
                "reranker_gap": round(avg_rel_score - avg_irr_score, 6) if avg_rel_score is not None and avg_irr_score is not None else None,
            },
            "reranker_diagnostics": {
                "category_counts": dict(sorted(category_counts.items())),
            },
            "config": {
                "chunking": {
                    "max_tokens": self.config.chunking.max_tokens,
                    "min_tokens": self.config.chunking.min_tokens,
                    "overlap_tokens": self.config.chunking.overlap_tokens,
                },
                "retrieval": {
                    "top_k": self.config.retrieval.top_k,
                    "candidate_pool_size": self.config.retrieval.candidate_pool_size,
                    "similarity_threshold": self.config.retrieval.similarity_threshold,
                    "vector_weight": self.config.retrieval.vector_weight,
                    "metadata_boost": self.config.retrieval.metadata_boost,
                    "metadata_filtering_enabled": self.config.retrieval.enable_metadata_filtering,
                },
                "bm25": {
                    "enabled": self.config.bm25.enabled,
                    "fusion_method": self.config.bm25.fusion_method,
                    "fusion_weight": self.config.bm25.fusion_weight,
                    "k1": self.config.bm25.k1,
                    "b": self.config.bm25.b,
                    "top_k": self.config.bm25.top_k,
                    "rrf_k": self.config.bm25.rrf_k,
                    "rrf_vector_weight": self.config.bm25.rrf_vector_weight,
                    "rrf_bm25_weight": self.config.bm25.rrf_bm25_weight,
                },
                "reranker": {
                    "enabled": self.config.reranker.enabled,
                    "mode": self.config.reranker.bge.mode,
                    "model": self.config.reranker.bge.model,
                    "max_candidates": self.config.reranker.bge.max_candidates,
                    "top_n": self.config.reranker.bge.top_n,
                    "blend_weight": self.config.reranker.bge.blend_weight,
                    "conditional_enabled": self.config.reranker.bge.conditional.enabled,
                    "conditional_strategy": self.config.reranker.bge.conditional.strategy,
                },
                "tools": {
                    "query_rewriter_enabled": self.config.query_rewriter.enabled,
                    "context_enabled": self.config.context.enabled,
                    "context_use_llm": self.config.context.use_llm,
                    "embedding_backend": self.config.embedding.backend,
                    "embedding_model": (
                        self.config.embedding.api.model
                        if self.config.embedding.backend == "api"
                        else self.config.embedding.model_name
                    ),
                    "vector_store_type": self.config.vector_store.type,
                },
            },
        }
        if reranker is not None:
            run_log["reranker_calls"] = {
                "attempted": reranker.calls_attempted,
                "skipped": reranker.calls_skipped,
                "failed": reranker.calls_failed,
            }

        log_path = "benchmarks/results/benchmark_runs.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_log, ensure_ascii=False) + "\n")
        print(f"Run summary appended to {log_path}")
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
            self._chunk_count = store.count()
            print("\n[skip] Store already has data — reusing existing index")
            print(f"  {self._chunk_count} chunks in {store_path}")
            articles = load_articles(
                self.config.data.path, id_map_path=id_map_path,
            )
            bm25 = None
            if self.config.bm25.enabled:
                bm25 = BM25Retriever.load(store_path + "/bm25_index.pkl")
                if bm25 is not None and not bm25.has_benchmark_metadata():
                    print("  [rebuild] BM25 index missing metadata — rebuilding from chunks cache…")
                    import pickle as _pickle
                    cache_path = Path("artifacts/cache/chunks_cache.pkl")
                    if cache_path.exists():
                        with open(cache_path, "rb") as f:
                            chunks = _pickle.load(f)
                    else:
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
        self._chunk_count = len(chunks)
        print(f"  Produced {self._chunk_count} chunks in {t1 - t0:.1f}s")

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

    # -- metric helpers -------------------------------------------------------

    @staticmethod
    def _first_relevant_rank(
        ranked_ids: list[str], relevant: set[str], k: int = 10
    ) -> int:
        for i, bid in enumerate(ranked_ids[:k]):
            if bid in relevant:
                return i + 1
        return 0

    @staticmethod
    def _calc_mrr(ranked_ids: list[str], relevant: set[str]) -> float:
        rank = Benchmark._first_relevant_rank(ranked_ids, relevant)
        return 1.0 / rank if rank > 0 else 0.0

    @staticmethod
    def _calc_ndcg(ranked_ids: list[str], relevant: set[str]) -> float:
        seen = set()
        dcg = 0.0
        for i, bid in enumerate(ranked_ids[:10]):
            if bid in relevant and bid not in seen:
                seen.add(bid)
                dcg += 1.0 / math.log2(i + 2)
        ideal_count = min(len(relevant), 10)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _calc_r_precision(ranked_ids: list[str], relevant: set[str]) -> float:
        r = len(relevant)
        if r == 0:
            return 0.0
        top_r_unique = set()
        for bid in ranked_ids[:r]:
            if bid in relevant:
                top_r_unique.add(bid)
        return len(top_r_unique) / r

    @staticmethod
    def _calc_normalized_precision(
        ranked_ids: list[str], relevant: set[str], k: int
    ) -> float:
        r = len(relevant)
        max_possible = min(r, k)
        if max_possible == 0:
            return 0.0
        found = len(set(ranked_ids[:k]) & relevant)
        return found / max_possible


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
        package_dir = Path(__file__).resolve().parent.parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate
    config = load_config(str(config_path))

    # Resolve paths relative to config
    config_dir = config_path.parent
    if not Path(config.data.path).is_absolute():
        config.data.path = str(config_dir / config.data.path)
    if not Path(config.vector_store.path).is_absolute():
        config.vector_store.path = str(config_dir / config.vector_store.path)
    if not Path(config.bm25.index_path).is_absolute():
        config.bm25.index_path = str(config_dir / config.bm25.index_path)

    benchmark = Benchmark(config)
    result = benchmark.run(
        questions_path=args.questions,
        id_map_path=args.id_map,
        max_questions=args.max_questions,
        force_rechunk=args.rechunk,
    )

    out_path = Path("benchmarks/results/benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
