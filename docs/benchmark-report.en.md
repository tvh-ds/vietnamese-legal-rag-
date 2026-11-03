# Vietnamese Legal Retrieval Benchmark

[Tiếng Việt](benchmark-report.md)

This report covers the **1,238-query reranking experiment** recorded on July 27, 2026. The [project README](../README.md#internship-results) summarizes the separate 8,238-query internship evaluation.

## 1. Configuration comparison

| Configuration | R@1 | R@3 | R@10 | MRR@10 | nDCG@10 | Normalized P@10 | Total (s) | Mean (ms/query) | Queries/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Hybrid without reranking** | **0.7916** | 0.9249 | 0.9693 | **0.8607** | **0.8436** | 0.9142 | 954.65 | 771.12 | **1.30** |
| Global rerank, top 10, blend 0.5 | 0.7746 | 0.8853 | 0.9047 | 0.8293 | 0.8037 | 0.8464 | 2532.90 | 2045.96 | 0.49 |
| Global rerank, top 30, blend 0.5 | 0.7746 | 0.8853 | 0.9079 | 0.8295 | 0.8050 | 0.8505 | 1456.48 | 1176.48 | 0.85 |
| Global rerank, 3 candidates, blend 0.5 | 0.7771 | 0.8772 | 0.9241 | 0.8319 | 0.8111 | 0.8677 | 4365.45 | 3526.21 | 0.28 |
| Reorder top 3, blend 1.0 | 0.7835 | **0.9249** | **0.9693** | 0.8579 | 0.8428 | **0.9142** | 1433.04 | 1157.54 | 0.86 |

Hybrid retrieval without reranking had the highest Recall@1, MRR@10, and nDCG@10, and the lowest latency. Global reranking reduced Recall@3 and Recall@10 across all tested variants. Restricting the reorder window to three candidates preserved Recall@3 and Recall@10, with a small decline in Recall@1 and MRR@10.

## 2. Top-three reorder experiment

The run began at 10:15 on July 27, 2026. It used `mode: reorder_window`, `max_candidates: 3`, `top_n: 3`, and `blend_weight: 1.0`. Conditional reranking was disabled. All 1,238 reranker calls were attempted, with zero skips and zero failures.

| Metric | Result |
| --- | ---: |
| Recall@1 | 0.7835 (970/1,238) |
| Recall@3 | 0.9249 (1,145/1,238) |
| Recall@10 | 0.9693 (1,200/1,238) |
| Precision@1 | 0.7835 |
| Precision@3 | 0.3250 |
| Precision@10 | 0.1090 |
| MRR@10 | 0.8579 |
| nDCG@10 | 0.8428 |
| R-Precision | 0.7326 |
| Normalized Precision@10 | 0.9142 |
| Ideal Precision@10 | 0.1270 |

| Metric | Before reranking | After reranking | Change |
| --- | ---: | ---: | ---: |
| MRR@10 | 0.8607 | 0.8579 | −0.0028 |
| nDCG@10 | 0.8436 | 0.8428 | −0.0008 |
| Normalized Precision@10 | 0.9142 | 0.9142 | 0.0000 |

For the pool of 50 candidates, the run recorded average candidate recall and oracle Recall@10 of **0.9621**. The mean API score was **0.9356** for relevant results and **0.7716** for irrelevant results, a gap of **+0.1640**.

| Query outcome | Count | Meaning |
| --- | ---: | --- |
| Unchanged hit | 1,004 | A relevant result remained in the top 10 with no rank change |
| Rank worsened | 100 | A relevant result stayed in the top 10 but ranked lower |
| Rank improved | 96 | A relevant result stayed in the top 10 and ranked higher |
| Unchanged miss | 25 | No relevant result reached the top 10 |
| Missing from candidates | 13 | The candidate pool contained no relevant result |

## 3. Dataset and evaluation

The benchmark used 45 JSON topic files with 76,459 articles, sourced from Vietnam’s public Pháp điển codification. [Ministry of Justice Pháp điển portal](https://phapdien.moj.gov.vn/Pages/chi-tiet-bo-phap-dien.aspx)

Content matching assigned benchmark IDs to approximately 32,700 articles and covered all 1,325 distinct relevance IDs. The benchmark ID was stored in retrieval metadata for evaluation. The embedding model received passage text and context.

| Relevant laws per query | Queries |
| --- | ---: |
| 1 | 993 |
| 2 | 190 |
| 3 | 34 |
| 4 or more | 21 |

The mean was 1.27 relevant laws per query. The benchmark counted a hit when at least one of the top-k chunks had a benchmark ID listed in the query's relevance labels. It tracked Recall and Precision at 1, 3, and 10, plus MRR@10, nDCG@10, R-Precision, and normalized Precision@10. Reranker diagnostics tracked candidate coverage, score separation, and rank changes per query.

## 4. Pipeline and parameters

| Stage | Configuration in this experiment |
| --- | --- |
| Load and label | Flatten topic → section → chapter → article and map article text to benchmark IDs |
| Chunk | Recursively split through clauses, points, paragraphs, and sentences. Use a 1,024-token maximum, a 30-token minimum, and 50 tokens of overlap |
| Add context | Prepend the article title and first sentence. LLM context generation was disabled |
| Embed | FPT Cloud Vietnamese_Embedding API, 1,024 dimensions, 2,048-token model limit, batches of up to 64 texts, and up to three retries with exponential backoff |
| Store dense index | Persistent ChromaDB with HNSW and cosine distance |
| Build sparse index | Okapi BM25 with PyVi word segmentation, `k1: 1.2`, `b: 0.75`. Persist the tokenized index alongside ChromaDB |
| Retrieve and fuse | Up to 50 candidates per retrieval method. Weighted fusion used 95% dense and 5% sparse scores, with a similarity threshold of 0.0 and 10 returned results |
| Rerank | Optional BGE global sorting or reordering within a fixed window |
| Optional features | Query rewriting and metadata boosting were disabled in the recorded configuration |

The source code separates the CLI, configuration, data loading, indexing, retrieval, and shared types under `rag_pipeline/`. Generated indexes and chunk caches go under `artifacts/`. Evaluation output goes under `benchmarks/results/`.

## 5. Conclusions and follow-up

The earlier BM25 grid search selected a sparse weight of **0.05**. Hybrid retrieval alone reached **0.9693 Recall@10** at about **771 ms per query**. The gap between Recall@1 and Recall@10 motivated work on first-position ranking. Reranking improved 96 query rankings and worsened 100, despite the positive average score gap between relevant and irrelevant passages. The experiment also records that Gemma-based selection from the top three and MMR diversity reranking were removed from the pipeline.

The report proposed testing conditional reranking based on the gap between the top two hybrid scores, expanding the reorder window to five or ten candidates, and reducing the blend weight to 0.2–0.3. It also proposed trying a stronger reranker or fine-tuning one on Vietnamese legal data.
