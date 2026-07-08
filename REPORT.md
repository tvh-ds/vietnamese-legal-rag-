# Vietnamese Legal RAG — Pipeline Report

## Dataset

| Property | Value |
|----------|-------|
| Source | 45 JSON files (`chu_de_1.json` … `chu_de_45.json`) |
| Format | VBQPPL parsed: topic → section → chapter → article |
| Total articles | **76,459** |
| Articles with benchmark IDs | **32,768** (content-matched via `Correct ID.json`) |
| Benchmark questions | 1,238 (from `1238_question_map_phap_dien.json`) |
| Unique relevant law IDs | 1,325 |

---

## Result

| Metric | Value |
|--------|-------|
| **Recall@10** | **0.9257** (1,146 / 1,238) |
| LLM context injection | Disabled (heuristic: title + first sentence) |

---

## Pipeline Architecture

### Phase 1 — Structure Parsing

| Component | Tool | Details |
|-----------|------|---------|
| Data loader | `data_loader.py` | Flattens topic→section→chapter→article hierarchy |
| Benchmark ID mapping | Content matching | `Correct ID.json` dict: `article_text → numeric_id` |
| ID match rate | 32,768 / 76,459 | All 1,325 benchmark-relevant IDs covered |

**Extracted metadata per article:**

| Field | Source |
|-------|--------|
| `article_id` (UUID) | `Điều ID` |
| `title` | `Tên điều` |
| `content` | `Nội dung` |
| `document_id` | Regex from `Ghi chú` text |
| `document_type` | `Luật` / `Nghị định` / `Thông tư` |
| `effective_date` | `metadata[Hiệu lực]` |
| `status` | `metadata[Thi hành]` |
| `topic_id`, `topic_name` | Topic hierarchy |
| `benchmark_id` | Content-matched from `Correct ID.json` |

---

### Phase 2 — Recursive Semantic Chunking

| Component | Tool | Details |
|-----------|------|---------|
| Tokenizer | HuggingFace `AutoTokenizer` | Matches embedding model for accurate counts |
| Chunking algorithm | `chunker.py` | Recursive Clause → Point → Paragraph → Sentence |
| Clause pattern | Regex `^\d+\.\s` | Numbered clauses (Khoản) |
| Point pattern | Regex `^[a-đ]\)\s` | Lettered points (Điểm) |
| Paragraph split | Double-newline boundary | Late chunking fallback |
| Sentence split | Vietnamese sentence boundaries | Final fallback |

**Chunking parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_tokens` | 1,024 | Split threshold |
| `min_tokens` | 30 | Merge shorter chunks with adjacent |
| `overlap_tokens` | 50 | Token overlap at chunk boundaries |
| Chunk cache | `chunks_cache.pkl` | Disk-persisted for instant reload |

**Chunk unit types:** ARTICLE, CLAUSE, POINT, PARAGRAPH, SENTENCE

**Chunk output fields (stored in ChromaDB):**

| Field | Type | Embedded? |
|-------|------|-----------|
| `content` (context + raw) | text | ✅ Yes |
| `chunk_id` | UUID | ❌ Metadata |
| `article_id` | UUID | ❌ Metadata |
| `unit_type` | enum | ❌ Metadata |
| `hierarchy_path` | string | ❌ Metadata |
| `document_id`, `document_type` | string | ❌ Metadata |
| `effective_date`, `status` | string | ❌ Metadata |
| `topic_id`, `topic_name` | string | ❌ Metadata |
| `benchmark_id` | string | ❌ Metadata (evaluation only) |

---

### Phase 3 — Context Generation

| Component | Tool | Details |
|-----------|------|---------|
| Method | Heuristic | Article title + first sentence |
| LLM option | `gemma-4-31B-it` via API | Disabled for this run (`use_llm: false`) |
| Per-chunk prepend | `[context] + [chunk_content]` | Injected before embedding |

---

### Phase 4 — Embedding

| Component | Tool | Details |
|-----------|------|---------|
| Backend | Company API (`requests`) | `POST /v1/embeddings` |
| Model | **Vietnamese_Embedding** | 1024-dim vectors |
| Base URL | `https://mkp-api.fptcloud.com` | OpenAI-compatible endpoint |
| Authentication | Bearer token | `$FPT_API_KEY` environment variable |
| Max sequence length | 2,048 tokens | Input chunks ≤ 1,024 fit comfortably |
| Batch size | 64 texts per API call | |
| Retry logic | 3 attempts, exponential backoff | Handles transient network failures |
| Progress | `tqdm` progress bar | Per-batch tracking |

**Embedding isolation:** The `benchmark_id` field is stored in ChromaDB metadata only. The embedding model never receives it. Retrieval never uses it. It is only read at the final evaluation step to compute Recall@10.

---

### Phase 5 — Vector Storage

| Component | Tool | Details |
|-----------|------|---------|
| Database | **ChromaDB** (persistent) | SQLite + binary segment files |
| Index type | HNSW | Approximate nearest neighbor |
| Distance metric | Cosine | Vectors are L2-normalized |
| Collection | `bench_data` | Stored in `chroma_db_bench/` |

---

### Phase 6 — BM25 Sparse Index

| Component | Tool | Details |
|-----------|------|---------|
| Library | **rank-bm25** | Okapi BM25 implementation |
| Tokenizer | **pyvi** | Vietnamese word segmentation |
| `k1` | 1.5 | Term frequency saturation |
| `b` | 0.75 | Document length normalization |
| Persistence | Pickle file | Saved alongside ChromaDB |
| Compound words | `đấu_giá`, `giấy_phép`, `xây_dựng` | pyvi handles correctly |

---

### Phase 7 — Query Rewriting

| Component | Tool | Details |
|-----------|------|---------|
| Abbreviation expansion | `query_rewriter.py` | 80+ legal abbreviations (`dk` → `điều kiện`, `bhxh` → `bảo hiểm xã hội`) |
| Domain context | Append phrase | `"theo pháp luật hiện hành"` for short queries |

---

### Phase 8 — Hybrid Retrieval (per query)

#### 8a. Vector Search

| Parameter | Value |
|-----------|-------|
| Candidate pool | `top_k × 2 = 20` |
| Similarity metric | Cosine (1 − distance/2) |

#### 8b. BM25 Search (parallel)

| Parameter | Value |
|-----------|-------|
| Candidate pool | 20 |
| Scoring | Okapi BM25 (`k1=1.5`, `b=0.75`) |

#### 8c. Score Fusion

| Parameter | Value |
|-----------|-------|
| Fusion method | Weighted sum |
| `fusion_weight` | 0.5 (50% vector, 50% BM25) |
| BM25-only candidates | Added with `bm25_score × 0.5` |

#### 8d. Metadata Boost

| Rule | Boost |
|------|-------|
| `status` contains `CÓ_HIỆU_LỰC` | +0.10 |
| `unit_type == ARTICLE` | +0.04 |
| `unit_type == CLAUSE` | +0.02 |

#### 8e. Reranker — MMR Diversity

| Component | Tool | Details |
|-----------|------|---------|
| Algorithm | Maximal Marginal Relevance | Balances relevance + diversity |
| Parameter | Value | Description |
| `lambda_mmr` | 0.7 | 70% relevance, 30% diversity |
| `keyword_boost` | 0.15 | Jaccard overlap bonus |
| `diversity_weight` | 0.15 | Penalty for similarity to already-selected chunks |
| Similarity metric | Jaccard on token sets | Vietnamese-word-aware comparison |

---

### Phase 9 — Benchmark Evaluation

| Component | Details |
|-----------|---------|
| Metric | Recall@10 |
| Formula | `hits / total_questions` |
| Hit definition | At least 1 of top-10 chunks has `benchmark_id` matching `relevant_laws` |
| Query embedding | All 1,238 queries batched in one API call |
| Retrieval per query | Full pipeline (vector + BM25 + boost + MMR) |

---

## Complete Parameter Summary

| Category | Parameter | Value |
|----------|-----------|-------|
| **Chunking** | `max_tokens` | 1,024 |
| | `min_tokens` | 30 |
| | `overlap_tokens` | 50 |
| **Embedding** | Backend | API |
| | Model | `Vietnamese_Embedding` |
| | Dimension | 1,024 |
| | Max seq length | 2,048 |
| | API batch size | 64 |
| | Retries | 3 (exponential backoff) |
| **Vector Store** | Database | ChromaDB (HNSW) |
| | Distance | Cosine |
| **BM25** | `k1` | 1.5 |
| | `b` | 0.75 |
| | `fusion_weight` | 0.5 |
| | Candidates | 20 |
| **Retrieval** | `top_k` | 10 |
| | `similarity_threshold` | 0.0 |
| | `vector_weight` | 0.7 |
| | `metadata_boost` | 0.2 |
| **Reranker** | `lambda_mmr` | 0.7 |
| | `keyword_boost` | 0.15 |
| | `diversity_weight` | 0.15 |
| **Context** | `use_llm` | false (heuristic) |
| **Query Rewriter** | `enabled` | true |
| | `expand_abbreviations` | true |
| | `append_context` | true |

---

## Recall@10 Progression

| Questions | Recall@10 |
|-----------|-----------|
| 100 | 0.9500 |
| 200 | 0.9400 |
| 300 | 0.9267 |
| 400 | 0.9200 |
| 500 | 0.9180 |
| 600 | 0.9183 |
| 700 | 0.9286 |
| 800 | 0.9250 |
| 900 | 0.9278 |
| 1,000 | 0.9290 |
| 1,100 | 0.9291 |
| 1,200 | 0.9275 |
| **1,238** | **0.9257** |


## Hyperparameters Tuning

| Parameter | Current Value | Range | Location |
|-----------|---------------|-------|----------|
| max_tokens | 1024 | 256–2048 | config.yaml |
| min_tokens | 30 | 10–100 | config.yaml |
| overlap_tokens | 50 | 0–200 | config.yaml |
| fusion_weight (BM25) | 0.5 | 0.0–1.0 | config.yaml |
| bm25.k1 | 1.5 | 0.5–2.0 | config.yaml |
| bm25.b | 0.75 | 0.0–1.0 | config.yaml |
| bm25.top_k (candidates) | 20 | 10–50 | config.yaml |
| retrieval.top_k | 10 | 5–30 | config.yaml |
| similarity_threshold | 0.0 | 0.0–0.5 | config.yaml |
| vector_weight | 0.7 | 0.0–1.0 | config.yaml |
| metadata_boost | 0.2 | 0.0–0.5 | config.yaml |
| reranker.lambda_mmr | 0.7 | 0.0–1.0 | config.yaml |
| reranker.keyword_boost | 0.15 | 0.0–0.3 | config.yaml |
| reranker.diversity_weight | 0.15 | 0.0–0.3 | config.yaml |
| context.use_llm | false | true/false | config.yaml |
| context.llm.temperature | 0.3 | 0.0–1.0 | config.yaml |
| context.max_context_tokens | 100 | 60–200 | config.yaml |
| query_rewriter.enabled | true | true/false | config.yaml |
| query_rewriter.append_context | true | true/false | config.yaml |
| Embedding model dim | 1024 | — | Fixed by API |