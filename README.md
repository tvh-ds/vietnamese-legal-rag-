# Vietnamese Legal RAG — Chunking + Retrieval Pipeline

A production-ready pipeline for chunking Vietnamese legal documents (VBQPPL), generating embeddings, and performing hybrid retrieval (dense + BM25 + BGE reranker) for RAG-powered legal Q&A.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/tvh-ds/vietnamese-legal-rag-.git
cd vietnamese-legal-rag-
pip install -r rag_pipeline/requirements.txt
```

### 2. Set API keys

Two separate API keys are needed:

```powershell
# PowerShell
$env:FPT_API_KEY = "sk-your-embedding-key"
$env:FPT_RERANKER_API_KEY = "sk-your-reranker-key"
```

```bash
# Bash
export FPT_API_KEY="sk-your-embedding-key"
export FPT_RERANKER_API_KEY="sk-your-reranker-key"
```

### 3. Add your data

Place the following files in the project root (see .gitignore):

- `Correct ID.json` — article content → benchmark ID mapping (50 MB)
- `1238_question_map_phap_dien.json` — 1,238 benchmark questions

JSON data files go in `data/phap_dien_dataset_45_chu_de/` (already configured).

### 4. Run the benchmark

```bash
python rag_pipeline/benchmark.py \
  --config rag_pipeline/config.yaml \
  --questions 1238_question_map_phap_dien.json \
  --id-map "Correct ID.json"
```

This uses the precomputed embeddings in `chroma_db_bench/` — no indexing needed.
Reports Recall@1, Recall@3, and Recall@10.

### 5. Search the index interactively

```bash
python rag_pipeline/main.py -c rag_pipeline/config.yaml search "thủ tục đấu giá tài sản"
```

---

## Project structure

```
.
├── README.md
├── REPORT.md                           # Latest benchmark results
├── data/                               # Input JSON files (your data)
│   ├── chu_de_1.json … chu_de_45.json
├── rag_pipeline/
│   ├── config.yaml                     # All tunable parameters
│   ├── config.py                       # Config loader (env var resolution)
│   ├── main.py                         # CLI: index / search
│   ├── benchmark.py                    # Recall@1 / @3 / @10 evaluation
│   │
│   ├── data_loader.py                  # JSON parser + benchmark ID mapping
│   ├── chunker.py                      # Recursive semantic chunker + disk cache
│   ├── context_generator.py            # LLM-powered article summarizer
│   ├── embedder.py                     # Local + API embedding backends
│   ├── vector_store.py                 # ChromaDB persistent store
│   ├── bm25_retriever.py              # BM25 + pyvi Vietnamese word segmentation
│   ├── query_rewriter.py              # Vietnamese query normalization
│   ├── retriever.py                   # Hybrid: vector + BM25 + rerank
│   ├── reranker.py                    # BGE API reranker
│   ├── tune_bm25.py                   # BM25 parameter grid search
│   ├── pipeline_types.py              # Shared dataclasses
│   └── requirements.txt               # Python dependencies
├── chroma_db_bench/                    # Precomputed vector store + BM25 index
├── chunks_cache.pkl                    # Precomputed chunk cache
└── .gitignore
```

---

## Data input

### VBQPPL parsed JSON (`data/chu_de_*.json`)

Used by `data_loader.py`. Hierarchical structure: topic → section → chapter → article.

```json
[
  {
    "Chủ đề ID": "<uuid>",
    "Tên chủ đề": "Bổ trợ tư pháp",
    "Chương ID": "<uuid>",
    "Tên chương": "Chương I NHỮNG QUY ĐỊNH CHUNG",
    "Các điều": [
      {
        "Điều ID": "D45AE0D8-...",
        "Tên điều": "Điều 4.1.LQ.1. Phạm vi điều chỉnh",
        "Nội dung": "Luật này quy định về...",
        "Ghi chú": { "metadata": [{ "Hiệu lực": "...", "Thi hành": "CÓ_HIỆU_LỰC_THI_HÀNH" }] },
        "Chỉ dẫn": [{ "Điều liên quan": "Điều 28.3.LQ.82...", "ID Điều liên quan": "..." }]
      }
    ]
  }
]
```

### Benchmark ID mapping (`Correct ID.json`)

A dict of `article_content → numeric_id` (31,721 entries). During loading, each article's `Nội dung` is matched against this dict by exact content comparison. Matched articles get a `benchmark_id` stored in ChromaDB metadata — **never embedded**. This ID is only used to match retrieved chunks against `relevant_laws` during evaluation.

### Adding new data

Point the config at your directory, then run with `--rechunk`:

```yaml
data:
  path: "../my_data"
```

---

## Full pipeline — phase by phase

| Phase                           | File                     | What happens                                                                                                                                                       |
| ------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Structure parsing**  | `data_loader.py`       | JSON walker — flattens topic → section → chapter → article hierarchy. Extracts metadata, cross-references, and benchmark IDs via content matching.             |
| **2. Recursive chunking** | `chunker.py`           | Recursive Clause → Point → Paragraph → Sentence splitting. Merge short chunks, deduplicate, preserve sentence boundaries. Overlap tokens prevent boundary cuts. |
| **3. Context injection**  | `context_generator.py` | 60–120 token Vietnamese legal summary generated per article (LLM or heuristic fallback), prepended to every chunk before embedding.                               |
| **4. Embedding**          | `embedder.py`          | Company`Vietnamese_Embedding` API (1024-dim) or local `sentence-transformers`. Batched 64/chunk. 3-retry exponential backoff.                                  |
| **5. Vector store**       | `vector_store.py`      | ChromaDB — persistent, cosine-distance, HNSW-indexed.                                                                                                             |
| **6. BM25 index**         | `bm25_retriever.py`    | Okapi BM25 with pyvi Vietnamese word segmentation (`k1=1.2`, `b=0.75`). Index persisted alongside ChromaDB.                                                    |
| **7. Retrieval**          | `retriever.py`         | Hybrid: vector cosine + BM25 sparse → fusion → reranker → top-10.                                                                                               |
| **8. Benchmarking**       | `benchmark.py`         | Recall@1 / @3 / @10 on 1,238 questions.                                                                                                                            |

### Retrieval (Phase 7 — per query)

| Stage                     | What happens                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------- |
| **Query rewriting** | Expands abbreviations, appends legal-domain context for short queries                   |
| **Vector search**   | Cosine similarity top-50 via ChromaDB                                                   |
| **BM25 search**     | Okapi BM25 top-50 via rank-bm25                                                         |
| **Score fusion**    | `(1 - fusion_weight) × vector + fusion_weight × BM25` (tuned at `0.05`)           |
| **BGE reranker**    | `POST /v1/rerank` — sorts by BGE relevance score with optional blend of hybrid score |
| **Final top-k**     | Returns 10 highest-scoring chunks                                                       |

### Benchmarking (Phase 8)

Queries are pre-embedded in one API batch. For each question: retrieve top-10 chunks, check if any chunk's `benchmark_id` matches the question's `relevant_laws`. Reports hits / total at each cutoff.

**Current best (without BGE reranker):** `R@1=0.7763`, `R@3=0.9208`, `R@10=0.9693`

---

## Configuration reference

Key tunable parameters in `config.yaml`:

### `reranker.bge`

| Key                | Default                     | Description                                     |
| ------------------ | --------------------------- | ----------------------------------------------- |
| `enabled`        | `true`                    | Enable API BGE reranking                        |
| `model`          | `bge-reranker-v2-m3`      | Reranker model name                             |
| `api_key`        | `${FPT_RERANKER_API_KEY}` | Separate key from embedding                     |
| `endpoint`       | `/v1/rerank`              | API endpoint path (uses shared`api.base_url`) |
| `max_candidates` | `10`                      | Candidates sent to the reranker per query       |
| `top_n`          | `10`                      | Results returned by the API                     |
| `blend_weight`   | `0.05`                    | Small BGE correction blended with hybrid score  |

### `reranker.bge.conditional`

Conditional reranking skips BGE when hybrid retrieval is already confident. This reduces API calls and avoids unnecessary reranker noise.

| Key                | Current           | Description                                           |
| ------------------ | ----------------- | ----------------------------------------------------- |
| `enabled`        | `true`          | Enable score-gap based conditional reranking          |
| `strategy`       | `top1_top2_gap` | Rerank only when rank 1 and rank 2 are close          |
| `top1_top2_gap`  | `0.03`          | Call BGE when`score_1 - score_2 < threshold`        |
| `top1_top10_gap` | `0.08`          | Alternative threshold for`strategy: top1_top10_gap` |
| `min_candidates` | `2`             | Minimum candidates required for conditional decision  |

### `bm25`

| Key               | Current  | Description                                            |
| ----------------- | -------- | ------------------------------------------------------ |
| `k1`            | `1.2`  | Term frequency saturation                              |
| `b`             | `0.75` | Document length normalization                          |
| `fusion_weight` | `0.05` | BM25 weight in hybrid (0 = pure vector, 1 = pure BM25) |
| `top_k`         | `50`   | Candidates from BM25 retrieval                         |

### `retrieval`

| Key                           | Current   | Description                             |
| ----------------------------- | --------- | --------------------------------------- |
| `candidate_pool_size`       | `50`    | Pool size before reranking              |
| `top_k`                     | `10`    | Final results per query                 |
| `enable_metadata_filtering` | `false` | Metadata post-filtering (status, topic) |
Full reference: comments in `config.yaml`.

---

## Reranker API

The BGE reranker is called via:

```text
POST {api.base_url}/v1/rerank
Authorization: Bearer {FPT_RERANKER_API_KEY}
Content-Type: application/json
```

**Request:**

```json
{
  "model": "bge-reranker-v2-m3",
  "query": "question text",
  "documents": ["chunk content 1", "chunk content 2"],
  "top_n": 10
}
```

**Response:**

```json
{
  "results": [
    {"index": 2, "relevance_score": 0.97},
    {"index": 0, "relevance_score": 0.42}
  ]
}
```

Also supports `{"scores": [...]}`, `{"data": [{"index": ..., "score": ...}]}`.

---

## Chunk output schema

| Field              | Type | Example                                   |
| ------------------ | ---- | ----------------------------------------- |
| `chunk_id`       | UUID | `017eb4cf-...`                          |
| `article_id`     | str  | `D45AE0D8-...`                          |
| `unit_type`      | enum | `ARTICLE` / `CLAUSE` / `POINT`      |
| `content`        | str  | `"Điều: Nguyên tắc bầu cử.\n..."` |
| `token_count`    | int  | `87`                                    |
| `hierarchy_path` | str  | `article_id/clause_1/point_b`           |
| `document_id`    | str  | `85/2015/QH13`                          |
| `status`         | str  | `CÓ_HIỆU_LỰC_THI_HÀNH`              |
| `topic_name`     | str  | `Tổ chức bộ máy nhà nước`        |

---

## Offline pipeline (from scratch)

```bash
# Full re-index (rebuild chunks + embeddings + BM25)
python rag_pipeline/benchmark.py \
  --config rag_pipeline/config.yaml \
  --questions 1238_question_map_phap_dien.json \
  --id-map "Correct ID.json" \
  --rechunk

# BM25 parameter grid search (no reranker during tuning)
python rag_pipeline/tune_bm25.py \
  --config rag_pipeline/config.yaml \
  --questions 1238_question_map_phap_dien.json \
  --id-map "Correct ID.json"
```
