# Vietnamese Legal RAG — Chunking + Retrieval Pipeline

A production-ready pipeline for chunking Vietnamese legal documents (VBQPPL), generating embeddings, and performing hybrid retrieval (dense vector + BM25 weighted fusion + optional BGE reranker) for RAG-powered legal Q&A.

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

Place the following files in `benchmarks/input/` (see .gitignore):

- `benchmarks/input/Correct ID.json` — article content → benchmark ID mapping
- `benchmarks/input/1238_question_map_phap_dien.json` — 1,238 benchmark questions

JSON data files go in `data/phap_dien_dataset_45_chu_de/` (already configured).

### 4. Run the benchmark

```bash
python -m rag_pipeline.cli.benchmark \
  --config rag_pipeline/config.yaml \
  --questions benchmarks/input/1238_question_map_phap_dien.json \
  --id-map "benchmarks/input/Correct ID.json" \
  --rechunk
```

This uses or creates precomputed embeddings in `artifacts/vector_store_bench/` — no indexing needed on repeated runs.

Output includes:
- Recall@1, Recall@3, Recall@10
- Precision@1, Precision@3, Precision@10
- MRR@10, nDCG@10, R-Precision, Normalized Precision@10
- Reranker diagnostics (calls, API score gap, per-query categorization)
- Per-question ranked results written to `benchmarks/results/`

### 5. Search the index interactively

```bash
python -m rag_pipeline.cli.main -c rag_pipeline/config.yaml search "thủ tục đấu giá tài sản"
```

---

## Project structure

```
.
├── README.md
├── REPORT.md                             # Latest benchmark results
├── data/                                 # Input JSON files (your data)
│   ├── chu_de_1.json … chu_de_45.json
├── rag_pipeline/
│   ├── __init__.py
│   ├── config.yaml                       # All tunable parameters
│   │
│   ├── cli/                              # CLI entry points
│   │   ├── main.py                       #   index / search
│   │   └── benchmark.py                  #   Full evaluation suite
│   │
│   ├── config/                           # Configuration
│   │   ├── config.py                     #   Dataclasses + YAML loader (env var resolution)
│   │   └── __init__.py
│   │
│   ├── data/                             # Data loading
│   │   ├── data_loader.py                #   JSON parser + benchmark ID mapping
│   │   └── __init__.py
│   │
│   ├── indexing/                         # Indexing pipeline
│   │   ├── chunker.py                    #   Recursive semantic chunker + disk cache
│   │   ├── context_generator.py          #   Heuristic or LLM article summarizer
│   │   ├── embedder.py                   #   Local + API embedding backends
│   │   ├── vector_store.py               #   ChromaDB persistent store
│   │   └── __init__.py
│   │
│   ├── retrieval/                        # Retrieval pipeline
│   │   ├── bm25_retriever.py            #   BM25 + pyvi Vietnamese word segmentation
│   │   ├── query_rewriter.py            #   Vietnamese query normalization
│   │   ├── reranker.py                  #   BGE API reranker (two modes)
│   │   ├── retriever.py                 #   Hybrid: vector + BM25 + rerank
│   │   └── __init__.py
│   │
│   ├── shared/                           # Shared types
│   │   ├── pipeline_types.py            #   RetrievalResult dataclass
│   │   └── __init__.py
│   │
│   └── requirements.txt                 # Python dependencies
├── benchmarks/                            # Benchmark artifacts
│   ├── input/                             #   Large input files (user-provided)
│   └── results/                           #   Run logs, per-question JSON, diagnostics
├── artifacts/
│   ├── cache/                             #   Precomputed chunk cache
│   ├── vector_store/                      #   ChromaDB persistent store (rebuild with index)
│   └── vector_store_bench/                #   Benchmark store (auto-created)
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

A dict of `article_content → numeric_id` (±31,000 entries). During loading, each article's `Nội dung` is matched against this dict by exact content comparison. Matched articles get a `benchmark_id` stored in ChromaDB metadata — **never embedded**. This ID is only used to match retrieved chunks against `relevant_laws` during evaluation.

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
| **2. Recursive chunking** | `chunker.py`           | Recursive Clause → Point → Paragraph → Sentence splitting (max 1024 tokens). Merge short chunks, preserve sentence boundaries, overlap tokens prevent cuts.   |
| **3. Context injection**  | `context_generator.py` | Article title + first sentence prepended to every chunk before embedding. LLM option available (`use_llm: true`).                                                  |
| **4. Embedding**          | `embedder.py`          | Company `Vietnamese_Embedding` API (1024-dim) or local `sentence-transformers`. Batched 64/chunk, 3-retry exponential backoff.                                     |
| **5. Vector store**       | `vector_store.py`      | ChromaDB — persistent, cosine-distance, HNSW-indexed.                                                                                                             |
| **6. BM25 index**         | `bm25_retriever.py`    | Okapi BM25 with pyvi Vietnamese word segmentation (`k1=1.2`, `b=0.75`). Index persisted alongside ChromaDB.                                                    |
| **7. Retrieval**          | `retriever.py`         | Hybrid: vector cosine + BM25 sparse → weighted fusion → optional reranker → top-10.                                                                            |
| **8. Benchmarking**       | `benchmark.py`         | Full evaluation on 1,238 questions: Recall, Precision, MRR, nDCG, R-Precision, Normalized Precision, reranker diagnostics.                                       |

### Retrieval (Phase 7 — per query)

| Stage                     | What happens                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------- |
| **Query rewriting** | Expands abbreviations, appends legal-domain context for short queries (disabled by default) |
| **Vector search**   | Cosine similarity top-50 via ChromaDB                                                   |
| **BM25 search**     | Okapi BM25 top-50 via rank-bm25                                                         |
| **Score fusion**    | `(1 - fusion_weight) × vector + fusion_weight × BM25` (tuned at `0.05`)           |
| **BGE reranker**    | Optional API call — two modes: `global_score_sort` or `reorder_window` (see below)    |
| **Final top-k**     | Returns 10 highest-scoring chunks                                                       |

### Reranker modes

The reranker supports two modes via `reranker.bge.mode`:

- **`global_score_sort`** (default if not set): all candidates are globally re-sorted by blended score (BGE API score + hybrid score). Powerful but can degrade top-3/top-10 if the reranker is inconsistent.
- **`reorder_window`**: only the first `max_candidates` candidates are reranked and reordered among themselves. Candidates beyond the window retain their original positions. Safer — preserves top-10 recall at the cost of weaker top-1 improvement.

### Benchmarking (Phase 8)

For each question: retrieve top-10 chunks, check if any chunk's `benchmark_id` matches the question's `relevant_laws`. Reports hits / total at each cutoff plus ranking metrics.

**Current best (no reranker):**

| Metric | Value |
|---|---|
| Recall@1 | 0.7916 |
| Recall@3 | 0.9249 |
| Recall@10 | 0.9693 |
| MRR@10 | 0.8607 |
| nDCG@10 | 0.8436 |
| Norm Precision@10 | 0.9142 |

See `REPORT.md` for detailed comparative results across reranker configurations.

---

## Configuration reference

Key tunable parameters in `config.yaml`:

### `reranker.bge`

| Key | Default | Description |
|---|---|---|
| `mode` | `global_score_sort` | Reranking strategy: `global_score_sort` or `reorder_window` |
| `enabled` | `true` | Enable API BGE reranking |
| `max_candidates` | `3` | Candidates sent to the reranker per query |
| `top_n` | `3` | Results returned by the API |
| `blend_weight` | `1.0` | BGE score weight in blended sort (1.0 = pure BGE) |

### `reranker.bge.conditional`

Conditional reranking skips BGE when hybrid retrieval is already confident.

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable score-gap based conditional reranking |
| `strategy` | `top1_top10_gap` | `top1_top2_gap` or `top1_top10_gap` |
| `top1_top2_gap` | `0.03` | Call BGE when `score_1 - score_2 < threshold` |
| `top1_top10_gap` | `0.08` | Alternative threshold for `strategy: top1_top10_gap` |
| `min_candidates` | `2` | Minimum candidates required for conditional decision |

### `bm25`

| Key | Default | Description |
|---|---|---|
| `k1` | `1.2` | Term frequency saturation |
| `b` | `0.75` | Document length normalization |
| `fusion_weight` | `0.05` | BM25 weight in hybrid (0 = pure vector, 1 = pure BM25) |
| `top_k` | `50` | Candidates from BM25 retrieval |

### `retrieval`

| Key | Default | Description |
|---|---|---|
| `candidate_pool_size` | `50` | Pool size before reranking |
| `top_k` | `10` | Final results per query |
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
  "top_n": 3
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

| Field | Type | Example |
|---|---|---|
| `chunk_id` | UUID | `017eb4cf-...` |
| `article_id` | str | `D45AE0D8-...` |
| `unit_type` | enum | `ARTICLE` / `CLAUSE` / `POINT` |
| `content` | str | `"Điều: Nguyên tắc bầu cử.\n..."` |
| `token_count` | int | `87` |
| `hierarchy_path` | str | `article_id/clause_1/point_b` |
| `document_id` | str | `85/2015/QH13` |
| `status` | str | `CÓ_HIỆU_LỰC_THI_HÀNH` |
| `topic_name` | str | `Tổ chức bộ máy nhà nước` |

---

## Offline pipeline (from scratch)

```bash
# Full re-index (rebuild chunks + embeddings + BM25)
python -m rag_pipeline.cli.benchmark \
  --config rag_pipeline/config.yaml \
  --questions benchmarks/input/1238_question_map_phap_dien.json \
  --id-map "benchmarks/input/Correct ID.json" \
  --rechunk
```
