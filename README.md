# Vietnamese Legal RAG — Chunking + Retrieval Pipeline

A production-ready pipeline for chunking Vietnamese legal documents (VBQPPL), generating embeddings, and performing hybrid retrieval for RAG-powered legal Q&A. Implements the specification in [`Pipeline.md`](Pipeline.md).

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/tvh-ds/vietnamese-legal-rag-.git
cd vietnamese-legal-rag-
pip install -r rag_pipeline/requirements.txt
```

### 2. Set API key

Both embedding and LLM use the same OpenAI-compatible endpoint. Set one environment variable:

```powershell
# PowerShell
$env:FPT_API_KEY = "sk-your-api-key"
```

```bash
# Bash
export FPT_API_KEY="sk-your-api-key"
```

### 3. Add your data

Place JSON files in `data/phap_dien_dataset_45_chu_de/`. See [Data input](#data-input) for format. The config already points there.

Also place these two files in the project root (not committed — gitignored):

- `Correct ID.json` — article content → benchmark ID mapping (50 MB)
- `1238_question_map_phap_dien.json` — 1,238 benchmark questions

### 4. Benchmark (index + evaluate in one command)

```bash
python rag_pipeline/benchmark.py \
  --config rag_pipeline/config.yaml \
  --questions 1238_question_map_phap_dien.json \
  --id-map "Correct ID.json" \
  -n 50
```

This does everything: load → chunk → embed → store → retrieve → Recall@10.

### 5. Subsequent runs (instant)

After the first run, chunks and embeddings are cached. No `--rechunk` needed unless you change chunking params:

```bash
# Same command — skips indexing, runs benchmark only
python rag_pipeline/benchmark.py --config rag_pipeline/config.yaml --questions 1238_question_map_phap_dien.json --id-map "Correct ID.json"

# Force full re-index after changing chunking/embedding config
python rag_pipeline/benchmark.py --config rag_pipeline/config.yaml --questions 1238_question_map_phap_dien.json --id-map "Correct ID.json" --rechunk
```

### 6. Search the index

```bash
python rag_pipeline/main.py -c rag_pipeline/config.yaml search "thủ tục đấu giá tài sản"
```

---

## Project structure

```
.
├── Pipeline.md                         # Full pipeline specification
├── data/                               # Input JSON files (your data)
│   ├── chu_de_1.json … chu_de_45.json
├── rag_pipeline/
│   ├── config.yaml                     # ⚙ All tunable parameters
│   ├── config.py                       # Config loader (env var resolution)
│   ├── main.py                         # CLI: index / search
│   ├── benchmark.py                    # Recall@10 evaluation
│   │
│   ├── data_loader.py                  # JSON parser + benchmark ID mapping
│   │
│   ├── chunker.py                      # Recursive semantic chunker + disk cache
│   ├── context_generator.py            # LLM-powered article summarizer (OpenAI)
│   │
│   ├── embedder.py                     # Local + API embedding backends
│   ├── vector_store.py                 # ChromaDB persistent store
│   ├── bm25_retriever.py              # BM25 + pyvi Vietnamese word segmentation
│   │
│   ├── query_rewriter.py              # Vietnamese query normalization
│   ├── retriever.py                   # Hybrid: vector + BM25 + boost + rerank
│   ├── reranker.py                    # MMR diversity + keyword overlap
│   ├── pipeline_types.py              # Shared dataclasses
│   │
│   └── requirements.txt               # Python dependencies
├── REPORT.md                           # Latest benchmark results
├── .gitignore
└── README.md
```

---

## Data input

### Format A — VBQPPL parsed JSON (`data/chu_de_*.json`)

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
        "Ghi chú": {
          "Ghi chú": "Điều 1 Luật số 01/2016/QH14...",
          "metadata": [
            {
              "Hiệu lực": "01/07/2017",
              "Thi hành": "CÓ_HIỆU_LỰC_THI_HÀNH",
              "Link": "http://vbpl.vn/..."
            }
          ]
        },
        "Chỉ dẫn": [
          {
            "Điều liên quan": "Điều 28.3.LQ.82...",
            "ID Điều liên quan": "28003000...",
            "Link": "#"
          }
        ]
      }
    ]
  }
]
```

The loader extracts: `article_id`, `title`, `content` (Nội dung — what gets chunked), document metadata from Ghi chú, and cross-references from Chỉ dẫn.

### Benchmark ID mapping (`Correct ID.json`)

A dict of `article_content → numeric_id` (31,721 entries). During loading, each article's `Nội dung` is matched against this dict by exact content comparison. Matched articles get a `benchmark_id` stored in ChromaDB metadata. This ID is **never embedded** — it's only used to match retrieved chunks against `relevant_laws` during evaluation.

### Adding new data

Point the config at your directory:

```yaml
data:
  path: "../my_data"    # relative to config.yaml
```

Then run with `--rechunk`. For custom schemas, adapt `data_loader.py` — the only contract is that it produces objects with `.article_id`, `.title`, `.content`, and `.references` attributes.

---

## Full pipeline — phase by phase

### Phase 1 — Structure parsing

| File | Tool / method |
|------|---------------|
| `data_loader.py` | JSON walker — flattens topic → section → chapter → article hierarchy |

Extracts document metadata (law number, effective date, validity status), structural metadata (chapter, topic, mapCode), cross-references (Chỉ dẫn), and benchmark IDs (via content matching against `Correct ID.json`). Only the Article level downward contains meaningful legal text.

### Phase 2 — Recursive semantic chunking

| File | Tool / method |
|------|---------------|
| `chunker.py` | **Recursive Clause → Point → Paragraph → Sentence splitting** |

The algorithm (`Pipeline.md` §3):

1. **Article fits under `max_tokens`?** → keep as one chunk (`ARTICLE`)
2. **Too large?** → split by numbered clauses: regex `^\d+\.\s` → `CLAUSE` chunks
3. **Clause too large?** → split by lettered points: regex `^[a-đ]\)\s` → `POINT` chunks
4. **No Clause/Point structure?** → split by double-newline paragraphs → `PARAGRAPH` chunks
5. **Still too large?** → sentence-aware split on Vietnamese boundaries → `SENTENCE` chunks

**Quality validation** (`Pipeline.md` §4):
- Merge short chunks (< `min_tokens`) with adjacent ones
- Remove exact-duplicate content
- Preserve sentence boundaries — no mid-sentence cuts

**Token counting**: Uses the HuggingFace tokenizer matching the configured embedding model for accurate counts. Falls back to character-heuristic if unavailable.

### Phase 3 — Context generation & injection

| File | Tool / method |
|------|---------------|
| `context_generator.py` | **LLM API** — `gemma-4-31B-it` via company chat completions endpoint |

Per `Pipeline.md` §5: A 60–120 token Vietnamese legal summary is generated for each article by the LLM. Prompt template:

> "Bạn là trợ lý pháp lý. Tóm tắt điều luật sau bằng tiếng Việt trong 2-3 câu..."

The summary is **prepended** to every chunk from that article before embedding. This ensures no chunk is isolated from its article's overall context. Falls back to heuristic (title + first sentence) on API failure.

Config: `context.use_llm`, `context.llm.*` in `config.yaml`.

### Phase 4 — Embedding

| File | Tool / method |
|------|---------------|
| `embedder.py` | **Company Embedding API** — `Vietnamese_Embedding` (1024-dim, 2048-token max) |

Chunks are batched (64 per API call) and sent to the embedding endpoint. Returns normalized 1024-dim float32 vectors.

**Also supports local mode** — set `embedding.backend: "local"` to use `AITeamVN/Vietnamese_Embedding` via sentence-transformers. The `create_embedder()` factory in `embedder.py` handles switching.

### Phase 5 — Vector storage

| File | Tool / method |
|------|---------------|
| `vector_store.py` | **ChromaDB** — persistent, cosine-distance, HNSW-indexed |

Each chunk stored as:
- `embedding_vector`: 1024-dim float32
- `content`: context-injected chunk text
- `metadata`: article_id, unit_type, document_id, status, topic_id, topic_name, chapter_title, map_code, hierarchy_path, etc.

Metadata is stored alongside embeddings but **not** used in the embedding process — used only for filtering/boosting during retrieval (`Pipeline.md` §6).

### Phase 6 — BM25 sparse index

| File | Tool / method |
|------|---------------|
| `bm25_retriever.py` | **rank-bm25** (Okapi BM25) with **pyvi** Vietnamese word segmentation |

Built in parallel with the vector store. pyvi handles compound Vietnamese words: `"đấu_giá"`, `"giấy_phép"`, `"xây_dựng"`. Parameters: `k1=1.5`, `b=0.75`. Index persisted as pickle alongside ChromaDB.

### Phase 7 — Retrieval (per query)

| Stage | File | What happens |
|-------|------|-------------|
| **Query rewriting** | `query_rewriter.py` | Expands legal abbreviations (`dk` → `điều kiện`, `bhxh` → `bảo hiểm xã hội`). Appends domain context (`"theo pháp luật hiện hành"`) for short queries. |
| **Vector search** | `vector_store.py` | Cosine similarity top-k via ChromaDB |
| **BM25 search** | `bm25_retriever.py` | Parallel exact-term matching via Okapi BM25 |
| **Score fusion** | `retriever.py` | Weighted blend: `(1 − fusion_weight) × vector + fusion_weight × BM25`. Default 50/50. |
| **Metadata boost** | `retriever.py` | Prefer ACTIVE documents, matching topic, ARTICLE/CLAUSE-level chunks |
| **Graph expansion** | `retriever.py` | Follows Chỉ dẫn cross-references to pull in related articles |
| **Reranker** | `reranker.py` | MMR (Maximal Marginal Relevance) balances relevance with diversity. `λ=0.7` relevance, keyword overlap bonus, diversity penalty. |

### Phase 8 — Benchmarking

| File | Tool / method |
|------|---------------|
| `benchmark.py` | **Recall@10** on 1,238 questions |

Queries are pre-embedded in one API batch. For each question: retrieve top-10 chunks via full pipeline, check if any chunk's `benchmark_id` (metadata only, never embedded) matches the question's `relevant_laws`. Reports `hits / total_questions`.

**Caching:** Chunks saved to `chunks_cache.pkl`. ChromaDB + BM25 index persisted. Subsequent runs skip indexing entirely unless `--rechunk` is passed.

---

## Configuration reference

Key sections in `config.yaml`:

### `chunking`

| Key | Default | Description |
|-----|---------|-------------|
| `max_tokens` | 1024 | Split threshold — articles larger than this are recursively split |
| `min_tokens` | 30 | Merge chunks shorter than this with adjacent ones |
| `overlap_tokens` | 50 | Token overlap between adjacent chunks (prevents boundary cuts) |

### `api` (shared)

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `https://mkp-api.fptcloud.com` | OpenAI-compatible base URL (used by both embedding and LLM) |
| `api_key` | `${FPT_API_KEY}` | API key from environment variable |

### `embedding`

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `api` | `api` (company endpoint) or `local` (sentence-transformers) |
| `model_name` | `AITeamVN/Vietnamese_Embedding` | Model for local mode |
| `api.model` | `Vietnamese_Embedding` | Model name sent to API |
| `api.max_seq_length` | 2048 | Max tokens per input |
| `api.batch_size` | 64 | Texts per API call |

### `context`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Whether to prepend context to chunks |
| `use_llm` | `false` | Use LLM API for summaries (false = heuristic title + first sentence) |
| `max_context_tokens` | 100 | Target summary length |
| `llm.model` | `gemma-4-31B-it` | Model for context generation |
| `llm.temperature` | 0.3 | Sampling temperature |
| `llm.max_output_tokens` | 120 | Max summary tokens |

### `bm25`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable BM25 sparse retrieval |
| `fusion_weight` | 0.5 | BM25 weight in hybrid score (0 = pure vector, 1 = pure BM25) |
| `k1` | 1.5 | Term frequency saturation |
| `b` | 0.75 | Document length normalization |

### `retrieval`

| Key | Default | Description |
|-----|---------|-------------|
| `top_k` | 10 | Max results per query |
| `similarity_threshold` | 0.0 | Minimum score cutoff (0 = keep all) |
| `vector_weight` | 0.7 | Vector relevance weight before BM25 fusion |
| `enable_graph_expansion` | `true` | Follow article cross-references |

### `reranker`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable MMR reranking |
| `lambda_mmr` | 0.7 | Relevance vs diversity (1 = pure relevance) |
| `keyword_boost` | 0.15 | Weight for keyword overlap bonus |
| `diversity_weight` | 0.15 | Weight for diversity penalty |

---

## Chunk output schema

Each chunk stored in ChromaDB:

| Field | Type | Example |
|-------|------|---------|
| `chunk_id` | UUID | `017eb4cf-...` |
| `article_id` | str | `D45AE0D8-...` |
| `unit_type` | enum | `ARTICLE` / `CLAUSE` / `POINT` / `PARAGRAPH` / `SENTENCE` |
| `content` | str | `"Điều: Nguyên tắc bầu cử.\nViệc bầu cử..."` |
| `raw_content` | str | `"Việc bầu cử..."` (before context injection) |
| `token_count` | int | `87` |
| `hierarchy_path` | str | `article_id/clause_1/point_b` |
| `document_id` | str | `85/2015/QH13` |
| `document_type` | str | `Luật` |
| `effective_date` | str | `01/09/2015` |
| `status` | str | `CÓ_HIỆU_LỰC_THI_HÀNH` |
| `topic_id` | str | `3fc1ee9d-...` |
| `topic_name` | str | `Tổ chức bộ máy nhà nước` |
| `chapter_title` | str | `Chương I NHỮNG QUY ĐỊNH CHUNG` |
| `map_code` | str | `35001000...` |

---

## API response formats

Both APIs share `api.base_url` (e.g. `https://mkp-api.fptcloud.com`). The pipeline auto-appends `/v1/embeddings` and `/v1/chat/completions`.

### Embedding API (`POST {base_url}/v1/embeddings`)

**Request:**
```json
{
  "model": "Vietnamese_Embedding",
  "input": ["text chunk 1", "text chunk 2"]
}
```

**Response:**
```json
{
  "data": [
    {"embedding": [0.012, -0.034, ...]},
    {"embedding": [0.008, 0.021, ...]}
  ]
}
```

Also supports `{"embeddings": [[...], [...]]}` format. Uses `requests` library with 3-retry exponential backoff on connection failures.

### LLM API (`POST {base_url}/v1/chat/completions`)

**Request:**
```json
{
  "model": "gemma-4-31B-it",
  "messages": [{"role": "user", "content": "Bạn là trợ lý pháp lý. Tóm tắt..."}],
  "temperature": 0.3,
  "max_tokens": 120
}
```

**Response:**
```json
{
  "choices": [{"message": {"content": "Điều luật này quy định về..."}}]
}
```

Uses `openai` Python library for the chat endpoint. If your API uses a different format, adapt `context_generator.py` (the `LLMContextGenerator.generate` method) and `embedder.py` (the `APIEmbedder.embed` method).
