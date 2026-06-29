# VBQPPL Legal RAG — Chunking + Retrieval Pipeline

Vietnamese legal document chunking and semantic retrieval pipeline for building a RAG-powered legal chatbot. Processes parsed VBQPPL JSON documents through recursive semantic chunking, embedding, and hybrid retrieval.

## Project structure

```
Chunking Pipeline/
├── Pipeline.md                  # Pipeline specification (design doc)
├── data/                        # Input: parsed VBQPPL JSON files (11 files)
│   ├── chu_de_4.json
│   ├── chu_de_5.json
│   └── ...
├── rag_pipeline/
│   ├── config.yaml              # All tunable parameters
│   ├── config.py                # Config loader (dataclass-based)
│   ├── data_loader.py           # JSON parser → Article objects
│   ├── chunker.py               # Recursive semantic chunker
│   ├── embedder.py              # Sentence-transformers wrapper
│   ├── vector_store.py          # ChromaDB vector store
│   ├── query_rewriter.py        # Vietnamese legal query normalizer
│   ├── reranker.py              # MMR diversity reranker
│   ├── retriever.py             # Hybrid retrieval orchestrator
│   ├── pipeline_types.py        # Shared types
│   ├── main.py                  # CLI entry point
│   ├── requirements.txt         # Python dependencies
│   └── chroma_db/               # Persisted vector database (24,395 chunks)
└── README.md                    # This file
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r rag_pipeline/requirements.txt
```

Requirements: `chromadb`, `sentence-transformers`, `transformers`, `pyyaml`, `numpy`, `tqdm`.

The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, ~420 MB) downloads automatically on first use and caches in `~/.cache/huggingface/`.

### 2. Search the existing index

The vector database is pre-built with 24,395 chunks from 14,962 Vietnamese legal articles. Searching works immediately:

```bash
# Basic search
python rag_pipeline/main.py -c rag_pipeline/config.yaml search "thủ tục đấu giá tài sản"

# Telex / no-diacritics input (query rewriter expands abbreviations)
python rag_pipeline/main.py -c rag_pipeline/config.yaml search "dk cap giay phep xay dung"

# Limit number of results
python rag_pipeline/main.py -c rag_pipeline/config.yaml search "điều kiện ứng cử đại biểu" -k 3

# Skip query rewriting (search raw input as-is)
python rag_pipeline/main.py -c rag_pipeline/config.yaml search "some raw query" --raw
```

### 3. Rebuild the index (optional)

To re-index from scratch after changing config or data:

```bash
python rag_pipeline/main.py -c rag_pipeline/config.yaml index
```

This clears the existing ChromaDB collection and runs the full pipeline: load → chunk → embed → store.

## Loading your own data

### Data format

The loader expects JSON files with this schema (same as the existing `data/*.json` files):

```
Root array
└── Topic entry
    ├── "Chủ đề ID": "<uuid>"
    ├── "Tên chủ đề": "<topic name>"
    ├── "Chương ID": "<uuid>"
    ├── "Tên chương": "<chapter title>"
    └── "Các điều": [                    ← array of articles
        {
          "Điều ID": "<uuid>",
          "Tên điều": "<article title>",
          "Nội dung": "<legal text>",    ← this gets chunked
          "Ghi chú": {
            "Ghi chú": "<citation>",     ← extracts document type & ID
            "metadata": [
              { "Hiệu lực": "<date>", "Thi hành": "<status>", "Link": "<url>" }
            ]
          },
          "Chỉ dẫn": [                   ← cross-references for graph expansion
            { "Điều liên quan": "...", "ID Điều li quan": "<id>", "Link": "#" }
          ]
        }
      ]
```

If your JSON files follow this schema, they work as-is. If the schema differs, adapt `data_loader.py`.

### Method A — Point the pipeline at a new data directory

1. Place your JSON files in a directory (e.g., `my_data/`)
2. Update `rag_pipeline/config.yaml`:

```yaml
data:
  path: "../my_data"    # relative to config.yaml; or an absolute path
```

3. Rebuild the index:

```bash
python rag_pipeline/main.py -c rag_pipeline/config.yaml index
```

### Method B — Incremental add via Python API

Add new articles without rebuilding the entire index:

```python
import sys; sys.path.insert(0, 'rag_pipeline')

from data_loader import load_articles
from chunker import Chunker
from embedder import Embedder
from vector_store import VectorStore

# 1. Load your new JSON files
articles = load_articles("path/to/your/data")

# 2. Chunk
chunker = Chunker(max_tokens=512, min_tokens=30)
chunks = []
for art in articles:
    chunks.extend(chunker.chunk_article(art))

print(f"Produced {len(chunks)} chunks")

# 3. Embed
embedder = Embedder()
embeddings = embedder.embed([c.content for c in chunks])

# 4. Store — merges with existing chunks (no overwrite, no re-index)
store = VectorStore(
    path="rag_pipeline/chroma_db",
    collection_name="vbqppl_legal_chunks",
)
store.upsert(chunks, embeddings)
print(f"Done. Total chunks in store: {store.count()}")
```

Each chunk gets a unique UUID, so existing data is never overwritten. Search immediately reflects new content.

## Seeing chunking results

### During indexing

The `index` command prints chunk statistics:

```
[2/4] Chunking articles…
  Produced 29792 chunks in 12.3s
```

Enable verbose logging for per-article detail:

```bash
python rag_pipeline/main.py -v -c rag_pipeline/config.yaml index
```

### Via Python API — inspect chunks

```python
import sys; sys.path.insert(0, 'rag_pipeline')

from data_loader import load_articles
from chunker import Chunker

articles = load_articles("data")
chunker = Chunker(max_tokens=512, min_tokens=30)

# Pick one article
art = articles[0]
print(f"Article: {art.title}")
print(f"Content length: {len(art.content)} chars\n")

# Chunk it
chunks = chunker.chunk_article(art)
for i, ch in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(f"  ID:           {ch.chunk_id}")
    print(f"  Type:         {ch.unit_type}")
    print(f"  Tokens:       {ch.token_count}")
    print(f"  Hierarchy:    {ch.hierarchy_path}")
    print(f"  Topic:        {ch.topic_name}")
    print(f"  Status:       {ch.status}")
    print(f"  Content:      {ch.content[:200]}...")
    print()
```

### Bulk inspect — chunk type distribution

```python
import sys; sys.path.insert(0, 'rag_pipeline')

from data_loader import load_articles
from chunker import Chunker

articles = load_articles("data")
chunker = Chunker(max_tokens=512, min_tokens=30)

type_counts = {}
size_buckets = {"<100": 0, "100-300": 0, "300-512": 0, ">512": 0}

for art in articles:
    for ch in chunker.chunk_article(art):
        type_counts[ch.unit_type] = type_counts.get(ch.unit_type, 0) + 1
        tc = ch.token_count
        if tc < 100:    size_buckets["<100"] += 1
        elif tc < 300:  size_buckets["100-300"] += 1
        elif tc <= 512: size_buckets["300-512"] += 1
        else:           size_buckets[">512"] += 1

print("Unit types:", type_counts)
print("Sizes:", size_buckets)
```

### Query the store directly — see stored chunks

```python
import sys; sys.path.insert(0, 'rag_pipeline')
from vector_store import VectorStore

store = VectorStore(
    path="rag_pipeline/chroma_db",
    collection_name="vbqppl_legal_chunks",
)

# Get all chunks for a specific article
chunks = store.get_by_article("D45AE0D8-627B-4E24-927F-F8B1E501D93E")
for c in chunks:
    print(f"  [{c['metadata']['unit_type']}] {c['content'][:150]}...")
```

## What a chunk looks like

Each chunk produced by `chunker.py` and stored in ChromaDB has these fields:

| Field | Example | Description |
|-------|---------|-------------|
| `chunk_id` | `017eb4cf-...` | Unique UUID |
| `article_id` | `D45AE0D8-...` | Parent article (Điều ID) |
| `unit_type` | `CLAUSE` | ARTICLE / CLAUSE / POINT / PARAGRAPH / SENTENCE |
| `parent_unit_id` | `article_id/clause_1` | Parent in hierarchy |
| `order` | `2` | Position within parent |
| `token_count` | `87` | WordPiece tokens (real tokenizer) |
| `hierarchy_path` | `article_id/clause_1/point_b` | Full path |
| `content` | `"Điều: Nguyên tắc...\nViệc bầu cử..."` | Context-injected chunk text |
| `raw_content` | `"Việc bầu cử..."` | Original text before context injection |
| `document_id` | `85/2015/QH13` | Law number |
| `document_type` | `Luật` | Law type |
| `effective_date` | `01/09/2015` | When it took effect |
| `status` | `CÓ_HIỆU_LỰC_THI_HÀNH` | Validity status |
| `topic_id` | `3fc1ee9d-...` | Legal topic ID |
| `topic_name` | `Tổ chức bộ máy nhà nước` | Legal topic name |
| `chapter_title` | `Chương I NHỮNG QUY ĐỊNH CHUNG` | Chapter heading |
| `map_code` | `35001000...` | Hierarchical navigation code |

Metadata is stored separately from the embedding vector (Pipeline.md §6) and used for filtering/boosting during retrieval.

## Chunking strategy

Follows Pipeline.md §3–5:

1. **Article fits under 512 tokens?** → Keep as one chunk (type `ARTICLE`)
2. **Too large?** → Split into numbered **Clauses** (`1. …`, `2. …` → type `CLAUSE`)
3. **Clause still too large?** → Split into lettered **Points** (`a) …`, `b) …` → type `POINT`)
4. **No Clause/Point structure?** → **Late chunking**: split by paragraphs (type `PARAGRAPH`)
5. **Paragraph too large?** → Sentence-aware split (type `SENTENCE`)
6. **Quality**: short chunks (< 30 tokens) merged with adjacent; duplicates removed
7. **Context**: article-level summary prepended to each chunk before embedding

## Configuration

All tunable parameters in `rag_pipeline/config.yaml`:

| Section | Key | Default | What it controls |
|---------|-----|---------|-----------------|
| `chunking` | `max_tokens` | 512 | Split threshold |
| `chunking` | `min_tokens` | 30 | Merge short chunks below this |
| `embedding` | `model_name` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (384-dim, Vietnamese-compatible) |
| `embedding` | `device` | `cpu` | `cpu` / `cuda` |
| `vector_store` | `path` | `./chroma_db` | Where ChromaDB persists data |
| `retrieval` | `top_k` | 10 | Max results per query |
| `retrieval` | `similarity_threshold` | 0.3 | Minimum score cutoff |
| `query_rewriter` | `enabled` | `true` | Expand abbreviations + add legal context |
| `reranker` | `enabled` | `true` | MMR diversity + keyword boost reranking |
| `context` | `enabled` | `true` | Prepend article summary to chunks |

To use a Vietnamese-specific embedding model (better retrieval quality), change:

```yaml
embedding:
  model_name: "keepitreal/vietnamese-sbert"
```

Then re-index: `python rag_pipeline/main.py -c rag_pipeline/config.yaml index`
