#!/usr/bin/env python3
"""GPU-offload workflow for embedding.

Step 1 (cloud desktop): python split_workflow.py chunk --config rag_pipeline/config.yaml --corpus corpus/vlqa/legal_corpus.json

Step 2 (Colab):      Upload chunks_text.pkl and chunks_data.pkl to Colab, run colab_embed.py

Step 3 (cloud desktop): python split_workflow.py store --config rag_pipeline/config.yaml
                        python split_workflow.py benchmark --config rag_pipeline/config.yaml --questions corpus/vlqa/1238_question_map_phap_dien.json -n 50
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from config import Config, load_config
from corpus_loader import load_corpus
from chunker import Chunker, set_tokenizer_model
from vector_store import VectorStore


# ---------------------------------------------------------------------------
# Step 1 — Chunk on CPU
# ---------------------------------------------------------------------------

def cmd_chunk(config: Config, corpus_path: str) -> None:
    print("=" * 50)
    print("Step 1: Chunking on CPU")
    print("=" * 50)

    print("\nLoading corpus...")
    articles = load_corpus(corpus_path)
    print(f"  {len(articles)} articles loaded")

    print("\nChunking...")
    set_tokenizer_model(config.embedding.model_name)
    chunker = Chunker(
        max_tokens=config.chunking.max_tokens,
        min_tokens=config.chunking.min_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
        enable_context=config.context.enabled,
        max_context_tokens=config.context.max_context_tokens,
    )
    chunks = []
    for art in articles:
        chunks.extend(chunker.chunk_article(art))
    print(f"  {len(chunks)} chunks produced")

    texts = [c.content for c in chunks]

    with open("chunks_text.pkl", "wb") as f:
        pickle.dump(texts, f)
    with open("chunks_data.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nSaved: chunks_text.pkl ({len(texts)} texts)")
    print(f"Saved: chunks_data.pkl ({len(chunks)} chunk objects)")
    print("\nUpload these two files to Google Colab and run colab_embed.py")
    print("Then download embeddings.npy back here and run: python split_workflow.py store")


# ---------------------------------------------------------------------------
# Step 2 — Colab embedding (copy-paste into Colab cell)
# ---------------------------------------------------------------------------

COLAB_CODE = '''\
# ============================================================
# Run this in Google Colab with GPU runtime (T4 is free)
# Runtime → Change runtime type → T4 GPU
# ============================================================

!pip install -q sentence-transformers

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Upload chunks_text.pkl (use the file upload button on the left panel)
# Or mount Google Drive and copy files there first

with open("chunks_text.pkl", "rb") as f:
    texts = pickle.load(f)

print(f"Loaded {len(texts)} chunk texts")

model = SentenceTransformer(
    "AITeamVN/Vietnamese_Embedding",
    device="cuda",
)
print(f"Model loaded, dim={model.get_sentence_embedding_dimension()}")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

np.save("embeddings.npy", embeddings)
print(f"Saved embeddings.npy — shape={embeddings.shape}")

# Download embeddings.npy from the file panel (left side)
# Move it back to your cloud desktop project folder\
'''


def cmd_colab_code() -> None:
    print(COLAB_CODE)


# ---------------------------------------------------------------------------
# Step 3 — Store embeddings + benchmark or search
# ---------------------------------------------------------------------------

def cmd_store(config: Config) -> None:
    print("=" * 50)
    print("Step 3: Storing embeddings")
    print("=" * 50)

    print("\nLoading chunks...")
    with open("chunks_data.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"  {len(chunks)} chunks")

    print("Loading embeddings...")
    embeddings = np.load("embeddings.npy")
    print(f"  shape={embeddings.shape}")

    store_path = config.vector_store.path + "_benchmark"
    print(f"\nStoring to {store_path}...")
    store = VectorStore(path=store_path, collection_name="benchmark_corpus")
    store.clear()
    store.upsert(chunks, embeddings)
    print(f"  {store.count()} chunks stored")

    print("\nDone. Run: python split_workflow.py benchmark --config rag_pipeline/config.yaml --questions corpus/vlqa/1238_question_map_phap_dien.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-offload embedding workflow")
    parser.add_argument("--config", default="rag_pipeline/config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command")

    chunk_p = sub.add_parser("chunk", help="Step 1: chunk corpus on CPU, save to disk")
    chunk_p.add_argument("--corpus", required=True, help="Path to legal_corpus.json")

    sub.add_parser("colab-code", help="Print the Colab embedding code to copy-paste")

    sub.add_parser("store", help="Step 3: load chunks + embeddings, store in ChromaDB")

    bench_p = sub.add_parser("benchmark", help="Run benchmark against stored chunks")
    bench_p.add_argument("--questions", required=True, help="Path to benchmark questions JSON")
    bench_p.add_argument("-n", "--max-questions", type=int, default=None)

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate
    config = load_config(str(config_path))
    config.vector_store.path = str(config_path.parent / config.vector_store.path)

    if args.command == "chunk":
        cmd_chunk(config, args.corpus)
    elif args.command == "colab-code":
        cmd_colab_code()
    elif args.command == "store":
        cmd_store(config)
    elif args.command == "benchmark":
        from benchmark import Benchmark
        bench = Benchmark(config)
        result = bench.evaluate_prebuilt(
            questions_path=args.questions,
            max_questions=args.max_questions,
        )
        import json
        with open("benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved benchmark_results.json")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
