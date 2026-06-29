#!/usr/bin/env python3
"""CLI entry point for the Vietnamese Legal RAG chunking + retrieval pipeline.

Subcommands:
    index     Load JSON data, chunk articles, embed, and store in vector DB.
    search    Run a natural-language query and print top results.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import Config, load_config
from data_loader import Article, load_articles
from chunker import Chunk, Chunker, chunk_articles
from embedder import Embedder
from query_rewriter import QueryRewriter
from reranker import Reranker
from retriever import Retriever
from vector_store import VectorStore

logger = logging.getLogger("rag_pipeline")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# index subcommand
# ---------------------------------------------------------------------------


def cmd_index(config: Config) -> None:
    """Run the full offline indexing pipeline."""
    print("=" * 60)
    print("VBQPPL Legal RAG — Indexing Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading articles…")
    t0 = time.perf_counter()
    articles = load_articles(config.data.path)
    t1 = time.perf_counter()
    print(f"  Loaded {len(articles)} articles in {t1 - t0:.1f}s")

    # 2. Chunk
    print("\n[2/4] Chunking articles…")
    t0 = time.perf_counter()
    chunker = Chunker(
        max_tokens=config.chunking.max_tokens,
        min_tokens=config.chunking.min_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
        enable_context=config.context.enabled,
        max_context_tokens=config.context.max_context_tokens,
    )
    chunks = chunk_articles(
        articles,
        max_tokens=config.chunking.max_tokens,
        min_tokens=config.chunking.min_tokens,
        enable_context=config.context.enabled,
        max_context_tokens=config.context.max_context_tokens,
    )
    t1 = time.perf_counter()
    print(f"  Produced {len(chunks)} chunks in {t1 - t0:.1f}s")

    # 3. Embed
    print("\n[3/4] Generating embeddings…")
    print(f"  Model: {config.embedding.model_name}")
    t0 = time.perf_counter()
    embedder = Embedder(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )
    texts = [ch.content for ch in chunks]
    embeddings = embedder.embed(texts, show_progress=True)
    t1 = time.perf_counter()
    print(f"  Embedded {len(chunks)} chunks ({embeddings.shape[1]}-dim) in {t1 - t0:.1f}s")

    # 4. Store
    print("\n[4/4] Storing in vector database…")
    t0 = time.perf_counter()
    store = VectorStore(
        path=config.vector_store.path,
        collection_name=config.vector_store.collection_name,
    )
    store.clear()  # Fresh start
    store.upsert(chunks, embeddings)
    t1 = time.perf_counter()
    print(f"  Stored {store.count()} chunks in {t1 - t0:.1f}s")

    print("\n[OK] Indexing complete.")


# ---------------------------------------------------------------------------
# search subcommand
# ---------------------------------------------------------------------------


def cmd_search(
    config: Config,
    query: str,
    top_k: int | None = None,
    raw: bool = False,
) -> None:
    """Run a retrieval query and display results."""
    import json as _json

    # Load articles for the retriever (graph expansion needs the lookup)
    print(f"Loading articles…", file=sys.stderr)
    articles = load_articles(config.data.path)

    embedder = Embedder(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
    )
    store = VectorStore(
        path=config.vector_store.path,
        collection_name=config.vector_store.collection_name,
    )

    if store.count() == 0:
        print("Error: Vector store is empty. Run 'index' first.", file=sys.stderr)
        sys.exit(1)

    # Query rewriting (Section 9)
    original_query = query
    if config.query_rewriter.enabled and not raw:
        rewriter = QueryRewriter(
            append_context=config.query_rewriter.append_context,
            context_phrase=config.query_rewriter.context_phrase,
            expand_abbreviations=config.query_rewriter.expand_abbreviations,
        )
        query = rewriter.rewrite(query)
        if query != original_query:
            print(f"Rewritten: {original_query} → {query}", file=sys.stderr)

    # Reranker (Section 11)
    reranker = None
    if config.reranker.enabled:
        reranker = Reranker(
            lambda_mmr=config.reranker.lambda_mmr,
            keyword_boost=config.reranker.keyword_boost,
            diversity_weight=config.reranker.diversity_weight,
        )

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        articles=articles,
        top_k=top_k or config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold,
        enable_metadata_filtering=config.retrieval.enable_metadata_filtering,
        enable_graph_expansion=config.retrieval.enable_graph_expansion,
        expansion_max_articles=config.retrieval.expansion_max_articles,
        vector_weight=config.retrieval.vector_weight,
        metadata_boost=config.retrieval.metadata_boost,
        graph_boost=config.retrieval.graph_boost,
        prefer_active=config.metadata.prefer_active,
        topic_boost=config.metadata.topic_boost,
        reranker=reranker,
    )

    results = retriever.retrieve(query)

    print(f"\nQuery: {query}")
    if query != original_query:
        print(f"Original: {original_query}")
    print(f"Results: {len(results)}")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n#{i}  score={r.score:.4f}  source={r.source}  type={r.unit_type}")
        print(f"    Article: {r.metadata.get('article_id', 'N/A')}")
        print(f"    Status: {r.metadata.get('status', 'N/A')}")
        print(f"    Topic: {r.metadata.get('topic_name', 'N/A')}")
        # Show first 300 chars of content
        content_preview = r.content[:300].replace("\n", " ")
        print(f"    Content: {content_preview}…")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vietnamese Legal RAG — Chunking + Retrieval Pipeline",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in rag_pipeline/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # index
    sub.add_parser("index", help="Run the full indexing pipeline")

    # search
    search_parser = sub.add_parser("search", help="Search for legal chunks")
    search_parser.add_argument("query", help="Query text in Vietnamese")
    search_parser.add_argument(
        "-k", "--top-k", type=int, default=None,
        help="Number of results (overrides config)",
    )
    search_parser.add_argument(
        "--raw", action="store_true",
        help="Skip query rewriting (use raw query as-is)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(config_path))

    # Resolve data and vector_store paths relative to the config file directory
    config_dir = config_path.parent
    if not Path(config.data.path).is_absolute():
        config.data.path = str(config_dir / config.data.path)
    if not Path(config.vector_store.path).is_absolute():
        config.vector_store.path = str(config_dir / config.vector_store.path)

    if args.command == "index":
        cmd_index(config)
    elif args.command == "search":
        cmd_search(config, args.query, args.top_k, raw=args.raw)


if __name__ == "__main__":
    main()
