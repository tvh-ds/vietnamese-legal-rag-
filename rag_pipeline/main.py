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

from bm25_retriever import BM25Retriever
from config import Config, load_config
from context_generator import LLMContextGenerator
from data_loader import load_articles
from chunker import chunk_articles, set_tokenizer_model
from embedder import create_embedder
from query_rewriter import QueryRewriter
from reranker import Reranker, LLMReranker
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
    print("\n[1/5] Loading articles…")
    t0 = time.perf_counter()
    articles = load_articles(config.data.path)
    t1 = time.perf_counter()
    print(f"  Loaded {len(articles)} articles in {t1 - t0:.1f}s")

    # 2. Chunk
    print("\n[2/5] Chunking articles…")
    set_tokenizer_model(config.embedding.model_name)
    t0 = time.perf_counter()

    # LLM context generator (optional)
    llm_gen = None
    if config.context.use_llm:
        llm_gen = LLMContextGenerator(
            base_url=getattr(config, "_api_base_url", ""),
            api_key=getattr(config, "_api_key", ""),
            model=config.context.llm.model,
            temperature=config.context.llm.temperature,
            max_output_tokens=config.context.llm.max_output_tokens,
            timeout_sec=config.context.llm.timeout_sec,
        )

    chunks = chunk_articles(
        articles,
        max_tokens=config.chunking.max_tokens,
        min_tokens=config.chunking.min_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
        enable_context=config.context.enabled,
        max_context_tokens=config.context.max_context_tokens,
        llm_context_generator=llm_gen,
    )
    t1 = time.perf_counter()
    print(f"  Produced {len(chunks)} chunks in {t1 - t0:.1f}s")

    # 3. Embed
    print("\n[3/5] Generating embeddings…")
    t0 = time.perf_counter()
    embedder = create_embedder(config)
    if config.embedding.backend == "api":
        print(f"  Backend: API ({config.embedding.api.model})")
    else:
        print(f"  Model: {config.embedding.model_name}")
    texts = [ch.content for ch in chunks]
    embeddings = embedder.embed(texts, show_progress=True)
    t1 = time.perf_counter()
    print(f"  Embedded {len(chunks)} chunks ({embeddings.shape[1]}-dim) in {t1 - t0:.1f}s")

    # 4. Store
    print("\n[4/5] Storing in vector database…")
    t0 = time.perf_counter()
    store = VectorStore(
        path=config.vector_store.path,
        collection_name=config.vector_store.collection_name,
    )
    store.clear()  # Fresh start
    store.upsert(chunks, embeddings)
    t1 = time.perf_counter()
    print(f"  Stored {store.count()} chunks in {t1 - t0:.1f}s")

    # 5. BM25 index
    if config.bm25.enabled:
        print("\n[5/5] Building BM25 index…")
        t0 = time.perf_counter()
        bm25 = BM25Retriever(
            k1=config.bm25.k1,
            b=config.bm25.b,
            top_k=config.bm25.top_k,
        )
        # Build index from chunks (convert Chunk → RetrievalResult-like)
        bm25.build_index(chunks)  # type: ignore[arg-type]
        bm25.save(config.bm25.index_path)
        t1 = time.perf_counter()
        print(f"  BM25 index built in {t1 - t0:.1f}s")
    else:
        print("\n[5/5] BM25 index skipped (disabled in config)")

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
    # Load articles for the retriever
    print(f"Loading articles…", file=sys.stderr)
    articles = load_articles(config.data.path)

    embedder = create_embedder(config)
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
        if config.reranker.mode == "llm_api":
            reranker = LLMReranker(
                base_url=getattr(config, "_api_base_url", ""),
                api_key=getattr(config, "_api_key", ""),
                model=config.reranker.llm.model,
                endpoint=config.reranker.llm.endpoint,
                max_candidates=config.reranker.llm.max_candidates,
                top_n=config.reranker.llm.top_n,
                temperature=config.reranker.llm.temperature,
                max_output_tokens=config.reranker.llm.max_output_tokens,
                max_candidate_chars=config.reranker.llm.max_candidate_chars,
                blend_weight=config.reranker.llm.blend_weight,
                timeout_sec=config.reranker.llm.timeout_sec,
                conditional_enabled=config.reranker.llm.conditional.enabled,
                conditional_strategy=config.reranker.llm.conditional.strategy,
                conditional_top1_top2_gap=config.reranker.llm.conditional.top1_top2_gap,
                conditional_top1_top10_gap=config.reranker.llm.conditional.top1_top10_gap,
                conditional_min_candidates=config.reranker.llm.conditional.min_candidates,
            )
        else:
            reranker = Reranker(
                base_url=getattr(config, "_api_base_url", ""),
                api_key=config.reranker.bge.api_key,
                model=config.reranker.bge.model,
                endpoint=config.reranker.bge.endpoint,
                max_candidates=config.reranker.bge.max_candidates,
                top_n=config.reranker.bge.top_n,
                timeout_sec=config.reranker.bge.timeout_sec,
                blend_weight=config.reranker.bge.blend_weight,
                conditional_enabled=config.reranker.bge.conditional.enabled,
                conditional_strategy=config.reranker.bge.conditional.strategy,
                conditional_top1_top2_gap=config.reranker.bge.conditional.top1_top2_gap,
                conditional_top1_top10_gap=config.reranker.bge.conditional.top1_top10_gap,
                conditional_min_candidates=config.reranker.bge.conditional.min_candidates,
            )

    # BM25 retriever (if available)
    bm25 = None
    if config.bm25.enabled:
        bm25 = BM25Retriever.load(config.bm25.index_path)
        if bm25 is None:
            print("Warning: BM25 index not found — run 'index' first to build it",
                  file=sys.stderr)

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        articles=articles,
        top_k=top_k or config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold,
        enable_metadata_filtering=config.retrieval.enable_metadata_filtering,
        vector_weight=config.retrieval.vector_weight,
        metadata_boost=config.retrieval.metadata_boost,
        prefer_active=config.metadata.prefer_active,
        topic_boost=config.metadata.topic_boost,
        reranker=reranker,
        bm25_retriever=bm25,
        bm25_weight=config.bm25.fusion_weight,
        fusion_method=config.bm25.fusion_method,
        rrf_k=config.bm25.rrf_k,
        rrf_vector_weight=config.bm25.rrf_vector_weight,
        rrf_bm25_weight=config.bm25.rrf_bm25_weight,
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

    # Resolve relative paths against config file directory
    config_dir = config_path.parent
    if not Path(config.data.path).is_absolute():
        config.data.path = str(config_dir / config.data.path)
    if not Path(config.vector_store.path).is_absolute():
        config.vector_store.path = str(config_dir / config.vector_store.path)
    if not Path(config.bm25.index_path).is_absolute():
        config.bm25.index_path = str(config_dir / config.bm25.index_path)

    if args.command == "index":
        cmd_index(config)
    elif args.command == "search":
        cmd_search(config, args.query, args.top_k, raw=args.raw)


if __name__ == "__main__":
    main()
