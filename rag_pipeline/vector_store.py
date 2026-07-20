"""ChromaDB-backed vector store for legal chunks.

Implements Section 8 of Pipeline.md: stores embedding vectors alongside
structured metadata that does NOT participate in the embedding process.

Metadata fields stored per chunk:
  - chunk_id, article_id, unit_type, parent_unit_id, order, hierarchy_path
  - document_id, document_type, effective_date, status
  - topic_id, topic_name, chapter_title, map_code
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for legal chunk embeddings.

    Usage::

        store = VectorStore(path="./chroma_db", collection_name="vbqppl_chunks")
        store.upsert(chunks, embeddings)
        results = store.query(query_embedding, top_k=10)
    """

    def __init__(
        self,
        path: str | Path = "./chroma_db",
        collection_name: str = "vbqppl_legal_chunks",
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    # -- lazy init -----------------------------------------------------------

    @property
    def client(self):
        """Lazy-load ChromaDB persistent client."""
        if self._client is None:
            import chromadb

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
        return self._client

    @property
    def collection(self):
        """Lazy-load or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # -- CRUD ----------------------------------------------------------------

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 500,
    ) -> int:
        """Insert or update chunks and their embeddings.

        Args:
            chunks: Chunk objects with metadata.
            embeddings: NumPy array of shape (len(chunks), dim).
            batch_size: Number of chunks per ChromaDB upsert batch.

        Returns:
            Total number of chunks upserted.
        """
        total = len(chunks)
        if total == 0:
            return 0

        if embeddings.shape[0] != total:
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) != chunks count ({total})"
            )

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [ch.chunk_id for ch in batch_chunks]
            documents = [ch.content for ch in batch_chunks]
            metadatas = [self._build_metadata(ch) for ch in batch_chunks]

            self.collection.upsert(
                ids=ids,
                embeddings=batch_embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

        logger.info("Upserted %d chunks into collection '%s'.", total, self.collection_name)
        return total

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Run a vector similarity search.

        Args:
            query_embedding: 1-D embedding vector for the query.
            top_k: Maximum number of results to return.
            where: Optional ChromaDB where-clause for metadata filtering.

        Returns:
            List of result dicts with keys: id, document, metadata, distance.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB's list-of-lists response
        out: list[dict] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                out.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return out

    def get_by_article(self, article_id: str) -> list[dict]:
        """Retrieve all chunks belonging to a specific article."""
        results = self.collection.get(
            where={"article_id": article_id},
            include=["documents", "metadatas"],
        )
        out: list[dict] = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                out.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return out

    def delete_by_article(self, article_id: str) -> int:
        """Remove all chunks for an article. Returns count of deleted chunks."""
        existing = self.collection.get(
            where={"article_id": article_id},
            include=[],
        )
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            count = len(existing["ids"])
            logger.info("Deleted %d chunks for article '%s'.", count, article_id)
            return count
        return 0

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete the entire collection and re-create it."""
        import chromadb.errors

        try:
            self.client.delete_collection(self.collection_name)
        except (chromadb.errors.NotFoundError, ValueError):
            # Collection doesn't exist yet — nothing to delete
            pass
        self._collection = None
        logger.info("Cleared collection '%s'.", self.collection_name)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict:
        """Build a ChromaDB-compatible metadata dict from a Chunk.

        ChromaDB only supports str | int | float | bool values, so we
        ensure all values are of those types.
        """
        return {
            "chunk_id": chunk.chunk_id,
            "article_id": chunk.article_id,
            "unit_type": chunk.unit_type,
            "parent_unit_id": chunk.parent_unit_id,
            "order": chunk.order,
            "hierarchy_path": chunk.hierarchy_path,
            "token_count": chunk.token_count,
            "document_id": chunk.document_id,
            "document_type": chunk.document_type,
            "effective_date": chunk.effective_date,
            "status": chunk.status,
            "topic_id": chunk.topic_id,
            "topic_name": chunk.topic_name,
            "chapter_title": chunk.chapter_title,
            "map_code": chunk.map_code,
            "benchmark_id": getattr(chunk, "benchmark_id", ""),
            "raw_content": chunk.raw_content,
        }
