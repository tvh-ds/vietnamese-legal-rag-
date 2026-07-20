"""BM25 sparse retriever with Vietnamese word segmentation.

Runs parallel to dense vector search. BM25 excels at exact term matching
(e.g. "đấu giá tài sản", "giấy phép xây dựng") which is critical for
legal text where specific terminology matters as much as semantics.

Vietnamese word segmentation via pyvi handles compound words like
"học_sinh", "đấu_giá", "xây_dựng" correctly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from pipeline_types import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vietnamese tokenizer
# ---------------------------------------------------------------------------

_vi_tokenizer = None


def _get_vi_tokenizer():
    """Lazy-load pyvi Vietnamese word segmenter."""
    global _vi_tokenizer
    if _vi_tokenizer is None:
        try:
            from pyvi import ViTokenizer
            _vi_tokenizer = ViTokenizer.tokenize
        except ImportError:
            logger.warning("pyvi not installed — falling back to whitespace tokenizer")
            _vi_tokenizer = lambda text: text  # noqa: E731
    return _vi_tokenizer


def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese text into words.

    With pyvi: "học sinh giỏi" → ["học_sinh", "giỏi"]
    Without:   "học sinh giỏi" → ["học", "sinh", "giỏi"]
    """
    tok = _get_vi_tokenizer()
    tokenized = tok(text)
    return [t for t in tokenized.split() if len(t) >= 1]


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 sparse retrieval over chunk contents.

    Builds an in-memory BM25 index during indexing and persists it to disk.
    During search, returns top-k chunks by BM25 score.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 20,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.top_k = top_k

        # Populated by build_index()
        self._chunks: list[dict] = []          # chunk metadata per document
        self._doc_texts: list[str] = []        # raw content per document
        self._tokenized_corpus: list[list[str]] = []
        self._bm25 = None

    # -- build ---------------------------------------------------------------

    def build_index(self, chunks: list[RetrievalResult]) -> None:
        """Build BM25 index from chunk contents.

        Args:
            chunks: List of RetrievalResult or Chunk objects.
        """
        from rank_bm25 import BM25Okapi

        self._chunks = []
        self._doc_texts = []
        self._tokenized_corpus = []

        for ch in chunks:
            content = getattr(ch, "content", "") or getattr(ch, "raw_content", "")
            if not content.strip():
                continue
            tokens = tokenize_vi(content)
            if not tokens:
                continue

            # Build metadata — try .metadata dict first (RetrievalResult),
            # fall back to Chunk attributes
            ch_meta = getattr(ch, "metadata", {})
            if ch_meta:
                meta = {k: str(v) if isinstance(v, (int, float, bool)) else v for k, v in ch_meta.items()}
            else:
                meta = {
                    "article_id": getattr(ch, "article_id", ""),
                    "unit_type": getattr(ch, "unit_type", "ARTICLE"),
                    "order": getattr(ch, "order", 0),
                    "hierarchy_path": getattr(ch, "hierarchy_path", ""),
                    "document_id": getattr(ch, "document_id", ""),
                    "document_type": getattr(ch, "document_type", ""),
                    "effective_date": getattr(ch, "effective_date", ""),
                    "status": getattr(ch, "status", ""),
                    "topic_id": getattr(ch, "topic_id", ""),
                    "topic_name": getattr(ch, "topic_name", ""),
                    "chapter_title": getattr(ch, "chapter_title", ""),
                    "map_code": getattr(ch, "map_code", ""),
                    "benchmark_id": str(getattr(ch, "benchmark_id", "") or ""),
                }

            self._chunks.append({
                "chunk_id": getattr(ch, "chunk_id", ""),
                "article_id": getattr(ch, "article_id", ""),
                "unit_type": getattr(ch, "unit_type", "ARTICLE"),
                "content": content,
                "raw_content": getattr(ch, "raw_content", content),
                "metadata": meta,
            })
            self._doc_texts.append(content)
            self._tokenized_corpus.append(tokens)

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )

        logger.info(
            "BM25 index built: %d documents, vocabulary from %d unique tokens",
            len(self._tokenized_corpus),
            sum(len(set(t)) for t in self._tokenized_corpus),
        )

    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._tokenized_corpus) > 0

    def has_benchmark_metadata(self) -> bool:
        """Check if stored chunks carry benchmark_id metadata."""
        if not self._chunks:
            return False
        return any(
            item.get("metadata", {}).get("benchmark_id", "")
            for item in self._chunks
        )

    # -- search --------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """Search BM25 index and return top-k results.

        Args:
            query: Vietnamese query string.

        Returns:
            List of dicts with keys: chunk_id, content, article_id, unit_type,
            bm25_score.
        """
        if not self.is_built():
            return []

        query_tokens = tokenize_vi(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        k = min(self.top_k, len(scores))
        if k == 0:
            return []

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1.0

        results: list[dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                **self._chunks[idx],
                "bm25_score": min(scores[idx] / max_score, 1.0),
            })

        return results

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist BM25 index to disk via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self._chunks,
            "doc_texts": self._doc_texts,
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
            "top_k": self.top_k,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved to %s (%d docs)", path, len(self._chunks))

    @classmethod
    def load(cls, path: str | Path) -> Optional["BM25Retriever"]:
        """Load BM25 index from disk. Returns None if file missing."""
        path = Path(path)
        if not path.exists():
            return None

        with open(path, "rb") as f:
            data = pickle.load(f)

        retriever = cls(
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            top_k=data.get("top_k", 20),
        )
        retriever._chunks = data["chunks"]
        retriever._doc_texts = data["doc_texts"]
        retriever._tokenized_corpus = data["tokenized_corpus"]

        # Rebuild BM25Okapi
        from rank_bm25 import BM25Okapi
        if retriever._tokenized_corpus:
            retriever._bm25 = BM25Okapi(
                retriever._tokenized_corpus,
                k1=retriever.k1,
                b=retriever.b,
            )

        logger.info("BM25 index loaded from %s (%d docs)", path, len(retriever._chunks))
        return retriever
