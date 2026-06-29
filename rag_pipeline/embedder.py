"""Embedding model wrapper for Vietnamese legal text.

Uses sentence-transformers with a multilingual model capable of handling
Vietnamese. The input to the embedder is the combined [Context] + [Chunk Content]
as specified in Section 7 of Pipeline.md.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Read HF_HUB_OFFLINE once at import time
_HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


class Embedder:
    """Lightweight wrapper around a sentence-transformers model.

    Usage::

        embedder = Embedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectors = embedder.embed(["chunk text 1", "chunk text 2"])
        # vectors.shape == (2, 384)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    # -- lazy load -----------------------------------------------------------

    @property
    def model(self):
        """Lazy-load the sentence-transformers model on first access.

        Tries local cache first to avoid HF Hub round-trips on every run.
        Falls back to downloading if the model isn't cached yet.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Silence httpx INFO logs from HF Hub checks
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)

            # Try local-only first (no network). Fall back to download.
            kwargs: dict = {"device": self.device}
            if not _HF_OFFLINE:
                try:
                    self._model = SentenceTransformer(
                        self.model_name, local_files_only=True, **kwargs
                    )
                    return self._model
                except (OSError, FileNotFoundError, EnvironmentError):
                    logger.info("Model not cached locally — downloading from HuggingFace…")

            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        # get_sentence_embedding_dimension was renamed in newer versions
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]

    # -- embed ---------------------------------------------------------------

    def embed(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Convert a sequence of texts into embedding vectors.

        Args:
            texts: List of chunk content strings (already context-injected).
            show_progress: Show a progress bar via tqdm.

        Returns:
            NumPy array of shape (len(texts), dim) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,          # cosine similarity ready
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns 1-D array of shape (dim,)."""
        return self.embed([text])[0]
