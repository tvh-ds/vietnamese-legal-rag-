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


# ---------------------------------------------------------------------------
# API-based embedder (company embedding API)
# ---------------------------------------------------------------------------

class APIEmbedder:
    """Embedding via OpenAI-compatible company API.

    Same interface as Embedder::

        embedder = APIEmbedder(base_url="https://...", api_key="...", model="...")
        vectors = embedder.embed(["chunk text 1", "chunk text 2"])
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "Vietnamese_Embedding",
        batch_size: int = 64,
        timeout_sec: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout_sec = timeout_sec
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Dimensionality — determined on first embed call."""
        if self._dim is None:
            v = self.embed(["dim probe"])  # type: ignore[assignment]
            self._dim = int(v.shape[1])
        return self._dim

    def embed(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Embed texts via API, returning (len(texts), dim) float32 array."""
        import time as _time
        import requests

        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        all_vectors: list[np.ndarray] = []
        texts_list = list(texts)
        total_batches = (len(texts_list) + self.batch_size - 1) // self.batch_size

        it = range(0, len(texts_list), self.batch_size)
        if show_progress:
            from tqdm import tqdm
            it = tqdm(it, total=total_batches, desc="  Embedding", unit="batch")

        url = f"{self.base_url}/v1/embeddings"
        for i in it:
            batch = texts_list[i : i + self.batch_size]

            for attempt in range(3):
                try:
                    resp = requests.post(
                        url,
                        json={"model": self.model, "input": batch},
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=self.timeout_sec,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if "data" in data:
                        for item in data["data"]:
                            all_vectors.append(np.array(item["embedding"], dtype=np.float32))
                    elif "embeddings" in data:
                        for emb in data["embeddings"]:
                            all_vectors.append(np.array(emb, dtype=np.float32))
                    else:
                        raise ValueError(f"Unexpected API response: {list(data.keys())}")
                    break

                except (requests.ConnectionError, requests.Timeout) as e:
                    if attempt < 2:
                        wait = (attempt + 1) * 2
                        logger.warning("Embedding API retry %d/3 after %ds: %s", attempt + 1, wait, e)
                        _time.sleep(wait)
                    else:
                        raise

        result = np.stack(all_vectors)
        return result.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]


def create_embedder(config) -> Embedder | APIEmbedder:
    """Factory: return the right embedder based on config."""
    if config.embedding.backend == "api":
        return APIEmbedder(
            base_url=getattr(config, "_api_base_url", ""),
            api_key=getattr(config, "_api_key", ""),
            model=config.embedding.api.model,
            batch_size=config.embedding.api.batch_size,
            timeout_sec=config.embedding.api.timeout_sec,
        )
    else:
        return Embedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
        )
