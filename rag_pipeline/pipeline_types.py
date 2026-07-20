"""Shared types used across the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata and score."""

    chunk_id: str
    content: str
    raw_content: str              # Without context prefix
    article_id: str
    unit_type: str                 # ARTICLE | CLAUSE | POINT | PARAGRAPH | SENTENCE
    order: int
    hierarchy_path: str
    score: float                   # Combined score (0.0 – 1.0, higher is better)
    source: str                    # "vector"
    metadata: dict
