"""Recursive semantic chunker for Vietnamese legal articles.

Implements Sections 3, 4, and 5 of Pipeline.md:

  Step 1: Check Article size → keep as one chunk if small.
  Step 2: Split into numbered Clauses (Khoản) — "1. …", "2. …"
  Step 3: Split oversized Clauses into lettered Points (Điểm) — "a) …", "b) …"
  Step 4: Late chunking — paragraph split for articles without Clause/Point structure.
  Fallback: Sentence-aware splitting for paragraphs that remain too large.

After chunking:
  - Quality validation: merge short chunks, enforce sentence boundaries, deduplicate.
  - Context injection: prepend a heuristic article-level summary to each chunk.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from data_loader import Article


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# Vietnamese + multilingual models: ~2.5–3 chars per token for Latin-script text.
# Used only as a fallback when the real tokenizer is unavailable.
_CHARS_PER_TOKEN = 3.0

# Cache for the real tokenizer (lazy-loaded)
_tokenizer = None
_tokenizer_model = "keepitreal/vietnamese-sbert"


def set_tokenizer_model(model_name: str) -> None:
    """Set the tokenizer model name (should match the embedding model)."""
    global _tokenizer, _tokenizer_model
    if model_name != _tokenizer_model:
        _tokenizer = None  # Force reload with new model
        _tokenizer_model = model_name


def _get_tokenizer():
    """Lazy-load the HuggingFace tokenizer for the embedding model."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            import logging as _logging
            # Suppress "Token indices sequence length" warning
            _logging.getLogger("transformers.tokenization_utils_base").setLevel(
                _logging.ERROR
            )
            # Silence HF Hub network checks when model is cached
            _logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)
            _logging.getLogger("httpx").setLevel(_logging.WARNING)
            _logging.getLogger("httpcore").setLevel(_logging.WARNING)

            # Try local-only first, fall back to download
            try:
                _tokenizer = AutoTokenizer.from_pretrained(
                    _tokenizer_model,
                    model_max_length=10_000_000,
                    local_files_only=True,
                )
            except (OSError, FileNotFoundError, EnvironmentError):
                _tokenizer = AutoTokenizer.from_pretrained(
                    _tokenizer_model,
                    model_max_length=10_000_000,
                )
        except Exception:
            _tokenizer = False  # Sentinel: fall back to char heuristic
    return _tokenizer if _tokenizer is not False else None


def estimate_tokens(text: str) -> int:
    """Count tokens using the real tokenizer, falling back to char heuristic.

    Uses the same tokenizer as the embedding model (WordPiece for MiniLM)
    for accurate chunk-size gating.
    """
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single semantic chunk ready for embedding."""

    chunk_id: str                           # Unique chunk UUID
    article_id: str                         # Parent article ID (Điều ID)
    content: str                            # Chunk text (context + original)
    raw_content: str                        # Original text before context injection
    unit_type: str                          # "ARTICLE" | "CLAUSE" | "POINT" | "PARAGRAPH" | "SENTENCE"
    parent_unit_id: str                     # Article or Clause ID
    order: int                              # Position within parent
    token_count: int                        # Estimated tokens
    hierarchy_path: str                     # e.g. "article_id/clause_1/point_a"

    # Document-level metadata (carried through from Article)
    document_id: str = ""
    document_type: str = ""
    effective_date: str = ""
    status: str = ""
    topic_id: str = ""
    topic_name: str = ""
    chapter_title: str = ""
    map_code: str = ""

    # Benchmark ID (content-matched — stored as metadata, never embedded)
    benchmark_id: str = ""


# ---------------------------------------------------------------------------
# Regex patterns for structure detection
# ---------------------------------------------------------------------------

# Numbered clause: "1. ", "2. ", "10. " — may be preceded by optional whitespace
_CLAUSE_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)", re.MULTILINE)

# Lettered point: "a) ", "b) ", "đ) " — Vietnamese alphabet, may be indented
_POINT_RE = re.compile(r"^(\s*)([a-đ])\)\s+(.*)", re.MULTILINE)

# Vietnamese sentence boundary characters
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ỸĐ])")

# Paragraph split: two or more newlines
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

@dataclass
class Chunker:
    """Recursive semantic chunker for Vietnamese legal text."""

    max_tokens: int = 512
    min_tokens: int = 30
    overlap_tokens: int = 0
    enable_context: bool = True
    max_context_tokens: int = 100
    token_counter: Callable[[str], int] = estimate_tokens
    llm_context_generator: object | None = None  # LLMContextGenerator instance

    # -- public API ----------------------------------------------------------

    def chunk_article(self, article: Article) -> list[Chunk]:
        """Produce semantic chunks for a single Article."""
        content = article.content.strip()
        if not content:
            return []

        token_count = self.token_counter(content)

        # Step 1: If the whole article fits under max_tokens, keep it as one chunk
        if token_count <= self.max_tokens:
            chunks = [self._make_chunk(
                article=article,
                content=content,
                unit_type="ARTICLE",
                parent_id=article.article_id,
                order=1,
            )]
        else:
            # Try recursive splitting
            chunks = self._recursive_split(article, content, article.article_id)

        # Quality validation
        chunks = self._validate_and_merge(chunks, article)

        # Context injection
        if self.enable_context:
            context = self._generate_context(article)
            chunks = self._inject_context(chunks, context)

        # Renumber orders after all splits/merges
        self._renumber(chunks)

        return chunks

    # -- splitting -----------------------------------------------------------

    def _recursive_split(
        self, article: Article, text: str, parent_id: str
    ) -> list[Chunk]:
        """Try Clause → Point → Paragraph → Sentence, stopping when chunks fit."""
        # Step 2: Split by numbered clauses
        clauses = self._split_by_clauses(text)
        if len(clauses) > 1 and self._any_too_large(clauses):
            # One or more clauses are still too big — try splitting each into points
            chunks: list[Chunk] = []
            for i, clause_text in enumerate(clauses, start=1):
                clause_id = f"{parent_id}/clause_{i}"
                if self.token_counter(clause_text) <= self.max_tokens:
                    chunks.append(self._make_chunk(
                        article, clause_text, "CLAUSE", parent_id, i,
                    ))
                else:
                    points = self._split_by_points(clause_text)
                    if len(points) > 1:
                        for j, point_text in enumerate(points, start=1):
                            point_id = f"{clause_id}/point_{chr(96 + j)}"
                            if self.token_counter(point_text) <= self.max_tokens:
                                chunks.append(self._make_chunk(
                                    article, point_text, "POINT", clause_id, j,
                                ))
                            else:
                                # Point still too large → sentence-aware fallback
                                chunks.extend(self._sentence_split(
                                    article, point_text, point_id, "POINT",
                                ))
                    else:
                        # No point structure → sentence-aware fallback
                        chunks.extend(self._sentence_split(
                            article, clause_text, clause_id, "CLAUSE",
                        ))
            return chunks
        elif len(clauses) > 1:
            # All clauses fit — use them directly
            return [
                self._make_chunk(article, t, "CLAUSE", parent_id, i)
                for i, t in enumerate(clauses, start=1)
            ]

        # Step 4: Late chunking — no Clause/Point structure
        paragraphs = self._split_paragraphs(text)
        if len(paragraphs) > 1:
            chunks = []
            for i, para in enumerate(paragraphs, start=1):
                if self.token_counter(para) <= self.max_tokens:
                    chunks.append(self._make_chunk(
                        article, para, "PARAGRAPH", parent_id, i,
                    ))
                else:
                    chunks.extend(self._sentence_split(
                        article, para, f"{parent_id}/para_{i}", "PARAGRAPH",
                    ))
            return chunks

        # Final fallback: sentence-aware chunking for the whole text
        return self._sentence_split(article, text, parent_id, "ARTICLE")

    # -- structure detectors -------------------------------------------------

    @staticmethod
    def _split_by_clauses(text: str) -> list[str]:
        """Split text into numbered clauses (1. … 2. …).

        Returns a list of clause texts. If no clause numbering is found,
        returns a single-element list containing *text*.
        """
        # Find all clause-start positions
        starts: list[int] = []
        for m in _CLAUSE_RE.finditer(text):
            starts.append(m.start())

        if len(starts) < 2:
            return [text]

        clauses: list[str] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            clause_text = text[start:end].strip()
            if clause_text:
                clauses.append(clause_text)
        return clauses

    @staticmethod
    def _split_by_points(text: str) -> list[str]:
        """Split text into lettered points (a) … b) …).

        Returns a list of point texts. If no point letters are found,
        returns a single-element list containing *text*.
        """
        starts: list[int] = []
        for m in _POINT_RE.finditer(text):
            starts.append(m.start())

        if len(starts) < 2:
            return [text]

        points: list[str] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            point_text = text[start:end].strip()
            if point_text:
                points.append(point_text)
        return points

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split text into paragraphs on double-newline boundaries."""
        parts = _PARAGRAPH_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _sentence_split(
        self, article: Article, text: str, parent_id: str, unit_type: str,
    ) -> list[Chunk]:
        """Sentence-aware fallback: split on sentence boundaries, merge into
        max_tokens-sized chunks."""
        sentences = _SENTENCE_BOUNDARY.split(text)
        if len(sentences) <= 1:
            # Can't split further — force as one chunk
            return [self._make_chunk(article, text, unit_type, parent_id, 1)]

        chunks: list[Chunk] = []
        buffer = ""
        order = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate = f"{buffer} {sent}".strip() if buffer else sent
            if self.token_counter(candidate) <= self.max_tokens:
                buffer = candidate
            else:
                if buffer:
                    order += 1
                    chunks.append(self._make_chunk(
                        article, buffer, "SENTENCE", parent_id, order,
                    ))
                # If the single sentence itself exceeds max, force it as a chunk
                if self.token_counter(sent) > self.max_tokens:
                    order += 1
                    chunks.append(self._make_chunk(
                        article, sent, "SENTENCE", parent_id, order,
                    ))
                    buffer = ""
                else:
                    buffer = sent

        if buffer:
            order += 1
            chunks.append(self._make_chunk(
                article, buffer, "SENTENCE", parent_id, order,
            ))

        return chunks

    # -- quality validation (Section 4) --------------------------------------

    def _validate_and_merge(
        self, chunks: list[Chunk], article: Article,
    ) -> list[Chunk]:
        """Validate chunk quality and merge undersized chunks."""
        if not chunks:
            return []

        # Merge short chunks with adjacent ones
        merged: list[Chunk] = []
        for ch in chunks:
            if ch.token_count < self.min_tokens and merged:
                # Merge with previous chunk
                prev = merged[-1]
                prev.content = f"{prev.raw_content}\n{ch.raw_content}"
                prev.raw_content = prev.content
                prev.token_count = self.token_counter(prev.content)
            else:
                merged.append(ch)

        # Deduplicate by content hash
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for ch in merged:
            key = ch.raw_content.strip()
            if key not in seen:
                seen.add(key)
                deduped.append(ch)

        return deduped

    # -- context generation & injection (Section 5) --------------------------

    def _generate_context(self, article: Article) -> str:
        """Generate a short semantic context for the article.

        Uses LLM via API when configured, falls back to heuristic
        (title + first sentence) otherwise.
        """
        # Try LLM first
        if self.llm_context_generator is not None:
            llm_summary = self.llm_context_generator.generate(
                title=article.title,
                content=article.content,
            )
            if llm_summary:
                return self._truncate_context(llm_summary)

        # Heuristic fallback
        content = article.content.strip()
        first_sent = content.split(".")[0].strip() if content else ""
        title_clean = article.title.split(". ", 1)[-1] if ". " in article.title else article.title
        combined = f"Điều: {title_clean}"
        if first_sent and first_sent != title_clean:
            combined = f"{combined}. {first_sent}"
        return self._truncate_context(combined)

    def _truncate_context(self, text: str) -> str:
        """Truncate context to approximately max_context_tokens."""
        max_chars = self.max_context_tokens * _CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text

        # Cut at last sentence boundary within limit
        truncated = text[: int(max_chars)]
        last_dot = truncated.rfind(".")
        if last_dot > max_chars * 0.5:
            return truncated[: last_dot + 1]
        return truncated.rsplit(" ", 1)[0] + "…"

    def _inject_context(self, chunks: list[Chunk], context: str) -> list[Chunk]:
        """Prepend the article context to each chunk's content."""
        for ch in chunks:
            ch.content = f"{context}\n{ch.raw_content}"
            ch.token_count = self.token_counter(ch.content)
        return chunks

    # -- helpers -------------------------------------------------------------

    def _make_chunk(
        self,
        article: Article,
        content: str,
        unit_type: str,
        parent_id: str,
        order: int,
    ) -> Chunk:
        chunk_id = str(uuid.uuid4())
        return Chunk(
            chunk_id=chunk_id,
            article_id=article.article_id,
            content=content,
            raw_content=content,
            unit_type=unit_type,
            parent_unit_id=parent_id,
            order=order,
            token_count=self.token_counter(content),
            hierarchy_path=f"{parent_id}/{unit_type.lower()}_{order}",
            document_id=article.document_id,
            document_type=article.document_type,
            effective_date=article.effective_date,
            status=article.status,
            topic_id=article.topic_id,
            topic_name=article.topic_name,
            chapter_title=article.chapter_title,
            map_code=article.map_code,
            benchmark_id=getattr(article, "benchmark_id", ""),
        )

    @staticmethod
    def _renumber(chunks: list[Chunk]) -> None:
        """Reassign order numbers sequentially."""
        for i, ch in enumerate(chunks, start=1):
            ch.order = i

    @staticmethod
    def _any_too_large(texts: list[str]) -> bool:
        """Return True if any text in the list exceeds max_tokens."""
        return any(estimate_tokens(t) > 512 for t in texts)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def chunk_articles(
    articles: list[Article],
    max_tokens: int = 512,
    min_tokens: int = 30,
    enable_context: bool = True,
    max_context_tokens: int = 100,
    llm_context_generator: object | None = None,
) -> list[Chunk]:
    """Convenience: chunk a list of articles with default settings."""
    chunker = Chunker(
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        enable_context=enable_context,
        max_context_tokens=max_context_tokens,
        llm_context_generator=llm_context_generator,
    )
    all_chunks: list[Chunk] = []
    for article in articles:
        all_chunks.extend(chunker.chunk_article(article))
    return all_chunks


# ---------------------------------------------------------------------------
# Smoketest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    from data_loader import load_articles

    articles = load_articles(data_dir)
    print(f"Loaded {len(articles)} articles")

    chunker = Chunker(max_tokens=512, min_tokens=30)

    total_chunks = 0
    type_counts: dict[str, int] = {}
    size_dist: dict[str, int] = {"<100": 0, "100-300": 0, "300-512": 0, ">512": 0}

    for i, art in enumerate(articles):
        chunks = chunker.chunk_article(art)
        total_chunks += len(chunks)
        for ch in chunks:
            type_counts[ch.unit_type] = type_counts.get(ch.unit_type, 0) + 1
            tc = ch.token_count
            if tc < 100:
                size_dist["<100"] += 1
            elif tc < 300:
                size_dist["100-300"] += 1
            elif tc <= 512:
                size_dist["300-512"] += 1
            else:
                size_dist[">512"] += 1

        if (i + 1) % 3000 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles…")

    print(f"\nTotal chunks: {total_chunks}")
    print(f"Unit type distribution: {type_counts}")
    print(f"Size distribution: {size_dist}")

    # Show a few example chunks
    print("\n--- Example chunks ---")
    examples = chunker.chunk_article(articles[0])
    for ch in examples[:5]:
        print(f"  [{ch.unit_type}] {ch.chunk_id[:8]}… tokens={ch.token_count}")
        print(f"    Content: {ch.content[:150]}…")
        print()
