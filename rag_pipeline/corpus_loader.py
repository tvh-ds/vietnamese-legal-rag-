"""Corpus loader for the VLQA legal corpus format.

The VLQA corpus (legal_corpus.json) has a different structure than the
chu_de_*.json files. This module loads it and produces Article-like objects
that the existing chunker + embedder + vector store pipeline can consume.

Corpus format:
    [
      {
        "id": 0,
        "law_id": "14/2022/TT-NHNN",
        "content": [
          {"aid": 56789, "content_Article": "1. ..."},
          ...
        ]
      },
      ...
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CorpusArticle:
    """Minimal Article-like object for corpus indexing.

    Has the same interface that chunker.Chunker expects:
      - article_id: str (uses corpus aid as string)
      - title: str
      - content: str (the article text)
      - document_id, document_type, effective_date, status, etc.
    """

    article_id: str                     # corpus "aid" as string
    map_code: str = ""
    title: str = ""                     # "law_id + aid"
    content: str = ""                   # content_Article

    document_id: str = ""
    document_type: str = ""
    effective_date: str = ""
    status: str = ""
    source_link: str = ""

    topic_id: str = ""
    topic_name: str = ""
    topic_number: str = ""
    section_id: str = ""
    section_name: str = ""
    chapter_id: str = ""
    chapter_title: str = ""
    chapter_index: str = ""

    references: list[dict] = field(default_factory=list)
    appendix: list = field(default_factory=list)
    source_file: str = ""

    # Extra: corpus-specific
    corpus_law_id: str = ""             # law_id from corpus
    corpus_doc_id: int = 0              # document id from corpus


def load_corpus(path: str | Path) -> list[CorpusArticle]:
    """Load the VLQA legal corpus and return Article-like objects.

    Each article in the corpus becomes one CorpusArticle.
    The `article_id` is set to the string form of the corpus `aid`,
    so benchmark `relevant_laws` can be matched directly.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    articles: list[CorpusArticle] = []
    for doc in raw:
        law_id = doc.get("law_id", "")
        doc_id = doc.get("id", 0)
        for art in doc.get("content", []):
            aid = art.get("aid", 0)
            content_text = art.get("content_Article", "")
            if not content_text.strip():
                continue

            articles.append(CorpusArticle(
                article_id=str(aid),              # KEY: aid as string for benchmark matching
                title=f"{law_id} - điều {aid}",
                content=content_text,
                document_id=law_id,
                document_type=_guess_doc_type(law_id),
                status="CÓ_HIỆU_LỰC_THI_HÀNH",    # assumed for benchmark
                corpus_law_id=law_id,
                corpus_doc_id=doc_id,
            ))

    return articles


def _guess_doc_type(law_id: str) -> str:
    """Guess document type from law_id prefix."""
    law_id_lower = law_id.lower()
    if "nđ" in law_id_lower:
        return "Nghị định"
    if "tt" in law_id_lower:
        return "Thông tư"
    if "qđ" in law_id_lower:
        return "Quyết định"
    if "pl" in law_id_lower and "qh" in law_id_lower:
        return "Luật"
    if "nq" in law_id_lower:
        return "Nghị quyết"
    return "Văn bản"
