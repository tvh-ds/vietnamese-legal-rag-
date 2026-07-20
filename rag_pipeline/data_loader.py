"""Data loader: parses VBQPPL JSON files into structured Article objects.

The JSON files follow a nested hierarchy:
  Chủ đề → Đề mục → Chương → Điều (Article)

Only the Article level carries meaningful legal content for retrieval, so the loader
flattens everything above into Article-level metadata.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """A single legal article with all hierarchical metadata attached."""

    article_id: str                     # Điều ID (UUID)
    map_code: str                       # Điều MAPC (hierarchical navigation code)
    title: str                          # Tên điều (e.g. "Điều 4.1.LQ.1. Phạm vi điều chỉnh")
    content: str                        # Nội dung — the legal text

    # Document-level metadata extracted from Ghi chú
    document_id: str                    # Normalized law number (e.g. "01/2016/QH14")
    document_type: str                  # "Luật", "Nghị định", "Thông tư", "Bộ luật", etc.
    effective_date: str                 # Hiệu lực date (DD/MM/YYYY)
    status: str                         # "CÓ_HIỆU_LỰC_THI_HÀNH" | "HẾT_HIỆU_LỰC" | etc.
    source_link: str                    # Link to full document

    # Hierarchical metadata
    topic_id: str                       # Chủ đề ID
    topic_name: str                     # Tên chủ đề
    topic_number: str                   # Chủ đề số
    section_id: str                     # Đề mục ID
    section_name: str                   # Tên đề mục
    chapter_id: str                     # Chương ID
    chapter_title: str                  # Tên chương
    chapter_index: str                  # Chỉ mục (I, II, III, …)

    # Cross-references
    references: list[dict] = field(default_factory=list)   # Chỉ dẫn

    # Bookkeeping
    appendix: list = field(default_factory=list)            # Phụ lục
    source_file: str = ""                                   # Originating JSON file

    # Benchmark ID (from content-matched Correct ID.json — stored as metadata, never embedded)
    benchmark_id: str = ""


# ---------------------------------------------------------------------------
# Document ID normalization patterns
# ---------------------------------------------------------------------------

# Extract law number: "Luật số 01/2016/QH14", "Nghị định số 162/2017/NĐ-CP",
# "Bộ luật số 91/2015/QH13", "Thông tư số 08/2016/TT-BGDĐT"
_DOC_PATTERN = re.compile(
    r"(Luật|Nghị\s*định|Thông\s*tư|Bộ\s*luật|Quyết\s*định|Pháp\s*lệnh|Nghị\s*quyết)"
    r"\s*số\s+([\d]+/[\d]+/[\w\-]+)",
)

# Also match abbreviated: "Luật số 01/2016/QH14"
_DOC_TYPE_PATTERN = re.compile(
    r"^(Luật|Nghị\s*định|Thông\s*tư|Bộ\s*luật|Quyết\s*định|Pháp\s*lệnh|Nghị\s*quyết)",
)


def _normalize_document_id(ghi_chu_text: str) -> tuple[str, str]:
    """Extract (document_type, document_id) from a Ghi chú string.

    Example input:
      "Điều 1 Luật số 01/2016/QH14 Đấu giá tài sản ngày 17/11/2016 …"
    Returns:
      ("Luật", "01/2016/QH14")
    """
    match = _DOC_PATTERN.search(ghi_chu_text)
    if match:
        doc_type = match.group(1).replace("\xa0", " ").strip()
        doc_id = match.group(2).strip()
        return (doc_type, doc_id)

    # Fallback: try to guess document type from the first word
    type_match = _DOC_TYPE_PATTERN.match(ghi_chu_text)
    doc_type = type_match.group(1) if type_match else "Văn bản"
    return (doc_type, "")


def _extract_effective_date(metadata_list: list[dict]) -> str:
    """Pull the effective date from the first metadata entry."""
    for entry in metadata_list:
        date = entry.get("Hiệu lực", "")
        if date:
            return date
    return ""


def _extract_status(metadata_list: list[dict]) -> str:
    """Pull the validity status from the first metadata entry."""
    for entry in metadata_list:
        status = entry.get("Thi hành", "")
        if status:
            return status
    return ""


def _extract_source_link(metadata_list: list[dict]) -> str:
    """Pull the source URL from the first metadata entry."""
    for entry in metadata_list:
        link = entry.get("Link", "")
        if link:
            return link
    return ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_articles(
    data_dir: str | Path = "data",
    id_map_path: str | Path | None = None,
) -> list[Article]:
    """Load all JSON files from *data_dir* and return a flat list of Articles.

    Each JSON file contains a root array of topic entries. Each topic entry
    contains chapters, and each chapter contains articles (Điều).

    If *id_map_path* is provided, loads a content→benchmark_id mapping
    (Correct ID.json format) and attaches benchmark IDs to articles via
    exact content matching.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Load benchmark ID mapping (content → numeric_id)
    id_map: dict[str, int] = {}
    if id_map_path is not None:
        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

    articles: list[Article] = []
    matched: int = 0
    json_files = sorted(data_path.glob("*.json"))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            raw_topics = json.load(f)

        for topic_entry in raw_topics:
            topic_id = topic_entry.get("Chủ đề ID", "")
            topic_name = topic_entry.get("Tên chủ đề", "")
            topic_number = topic_entry.get("Chủ đề số", "")
            section_id = topic_entry.get("Đề mục ID", "")
            section_name = topic_entry.get("Tên đề mục", "")
            chapter_id = topic_entry.get("Chương ID", "")
            chapter_title = topic_entry.get("Tên chương", "")
            chapter_index = topic_entry.get("Chỉ mục", "")

            # Each topic entry has one chapter; the articles live inside
            dieu_list = topic_entry.get("Các điều", [])
            if not dieu_list:
                continue

            for dieu in dieu_list:
                # --- Article-level fields ---
                article_id = dieu.get("Điều ID", "")
                map_code = dieu.get("Điều MAPC", "")
                title = dieu.get("Tên điều", "")
                content = dieu.get("Nội dung", "")
                appendix = dieu.get("Phụ lục", [])

                # --- References ---
                raw_refs = dieu.get("Chỉ dẫn", [])
                references: list[dict] = []
                for ref in raw_refs:
                    references.append({
                        "name": ref.get("Điều liên quan", ""),
                        "id": ref.get("ID Điều liên quan"),
                        "link": ref.get("Link", "#"),
                    })

                # --- Ghi chú (document metadata) ---
                ghi_chu = dieu.get("Ghi chú", {})
                ghi_chu_text = ghi_chu.get("Ghi chú", "") if isinstance(ghi_chu, dict) else ""
                metadata_list = ghi_chu.get("metadata", []) if isinstance(ghi_chu, dict) else []

                doc_type, doc_id = _normalize_document_id(ghi_chu_text)
                effective_date = _extract_effective_date(metadata_list)
                status = _extract_status(metadata_list)
                source_link = _extract_source_link(metadata_list)

                # Benchmark ID — content-based matching
                benchmark_id = ""
                if id_map:
                    bid = id_map.get(content.strip())
                    if bid is not None:
                        benchmark_id = str(bid)
                        matched += 1

                articles.append(Article(
                    article_id=article_id,
                    map_code=map_code,
                    title=title,
                    content=content,
                    document_id=doc_id,
                    document_type=doc_type,
                    effective_date=effective_date,
                    status=status,
                    source_link=source_link,
                    topic_id=topic_id,
                    topic_name=topic_name,
                    topic_number=topic_number,
                    section_id=section_id,
                    section_name=section_name,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    chapter_index=chapter_index,
                    references=references,
                    appendix=appendix,
                    source_file=json_file.name,
                    benchmark_id=benchmark_id,
                ))

    if id_map:
        import logging
        logging.getLogger(__name__).info(
            "Benchmark ID matches: %d/%d articles", matched, len(articles),
        )
    return articles


# ---------------------------------------------------------------------------
# CLI smoketest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    arts = load_articles(data_dir)
    print(f"Loaded {len(arts)} articles from {data_dir}")

    # Print first 3 articles summary
    for a in arts[:3]:
        print(f"  [{a.document_type} {a.document_id}] {a.title}")
        print(f"    Status: {a.status} | Effective: {a.effective_date}")
        print(f"    Topic: {a.topic_name} | Chapter: {a.chapter_title}")
        print(f"    Content preview: {a.content[:120]}…")
        print(f"    Refs: {len(a.references)}")
        print()
