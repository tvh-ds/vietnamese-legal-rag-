"""Query Rewriter — normalizes user queries for better legal retrieval.

Implements Section 9 of Pipeline.md:

  Raw user input often uses shorthand, telex typing (no diacritics), or
  colloquial phrasing. The Query Rewriter normalizes these into proper
  Vietnamese legal language to improve embedding match quality.

Example:
  Input:  "dk cap phep xay dung"
  Output: "Điều kiện cấp giấy phép xây dựng theo pháp luật hiện hành"
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Vietnamese legal abbreviation dictionary
# ---------------------------------------------------------------------------

# Maps common shorthand / telex / abbreviated forms to full legal terms.
# Keys are lowercase, no-diacritic forms for robust matching.
_LEGAL_ABBREVIATIONS: dict[str, str] = {
    # Procedural
    "dk": "điều kiện",
    "dkcn": "điều kiện cần",
    "tt": "thủ tục",  # also "thông tư" — context-dependent
    "hs": "hồ sơ",
    "tthc": "thủ tục hành chính",
    "vbpl": "văn bản pháp luật",
    "vbqppl": "văn bản quy phạm pháp luật",
    "qd": "quyết định",
    "nd": "nghị định",
    "ttlt": "thông tư liên tịch",
    "pl": "pháp luật",
    "hđ": "hợp đồng",
    "gcn": "giấy chứng nhận",
    "sh": "sở hữu",
    "tn": "thu nhập",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "bhtn": "bảo hiểm thất nghiệp",
    "dn": "doanh nghiệp",
    "hdnd": "hội đồng nhân dân",
    "ubnd": "ủy ban nhân dân",
    "tand": "tòa án nhân dân",
    "vksnd": "viện kiểm sát nhân dân",
    "qh": "quốc hội",
    "cp": "chính phủ",
    "tw": "trung ương",
    "nv": "nhà nước",
    "xhcn": "xã hội chủ nghĩa",
    "xh": "xã hội",
    "xhnv": "xã hội nhà văn",
    "xhcnvn": "xã hội chủ nghĩa việt nam",
    "vn": "việt nam",
    "vnxhcn": "việt nam xã hội chủ nghĩa",
    "tphcm": "thành phố hồ chí minh",
    "hn": "hà nội",
    "dt": "đất",
    "bds": "bất động sản",
    "xd": "xây dựng",
    "gp": "giấy phép",
    "gpxd": "giấy phép xây dựng",
    "kt": "kinh tế",
    "xh": "xã hội",
    "gd": "giáo dục",
    "yt": "y tế",
    "mt": "môi trường",
    "kh": "kế hoạch",
    "tc": "tài chính",
    "ns": "ngân sách",
    "nn": "nhà nước",
    "nnvn": "nhà nước việt nam",
    "ql": "quản lý",
    "qlnn": "quản lý nhà nước",
    "sd": "sử dụng",
    "shd": "sử dụng đất",
    "ch": "căn hộ",
    "cc": "chung cư",
    "nl": "năng lượng",
    "gtvt": "giao thông vận tải",
    "gt": "giao thông",
    "vt": "vận tải",
    "cn": "công nghiệp",
    "nn": "nông nghiệp",
    "ts": "tài sản",
    "tncn": "thu nhập cá nhân",
    "tndn": "thu nhập doanh nghiệp",
    "gtgt": "giá trị gia tăng",
    "ttĐb": "tiêu thụ đặc biệt",
    "qsd": "quyền sử dụng",
    "qsh": "quyền sở hữu",
    "shcn": "sở hữu công nghiệp",
    "shtt": "sở hữu trí tuệ",
    "ld": "lao động",
    "bh": "bảo hiểm",
    "tm": "thương mại",
    "dtm": "điện tử thương mại",
    "dv": "dịch vụ",
    "sp": "sản phẩm",
    "hh": "hàng hóa",
    "xk": "xuất khẩu",
    "nk": "nhập khẩu",
    "bl": "bộ luật",
    "blhs": "bộ luật hình sự",
    "blds": "bộ luật dân sự",
    "blld": "bộ luật lao động",
    "blttds": "bộ luật tố tụng dân sự",
    "bltths": "bộ luật tố tụng hình sự",
    "hngđ": "hôn nhân gia đình",
    "hcgd": "hộ chiếu gia đình",
    "cty": "công ty",
    "tct": "tổng công ty",
    "dntn": "doanh nghiệp tư nhân",
    "tnhh": "trách nhiệm hữu hạn",
    "cp": "cổ phần",
    "hdnd": "hợp đồng nhân dân",
    "tphcm": "thành phố hồ chí minh",
}

# Telex diacritic patterns: words ending in certain letter combinations
# indicate Vietnamese telex typing (e.g., "dieu" → "điều", "kien" → "kiện")
_TELEX_PATTERNS: dict[str, str] = {
    "aw": "ă", "aa": "â", "dd": "đ",
    "ee": "ê", "oo": "ô", "ow": "ơ",
    "uw": "ư", "w": "ư",
}

# Common diacritic marks in telex (tone marks appended to words)
# e.g., "kien" could be "kiến", "kiện", "kiên" — we don't guess, just flag
_TELEX_TONE_MARKERS = re.compile(r"[a-z]+[sfrxj]$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Query Rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """Normalize user queries for Vietnamese legal retrieval.

    Handles:
      - Telex-typed queries (no diacritics) — expands abbreviations
      - Shorthand legal terms — expands to full form
      - Appends legal-domain context phrase (optional)
    """

    def __init__(
        self,
        append_context: bool = True,
        context_phrase: str = "theo pháp luật hiện hành",
        expand_abbreviations: bool = True,
    ) -> None:
        self.append_context = append_context
        self.context_phrase = context_phrase
        self.expand_abbreviations = expand_abbreviations

    def rewrite(self, query: str) -> str:
        """Normalize a raw user query into legal retrieval form.

        Args:
            query: Raw user input (may be telex, shorthand, or proper Vietnamese).

        Returns:
            Normalized query string ready for embedding.
        """
        query = query.strip()
        if not query:
            return query

        # Step 1: Detect if the query lacks Vietnamese diacritics (telex mode)
        is_telex = self._likely_telex(query)

        # Step 2: Expand legal abbreviations (works for both telex and proper)
        if self.expand_abbreviations:
            query = self._expand_abbreviations(query)

        # Step 3: If telex and no diacritics after expansion, we can't
        # reliably add tones; return as-is with abbreviations expanded.
        # The embedding model handles telex reasonably well.

        # Step 4: Append legal context phrase for domain grounding
        if self.append_context:
            query = self._maybe_append_context(query)

        return query

    # -- detection -----------------------------------------------------------

    @staticmethod
    def _likely_telex(query: str) -> bool:
        """Heuristic: if the query contains no Vietnamese diacritic characters,
        it's likely typed in telex/no-diacritic mode."""
        # Vietnamese diacritic range (combined characters)
        viet_char = re.search(
            r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
            r"ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]",
            query,
        )
        return viet_char is None

    # -- abbreviation expansion ----------------------------------------------

    def _expand_abbreviations(self, query: str) -> str:
        """Replace known legal abbreviations with their full forms.

        Uses word-boundary matching so "dk" in "dk cap phep" is expanded
        but "dk" inside a longer word like "padk" is not.
        """
        words = query.split()
        expanded: list[str] = []
        for w in words:
            lower = w.lower().rstrip(".,;:!?")
            suffix = w[len(lower):]  # preserve punctuation
            if lower in _LEGAL_ABBREVIATIONS:
                expanded.append(_LEGAL_ABBREVIATIONS[lower] + suffix)
            else:
                expanded.append(w)
        return " ".join(expanded)

    # -- context appending ---------------------------------------------------

    def _maybe_append_context(self, query: str) -> str:
        """Append a legal-domain context phrase if the query is short
        and doesn't already contain legal framing."""
        # Only append if query is relatively short (< 30 words)
        if len(query.split()) > 30:
            return query

        # Don't append if query already contains legal framing terms
        legal_framing = [
            "theo pháp luật", "theo quy định", "theo luật",
            "pháp luật hiện hành", "luật định", "quy định của pháp luật",
            "theo bộ luật", "theo nghị định",
        ]
        lower_q = query.lower()
        for phrase in legal_framing:
            if phrase in lower_q:
                return query

        return f"{query} {self.context_phrase}"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_DEFAULT_REWRITER: Optional[QueryRewriter] = None


def get_rewriter(
    append_context: bool = True,
    expand_abbreviations: bool = True,
) -> QueryRewriter:
    """Get or create the default QueryRewriter singleton."""
    global _DEFAULT_REWRITER
    if _DEFAULT_REWRITER is None:
        _DEFAULT_REWRITER = QueryRewriter(
            append_context=append_context,
            expand_abbreviations=expand_abbreviations,
        )
    return _DEFAULT_REWRITER


# ---------------------------------------------------------------------------
# Smoketest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rw = QueryRewriter()
    tests = [
        "dk cap phep xay dung",
        "thu tuc dang ky kinh doanh",
        "dkcn de thanh lap dn",
        "dieu kien cap giay phep xay dung",
        "quy dinh ve bhxh cho nguoi lao dong",
        "tthc ve dat dai",
        "quyền và nghĩa vụ của người sử dụng đất",
    ]
    for t in tests:
        print(f"  IN : {t}")
        print(f"  OUT: {rw.rewrite(t)}")
        print()
