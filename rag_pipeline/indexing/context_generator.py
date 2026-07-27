"""LLM-powered context generation for Vietnamese legal articles.

Implements Section 5 of Pipeline.md: generates a 60-120 token semantic
summary of each article, which is then prepended to every chunk from
that article before embedding.

Uses the company's gemma-4-31B-it model via API.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Prompt template for Vietnamese legal article summarization
_SUMMARIZE_PROMPT = """Bạn là trợ lý pháp lý. Tóm tắt điều luật sau bằng tiếng Việt trong 2-3 câu (khoảng 60-120 token).
Chỉ tóm tắt nội dung chính, không thêm bình luận, không lặp lại tiêu đề.

Tiêu đề: {title}

Nội dung:
{content}

Tóm tắt:"""


class LLMContextGenerator:
    """Generate article-level context summaries via OpenAI-compatible LLM API.

    Usage::

        gen = LLMContextGenerator(base_url="https://...", api_key="...", model="gemma-4-31B-it")
        summary = gen.generate(article_title, article_content)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gemma-4-31B-it",
        temperature: float = 0.3,
        max_output_tokens: int = 120,
        timeout_sec: int = 30,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
        self.call_count = 0
        self.error_count = 0

    def generate(self, title: str, content: str) -> str:
        """Generate a Vietnamese legal summary for the article.

        Args:
            title: Article title (e.g. "Điều 4.1.LQ.1. Phạm vi điều chỉnh")
            content: Full article text.

        Returns:
            Summary string (60-120 tokens), or heuristic fallback on error.
        """
        content_trimmed = content[:8000]

        prompt = _SUMMARIZE_PROMPT.format(
            title=title,
            content=content_trimmed,
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            self.call_count += 1
            summary = resp.choices[0].message.content.strip()

            for prefix in ["Tóm tắt:", "Tóm tắt điều luật:", "Nội dung chính:"]:
                if summary.startswith(prefix):
                    summary = summary[len(prefix):].strip()

            if summary:
                if self.call_count % 500 == 0:
                    logger.info("LLM: %d summaries generated (%d errors)", self.call_count, self.error_count)
                return summary

        except Exception as exc:
            self.error_count += 1
            logger.warning("LLM context generation failed, using heuristic fallback: %s", exc)

        return self._heuristic_fallback(title, content)

    @staticmethod
    def _heuristic_fallback(title: str, content: str) -> str:
        """Fallback: title + first sentence."""
        title_clean = title.split(". ", 1)[-1] if ". " in title else title
        first_sent = content.split(".")[0].strip() if content else ""
        if first_sent and first_sent != title_clean:
            return f"Điều: {title_clean}. {first_sent}"
        return f"Điều: {title_clean}"
