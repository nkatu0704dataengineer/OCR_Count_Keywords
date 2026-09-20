import re
from pathlib import Path
from typing import List

from src.ocr.domain.models.keyword import KeywordCount, KeywordReport


# Default keywords to write when config/keywords.txt does not exist
_DEFAULT_KEYWORDS = [
    "Trí tuệ nhân tạo",
    "Đổi mới",
    "Chuyển đổi số",
    "Xấu xí",
]


class KeywordCounter:
    """
    Reads keywords from a text file and counts their occurrences in a document's
    result text.  Matching is case-insensitive and uses simple substring search
    with proper Unicode handling for Vietnamese diacritics.
    """

    @staticmethod
    def load_keywords(keywords_file: str) -> List[str]:
        """
        Loads keywords from the given file path (one keyword per line).
        If the file does not exist, creates it with a default set of sample keywords.
        Blank lines and leading/trailing whitespace are stripped.
        """
        kw_path = Path(keywords_file)

        if not kw_path.exists():
            kw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(kw_path, "w", encoding="utf-8") as f:
                for kw in _DEFAULT_KEYWORDS:
                    f.write(kw + "\n")
            print(f"[KEYWORDS] Created default keywords file: {kw_path}")

        keywords: List[str] = []
        with open(kw_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    keywords.append(stripped)

        return keywords

    @staticmethod
    def count_keywords(
        text: str,
        keywords: List[str],
        document_id: str = "",
        source_file: str = ""
    ) -> KeywordReport:
        """
        Counts occurrences of each keyword in *text* (case-insensitive).

        Uses ``re.findall`` with ``re.IGNORECASE | re.UNICODE`` so that
        Vietnamese diacritics are handled correctly.
        """
        text_lower = text.lower()
        results: List[KeywordCount] = []
        total_matches = 0

        for kw in keywords:
            # Use regex with escaped keyword for safe substring matching
            pattern = re.escape(kw.lower())
            matches = re.findall(pattern, text_lower, flags=re.UNICODE)
            count = len(matches)
            results.append(KeywordCount(keyword=kw, count=count))
            total_matches += count

        return KeywordReport(
            document_id=document_id,
            source_file=source_file,
            keywords=results,
            total_keywords_searched=len(keywords),
            total_matches=total_matches,
        )
