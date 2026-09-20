from pathlib import Path

from src.ocr.domain.models.keyword import KeywordReport


class KeywordExporter:
    """
    Exports a KeywordReport as a Markdown table written to ``count_keywords.md``
    in the same directory as ``result.md``.
    """

    @staticmethod
    def export_markdown(report: KeywordReport, output_dir: Path) -> Path:
        """
        Writes ``count_keywords.md`` inside *output_dir* and returns its path.

        Format:
            | Từ khóa / Cụm từ | Số lần xuất hiện |
            | :--- | :---: |
            | keyword | count |
        """
        lines = [
            "# Báo Cáo Tần Suất Từ Khóa",
            "",
            f"**Tài liệu:** `{report.document_id}`  ",
            f"**Tổng số từ khóa tìm kiếm:** {report.total_keywords_searched}  ",
            f"**Tổng số lần khớp:** {report.total_matches}",
            "",
            "| Từ khóa / Cụm từ | Số lần xuất hiện |",
            "| :--- | :---: |",
        ]

        for kw in report.keywords:
            lines.append(f"| {kw.keyword} | {kw.count} |")

        lines.append("")  # trailing newline

        out_path = output_dir / "count_keywords.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[KEYWORDS] Exported keyword report -> {out_path}")
        return out_path
