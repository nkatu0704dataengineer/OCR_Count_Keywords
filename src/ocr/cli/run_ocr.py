import argparse
import sys
from pathlib import Path
from typing import List, Optional
from tabulate import tabulate

from src.ocr.application.convert_document import ConvertDocumentUseCase

def parse_pages_arg(pages_str: Optional[str]) -> Optional[List[int]]:
    if not pages_str:
        return None

    pages = set()
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            sub = part.split("-")
            if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                start, end = int(sub[0]), int(sub[1])
                for p in range(start, end + 1):
                    pages.add(p)
        elif part.isdigit():
            pages.add(int(part))

    return sorted(list(pages)) if pages else None

def main():
    parser = argparse.ArgumentParser(
        description="Codex Redesign OCR v2.0 - Hybrid Document Extraction Pipeline"
    )
    parser.add_argument("input", help="Path to input PDF document")
    parser.add_argument("--profile", choices=["fast", "balanced", "accuracy"], default="balanced",
                        help="Processing profile (default: balanced)")
    parser.add_argument("--pages", type=str, default=None,
                        help="Specific pages or ranges (e.g. '1-5' or '1,3,5,11')")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: 'output')")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode and print timing metrics")
    parser.add_argument("--accuracy-report", action="store_true",
                        help="Evaluate CER, WER, and diacritic accuracy against ground truth")
    parser.add_argument("--ground-truth", type=str, default="benchmark/ground_truth",
                        help="Ground truth directory (default: 'benchmark/ground_truth')")
    parser.add_argument("--resume", action="store_true",
                        help="Resume processing by skipping pages cached in .cache/")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    parser.add_argument("--keywords-file", type=str, default=None,
                        help="Path to keywords text file (default: config/keywords.txt)")

    args = parser.parse_args()

    pages_list = parse_pages_arg(args.pages)
    converter = ConvertDocumentUseCase()

    print("=" * 60)
    print("CODEX OCR v2.0 HYBRID DOCUMENT EXTRACTION PIPELINE")
    print(f"File: {args.input} | Profile: {args.profile}")
    if pages_list:
        print(f"Pages: {pages_list}")
    if args.resume:
        print("Resume: ENABLED (using .cache)")
    print("=" * 60)

    try:
        doc = converter.execute(
            file_path=args.input,
            profile_name=args.profile,
            pages=pages_list,
            output_dir=args.output,
            resume=args.resume,
            benchmark=args.benchmark,
            accuracy_report=args.accuracy_report,
            ground_truth_dir=args.ground_truth,
            keywords_file=args.keywords_file
        )

        print("=" * 60)
        print("EXTRACTION COMPLETED SUCCESSFULLY")
        print(f"Document ID: {doc.document_id}")
        print(f"Total pages processed: {len(doc.pages)}")
        print(f"Total time: {doc.quality.total_time_seconds:.2f}s "
              f"({doc.quality.total_time_seconds / max(1, len(doc.pages)):.2f}s/page)")
        print(f"Pages native: {doc.quality.pages_processed_native} | "
              f"Pages OCR: {doc.quality.pages_processed_ocr} | "
              f"Fallback regions: {doc.quality.total_fallback_regions}")
        print(f"Output written to: {args.output}/{doc.document_id}/")
        print("=" * 60)

        if args.benchmark:
            sec_per_page = doc.quality.total_time_seconds / max(1, len(doc.pages))
            table_data = [
                ["Metric", "Value"],
                ["Total Wall Time (s)", f"{doc.quality.total_time_seconds:.2f}"],
                ["Seconds Per Page", f"{sec_per_page:.2f}"],
                ["Pages Per Minute", f"{(len(doc.pages) / max(0.01, doc.quality.total_time_seconds)) * 60:.2f}"],
                ["Mean OCR Confidence", f"{doc.quality.mean_ocr_confidence:.4f}"],
                ["Mean Quality Score", f"{doc.quality.mean_quality_score:.4f}"],
                ["Primary Lines", f"{doc.quality.metrics.primary_lines}"],
                ["Fallback Lines", f"{doc.quality.metrics.fallback_lines}"],
                ["Fallback Ratio", f"{doc.quality.metrics.fallback_ratio:.2%}"]
            ]
            print("\nBENCHMARK REPORT:")
            print(tabulate(table_data, headers="firstrow", tablefmt="github"))

        if args.accuracy_report and hasattr(doc, "accuracy") and doc.accuracy:
            acc = doc.accuracy
            tab_acc_str = f"{acc.mean_table_accuracy:.2%}" if acc.mean_table_accuracy is not None else "N/A"
            acc_data = [
                ["Metric", "Score"],
                ["Mean CER", f"{acc.mean_cer:.2%}"],
                ["Mean WER", f"{acc.mean_wer:.2%}"],
                ["Mean Diacritic Accuracy", f"{acc.mean_diacritic_accuracy:.2%}"],
                ["Mean Table Accuracy", tab_acc_str],
                ["Mean Reading Order Accuracy", f"{acc.mean_reading_order_accuracy:.2%}"],
                ["Pages Evaluated", f"{acc.pages_evaluated}"]
            ]
            print("\nACCURACY EVALUATION REPORT (vs Ground Truth):")
            print(tabulate(acc_data, headers="firstrow", tablefmt="github"))

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
