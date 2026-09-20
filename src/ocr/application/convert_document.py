import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf
import yaml

from src.ocr.domain.models.document import ExtractedDocument
from src.ocr.domain.models.quality import DocumentQualityReport, QualityMetrics
from src.ocr.domain.routing.routing_policy import RoutingPolicy
from src.ocr.application.route_page import RoutePageUseCase
from src.ocr.application.process_page import ProcessPageUseCase
from src.ocr.application.export_document import ExportDocumentUseCase
from src.ocr.infrastructure.cache.artifact_cache import ArtifactCache
from src.ocr.infrastructure.export.manifest_exporter import ManifestExporter
from src.ocr.application.count_keywords import CountKeywordsUseCase

class ConvertDocumentUseCase:
    """
    Main orchestration use case to convert a PDF document using the Hybrid OCR v2 Pipeline.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            env_config = os.getenv("OCR_CONFIG_PATH")
            root_config = Path("config/profiles.yaml").resolve()
            bundled_config = Path(__file__).parent.parent / "config" / "profiles.yaml"

            if env_config and os.path.exists(env_config):
                config_path = env_config
            elif root_config.exists():
                config_path = str(root_config)
            else:
                config_path = str(bundled_config)

        self.profiles = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                self.profiles = loaded.get("profiles", {})

    def execute(
        self,
        file_path: str,
        profile_name: str = "balanced",
        pages: Optional[List[int]] = None,
        output_dir: str = "output",
        resume: bool = False,
        benchmark: bool = False,
        accuracy_report: bool = False,
        ground_truth_dir: str = "benchmark/ground_truth",
        keywords_file: Optional[str] = None
    ) -> ExtractedDocument:
        t0 = time.time()
        pdf_path = Path(file_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc_id = pdf_path.stem
        file_hash = ArtifactCache.compute_file_hash(str(pdf_path))

        config = self.profiles.get(profile_name, {
            "profile": profile_name,
            "render_dpi": 180,
            "fallback_enabled": True,
            "min_confidence": 0.80
        })

        doc = pymupdf.open(str(pdf_path))
        total_pages = len(doc)

        if pages is None or len(pages) == 0:
            target_pages = list(range(1, total_pages + 1))
        else:
            target_pages = [p for p in pages if 1 <= p <= total_pages]

        cache = ArtifactCache()
        routing_policy = RoutingPolicy(
            reliable_threshold=config.get("native_quality", {}).get("reliable_threshold", 0.90),
            review_threshold=config.get("native_quality", {}).get("review_threshold", 0.70)
        )
        router = RoutePageUseCase(routing_policy)
        page_processor = ProcessPageUseCase(doc, config, file_hash, cache)

        extracted_pages = []
        pages_native = 0
        pages_ocr = 0
        pages_mixed = 0
        total_fallback = 0

        for page_num in target_pages:
            p_result = page_processor.execute(
                page_num=page_num,
                total_pages=total_pages,
                router=router,
                resume=resume
            )
            extracted_pages.append(p_result)

            if p_result.page_type.value == "NATIVE_TEXT":
                pages_native += 1
            elif p_result.page_type.value == "MIXED":
                pages_mixed += 1
            else:
                pages_ocr += 1

            total_fallback += p_result.quality.fallback_count

        total_duration = time.time() - t0
        mean_native_score = sum(p.quality.native_text_score for p in extracted_pages) / len(extracted_pages) if extracted_pages else 1.0
        mean_ocr_conf = sum(p.quality.ocr_confidence for p in extracted_pages) / len(extracted_pages) if extracted_pages else 1.0
        mean_qual_score = sum(p.quality.quality_score for p in extracted_pages) / len(extracted_pages) if extracted_pages else 1.0

        total_primary_lines = sum(p.quality.total_blocks for p in extracted_pages)
        fb_ratio = (total_fallback / total_primary_lines) if total_primary_lines > 0 else 0.0

        quality_report = DocumentQualityReport(
            document_id=doc_id,
            total_pages=total_pages,
            mean_native_score=round(mean_native_score, 4),
            mean_ocr_confidence=round(mean_ocr_conf, 4),
            mean_quality_score=round(mean_qual_score, 4),
            pages_processed_native=pages_native,
            pages_processed_ocr=pages_ocr,
            pages_processed_mixed=pages_mixed,
            total_fallback_regions=total_fallback,
            total_time_seconds=round(total_duration, 2),
            page_reports=[p.quality for p in extracted_pages],
            metrics=QualityMetrics(
                mean_confidence=round(mean_ocr_conf, 4),
                primary_lines=total_primary_lines,
                fallback_lines=total_fallback,
                fallback_ratio=round(fb_ratio, 4)
            )
        )

        extracted_doc = ExtractedDocument(
            document_id=doc_id,
            file_path=str(pdf_path),
            file_hash=file_hash,
            total_pages=total_pages,
            selected_pages=target_pages,
            profile=profile_name,
            pages=extracted_pages,
            quality=quality_report
        )

        # Generate manifest
        extracted_doc.manifest = ManifestExporter.generate_manifest(extracted_doc, config)

        # Benchmark data if requested
        benchmark_data = None
        if benchmark:
            benchmark_data = {
                "document_id": doc_id,
                "profile": profile_name,
                "pages_benchmarked": target_pages,
                "wall_clock_seconds": round(total_duration, 2),
                "seconds_per_page": round(total_duration / len(target_pages), 2),
                "pages_per_minute": round((len(target_pages) / total_duration) * 60, 2),
                "primary_lines": total_primary_lines,
                "fallback_lines": total_fallback,
                "fallback_ratio": round(fb_ratio, 4),
                "mean_confidence": round(mean_ocr_conf, 4),
                "mean_quality_score": round(mean_qual_score, 4)
            }

        # Export all artifacts to disk
        out_path = Path(output_dir)
        doc_dir = ExportDocumentUseCase.execute(extracted_doc, out_path, benchmark_data)

        # Accuracy evaluation against ground truth if requested
        if accuracy_report:
            gt_path = Path(ground_truth_dir)
            if gt_path.exists():
                try:
                    import json
                    from benchmark.evaluate import evaluate_document
                    acc_report = evaluate_document(doc_dir, gt_path)
                    extracted_doc.accuracy = acc_report
                    acc_dict = acc_report.model_dump(mode="json")
                    with open(doc_dir / "accuracy.json", "w", encoding="utf-8") as f:
                        json.dump(acc_dict, f, ensure_ascii=False, indent=2)

                    # Update quality metrics with real CER and WER
                    extracted_doc.quality.metrics.cer = acc_report.mean_cer
                    extracted_doc.quality.metrics.wer = acc_report.mean_wer
                    with open(doc_dir / "quality.json", "w", encoding="utf-8") as f:
                        json.dump(extracted_doc.quality.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[WARN] Accuracy evaluation failed: {e}")

        # Keyword counting (runs after result.md has been written)
        try:
            CountKeywordsUseCase.execute(
                doc_dir=doc_dir,
                document_id=doc_id,
                keywords_file=keywords_file,
            )
        except Exception as e:
            print(f"[WARN] Keyword counting failed: {e}")

        return extracted_doc
