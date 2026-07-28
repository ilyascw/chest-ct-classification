"""Command-line interface for batch chest CT archive processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core_pipeline import CTPathologyPipeline
from .data_models import PipelineConfig


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CT pathology screening pipeline",
    )
    parser.add_argument("zip_paths", nargs="+", help="Input ZIP archives")
    parser.add_argument(
        "--ctclip-checkpoint",
        required=True,
        help="Path to the CT-CLIP checkpoint",
    )
    parser.add_argument(
        "--catboost-model",
        required=True,
        help="Path to the CatBoost model",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ct_pathology_results.xlsx",
        help="Output XLSX report",
    )
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=1,
        help="Number of study-processing workers",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    required_files = [
        *[Path(path) for path in args.zip_paths],
        Path(args.ctclip_checkpoint),
        Path(args.catboost_model),
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        parser.error(f"Files not found: {', '.join(missing_files)}")

    config = PipelineConfig(
        ct_clip_checkpoint=args.ctclip_checkpoint,
        catboost_model=args.catboost_model,
        device=args.device,
        max_workers=args.max_workers,
        log_level=args.log_level,
    )

    pipeline = CTPathologyPipeline(config)
    report = pipeline.process_zip_archives(args.zip_paths, args.output)
    status_counts = report["processing_status"].value_counts().to_dict()

    print(f"Report saved to: {Path(args.output).resolve()}")
    print(f"Studies processed: {len(report)}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
