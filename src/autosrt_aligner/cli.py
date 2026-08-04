"""Command-line entrypoint for advanced/local batch use."""

from __future__ import annotations

import argparse
from pathlib import Path

from .errors import AutosrtError, ExportValidationError
from .engines.factory import ENGINE_CHOICES
from .pipeline_v2 import run_alignment_job_v2 as run_alignment_job
from .profiles import normalize_language


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SRT/VTT from narration audio and source script.")
    parser.add_argument("--audio", required=True, help="Input audio path, e.g. input.mp3")
    parser.add_argument("--text", required=True, help="Input script txt path")
    parser.add_argument("--language", required=True, choices=["zh", "zh-TW", "ja", "en", "ko"])
    parser.add_argument(
        "--profile",
        default="youtube_long",
        choices=["youtube_long", "standard", "short", "slow_elder"],
        help="Subtitle style profile",
    )
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--vtt", action="store_true", help="Also export output.vtt")
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--max-chars-per-line", type=int, default=None)
    parser.add_argument(
        "--engine",
        default="qwen-mlx",
        choices=ENGINE_CHOICES,
        help="对齐引擎；本工具固定使用 Qwen MLX",
    )
    args = parser.parse_args(argv)

    try:
        script_text = Path(args.text).read_text(encoding="utf-8-sig")
        result = run_alignment_job(
            audio_path=args.audio,
            script_text=script_text,
            language=normalize_language(args.language),
            subtitle_profile=args.profile,
            output_dir=args.out_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            max_chars_per_line=args.max_chars_per_line,
            generate_vtt=args.vtt,
            alignment_engine=args.engine,
        )
    except ExportValidationError as exc:
        print(f"ERROR: {exc}")
        print(f"quality_report: {Path(args.out_dir) / 'quality_report.json'}")
        print(f"alignment_json: {Path(args.out_dir) / 'alignment.json'}")
        return 2
    except AutosrtError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("\n".join(result.logs))
    print(f"SRT: {result.srt_path}")
    if result.vtt_path:
        print(f"VTT: {result.vtt_path}")
    print(f"quality_report: {result.quality_report_path}")
    print(f"alignment_json: {result.alignment_json_path}")
    print(f"quality_score: {result.quality_report['quality_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
