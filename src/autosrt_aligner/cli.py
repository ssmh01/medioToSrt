"""Command-line entrypoint for advanced/local batch use."""

from __future__ import annotations

import argparse
from pathlib import Path

from .errors import AutosrtError
from .pipeline import run_alignment_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SRT/VTT from narration audio and source script.")
    parser.add_argument("--audio", required=True, help="Input audio path, e.g. input.mp3")
    parser.add_argument("--text", required=True, help="Input script txt path")
    parser.add_argument("--language", required=True, choices=["zh", "zh-TW", "ja", "en"])
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
    args = parser.parse_args(argv)

    try:
        script_text = Path(args.text).read_text(encoding="utf-8-sig")
        result = run_alignment_job(
            audio_path=args.audio,
            script_text=script_text,
            language=args.language,
            subtitle_profile=args.profile,
            output_dir=args.out_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            max_chars_per_line=args.max_chars_per_line,
            generate_vtt=args.vtt,
        )
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
