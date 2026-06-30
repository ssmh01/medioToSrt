"""End-to-end alignment, split, validation, and export pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .audio import preprocess_audio
from .engines.base import AlignmentEngine
from .engines.stable_ts import StableTsEngine
from .formats import export_srt, export_vtt
from .models import AlignmentToken, JobResult, SubtitleCue
from .profiles import resolve_profile
from .quality import build_quality_report
from .splitter import split_subtitles
from .text import clean_script_text, map_tokens_to_display


def run_alignment_job(
    audio_path: str | Path,
    script_text: str,
    language: str = "zh",
    subtitle_profile: str = "youtube_long",
    output_dir: str | Path | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    max_chars_per_line: int | None = None,
    generate_vtt: bool = True,
    preserve_punctuation: bool = True,
    engine: AlignmentEngine | None = None,
) -> JobResult:
    logs: list[str] = []
    engine = engine or StableTsEngine()
    profile = resolve_profile(
        subtitle_profile,
        language,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars_per_line=max_chars_per_line,
    )
    cleaned = clean_script_text(script_text, preserve_punctuation=preserve_punctuation)
    logs.append(f"文案字符数: {len(cleaned.display_text)}")

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="autosrt_aligner_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if getattr(engine, "requires_audio_preprocessing", True):
        logs.append("开始音频预处理: 16kHz mono wav")
        audio_info = preprocess_audio(audio_path, work_dir)
        align_audio_path = audio_info.wav_path
        audio_duration = audio_info.duration
    else:
        align_audio_path = Path(audio_path)
        audio_duration = None

    alignment = engine.align(align_audio_path, cleaned, language, logs)
    if alignment.audio_duration is not None:
        audio_duration = alignment.audio_duration
    logs.append(f"对齐 token 数: {len(alignment.tokens)}")

    mapped_tokens = _ensure_mapped_tokens(alignment.tokens, cleaned)
    cues = split_subtitles(cleaned.display_text, mapped_tokens, language, profile)
    quality_report = build_quality_report(cues, cleaned.display_text, audio_duration, profile, language)

    srt_path = out_dir / "output.srt"
    srt_path.write_text(export_srt(cues), encoding="utf-8")
    vtt_path = None
    if generate_vtt:
        vtt_path = out_dir / "output.vtt"
        vtt_path.write_text(export_vtt(cues), encoding="utf-8")

    alignment_payload = _build_alignment_payload(
        cleaned_display=cleaned.display_text,
        cleaned_align=cleaned.align_text,
        language=language,
        profile_key=subtitle_profile,
        alignment_raw=alignment.raw,
        tokens=mapped_tokens,
        cues=cues,
        audio_duration=audio_duration,
    )
    alignment_json_path = out_dir / "alignment.json"
    alignment_json_path.write_text(
        json.dumps(alignment_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quality_report_path = out_dir / "quality_report.json"
    quality_report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logs.append("导出完成")

    return JobResult(
        output_dir=out_dir,
        srt_path=srt_path,
        vtt_path=vtt_path,
        alignment_json_path=alignment_json_path,
        quality_report_path=quality_report_path,
        cues=cues,
        quality_report=quality_report,
        alignment_payload=alignment_payload,
        logs=logs,
    )


def _ensure_mapped_tokens(tokens: list[AlignmentToken], cleaned: Any) -> list[AlignmentToken]:
    if all(token.start_char is not None and token.end_char is not None for token in tokens):
        return tokens
    return map_tokens_to_display(tokens, cleaned)


def _build_alignment_payload(
    cleaned_display: str,
    cleaned_align: str,
    language: str,
    profile_key: str,
    alignment_raw: dict[str, Any],
    tokens: list[AlignmentToken],
    cues: list[SubtitleCue],
    audio_duration: float | None,
) -> dict[str, Any]:
    return {
        "language": language,
        "subtitle_profile": profile_key,
        "audio_duration": audio_duration,
        "display_text_length": len(cleaned_display),
        "align_text_length": len(cleaned_align),
        "engine_result": alignment_raw,
        "tokens": [
            {
                "text": token.text,
                "start": token.start,
                "end": token.end,
                "start_char": token.start_char,
                "end_char": token.end_char,
                "confidence": token.confidence,
            }
            for token in tokens
        ],
        "cues": [
            {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "start_char": cue.start_char,
                "end_char": cue.end_char,
                "warnings": cue.warnings,
            }
            for cue in cues
        ],
    }
