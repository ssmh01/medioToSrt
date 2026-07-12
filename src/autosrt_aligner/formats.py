"""SRT and VTT formatting helpers."""

from __future__ import annotations

import unicodedata

from .models import SubtitleCue


def srt_timestamp(seconds: float) -> str:
    whole_ms = max(0, int(round(seconds * 1000)))
    ms = whole_ms % 1000
    total_seconds = whole_ms // 1000
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def _strip_trailing_punctuation_per_line(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        cleaned = line.rstrip()
        while cleaned and unicodedata.category(cleaned[-1]).startswith("P"):
            cleaned = cleaned[:-1].rstrip()
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def export_srt(cues: list[SubtitleCue]) -> str:
    blocks: list[str] = []
    for cue in cues:
        text = _strip_trailing_punctuation_per_line(cue.text)
        blocks.append(
            f"{cue.index}\n{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


def export_vtt(cues: list[SubtitleCue]) -> str:
    blocks = ["WEBVTT", ""]
    for cue in cues:
        blocks.append(f"{vtt_timestamp(cue.start)} --> {vtt_timestamp(cue.end)}\n{cue.text}")
        blocks.append("")
    return "\n".join(blocks)
