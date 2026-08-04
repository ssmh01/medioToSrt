"""SRT and VTT formatting helpers."""

from __future__ import annotations

import unicodedata

from .models import SubtitleCue
from .profiles import language_group


_NEUTRAL_TERMINAL_PUNCTUATION = frozenset("。．｡.,，、､")
_ELLIPSIS_CHARACTERS = frozenset({"…", "⋯", "︙"})
_CLOSING_DELIMITER_CATEGORIES = frozenset({"Pe", "Pf"})
_ASCII_CLOSING_QUOTES = frozenset({"'", '"'})


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


def _is_closing_delimiter(char: str) -> bool:
    return (
        unicodedata.category(char) in _CLOSING_DELIMITER_CATEGORIES
        or char in _ASCII_CLOSING_QUOTES
    )


def _has_ellipsis_suffix(text: str) -> bool:
    return text.endswith("...") or any(text.endswith(char) for char in _ELLIPSIS_CHARACTERS)


def _strip_trailing_punctuation_per_line(text: str, language: str) -> str:
    if language_group(language) == "en":
        return text

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        cleaned = line.rstrip()

        closing_delimiters = ""
        while cleaned and _is_closing_delimiter(cleaned[-1]):
            closing_delimiters = cleaned[-1] + closing_delimiters
            cleaned = cleaned[:-1].rstrip()

        if not _has_ellipsis_suffix(cleaned):
            while cleaned and cleaned[-1] in _NEUTRAL_TERMINAL_PUNCTUATION:
                cleaned = cleaned[:-1].rstrip()

        cleaned += closing_delimiters
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def export_srt(
    cues: list[SubtitleCue],
    strip_trailing_punctuation: bool = True,
    *,
    language: str = "zh",
) -> str:
    blocks: list[str] = []
    for cue in cues:
        text = (
            _strip_trailing_punctuation_per_line(cue.text, language)
            if strip_trailing_punctuation
            else cue.text
        )
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
