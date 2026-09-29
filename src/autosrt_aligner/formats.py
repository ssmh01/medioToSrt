"""SRT and VTT formatting helpers."""

from __future__ import annotations

import re
import unicodedata

from .models import SubtitleCue
from .profiles import language_group


_NEUTRAL_TERMINAL_PUNCTUATION = frozenset("。．｡.,，、､")
_ELLIPSIS_CHARACTERS = frozenset({"…", "⋯", "︙"})
_CLOSING_DELIMITER_CATEGORIES = frozenset({"Pe", "Pf"})
_ASCII_CLOSING_QUOTES = frozenset({"'", '"'})
_ENGLISH_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "etc.",
        "jr.",
        "mr.",
        "mrs.",
        "ms.",
        "ph.d.",
        "prof.",
        "sr.",
        "st.",
        "vs.",
    }
)
_ENGLISH_INITIAL = re.compile(r"[A-Z]\.")
_ENGLISH_INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}")
_REPEATED_EMPHATIC_PUNCTUATION = re.compile(r"[!?]{2,}")
_ENGLISH_TOKEN_OPENING_DELIMITERS = "'\"([{\u2018\u201c"


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


def _normalize_emphatic_punctuation(match: re.Match[str]) -> str:
    punctuation = match.group(0)
    if "?" in punctuation and "!" in punctuation:
        return "?!"
    return punctuation[0]


def _is_english_abbreviation_suffix(text: str) -> bool:
    token = text.rsplit(maxsplit=1)[-1].lstrip(_ENGLISH_TOKEN_OPENING_DELIMITERS)
    return (
        token.casefold() in _ENGLISH_ABBREVIATIONS
        or _ENGLISH_INITIAL.fullmatch(token) is not None
        or _ENGLISH_INITIALISM.fullmatch(token) is not None
    )


def _clean_english_punctuation_per_line(text: str) -> str:
    """Apply the concise English display style without changing source text."""

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        cleaned = _REPEATED_EMPHATIC_PUNCTUATION.sub(
            _normalize_emphatic_punctuation,
            line.rstrip(),
        )

        closing_delimiters = ""
        while cleaned and _is_closing_delimiter(cleaned[-1]):
            closing_delimiters = cleaned[-1] + closing_delimiters
            cleaned = cleaned[:-1].rstrip()

        if (
            cleaned.endswith(".")
            and not _has_ellipsis_suffix(cleaned)
            and not _is_english_abbreviation_suffix(cleaned)
        ):
            cleaned = cleaned[:-1].rstrip()

        cleaned_lines.append(cleaned + closing_delimiters)
    return "\n".join(cleaned_lines)


def _strip_trailing_punctuation_per_line(text: str, language: str) -> str:
    if language_group(language) == "en":
        return _clean_english_punctuation_per_line(text)

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


def export_vtt(
    cues: list[SubtitleCue],
    *,
    language: str | None = None,
    clean_punctuation: bool = True,
) -> str:
    blocks = ["WEBVTT", ""]
    for cue in cues:
        text = (
            _clean_english_punctuation_per_line(cue.text)
            if clean_punctuation and language is not None and language_group(language) == "en"
            else cue.text
        )
        blocks.append(f"{vtt_timestamp(cue.start)} --> {vtt_timestamp(cue.end)}\n{text}")
        blocks.append("")
    return "\n".join(blocks)
