"""Text cleanup, alignment mapping, and subtitle continuity checks."""

from __future__ import annotations

import re
import unicodedata

from .errors import InputError
from .models import AlignmentToken, CleanedText, SubtitleCue

ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
MARKDOWN_NOISE = set("#>*_`~")


def clean_script_text(script_text: str, preserve_punctuation: bool = True) -> CleanedText:
    if script_text is None:
        raise InputError("文案为空")

    display = script_text.replace("\r\n", "\n").replace("\r", "\n")
    display = "".join(ch for ch in display if ch not in ZERO_WIDTH)
    display = unicodedata.normalize("NFC", display).strip()
    if not display:
        raise InputError("文案为空")

    align_chars: list[str] = []
    align_to_display: list[int] = []
    last_was_space = False
    for idx, ch in enumerate(display):
        if ch in MARKDOWN_NOISE:
            continue
        if not preserve_punctuation and unicodedata.category(ch).startswith("P"):
            continue
        if ch.isspace() or ch == "\u3000":
            if align_chars and not last_was_space:
                align_chars.append(" ")
                align_to_display.append(idx)
                last_was_space = True
            continue
        align_chars.append(ch)
        align_to_display.append(idx)
        last_was_space = False

    while align_chars and align_chars[0].isspace():
        align_chars.pop(0)
        align_to_display.pop(0)
    while align_chars and align_chars[-1].isspace():
        align_chars.pop()
        align_to_display.pop()
    align_text = "".join(align_chars)
    if not align_text:
        raise InputError("清洗后的文案为空，无法对齐")
    return CleanedText(display_text=display, align_text=align_text, align_to_display=align_to_display)


def normalize_for_alignment(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch not in ZERO_WIDTH and ch not in MARKDOWN_NOISE)
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_for_compare(value: str) -> str:
    value = "".join(ch for ch in value if ch not in ZERO_WIDTH)
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", "", value)


def render_display_segment(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def map_tokens_to_display(tokens: list[AlignmentToken], cleaned: CleanedText) -> list[AlignmentToken]:
    mapped: list[AlignmentToken] = []
    cursor = 0
    display_cursor = 0

    for token in tokens:
        token_text = normalize_for_alignment(token.text)
        if not token_text:
            continue

        align_index = cleaned.align_text.find(token_text, cursor)
        if align_index >= 0:
            align_end = align_index + len(token_text)
            start_char = cleaned.align_to_display[min(align_index, len(cleaned.align_to_display) - 1)]
            end_char = cleaned.align_to_display[min(align_end - 1, len(cleaned.align_to_display) - 1)] + 1
            cursor = align_end
        else:
            raw_index = cleaned.display_text.find(token.text.strip(), display_cursor)
            if raw_index < 0:
                raw_index = display_cursor
            start_char = raw_index
            end_char = min(len(cleaned.display_text), raw_index + len(token.text.strip()))
            cursor = min(len(cleaned.align_text), cursor + len(token_text))

        display_cursor = max(display_cursor, end_char)
        mapped.append(
            AlignmentToken(
                text=token.text,
                start=max(0.0, token.start),
                end=max(token.start, token.end),
                start_char=start_char,
                end_char=end_char,
                confidence=token.confidence,
            )
        )

    return mapped


def validate_subtitle_continuity(cues: list[SubtitleCue], display_text: str) -> bool:
    actual = normalize_for_compare("".join(cue.text for cue in cues))
    expected = normalize_for_compare(display_text)
    return actual == expected


def unaligned_text_ratio(cues: list[SubtitleCue], display_text: str) -> float:
    actual = normalize_for_compare("".join(cue.text for cue in cues))
    expected = normalize_for_compare(display_text)
    if not expected:
        return 1.0
    if actual == expected:
        return 0.0
    return min(1.0, abs(len(expected) - len(actual)) / len(expected))
