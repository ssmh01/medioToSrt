"""Text cleanup, alignment mapping, and subtitle continuity checks."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

from .errors import AlignmentError, InputError
from .models import AlignmentToken, CleanedText, SourceDocument, SubtitleCue

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


def build_source_document(
    script_text: str,
    language: str,
    preserve_punctuation: bool = True,
) -> SourceDocument:
    """Build the immutable source contract used by the refactored pipeline."""

    cleaned = clean_script_text(script_text, preserve_punctuation=preserve_punctuation)
    return SourceDocument(
        raw_text=script_text,
        display_text=cleaned.display_text,
        align_text=cleaned.align_text,
        align_to_display=tuple(cleaned.align_to_display),
        language=language,
        source_hash=sha256(script_text.encode("utf-8")).hexdigest(),
    )


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


def map_tokens_to_source_strict(
    tokens: list[AlignmentToken],
    source: SourceDocument,
    source_offset: int = 0,
    chunk_id: str | None = None,
) -> list[AlignmentToken]:
    """Map engine tokens without silently inventing source positions.

    A forced aligner may omit whitespace and unspoken punctuation, so those
    characters are allowed between successive token matches and after the
    final token. Any non-ignorable source text that a returned token cannot
    match is an alignment failure.
    ``source`` may represent a local chunk; ``source_offset`` then restores
    its display-text offset in the full document.
    """

    mapped: list[AlignmentToken] = []
    cursor = 0
    for token in tokens:
        token_text = normalize_for_alignment(token.text)
        if not token_text:
            continue

        align_index = source.align_text.find(token_text, cursor)
        if align_index < 0:
            raise AlignmentError(
                f"对齐 token 无法映射回原文: {token.text!r}, cursor={cursor}"
            )
        skipped = source.align_text[cursor:align_index]
        if any(not _is_ignorable_alignment_gap(char) for char in skipped):
            raise AlignmentError(
                f"对齐 token 跳过了原文内容: {skipped[:24]!r}"
            )

        align_end = align_index + len(token_text)
        if align_end - 1 >= len(source.align_to_display):
            raise AlignmentError("对齐 token 超出原文映射范围")
        start_char = source.align_to_display[align_index] + source_offset
        end_char = source.align_to_display[align_end - 1] + source_offset + 1
        cursor = align_end
        mapped.append(
            AlignmentToken(
                text=token.text,
                start=max(0.0, token.start),
                end=max(token.start, token.end),
                start_char=start_char,
                end_char=end_char,
                confidence=token.confidence,
                chunk_id=chunk_id or token.chunk_id,
                unit_type=token.unit_type,
            )
        )

    if not mapped:
        raise AlignmentError("对齐结果没有可映射 token")
    trailing = source.align_text[cursor:]
    if any(not _is_ignorable_alignment_gap(char) for char in trailing):
        raise AlignmentError(
            f"对齐 token 未覆盖原文尾部: {trailing[:24]!r}"
        )
    return mapped


def _is_ignorable_alignment_gap(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P") or char in MARKDOWN_NOISE


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
