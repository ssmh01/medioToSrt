"""Deterministic source/audio chunk planning for the V2 pipeline."""

from __future__ import annotations

from .models import AudioChunk, SourceDocument

STRONG_PUNCT = set("。！？!?．.!？")
MID_PUNCT = set("，、,;；:：")
OPEN_QUOTES = set("「『“‘（([【\"")
CLOSE_QUOTES = set("」』”’）)]】\"")


def plan_audio_chunks(
    source: SourceDocument,
    audio_duration: float,
    *,
    target_seconds: float = 45.0,
    overlap_seconds: float = 1.5,
    max_chunks: int = 128,
) -> list[AudioChunk]:
    """Plan overlapping chunks without changing source text.

    The first implementation deliberately uses a conservative proportional
    estimate only to choose search windows. Actual timestamps still come from
    the engine and are reconciled later.
    """

    if audio_duration <= 0:
        raise ValueError("音频时长必须大于 0")
    visible_positions = [
        index for index, char in enumerate(source.display_text) if not char.isspace()
    ]
    if not visible_positions:
        raise ValueError("原文没有可对齐字符")
    if len(visible_positions) <= 220 or audio_duration <= target_seconds * 1.25:
        return [
            AudioChunk(
                chunk_id="chunk-001",
                source_start=0,
                source_end=len(source.display_text),
                audio_start=0.0,
                audio_end=audio_duration,
                core_source_start=0,
                core_source_end=len(source.display_text),
                core_audio_start=0.0,
                core_audio_end=audio_duration,
            )
        ]

    units_per_second = len(visible_positions) / audio_duration
    target_units = max(64, int(round(target_seconds * units_per_second)))
    overlap_units = max(8, int(round(overlap_seconds * units_per_second)))
    chunks: list[AudioChunk] = []
    core_start_unit = 0
    chunk_number = 1

    while core_start_unit < len(visible_positions) and chunk_number <= max_chunks:
        target_unit = min(len(visible_positions), core_start_unit + target_units)
        if target_unit >= len(visible_positions):
            core_end_unit = len(visible_positions)
        else:
            core_end_unit = _choose_boundary(
                source.display_text,
                visible_positions,
                core_start_unit,
                target_unit,
                source.language,
            )
        if core_end_unit <= core_start_unit:
            core_end_unit = min(len(visible_positions), core_start_unit + target_units)

        source_start_unit = max(0, core_start_unit - overlap_units)
        source_end_unit = min(len(visible_positions), core_end_unit + overlap_units)
        source_start = visible_positions[source_start_unit]
        source_end = visible_positions[source_end_unit - 1] + 1
        core_source_start = (
            0
            if core_start_unit == 0
            else visible_positions[core_start_unit]
        )
        core_source_end = (
            len(source.display_text)
            if core_end_unit >= len(visible_positions)
            else visible_positions[core_end_unit]
        )
        # ``units_per_second`` is based on visible-unit indexes, not display
        # character offsets (which include spaces and punctuation).
        core_audio_start = core_start_unit / units_per_second
        core_audio_end = core_end_unit / units_per_second
        audio_start = max(0.0, core_audio_start - overlap_seconds)
        audio_end = min(audio_duration, max(core_audio_end + overlap_seconds, audio_start + 0.5))

        chunks.append(
            AudioChunk(
                chunk_id=f"chunk-{chunk_number:03d}",
                source_start=source_start,
                source_end=source_end,
                audio_start=audio_start,
                audio_end=audio_end,
                overlap_before=max(0.0, core_audio_start - audio_start),
                overlap_after=max(0.0, audio_end - core_audio_end),
                core_source_start=core_source_start,
                core_source_end=core_source_end,
                core_audio_start=core_audio_start,
                core_audio_end=core_audio_end,
            )
        )
        core_start_unit = core_end_unit
        chunk_number += 1

    if core_start_unit < len(visible_positions):
        raise ValueError(f"原文过长，超过最大分块数 {max_chunks}")
    return chunks


def _choose_boundary(
    text: str,
    visible_positions: list[int],
    start_unit: int,
    target_unit: int,
    language: str,
) -> int:
    radius = max(18, min(80, int((target_unit - start_unit) * 0.28)))
    low = max(start_unit + 16, target_unit - radius)
    high = min(len(visible_positions), target_unit + radius)
    best_unit = target_unit
    best_score = float("-inf")
    for unit in range(low, high + 1):
        char_end = visible_positions[unit - 1] + 1
        if not _safe_chunk_boundary(text, char_end, language):
            continue
        prev = _previous_visible(text, char_end)
        distance_penalty = abs(unit - target_unit) * 0.8
        score = -distance_penalty
        if prev in STRONG_PUNCT:
            score += 80
        elif prev in MID_PUNCT:
            score += 35
        elif prev.isspace():
            score += 18
        if _inside_quote(text, char_end):
            score -= 12
        if score > best_score:
            best_score = score
            best_unit = unit
    return best_unit


def _safe_chunk_boundary(text: str, char_end: int, language: str) -> bool:
    if char_end <= 0 or char_end >= len(text):
        return True
    previous = _previous_visible(text, char_end)
    following = _next_visible(text, char_end)
    if not previous or not following:
        return True
    if (
        language == "en"
        and previous.isalnum()
        and following.isalnum()
        and not _has_boundary_space(text, char_end)
    ):
        return False
    if language in {"ja", "ko"} and previous in OPEN_QUOTES:
        return False
    return True


def _previous_visible(text: str, end: int) -> str:
    for char in reversed(text[:end]):
        if not char.isspace():
            return char
    return ""


def _next_visible(text: str, start: int) -> str:
    for char in text[start:]:
        if not char.isspace():
            return char
    return ""


def _has_boundary_space(text: str, char_end: int) -> bool:
    previous_index = None
    for index in range(min(char_end, len(text)) - 1, -1, -1):
        if not text[index].isspace():
            previous_index = index
            break
    if previous_index is None:
        return False
    return any(char.isspace() for char in text[previous_index + 1 : char_end])


def _inside_quote(text: str, char_end: int) -> bool:
    depth = 0
    for char in text[:char_end]:
        if char in OPEN_QUOTES:
            depth += 1
        elif char in CLOSE_QUOTES and depth:
            depth -= 1
    return depth > 0
