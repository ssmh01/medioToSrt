"""Evidence-constrained semantic cue segmentation for the V2 pipeline."""

from __future__ import annotations

import math
import re

from .errors import AlignmentError
from .models import AlignmentToken, SourceDocument, SubtitleCue, SubtitleProfile
from .profiles import language_group
from .text import render_display_segment, validate_subtitle_continuity

STRONG_PUNCT = set("。！？!?…．.")
MID_PUNCT = set("，、,;；:：")
CLOSE_QUOTES = set("」』”’）)]】")
OPEN_QUOTES = set("「『“‘（([【")
ZH_BAD_EDGE = set("的地得把被和与跟在是就也都要会能很又还再")
JA_BAD_EDGE = set("はがをにでもとのへやもねよからまで")
JA_SMALL_KANA = set("っゃゅょぁぃぅぇぉッャュョァィゥェォー")
EN_BAD_EDGE = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or"}


def segment_cues(
    source: SourceDocument,
    tokens: list[AlignmentToken],
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    """Select cue boundaries from existing token timestamps only."""

    ordered = sorted(
        (token for token in tokens if token.start_char is not None and token.end_char is not None),
        key=lambda token: (token.start_char or 0, token.start, token.end),
    )
    if not ordered:
        raise AlignmentError("没有可用于语义切分的对齐 token")

    path = _best_path(source, ordered, profile)
    cues: list[SubtitleCue] = []
    char_start = 0
    token_start = 0
    for token_end in path:
        next_start_char = (
            ordered[token_end + 1].start_char
            if token_end + 1 < len(ordered)
            else len(source.display_text)
        )
        char_end = max(char_start, next_start_char or len(source.display_text))
        text = render_display_segment(source.display_text[char_start:char_end])
        if text:
            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start=max(0.0, ordered[token_start].start),
                    end=max(ordered[token_start].start + 0.1, ordered[token_end].end),
                    text=text,
                    start_char=char_start,
                    end_char=char_end,
                )
            )
        char_start = char_end
        token_start = token_end + 1

    if not cues or not validate_subtitle_continuity(cues, source.display_text):
        raise AlignmentError("语义切分结果无法连续覆盖原文")
    return cues


def _best_path(
    source: SourceDocument,
    tokens: list[AlignmentToken],
    profile: SubtitleProfile,
) -> list[int]:
    count = len(tokens)
    costs = [math.inf] * (count + 1)
    next_end: list[int | None] = [None] * count
    costs[count] = 0.0

    for start in range(count - 1, -1, -1):
        start_char = tokens[start].start_char or 0
        for end in range(start, count):
            duration = tokens[end].end - tokens[start].start
            if end < count - 1 and duration < profile.min_duration:
                continue
            if duration > profile.max_duration + 0.001:
                break
            char_end = tokens[end + 1].start_char if end + 1 < count else len(source.display_text)
            if char_end is None:
                char_end = len(source.display_text)
            if end < count - 1 and not _is_safe_boundary(source.display_text, char_end, source.language):
                continue
            if not math.isfinite(costs[end + 1]):
                continue
            text = source.display_text[start_char:char_end]
            candidate_cost = _boundary_cost(
                source.display_text,
                char_end,
                text,
                duration,
                profile,
                source.language,
                tokens[end].end,
                tokens[end + 1].start - tokens[end].end if end + 1 < count else 0.0,
            )
            total = candidate_cost + costs[end + 1]
            if total < costs[start]:
                costs[start] = total
                next_end[start] = end

    if next_end[0] is None:
        # A very short or unusual sample may have no ideal-duration boundary;
        # keep the complete evidence span rather than modifying timestamps.
        return [count - 1]

    path: list[int] = []
    start = 0
    while start < count:
        end = next_end[start]
        if end is None:
            raise AlignmentError("语义切分路径不完整")
        path.append(end)
        start = end + 1
    return path


def _boundary_cost(
    display_text: str,
    char_end: int,
    text: str,
    duration: float,
    profile: SubtitleProfile,
    language: str,
    token_end: float,
    next_gap: float,
) -> float:
    compact = "".join(text.split())
    chars = len(compact)
    target_chars = max(4.0, profile.max_chars_total * 0.72)
    cost = abs(chars - target_chars) * 0.45
    ideal_mid = (profile.ideal_min_duration + profile.ideal_max_duration) / 2
    cost += abs(duration - ideal_mid) * 2.0
    previous = _previous_visible(display_text, char_end)
    following = _next_visible(display_text, char_end)
    if previous in STRONG_PUNCT:
        cost -= 48
    elif previous in MID_PUNCT:
        cost -= 22
    if next_gap >= 0.45:
        cost -= 20
    elif next_gap >= 0.2:
        cost -= 8
    if _has_boundary_space(display_text, char_end):
        cost -= 10
    if previous in OPEN_QUOTES or following in CLOSE_QUOTES:
        cost += 42
    if _has_bad_edge(compact, language):
        cost += 28
    if chars > profile.max_chars_total:
        cost += 80 + (chars - profile.max_chars_total) * 4
    if duration < profile.min_duration and char_end < len(display_text):
        cost += 90
    # Keep the value visible in debugging without making it a timing edit.
    _ = token_end
    return cost


def _is_safe_boundary(text: str, char_end: int, language: str) -> bool:
    group = language_group(language)
    previous = _previous_visible(text, char_end)
    following = _next_visible(text, char_end)
    if not previous or not following:
        return True
    if previous in OPEN_QUOTES or following in CLOSE_QUOTES:
        return False
    if language == "en":
        if previous.isalnum() and following.isalnum() and not _has_boundary_space(text, char_end):
            return False
        words = re.findall(r"[A-Za-z']+", text[:char_end])
        if words and words[-1].lower() in EN_BAD_EDGE and following.isalpha():
            return False
    elif group == "cjk":
        if previous in ZH_BAD_EDGE and following not in STRONG_PUNCT | MID_PUNCT:
            return False
    elif language == "ja":
        if previous in JA_SMALL_KANA or following in JA_SMALL_KANA:
            return False
        if previous in JA_BAD_EDGE and following not in STRONG_PUNCT | MID_PUNCT:
            return False
    elif language == "ko":
        if (
            _is_hangul(previous)
            and _is_hangul(following)
            and not _has_boundary_space(text, char_end)
        ):
            return False
    return True


def _has_bad_edge(text: str, language: str) -> bool:
    group = language_group(language)
    words = re.findall(r"[A-Za-z']+", text)
    if language == "en":
        return bool(words and words[-1].lower() in EN_BAD_EDGE)
    if not text:
        return False
    if group == "cjk":
        return text[-1] in ZH_BAD_EDGE
    if language == "ja":
        return text[-1] in JA_BAD_EDGE
    return False


def _is_hangul(char: str) -> bool:
    return bool(char and ("\uac00" <= char <= "\ud7a3"))


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
    """Return whether the selected boundary is already separated by whitespace."""

    previous_index = None
    for index in range(min(char_end, len(text)) - 1, -1, -1):
        if not text[index].isspace():
            previous_index = index
            break
    if previous_index is None:
        return False
    return any(char.isspace() for char in text[previous_index + 1 : char_end])
