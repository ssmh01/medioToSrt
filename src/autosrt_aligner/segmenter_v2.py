"""Evidence-constrained semantic cue segmentation for the V2 pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import replace

from .errors import AlignmentError
from .models import AlignmentToken, SourceDocument, SubtitleCue, SubtitleProfile
from .profiles import language_group
from .text import (
    is_nonspoken_alignment_char,
    render_display_segment,
    validate_subtitle_continuity,
)

STRONG_PUNCT = set("。！？!?…．.")
MID_PUNCT = set("，、,;；:：")
CLOSE_QUOTES = set("」』”’）)]】")
OPEN_QUOTES = set("「『“‘（([【")
ZH_BAD_EDGE = set("的地得把被和与跟在是就也都要会能很又还再")
JA_BAD_EDGE = set("はがをにでもとのへやもねよからまで")
JA_SMALL_KANA = set("っゃゅょぁぃぅぇぉッャュョァィゥェォー")
EN_BAD_EDGE = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or"}
MAX_MERGED_INTERNAL_GAP_SECONDS = 0.8
TERMINAL_JA_CPS_TOLERANCE = 1.0
TERMINAL_CUE_MIN_EVIDENCE_SECONDS = 0.5


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

    cues = _repair_korean_timing(cues, source.language, profile)
    cues = _merge_korean_fast_cues(source, cues, profile)
    if not cues or not validate_subtitle_continuity(cues, source.display_text):
        raise AlignmentError("语义切分结果无法连续覆盖原文")
    return cues


def _repair_korean_timing(
    cues: list[SubtitleCue],
    language: str,
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    """Use an existing inter-cue gap to soften a Korean fast cue.

    The next cue's real start is the upper bound. This only extends a cue into
    already-unassigned silence; it never moves a cue boundary past the next
    token or invents an audio timestamp.
    """

    if language != "ko" or len(cues) < 2 or profile.max_chars_per_second <= 0:
        return cues

    repaired = list(cues)
    for index, cue in enumerate(cues[:-1]):
        spoken_chars = _reading_char_count(cue.text)
        if spoken_chars <= 0 or cue.duration <= 0:
            continue
        target_duration = spoken_chars / profile.max_chars_per_second
        if cue.duration >= target_duration:
            continue

        next_start = cues[index + 1].start
        if next_start <= cue.end:
            continue
        repaired_end = min(next_start, cue.start + target_duration)
        if repaired_end > cue.end:
            repaired[index] = replace(cue, end=repaired_end)
    return repaired


def _merge_korean_fast_cues(
    source: SourceDocument,
    cues: list[SubtitleCue],
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    """Merge an isolated fast Korean cue when a compliant adjacent span exists.

    This keeps all token timestamps intact. A merge is allowed only when the
    combined cue still satisfies the configured duration, character, and
    reading-speed limits. Weak boundaries are removed before punctuation-led
    sentence boundaries so readability is not traded for a passing report.
    """

    if source.language != "ko" or len(cues) < 2 or profile.max_chars_per_second <= 0:
        return cues

    repaired = list(cues)
    index = 0
    while index < len(repaired):
        cue = repaired[index]
        reading_speed = _reading_char_count(cue.text) / max(cue.duration, 0.1)
        if reading_speed <= profile.max_chars_per_second + 0.001:
            index += 1
            continue

        candidates: list[tuple[tuple[float, ...], int, SubtitleCue]] = []
        for merge_start in (index - 1, index):
            if merge_start < 0 or merge_start + 1 >= len(repaired):
                continue
            left, right = repaired[merge_start : merge_start + 2]
            if right.start - left.end > MAX_MERGED_INTERNAL_GAP_SECONDS + 0.001:
                continue
            text = render_display_segment(
                source.display_text[left.start_char : right.end_char]
            )
            merged = SubtitleCue(
                index=left.index,
                start=left.start,
                end=right.end,
                text=text,
                start_char=left.start_char,
                end_char=right.end_char,
            )
            merged_speed = _reading_char_count(text) / max(merged.duration, 0.1)
            if (
                merged.duration > profile.max_duration + 0.001
                or len("".join(text.split())) > profile.max_chars_total
                or merged_speed > profile.max_chars_per_second + 0.001
            ):
                continue
            removed_boundary = right.start_char
            boundary_char = _previous_visible(source.display_text, removed_boundary)
            boundary_penalty = (
                2.0
                if boundary_char in STRONG_PUNCT
                else 1.0
                if boundary_char in MID_PUNCT
                else 0.0
            )
            ideal_mid = (profile.ideal_min_duration + profile.ideal_max_duration) / 2
            candidates.append(
                (
                    (
                        boundary_penalty,
                        abs(merged.duration - ideal_mid),
                        merged_speed,
                    ),
                    merge_start,
                    merged,
                )
            )
        if not candidates:
            index += 1
            continue

        _, merge_start, merged = min(candidates, key=lambda item: item[0])
        repaired[merge_start : merge_start + 2] = [merged]
        index = max(0, merge_start - 1)

    return [replace(cue, index=index + 1) for index, cue in enumerate(repaired)]


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
            if _exceeds_reading_speed_limit(
                text,
                duration,
                profile,
                source.language,
                is_terminal=end == count - 1,
            ):
                continue
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


def _reading_char_count(value: str) -> int:
    return sum(1 for char in value if not is_nonspoken_alignment_char(char))


def _exceeds_reading_speed_limit(
    text: str,
    duration: float,
    profile: SubtitleProfile,
    language: str,
    *,
    is_terminal: bool,
) -> bool:
    if duration <= 0 or profile.max_chars_per_second <= 0:
        return False
    limit = profile.max_chars_per_second
    if (
        language == "ja"
        and is_terminal
        and duration >= TERMINAL_CUE_MIN_EVIDENCE_SECONDS
    ):
        limit += TERMINAL_JA_CPS_TOLERANCE
    return _reading_char_count(text) / duration > limit + 0.001
