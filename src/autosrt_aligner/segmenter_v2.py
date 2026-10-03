"""Evidence-constrained semantic cue segmentation for the V2 pipeline."""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import replace

from .errors import AlignmentError
from .models import AlignmentToken, SourceDocument, SubtitleCue, SubtitleProfile
from .language_rules import (
    analyze,
    boundary_kind,
    display_boundaries,
    lexical_penalty,
    make_boundary_map,
    sentence_ends,
)
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
LANGUAGE_TARGETS = {
    "zh": (3.0, 22), "ja": (3.0, 23), "ko": (3.1, 27), "en": (3.3, 62),
}


def segment_cues(
    source: SourceDocument,
    tokens: list[AlignmentToken],
    profile: SubtitleProfile,
    audio_duration: float | None = None,
) -> list[SubtitleCue]:
    """Select cue boundaries from existing token timestamps only."""

    ordered = sorted(
        (token for token in tokens if token.start_char is not None and token.end_char is not None),
        key=lambda token: (token.start_char or 0, token.start, token.end),
    )
    if not ordered:
        raise AlignmentError("没有可用于语义切分的对齐 token")

    analysis = analyze(source.display_text, source.language)
    boundaries = display_boundaries(source.display_text, ordered)
    path = _best_path(source, ordered, profile, analysis, boundaries, audio_duration)
    cues: list[SubtitleCue] = []
    char_start = 0
    token_start = 0
    for token_end in path:
        char_end = boundaries[token_end]
        text = render_display_segment(source.display_text[char_start:char_end])
        if text:
            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start=ordered[token_start].start,
                    end=ordered[token_end].end,
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


def structural_ends(text: str) -> list[int]:
    """Ends of paragraph-separated sentences and standalone quotations."""
    result = []
    for match in re.finditer(r"\n+", text):
        left = text[: match.start()].rstrip()
        if not left or left[-1] not in '。！？!? .」』”’"':
            continue
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        if end < len(text):
            result.append(end)
    return result


def utterance_ends(text: str, language: str) -> set[int]:
    return set(sentence_ends(text, language)) | set(structural_ends(text))


def minimum_cue_duration(
    start_char: int,
    end_char: int,
    profile: SubtitleProfile,
    complete_ends: set[int],
) -> float:
    """A short complete utterance may stand alone; fragments use the normal limit."""
    if (start_char == 0 or start_char in complete_ends) and end_char in complete_ends:
        return (
            profile.min_complete_duration
            if profile.min_complete_duration is not None
            else profile.min_duration
        )
    return max(1.2, profile.min_duration)


def _best_path(
    source: SourceDocument,
    tokens: list[AlignmentToken],
    profile: SubtitleProfile,
    analysis: dict,
    boundaries: list[int],
    audio_duration: float | None,
) -> list[int]:
    text = source.display_text
    language = source.language
    count = len(tokens)
    maps = make_boundary_map(text, language, analysis)
    morphology = language != "ko"
    kinds = [boundary_kind(text, end, language) for end in boundaries]
    allowed = [_is_safe_boundary(text, end, language) for end in boundaries]
    if morphology:
        allowed = [
            safe and not maps["word_inside"][end]
            for safe, end in zip(allowed, boundaries)
        ]
    char_counts = [0]
    for char in text:
        char_counts.append(char_counts[-1] + int(not char.isspace()))
    complete_ends = utterance_ends(text, language)
    strong = sorted(complete_ends)
    for index, end in enumerate(boundaries):
        if end in complete_ends:
            kinds[index] = "sentence"
    critical = structural_ends(text)
    gap_counts = [0]
    for index, token in enumerate(tokens):
        gap = tokens[index + 1].start - token.end if index + 1 < count else 0
        gap_counts.append(gap_counts[-1] + int(gap >= 0.8 - 0.001))
    target_seconds, target_chars = LANGUAGE_TARGETS[language]
    costs = [(math.inf, math.inf, math.inf, math.inf)] * (count + 1)
    next_end: list[int | None] = [None] * count
    costs[count] = (0, 0, 0, 0.0)
    for start in range(count - 1, -1, -1):
        char_start = boundaries[start - 1] if start else 0
        for end in range(start, count):
            duration = tokens[end].end - tokens[start].start
            effective_duration = duration
            if end == count - 1 and audio_duration is not None:
                effective_duration = max(tokens[end].end, audio_duration) - tokens[start].start
            if effective_duration > profile.max_duration + 0.001:
                break
            char_end = boundaries[end]
            chars = char_counts[char_end] - char_counts[char_start]
            if chars > profile.max_chars_total:
                break
            if duration <= 0:
                continue
            if end < count - 1:
                minimum = minimum_cue_duration(
                    char_start, char_end, profile, complete_ends
                )
                if duration < minimum - 0.001 or not allowed[end]:
                    continue
            if not math.isfinite(costs[end + 1][0]):
                continue
            if _exceeds_reading_speed_limit(
                text[char_start:char_end],
                duration,
                profile,
                language,
                is_terminal=end == count - 1,
            ):
                continue
            kind = kinds[end]
            internal = bisect.bisect_left(strong, char_end) - bisect.bisect_right(
                strong, char_start
            )
            # Positive per-cue costs balance complete thoughts against reading length.
            cost = (
                12
                + 0.6 * (duration - target_seconds) ** 2
                + 8 * max(0, 2.0 - duration) ** 2
            )
            cost += 3.0 * ((chars - target_chars) / target_chars) ** 2
            if kind == "clause":
                cost += 5
                if text[:char_end].rstrip().endswith("、"):
                    cost += 20
                if morphology and maps["phrase_inside"][char_end]:
                    cost += 40
            elif kind == "weak":
                cost += lexical_penalty(language, char_end, maps) if morphology else 30
            cost += internal * (45 if kind == "sentence" else 90)
            cost += (
                bisect.bisect_left(critical, char_end)
                - bisect.bisect_right(critical, char_start)
            ) * 60
            cost += (gap_counts[end] - gap_counts[start]) * 110
            if end < count - 1:
                gap = tokens[end + 1].start - tokens[end].end
                if kind == "weak" and gap >= 0.45:
                    cost -= min(8, cost - 1)
                if morphology and maps["phrase_inside"][char_end] and kind == "weak":
                    cost += 70
            tail = costs[end + 1]
            if language == "en":
                # English paths prioritize core syntax, complete thoughts, then phrases.
                core_break = int(end < count - 1 and maps["core_inside"][char_end])
                incomplete_sentence = internal if kind != "sentence" else 0
                phrase_break = int(end < count - 1 and maps["phrase_inside"][char_end])
                total = (
                    core_break + tail[0],
                    incomplete_sentence + tail[1],
                    phrase_break + tail[2],
                    cost + tail[3],
                )
            else:
                total = (0, 0, 0, cost + tail[3])
            if total < costs[start]:
                costs[start] = total
                next_end[start] = end
    if next_end[0] is None:
        raise AlignmentError(
            "无法在现有 token 时间证据下满足字幕时长、字数和阅读速度限制"
        )
    path = []
    start = 0
    while start < count:
        end = next_end[start]
        if end is None:
            raise AlignmentError("语义切分路径不完整")
        path.append(end)
        start = end + 1
    return path

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
