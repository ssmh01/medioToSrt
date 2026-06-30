"""Rule-based subtitle splitting and export text formatting."""

from __future__ import annotations

import re
from dataclasses import replace

from .errors import AlignmentError, ExportValidationError
from .models import AlignmentToken, SubtitleCue, SubtitleProfile
from .profiles import language_group
from .text import render_display_segment, validate_subtitle_continuity

STRONG_PUNCT = set("。！？!?…")
MID_PUNCT = set("，、,;；:：")
JA_BAD_EDGE = {"は", "が", "を", "に", "で", "と", "も", "の"}
EN_BAD_EDGE = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or"}
OPEN_QUOTES = set("「『“‘（([【")
CLOSE_QUOTES = set("」』”’）)]】")
JA_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
JA_KATAKANA_RE = re.compile(r"[\u30a0-\u30ffー]")
JA_KANJI_RE = re.compile(r"[\u3400-\u9fff々]")
JA_NUMERAL_RE = re.compile(r"[0-9０-９一二三四五六七八九十百千万億兆]")
JA_COUNTER_RE = re.compile(r"[円年月日人個本枚台階歳才分秒]")
JA_SMALL_KANA = {
    "っ",
    "ゃ",
    "ゅ",
    "ょ",
    "ぁ",
    "ぃ",
    "ぅ",
    "ぇ",
    "ぉ",
    "ッ",
    "ャ",
    "ュ",
    "ョ",
    "ァ",
    "ィ",
    "ゥ",
    "ェ",
    "ォ",
    "ー",
}
JA_REPAIR_SCAN_CHARS = 12
VISUAL_GAP_TARGET_SECONDS = 0.20


def split_subtitles(
    display_text: str,
    tokens: list[AlignmentToken],
    language: str,
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    tokens = [token for token in tokens if token.start_char is not None and token.end_char is not None]
    tokens.sort(key=lambda token: (token.start, token.start_char or 0))
    if not tokens:
        raise AlignmentError("没有可用于字幕切分的 token 时间戳")

    cues: list[SubtitleCue] = []
    token_start = 0
    char_start = 0
    while token_start < len(tokens):
        token_end = _choose_break(tokens, token_start, display_text, language, profile)
        next_token_start_char = (
            tokens[token_end + 1].start_char if token_end + 1 < len(tokens) else len(display_text)
        )
        char_end = max(char_start, next_token_start_char or len(display_text))
        text = render_display_segment(display_text[char_start:char_end])
        if not text:
            char_start = char_end
            token_start = token_end + 1
            continue
        wrapped = wrap_subtitle_text(text, language, profile.max_chars_per_line)
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start=tokens[token_start].start,
                end=tokens[token_end].end,
                text=wrapped,
                start_char=char_start,
                end_char=char_end,
            )
        )
        char_start = char_end
        token_start = token_end + 1

    cues = _repair_unsafe_boundaries(cues, display_text, language, profile)
    cues = _repair_timing(cues, profile, language)
    _annotate_warnings(cues, profile)
    if not validate_subtitle_continuity(cues, display_text):
        raise ExportValidationError("字幕正文未能连续覆盖原文，已停止导出以避免改写/漏字")
    return cues


def _choose_break(
    tokens: list[AlignmentToken],
    start_index: int,
    display_text: str,
    language: str,
    profile: SubtitleProfile,
) -> int:
    best_index = start_index
    best_score = float("-inf")
    hard_limit = start_index
    for index in range(start_index, len(tokens)):
        duration = tokens[index].end - tokens[start_index].start
        hard_limit = index
        char_end = tokens[index + 1].start_char if index + 1 < len(tokens) else len(display_text)
        text = render_display_segment(display_text[tokens[start_index].start_char or 0 : char_end])
        chars = _visible_len(text)
        if duration >= profile.min_duration:
            score = _break_score(tokens, index, text, duration, language, profile, display_text, char_end)
            if score > best_score:
                best_score = score
                best_index = index
        if duration >= profile.max_duration or chars >= _soft_char_limit(profile):
            break
    if best_score == float("-inf"):
        return hard_limit
    return best_index


def _break_score(
    tokens: list[AlignmentToken],
    index: int,
    text: str,
    duration: float,
    language: str,
    profile: SubtitleProfile,
    display_text: str,
    char_end: int,
) -> float:
    score = 0.0
    compact = text.strip()
    last = compact[-1:] if compact else ""
    if last in STRONG_PUNCT:
        score += 70
    elif last in MID_PUNCT:
        score += 35

    if profile.ideal_min_duration <= duration <= profile.ideal_max_duration:
        score += 24
    else:
        ideal_mid = (profile.ideal_min_duration + profile.ideal_max_duration) / 2
        score -= abs(duration - ideal_mid) * 4

    chars = _visible_len(text)
    ideal_chars = profile.max_chars_total * 0.72
    score -= abs(chars - ideal_chars) * 0.55
    if chars > profile.max_chars_total:
        score -= 18 + (chars - profile.max_chars_total) * 2.2
    if chars < max(4, profile.max_chars_per_line * 0.45) and index + 1 < len(tokens):
        score -= 18

    if index + 1 < len(tokens):
        gap = tokens[index + 1].start - tokens[index].end
        if gap >= 0.45:
            score += 16
        elif gap >= 0.2:
            score += 8

    if _has_bad_line_edge(compact, language):
        score -= 26
    score += _boundary_score(display_text, char_end, language)
    return score


def wrap_subtitle_text(text: str, _language: str, _max_chars_per_line: int) -> str:
    return render_display_segment(text)


def _repair_timing(cues: list[SubtitleCue], profile: SubtitleProfile, language: str = "zh") -> list[SubtitleCue]:
    repaired = list(cues)
    for idx, cue in enumerate(repaired):
        start = max(0.0, cue.start)
        end = max(start + 0.1, cue.end)
        if idx + 1 < len(repaired):
            next_start = repaired[idx + 1].start
            desired_end = next_start - profile.gap_seconds
            if end > desired_end and desired_end > start + 0.1:
                end = desired_end
            elif end > next_start:
                end = next_start
        repaired[idx] = replace(cue, index=idx + 1, start=start, end=end)
    return _smooth_timing(repaired, profile, language)


def _repair_unsafe_boundaries(
    cues: list[SubtitleCue],
    display_text: str,
    language: str,
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    if language_group(language) != "ja" or len(cues) < 2:
        return cues

    repaired = list(cues)
    for index in range(len(repaired) - 1):
        prev = repaired[index]
        cur = repaired[index + 1]
        boundary = prev.end_char
        if not _is_high_risk_boundary(display_text, boundary, language):
            continue

        target = _find_repair_boundary(display_text, prev, cur, boundary, language)
        if target is None or target == boundary:
            continue

        moved = _move_boundary(prev, cur, target, display_text, profile, language)
        if moved is None:
            continue
        repaired[index], repaired[index + 1] = moved

    return [replace(cue, index=index + 1) for index, cue in enumerate(repaired)]


def _find_repair_boundary(
    display_text: str,
    prev: SubtitleCue,
    cur: SubtitleCue,
    boundary: int,
    language: str,
) -> int | None:
    upper = min(cur.end_char - 1, boundary + JA_REPAIR_SCAN_CHARS)
    forward = range(boundary + 1, upper + 1)
    target = _best_repair_candidate(display_text, forward, boundary, language)
    if target is not None:
        return target

    lower = max(prev.start_char + 1, boundary - JA_REPAIR_SCAN_CHARS)
    backward = range(lower, boundary)
    return _best_repair_candidate(display_text, backward, boundary, language)


def _best_repair_candidate(
    display_text: str,
    candidates: range,
    original_boundary: int,
    language: str,
) -> int | None:
    best_boundary: int | None = None
    best_score = float("-inf")
    for candidate in candidates:
        if _is_high_risk_boundary(display_text, candidate, language):
            continue
        prev_char = _previous_visible_char(display_text, candidate)
        next_char = _next_visible_char(display_text, candidate)
        if not prev_char or not next_char:
            continue
        score = _repair_candidate_score(display_text, candidate, original_boundary, language)
        if score > best_score:
            best_score = score
            best_boundary = candidate
    return best_boundary


def _repair_candidate_score(display_text: str, char_end: int, original_boundary: int, language: str) -> float:
    prev_char = _previous_visible_char(display_text, char_end)
    next_char = _next_visible_char(display_text, char_end)
    score = _boundary_score(display_text, char_end, language)
    if prev_char in STRONG_PUNCT or prev_char in CLOSE_QUOTES:
        score += 90
    elif prev_char in MID_PUNCT:
        score += 45
    if language_group(language) == "ja":
        if prev_char in JA_BAD_EDGE:
            score += 12
        if next_char in JA_BAD_EDGE:
            score -= 38
        if _is_inside_unclosed_quote(display_text, char_end):
            score -= 6
    score -= abs(char_end - original_boundary) * 0.6
    return score


def _move_boundary(
    prev: SubtitleCue,
    cur: SubtitleCue,
    boundary: int,
    display_text: str,
    profile: SubtitleProfile,
    language: str,
) -> tuple[SubtitleCue, SubtitleCue] | None:
    prev_text = wrap_subtitle_text(
        render_display_segment(display_text[prev.start_char:boundary]),
        language,
        profile.max_chars_per_line,
    )
    cur_text = wrap_subtitle_text(
        render_display_segment(display_text[boundary:cur.end_char]),
        language,
        profile.max_chars_per_line,
    )
    if not prev_text or not cur_text:
        return None

    prev_chars = _visible_len(prev_text)
    cur_chars = _visible_len(cur_text)
    total_chars = prev_chars + cur_chars
    if total_chars <= 0:
        return None

    total_start = prev.start
    total_end = max(cur.end, cur.start + 0.1)
    total_duration = max(0.2, total_end - total_start)
    boundary_time = total_start + total_duration * (prev_chars / total_chars)
    min_end = total_start + 0.1
    max_end = total_end - profile.gap_seconds - 0.1
    if max_end <= min_end:
        return None
    boundary_time = min(max(boundary_time, min_end), max_end)
    cur_start = boundary_time + profile.gap_seconds
    if boundary_time - total_start > profile.max_duration + 0.75:
        return None
    if total_end - cur_start < 0.1:
        return None

    return (
        replace(prev, end=boundary_time, text=prev_text, end_char=boundary),
        replace(cur, start=cur_start, text=cur_text, start_char=boundary),
    )


def _smooth_timing(cues: list[SubtitleCue], profile: SubtitleProfile, language: str = "zh") -> list[SubtitleCue]:
    if not cues:
        return cues

    target_gap = VISUAL_GAP_TARGET_SECONDS
    min_gap = profile.gap_seconds
    target_cps = _target_chars_per_second(profile, language)
    smoothed = list(cues)

    for idx in range(len(smoothed) - 1):
        prev = smoothed[idx]
        cur = smoothed[idx + 1]
        gap = cur.start - prev.end
        if gap <= target_gap:
            continue

        cur_chars = _visible_len(cur.text)
        cur_duration = max(cur.duration, 0.1)
        extra_needed = max(0.0, (cur_chars / target_cps) - cur_duration)
        if extra_needed > 0:
            available_for_start = max(0.0, gap - min_gap)
            max_duration_shift = max(0.0, profile.max_duration - cur.duration)
            shift_start = min(extra_needed, available_for_start, max_duration_shift)
            if shift_start > 0:
                cur = replace(cur, start=cur.start - shift_start)
                smoothed[idx + 1] = cur
                gap = cur.start - prev.end

        desired_end = cur.start - target_gap
        max_end = min(cur.start - min_gap, prev.start + profile.max_duration)
        new_end = min(desired_end, max_end)
        if new_end > prev.end:
            prev = replace(prev, end=new_end)
            smoothed[idx] = prev
            gap = cur.start - prev.end

        cur = _shift_cue_start_for_gap(cur, prev.end, target_gap, min_gap, profile)
        smoothed[idx + 1] = cur

    repaired: list[SubtitleCue] = []
    for idx, cue in enumerate(smoothed):
        start = max(0.0, cue.start)
        end = max(start + 0.1, cue.end)
        end = min(end, start + profile.max_duration)
        if idx + 1 < len(smoothed):
            end = min(end, smoothed[idx + 1].start - min_gap)
        repaired.append(replace(cue, index=idx + 1, start=start, end=max(start + 0.1, end)))
    return repaired


def _shift_cue_start_for_gap(
    cue: SubtitleCue,
    previous_end: float,
    target_gap: float,
    min_gap: float,
    profile: SubtitleProfile,
) -> SubtitleCue:
    gap = cue.start - previous_end
    if gap <= target_gap:
        return cue

    max_shift_by_gap = max(0.0, gap - target_gap)
    max_shift_by_duration = max(0.0, profile.max_duration - cue.duration)
    max_shift_by_min_gap = max(0.0, cue.start - previous_end - min_gap)
    shift = min(max_shift_by_gap, max_shift_by_duration, max_shift_by_min_gap)
    if shift <= 0:
        return cue
    return replace(cue, start=cue.start - shift)


def _annotate_warnings(cues: list[SubtitleCue], profile: SubtitleProfile) -> None:
    for index, cue in enumerate(cues):
        if cue.duration < profile.min_duration:
            cue.warnings.append("too_short")
        if cue.duration > profile.max_duration:
            cue.warnings.append("too_long")
        if _visible_len(cue.text) / max(cue.duration, 0.1) > profile.max_chars_per_second:
            cue.warnings.append("fast_reading")
        if _visible_len(cue.text) > profile.max_chars_total:
            cue.warnings.append("cue_too_long")
        if index and cue.start < cues[index - 1].end:
            cue.warnings.append("overlap")


def _visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _soft_char_limit(profile: SubtitleProfile) -> int:
    return profile.max_chars_total + max(8, int(profile.max_chars_per_line * 0.6))


def _target_chars_per_second(profile: SubtitleProfile, language: str) -> float:
    group = language_group(language)
    if group == "ja":
        return min(profile.max_chars_per_second, 8.5)
    if group == "cjk":
        return min(profile.max_chars_per_second, 9.5)
    return min(profile.max_chars_per_second, 18.0)


def _boundary_score(display_text: str, char_end: int, language: str) -> float:
    group = language_group(language)
    prev_char = _previous_visible_char(display_text, char_end)
    next_char = _next_visible_char(display_text, char_end)
    if not prev_char or not next_char:
        return 0.0

    score = 0.0
    if prev_char in STRONG_PUNCT or prev_char in CLOSE_QUOTES:
        score += 18
    elif prev_char in MID_PUNCT:
        score += 10
    if next_char in STRONG_PUNCT or next_char in MID_PUNCT or next_char in CLOSE_QUOTES:
        score -= 18
    if prev_char in OPEN_QUOTES:
        score -= 55

    if group == "ja":
        score += _ja_boundary_score(prev_char, next_char)
        if _is_inside_unclosed_quote(display_text, char_end) and not _is_clear_boundary(prev_char):
            score -= 28
    return score


def _ja_boundary_score(prev_char: str, next_char: str) -> float:
    if _is_katakana(prev_char) and _is_katakana(next_char):
        return -78
    if _is_kanji(prev_char) and _is_kanji(next_char):
        return -78
    if _is_kanji(prev_char) and _is_hiragana(next_char) and next_char not in JA_BAD_EDGE:
        return -68
    if _is_hiragana(prev_char) and _is_kanji(next_char) and prev_char not in JA_BAD_EDGE:
        return -68
    if _is_hiragana(prev_char) and _is_hiragana(next_char):
        if prev_char in JA_BAD_EDGE:
            return 0
        return -48
    if _is_numeral(prev_char) and (_is_numeral(next_char) or JA_COUNTER_RE.fullmatch(next_char)):
        return -78
    if JA_COUNTER_RE.fullmatch(prev_char) and _is_numeral(next_char):
        return -78
    if prev_char in {"っ", "ッ", "ー"} or next_char in JA_SMALL_KANA:
        return -78
    return 0.0


def _is_high_risk_boundary(display_text: str, char_end: int, language: str) -> bool:
    if language_group(language) != "ja":
        return False
    prev_char = _previous_visible_char(display_text, char_end)
    next_char = _next_visible_char(display_text, char_end)
    if not prev_char or not next_char:
        return False
    if _is_clear_boundary(prev_char):
        return False
    if prev_char in OPEN_QUOTES or next_char in CLOSE_QUOTES:
        return True
    return _ja_boundary_score(prev_char, next_char) <= -48


def _is_clear_boundary(char: str) -> bool:
    return char in STRONG_PUNCT or char in MID_PUNCT or char in CLOSE_QUOTES


def _is_inside_unclosed_quote(text: str, char_end: int) -> bool:
    depth = 0
    for char in text[:char_end]:
        if char in OPEN_QUOTES:
            depth += 1
        elif char in CLOSE_QUOTES and depth:
            depth -= 1
    return depth > 0


def _previous_visible_char(text: str, end: int) -> str:
    for char in reversed(text[:end]):
        if not char.isspace():
            return char
    return ""


def _next_visible_char(text: str, start: int) -> str:
    for char in text[start:]:
        if not char.isspace():
            return char
    return ""


def _is_hiragana(char: str) -> bool:
    return bool(JA_HIRAGANA_RE.fullmatch(char))


def _is_katakana(char: str) -> bool:
    return bool(JA_KATAKANA_RE.fullmatch(char))


def _is_kanji(char: str) -> bool:
    return bool(JA_KANJI_RE.fullmatch(char))


def _is_numeral(char: str) -> bool:
    return bool(JA_NUMERAL_RE.fullmatch(char))


def _has_bad_line_edge(text: str, language: str) -> bool:
    group = language_group(language)
    if group == "ja":
        return text[-1:] in JA_BAD_EDGE
    if group == "en":
        words = re.findall(r"[A-Za-z']+", text)
        return bool(words and words[-1].lower() in EN_BAD_EDGE)
    return False
