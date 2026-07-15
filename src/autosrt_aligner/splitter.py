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
ZH_BAD_EDGE = {
    "的",
    "地",
    "得",
    "把",
    "被",
    "和",
    "与",
    "跟",
    "在",
    "是",
    "就",
    "也",
    "都",
    "要",
    "会",
    "能",
    "很",
    "又",
    "还",
    "再",
}
ZH_QUOTE_FOLLOWERS = set("的一声地了着过吗呢吧啊呀嘛她他它们")
EN_BAD_EDGE = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or"}
OPEN_QUOTES = set("「『“‘（([【")
CLOSE_QUOTES = set("」』”’）)]】")
STRAIGHT_QUOTES = set("\"'")
QUOTE_CHARS = OPEN_QUOTES | CLOSE_QUOTES | STRAIGHT_QUOTES
ZH_CJK_RE = re.compile(r"[\u3400-\u9fff]")
JA_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
JA_KATAKANA_RE = re.compile(r"[\u30a0-\u30ffー]")
JA_KANJI_RE = re.compile(r"[\u3400-\u9fff々]")
JA_NUMERAL_RE = re.compile(r"[0-9０-９一二三四五六七八九十百千万億兆]")
JA_COUNTER_RE = re.compile(r"[円年月日人個本枚台階歳才分秒]")
KO_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
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
ZH_REPAIR_SCAN_CHARS = 12
KO_REPAIR_SCAN_CHARS = 16
ZH_FORCE_SPLIT_CHARS = 38
ZH_LONG_CUE_CHARS = 34
ZH_COMFORT_CPS = 8.5
KO_FORCE_SPLIT_CHARS = 44
KO_LONG_CUE_CHARS = 38
KO_COMFORT_CPS = 9.5
KO_SEVERE_FAST_CPS = 10.5
KO_REBALANCE_WINDOW_SIZE = 7
KO_REBALANCE_GAP_BARRIER_SECONDS = 0.35
VISUAL_GAP_TARGET_SECONDS = 0.20
TIMING_EPSILON = 0.001


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
    cues = _split_overlong_zh_cues(cues, display_text, tokens, language, profile)
    cues = _repair_unsafe_boundaries(cues, display_text, language, profile)
    cues = _repair_timing(cues, profile, language)
    cues = _rebalance_korean_timeline(cues, display_text, language, profile)
    cues = _repair_timing(cues, profile, language)
    cues = _merge_short_english_cues(cues, display_text, language, profile)
    cues = _repair_timing(cues, profile, language)
    _annotate_warnings(cues, profile, language)
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
        if duration >= profile.max_duration or chars >= _soft_char_limit(profile, language):
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
    ideal_chars = _ideal_chars(profile, language)
    score -= abs(chars - ideal_chars) * 0.55
    if chars > profile.max_chars_total:
        score -= 18 + (chars - profile.max_chars_total) * 2.2
    group = language_group(language)
    if group == "cjk":
        if chars > ZH_LONG_CUE_CHARS:
            score -= (chars - ZH_LONG_CUE_CHARS) * 3.8
        if chars > ZH_FORCE_SPLIT_CHARS:
            score -= 50 + (chars - ZH_FORCE_SPLIT_CHARS) * 5.0
    elif group == "ko":
        if chars > KO_LONG_CUE_CHARS:
            score -= (chars - KO_LONG_CUE_CHARS) * 3.2
        if chars > KO_FORCE_SPLIT_CHARS:
            score -= 46 + (chars - KO_FORCE_SPLIT_CHARS) * 4.4
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
    if language_group(language) not in {"ja", "cjk", "ko", "en"} or len(cues) < 2:
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


def _split_overlong_zh_cues(
    cues: list[SubtitleCue],
    display_text: str,
    tokens: list[AlignmentToken],
    language: str,
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    if language_group(language) not in {"cjk", "ko"}:
        return cues

    split_cues: list[SubtitleCue] = []
    for index, cue in enumerate(cues):
        next_gap = cues[index + 1].start - cue.end if index + 1 < len(cues) else 0.0
        pending = [cue]
        changed = True
        while changed:
            changed = False
            next_pending: list[SubtitleCue] = []
            for item in pending:
                if not _should_split_zh_cue(item, next_gap, profile, language):
                    next_pending.append(item)
                    continue
                moved = _split_zh_cue(item, display_text, tokens, language, profile)
                if moved is None:
                    next_pending.append(item)
                    continue
                first, second = moved
                next_pending.extend([first, second])
                changed = True
            pending = next_pending
        split_cues.extend(pending)

    return [replace(cue, index=index + 1) for index, cue in enumerate(split_cues)]


def _should_split_zh_cue(
    cue: SubtitleCue,
    gap_after: float,
    profile: SubtitleProfile,
    language: str,
) -> bool:
    chars = _visible_len(cue.text)
    cps = chars / max(cue.duration, 0.1)
    force_chars = _force_split_chars(language)
    long_chars = _long_cue_chars(language)
    comfort_cps = _comfort_chars_per_second(language)
    if language_group(language) == "ko" and chars > profile.max_chars_total:
        return True
    if chars > force_chars:
        return True
    if cps > comfort_cps:
        return True
    if cue.duration >= profile.max_duration - 0.05 and chars > max(32, int(long_chars * 0.85)):
        return True
    return gap_after > 0.8 and cue.duration >= profile.max_duration - 0.05


def _split_zh_cue(
    cue: SubtitleCue,
    display_text: str,
    tokens: list[AlignmentToken],
    language: str,
    profile: SubtitleProfile,
) -> tuple[SubtitleCue, SubtitleCue] | None:
    boundary = _best_zh_split_boundary(display_text, cue.start_char, cue.end_char, profile, language)
    if boundary is None:
        return None
    return _split_cue_at_boundary(cue, boundary, display_text, tokens, profile, language)


def _best_zh_split_boundary(
    display_text: str,
    start_char: int,
    end_char: int,
    profile: SubtitleProfile,
    language: str,
) -> int | None:
    total_visible = _visible_len(display_text[start_char:end_char])
    if total_visible < 20:
        return None

    best_boundary: int | None = None
    best_score = float("-inf")
    min_left = max(8, int(profile.max_chars_per_line * 0.55))
    min_right = 6
    long_chars = _long_cue_chars(language)
    force_chars = _force_split_chars(language)
    for candidate in range(start_char + 1, end_char):
        if _is_high_risk_boundary(display_text, candidate, language):
            continue
        left_chars = _visible_len(display_text[start_char:candidate])
        right_chars = _visible_len(display_text[candidate:end_char])
        if left_chars < min_left or right_chars < min_right:
            continue
        score = _repair_candidate_score(display_text, candidate, (start_char + end_char) // 2, language)
        score -= abs(left_chars - min(long_chars, max(24, total_visible // 2))) * 0.9
        if left_chars > force_chars:
            score -= 60
        if right_chars > force_chars:
            score -= 60
        if left_chars <= long_chars and right_chars <= long_chars:
            score += 18
        if score > best_score:
            best_score = score
            best_boundary = candidate
    return best_boundary


def _split_cue_at_boundary(
    cue: SubtitleCue,
    boundary: int,
    display_text: str,
    tokens: list[AlignmentToken],
    profile: SubtitleProfile,
    language: str,
) -> tuple[SubtitleCue, SubtitleCue] | None:
    first_text = wrap_subtitle_text(
        render_display_segment(display_text[cue.start_char:boundary]),
        language,
        profile.max_chars_per_line,
    )
    second_text = wrap_subtitle_text(
        render_display_segment(display_text[boundary:cue.end_char]),
        language,
        profile.max_chars_per_line,
    )
    if not first_text or not second_text:
        return None

    boundary_time = _boundary_time_from_tokens(tokens, boundary, cue)
    min_duration = profile.min_duration if language_group(language) == "ko" else 0.45
    if boundary_time - cue.start < min_duration or cue.end - boundary_time < min_duration:
        return None
    first_end = min(boundary_time, cue.end - profile.gap_seconds - min_duration)
    second_start = first_end + profile.gap_seconds
    if first_end <= cue.start + min_duration or second_start >= cue.end - min_duration:
        return None

    return (
        replace(cue, end=first_end, text=first_text, end_char=boundary, warnings=[]),
        replace(cue, start=second_start, text=second_text, start_char=boundary, warnings=[]),
    )


def _boundary_time_from_tokens(tokens: list[AlignmentToken], boundary: int, cue: SubtitleCue) -> float:
    prev_token = max(
        (token for token in tokens if token.end_char is not None and token.end_char <= boundary),
        key=lambda token: token.end_char or 0,
        default=None,
    )
    next_token = min(
        (token for token in tokens if token.start_char is not None and token.start_char >= boundary),
        key=lambda token: token.start_char or 0,
        default=None,
    )
    if prev_token and next_token and cue.start <= prev_token.end <= next_token.start <= cue.end:
        return (prev_token.end + next_token.start) / 2

    left_chars = max(0, boundary - cue.start_char)
    total_chars = max(1, cue.end_char - cue.start_char)
    ratio = min(0.9, max(0.1, left_chars / total_chars))
    return cue.start + cue.duration * ratio


def _find_repair_boundary(
    display_text: str,
    prev: SubtitleCue,
    cur: SubtitleCue,
    boundary: int,
    language: str,
) -> int | None:
    group = language_group(language)
    if group == "cjk":
        scan_chars = ZH_REPAIR_SCAN_CHARS
    elif group == "ko":
        scan_chars = KO_REPAIR_SCAN_CHARS
    else:
        scan_chars = JA_REPAIR_SCAN_CHARS
    upper = min(cur.end_char - 1, boundary + scan_chars)
    forward = range(boundary + 1, upper + 1)
    target = _best_repair_candidate(display_text, forward, boundary, language)
    if target is not None:
        return target

    lower = max(prev.start_char + 1, boundary - scan_chars)
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
    group = language_group(language)
    if group == "ja":
        if prev_char in JA_BAD_EDGE:
            score += 12
        if next_char in JA_BAD_EDGE:
            score -= 38
        if _is_inside_unclosed_quote(display_text, char_end):
            score -= 6
    elif group == "cjk":
        if prev_char in ZH_BAD_EDGE:
            score -= 24
        if next_char in ZH_BAD_EDGE:
            score -= 8
        if _is_zh_quote_boundary_risk(display_text, char_end):
            score -= 90
    elif group == "ko":
        if _is_hangul(prev_char) and _is_hangul(next_char) and not _has_space_at_boundary(display_text, char_end):
            score -= 70
        if _has_space_at_boundary(display_text, char_end):
            score += 24
        if _is_inside_unclosed_quote(display_text, char_end):
            score -= 6
    elif group == "en":
        if _has_space_at_boundary(display_text, char_end):
            score += 34
        if _is_english_word_internal_boundary(display_text, char_end):
            score -= 90
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
    timing_max_duration = _timing_max_duration(profile, language)
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
            max_duration_shift = max(0.0, timing_max_duration - cur.duration)
            shift_start = min(extra_needed, available_for_start, max_duration_shift)
            if shift_start > 0:
                cur = replace(cur, start=cur.start - shift_start)
                smoothed[idx + 1] = cur
                gap = cur.start - prev.end

        desired_end = cur.start - target_gap
        max_end = min(cur.start - min_gap, prev.start + timing_max_duration)
        new_end = min(desired_end, max_end)
        if new_end > prev.end:
            prev = replace(prev, end=new_end)
            smoothed[idx] = prev
            gap = cur.start - prev.end

        cur = _shift_cue_start_for_gap(cur, prev.end, target_gap, min_gap, timing_max_duration)
        smoothed[idx + 1] = cur

    repaired: list[SubtitleCue] = []
    for idx, cue in enumerate(smoothed):
        start = max(0.0, cue.start)
        end = max(start + 0.1, cue.end)
        end = min(end, start + timing_max_duration)
        if idx + 1 < len(smoothed):
            end = min(end, smoothed[idx + 1].start - min_gap)
        repaired.append(replace(cue, index=idx + 1, start=start, end=max(start + 0.1, end)))
    return repaired


def _shift_cue_start_for_gap(
    cue: SubtitleCue,
    previous_end: float,
    target_gap: float,
    min_gap: float,
    timing_max_duration: float,
) -> SubtitleCue:
    gap = cue.start - previous_end
    if gap <= target_gap:
        return cue

    max_shift_by_gap = max(0.0, gap - target_gap)
    max_shift_by_duration = max(0.0, timing_max_duration - cue.duration)
    max_shift_by_min_gap = max(0.0, cue.start - previous_end - min_gap)
    shift = min(max_shift_by_gap, max_shift_by_duration, max_shift_by_min_gap)
    if shift <= 0:
        return cue
    return replace(cue, start=cue.start - shift)


def _rebalance_korean_timeline(
    cues: list[SubtitleCue],
    display_text: str,
    language: str,
    profile: SubtitleProfile,
    trigger_cps: float = KO_SEVERE_FAST_CPS,
) -> list[SubtitleCue]:
    if language_group(language) != "ko" or len(cues) < 2:
        return cues

    rebalanced = _merge_short_korean_tail_cue(list(cues), display_text, profile, language)
    index = 0
    while index < len(rebalanced):
        if _korean_cue_cps(rebalanced[index]) <= trigger_cps:
            index += 1
            continue

        run_start, run_end = _korean_fast_run(rebalanced, index)
        candidate = _find_korean_rebalance_window(
            rebalanced,
            run_start,
            run_end,
            display_text,
            profile,
            language,
        )
        if candidate is None:
            index = run_end + 1
            continue

        window_start, window_end, window = candidate
        rebalanced[window_start : window_end + 1] = window
        index = window_end + 1

    rebalanced = _merge_short_korean_tail_cue(rebalanced, display_text, profile, language)
    return [replace(cue, index=index + 1) for index, cue in enumerate(rebalanced)]


def _finalize_korean_cues(
    cues: list[SubtitleCue],
    display_text: str,
    language: str,
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    if language_group(language) != "ko":
        return cues

    finalized = [replace(cue, warnings=[]) for cue in cues]
    finalized = _repair_timing(finalized, profile, language)
    finalized = _rebalance_korean_timeline(
        finalized,
        display_text,
        language,
        profile,
        trigger_cps=KO_COMFORT_CPS,
    )
    finalized = _split_overlong_zh_cues(finalized, display_text, [], language, profile)
    finalized = _repair_unsafe_boundaries(finalized, display_text, language, profile)
    finalized = _repair_timing(finalized, profile, language)
    finalized = _rebalance_korean_timeline(
        finalized,
        display_text,
        language,
        profile,
        trigger_cps=KO_COMFORT_CPS,
    )
    finalized = _repair_timing(finalized, profile, language)
    _annotate_warnings(finalized, profile, language)
    if not validate_subtitle_continuity(finalized, display_text):
        raise ExportValidationError("韩语最终质量收口未能连续覆盖原文")
    return finalized


def _korean_cue_cps(cue: SubtitleCue) -> float:
    chars = _visible_len(cue.text)
    return chars / max(cue.duration, 0.1)


def _korean_fast_run(cues: list[SubtitleCue], severe_index: int) -> tuple[int, int]:
    start_index = severe_index
    while (
        start_index > 0
        and _can_rebalance_across_korean_gap(cues[start_index - 1], cues[start_index])
        and _korean_cue_cps(cues[start_index - 1]) > KO_COMFORT_CPS
    ):
        start_index -= 1

    end_index = severe_index
    while (
        end_index + 1 < len(cues)
        and _can_rebalance_across_korean_gap(cues[end_index], cues[end_index + 1])
        and _korean_cue_cps(cues[end_index + 1]) > KO_COMFORT_CPS
    ):
        end_index += 1
    return start_index, end_index


def _find_korean_rebalance_window(
    cues: list[SubtitleCue],
    run_start: int,
    run_end: int,
    display_text: str,
    profile: SubtitleProfile,
    language: str,
) -> tuple[int, int, list[SubtitleCue]] | None:
    run_size = run_end - run_start + 1
    if run_size > KO_REBALANCE_WINDOW_SIZE:
        return None

    candidate_ranges: list[tuple[int, float, int, int]] = []
    min_start = max(0, run_end - KO_REBALANCE_WINDOW_SIZE + 1)
    max_end = min(len(cues) - 1, run_start + KO_REBALANCE_WINDOW_SIZE - 1)
    for start_index in range(min_start, run_start + 1):
        for end_index in range(run_end, max_end + 1):
            size = end_index - start_index + 1
            if size < 2 or size > KO_REBALANCE_WINDOW_SIZE:
                continue
            window = cues[start_index : end_index + 1]
            if any(
                not _can_rebalance_across_korean_gap(prev, cur)
                for prev, cur in zip(window, window[1:])
            ):
                continue
            slack = sum(max(0.0, cue.duration - (_visible_len(cue.text) / KO_COMFORT_CPS)) for cue in window)
            candidate_ranges.append((size, -slack, start_index, end_index))

    for _, _, start_index, end_index in sorted(candidate_ranges):
        window = _rebalance_korean_window(
            cues[start_index : end_index + 1],
            display_text,
            profile,
            language,
        )
        if window is not None:
            return start_index, end_index, window
    return None


def _can_rebalance_across_korean_gap(prev: SubtitleCue, cur: SubtitleCue) -> bool:
    gap = cur.start - prev.end
    return -TIMING_EPSILON <= gap <= KO_REBALANCE_GAP_BARRIER_SECONDS + TIMING_EPSILON


def _rebalance_korean_window(
    cues: list[SubtitleCue],
    display_text: str,
    profile: SubtitleProfile,
    language: str,
) -> list[SubtitleCue] | None:
    if len(cues) < 2:
        return None

    start_char = cues[0].start_char
    end_char = cues[-1].end_char
    boundaries = _balanced_korean_boundaries(display_text, start_char, end_char, len(cues), profile, language)
    if boundaries is None:
        return None

    ranges = list(zip([start_char, *boundaries], [*boundaries, end_char]))
    texts = [
        wrap_subtitle_text(render_display_segment(display_text[start:end]), language, profile.max_chars_per_line)
        for start, end in ranges
    ]
    if any(not text for text in texts):
        return None

    chars = [_visible_len(text) for text in texts]
    total_chars = sum(chars)
    if total_chars <= 0 or any(count > profile.max_chars_total for count in chars):
        return None

    min_gap = profile.gap_seconds
    total_start = cues[0].start
    total_end = cues[-1].end
    available_duration = total_end - total_start - min_gap * (len(cues) - 1)
    if available_duration <= profile.min_duration:
        return None
    target_cps = min(profile.max_chars_per_second, KO_COMFORT_CPS)
    if total_chars / available_duration > target_cps:
        return None

    durations = [available_duration * (count / total_chars) for count in chars]
    timing_max_duration = _timing_max_duration(profile, language)
    if any(duration + TIMING_EPSILON < profile.min_duration for duration in durations):
        return None
    if any(duration > timing_max_duration + TIMING_EPSILON for duration in durations):
        return None
    if any(count / max(duration, 0.1) > target_cps for count, duration in zip(chars, durations)):
        return None

    rebalanced: list[SubtitleCue] = []
    cursor = total_start
    for offset, ((start, end), text, duration) in enumerate(zip(ranges, texts, durations)):
        cue_end = total_end if offset == len(cues) - 1 else cursor + duration
        rebalanced.append(
            replace(
                cues[offset],
                start=cursor,
                end=cue_end,
                text=text,
                start_char=start,
                end_char=end,
                warnings=[],
            )
        )
        cursor = cue_end + min_gap
    return rebalanced


def _balanced_korean_boundaries(
    display_text: str,
    start_char: int,
    end_char: int,
    count: int,
    profile: SubtitleProfile,
    language: str,
) -> list[int] | None:
    boundaries: list[int] = []
    segment_start = start_char
    remaining_segments = count
    while remaining_segments > 1:
        remaining_visible = _visible_len(display_text[segment_start:end_char])
        target_visible = max(8, round(remaining_visible / remaining_segments))
        boundary = _best_korean_boundary_near_visible_count(
            display_text,
            segment_start,
            end_char,
            target_visible,
            remaining_segments - 1,
            profile,
            language,
        )
        if boundary is None:
            return None
        boundaries.append(boundary)
        segment_start = boundary
        remaining_segments -= 1
    return boundaries


def _best_korean_boundary_near_visible_count(
    display_text: str,
    start_char: int,
    end_char: int,
    target_visible: int,
    remaining_segments: int,
    profile: SubtitleProfile,
    language: str,
) -> int | None:
    best_boundary: int | None = None
    best_score = float("-inf")
    min_left = 8
    min_right = max(6, remaining_segments * 6)
    for candidate in range(start_char + 1, end_char):
        if _is_high_risk_boundary(display_text, candidate, language):
            continue
        left_chars = _visible_len(display_text[start_char:candidate])
        right_chars = _visible_len(display_text[candidate:end_char])
        if left_chars < min_left or right_chars < min_right:
            continue

        score = _repair_candidate_score(display_text, candidate, start_char + target_visible, language)
        score -= abs(left_chars - target_visible) * 2.6
        if left_chars > profile.max_chars_total:
            score -= (left_chars - profile.max_chars_total) * 3.0
        if right_chars > remaining_segments * profile.max_chars_total:
            score -= 35
        if _previous_visible_char(display_text, candidate) in STRONG_PUNCT:
            score += 30
        if score > best_score:
            best_score = score
            best_boundary = candidate
    return best_boundary


def _merge_short_korean_tail_cue(
    cues: list[SubtitleCue],
    display_text: str,
    profile: SubtitleProfile,
    language: str,
) -> list[SubtitleCue]:
    if len(cues) < 2:
        return cues

    last = cues[-1]
    prev = cues[-2]
    if last.duration >= profile.min_duration:
        return cues

    merged_text = wrap_subtitle_text(
        render_display_segment(display_text[prev.start_char : last.end_char]),
        language,
        profile.max_chars_per_line,
    )
    merged_chars = _visible_len(merged_text)
    merged_duration = last.end - prev.start
    if (
        not merged_text
        or merged_duration > _timing_max_duration(profile, language) + TIMING_EPSILON
        or merged_chars > profile.max_chars_total
        or merged_chars / max(merged_duration, 0.1) > profile.max_chars_per_second
    ):
        return cues

    merged = replace(prev, end=last.end, text=merged_text, end_char=last.end_char, warnings=[])
    return [*cues[:-2], merged]


def _merge_short_english_cues(
    cues: list[SubtitleCue],
    display_text: str,
    language: str,
    profile: SubtitleProfile,
) -> list[SubtitleCue]:
    if language_group(language) != "en" or len(cues) < 2:
        return cues

    merged = list(cues)
    index = len(merged) - 1
    while index > 0:
        cue = merged[index]
        if cue.duration >= profile.min_duration:
            index -= 1
            continue

        prev = merged[index - 1]
        merged_text = wrap_subtitle_text(
            render_display_segment(display_text[prev.start_char : cue.end_char]),
            language,
            profile.max_chars_per_line,
        )
        merged_duration = cue.end - prev.start
        merged_chars = _visible_len(merged_text)
        if (
            merged_text
            and merged_duration <= _timing_max_duration(profile, language) + TIMING_EPSILON
            and merged_chars <= profile.max_chars_total
            and merged_chars / max(merged_duration, 0.1) <= profile.max_chars_per_second
        ):
            merged[index - 1 : index + 1] = [
                replace(prev, end=cue.end, text=merged_text, end_char=cue.end_char, warnings=[])
            ]
        index -= 1

    return [replace(cue, index=index + 1) for index, cue in enumerate(merged)]


def _annotate_warnings(cues: list[SubtitleCue], profile: SubtitleProfile, language: str) -> None:
    timing_max_duration = _timing_max_duration(profile, language)
    for index, cue in enumerate(cues):
        if cue.duration < profile.min_duration:
            cue.warnings.append("too_short")
        if cue.duration > timing_max_duration + TIMING_EPSILON:
            cue.warnings.append("too_long")
        if _visible_len(cue.text) / max(cue.duration, 0.1) > profile.max_chars_per_second:
            cue.warnings.append("fast_reading")
        if _visible_len(cue.text) > profile.max_chars_total:
            cue.warnings.append("cue_too_long")
        if index and cue.start < cues[index - 1].end:
            cue.warnings.append("overlap")


def _visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _soft_char_limit(profile: SubtitleProfile, language: str) -> int:
    group = language_group(language)
    if group == "cjk":
        return ZH_FORCE_SPLIT_CHARS
    if group == "ko":
        return KO_FORCE_SPLIT_CHARS
    return profile.max_chars_total + max(8, int(profile.max_chars_per_line * 0.6))


def _ideal_chars(profile: SubtitleProfile, language: str) -> float:
    group = language_group(language)
    if group == "cjk":
        return min(profile.max_chars_total, 30)
    if group == "ko":
        return min(profile.max_chars_total, 34)
    return profile.max_chars_total * 0.72


def _target_chars_per_second(profile: SubtitleProfile, language: str) -> float:
    group = language_group(language)
    if group == "ja":
        return min(profile.max_chars_per_second, 8.5)
    if group == "cjk":
        return min(profile.max_chars_per_second, ZH_COMFORT_CPS)
    if group == "ko":
        return min(profile.max_chars_per_second, KO_COMFORT_CPS)
    return min(profile.max_chars_per_second, 18.0)


def _timing_max_duration(profile: SubtitleProfile, language: str) -> float:
    if language_group(language) == "ko" and profile.max_duration < 5.8:
        return min(5.8, profile.max_duration + 1.8)
    return profile.max_duration


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
    elif group == "cjk":
        score += _zh_boundary_score(display_text, char_end, prev_char, next_char)
    elif group == "ko":
        score += _ko_boundary_score(display_text, char_end, prev_char, next_char)
    elif group == "en":
        if _has_space_at_boundary(display_text, char_end):
            score += 34
        if _is_english_word_internal_boundary(display_text, char_end):
            score -= 90
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


def _zh_boundary_score(display_text: str, char_end: int, prev_char: str, next_char: str) -> float:
    if _is_zh_quote_boundary_risk(display_text, char_end):
        return -95
    score = 0.0
    if prev_char in ZH_BAD_EDGE:
        score -= 24
    if next_char in ZH_BAD_EDGE:
        score -= 8
    if _is_cjk_char(prev_char) and _is_cjk_char(next_char):
        score -= 6
    return score


def _ko_boundary_score(display_text: str, char_end: int, prev_char: str, next_char: str) -> float:
    score = 0.0
    if _has_space_at_boundary(display_text, char_end):
        score += 34
    if _is_hangul(prev_char) and _is_hangul(next_char) and not _has_space_at_boundary(display_text, char_end):
        score -= 74
    if prev_char in OPEN_QUOTES or next_char in CLOSE_QUOTES:
        score -= 55
    return score


def _is_zh_quote_boundary_risk(text: str, char_end: int) -> bool:
    prev_char = _previous_visible_char(text, char_end)
    next_char = _next_visible_char(text, char_end)
    if not prev_char or not next_char:
        return False
    if next_char in STRONG_PUNCT or next_char in MID_PUNCT:
        return False
    if prev_char in OPEN_QUOTES or prev_char in STRAIGHT_QUOTES:
        return _is_cjk_char(next_char) or next_char in ZH_QUOTE_FOLLOWERS
    if next_char in CLOSE_QUOTES or next_char in STRAIGHT_QUOTES:
        return _is_cjk_char(prev_char)
    if prev_char in CLOSE_QUOTES or prev_char in STRAIGHT_QUOTES:
        return next_char in ZH_QUOTE_FOLLOWERS
    return False


def _is_high_risk_boundary(display_text: str, char_end: int, language: str) -> bool:
    group = language_group(language)
    prev_char = _previous_visible_char(display_text, char_end)
    next_char = _next_visible_char(display_text, char_end)
    if not prev_char or not next_char:
        return False
    if group == "cjk":
        return _is_zh_quote_boundary_risk(display_text, char_end)
    if _is_clear_boundary(prev_char):
        return False
    if group == "ko":
        if prev_char in OPEN_QUOTES or next_char in CLOSE_QUOTES:
            return True
        return _is_hangul(prev_char) and _is_hangul(next_char) and not _has_space_at_boundary(display_text, char_end)
    if group == "en":
        return _is_english_word_internal_boundary(display_text, char_end)
    if group != "ja":
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


def _is_cjk_char(char: str) -> bool:
    return bool(ZH_CJK_RE.fullmatch(char))


def _is_hangul(char: str) -> bool:
    return bool(KO_HANGUL_RE.fullmatch(char))


def _has_space_at_boundary(text: str, char_end: int) -> bool:
    return (
        (char_end > 0 and text[char_end - 1].isspace())
        or (char_end < len(text) and text[char_end].isspace())
    )


def _is_english_word_internal_boundary(text: str, char_end: int) -> bool:
    if char_end <= 0 or char_end >= len(text) or _has_space_at_boundary(text, char_end):
        return False
    prev_char = text[char_end - 1]
    next_char = text[char_end]
    word_chars = set("'-’")
    prev_is_word = (prev_char.isascii() and prev_char.isalnum()) or prev_char in word_chars
    next_is_word = (next_char.isascii() and next_char.isalnum()) or next_char in word_chars
    return prev_is_word and next_is_word


def _long_cue_chars(language: str) -> int:
    return KO_LONG_CUE_CHARS if language_group(language) == "ko" else ZH_LONG_CUE_CHARS


def _force_split_chars(language: str) -> int:
    return KO_FORCE_SPLIT_CHARS if language_group(language) == "ko" else ZH_FORCE_SPLIT_CHARS


def _comfort_chars_per_second(language: str) -> float:
    return KO_COMFORT_CPS if language_group(language) == "ko" else ZH_COMFORT_CPS


def _has_bad_line_edge(text: str, language: str) -> bool:
    group = language_group(language)
    if group == "ja":
        return text[-1:] in JA_BAD_EDGE
    if group == "cjk":
        return text[-1:] in ZH_BAD_EDGE or text[-1:] in OPEN_QUOTES or text[-1:] in STRAIGHT_QUOTES
    if group == "ko":
        return text[-1:] in OPEN_QUOTES or text[-1:] in STRAIGHT_QUOTES
    if group == "en":
        words = re.findall(r"[A-Za-z']+", text)
        return bool(words and words[-1].lower() in EN_BAD_EDGE)
    return False
