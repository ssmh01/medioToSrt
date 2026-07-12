"""Quality report generation for subtitle cues."""

from __future__ import annotations
from statistics import mean
from typing import Any

from .models import SubtitleCue, SubtitleProfile
from .profiles import language_group
from .splitter import TIMING_EPSILON, _is_high_risk_boundary, _timing_max_duration
from .text import unaligned_text_ratio

ZH_LONG_CUE_CHARS = 34
ZH_FAST_CPS = 8.5
ZH_SEVERE_FAST_CPS = 9.5
KO_LONG_CUE_CHARS = 38
KO_FAST_CPS = 9.5
KO_SEVERE_FAST_CPS = 10.5


def build_quality_report(
    cues: list[SubtitleCue],
    display_text: str,
    audio_duration: float | None,
    profile: SubtitleProfile,
    language: str = "zh",
) -> dict[str, Any]:
    durations = [cue.duration for cue in cues]
    cps_values = [_chars(cue.text) / max(cue.duration, 0.1) for cue in cues]
    cue_lengths = [_chars(cue.text) for cue in cues]
    gaps = [cur.start - prev.end for prev, cur in zip(cues, cues[1:])]
    timing_max_duration = _timing_max_duration(profile, language)

    overlap_count = sum(1 for prev, cur in zip(cues, cues[1:]) if cur.start < prev.end)
    too_short_count = sum(1 for cue in cues if cue.duration < profile.min_duration)
    too_long_count = sum(1 for cue in cues if cue.duration > timing_max_duration + TIMING_EPSILON)
    empty_count = sum(1 for cue in cues if not cue.text.strip())
    large_gap_count = sum(1 for gap in gaps if gap > 0.8)
    weak_boundary_count = _weak_boundary_count(cues, display_text, language)
    zh_metrics = _zh_quality_metrics(cues, display_text, profile, language)
    ko_metrics = _ko_quality_metrics(cues, display_text, profile, language)
    ratio = unaligned_text_ratio(cues, display_text)

    warnings: list[str] = []
    if overlap_count:
        warnings.append("存在字幕时间轴重叠")
    if too_short_count:
        warnings.append("存在过短字幕")
    if too_long_count:
        warnings.append("存在过长字幕")
    if ratio:
        warnings.append("字幕文本与原文未完全连续一致")
    if max(cps_values, default=0.0) > profile.max_chars_per_second:
        warnings.append("存在阅读速度过快的字幕")
    if max(cue_lengths, default=0) > profile.max_chars_total:
        warnings.append("存在单条字幕过长")
    if large_gap_count:
        warnings.append("存在疑似异常长间隔")
    if zh_metrics["zh_unsafe_boundary_count"]:
        warnings.append("存在中文不安全切段")
    if zh_metrics["zh_timeline_risk_count"]:
        warnings.append("存在中文时间轴风险")
    if ko_metrics["ko_unsafe_boundary_count"]:
        warnings.append("存在韩语不安全切段")
    if ko_metrics["ko_timeline_risk_count"]:
        warnings.append("存在韩语时间轴风险")

    score = 100
    score -= overlap_count * 18
    score -= too_short_count * 6
    score -= too_long_count * 6
    score -= empty_count * 12
    score -= int(ratio * 60)
    score -= sum(1 for value in cps_values if value > profile.max_chars_per_second) * 4
    score -= sum(1 for length in cue_lengths if length > profile.max_chars_total) * 3
    score -= large_gap_count * 4
    score -= weak_boundary_count * 2
    score -= zh_metrics["zh_fast_cue_count"] * 4
    score -= zh_metrics["zh_large_gap_after_long_cue_count"] * 6
    score -= zh_metrics["zh_unsafe_boundary_count"] * 5
    score -= zh_metrics["zh_timeline_risk_count"] * 6
    score -= ko_metrics["ko_fast_cue_count"] * 4
    score -= ko_metrics["ko_unsafe_boundary_count"] * 5
    score -= ko_metrics["ko_timeline_risk_count"] * 6
    score = max(0, min(100, score))

    return {
        "audio_duration": audio_duration,
        "subtitle_count": len(cues),
        "avg_subtitle_duration": _rounded(mean(durations) if durations else 0.0),
        "min_subtitle_duration": _rounded(min(durations) if durations else 0.0),
        "max_subtitle_duration": _rounded(max(durations) if durations else 0.0),
        "avg_chars_per_second": _rounded(mean(cps_values) if cps_values else 0.0),
        "max_chars_per_second": _rounded(max(cps_values) if cps_values else 0.0),
        "p95_chars_per_second": _rounded(_percentile(cps_values, 0.95)),
        "max_chars_per_cue": max(cue_lengths, default=0),
        "large_gap_count": large_gap_count,
        "max_gap_seconds": _rounded(max(gaps, default=0.0)),
        "weak_boundary_count": weak_boundary_count,
        "overlap_count": overlap_count,
        "too_short_count": too_short_count,
        "too_long_count": too_long_count,
        "empty_subtitle_count": empty_count,
        "unaligned_text_ratio": _rounded(ratio),
        "suspicious_gap_count": large_gap_count,
        **zh_metrics,
        **ko_metrics,
        "quality_score": score,
        "warnings": warnings,
    }


def _chars(text: str) -> int:
    return len("".join(text.split()))


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _has_natural_boundary(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    return compact[-1] in "。！？!?…、，,;；:：」』”’）)]】"


def _weak_boundary_count(cues: list[SubtitleCue], display_text: str, language: str) -> int:
    if _uses_japanese_boundary_rules(display_text, language):
        return sum(1 for cue in cues[:-1] if _is_high_risk_boundary(display_text, cue.end_char, "ja"))
    group = language_group(language)
    if group == "cjk":
        unsafe = sum(1 for cue in cues[:-1] if _is_high_risk_boundary(display_text, cue.end_char, language))
        unnatural_tail = sum(1 for cue in cues if not _has_natural_boundary(cue.text))
        return unsafe + unnatural_tail
    if group == "ko":
        return sum(1 for cue in cues[:-1] if _is_high_risk_boundary(display_text, cue.end_char, language))
    return sum(1 for cue in cues if not _has_natural_boundary(cue.text))


def _uses_japanese_boundary_rules(display_text: str, language: str) -> bool:
    return language == "ja"


def _zh_quality_metrics(
    cues: list[SubtitleCue],
    display_text: str,
    profile: SubtitleProfile,
    language: str,
) -> dict[str, int]:
    if language_group(language) != "cjk":
        return {
            "zh_long_cue_count": 0,
            "zh_fast_cue_count": 0,
            "zh_unsafe_boundary_count": 0,
            "zh_large_gap_after_long_cue_count": 0,
            "zh_timeline_risk_count": 0,
        }

    long_count = 0
    fast_count = 0
    unsafe_count = 0
    large_gap_after_long_count = 0
    timeline_risk_count = 0
    for index, cue in enumerate(cues):
        chars = _chars(cue.text)
        cps = chars / max(cue.duration, 0.1)
        if chars > ZH_LONG_CUE_CHARS:
            long_count += 1
        if cps > ZH_FAST_CPS:
            fast_count += 1
        if index < len(cues) - 1 and _is_high_risk_boundary(display_text, cue.end_char, language):
            unsafe_count += 1
            timeline_risk_count += 1
        if index < len(cues) - 1:
            gap = cues[index + 1].start - cue.end
            long_or_maxed = chars > ZH_LONG_CUE_CHARS or cue.duration >= profile.max_duration - 0.05
            if gap > 0.8 and long_or_maxed:
                large_gap_after_long_count += 1
                timeline_risk_count += 1
            elif gap > 1.5:
                timeline_risk_count += 1
        if cps > ZH_SEVERE_FAST_CPS:
            timeline_risk_count += 1

    return {
        "zh_long_cue_count": long_count,
        "zh_fast_cue_count": fast_count,
        "zh_unsafe_boundary_count": unsafe_count,
        "zh_large_gap_after_long_cue_count": large_gap_after_long_count,
        "zh_timeline_risk_count": timeline_risk_count,
    }


def _ko_quality_metrics(
    cues: list[SubtitleCue],
    display_text: str,
    profile: SubtitleProfile,
    language: str,
) -> dict[str, int]:
    if language_group(language) != "ko":
        return {
            "ko_long_cue_count": 0,
            "ko_fast_cue_count": 0,
            "ko_unsafe_boundary_count": 0,
            "ko_timeline_risk_count": 0,
        }

    long_count = 0
    fast_count = 0
    unsafe_count = 0
    timeline_risk_count = 0
    for index, cue in enumerate(cues):
        chars = _chars(cue.text)
        cps = chars / max(cue.duration, 0.1)
        if chars > KO_LONG_CUE_CHARS:
            long_count += 1
        if cps > KO_FAST_CPS:
            fast_count += 1
        if index < len(cues) - 1 and _is_high_risk_boundary(display_text, cue.end_char, language):
            unsafe_count += 1
            timeline_risk_count += 1
        if index < len(cues) - 1:
            gap = cues[index + 1].start - cue.end
            long_or_maxed = chars > KO_LONG_CUE_CHARS or cue.duration >= profile.max_duration - 0.05
            if (gap > 0.8 and long_or_maxed) or gap > 1.5:
                timeline_risk_count += 1
        if cps > KO_SEVERE_FAST_CPS:
            timeline_risk_count += 1

    return {
        "ko_long_cue_count": long_count,
        "ko_fast_cue_count": fast_count,
        "ko_unsafe_boundary_count": unsafe_count,
        "ko_timeline_risk_count": timeline_risk_count,
    }
