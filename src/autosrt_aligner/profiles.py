"""Subtitle profile presets and language-sensitive defaults."""

from __future__ import annotations

from .errors import InputError
from .models import SubtitleProfile


PROFILE_LABELS = {
    "youtube_long": "YouTube 长视频",
    "standard": "标准字幕",
    "short": "短字幕",
    "slow_elder": "老年频道慢节奏朗读字幕",
}

CANONICAL_LANGUAGES = ("zh", "ja", "en", "ko")
LANGUAGE_ALIASES = {"zh-TW": "zh"}
SUPPORTED_LANGUAGES = set(CANONICAL_LANGUAGES) | set(LANGUAGE_ALIASES)


def normalize_language(language: str) -> str:
    normalized = LANGUAGE_ALIASES.get(language, language)
    if normalized not in CANONICAL_LANGUAGES:
        raise InputError(f"不支持的语言参数: {language}")
    return normalized


def language_group(language: str) -> str:
    language = normalize_language(language)
    if language == "zh":
        return "cjk"
    if language == "ja":
        return "ja"
    if language == "ko":
        return "ko"
    return "en"


def resolve_profile(
    profile_key: str,
    language: str,
    min_duration: float | None = None,
    max_duration: float | None = None,
    max_chars_per_line: int | None = None,
    max_chars_total: int | None = None,
) -> SubtitleProfile:
    if profile_key not in PROFILE_LABELS:
        raise InputError(f"不支持的字幕风格: {profile_key}")

    group = language_group(language)
    if group == "en":
        base_line = 42
        base_total = 84
        cps = 20.0
    elif group == "ja":
        base_line = 17
        base_total = 34
        cps = 12.0
    elif group == "ko":
        base_line = 20
        base_total = 38
        cps = 10.5
    else:
        base_line = 18
        base_total = 34
        cps = 8.5

    if profile_key == "short":
        default_min, default_max = 1.0, 4.2
        base_line = max(10, int(base_line * 0.82))
        base_total = max(base_line, int(base_total * 0.78))
    elif profile_key == "slow_elder":
        default_min, default_max = 1.5, 7.0
        base_line = max(10, int(base_line * 0.9))
        if group != "en":
            base_total = max(base_line, int(base_total * 0.86))
        cps *= 0.85
    elif profile_key == "standard":
        default_min, default_max = 1.2, 6.0
    else:
        default_min, default_max = 1.2, 6.0 if group == "en" else 6.5

    effective_min = min_duration if min_duration is not None else default_min
    effective_max = max_duration if max_duration is not None else default_max
    if effective_min <= 0:
        raise InputError("每条字幕最短时长必须大于 0")
    if effective_max <= effective_min:
        raise InputError("每条字幕最长时长必须大于最短时长")

    effective_line = max_chars_per_line if max_chars_per_line is not None else base_line
    if effective_line <= 0:
        raise InputError("字幕切分参考字符数必须大于 0")

    effective_total = (
        max_chars_total if max_chars_total is not None else max(effective_line, base_total)
    )
    if effective_total <= 0:
        raise InputError("每条字幕最大字符数必须大于 0")
    complete_min = 0.8 if profile_key == "youtube_long" else default_min
    if effective_min > default_min:
        complete_min = effective_min

    return SubtitleProfile(
        key=profile_key,
        label=PROFILE_LABELS[profile_key],
        min_duration=effective_min,
        max_duration=effective_max,
        ideal_min_duration=max(1.0, effective_min + 0.4),
        ideal_max_duration=min(effective_max, max(effective_min + 0.8, 4.8)),
        max_chars_per_line=effective_line,
        max_chars_total=effective_total,
        max_chars_per_second=cps,
        min_complete_duration=min(effective_min, complete_min),
    )
