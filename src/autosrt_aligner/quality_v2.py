"""Independent text, timing, and semantic quality gates for V2."""

from __future__ import annotations

import math
import unicodedata
from statistics import mean
from typing import Any

from .models import (
    AlignmentToken,
    AudioChunk,
    ChunkAlignment,
    SourceDocument,
    SubtitleCue,
    SubtitleProfile,
)
from .reconcile import ReconcileReport
from .segmenter_v2 import _exceeds_reading_speed_limit, _is_safe_boundary
from .text import normalize_for_compare, validate_subtitle_continuity


def build_v2_quality_report(
    source: SourceDocument,
    tokens: list[AlignmentToken],
    cues: list[SubtitleCue],
    chunks: list[AudioChunk],
    reconcile: ReconcileReport,
    profile: SubtitleProfile,
    audio_duration: float | None = None,
    alignments: list[ChunkAlignment] | None = None,
) -> dict[str, Any]:
    text_exact = validate_subtitle_continuity(cues, source.display_text)
    text_verified = text_exact and reconcile.source_token_coverage >= 1.0
    text_status = "pass" if text_verified else "fail"
    timing_issues = list(reconcile.issues)
    timing_issues.extend(
        _timing_issues(
            tokens,
            source.display_text,
            chunks,
            audio_duration,
            alignments=alignments,
        )
    )
    timing_status = "pass" if not timing_issues else "review"
    segmentation_issues = _segmentation_issues(source, cues, profile)
    segmentation_status = "pass" if not segmentation_issues else "review"

    publishable = (
        text_status == "pass"
        and timing_status == "pass"
        and segmentation_status == "pass"
    )
    warnings = sorted(set(timing_issues + segmentation_issues))
    if text_status != "pass":
        warnings.append(
            "原文 token 时间证据未完整覆盖"
            if text_exact
            else "字幕文字未能连续覆盖原文"
        )
    warnings = sorted(set(warnings))
    durations = [cue.duration for cue in cues]
    cps_values = [
        _reading_char_count(cue.text) / max(cue.duration, 0.1)
        for cue in cues
    ]
    gaps = [cur.start - prev.end for prev, cur in zip(cues, cues[1:])]
    score = 100 if publishable else 0
    if text_status == "pass" and (timing_status != "pass" or segmentation_status != "pass"):
        score = 50

    return {
        "schema_version": "2",
        "text_integrity_status": "pass" if text_exact else "fail",
        "token_evidence_status": (
            "pass" if reconcile.source_token_coverage >= 1.0 else "fail"
        ),
        "text_status": text_status,
        "timing_status": timing_status,
        "segmentation_status": segmentation_status,
        "publish_status": "pass" if publishable else "blocked",
        "timeline_status": "ok" if timing_status == "pass" else "needs_review",
        "timeline_confidence_score": 100 if timing_status == "pass" else 0,
        "low_confidence_ranges": _low_confidence_ranges(tokens),
        "timing_verification": "structural_evidence_only",
        "audio_duration": round(audio_duration, 3) if audio_duration is not None else None,
        "source_hash": source.source_hash,
        "source_visible_length": source.visible_length,
        "subtitle_count": len(cues),
        "chunk_count": len(chunks),
        "source_token_coverage": reconcile.source_token_coverage,
        "duplicate_token_count": reconcile.duplicate_token_count,
        "token_overlap_count": reconcile.overlap_count,
        "unresolved_overlap_count": reconcile.overlap_count,
        "resolved_overlap_count": reconcile.resolved_overlap_count,
        "uncovered_source_ranges": _uncovered_source_ranges(
            source,
            reconcile.uncovered_source_ranges,
        ),
        "time_reversal_count": reconcile.time_reversal_count,
        "low_confidence_token_count": reconcile.low_confidence_count,
        "missing_confidence_token_count": reconcile.missing_confidence_count,
        "confidence_status": (
            "available" if reconcile.confidence_available else "not_provided"
        ),
        # The segmenter already permits a terminal cue to end on the real
        # audio evidence even when it has no following cue to merge with.
        "too_short_count": sum(
            1
            for cue in cues[:-1]
            if cue.duration < profile.min_duration - 0.001
        ),
        "too_long_count": sum(1 for cue in cues if cue.duration > profile.max_duration + 0.001),
        "max_chars_total": profile.max_chars_total,
        "avg_subtitle_duration": round(mean(durations), 3) if durations else 0.0,
        "max_subtitle_duration": round(max(durations, default=0.0), 3),
        "max_chars_per_second": round(max(cps_values, default=0.0), 3),
        "p95_chars_per_second": round(_percentile(cps_values, 0.95), 3),
        "max_gap_seconds": round(max(gaps, default=0.0), 3),
        "overlap_count": sum(1 for gap in gaps if gap < -0.001),
        "large_gap_count": sum(1 for gap in gaps if gap > 0.8),
        "unaligned_text_ratio": 0.0 if text_exact else _text_ratio(cues, source.display_text),
        "timing_issues": timing_issues,
        "segmentation_issues": segmentation_issues,
        "warnings": warnings,
        # Kept temporarily for old UI/CLI consumers. V2 status fields are the
        # authoritative gate and this value is intentionally binary-ish.
        "quality_score": score,
    }


def _timing_issues(
    tokens: list[AlignmentToken],
    display_text: str,
    chunks: list[AudioChunk],
    audio_duration: float | None,
    *,
    alignments: list[ChunkAlignment] | None = None,
) -> list[str]:
    issues: list[str] = []
    if not tokens:
        return ["没有可验证的时间轴 token"]
    ordered = sorted(tokens, key=lambda token: (token.start_char or 0, token.start))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.start - 0.05:
            issues.append("时间轴按原文顺序发生倒退")
        if current.start < 0 or current.end < current.start:
            issues.append("存在无效 token 时间")
    for token in ordered:
        if not math.isfinite(token.start) or not math.isfinite(token.end):
            issues.append("存在非有限 token 时间")
            continue
        if token.start_char is None or token.end_char is None:
            issues.append("存在未映射 source 的 token")
            break
        if token.end_char > len(display_text):
            issues.append("token 超出原文范围")
            break
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    effective_ranges = {
        alignment.chunk.chunk_id: (
            alignment.effective_audio_start
            if alignment.effective_audio_start is not None
            else alignment.chunk.audio_start,
            alignment.effective_audio_end
            if alignment.effective_audio_end is not None
            else alignment.chunk.audio_end,
        )
        for alignment in (alignments or [])
    }
    for token in ordered:
        chunk = chunk_by_id.get(token.chunk_id or "")
        effective_range = effective_ranges.get(token.chunk_id or "")
        if chunk is not None:
            audio_start, audio_end = effective_range or (
                chunk.audio_start,
                chunk.audio_end,
            )
        else:
            audio_start = audio_end = None
        if (
            audio_start is not None
            and audio_end is not None
            and (token.start < audio_start - 0.25 or token.end > audio_end + 0.25)
        ):
            issues.append("token 时间超出所属音频分块范围")
            break
    if audio_duration is not None and ordered and ordered[-1].end > audio_duration + 0.25:
        issues.append("token 时间超出音频总时长")
    return sorted(set(issues))


def _low_confidence_ranges(
    tokens: list[AlignmentToken],
    threshold: float = 0.20,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (token for token in tokens if token.start_char is not None and token.end_char is not None),
        key=lambda token: (token.start_char or 0, token.end_char or 0),
    )
    ranges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for token in ordered:
        if token.confidence is None or token.confidence >= threshold:
            current = None
            continue
        start = token.start_char or 0
        end = token.end_char or start
        if current is None or start > current["end_char"]:
            current = {
                "start_char": start,
                "end_char": end,
                "start": round(token.start, 3),
                "end": round(token.end, 3),
            }
            ranges.append(current)
        else:
            current["end_char"] = max(current["end_char"], end)
            current["start"] = round(min(current["start"], token.start), 3)
            current["end"] = round(max(current["end"], token.end), 3)
    return ranges


def _segmentation_issues(
    source: SourceDocument,
    cues: list[SubtitleCue],
    profile: SubtitleProfile,
) -> list[str]:
    issues: list[str] = []
    if not cues:
        return ["没有字幕 cue"]
    for index, cue in enumerate(cues):
        if cue.duration <= 0:
            issues.append("存在非正时长 cue")
        if index < len(cues) - 1 and cue.duration < profile.min_duration - 0.001:
            issues.append("存在短于配置下限的 cue")
        if cue.duration > profile.max_duration + 0.001:
            issues.append("存在超过配置上限的 cue")
        if index and cue.start < cues[index - 1].end - 0.001:
            issues.append("字幕 cue 时间重叠")
        if "\n" in cue.text:
            issues.append("字幕 cue 包含内部换行")
        compact_length = len("".join(cue.text.split()))
        if compact_length > profile.max_chars_total:
            issues.append("存在超过字符上限的 cue")
        if (
            cue.duration > 0
            and _exceeds_reading_speed_limit(
                cue.text,
                cue.duration,
                profile,
                source.language,
                is_terminal=index == len(cues) - 1,
            )
        ):
            issues.append("存在超过阅读速度上限的 cue")
    for cue in cues[:-1]:
        if not _is_safe_boundary(source.display_text, cue.end_char, source.language):
            issues.append("存在不自然语义边界")
    return sorted(set(issues))


def _reading_char_count(value: str) -> int:
    """Count spoken characters; whitespace and punctuation have no voice time."""

    return sum(
        1
        for char in value
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _text_ratio(cues: list[SubtitleCue], display_text: str) -> float:
    actual = normalize_for_compare("".join(cue.text for cue in cues))
    expected = normalize_for_compare(display_text)
    if not expected:
        return 1.0
    return min(1.0, abs(len(expected) - len(actual)) / len(expected))


def _uncovered_source_ranges(
    source: SourceDocument,
    ranges: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    return [
        {
            "start_char": start,
            "end_char": end,
            "text": source.display_text[start:end],
        }
        for start, end in ranges
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]
