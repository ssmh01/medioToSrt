"""End-to-end alignment, split, validation, and export pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .audio import preprocess_audio
from .engines.base import AlignmentEngine
from .engines.stable_ts import StableTsEngine
from .formats import export_srt, export_vtt
from .models import AlignmentToken, JobResult, SubtitleCue
from .profiles import resolve_profile
from .quality import build_quality_report
from .splitter import VISUAL_GAP_TARGET_SECONDS, split_subtitles
from .text import clean_script_text, map_tokens_to_display


def run_alignment_job(
    audio_path: str | Path,
    script_text: str,
    language: str = "zh",
    subtitle_profile: str = "youtube_long",
    output_dir: str | Path | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    max_chars_per_line: int | None = None,
    generate_vtt: bool = True,
    preserve_punctuation: bool = True,
    engine: AlignmentEngine | None = None,
) -> JobResult:
    logs: list[str] = []
    engine = engine or StableTsEngine()
    profile = resolve_profile(
        subtitle_profile,
        language,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars_per_line=max_chars_per_line,
    )
    cleaned = clean_script_text(script_text, preserve_punctuation=preserve_punctuation)
    logs.append(f"文案字符数: {len(cleaned.display_text)}")

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="autosrt_aligner_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if getattr(engine, "requires_audio_preprocessing", True):
        logs.append("开始音频预处理: 16kHz mono wav")
        audio_info = preprocess_audio(audio_path, work_dir)
        align_audio_path = audio_info.wav_path
        audio_duration = audio_info.duration
    else:
        align_audio_path = Path(audio_path)
        audio_duration = None

    alignment = engine.align(align_audio_path, cleaned, language, logs)
    if alignment.audio_duration is not None:
        audio_duration = alignment.audio_duration
    logs.append(f"对齐 token 数: {len(alignment.tokens)}")

    mapped_tokens = _ensure_mapped_tokens(alignment.tokens, cleaned)
    cues = split_subtitles(cleaned.display_text, mapped_tokens, language, profile)
    cues, timeline_repair = _repair_timeline_if_needed(
        cues,
        cleaned.display_text,
        mapped_tokens,
        profile,
        language,
        audio_duration,
        logs,
    )
    quality_report = build_quality_report(cues, cleaned.display_text, audio_duration, profile, language)
    if timeline_repair is not None:
        quality_report["timeline_repaired"] = True
        quality_report["timeline_repair"] = timeline_repair
        quality_report["timeline_repair_mode"] = timeline_repair.get("mode")
        quality_report["timeline_repair_confidence"] = timeline_repair.get("confidence")
        warning = "检测到对齐时间轴断层，已用可信锚点重建后段时间轴"
        if timeline_repair.get("confidence") == "low":
            warning = "检测到对齐时间轴断层，后段时间轴为低置信估算，可能存在局部漂移"
            quality_report["quality_score"] = min(quality_report.get("quality_score", 100), 86)
        quality_report.setdefault("warnings", []).append(warning)

    srt_path = out_dir / "output.srt"
    srt_path.write_text(export_srt(cues), encoding="utf-8")
    vtt_path = None
    if generate_vtt:
        vtt_path = out_dir / "output.vtt"
        vtt_path.write_text(export_vtt(cues), encoding="utf-8")

    alignment_payload = _build_alignment_payload(
        cleaned_display=cleaned.display_text,
        cleaned_align=cleaned.align_text,
        language=language,
        profile_key=subtitle_profile,
        alignment_raw=alignment.raw,
        tokens=mapped_tokens,
        cues=cues,
        audio_duration=audio_duration,
    )
    alignment_json_path = out_dir / "alignment.json"
    alignment_json_path.write_text(
        json.dumps(alignment_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quality_report_path = out_dir / "quality_report.json"
    quality_report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logs.append("导出完成")

    return JobResult(
        output_dir=out_dir,
        srt_path=srt_path,
        vtt_path=vtt_path,
        alignment_json_path=alignment_json_path,
        quality_report_path=quality_report_path,
        cues=cues,
        quality_report=quality_report,
        alignment_payload=alignment_payload,
        logs=logs,
    )


def _repair_timeline_if_needed(
    cues: list[SubtitleCue],
    display_text: str,
    mapped_tokens: list[AlignmentToken] | None,
    profile: Any,
    language: str,
    audio_duration: float | None,
    logs: list[str],
) -> tuple[list[SubtitleCue], dict[str, Any] | None]:
    break_index = _find_timeline_break(cues, profile)
    if break_index is None:
        return cues, None

    tail_end_time = audio_duration if audio_duration and audio_duration > 0 else max((cue.end for cue in cues), default=0.0)
    if tail_end_time <= 0:
        return cues, None

    repair_start_index = _repair_start_index(cues, break_index, profile)
    keep_count = max(0, repair_start_index)
    kept = cues[:keep_count]
    tail_start_char = cues[repair_start_index].start_char
    tail_start_time = _tail_start_time(cues, repair_start_index, tail_end_time)
    detected_tail_start_char = cues[break_index].start_char
    detected_tail_start_time = _tail_start_time(cues, break_index, tail_end_time)

    tail_text = display_text[tail_start_char:]
    tail_visible_chars = _visible_len(tail_text)
    available_duration = tail_end_time - tail_start_time
    if tail_visible_chars <= 0 or available_duration <= 0.5:
        return cues, None

    tail_tokens, mode, confidence, anchor_count = _repair_tokens_for_tail(
        display_text,
        tail_start_char,
        tail_start_time,
        tail_end_time,
        mapped_tokens,
        profile,
        detected_tail_start_char,
        detected_tail_start_time,
    )
    if not tail_tokens:
        return cues, None

    repaired_tail = split_subtitles(tail_text, tail_tokens, language, profile)
    adjusted_tail = [
        _replace_cue_chars(cue, cue.start_char + tail_start_char, cue.end_char + tail_start_char)
        for cue in repaired_tail
    ]
    repaired = _reindex_cues([*kept, *adjusted_tail])

    repair_info = {
        "start_index": repair_start_index + 1,
        "detected_index": break_index + 1,
        "start_time": round(tail_start_time, 3),
        "end_time": round(tail_end_time, 3),
        "tail_chars": tail_visible_chars,
        "mode": mode,
        "confidence": confidence,
        "anchor_count": anchor_count,
        "reason": _timeline_break_reason(cues, break_index, profile),
    }
    if confidence == "low":
        logs.append(
            "检测到对齐时间轴断层，已从第 "
            f"{repair_start_index + 1} 条字幕开始用低置信估算重建后段时间轴"
        )
    else:
        logs.append(
            "检测到对齐时间轴断层，已从第 "
            f"{repair_start_index + 1} 条字幕开始用可信锚点重建后段时间轴"
        )
    return repaired, repair_info


def _repair_start_index(cues: list[SubtitleCue], break_index: int, profile: Any) -> int:
    start_index = break_index
    lower = max(0, break_index - 3)
    for index in range(break_index - 1, lower - 1, -1):
        cue = cues[index]
        next_cue = cues[index + 1]
        next_gap = next_cue.start - cue.end
        chars = _visible_len(cue.text)
        stretched = cue.duration >= profile.max_duration - 0.05 and chars < profile.max_chars_total * 0.75
        stranded_before_break = next_gap > max(VISUAL_GAP_TARGET_SECONDS * 4.0, 0.8)
        if stretched or stranded_before_break:
            start_index = index
            continue
        break
    return start_index


def _find_timeline_break(cues: list[SubtitleCue], profile: Any) -> int | None:
    if len(cues) < 2:
        return None

    for index, cue in enumerate(cues):
        if _is_severe_timing_anomaly(cue, profile):
            return max(0, index)
        if index:
            gap = cue.start - cues[index - 1].end
            if _is_suspicious_repair_gap(cues[index - 1], cue, gap, profile):
                return max(0, index)
            if gap > _severe_gap_threshold(profile):
                return max(0, index)
            if cue.start < cues[index - 1].end - 0.1:
                return max(0, index)
    return None


def _timeline_break_reason(cues: list[SubtitleCue], index: int, profile: Any) -> str:
    cue = cues[index]
    chars = _visible_len(cue.text)
    cps = chars / max(cue.duration, 0.1)
    if cue.duration > _severe_duration_threshold(profile):
        return f"cue_too_long:{cue.duration:.3f}s"
    if chars > profile.max_chars_total and cue.duration <= 0.5:
        return f"cue_too_fast:{cps:.3f}cps"
    if cps > max(profile.max_chars_per_second * 4.0, 45.0) and chars > profile.max_chars_total:
        return f"cue_cps_spike:{cps:.3f}cps"
    if index:
        gap = cue.start - cues[index - 1].end
        if _is_suspicious_repair_gap(cues[index - 1], cue, gap, profile):
            return f"suspicious_gap:{gap:.3f}s"
        if gap > _severe_gap_threshold(profile):
            return f"large_gap:{gap:.3f}s"
        if cue.start < cues[index - 1].end - 0.1:
            return "overlap"
    return "unknown"


def _is_severe_timing_anomaly(cue: SubtitleCue, profile: Any) -> bool:
    chars = _visible_len(cue.text)
    cps = chars / max(cue.duration, 0.1)
    if cue.duration > _severe_duration_threshold(profile):
        return True
    if chars > profile.max_chars_total and cue.duration <= 0.5:
        return True
    return cps > max(profile.max_chars_per_second * 4.0, 45.0) and chars > profile.max_chars_total


def _severe_duration_threshold(profile: Any) -> float:
    return max(12.0, profile.max_duration * 1.8)


def _severe_gap_threshold(profile: Any) -> float:
    return max(30.0, profile.max_duration * 4.0)


def _is_suspicious_repair_gap(prev: SubtitleCue, cur: SubtitleCue, gap: float, profile: Any) -> bool:
    if gap <= max(2.0, profile.max_duration * 0.45):
        return False
    prev_sparse_stretched = (
        prev.duration >= profile.max_duration - 0.05
        and _visible_len(prev.text) < profile.max_chars_total * 0.65
    )
    cur_sparse_stretched = (
        cur.duration >= profile.max_duration - 0.05
        and _visible_len(cur.text) < profile.max_chars_total * 0.65
    )
    return prev_sparse_stretched or cur_sparse_stretched


def _repair_tokens_for_tail(
    display_text: str,
    tail_start_char: int,
    tail_start_time: float,
    tail_end_time: float,
    mapped_tokens: list[AlignmentToken] | None,
    profile: Any,
    detected_tail_start_char: int | None = None,
    detected_tail_start_time: float | None = None,
) -> tuple[list[AlignmentToken], str, str, int]:
    tail_text = display_text[tail_start_char:]
    if mapped_tokens:
        anchors = _trusted_timeline_anchors(
            display_text,
            mapped_tokens,
            tail_start_char,
            tail_start_time,
            len(display_text),
            tail_end_time,
            profile,
        )
        anchors = _with_convergence_anchor(
            display_text,
            anchors,
            tail_start_char,
            len(display_text),
            tail_end_time,
            detected_tail_start_char,
            detected_tail_start_time,
        )
        trusted_anchor_count = max(0, len(anchors) - 2)
        if trusted_anchor_count:
            relative_anchors = [(char - tail_start_char, time) for char, time in anchors]
            confidence = _anchor_repair_confidence(
                trusted_anchor_count,
                anchors[-2][0] - tail_start_char,
                len(tail_text),
            )
            return (
                _interpolated_tokens_for_text(tail_text, relative_anchors),
                "anchor_interpolated",
                confidence,
                trusted_anchor_count,
            )

    return (
        _synthetic_tokens_for_text(tail_text, tail_start_time, tail_end_time),
        "tail_linear_low_confidence",
        "low",
        0,
    )


def _tail_start_time(cues: list[SubtitleCue], start_index: int, tail_end_time: float) -> float:
    if start_index > 0:
        return min(tail_end_time, cues[start_index - 1].end + VISUAL_GAP_TARGET_SECONDS)
    return min(tail_end_time, cues[start_index].start)


def _with_convergence_anchor(
    display_text: str,
    anchors: list[tuple[int, float]],
    tail_start_char: int,
    tail_end_char: int,
    tail_end_time: float,
    detected_tail_start_char: int | None,
    detected_tail_start_time: float | None,
) -> list[tuple[int, float]]:
    if (
        detected_tail_start_char is None
        or detected_tail_start_time is None
        or detected_tail_start_char <= tail_start_char
        or not anchors
    ):
        return anchors

    tail_start_time = anchors[0][1]
    backtrack_time = detected_tail_start_time - tail_start_time
    if backtrack_time <= 1.0 or tail_end_time <= detected_tail_start_time:
        return anchors

    total_visible = _visible_len(display_text[tail_start_char:tail_end_char])
    backtrack_visible = _visible_len(display_text[tail_start_char:detected_tail_start_char])
    detected_tail_visible = _visible_len(display_text[detected_tail_start_char:tail_end_char])
    if total_visible <= 0 or backtrack_visible <= 0 or detected_tail_visible <= 0:
        return anchors

    convergence_visible = min(
        max(900, backtrack_visible * 70),
        1200,
        max(backtrack_visible + 1, total_visible - 1),
    )
    convergence_char = _char_after_visible_count(
        display_text,
        tail_start_char,
        tail_end_char,
        convergence_visible,
    )
    if convergence_char <= detected_tail_start_char or convergence_char >= tail_end_char:
        return anchors

    visible_from_detected = _visible_len(display_text[detected_tail_start_char:convergence_char])
    old_tail_ratio = visible_from_detected / detected_tail_visible
    convergence_time = detected_tail_start_time + old_tail_ratio * (tail_end_time - detected_tail_start_time)
    if convergence_time <= tail_start_time or convergence_time >= tail_end_time:
        return anchors

    return _normalize_absolute_anchors([*anchors, (convergence_char, convergence_time)], tail_start_char, tail_end_char)


def _char_after_visible_count(text: str, start_char: int, end_char: int, visible_count: int) -> int:
    seen = 0
    for index in range(start_char, end_char):
        if text[index].isspace():
            continue
        seen += 1
        if seen >= visible_count:
            return index + 1
    return end_char


def _normalize_absolute_anchors(
    anchors: list[tuple[int, float]],
    start_char: int,
    end_char: int,
) -> list[tuple[int, float]]:
    cleaned: list[tuple[int, float]] = []
    for char, time in sorted(anchors, key=lambda item: (item[0], item[1])):
        char = min(max(start_char, char), end_char)
        time = max(0.0, time)
        if cleaned and (char <= cleaned[-1][0] or time <= cleaned[-1][1]):
            continue
        cleaned.append((char, time))
    if not cleaned:
        return [(start_char, 0.0), (end_char, 0.1)]
    if cleaned[0][0] != start_char:
        cleaned.insert(0, (start_char, cleaned[0][1]))
    if cleaned[-1][0] != end_char:
        cleaned.append((end_char, cleaned[-1][1] + 0.1))
    return cleaned


def _trusted_timeline_anchors(
    display_text: str,
    mapped_tokens: list[AlignmentToken],
    tail_start_char: int,
    tail_start_time: float,
    tail_end_char: int,
    tail_end_time: float,
    profile: Any,
) -> list[tuple[int, float]]:
    anchors: list[tuple[int, float]] = [(tail_start_char, tail_start_time)]
    last_char = tail_start_char
    last_time = tail_start_time
    tokens = sorted(
        (
            token
            for token in mapped_tokens
            if token.start_char is not None
            and token.end_char is not None
            and token.end_char > tail_start_char
            and token.start_char < tail_end_char
        ),
        key=lambda token: (token.start_char or 0, token.end_char or 0),
    )

    for token in tokens:
        end_char = min(tail_end_char, token.end_char or tail_end_char)
        if end_char <= last_char:
            continue
        token_time = max(token.start, token.end)
        if token_time > tail_end_time + 0.25:
            continue
        if not _is_trusted_anchor_step(display_text, last_char, end_char, last_time, token_time, token, profile):
            continue
        anchors.append((end_char, token_time))
        last_char = end_char
        last_time = token_time

    if anchors[-1][0] < tail_end_char and tail_end_time > anchors[-1][1]:
        anchors.append((tail_end_char, tail_end_time))
    return anchors


def _is_trusted_anchor_step(
    display_text: str,
    last_char: int,
    end_char: int,
    last_time: float,
    token_time: float,
    token: AlignmentToken,
    profile: Any,
) -> bool:
    visible_delta = _visible_len(display_text[last_char:end_char])
    if visible_delta <= 0:
        return False
    time_delta = token_time - last_time
    if time_delta <= 0.02:
        return False

    confidence = token.confidence if token.confidence is not None else 0.0
    if confidence < 0.25:
        return False

    cps = visible_delta / time_delta
    max_cps = max(profile.max_chars_per_second * 1.8, 16.0)
    min_cps = 1.8
    if cps < min_cps or cps > max_cps:
        return False

    # A small number of characters cannot plausibly justify a long timestamp jump.
    if visible_delta <= 4 and time_delta > max(1.2, visible_delta / min_cps):
        return False
    return True


def _anchor_repair_confidence(anchor_count: int, trusted_span_chars: int, tail_chars: int) -> str:
    if anchor_count <= 0:
        return "low"
    if trusted_span_chars >= max(400, tail_chars * 0.25):
        return "high"
    return "medium"


def _interpolated_tokens_for_text(text: str, anchors: list[tuple[int, float]]) -> list[AlignmentToken]:
    normalized_anchors = _normalize_anchors(anchors, len(text))
    tokens: list[AlignmentToken] = []
    for (start_char, start_time), (end_char, end_time) in zip(normalized_anchors, normalized_anchors[1:]):
        visible_positions = [
            (index, char)
            for index, char in enumerate(text[start_char:end_char], start=start_char)
            if not char.isspace()
        ]
        if not visible_positions:
            continue
        duration = max(0.1, end_time - start_time)
        step = duration / len(visible_positions)
        for visible_index, (char_index, char) in enumerate(visible_positions):
            token_start = start_time + visible_index * step
            token_end = start_time + (visible_index + 1) * step
            tokens.append(AlignmentToken(char, token_start, token_end, char_index, char_index + 1))
    return tokens


def _normalize_anchors(anchors: list[tuple[int, float]], text_length: int) -> list[tuple[int, float]]:
    cleaned: list[tuple[int, float]] = []
    for char, time in sorted(anchors, key=lambda item: (item[0], item[1])):
        char = min(max(0, char), text_length)
        time = max(0.0, time)
        if cleaned and (char <= cleaned[-1][0] or time <= cleaned[-1][1]):
            continue
        cleaned.append((char, time))
    if not cleaned:
        return [(0, 0.0), (text_length, 0.1)]
    if cleaned[0][0] != 0:
        cleaned.insert(0, (0, cleaned[0][1]))
    if cleaned[-1][0] != text_length:
        cleaned.append((text_length, cleaned[-1][1] + 0.1))
    return cleaned


def _synthetic_tokens_for_text(text: str, start_time: float, end_time: float) -> list[AlignmentToken]:
    visible_positions = [(index, char) for index, char in enumerate(text) if not char.isspace()]
    if not visible_positions:
        return []

    duration = max(0.1, end_time - start_time)
    step = duration / len(visible_positions)
    tokens: list[AlignmentToken] = []
    for visible_index, (char_index, char) in enumerate(visible_positions):
        token_start = start_time + visible_index * step
        token_end = start_time + (visible_index + 1) * step
        tokens.append(AlignmentToken(char, token_start, token_end, char_index, char_index + 1))
    return tokens


def _replace_cue_chars(cue: SubtitleCue, start_char: int, end_char: int) -> SubtitleCue:
    return SubtitleCue(
        index=cue.index,
        start=cue.start,
        end=cue.end,
        text=cue.text,
        start_char=start_char,
        end_char=end_char,
        warnings=list(cue.warnings),
    )


def _reindex_cues(cues: list[SubtitleCue]) -> list[SubtitleCue]:
    return [
        SubtitleCue(
            index=index,
            start=cue.start,
            end=cue.end,
            text=cue.text,
            start_char=cue.start_char,
            end_char=cue.end_char,
            warnings=list(cue.warnings),
        )
        for index, cue in enumerate(cues, start=1)
    ]


def _visible_len(text: str) -> int:
    return len("".join(text.split()))


def _ensure_mapped_tokens(tokens: list[AlignmentToken], cleaned: Any) -> list[AlignmentToken]:
    if all(token.start_char is not None and token.end_char is not None for token in tokens):
        return tokens
    return map_tokens_to_display(tokens, cleaned)


def _build_alignment_payload(
    cleaned_display: str,
    cleaned_align: str,
    language: str,
    profile_key: str,
    alignment_raw: dict[str, Any],
    tokens: list[AlignmentToken],
    cues: list[SubtitleCue],
    audio_duration: float | None,
) -> dict[str, Any]:
    return {
        "language": language,
        "subtitle_profile": profile_key,
        "audio_duration": audio_duration,
        "display_text_length": len(cleaned_display),
        "align_text_length": len(cleaned_align),
        "engine_result": alignment_raw,
        "tokens": [
            {
                "text": token.text,
                "start": token.start,
                "end": token.end,
                "start_char": token.start_char,
                "end_char": token.end_char,
                "confidence": token.confidence,
            }
            for token in tokens
        ],
        "cues": [
            {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "start_char": cue.start_char,
                "end_char": cue.end_char,
                "warnings": cue.warnings,
            }
            for cue in cues
        ],
    }
