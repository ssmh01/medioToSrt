"""End-to-end alignment, split, validation, and export pipeline."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .audio import preprocess_audio
from .engines.base import AlignmentEngine
from .engines.stable_ts import StableTsEngine
from .formats import export_srt, export_vtt
from .models import AlignmentToken, JobResult, SubtitleCue
from .profiles import language_group, resolve_profile
from .quality import build_quality_report
from .splitter import VISUAL_GAP_TARGET_SECONDS, _is_high_risk_boundary, split_subtitles
from .text import clean_script_text, map_tokens_to_display

CHECKPOINT_MIN_VISIBLE_CHARS = 240
CHECKPOINT_SMALL_TEXT_VISIBLE_CHARS = 1800
CHECKPOINT_LONG_TEXT_STRIDE = 850
CHECKPOINT_MAX_COUNT = 18
CHECKPOINT_TARGET_MIN_CHARS = 70
CHECKPOINT_TARGET_MAX_CHARS = 180
CHECKPOINT_MEDIUM_MEDIAN_DRIFT = 0.75
CHECKPOINT_MEDIUM_P95_DRIFT = 1.25
CHECKPOINT_HIGH_MEDIAN_DRIFT = 1.0
CHECKPOINT_HIGH_P95_DRIFT = 1.5
CHECKPOINT_MEDIUM_SUPPORTING_MEDIAN_DRIFT = 0.25
CHECKPOINT_HIGH_SUPPORTING_MEDIAN_DRIFT = 0.35
CHECKPOINT_REPAIR_MAX_RANGES = 6
MICRO_DRIFT_MAX_RANGES = 16
MICRO_DRIFT_SAMPLE_THRESHOLD = 0.65
MICRO_DRIFT_MEDIAN_THRESHOLD = 0.75
MICRO_DRIFT_P75_THRESHOLD = 1.05
MICRO_DRIFT_HIGH_MEDIAN_THRESHOLD = 1.0
MICRO_DRIFT_HIGH_P75_THRESHOLD = 1.5
MICRO_DRIFT_MIN_SAMPLES = 6
MICRO_DRIFT_MIN_VISIBLE_CHARS = 12
MICRO_DRIFT_REPAIRED_TARGET_SECONDS = 0.6
MICRO_DRIFT_MIN_IMPROVEMENT_RATIO = 0.55


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
    pre_repair_token_diagnostics = _token_timeline_diagnostics(
        mapped_tokens,
        cleaned.display_text,
        audio_duration,
        profile,
    )
    mapped_tokens, timeline_summary = _repair_tokens_with_local_realign(
        mapped_tokens,
        cleaned.display_text,
        engine,
        align_audio_path,
        work_dir,
        language,
        audio_duration,
        profile,
        logs,
    )
    post_repair_token_diagnostics = _token_timeline_diagnostics(
        mapped_tokens,
        cleaned.display_text,
        audio_duration,
        profile,
    )
    mapped_tokens, checkpoint_repaired = _repair_checkpoint_drifts_with_local_realign(
        mapped_tokens,
        cleaned.display_text,
        engine,
        align_audio_path,
        work_dir,
        language,
        audio_duration,
        profile,
        logs,
        timeline_summary,
    )
    if checkpoint_repaired:
        post_repair_token_diagnostics = _token_timeline_diagnostics(
            mapped_tokens,
            cleaned.display_text,
            audio_duration,
            profile,
        )
    cues = split_subtitles(cleaned.display_text, mapped_tokens, language, profile)
    pre_repair_cue_diagnostics = _cue_timeline_diagnostics(cues, cleaned.display_text, profile, language)
    mapped_tokens, cue_risk_repaired = _repair_cue_risks_with_local_realign(
        mapped_tokens,
        cues,
        cleaned.display_text,
        engine,
        align_audio_path,
        work_dir,
        language,
        audio_duration,
        profile,
        logs,
        timeline_summary,
    )
    if cue_risk_repaired:
        post_repair_token_diagnostics = _token_timeline_diagnostics(
            mapped_tokens,
            cleaned.display_text,
            audio_duration,
            profile,
        )
        cues = split_subtitles(cleaned.display_text, mapped_tokens, language, profile)
    mapped_tokens, micro_drift_repaired = _repair_micro_drifts_with_local_realign(
        mapped_tokens,
        cues,
        cleaned.display_text,
        engine,
        align_audio_path,
        work_dir,
        language,
        audio_duration,
        profile,
        logs,
        timeline_summary,
    )
    if micro_drift_repaired:
        post_repair_token_diagnostics = _token_timeline_diagnostics(
            mapped_tokens,
            cleaned.display_text,
            audio_duration,
            profile,
        )
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
        timeline_summary.register_fallback(timeline_repair)
    post_repair_cue_diagnostics = _cue_timeline_diagnostics(cues, cleaned.display_text, profile, language)
    _register_unresolved_cue_timeline_risks(
        timeline_summary,
        _detect_cue_timeline_ranges(cues, cleaned.display_text, audio_duration, profile, language),
        language,
    )
    _apply_timeline_quality(quality_report, timeline_summary)
    if timeline_repair is not None:
        quality_report["timeline_repaired"] = True
        quality_report["timeline_repair"] = timeline_repair
        quality_report["timeline_repair_mode"] = timeline_repair.get("mode")
        quality_report["timeline_repair_confidence"] = timeline_repair.get("confidence")

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
        timeline_summary=timeline_summary.to_payload(),
        token_diagnostics={
            "before_repair": pre_repair_token_diagnostics,
            "after_repair": post_repair_token_diagnostics,
        },
        cue_diagnostics={
            "before_repair": pre_repair_cue_diagnostics,
            "after_repair": post_repair_cue_diagnostics,
        },
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


@dataclass
class TimelineRange:
    start_char: int
    end_char: int
    audio_start: float
    audio_end: float
    reasons: list[str]
    severity: str = "medium"
    index: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.audio_end - self.audio_start)

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "audio_start": round(self.audio_start, 3),
            "audio_end": round(self.audio_end, 3),
            "duration": round(self.duration, 3),
            "reasons": self.reasons,
            "severity": self.severity,
        }


@dataclass
class DriftCheckpoint:
    index: int
    start_char: int
    end_char: int
    audio_start: float
    audio_end: float
    sample_count: int
    median_drift: float
    p95_drift: float
    max_drift: float
    signed_median_drift: float
    severity: str
    local_tokens: list[AlignmentToken] = field(repr=False)

    @property
    def direction(self) -> int:
        return 1 if self.signed_median_drift >= 0 else -1

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "audio_start": round(self.audio_start, 3),
            "audio_end": round(self.audio_end, 3),
            "sample_count": self.sample_count,
            "median_drift": round(self.median_drift, 3),
            "p95_drift": round(self.p95_drift, 3),
            "max_drift": round(self.max_drift, 3),
            "signed_median_drift": round(self.signed_median_drift, 3),
            "severity": self.severity,
        }


@dataclass
class DriftRange:
    timeline_range: TimelineRange
    checkpoint_count: int
    median_drift: float
    p95_drift: float
    max_drift: float
    signed_median_drift: float
    direction: int

    @property
    def severity(self) -> str:
        return self.timeline_range.severity

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.timeline_range.to_payload(),
            "checkpoint_count": self.checkpoint_count,
            "median_drift": round(self.median_drift, 3),
            "p95_drift": round(self.p95_drift, 3),
            "max_drift": round(self.max_drift, 3),
            "signed_median_drift": round(self.signed_median_drift, 3),
        }


@dataclass
class MicroDriftCandidate:
    timeline_range: TimelineRange
    cue_start_index: int
    cue_end_index: int
    reasons: list[str]
    score: float


@dataclass
class MicroDriftRange:
    timeline_range: TimelineRange
    cue_start_index: int
    cue_end_index: int
    sample_count: int
    visible_chars: int
    median_drift: float
    p75_drift: float
    max_drift: float
    signed_median_drift: float
    source: str

    @property
    def severity(self) -> str:
        return self.timeline_range.severity

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.timeline_range.to_payload(),
            "cue_start_index": self.cue_start_index + 1,
            "cue_end_index": self.cue_end_index + 1,
            "sample_count": self.sample_count,
            "visible_chars": self.visible_chars,
            "median_drift": round(self.median_drift, 3),
            "p75_drift": round(self.p75_drift, 3),
            "max_drift": round(self.max_drift, 3),
            "signed_median_drift": round(self.signed_median_drift, 3),
            "source": self.source,
        }


@dataclass
class TimelineRepairSummary:
    suspect_ranges: list[dict[str, Any]] = field(default_factory=list)
    repaired_ranges: list[dict[str, Any]] = field(default_factory=list)
    low_confidence_ranges: list[dict[str, Any]] = field(default_factory=list)
    local_realign_attempts: list[dict[str, Any]] = field(default_factory=list)
    max_anchor_gap_seconds: float = 0.0
    verified_checkpoint_count: int = 0
    checkpoint_drift_count: int = 0
    max_checkpoint_drift_seconds: float = 0.0
    p95_checkpoint_drift_seconds: float = 0.0
    drift_suspect_ranges: list[dict[str, Any]] = field(default_factory=list)
    drift_repaired_ranges: list[dict[str, Any]] = field(default_factory=list)
    unresolved_drift_ranges: list[dict[str, Any]] = field(default_factory=list)
    micro_drift_candidate_count: int = 0
    micro_drift_run_count: int = 0
    micro_drift_repaired_count: int = 0
    micro_drift_unresolved_count: int = 0
    max_micro_drift_seconds: float = 0.0
    micro_drift_ranges: list[dict[str, Any]] = field(default_factory=list)
    micro_drift_repaired_ranges: list[dict[str, Any]] = field(default_factory=list)
    micro_drift_unresolved_ranges: list[dict[str, Any]] = field(default_factory=list)

    def register_fallback(self, repair_info: dict[str, Any]) -> None:
        payload = dict(repair_info)
        payload["mode"] = repair_info.get("mode") or "fallback_estimate"
        payload["confidence"] = "low"
        self.low_confidence_ranges.append(payload)

    @property
    def status(self) -> str:
        if self.low_confidence_ranges or self.unresolved_drift_ranges or self.micro_drift_unresolved_ranges:
            return "needs_review"
        if self.repaired_ranges or self.drift_repaired_ranges or self.micro_drift_repaired_ranges:
            return "repaired"
        return "ok"

    @property
    def confidence_score(self) -> int:
        if self.low_confidence_ranges or self.unresolved_drift_ranges or self.micro_drift_unresolved_ranges:
            return 60
        if self.repaired_ranges or self.drift_repaired_ranges or self.micro_drift_repaired_ranges:
            return 92
        if self.suspect_ranges or self.drift_suspect_ranges or self.micro_drift_ranges:
            return 88
        return 100

    def to_payload(self) -> dict[str, Any]:
        return {
            "timeline_status": self.status,
            "timeline_confidence_score": self.confidence_score,
            "suspect_ranges": self.suspect_ranges,
            "repaired_ranges": self.repaired_ranges,
            "low_confidence_ranges": self.low_confidence_ranges,
            "local_realign_attempts": self.local_realign_attempts,
            "max_anchor_gap_seconds": round(self.max_anchor_gap_seconds, 3),
            "checkpoint_drift_count": self.checkpoint_drift_count,
            "max_checkpoint_drift_seconds": round(self.max_checkpoint_drift_seconds, 3),
            "p95_checkpoint_drift_seconds": round(self.p95_checkpoint_drift_seconds, 3),
            "verified_checkpoint_count": self.verified_checkpoint_count,
            "drift_suspect_ranges": self.drift_suspect_ranges,
            "drift_repaired_ranges": self.drift_repaired_ranges,
            "unresolved_drift_ranges": self.unresolved_drift_ranges,
            "micro_drift_candidate_count": self.micro_drift_candidate_count,
            "micro_drift_run_count": self.micro_drift_run_count,
            "micro_drift_repaired_count": self.micro_drift_repaired_count,
            "micro_drift_unresolved_count": self.micro_drift_unresolved_count,
            "max_micro_drift_seconds": round(self.max_micro_drift_seconds, 3),
            "micro_drift_ranges": self.micro_drift_ranges,
            "micro_drift_repaired_ranges": self.micro_drift_repaired_ranges,
            "micro_drift_unresolved_ranges": self.micro_drift_unresolved_ranges,
        }


def _repair_tokens_with_local_realign(
    mapped_tokens: list[AlignmentToken],
    display_text: str,
    engine: AlignmentEngine,
    align_audio_path: Path,
    work_dir: Path,
    language: str,
    audio_duration: float | None,
    profile: Any,
    logs: list[str],
) -> tuple[list[AlignmentToken], TimelineRepairSummary]:
    summary = TimelineRepairSummary(
        max_anchor_gap_seconds=_max_trusted_anchor_gap(display_text, mapped_tokens, audio_duration, profile)
    )
    if not audio_duration or audio_duration <= 0:
        return mapped_tokens, summary

    suspect_ranges = _detect_suspect_timeline_ranges(display_text, mapped_tokens, audio_duration, profile)
    if not suspect_ranges:
        return mapped_tokens, summary

    summary.suspect_ranges = [range_.to_payload() for range_ in suspect_ranges]
    logs.append(f"检测到 {len(suspect_ranges)} 个疑似时间轴坏段，开始局部重对齐")

    repaired_tokens = list(mapped_tokens)
    local_realign = getattr(engine, "realign_fragment", None)
    for range_ in suspect_ranges:
        if not callable(local_realign):
            summary.low_confidence_ranges.append(
                {
                    **range_.to_payload(),
                    "mode": "local_realign_unavailable",
                    "confidence": "low",
                }
            )
            continue

        result = _try_local_realign_range(
            local_realign,
            align_audio_path,
            display_text,
            range_,
            work_dir,
            language,
            profile,
            logs,
            summary,
        )
        if result:
            repaired_tokens = _replace_tokens_in_char_range(
                repaired_tokens,
                range_.start_char,
                range_.end_char,
                result,
            )
            summary.repaired_ranges.append(
                {
                    **range_.to_payload(),
                    "mode": "local_realign",
                    "confidence": "high",
                    "token_count": len(result),
                }
            )
        else:
            summary.low_confidence_ranges.append(
                {
                    **range_.to_payload(),
                    "mode": "local_realign_failed",
                    "confidence": "low",
                }
            )

    repaired_tokens.sort(key=lambda token: ((token.start_char or 0), token.start, token.end))
    return repaired_tokens, summary


def _repair_cue_risks_with_local_realign(
    mapped_tokens: list[AlignmentToken],
    cues: list[SubtitleCue],
    display_text: str,
    engine: AlignmentEngine,
    align_audio_path: Path,
    work_dir: Path,
    language: str,
    audio_duration: float | None,
    profile: Any,
    logs: list[str],
    summary: TimelineRepairSummary,
) -> tuple[list[AlignmentToken], bool]:
    suspect_ranges = _detect_cue_timeline_ranges(cues, display_text, audio_duration, profile, language)
    if not suspect_ranges:
        return mapped_tokens, False

    known_ranges = {(item.get("start_char"), item.get("end_char"), tuple(item.get("reasons", []))) for item in summary.suspect_ranges}
    for range_ in suspect_ranges:
        key = (range_.start_char, range_.end_char, tuple(range_.reasons))
        if key not in known_ranges:
            summary.suspect_ranges.append(range_.to_payload())
            known_ranges.add(key)
    logs.append(f"检测到 {len(suspect_ranges)} 个 cue 级时间轴风险，开始局部重对齐")

    local_realign = getattr(engine, "realign_fragment", None)
    if not callable(local_realign):
        for range_ in suspect_ranges:
            _append_low_confidence_range(summary, range_, _cue_risk_mode(language, "local_realign_unavailable"))
        return mapped_tokens, False

    repaired_tokens = list(mapped_tokens)
    changed = False
    for range_ in suspect_ranges[:4]:
        result = _try_local_realign_range(
            local_realign,
            align_audio_path,
            display_text,
            range_,
            work_dir,
            language,
            profile,
            logs,
            summary,
        )
        if result:
            repaired_tokens = _replace_tokens_in_char_range(
                repaired_tokens,
                range_.start_char,
                range_.end_char,
                result,
            )
            summary.repaired_ranges.append(
                {
                    **range_.to_payload(),
                    "mode": _cue_risk_mode(language, "cue_local_realign"),
                    "confidence": "high",
                    "token_count": len(result),
                }
            )
            changed = True
        else:
            _append_low_confidence_range(summary, range_, _cue_risk_mode(language, "cue_local_realign_failed"))

    repaired_tokens.sort(key=lambda token: ((token.start_char or 0), token.start, token.end))
    return repaired_tokens, changed


def _repair_zh_cue_risks_with_local_realign(
    mapped_tokens: list[AlignmentToken],
    cues: list[SubtitleCue],
    display_text: str,
    engine: AlignmentEngine,
    align_audio_path: Path,
    work_dir: Path,
    language: str,
    audio_duration: float | None,
    profile: Any,
    logs: list[str],
    summary: TimelineRepairSummary,
) -> tuple[list[AlignmentToken], bool]:
    return _repair_cue_risks_with_local_realign(
        mapped_tokens,
        cues,
        display_text,
        engine,
        align_audio_path,
        work_dir,
        language,
        audio_duration,
        profile,
        logs,
        summary,
    )


def _repair_checkpoint_drifts_with_local_realign(
    mapped_tokens: list[AlignmentToken],
    display_text: str,
    engine: AlignmentEngine,
    align_audio_path: Path,
    work_dir: Path,
    language: str,
    audio_duration: float | None,
    profile: Any,
    logs: list[str],
    summary: TimelineRepairSummary,
) -> tuple[list[AlignmentToken], bool]:
    if not _uses_dense_text_checkpoint_detection(language) or not audio_duration or audio_duration <= 0:
        return mapped_tokens, False

    if _visible_len(display_text) < CHECKPOINT_MIN_VISIBLE_CHARS:
        return mapped_tokens, False

    local_realign = getattr(engine, "realign_fragment", None)
    if not callable(local_realign):
        return mapped_tokens, False

    checkpoint_ranges = _build_checkpoint_ranges(display_text, mapped_tokens, audio_duration)
    if not checkpoint_ranges:
        return mapped_tokens, False

    checkpoints = _verify_checkpoint_ranges(
        local_realign,
        align_audio_path,
        display_text,
        mapped_tokens,
        checkpoint_ranges,
        work_dir,
        language,
        profile,
        logs,
        summary,
    )
    if not checkpoints:
        return mapped_tokens, False

    _update_checkpoint_drift_stats(summary, checkpoints)
    drift_ranges = _drift_ranges_from_checkpoints(display_text, mapped_tokens, checkpoints, audio_duration)
    if not drift_ranges:
        return mapped_tokens, False

    _register_drift_suspects(summary, drift_ranges)
    logs.append(f"检测到 {len(drift_ranges)} 个局部时间轴漂移段，开始强制局部重对齐")

    repaired_tokens = list(mapped_tokens)
    changed = False
    for drift_range in _prioritize_drift_ranges(drift_ranges)[:CHECKPOINT_REPAIR_MAX_RANGES]:
        result = _try_local_realign_range(
            local_realign,
            align_audio_path,
            display_text,
            drift_range.timeline_range,
            work_dir,
            language,
            profile,
            logs,
            summary,
            paddings=(10.0, 18.0),
        )
        if result:
            repaired_tokens = _replace_tokens_in_char_range(
                repaired_tokens,
                drift_range.timeline_range.start_char,
                drift_range.timeline_range.end_char,
                result,
            )
            payload = {
                **drift_range.to_payload(),
                "mode": "checkpoint_drift_local_realign",
                "confidence": "high",
                "token_count": len(result),
            }
            summary.drift_repaired_ranges.append(payload)
            summary.repaired_ranges.append(payload)
            changed = True
            continue

        if drift_range.severity == "high":
            _append_unresolved_drift_range(summary, drift_range, "checkpoint_drift_local_realign_failed")
            _append_low_confidence_range(
                summary,
                drift_range.timeline_range,
                "checkpoint_drift_local_realign_failed",
            )

    if not changed:
        return mapped_tokens, False

    repaired_tokens.sort(key=lambda token: ((token.start_char or 0), token.start, token.end))
    post_checkpoints = [
        _recompute_checkpoint_against_tokens(checkpoint, repaired_tokens, profile)
        for checkpoint in checkpoints
    ]
    _update_checkpoint_drift_stats(summary, post_checkpoints)
    for drift_range in _drift_ranges_from_checkpoints(display_text, repaired_tokens, post_checkpoints, audio_duration):
        if drift_range.severity == "high":
            drift_range.timeline_range.reasons = sorted(
                set(drift_range.timeline_range.reasons) | {"post_repair_drift"}
            )
            _append_unresolved_drift_range(summary, drift_range, "post_repair_drift")
            _append_low_confidence_range(summary, drift_range.timeline_range, "post_repair_drift")

    return repaired_tokens, True


def _repair_micro_drifts_with_local_realign(
    mapped_tokens: list[AlignmentToken],
    cues: list[SubtitleCue],
    display_text: str,
    engine: AlignmentEngine,
    align_audio_path: Path,
    work_dir: Path,
    language: str,
    audio_duration: float | None,
    profile: Any,
    logs: list[str],
    summary: TimelineRepairSummary,
) -> tuple[list[AlignmentToken], bool]:
    if not _uses_dense_text_checkpoint_detection(language) or not audio_duration or audio_duration <= 0:
        return mapped_tokens, False

    local_realign = getattr(engine, "realign_fragment", None)
    if not callable(local_realign):
        return mapped_tokens, False

    candidates = _detect_micro_drift_candidates(cues, display_text, audio_duration, profile, language)
    summary.micro_drift_candidate_count = len(candidates)
    if not candidates:
        return mapped_tokens, False

    logs.append(f"检测到 {len(candidates)} 个 micro drift 候选短段，开始句级复核")
    repaired_tokens = list(mapped_tokens)
    changed = False
    unresolved: list[MicroDriftRange] = []

    for index, candidate in enumerate(candidates[:MICRO_DRIFT_MAX_RANGES], start=1):
        candidate.timeline_range.index = index
        local_tokens = _try_local_realign_range(
            local_realign,
            align_audio_path,
            display_text,
            candidate.timeline_range,
            work_dir,
            language,
            profile,
            logs,
            summary,
            paddings=(8.0, 14.0),
        )
        if not local_tokens:
            continue

        micro_range = _micro_drift_range_from_local_tokens(
            candidate,
            local_tokens,
            repaired_tokens,
            display_text,
            audio_duration,
            profile,
        )
        if not micro_range:
            continue

        _register_micro_drift_range(summary, micro_range)
        replacement = _tokens_in_char_range(
            local_tokens,
            micro_range.timeline_range.start_char,
            micro_range.timeline_range.end_char,
        )
        validation_error = _validate_micro_replacement_tokens(
            replacement,
            display_text,
            micro_range.timeline_range.start_char,
            micro_range.timeline_range.end_char,
            profile,
        )
        compatibility_error = _validate_replacement_timeline_compatible(
            repaired_tokens,
            replacement,
            micro_range.timeline_range.start_char,
            micro_range.timeline_range.end_char,
        )
        if (
            validation_error is not None
            or compatibility_error is not None
            or not _micro_repair_improves(micro_range.median_drift, 0.0)
        ):
            if micro_range.severity == "high":
                unresolved.append(micro_range)
                _append_unresolved_micro_drift_range(
                    summary,
                    micro_range,
                    validation_error or compatibility_error or "micro_drift_no_significant_improvement",
                )
            continue

        repaired_tokens = _replace_tokens_in_char_range(
            repaired_tokens,
            micro_range.timeline_range.start_char,
            micro_range.timeline_range.end_char,
            replacement,
        )
        payload = {
            **micro_range.to_payload(),
            "mode": "micro_drift_local_realign",
            "confidence": "high",
            "token_count": len(replacement),
        }
        summary.micro_drift_repaired_ranges.append(payload)
        summary.micro_drift_repaired_count = len(summary.micro_drift_repaired_ranges)
        summary.repaired_ranges.append(payload)
        changed = True

    if unresolved:
        for micro_range in unresolved:
            _append_low_confidence_range(summary, micro_range.timeline_range, "micro_drift_unresolved")

    if not changed:
        return mapped_tokens, False

    repaired_tokens.sort(key=lambda token: ((token.start_char or 0), token.start, token.end))
    return repaired_tokens, True


def _detect_micro_drift_candidates(
    cues: list[SubtitleCue],
    display_text: str,
    audio_duration: float | None,
    profile: Any,
    language: str,
) -> list[MicroDriftCandidate]:
    if not _uses_dense_text_checkpoint_detection(language) or len(cues) < 2:
        return []

    raw: list[MicroDriftCandidate] = []
    max_audio = audio_duration or max((cue.end for cue in cues), default=0.0)
    for index, cue in enumerate(cues):
        reasons, severity, score = _micro_candidate_reasons(cues, index, profile)
        if not reasons:
            continue
        start_index, end_index = _micro_candidate_cue_window(cues, index, profile)
        start_char = cues[start_index].start_char
        end_char = cues[end_index].end_char
        expanded_start = _previous_sentence_boundary(display_text, start_char, 90)
        expanded_end = _next_sentence_boundary(display_text, end_char, 120)
        expanded_start, expanded_end = _trim_char_range(display_text, expanded_start, expanded_end)
        if expanded_end <= expanded_start:
            continue
        raw.append(
            MicroDriftCandidate(
                timeline_range=TimelineRange(
                    start_char=expanded_start,
                    end_char=expanded_end,
                    audio_start=max(0.0, cues[start_index].start),
                    audio_end=min(max_audio, max(cues[end_index].end, cues[start_index].start + 0.5)),
                    reasons=sorted(set(reasons) | {"micro_drift_probe"}),
                    severity=severity,
                ),
                cue_start_index=start_index,
                cue_end_index=end_index,
                reasons=sorted(set(reasons)),
                score=score,
            )
        )

    merged = _merge_micro_drift_candidates(raw)
    return sorted(
        merged,
        key=lambda item: (
            0 if item.timeline_range.severity == "high" else 1,
            -item.score,
            item.timeline_range.start_char,
        ),
    )[:MICRO_DRIFT_MAX_RANGES]


def _micro_candidate_reasons(
    cues: list[SubtitleCue],
    index: int,
    profile: Any,
) -> tuple[list[str], str, float]:
    cue = cues[index]
    prev_cue = cues[index - 1] if index else None
    next_cue = cues[index + 1] if index + 1 < len(cues) else None
    chars = _visible_len(cue.text)
    cps = chars / max(cue.duration, 0.1)
    near_max = cue.duration >= profile.max_duration - 0.05
    prev_near_max = bool(prev_cue and prev_cue.duration >= profile.max_duration - 0.05)
    next_near_max = bool(next_cue and next_cue.duration >= profile.max_duration - 0.05)
    next_gap = next_cue.start - cue.end if next_cue else 0.0
    prev_gap = cue.start - prev_cue.end if prev_cue else 0.0

    reasons: list[str] = []
    score = 0.0
    severity = "medium"
    sparse_threshold = max(18, int(profile.max_chars_total * 0.65))
    short_duration = max(profile.min_duration + 1.4, min(3.4, profile.max_duration * 0.55))

    if near_max and chars <= sparse_threshold:
        reasons.append("micro_sparse_maxed_cue")
        score += 4.0 + (sparse_threshold - chars) / max(1, sparse_threshold)
        severity = "high"
    if near_max and (prev_near_max or next_near_max):
        reasons.append("micro_consecutive_maxed_cues")
        score += 3.0
        severity = "high"
    if near_max and next_cue and next_cue.duration <= short_duration and _visible_len(next_cue.text) >= 12:
        reasons.append("micro_maxed_then_short_cue")
        score += 2.4
    if prev_near_max and cue.duration <= short_duration and chars >= 12:
        reasons.append("micro_short_after_maxed_cue")
        score += 2.2
    if near_max and next_gap > 0.35:
        reasons.append("micro_gap_after_maxed_cue")
        score += min(2.0, next_gap)
    if prev_near_max and prev_gap > 0.35:
        reasons.append("micro_gap_before_after_maxed_cue")
        score += min(1.6, prev_gap)
    if cue.duration >= profile.max_duration - 0.25 and chars >= 28 and cps < 5.0:
        reasons.append("micro_low_cps_long_cue")
        score += 1.2

    return sorted(set(reasons)), severity, score


def _micro_candidate_cue_window(
    cues: list[SubtitleCue],
    index: int,
    profile: Any,
) -> tuple[int, int]:
    start_index = max(0, index - 1)
    end_index = min(len(cues) - 1, index + 1)
    cue = cues[index]
    next_cue = cues[index + 1] if index + 1 < len(cues) else None
    prev_cue = cues[index - 1] if index else None
    short_duration = max(profile.min_duration + 1.4, min(3.4, profile.max_duration * 0.55))
    if next_cue and cue.duration >= profile.max_duration - 0.05 and next_cue.duration <= short_duration:
        end_index = min(len(cues) - 1, index + 2)
    if prev_cue and prev_cue.duration >= profile.max_duration - 0.05 and cue.duration <= short_duration:
        start_index = max(0, index - 2)
    return start_index, end_index


def _merge_micro_drift_candidates(candidates: list[MicroDriftCandidate]) -> list[MicroDriftCandidate]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: (item.timeline_range.start_char, -item.score))
    merged: list[MicroDriftCandidate] = []
    for item in ordered:
        if not merged or item.timeline_range.start_char > merged[-1].timeline_range.end_char + 40:
            merged.append(item)
            continue
        prev = merged[-1]
        severity = "high" if "high" in {prev.timeline_range.severity, item.timeline_range.severity} else "medium"
        merged[-1] = MicroDriftCandidate(
            timeline_range=TimelineRange(
                start_char=min(prev.timeline_range.start_char, item.timeline_range.start_char),
                end_char=max(prev.timeline_range.end_char, item.timeline_range.end_char),
                audio_start=min(prev.timeline_range.audio_start, item.timeline_range.audio_start),
                audio_end=max(prev.timeline_range.audio_end, item.timeline_range.audio_end),
                reasons=sorted(set(prev.timeline_range.reasons) | set(item.timeline_range.reasons)),
                severity=severity,
            ),
            cue_start_index=min(prev.cue_start_index, item.cue_start_index),
            cue_end_index=max(prev.cue_end_index, item.cue_end_index),
            reasons=sorted(set(prev.reasons) | set(item.reasons)),
            score=max(prev.score, item.score),
        )
    return merged


def _micro_drift_range_from_local_tokens(
    candidate: MicroDriftCandidate,
    local_tokens: list[AlignmentToken],
    mapped_tokens: list[AlignmentToken],
    display_text: str,
    audio_duration: float,
    profile: Any,
) -> MicroDriftRange | None:
    samples = [
        sample
        for sample in _drift_samples_against_tokens(local_tokens, mapped_tokens)
        if sample[1] > candidate.timeline_range.start_char and sample[0] < candidate.timeline_range.end_char
    ]
    run = _strongest_micro_drift_run(samples, display_text)
    if not run:
        return None

    start_char, end_char, sample_count, visible_chars, median_drift, p75_drift, max_drift, signed_median = run
    expanded_start = max(candidate.timeline_range.start_char, _previous_sentence_boundary(display_text, start_char, 90))
    expanded_end = min(candidate.timeline_range.end_char, _next_sentence_boundary(display_text, end_char, 120))
    expanded_start, expanded_end = _trim_char_range(display_text, expanded_start, expanded_end)
    if expanded_end <= expanded_start:
        return None
    audio_start, audio_end = _audio_window_for_char_range(mapped_tokens, expanded_start, expanded_end, audio_duration)
    severity = "high" if (
        median_drift >= MICRO_DRIFT_HIGH_MEDIAN_THRESHOLD
        or p75_drift >= MICRO_DRIFT_HIGH_P75_THRESHOLD
        or max_drift >= 1.5
    ) else "medium"
    return MicroDriftRange(
        timeline_range=TimelineRange(
            start_char=expanded_start,
            end_char=expanded_end,
            audio_start=audio_start,
            audio_end=audio_end,
            reasons=sorted(set(candidate.timeline_range.reasons) | {"micro_drift", "silent_timeline_drift"}),
            severity=severity,
            index=0,
        ),
        cue_start_index=candidate.cue_start_index,
        cue_end_index=candidate.cue_end_index,
        sample_count=sample_count,
        visible_chars=visible_chars,
        median_drift=median_drift,
        p75_drift=p75_drift,
        max_drift=max_drift,
        signed_median_drift=signed_median,
        source="cue_shape",
    )


def _strongest_micro_drift_run(
    samples: list[tuple[int, int, float, float]],
    display_text: str,
) -> tuple[int, int, int, int, float, float, float, float] | None:
    ordered = sorted(samples, key=lambda item: (item[0], item[1]))
    runs: list[list[tuple[int, int, float, float]]] = []
    current: list[tuple[int, int, float, float]] = []
    current_sign = 0
    last_end = -1
    for sample in ordered:
        start_char, end_char, signed, abs_value = sample
        if abs_value < MICRO_DRIFT_SAMPLE_THRESHOLD:
            if current:
                runs.append(current)
                current = []
                current_sign = 0
            continue
        sign = 1 if signed >= 0 else -1
        continuous = bool(current) and sign == current_sign and start_char <= last_end + 4
        if current and not continuous:
            runs.append(current)
            current = []
        if not current:
            current_sign = sign
        current.append(sample)
        last_end = max(last_end, end_char)
    if current:
        runs.append(current)

    best: tuple[int, int, int, int, float, float, float, float] | None = None
    best_score = 0.0
    for run in runs:
        if len(run) < MICRO_DRIFT_MIN_SAMPLES:
            continue
        start_char = min(item[0] for item in run)
        end_char = max(item[1] for item in run)
        visible_chars = _visible_len(display_text[start_char:end_char])
        if visible_chars < MICRO_DRIFT_MIN_VISIBLE_CHARS:
            continue
        abs_values = [item[3] for item in run]
        signed_values = [item[2] for item in run]
        median_drift = median(abs_values)
        p75_drift = _percentile_float(abs_values, 0.75)
        max_drift = max(abs_values)
        if median_drift < MICRO_DRIFT_MEDIAN_THRESHOLD and p75_drift < MICRO_DRIFT_P75_THRESHOLD:
            continue
        signed_median = median(signed_values)
        score = median_drift * 2.0 + p75_drift + min(2.0, visible_chars / 24.0)
        if score > best_score:
            best_score = score
            best = (start_char, end_char, len(run), visible_chars, median_drift, p75_drift, max_drift, signed_median)
    return best


def _drift_samples_against_tokens(
    local_tokens: list[AlignmentToken],
    mapped_tokens: list[AlignmentToken],
) -> list[tuple[int, int, float, float]]:
    mapped = sorted(
        [
            token
            for token in mapped_tokens
            if token.start_char is not None and token.end_char is not None
        ],
        key=lambda token: (token.start_char or 0, token.end_char or 0, token.start),
    )
    samples: list[tuple[int, int, float, float]] = []
    for token in local_tokens:
        if token.start_char is None or token.end_char is None:
            continue
        sample_char = max(token.start_char, min(token.end_char - 1, (token.start_char + token.end_char - 1) // 2))
        global_time = _token_time_at_char(mapped, sample_char)
        if global_time is None:
            continue
        local_time = (token.start + token.end) / 2
        signed = local_time - global_time
        samples.append((token.start_char, token.end_char, signed, abs(signed)))
    return samples


def _tokens_in_char_range(
    tokens: list[AlignmentToken],
    start_char: int,
    end_char: int,
) -> list[AlignmentToken]:
    return sorted(
        [
            token
            for token in tokens
            if token.start_char is not None
            and token.end_char is not None
            and token.end_char > start_char
            and token.start_char < end_char
        ],
        key=lambda token: (token.start_char or 0, token.start, token.end),
    )


def _validate_micro_replacement_tokens(
    tokens: list[AlignmentToken],
    display_text: str,
    fragment_start: int,
    fragment_end: int,
    profile: Any,
) -> str | None:
    if not tokens:
        return "no_tokens"
    tokens = sorted(tokens, key=lambda token: (token.start_char or 0, token.start))
    if (tokens[0].start_char or 0) > fragment_start + 6:
        return "missing_fragment_start"
    if (tokens[-1].end_char or 0) < fragment_end - 6:
        return "missing_fragment_end"
    last_time = tokens[0].start
    low_conf_chars = 0
    total_chars = max(1, _visible_len(display_text[fragment_start:fragment_end]))
    for token in tokens:
        if token.start < last_time - 0.12:
            return "time_reversal"
        last_time = max(last_time, token.end)
        if token.confidence is not None and token.confidence < 0.08:
            low_conf_chars += _visible_len(display_text[token.start_char or 0 : token.end_char or 0])
    duration = max(0.1, tokens[-1].end - tokens[0].start)
    cps = total_chars / duration
    if cps < 1.2:
        return f"too_slow:{cps:.3f}"
    if cps > max(profile.max_chars_per_second * 1.9, 18.0):
        return f"too_fast:{cps:.3f}"
    if low_conf_chars / total_chars > 0.55:
        return "low_confidence_ratio"
    return None


def _validate_replacement_timeline_compatible(
    existing_tokens: list[AlignmentToken],
    replacement: list[AlignmentToken],
    start_char: int,
    end_char: int,
) -> str | None:
    if not replacement:
        return "no_tokens"
    replacement_start = min(token.start for token in replacement)
    replacement_end = max(token.end for token in replacement)
    prev_token = max(
        (
            token
            for token in existing_tokens
            if token.end_char is not None and token.end_char <= start_char
        ),
        key=lambda token: token.end_char or 0,
        default=None,
    )
    next_token = min(
        (
            token
            for token in existing_tokens
            if token.start_char is not None and token.start_char >= end_char
        ),
        key=lambda token: token.start_char or 0,
        default=None,
    )
    if prev_token and replacement_start < prev_token.end - 0.12:
        return "overlap_previous"
    if next_token and replacement_end > next_token.start + 0.12:
        return "overlap_next"
    return None


def _micro_repair_improves(before_median: float, after_median: float) -> bool:
    if after_median <= MICRO_DRIFT_REPAIRED_TARGET_SECONDS:
        return True
    if before_median <= 0:
        return False
    return (before_median - after_median) / before_median >= MICRO_DRIFT_MIN_IMPROVEMENT_RATIO


def _register_micro_drift_range(summary: TimelineRepairSummary, micro_range: MicroDriftRange) -> None:
    payload = micro_range.to_payload()
    existing = {
        (item.get("start_char"), item.get("end_char"), tuple(item.get("reasons", [])))
        for item in summary.micro_drift_ranges
    }
    key = (payload.get("start_char"), payload.get("end_char"), tuple(payload.get("reasons", [])))
    if key not in existing:
        summary.micro_drift_ranges.append(payload)
    summary.micro_drift_run_count = len(summary.micro_drift_ranges)
    summary.max_micro_drift_seconds = max(summary.max_micro_drift_seconds, micro_range.max_drift)


def _append_unresolved_micro_drift_range(
    summary: TimelineRepairSummary,
    micro_range: MicroDriftRange,
    mode: str,
) -> None:
    payload = {
        **micro_range.to_payload(),
        "mode": mode,
        "confidence": "low",
    }
    existing = {
        (item.get("start_char"), item.get("end_char"), item.get("mode"))
        for item in summary.micro_drift_unresolved_ranges
    }
    key = (payload.get("start_char"), payload.get("end_char"), payload.get("mode"))
    if key not in existing:
        summary.micro_drift_unresolved_ranges.append(payload)
    summary.micro_drift_unresolved_count = len(summary.micro_drift_unresolved_ranges)


def _uses_dense_text_checkpoint_detection(language: str) -> bool:
    return language_group(language) in {"cjk", "ko"}


def _uses_zh_checkpoint_detection(language: str) -> bool:
    return _uses_dense_text_checkpoint_detection(language)


def _build_checkpoint_ranges(
    display_text: str,
    mapped_tokens: list[AlignmentToken],
    audio_duration: float,
) -> list[TimelineRange]:
    visible_total = _visible_len(display_text)
    if visible_total < CHECKPOINT_MIN_VISIBLE_CHARS:
        return []

    if visible_total < CHECKPOINT_SMALL_TEXT_VISIBLE_CHARS:
        target_visible_positions = [max(1, visible_total // 2)]
    else:
        target_visible_positions = list(
            range(
                CHECKPOINT_LONG_TEXT_STRIDE,
                max(CHECKPOINT_LONG_TEXT_STRIDE + 1, visible_total - CHECKPOINT_LONG_TEXT_STRIDE // 2),
                CHECKPOINT_LONG_TEXT_STRIDE,
            )
        )

    if len(target_visible_positions) > CHECKPOINT_MAX_COUNT:
        step = max(1, len(target_visible_positions) / CHECKPOINT_MAX_COUNT)
        target_visible_positions = [
            target_visible_positions[min(len(target_visible_positions) - 1, round(index * step))]
            for index in range(CHECKPOINT_MAX_COUNT)
        ]

    ranges: list[TimelineRange] = []
    seen_ranges: set[tuple[int, int]] = set()
    for index, visible_pos in enumerate(target_visible_positions, start=1):
        target_char = _char_after_visible_count(display_text, 0, len(display_text), visible_pos)
        start_char, end_char = _checkpoint_fragment_range(display_text, target_char)
        if end_char <= start_char:
            continue
        key = (start_char, end_char)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        audio_start, audio_end = _audio_window_for_char_range(mapped_tokens, start_char, end_char, audio_duration)
        if audio_end <= audio_start:
            continue
        ranges.append(
            TimelineRange(
                start_char=start_char,
                end_char=end_char,
                audio_start=audio_start,
                audio_end=audio_end,
                reasons=["checkpoint_drift_probe"],
                severity="medium",
                index=index,
            )
        )
    return ranges


def _checkpoint_fragment_range(display_text: str, target_char: int) -> tuple[int, int]:
    target_char = min(max(0, target_char), len(display_text))
    start = _previous_sentence_boundary(display_text, target_char, 110)
    end = _next_sentence_boundary(display_text, target_char, 150)

    for _ in range(3):
        visible = _visible_len(display_text[start:end])
        if visible >= CHECKPOINT_TARGET_MIN_CHARS or (start == 0 and end == len(display_text)):
            break
        start = _previous_sentence_boundary(display_text, start, 140) if start > 0 else start
        end = _next_sentence_boundary(display_text, end, 180) if end < len(display_text) else end

    if _visible_len(display_text[start:end]) > CHECKPOINT_TARGET_MAX_CHARS + 80:
        soft_start = _char_after_visible_count(display_text, start, target_char, 35)
        soft_end = _char_after_visible_count(display_text, target_char, end, CHECKPOINT_TARGET_MAX_CHARS - 35)
        if soft_end > soft_start and _visible_len(display_text[soft_start:soft_end]) >= CHECKPOINT_TARGET_MIN_CHARS:
            start, end = soft_start, soft_end

    return _trim_char_range(display_text, start, end)


def _verify_checkpoint_ranges(
    local_realign: Any,
    align_audio_path: Path,
    display_text: str,
    mapped_tokens: list[AlignmentToken],
    ranges: list[TimelineRange],
    work_dir: Path,
    language: str,
    profile: Any,
    logs: list[str],
    summary: TimelineRepairSummary,
) -> list[DriftCheckpoint]:
    checkpoints: list[DriftCheckpoint] = []
    for range_ in ranges:
        local_tokens = _try_local_realign_range(
            local_realign,
            align_audio_path,
            display_text,
            range_,
            work_dir,
            language,
            profile,
            logs,
            summary,
            paddings=(10.0,),
        )
        if not local_tokens:
            continue
        checkpoint = _checkpoint_from_local_tokens(range_, local_tokens, mapped_tokens, profile)
        if checkpoint:
            checkpoints.append(checkpoint)
    return checkpoints


def _checkpoint_from_local_tokens(
    range_: TimelineRange,
    local_tokens: list[AlignmentToken],
    mapped_tokens: list[AlignmentToken],
    profile: Any,
) -> DriftCheckpoint | None:
    drifts = _drifts_against_tokens(local_tokens, mapped_tokens)
    if len(drifts) < 4:
        return None
    signed_values = [signed for signed, _abs_value in drifts]
    abs_values = [abs_value for _signed, abs_value in drifts]
    median_drift = median(abs_values)
    p95_drift = _percentile_float(abs_values, 0.95)
    max_drift = max(abs_values)
    signed_median = median(signed_values)
    severity = _checkpoint_drift_severity(median_drift, p95_drift, max_drift)
    return DriftCheckpoint(
        index=range_.index,
        start_char=range_.start_char,
        end_char=range_.end_char,
        audio_start=range_.audio_start,
        audio_end=range_.audio_end,
        sample_count=len(drifts),
        median_drift=median_drift,
        p95_drift=p95_drift,
        max_drift=max_drift,
        signed_median_drift=signed_median,
        severity=severity,
        local_tokens=local_tokens,
    )


def _recompute_checkpoint_against_tokens(
    checkpoint: DriftCheckpoint,
    mapped_tokens: list[AlignmentToken],
    profile: Any,
) -> DriftCheckpoint:
    recomputed = _checkpoint_from_local_tokens(
        TimelineRange(
            start_char=checkpoint.start_char,
            end_char=checkpoint.end_char,
            audio_start=checkpoint.audio_start,
            audio_end=checkpoint.audio_end,
            reasons=["checkpoint_drift_probe"],
            severity="medium",
            index=checkpoint.index,
        ),
        checkpoint.local_tokens,
        mapped_tokens,
        profile,
    )
    return recomputed or checkpoint


def _drifts_against_tokens(
    local_tokens: list[AlignmentToken],
    mapped_tokens: list[AlignmentToken],
) -> list[tuple[float, float]]:
    mapped = sorted(
        [
            token
            for token in mapped_tokens
            if token.start_char is not None and token.end_char is not None
        ],
        key=lambda token: (token.start_char or 0, token.end_char or 0, token.start),
    )
    drifts: list[tuple[float, float]] = []
    for token in local_tokens:
        if token.start_char is None or token.end_char is None:
            continue
        sample_char = max(token.start_char, min(token.end_char - 1, (token.start_char + token.end_char - 1) // 2))
        global_time = _token_time_at_char(mapped, sample_char)
        if global_time is None:
            continue
        local_time = (token.start + token.end) / 2
        signed = local_time - global_time
        drifts.append((signed, abs(signed)))
    return drifts


def _token_time_at_char(tokens: list[AlignmentToken], char_pos: int) -> float | None:
    nearest: tuple[int, AlignmentToken] | None = None
    for token in tokens:
        start_char = token.start_char or 0
        end_char = token.end_char or start_char
        if start_char <= char_pos < end_char:
            span = max(1, end_char - start_char)
            ratio = min(1.0, max(0.0, (char_pos + 0.5 - start_char) / span))
            return token.start + (token.end - token.start) * ratio
        distance = min(abs(char_pos - start_char), abs(char_pos - max(start_char, end_char - 1)))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, token)
    if nearest and nearest[0] <= 3:
        token = nearest[1]
        return (token.start + token.end) / 2
    return None


def _checkpoint_drift_severity(median_drift: float, p95_drift: float, max_drift: float) -> str:
    if median_drift >= CHECKPOINT_HIGH_MEDIAN_DRIFT:
        return "high"
    if p95_drift >= CHECKPOINT_HIGH_P95_DRIFT and median_drift >= CHECKPOINT_HIGH_SUPPORTING_MEDIAN_DRIFT:
        return "high"
    if median_drift >= CHECKPOINT_MEDIUM_MEDIAN_DRIFT:
        return "medium"
    if p95_drift >= CHECKPOINT_MEDIUM_P95_DRIFT and median_drift >= CHECKPOINT_MEDIUM_SUPPORTING_MEDIAN_DRIFT:
        return "medium"
    return "ok"


def _drift_ranges_from_checkpoints(
    display_text: str,
    mapped_tokens: list[AlignmentToken],
    checkpoints: list[DriftCheckpoint],
    audio_duration: float,
) -> list[DriftRange]:
    drifted = [checkpoint for checkpoint in checkpoints if checkpoint.severity != "ok"]
    if not drifted:
        return []

    raw_ranges: list[DriftRange] = []
    for checkpoint in drifted:
        start_char = _previous_sentence_boundary(display_text, checkpoint.start_char, 260)
        end_char = _next_sentence_boundary(display_text, checkpoint.end_char, 320)
        audio_start, audio_end = _audio_window_for_char_range(mapped_tokens, start_char, end_char, audio_duration)
        timeline_range = TimelineRange(
            start_char=start_char,
            end_char=end_char,
            audio_start=audio_start,
            audio_end=audio_end,
            reasons=["checkpoint_drift", "silent_timeline_drift"],
            severity=checkpoint.severity,
            index=checkpoint.index,
        )
        raw_ranges.append(
            DriftRange(
                timeline_range=timeline_range,
                checkpoint_count=1,
                median_drift=checkpoint.median_drift,
                p95_drift=checkpoint.p95_drift,
                max_drift=checkpoint.max_drift,
                signed_median_drift=checkpoint.signed_median_drift,
                direction=checkpoint.direction,
            )
        )

    return _merge_drift_ranges(raw_ranges)


def _merge_drift_ranges(ranges: list[DriftRange]) -> list[DriftRange]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: item.timeline_range.start_char)
    merged: list[DriftRange] = []
    for item in ordered:
        if (
            not merged
            or item.timeline_range.start_char > merged[-1].timeline_range.end_char + 240
            or item.direction != merged[-1].direction
        ):
            item.timeline_range.index = len(merged) + 1
            merged.append(item)
            continue
        prev = merged[-1]
        timeline_range = TimelineRange(
            start_char=prev.timeline_range.start_char,
            end_char=max(prev.timeline_range.end_char, item.timeline_range.end_char),
            audio_start=min(prev.timeline_range.audio_start, item.timeline_range.audio_start),
            audio_end=max(prev.timeline_range.audio_end, item.timeline_range.audio_end),
            reasons=sorted(set(prev.timeline_range.reasons) | set(item.timeline_range.reasons)),
            severity="high" if "high" in {prev.severity, item.severity} else "medium",
            index=prev.timeline_range.index,
        )
        merged[-1] = DriftRange(
            timeline_range=timeline_range,
            checkpoint_count=prev.checkpoint_count + item.checkpoint_count,
            median_drift=max(prev.median_drift, item.median_drift),
            p95_drift=max(prev.p95_drift, item.p95_drift),
            max_drift=max(prev.max_drift, item.max_drift),
            signed_median_drift=prev.signed_median_drift
            if abs(prev.signed_median_drift) >= abs(item.signed_median_drift)
            else item.signed_median_drift,
            direction=prev.direction,
        )
    return merged


def _prioritize_drift_ranges(ranges: list[DriftRange]) -> list[DriftRange]:
    return sorted(
        ranges,
        key=lambda item: (
            0 if item.severity == "high" else 1,
            -item.max_drift,
            item.timeline_range.start_char,
        ),
    )


def _register_drift_suspects(summary: TimelineRepairSummary, drift_ranges: list[DriftRange]) -> None:
    existing = {
        (item.get("start_char"), item.get("end_char"), tuple(item.get("reasons", [])))
        for item in summary.suspect_ranges
    }
    existing_drift = {
        (item.get("start_char"), item.get("end_char"), tuple(item.get("reasons", [])))
        for item in summary.drift_suspect_ranges
    }
    for index, drift_range in enumerate(drift_ranges, start=1):
        drift_range.timeline_range.index = index
        payload = drift_range.to_payload()
        key = (payload.get("start_char"), payload.get("end_char"), tuple(payload.get("reasons", [])))
        if key not in existing_drift:
            summary.drift_suspect_ranges.append(payload)
            existing_drift.add(key)
        if key not in existing:
            summary.suspect_ranges.append(payload)
            existing.add(key)


def _append_unresolved_drift_range(summary: TimelineRepairSummary, drift_range: DriftRange, mode: str) -> None:
    payload = {
        **drift_range.to_payload(),
        "mode": mode,
        "confidence": "low",
    }
    existing = {
        (item.get("start_char"), item.get("end_char"), item.get("mode"))
        for item in summary.unresolved_drift_ranges
    }
    key = (payload.get("start_char"), payload.get("end_char"), payload.get("mode"))
    if key not in existing:
        summary.unresolved_drift_ranges.append(payload)


def _update_checkpoint_drift_stats(summary: TimelineRepairSummary, checkpoints: list[DriftCheckpoint]) -> None:
    drifts = [checkpoint.max_drift for checkpoint in checkpoints]
    summary.verified_checkpoint_count = len(checkpoints)
    summary.checkpoint_drift_count = sum(1 for checkpoint in checkpoints if checkpoint.severity != "ok")
    summary.max_checkpoint_drift_seconds = max(drifts, default=0.0)
    summary.p95_checkpoint_drift_seconds = _percentile_float(drifts, 0.95)


def _audio_window_for_char_range(
    mapped_tokens: list[AlignmentToken],
    start_char: int,
    end_char: int,
    audio_duration: float,
) -> tuple[float, float]:
    tokens = [
        token
        for token in mapped_tokens
        if token.start_char is not None
        and token.end_char is not None
        and token.end_char > start_char
        and token.start_char < end_char
    ]
    if tokens:
        return max(0.0, min(token.start for token in tokens)), min(audio_duration, max(token.end for token in tokens))

    prev_token = _nearest_token_before(mapped_tokens, start_char)
    next_token = _nearest_token_after(mapped_tokens, end_char, audio_duration)
    audio_start = prev_token.end if prev_token else 0.0
    audio_end = next_token.start if next_token else audio_duration
    return max(0.0, audio_start), min(audio_duration, max(audio_start + 0.5, audio_end))


def _detect_cue_timeline_ranges(
    cues: list[SubtitleCue],
    display_text: str,
    audio_duration: float | None,
    profile: Any,
    language: str,
) -> list[TimelineRange]:
    group = language_group(language)
    if group not in {"cjk", "ko"} or len(cues) < 2:
        return []

    raw_ranges: list[tuple[int, int, list[str], str]] = []
    prefix = _cue_risk_prefix(language)
    long_chars = _cue_risk_long_chars(language)
    fast_cps = _cue_risk_fast_cps(language)
    severe_fast_cps = _cue_risk_severe_fast_cps(language)
    for index, cue in enumerate(cues):
        next_cue = cues[index + 1] if index + 1 < len(cues) else None
        gap = (next_cue.start - cue.end) if next_cue else 0.0
        chars = _visible_len(cue.text)
        cps = chars / max(cue.duration, 0.1)
        near_max = cue.duration >= profile.max_duration - 0.05
        reasons: list[str] = []
        severity = "medium"

        if next_cue and gap > 1.5:
            reasons.append(f"{prefix}_severe_gap")
            severity = "high"
        if next_cue and gap > 0.8 and (near_max or chars > long_chars):
            reasons.append(f"{prefix}_long_cue_gap")
            severity = "high"
        if cps > severe_fast_cps or (cps > fast_cps and chars > 18):
            reasons.append(f"{prefix}_fast_cue")
            if cps > severe_fast_cps:
                severity = "high"
        if next_cue and _is_high_risk_boundary(display_text, cue.end_char, language):
            reasons.append("zh_quote_boundary" if group == "cjk" else "ko_unsafe_boundary")
        if _is_dense_maxed_duration_cluster(cues, index, profile, language):
            reasons.append(f"{prefix}_maxed_duration_cluster")

        if not reasons:
            continue
        start_index = max(0, index - 1)
        end_index = min(len(cues) - 1, index + 1)
        raw_ranges.append((start_index, end_index, sorted(set(reasons)), severity))

    merged = _merge_zh_cue_ranges(cues, raw_ranges, audio_duration)
    for index, range_ in enumerate(merged, start=1):
        range_.index = index
    return merged[:4]


def _detect_zh_cue_timeline_ranges(
    cues: list[SubtitleCue],
    display_text: str,
    audio_duration: float | None,
    profile: Any,
    language: str,
) -> list[TimelineRange]:
    return _detect_cue_timeline_ranges(cues, display_text, audio_duration, profile, language)


def _is_dense_maxed_duration_cluster(cues: list[SubtitleCue], index: int, profile: Any, language: str) -> bool:
    if index + 2 >= len(cues):
        return False
    window = cues[index : index + 3]
    maxed = [cue for cue in window if cue.duration >= profile.max_duration - 0.05]
    if len(maxed) < 3:
        return False
    avg_chars = sum(_visible_len(cue.text) for cue in window) / 3
    return avg_chars > _cue_risk_cluster_chars(language)


def _merge_zh_cue_ranges(
    cues: list[SubtitleCue],
    raw_ranges: list[tuple[int, int, list[str], str]],
    audio_duration: float | None,
) -> list[TimelineRange]:
    if not raw_ranges:
        return []
    ordered = sorted(raw_ranges, key=lambda item: item[0])
    merged: list[tuple[int, int, set[str], str]] = []
    for start_index, end_index, reasons, severity in ordered:
        if not merged or start_index > merged[-1][1] + 1:
            merged.append((start_index, end_index, set(reasons), severity))
            continue
        prev_start, prev_end, prev_reasons, prev_severity = merged[-1]
        merged[-1] = (
            prev_start,
            max(prev_end, end_index),
            prev_reasons | set(reasons),
            "high" if "high" in {prev_severity, severity} else "medium",
        )

    ranges: list[TimelineRange] = []
    max_audio = audio_duration or max((cue.end for cue in cues), default=0.0)
    for start_index, end_index, reasons, severity in merged:
        start_cue = cues[start_index]
        end_cue = cues[end_index]
        ranges.append(
            TimelineRange(
                start_char=start_cue.start_char,
                end_char=end_cue.end_char,
                audio_start=max(0.0, start_cue.start),
                audio_end=min(max_audio, max(end_cue.end, start_cue.start + 0.5)),
                reasons=sorted(reasons),
                severity=severity,
            )
        )
    return ranges


def _cue_risk_prefix(language: str) -> str:
    return "ko" if language_group(language) == "ko" else "zh"


def _cue_risk_mode(language: str, suffix: str) -> str:
    return f"{_cue_risk_prefix(language)}_{suffix}"


def _cue_risk_long_chars(language: str) -> int:
    return 38 if language_group(language) == "ko" else 34


def _cue_risk_cluster_chars(language: str) -> int:
    return 34 if language_group(language) == "ko" else 30


def _cue_risk_fast_cps(language: str) -> float:
    return 9.5 if language_group(language) == "ko" else 8.5


def _cue_risk_severe_fast_cps(language: str) -> float:
    return 10.5 if language_group(language) == "ko" else 9.5


def _append_low_confidence_range(summary: TimelineRepairSummary, range_: TimelineRange, mode: str) -> None:
    payload = {
        **range_.to_payload(),
        "mode": mode,
        "confidence": "low",
    }
    existing = {
        (item.get("start_char"), item.get("end_char"), item.get("mode"))
        for item in summary.low_confidence_ranges
    }
    key = (payload.get("start_char"), payload.get("end_char"), payload.get("mode"))
    if key not in existing:
        summary.low_confidence_ranges.append(payload)


def _register_unresolved_cue_timeline_risks(
    summary: TimelineRepairSummary,
    suspect_ranges: list[TimelineRange],
    language: str,
) -> None:
    repaired = {
        (item.get("start_char"), item.get("end_char"))
        for item in summary.repaired_ranges
        if item.get("confidence") == "high"
    }
    for range_ in suspect_ranges:
        if (range_.start_char, range_.end_char) in repaired:
            continue
        if range_.severity == "high":
            _append_low_confidence_range(summary, range_, _cue_risk_mode(language, "unresolved_cue_risk"))


def _register_unresolved_zh_timeline_risks(
    summary: TimelineRepairSummary,
    suspect_ranges: list[TimelineRange],
) -> None:
    _register_unresolved_cue_timeline_risks(summary, suspect_ranges, "zh")


def _try_local_realign_range(
    local_realign: Any,
    align_audio_path: Path,
    display_text: str,
    range_: TimelineRange,
    work_dir: Path,
    language: str,
    profile: Any,
    logs: list[str],
    summary: TimelineRepairSummary,
    paddings: tuple[float, ...] = (4.0, 10.0),
) -> list[AlignmentToken] | None:
    fragment_start, fragment_end = _trim_char_range(display_text, range_.start_char, range_.end_char)
    if fragment_end <= fragment_start:
        return None

    fragment_text = display_text[fragment_start:fragment_end]
    try:
        fragment_cleaned = clean_script_text(fragment_text, preserve_punctuation=True)
    except Exception as exc:
        summary.local_realign_attempts.append(
            _attempt_payload(range_, 0, range_.audio_start, range_.audio_end, "failed", f"text_clean_failed:{exc}")
        )
        return None

    for attempt_index, padding in enumerate(paddings, start=1):
        audio_start = max(0.0, range_.audio_start - padding)
        audio_end = range_.audio_end + padding
        attempt_id = f"{range_.index}_{attempt_index}"
        try:
            alignment = local_realign(
                align_audio_path,
                fragment_cleaned,
                language,
                audio_start,
                audio_end,
                work_dir,
                logs,
                attempt_id,
            )
            local_tokens = _ensure_mapped_tokens(alignment.tokens, fragment_cleaned)
            global_tokens = _offset_local_tokens(local_tokens, fragment_start, audio_start)
            validation_error = _validate_local_realign_tokens(
                global_tokens,
                display_text,
                fragment_start,
                fragment_end,
                audio_start,
                audio_end,
                profile,
            )
            if validation_error is None:
                summary.local_realign_attempts.append(
                    _attempt_payload(range_, attempt_index, audio_start, audio_end, "succeeded", None, len(global_tokens))
                )
                return global_tokens
            summary.local_realign_attempts.append(
                _attempt_payload(range_, attempt_index, audio_start, audio_end, "failed", validation_error, len(global_tokens))
            )
        except Exception as exc:
            summary.local_realign_attempts.append(
                _attempt_payload(range_, attempt_index, audio_start, audio_end, "failed", str(exc))
            )

    return None


def _attempt_payload(
    range_: TimelineRange,
    attempt_index: int,
    audio_start: float,
    audio_end: float,
    status: str,
    reason: str | None,
    token_count: int = 0,
) -> dict[str, Any]:
    return {
        "range_index": range_.index,
        "attempt": attempt_index,
        "audio_start": round(audio_start, 3),
        "audio_end": round(audio_end, 3),
        "status": status,
        "reason": reason,
        "token_count": token_count,
    }


def _detect_suspect_timeline_ranges(
    display_text: str,
    tokens: list[AlignmentToken],
    audio_duration: float,
    profile: Any,
) -> list[TimelineRange]:
    sorted_tokens = sorted(
        [
            token
            for token in tokens
            if token.start_char is not None and token.end_char is not None
        ],
        key=lambda token: (token.start_char or 0, token.end_char or 0),
    )
    if len(sorted_tokens) < 2:
        return []

    raw_ranges: list[tuple[int, int, str, str]] = []
    low_start: int | None = None
    low_end: int | None = None
    zero_start: int | None = None
    zero_end: int | None = None

    for prev, cur in zip(sorted_tokens, sorted_tokens[1:]):
        prev_end = prev.end_char or 0
        cur_end = cur.end_char or prev_end
        if cur_end <= prev_end:
            continue
        visible_delta = _visible_len(display_text[prev_end:cur_end])
        time_delta = cur.start - max(prev.end, prev.start)
        if cur.start < prev.start - 0.12 or cur.end < prev.end - 0.12:
            raw_ranges.append((max(0, prev_end - 1), cur_end, "time_reversal", "high"))
        elif visible_delta > 0:
            cps = visible_delta / max(time_delta, 0.02)
            if time_delta > 6.0 and (visible_delta <= 24 or cps < 1.25):
                raw_ranges.append((prev_end, cur_end, "timestamp_jump", "high"))
            elif cps > max(profile.max_chars_per_second * 2.8, 28.0) and visible_delta >= 8:
                raw_ranges.append((prev_end, cur_end, "cps_spike", "medium"))

        low_conf = cur.confidence is not None and cur.confidence < 0.12
        if low_conf:
            low_start = cur.start_char if low_start is None else min(low_start, cur.start_char or low_start)
            low_end = max(low_end or 0, cur.end_char or 0)
        elif low_start is not None and low_end is not None:
            if _visible_len(display_text[low_start:low_end]) >= 18:
                raw_ranges.append((low_start, low_end, "low_confidence_run", "medium"))
            low_start = None
            low_end = None

        zero_like = cur.duration <= 0.02
        if zero_like:
            zero_start = cur.start_char if zero_start is None else min(zero_start, cur.start_char or zero_start)
            zero_end = max(zero_end or 0, cur.end_char or 0)
        elif zero_start is not None and zero_end is not None:
            if _visible_len(display_text[zero_start:zero_end]) >= 20:
                raw_ranges.append((zero_start, zero_end, "zero_duration_run", "medium"))
            zero_start = None
            zero_end = None

        if cur.start >= audio_duration - 0.5:
            remaining = _visible_len(display_text[cur.start_char or 0 :])
            if remaining >= 80:
                raw_ranges.append((cur.start_char or prev_end, len(display_text), "end_collapse", "high"))
                break

    if low_start is not None and low_end is not None and _visible_len(display_text[low_start:low_end]) >= 18:
        raw_ranges.append((low_start, low_end, "low_confidence_run", "medium"))
    if zero_start is not None and zero_end is not None and _visible_len(display_text[zero_start:zero_end]) >= 20:
        raw_ranges.append((zero_start, zero_end, "zero_duration_run", "medium"))

    expanded = [
        _expand_raw_suspect_range(display_text, start, end, reason, severity)
        for start, end, reason, severity in raw_ranges
        if end > start
    ]
    merged = _merge_suspect_ranges(expanded)
    chunked = _chunk_suspect_ranges(display_text, merged, sorted_tokens, audio_duration, profile)
    for index, range_ in enumerate(chunked, start=1):
        range_.index = index
    return chunked[:8]


def _expand_raw_suspect_range(
    display_text: str,
    start: int,
    end: int,
    reason: str,
    severity: str,
) -> tuple[int, int, set[str], str]:
    expanded_start = _previous_sentence_boundary(display_text, start, 90)
    expanded_end = _next_sentence_boundary(display_text, end, 140)
    return expanded_start, expanded_end, {reason}, severity


def _previous_sentence_boundary(text: str, start: int, limit: int) -> int:
    lower = max(0, start - limit)
    for index in range(start - 1, lower - 1, -1):
        if text[index] in "。！？!?；;」』”’":
            return index + 1
    return lower


def _next_sentence_boundary(text: str, end: int, limit: int) -> int:
    upper = min(len(text), end + limit)
    for index in range(max(0, end), upper):
        if text[index] in "。！？!?；;」』”’":
            return index + 1
    return upper


def _merge_suspect_ranges(ranges: list[tuple[int, int, set[str], str]]) -> list[tuple[int, int, set[str], str]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[int, int, set[str], str]] = []
    for start, end, reasons, severity in ordered:
        if not merged or start > merged[-1][1] + 240:
            merged.append((start, end, set(reasons), severity))
            continue
        prev_start, prev_end, prev_reasons, prev_severity = merged[-1]
        merged[-1] = (
            prev_start,
            max(prev_end, end),
            prev_reasons | reasons,
            "high" if "high" in {prev_severity, severity} else "medium",
        )
    return merged


def _chunk_suspect_ranges(
    display_text: str,
    ranges: list[tuple[int, int, set[str], str]],
    tokens: list[AlignmentToken],
    audio_duration: float,
    profile: Any,
) -> list[TimelineRange]:
    chunked: list[TimelineRange] = []
    max_visible = max(420, profile.max_chars_total * 16)
    for start, end, reasons, severity in ranges:
        cursor = start
        while cursor < end:
            target = _char_after_visible_count(display_text, cursor, end, max_visible)
            chunk_end = _next_sentence_boundary(display_text, target, 100) if target < end else end
            chunk_end = max(chunk_end, min(end, target))
            audio_start, audio_end = _estimate_audio_window(
                display_text,
                tokens,
                cursor,
                chunk_end,
                start,
                end,
                audio_duration,
            )
            chunked.append(
                TimelineRange(
                    start_char=cursor,
                    end_char=chunk_end,
                    audio_start=audio_start,
                    audio_end=audio_end,
                    reasons=sorted(reasons),
                    severity=severity,
                )
            )
            cursor = chunk_end
    return chunked


def _estimate_audio_window(
    display_text: str,
    tokens: list[AlignmentToken],
    chunk_start: int,
    chunk_end: int,
    range_start: int,
    range_end: int,
    audio_duration: float,
) -> tuple[float, float]:
    prev_token = _nearest_token_before(tokens, range_start)
    next_token = _nearest_token_after(tokens, range_end, audio_duration)
    anchor_start_time = max(0.0, (prev_token.end if prev_token else 0.0))
    anchor_end_time = min(audio_duration, (next_token.start if next_token else audio_duration))
    if anchor_end_time <= anchor_start_time:
        anchor_end_time = audio_duration

    total_visible = max(1, _visible_len(display_text[range_start:range_end]))
    chunk_start_visible = _visible_len(display_text[range_start:chunk_start])
    chunk_end_visible = _visible_len(display_text[range_start:chunk_end])
    duration = max(1.0, anchor_end_time - anchor_start_time)
    estimated_start = anchor_start_time + duration * (chunk_start_visible / total_visible)
    estimated_end = anchor_start_time + duration * (chunk_end_visible / total_visible)
    min_duration = max(8.0, _visible_len(display_text[chunk_start:chunk_end]) / 5.0)
    if estimated_end - estimated_start < min_duration:
        center = (estimated_start + estimated_end) / 2
        estimated_start = center - min_duration / 2
        estimated_end = center + min_duration / 2
    return max(0.0, estimated_start), min(audio_duration, max(estimated_start + 0.5, estimated_end))


def _nearest_token_before(tokens: list[AlignmentToken], char_pos: int) -> AlignmentToken | None:
    candidates = [
        token
        for token in tokens
        if token.end_char is not None and token.end_char <= char_pos and (token.confidence is None or token.confidence >= 0.2)
    ]
    return max(candidates, key=lambda token: token.end_char or 0, default=None)


def _nearest_token_after(tokens: list[AlignmentToken], char_pos: int, audio_duration: float) -> AlignmentToken | None:
    candidates = [
        token
        for token in tokens
        if token.start_char is not None
        and token.start_char >= char_pos
        and token.start < audio_duration - 0.5
        and (token.confidence is None or token.confidence >= 0.2)
    ]
    return min(candidates, key=lambda token: token.start_char or 0, default=None)


def _trim_char_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _offset_local_tokens(
    local_tokens: list[AlignmentToken],
    fragment_start: int,
    audio_start: float,
) -> list[AlignmentToken]:
    return [
        AlignmentToken(
            text=token.text,
            start=max(0.0, token.start + audio_start),
            end=max(token.start + audio_start, token.end + audio_start),
            start_char=(token.start_char or 0) + fragment_start,
            end_char=(token.end_char or 0) + fragment_start,
            confidence=token.confidence,
        )
        for token in local_tokens
        if token.start_char is not None and token.end_char is not None
    ]


def _validate_local_realign_tokens(
    tokens: list[AlignmentToken],
    display_text: str,
    fragment_start: int,
    fragment_end: int,
    audio_start: float,
    audio_end: float,
    profile: Any,
) -> str | None:
    if not tokens:
        return "no_tokens"
    tokens = sorted(tokens, key=lambda token: (token.start_char or 0, token.start))
    if (tokens[0].start_char or 0) > fragment_start + 6:
        return "missing_fragment_start"
    if (tokens[-1].end_char or 0) < fragment_end - 6:
        return "missing_fragment_end"
    last_time = audio_start
    low_conf_chars = 0
    total_chars = max(1, _visible_len(display_text[fragment_start:fragment_end]))
    for token in tokens:
        if token.start < audio_start - 0.2 or token.end > audio_end + 0.2:
            return "token_outside_audio_window"
        if token.start < last_time - 0.12:
            return "time_reversal"
        last_time = max(last_time, token.end)
        if token.confidence is not None and token.confidence < 0.08:
            low_conf_chars += _visible_len(display_text[token.start_char or 0 : token.end_char or 0])
    duration = max(0.1, tokens[-1].end - tokens[0].start)
    cps = total_chars / duration
    if cps < 1.6:
        return f"too_slow:{cps:.3f}"
    if cps > max(profile.max_chars_per_second * 1.8, 18.0):
        return f"too_fast:{cps:.3f}"
    if low_conf_chars / total_chars > 0.55:
        return "low_confidence_ratio"
    return None


def _replace_tokens_in_char_range(
    tokens: list[AlignmentToken],
    start_char: int,
    end_char: int,
    replacement: list[AlignmentToken],
) -> list[AlignmentToken]:
    kept = [
        token
        for token in tokens
        if token.start_char is None
        or token.end_char is None
        or token.end_char <= start_char
        or token.start_char >= end_char
    ]
    return [*kept, *replacement]


def _max_trusted_anchor_gap(
    display_text: str,
    tokens: list[AlignmentToken],
    audio_duration: float | None,
    profile: Any,
) -> float:
    if not audio_duration:
        return 0.0
    anchors = _trusted_timeline_anchors(display_text, tokens, 0, 0.0, len(display_text), audio_duration, profile)
    if len(anchors) < 2:
        return 0.0
    return max((cur_time - prev_time for (_, prev_time), (_, cur_time) in zip(anchors, anchors[1:])), default=0.0)


def _token_timeline_diagnostics(
    tokens: list[AlignmentToken],
    display_text: str,
    audio_duration: float | None,
    profile: Any,
) -> dict[str, Any]:
    mapped = sorted(
        [
            token
            for token in tokens
            if token.start_char is not None and token.end_char is not None
        ],
        key=lambda token: (token.start_char or 0, token.end_char or 0, token.start),
    )
    if not mapped:
        return {
            "token_count": len(tokens),
            "mapped_token_count": 0,
            "text_coverage_ratio": 0.0,
            "time_reversal_count": 0,
            "zero_duration_count": 0,
            "low_confidence_token_count": 0,
            "timestamp_jump_count": 0,
            "cps_spike_count": 0,
            "max_token_gap_seconds": 0.0,
            "end_collapse_visible_chars": 0,
        }

    covered_positions: set[int] = set()
    low_confidence_token_count = 0
    low_confidence_visible_chars = 0
    zero_duration_count = 0
    for token in mapped:
        start_char = max(0, token.start_char or 0)
        end_char = min(len(display_text), token.end_char or start_char)
        for char_index in range(start_char, end_char):
            if not display_text[char_index].isspace():
                covered_positions.add(char_index)
        if token.duration <= 0.02:
            zero_duration_count += 1
        if token.confidence is not None and token.confidence < 0.12:
            low_confidence_token_count += 1
            low_confidence_visible_chars += _visible_len(display_text[start_char:end_char])

    time_reversal_count = 0
    timestamp_jump_count = 0
    cps_spike_count = 0
    max_token_gap = 0.0
    for prev, cur in zip(mapped, mapped[1:]):
        if cur.start < prev.start - 0.12 or cur.end < prev.end - 0.12:
            time_reversal_count += 1
        time_gap = cur.start - max(prev.end, prev.start)
        max_token_gap = max(max_token_gap, time_gap)
        prev_end_char = prev.end_char or 0
        cur_end_char = cur.end_char or prev_end_char
        visible_delta = _visible_len(display_text[prev_end_char:cur_end_char])
        if visible_delta <= 0:
            continue
        cps = visible_delta / max(time_gap, 0.02)
        if time_gap > 6.0 and (visible_delta <= 24 or cps < 1.25):
            timestamp_jump_count += 1
        if cps > max(profile.max_chars_per_second * 2.8, 28.0) and visible_delta >= 8:
            cps_spike_count += 1

    end_collapse_visible_chars = 0
    if audio_duration:
        for token in mapped:
            if token.start >= audio_duration - 0.5:
                end_collapse_visible_chars = _visible_len(display_text[token.start_char or 0 :])
                break

    visible_total = max(1, _visible_len(display_text))
    return {
        "token_count": len(tokens),
        "mapped_token_count": len(mapped),
        "first_token_time": round(mapped[0].start, 3),
        "last_token_time": round(mapped[-1].end, 3),
        "text_coverage_ratio": round(len(covered_positions) / visible_total, 6),
        "time_reversal_count": time_reversal_count,
        "zero_duration_count": zero_duration_count,
        "low_confidence_token_count": low_confidence_token_count,
        "low_confidence_visible_chars": low_confidence_visible_chars,
        "timestamp_jump_count": timestamp_jump_count,
        "cps_spike_count": cps_spike_count,
        "max_token_gap_seconds": round(max_token_gap, 3),
        "end_collapse_visible_chars": end_collapse_visible_chars,
    }


def _cue_timeline_diagnostics(
    cues: list[SubtitleCue],
    display_text: str,
    profile: Any,
    language: str,
) -> dict[str, Any]:
    gaps = [cur.start - prev.end for prev, cur in zip(cues, cues[1:])]
    cps_values = [_visible_len(cue.text) / max(cue.duration, 0.1) for cue in cues]
    lengths = [_visible_len(cue.text) for cue in cues]
    cue_ranges = _detect_cue_timeline_ranges(cues, display_text, None, profile, language)
    group = language_group(language)
    return {
        "cue_count": len(cues),
        "max_gap_seconds": round(max(gaps, default=0.0), 3),
        "large_gap_count": sum(1 for gap in gaps if gap > 0.8),
        "max_chars_per_second": round(max(cps_values, default=0.0), 3),
        "max_chars_per_cue": max(lengths, default=0),
        "overlap_count": sum(1 for prev, cur in zip(cues, cues[1:]) if cur.start < prev.end),
        "cue_timeline_risk_count": len(cue_ranges),
        "cue_timeline_risks": [range_.to_payload() for range_ in cue_ranges],
        "zh_timeline_risk_count": len(cue_ranges) if group == "cjk" else 0,
        "zh_timeline_risks": [range_.to_payload() for range_ in cue_ranges] if group == "cjk" else [],
        "ko_timeline_risk_count": len(cue_ranges) if group == "ko" else 0,
        "ko_timeline_risks": [range_.to_payload() for range_ in cue_ranges] if group == "ko" else [],
    }


def _apply_timeline_quality(quality_report: dict[str, Any], summary: TimelineRepairSummary) -> None:
    payload = summary.to_payload()
    quality_report.update(payload)
    warnings = quality_report.setdefault("warnings", [])
    if summary.status == "needs_review":
        quality_report["quality_score"] = min(quality_report.get("quality_score", 100), 75)
        if summary.unresolved_drift_ranges:
            _append_warning_once(warnings, "存在局部时间轴漂移，需检查时间轴")
        if summary.micro_drift_unresolved_ranges:
            _append_warning_once(warnings, "存在局部短段时间轴漂移，需检查时间轴")
        if summary.low_confidence_ranges:
            _append_warning_once(warnings, "存在低置信时间轴估算段，需人工检查时间轴")
    elif summary.status == "repaired":
        quality_report["quality_score"] = min(quality_report.get("quality_score", 100), 95)
    elif summary.checkpoint_drift_count:
        quality_report["quality_score"] = min(quality_report.get("quality_score", 100), 90)


def _append_warning_once(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


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
        "confidence": "low",
        "estimated_confidence": confidence,
        "anchor_count": anchor_count,
        "reason": _timeline_break_reason(cues, break_index, profile),
    }
    logs.append(
        "检测到对齐时间轴断层，已从第 "
        f"{repair_start_index + 1} 条字幕开始用低置信估算重建后段时间轴"
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


def _percentile_float(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


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
    timeline_summary: dict[str, Any] | None = None,
    token_diagnostics: dict[str, Any] | None = None,
    cue_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "language": language,
        "subtitle_profile": profile_key,
        "audio_duration": audio_duration,
        "display_text_length": len(cleaned_display),
        "align_text_length": len(cleaned_align),
        "timeline_diagnostics": timeline_summary or {},
        "token_diagnostics": token_diagnostics or {},
        "cue_diagnostics": cue_diagnostics or {},
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
