"""Evidence-first alignment pipeline used by the refactored application."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .audio import preprocess_audio, probe_audio_duration
from .engines.base import AlignmentEngine
from .engines.factory import create_alignment_engine
from .errors import AlignmentError, ExportValidationError
from .formats import export_srt, export_vtt
from .models import (
    AlignmentResult,
    AlignmentToken,
    AudioChunk,
    ChunkAlignment,
    CleanedText,
    JobResult,
    SourceDocument,
)
from .chunking import plan_audio_chunks
from .quality_v2 import build_v2_quality_report
from .reconcile import reconcile_chunk_alignments
from .segmenter_v2 import segment_cues
from .text import build_source_document, map_tokens_to_source_strict
from .profiles import resolve_profile

FULL_CONTEXT_MAX_SECONDS = 300.0
INSTANT_TOKEN_RUN_RETRY_THRESHOLD = 5


class _ChunkTimingWindowError(AlignmentError):
    """A token was placed outside the real audio window for this attempt."""

    def __init__(
        self,
        message: str,
        *,
        start_overrun: float = 0.0,
        end_overrun: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.start_overrun = start_overrun
        self.end_overrun = end_overrun


def run_alignment_job_v2(
    audio_path: str | Path,
    script_text: str,
    language: str = "zh",
    subtitle_profile: str = "youtube_long",
    output_dir: str | Path | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    max_chars_per_line: int | None = None,
    generate_vtt: bool = True,
    preserve_punctuation: bool | None = None,
    engine: AlignmentEngine | None = None,
    alignment_engine: str | None = None,
) -> JobResult:
    """Run the V2 pipeline without timestamp invention or silent text mapping."""

    logs: list[str] = []
    engine = engine or create_alignment_engine(alignment_engine)
    logs.append(
        "V2 对齐引擎: "
        f"{getattr(engine, 'engine_name', type(engine).__name__)}"
    )
    profile = resolve_profile(
        subtitle_profile,
        language,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars_per_line=max_chars_per_line,
    )
    source = build_source_document(script_text, language, preserve_punctuation=True)
    logs.append(f"V2 文案字符数: {len(source.display_text)}")

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="autosrt_aligner_v2_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if getattr(engine, "requires_audio_preprocessing", True):
        logs.append("V2 开始音频预处理: 16kHz mono wav")
        audio_info = preprocess_audio(audio_path, work_dir)
        align_audio_path = audio_info.wav_path
        audio_duration = audio_info.duration
    else:
        align_audio_path = Path(audio_path)
        audio_duration = _safe_probe_duration(align_audio_path)

    try:
        chunks, alignments, audio_duration = _align_source(
            engine=engine,
            source=source,
            align_audio_path=align_audio_path,
            audio_duration=audio_duration,
            work_dir=work_dir,
            language=language,
            logs=logs,
        )
    except AlignmentError as exc:
        _write_artifacts(
            out_dir=out_dir,
            source=source,
            chunks=[],
            tokens=[],
            cues=[],
            quality_report=_failure_quality_report(source, audio_duration, str(exc)),
        )
        raise ExportValidationError(f"V2 对齐证据不足，已阻断导出: {exc}") from exc
    tokens, reconcile = reconcile_chunk_alignments(alignments, source)
    logs.append(
        f"V2 分块 {len(chunks)} 个，对齐 token {len(tokens)} 个，"
        f"source 覆盖率 {reconcile.source_token_coverage:.6f}"
    )
    if reconcile.issues:
        logs.append("V2 时间证据存在问题: " + "；".join(reconcile.issues))

    try:
        cues = segment_cues(source, tokens, profile)
    except AlignmentError as exc:
        failure_report = _failure_quality_report(source, audio_duration, str(exc))
        failure_report.update(
            {
                "source_token_coverage": reconcile.source_token_coverage,
                "token_evidence_status": (
                    "pass" if reconcile.source_token_coverage >= 1.0 else "fail"
                ),
                "chunk_count": len(chunks),
                "subtitle_count": 0,
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
                "confidence_status": _confidence_status(reconcile),
                "timing_issues": list(reconcile.issues),
            }
        )
        _write_artifacts(
            out_dir=out_dir,
            source=source,
            chunks=chunks,
            tokens=tokens,
            cues=[],
            quality_report=failure_report,
            alignments=alignments,
        )
        raise ExportValidationError(f"V2 语义切分证据不足，已阻断导出: {exc}") from exc
    quality_report = build_v2_quality_report(
        source,
        tokens,
        cues,
        chunks,
        reconcile,
        profile,
        audio_duration=audio_duration,
        alignments=alignments,
    )

    alignment_payload = _build_v2_alignment_payload(
        source,
        chunks,
        tokens,
        cues,
        quality_report,
        alignments=alignments,
    )
    alignment_json_path = out_dir / "alignment.json"
    quality_report_path = out_dir / "quality_report.json"
    _write_artifacts(
        out_dir=out_dir,
        source=source,
        chunks=chunks,
        tokens=tokens,
        cues=cues,
        quality_report=quality_report,
        alignments=alignments,
    )

    if quality_report["publish_status"] != "pass":
        raise ExportValidationError(
            "V2 质量门禁阻断导出: "
            + "；".join(quality_report["warnings"] or ["未知质量问题"])
        )

    srt_path = out_dir / "output.srt"
    srt_path.write_text(
        export_srt(
            cues,
            strip_trailing_punctuation=preserve_punctuation is not True,
            language=language,
        ),
        encoding="utf-8",
    )
    vtt_path = None
    if generate_vtt:
        vtt_path = out_dir / "output.vtt"
        vtt_path.write_text(export_vtt(cues), encoding="utf-8")
    logs.append("V2 质量门禁通过，导出完成")

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


def _align_source(
    *,
    engine: AlignmentEngine,
    source: SourceDocument,
    align_audio_path: Path,
    audio_duration: float | None,
    work_dir: Path,
    language: str,
    logs: list[str],
) -> tuple[list[AudioChunk], list[ChunkAlignment], float]:
    chunk_aligner = getattr(engine, "align_chunk", None)
    if not audio_duration or audio_duration <= 0:
        return _align_full(
            engine=engine,
            source=source,
            align_audio_path=align_audio_path,
            audio_duration=audio_duration,
            language=language,
            logs=logs,
        )

    # Preserve the engine's global context whenever one strict pass can map
    # the complete source. Overlapping chunks are a recovery path, not the
    # default, because independently aligned windows can disagree at seams.
    engine_context_limit = _full_context_limit(engine)
    if not callable(chunk_aligner) or audio_duration <= engine_context_limit:
        try:
            logs.append("V2 先尝试整段严格对齐，保留全局上下文")
            return _align_full(
                engine=engine,
                source=source,
                align_audio_path=align_audio_path,
                audio_duration=audio_duration,
                language=language,
                logs=logs,
            )
        except AlignmentError as exc:
            if not callable(chunk_aligner):
                raise
            logs.append(f"V2 整段严格对齐失败，改用重叠分块: {exc}")
    else:
        logs.append(
            f"V2 音频 {audio_duration:.1f}s 超过整段上下文上限 "
            f"{engine_context_limit:.0f}s，直接使用重叠分块"
        )

    chunks = plan_audio_chunks(
        source,
        audio_duration,
        target_seconds=_chunk_target_seconds(engine),
        overlap_seconds=_chunk_overlap_seconds(engine),
    )
    alignments: list[ChunkAlignment] = []
    for index, planned_chunk in enumerate(chunks, start=1):
        previous_core_end_time = (
            _previous_core_end_time(alignments[-1], planned_chunk)
            if alignments and getattr(engine, "enforce_chunk_timing", False)
            else None
        )
        chunk = _anchor_chunk_to_previous(
            planned_chunk,
            previous_core_end_time,
            audio_duration,
        )
        if chunk != planned_chunk:
            chunks[index - 1] = chunk
            logs.append(
                f"{chunk.chunk_id} 使用上一分块实际时间锚定窗口: "
                f"{planned_chunk.audio_start:.3f}-{planned_chunk.audio_end:.3f}s -> "
                f"{chunk.audio_start:.3f}-{chunk.audio_end:.3f}s"
            )
        chunk_text = source.display_text[chunk.source_start : chunk.source_end]
        chunk_source = build_source_document(chunk_text, language, preserve_punctuation=True)
        cleaned = CleanedText(
            display_text=chunk_source.display_text,
            align_text=chunk_source.align_text,
            align_to_display=list(chunk_source.align_to_display),
        )
        logs.append(
            f"V2 对齐 {chunk.chunk_id}: source {chunk.source_start}-{chunk.source_end}, "
            f"audio {chunk.audio_start:.3f}-{chunk.audio_end:.3f}s"
        )
        mapped, result = _align_chunk_with_retry(
            chunk_aligner=chunk_aligner,
            align_audio_path=align_audio_path,
            cleaned=cleaned,
            chunk=chunk,
            chunk_source=chunk_source,
            audio_duration=audio_duration,
            language=language,
            work_dir=work_dir,
            logs=logs,
            attempt_prefix=f"{index:03d}",
            previous_core_end_time=previous_core_end_time,
        )
        alignments.append(_chunk_alignment(chunk, mapped, result))
    return chunks, alignments, audio_duration


def _align_full(
    *,
    engine: AlignmentEngine,
    source: SourceDocument,
    align_audio_path: Path,
    audio_duration: float | None,
    language: str,
    logs: list[str],
) -> tuple[list[AudioChunk], list[ChunkAlignment], float]:
    result = engine.align(
        align_audio_path,
        CleanedText(
            display_text=source.display_text,
            align_text=source.align_text,
            align_to_display=list(source.align_to_display),
        ),
        language,
        logs,
    )
    duration = result.audio_duration or audio_duration or 0.0
    if duration <= 0:
        raise AlignmentError("对齐引擎没有返回有效音频时长")
    chunk = AudioChunk(
        "chunk-001",
        0,
        len(source.display_text),
        0.0,
        duration,
        core_source_start=0,
        core_source_end=len(source.display_text),
        core_audio_start=0.0,
        core_audio_end=duration,
    )
    mapped = map_tokens_to_source_strict(result.tokens, source, chunk_id=chunk.chunk_id)
    return [chunk], [_chunk_alignment(chunk, mapped, result)], duration


def _align_chunk_with_retry(
    *,
    chunk_aligner: Any,
    align_audio_path: Path,
    cleaned: CleanedText,
    chunk: AudioChunk,
    chunk_source: SourceDocument,
    audio_duration: float,
    language: str,
    work_dir: Path,
    logs: list[str],
    attempt_prefix: str,
    previous_core_end_time: float | None = None,
) -> tuple[list[AlignmentToken], AlignmentResult]:
    errors: list[str] = []
    best_collapse: tuple[int, list[AlignmentToken], AlignmentResult, int] | None = None
    required_start_padding = 0.0
    required_end_padding = 0.0
    window_retry = False
    for attempt, padding in enumerate(
        _retry_padding_seconds(chunk_aligner),
        start=1,
    ):
        if window_retry:
            start_padding = max(
                required_start_padding,
                padding if required_start_padding > 0 else 0.0,
            )
            end_padding = max(
                required_end_padding,
                padding if required_end_padding > 0 else 0.0,
            )
        else:
            start_padding = padding
            end_padding = padding
        attempt_chunk = replace(
            chunk,
            audio_start=max(0.0, chunk.audio_start - start_padding),
            audio_end=min(audio_duration, chunk.audio_end + end_padding),
        )
        try:
            result = chunk_aligner(
                align_audio_path,
                cleaned,
                language,
                attempt_chunk,
                work_dir,
                logs,
                f"{attempt_prefix}-{attempt}",
            )
            mapped = map_tokens_to_source_strict(
                result.tokens,
                chunk_source,
                source_offset=chunk.source_start,
                chunk_id=chunk.chunk_id,
            )
            _validate_chunk_core_monotonicity(
                mapped,
                chunk,
                previous_core_end_time,
            )
            engine_instance = getattr(chunk_aligner, "__self__", None)
            if getattr(engine_instance, "enforce_chunk_timing", False):
                try:
                    _validate_chunk_attempt_timing(mapped, result, attempt_chunk)
                except AlignmentError as exc:
                    if "时间塌缩" not in str(exc):
                        raise
                    collapse_score = _longest_instant_run(mapped)
                    if best_collapse is None or collapse_score < best_collapse[0]:
                        best_collapse = (collapse_score, mapped, result, attempt)
                    errors.append(f"attempt {attempt}: {exc}")
                    logs.append(f"{chunk.chunk_id} 检测到时间塌缩，继续扩大音频上下文")
                    continue
            result.raw["attempts"] = attempt
            result.raw.setdefault("effective_audio_start", attempt_chunk.audio_start)
            result.raw.setdefault("effective_audio_end", attempt_chunk.audio_end)
            return mapped, result
        except _ChunkTimingWindowError as exc:
            required_start_padding = max(
                required_start_padding,
                exc.start_overrun + 0.5 if exc.start_overrun > 0 else 0.0,
            )
            required_end_padding = max(
                required_end_padding,
                exc.end_overrun + 0.5 if exc.end_overrun > 0 else 0.0,
            )
            window_retry = True
            errors.append(f"attempt {attempt}: {exc}")
            directions = []
            if exc.start_overrun > 0:
                directions.append(f"前方至少 {required_start_padding:.3f}s")
            if exc.end_overrun > 0:
                directions.append(f"后方至少 {required_end_padding:.3f}s")
            logs.append(
                f"{chunk.chunk_id} 第 {attempt} 次时间窗口越界，"
                f"按实际越界方向扩大音频上下文（{'、'.join(directions)}）"
            )
        except AlignmentError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            logs.append(f"{chunk.chunk_id} 第 {attempt} 次严格映射失败，扩大音频上下文")
    if best_collapse is not None:
        collapse_score, mapped, result, attempt = best_collapse
        result.raw["attempts"] = attempt
        result.raw.setdefault("effective_audio_start", chunk.audio_start)
        result.raw.setdefault("effective_audio_end", chunk.audio_end)
        logs.append(
            f"{chunk.chunk_id} 三次上下文均有时间塌缩，保留塌缩最少的真实候选: "
            f"连续 {collapse_score} 个 token"
        )
        return mapped, result
    raise AlignmentError(f"{chunk.chunk_id} 无法完成严格对齐: {'; '.join(errors)}")


def _chunk_alignment(
    chunk: AudioChunk,
    tokens: list[AlignmentToken],
    result: AlignmentResult,
) -> ChunkAlignment:
    confidences = [token.confidence for token in tokens if token.confidence is not None]
    effective_audio_start = _optional_float(
        result.raw.get("effective_audio_start"),
        chunk.audio_start,
    )
    effective_audio_end = _optional_float(
        result.raw.get("effective_audio_end"),
        chunk.audio_end,
    )
    return ChunkAlignment(
        chunk=chunk,
        tokens=tokens,
        engine=str(result.raw.get("engine", "unknown")),
        model=result.raw.get("model"),
        confidence=sum(confidences) / len(confidences) if confidences else None,
        attempts=int(result.raw.get("attempts", 1)),
        warnings=[],
        effective_audio_start=effective_audio_start,
        effective_audio_end=effective_audio_end,
        confidence_available=bool(result.raw.get("confidence_available", True)),
    )


def _previous_core_end_time(
    previous: ChunkAlignment,
    current: AudioChunk,
) -> float | None:
    boundary = current.core_source_start
    if boundary is None:
        return None
    previous_core_tokens = [
        token
        for token in previous.tokens
        if token.end_char is not None and token.end_char <= boundary
    ]
    if not previous_core_tokens:
        return None
    return max(token.end for token in previous_core_tokens)


def _anchor_chunk_to_previous(
    chunk: AudioChunk,
    previous_core_end_time: float | None,
    audio_duration: float,
) -> AudioChunk:
    """Shift a planned window to the last confirmed source/audio boundary."""

    if (
        previous_core_end_time is None
        or chunk.core_audio_start is None
        or chunk.core_audio_end is None
    ):
        return chunk

    shift = previous_core_end_time - chunk.core_audio_start
    if abs(shift) <= 0.05:
        return chunk

    audio_start = max(0.0, chunk.audio_start + shift)
    audio_end = min(audio_duration, chunk.audio_end + shift)
    core_audio_start = max(0.0, chunk.core_audio_start + shift)
    core_audio_end = min(audio_duration, chunk.core_audio_end + shift)
    if audio_end <= audio_start or core_audio_end < core_audio_start:
        return chunk

    return replace(
        chunk,
        audio_start=audio_start,
        audio_end=audio_end,
        overlap_before=max(0.0, core_audio_start - audio_start),
        overlap_after=max(0.0, audio_end - core_audio_end),
        core_audio_start=core_audio_start,
        core_audio_end=core_audio_end,
    )


def _validate_chunk_core_monotonicity(
    tokens: list[AlignmentToken],
    chunk: AudioChunk,
    previous_core_end_time: float | None,
) -> None:
    if previous_core_end_time is None or chunk.core_source_start is None:
        return
    current_core_tokens = [
        token
        for token in tokens
        if token.start_char is not None and token.start_char >= chunk.core_source_start
    ]
    if not current_core_tokens:
        return
    first_start = min(token.start for token in current_core_tokens)
    if first_start < previous_core_end_time - 0.05:
        raise AlignmentError(
            f"{chunk.chunk_id} 当前核心 token 时间早于上一分块: "
            f"{first_start:.3f}s < {previous_core_end_time:.3f}s"
        )


def _validate_chunk_attempt_timing(
    tokens: list[AlignmentToken],
    result: AlignmentResult,
    attempt_chunk: AudioChunk,
) -> None:
    effective_start = _optional_float(
        result.raw.get("effective_audio_start"),
        attempt_chunk.audio_start,
    )
    effective_end = _optional_float(
        result.raw.get("effective_audio_end"),
        attempt_chunk.audio_end,
    )
    window_overruns: list[tuple[float, float, AlignmentToken]] = []
    for token in tokens:
        start_overrun = max(0.0, effective_start - token.start)
        end_overrun = max(0.0, token.end - effective_end)
        if start_overrun > 0.25 or end_overrun > 0.25:
            window_overruns.append((start_overrun, end_overrun, token))
    if window_overruns:
        start_overrun, end_overrun, token = max(
            window_overruns,
            key=lambda item: max(item[0], item[1]),
        )
        raise _ChunkTimingWindowError(
            f"{attempt_chunk.chunk_id} 返回 token 超出实际音频窗口: "
            f"{token.text!r} {token.start:.3f}-{token.end:.3f}s, "
            f"有效范围 {effective_start:.3f}-{effective_end:.3f}s",
            start_overrun=start_overrun,
            end_overrun=end_overrun,
        )
    for previous, current in zip(tokens, tokens[1:]):
        if current.start < previous.start - 0.05:
            raise AlignmentError(
                f"{attempt_chunk.chunk_id} 返回 token 时间倒退: "
                f"{previous.text!r}->{current.text!r}"
            )
    if len(tokens) >= 50:
        longest_instant_run = _longest_instant_run(tokens)
        if longest_instant_run >= INSTANT_TOKEN_RUN_RETRY_THRESHOLD:
            raise AlignmentError(
                f"{attempt_chunk.chunk_id} 存在时间塌缩: "
                f"连续 {longest_instant_run} 个 token 时长小于 20ms"
            )


def _longest_instant_run(tokens: list[AlignmentToken]) -> int:
    longest = 0
    current = 0
    for token in tokens:
        if token.duration < 0.02:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _build_v2_alignment_payload(
    source: SourceDocument,
    chunks: list[AudioChunk],
    tokens: list[AlignmentToken],
    cues: list[Any],
    quality_report: dict[str, Any],
    alignments: list[ChunkAlignment] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "source": {
            "language": source.language,
            "source_hash": source.source_hash,
            "display_text": source.display_text,
            "align_text": source.align_text,
        },
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source_start": chunk.source_start,
                "source_end": chunk.source_end,
                "audio_start": round(chunk.audio_start, 3),
                "audio_end": round(chunk.audio_end, 3),
                "overlap_before": round(chunk.overlap_before, 3),
                "overlap_after": round(chunk.overlap_after, 3),
                "core_source_start": chunk.core_source_start,
                "core_source_end": chunk.core_source_end,
                "core_audio_start": (
                    round(chunk.core_audio_start, 3)
                    if chunk.core_audio_start is not None
                    else None
                ),
                "core_audio_end": (
                    round(chunk.core_audio_end, 3)
                    if chunk.core_audio_end is not None
                    else None
                ),
            }
            for chunk in chunks
        ],
        "chunk_evidence": [
            {
                "chunk_id": item.chunk.chunk_id,
                "engine": item.engine,
                "model": item.model,
                "attempts": item.attempts,
                "confidence": item.confidence,
                "token_count": len(item.tokens),
                "warnings": item.warnings,
                "effective_audio_start": (
                    round(item.effective_audio_start, 3)
                    if item.effective_audio_start is not None
                    else None
                ),
                "effective_audio_end": (
                    round(item.effective_audio_end, 3)
                    if item.effective_audio_end is not None
                    else None
                ),
                "confidence_available": item.confidence_available,
            }
            for item in (alignments or [])
        ],
        "tokens": [_token_payload(token) for token in tokens],
        "cues": [
            {
                "index": cue.index,
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "start_char": cue.start_char,
                "end_char": cue.end_char,
                "text": cue.text,
            }
            for cue in cues
        ],
        "quality": quality_report,
    }


def _token_payload(token: AlignmentToken) -> dict[str, Any]:
    return {
        "text": token.text,
        "start": round(token.start, 3),
        "end": round(token.end, 3),
        "start_char": token.start_char,
        "end_char": token.end_char,
        "confidence": token.confidence,
        "chunk_id": token.chunk_id,
        "unit_type": token.unit_type,
    }


def _failure_quality_report(
    source: SourceDocument,
    audio_duration: float | None,
    warning: str,
) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "text_integrity_status": "unknown",
        "token_evidence_status": "unknown",
        "text_status": "unknown",
        "timing_status": "review",
        "segmentation_status": "review",
        "publish_status": "blocked",
        "timeline_status": "needs_review",
        "timeline_confidence_score": 0,
        "timing_verification": "not_available",
        "audio_duration": round(audio_duration, 3) if audio_duration is not None else None,
        "source_hash": source.source_hash,
        "source_visible_length": source.visible_length,
        "subtitle_count": 0,
        "chunk_count": 0,
        "source_token_coverage": 0.0,
        "duplicate_token_count": 0,
        "token_overlap_count": 0,
        "unresolved_overlap_count": 0,
        "resolved_overlap_count": 0,
        "uncovered_source_ranges": [],
        "time_reversal_count": 0,
        "low_confidence_token_count": 0,
        "missing_confidence_token_count": 0,
        "confidence_status": "not_available",
        "warnings": [warning],
        "timing_issues": [warning],
        "segmentation_issues": [],
        "quality_score": 0,
    }


def _optional_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _full_context_limit(engine: AlignmentEngine) -> float:
    value = getattr(engine, "max_full_context_seconds", FULL_CONTEXT_MAX_SECONDS)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return FULL_CONTEXT_MAX_SECONDS
    if not math.isfinite(value) or value <= 0:
        return FULL_CONTEXT_MAX_SECONDS
    return min(FULL_CONTEXT_MAX_SECONDS, value)


def _chunk_target_seconds(engine: AlignmentEngine) -> float:
    value = getattr(engine, "chunk_target_seconds", 45.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 45.0
    return value if math.isfinite(value) and value > 0 else 45.0


def _chunk_overlap_seconds(engine: AlignmentEngine) -> float:
    value = getattr(engine, "chunk_overlap_seconds", 1.5)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.5
    return value if math.isfinite(value) and value >= 0 else 1.5


def _retry_padding_seconds(chunk_aligner: Any) -> tuple[float, ...]:
    engine_instance = getattr(chunk_aligner, "__self__", None)
    values = getattr(engine_instance, "retry_padding_seconds", (0.0, 4.0, 8.0))
    try:
        paddings = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return (0.0, 4.0, 8.0)
    if not paddings or paddings[0] != 0.0 or any(
        not math.isfinite(value) or value < 0 for value in paddings
    ):
        return (0.0, 4.0, 8.0)
    return paddings


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


def _confidence_status(reconcile: Any) -> str:
    return "available" if reconcile.confidence_available else "not_provided"


def _write_artifacts(
    *,
    out_dir: Path,
    source: SourceDocument,
    chunks: list[AudioChunk],
    tokens: list[AlignmentToken],
    cues: list[Any],
    quality_report: dict[str, Any],
    alignments: list[ChunkAlignment] | None = None,
) -> None:
    alignment_payload = _build_v2_alignment_payload(
        source,
        chunks,
        tokens,
        cues,
        quality_report,
        alignments=alignments,
    )
    (out_dir / "alignment.json").write_text(
        json.dumps(alignment_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _safe_probe_duration(audio_path: Path) -> float | None:
    if not audio_path.exists():
        return None
    try:
        return probe_audio_duration(audio_path)
    except Exception:
        return None
