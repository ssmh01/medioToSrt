"""Shared data models for alignment, splitting, and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubtitleProfile:
    key: str
    label: str
    min_duration: float
    max_duration: float
    ideal_min_duration: float
    ideal_max_duration: float
    max_chars_per_line: int
    max_chars_total: int
    max_chars_per_second: float
    gap_seconds: float = 0.08


@dataclass
class CleanedText:
    display_text: str
    align_text: str
    align_to_display: list[int]


@dataclass(frozen=True)
class SourceDocument:
    """Immutable source-text contract used by the V2 alignment pipeline."""

    raw_text: str
    display_text: str
    align_text: str
    align_to_display: tuple[int, ...]
    language: str
    source_hash: str

    @property
    def visible_length(self) -> int:
        return len("".join(self.display_text.split()))


@dataclass(frozen=True)
class AudioChunk:
    """A source span and its corresponding audio search window."""

    chunk_id: str
    source_start: int
    source_end: int
    audio_start: float
    audio_end: float
    overlap_before: float = 0.0
    overlap_after: float = 0.0
    core_source_start: int | None = None
    core_source_end: int | None = None
    core_audio_start: float | None = None
    core_audio_end: float | None = None


@dataclass
class ChunkAlignment:
    """Alignment evidence produced for one audio chunk."""

    chunk: AudioChunk
    tokens: list["AlignmentToken"]
    engine: str
    model: str | None = None
    confidence: float | None = None
    attempts: int = 1
    warnings: list[str] = field(default_factory=list)
    effective_audio_start: float | None = None
    effective_audio_end: float | None = None
    confidence_available: bool = True


@dataclass
class AlignmentToken:
    text: str
    start: float
    end: float
    start_char: int | None = None
    end_char: int | None = None
    confidence: float | None = None
    chunk_id: str | None = None
    unit_type: str = "unknown"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AlignmentResult:
    tokens: list[AlignmentToken]
    raw: dict[str, Any] = field(default_factory=dict)
    audio_duration: float | None = None
    language: str = "zh"


@dataclass
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str
    start_char: int
    end_char: int
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class JobResult:
    output_dir: Path
    srt_path: Path
    vtt_path: Path | None
    alignment_json_path: Path
    quality_report_path: Path
    cues: list[SubtitleCue]
    quality_report: dict[str, Any]
    alignment_payload: dict[str, Any]
    logs: list[str]

    def preview_rows(self) -> list[list[Any]]:
        return [
            [
                cue.index,
                seconds_to_preview(cue.start),
                seconds_to_preview(cue.end),
                cue.text,
                round(cue.duration, 3),
                " / ".join(cue.warnings),
            ]
            for cue in self.cues
        ]


def seconds_to_preview(seconds: float) -> str:
    whole_ms = max(0, int(round(seconds * 1000)))
    ms = whole_ms % 1000
    total_seconds = whole_ms // 1000
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
