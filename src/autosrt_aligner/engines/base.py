"""Alignment engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from autosrt_aligner.models import AlignmentResult, AudioChunk, CleanedText


class AlignmentEngine(Protocol):
    requires_audio_preprocessing: bool

    def align(
        self,
        audio_path: Path,
        cleaned_text: CleanedText,
        language: str,
        logs: list[str],
    ) -> AlignmentResult:
        """Align cleaned text to audio and return timestamped tokens."""


class ChunkAlignmentEngine(AlignmentEngine, Protocol):
    def align_chunk(
        self,
        audio_path: Path,
        cleaned_text: CleanedText,
        language: str,
        chunk: AudioChunk,
        work_dir: Path,
        logs: list[str],
        attempt_id: str,
    ) -> AlignmentResult:
        """Align a bounded audio window and return global timestamps."""
