"""Alignment engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from autosrt_aligner.models import AlignmentResult, CleanedText


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

