"""Qwen3-ForcedAligner running through mlx-audio on Apple Silicon.

The adapter deliberately accepts the existing cleaned source text instead of
running a second transcription pass.  Qwen's returned timestamps are kept as
real model evidence; this module never splits tokens or fabricates confidence
scores.
"""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from autosrt_aligner.audio import clip_audio_segment, ensure_ffmpeg_on_path, probe_audio_duration
from autosrt_aligner.errors import AlignmentError, DependencyError
from autosrt_aligner.models import AlignmentResult, AlignmentToken, AudioChunk, CleanedText


DEFAULT_QWEN_MLX_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

QWEN_MLX_LANGUAGES = {
    "zh": "Chinese",
    "zh-TW": "Chinese",
    "ja": "Japanese",
    "en": "English",
    "ko": "Korean",
}


class QwenMlxEngine:
    """Known-transcript forced alignment using the MLX Qwen model."""

    requires_audio_preprocessing = True
    engine_name = "qwen3-forced-aligner-mlx"
    enforce_chunk_timing = True
    # Keep a single call below the model's documented short-input budget. Long
    # recordings continue through the existing overlap/retry pipeline.
    max_full_context_seconds = 180.0

    def __init__(
        self,
        model_name: str | None = None,
        *,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "AUTOSRT_QWEN_MLX_MODEL",
            DEFAULT_QWEN_MLX_MODEL,
        )
        self._loader = loader
        self._model: Any | None = None

    def _load_model(self, logs: list[str]) -> Any:
        if self._model is not None:
            return self._model

        loader = self._loader
        if loader is None:
            if not qwen_mlx_runtime_available():
                raise DependencyError(
                    "当前 Python 进程没有可用的 MLX/Metal 设备，"
                    "无法启动 Qwen3-ForcedAligner。请在可访问 Apple GPU 的桌面进程中运行。"
                )
            try:
                from mlx_audio.stt import load  # type: ignore
            except Exception as exc:  # pragma: no cover - local optional dependency
                raise DependencyError(
                    "未安装 mlx-audio，无法使用 Qwen3-ForcedAligner。"
                    "请在 Apple Silicon 环境运行: pip install -U mlx-audio"
                ) from exc
            loader = load

        logs.append(f"加载 Qwen3-ForcedAligner/MLX 模型: {self.model_name}")
        try:
            self._model = loader(self.model_name)
        except Exception as exc:  # pragma: no cover - model download/runtime dependent
            raise DependencyError(
                f"Qwen3-ForcedAligner/MLX 模型加载失败: {exc}"
            ) from exc
        return self._model

    def align(
        self,
        audio_path: Path,
        cleaned_text: CleanedText,
        language: str,
        logs: list[str],
    ) -> AlignmentResult:
        qwen_language = _qwen_language(language)
        ensure_ffmpeg_on_path()
        model = self._load_model(logs)
        logs.append(
            f"开始 Qwen3-ForcedAligner/MLX 已知原文对齐，language={qwen_language}"
        )
        try:
            result = model.generate(
                str(audio_path),
                text=cleaned_text.align_text,
                language=qwen_language,
            )
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise AlignmentError(f"Qwen3-ForcedAligner/MLX 对齐失败: {exc}") from exc

        tokens = _extract_tokens(result)
        if not tokens:
            raise AlignmentError("Qwen3-ForcedAligner/MLX 未返回可用 token 时间戳")

        confidence_available = any(token.confidence is not None for token in tokens)
        raw = {
            "engine": self.engine_name,
            "model": self.model_name,
            "qwen_language": qwen_language,
            "requested_language": language,
            "confidence_available": confidence_available,
            "duration": _as_optional_float(_get_attr_or_item(result, "duration")),
        }
        return AlignmentResult(
            tokens=tokens,
            raw=raw,
            audio_duration=raw["duration"],
            language=language,
        )

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
        """Align a clipped window and restore its timestamps to global time."""

        clip_path = clip_audio_segment(
            audio_path,
            work_dir / f"chunk_{chunk.chunk_id}_{attempt_id}.wav",
            chunk.audio_start,
            chunk.audio_end,
        )
        result = self.align(clip_path, cleaned_text, language, logs)
        offset = chunk.audio_start
        clip_duration = _clip_duration_or_default(
            clip_path,
            chunk.audio_end - chunk.audio_start,
        )
        tokens = [
            AlignmentToken(
                text=token.text,
                start=max(0.0, token.start + offset),
                end=max(token.start + offset, token.end + offset),
                confidence=token.confidence,
                chunk_id=chunk.chunk_id,
                unit_type=token.unit_type,
            )
            for token in result.tokens
        ]
        return AlignmentResult(
            tokens=tokens,
            raw={
                **result.raw,
                "chunk_id": chunk.chunk_id,
                "effective_audio_start": offset,
                "effective_audio_end": offset + clip_duration,
            },
            audio_duration=clip_duration,
            language=language,
        )


def _qwen_language(language: str) -> str:
    try:
        return QWEN_MLX_LANGUAGES[language]
    except KeyError as exc:
        supported = ", ".join(QWEN_MLX_LANGUAGES)
        raise AlignmentError(
            f"Qwen3-ForcedAligner/MLX 暂不支持语言 {language!r}，支持: {supported}"
        ) from exc


def qwen_mlx_runtime_available() -> bool:
    """Probe MLX in a child process because a headless Metal import can abort."""

    try:
        if (
            importlib.util.find_spec("mlx_audio") is None
            or importlib.util.find_spec("mlx") is None
        ):
            return False
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mlx_audio.stt import load; "
                "import mlx.core as mx; print(mx.default_device())",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _extract_tokens(result: Any) -> list[AlignmentToken]:
    tokens: list[AlignmentToken] = []
    for item in _iter_items(result):
        nested = _get_attr_or_item(item, "items", None)
        if nested is None:
            nested = _get_attr_or_item(item, "words", None)
        if nested is not None:
            tokens.extend(_extract_tokens(nested))
            continue

        text = (
            _get_attr_or_item(item, "text", None)
            or _get_attr_or_item(item, "word", None)
            or _get_attr_or_item(item, "char", None)
            or ""
        )
        start = _as_float(
            _get_attr_or_item(
                item,
                "start_time",
                _get_attr_or_item(item, "start", 0.0),
            )
        )
        end = _as_float(
            _get_attr_or_item(
                item,
                "end_time",
                _get_attr_or_item(item, "end", start),
            )
        )
        confidence = _as_optional_float(
            _get_attr_or_item(
                item,
                "confidence",
                _get_attr_or_item(
                    item,
                    "score",
                    _get_attr_or_item(item, "probability", None),
                ),
            )
        )
        if text and end >= start:
            normalized_text = str(text)
            unit_type = "character" if _is_single_cjk_unit(normalized_text) else "word"
            tokens.append(
                AlignmentToken(
                    text=normalized_text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    unit_type=unit_type,
                )
            )
    return tokens


def _iter_items(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, dict):
        for key in ("items", "timestamps", "words", "segments", "results"):
            nested = value.get(key)
            if nested is not None:
                return _iter_items(nested)
        return (value,)
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, (list, tuple)):
        return value
    if hasattr(value, "__iter__") and not hasattr(value, "text"):
        return value
    return (value,)


def _is_single_cjk_unit(value: str) -> bool:
    visible = "".join(value.split())
    return len(visible) == 1 and any(
        "CJK" in _unicode_name(char)
        or "HIRAGANA" in _unicode_name(char)
        or "KATAKANA" in _unicode_name(char)
        or "HANGUL" in _unicode_name(char)
        for char in visible
    )


def _unicode_name(value: str) -> str:
    import unicodedata

    return unicodedata.name(value, "")


def _get_attr_or_item(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _clip_duration_or_default(audio_path: Path, fallback: float) -> float:
    try:
        duration = probe_audio_duration(audio_path)
    except Exception:
        return fallback
    return duration if duration > 0 else fallback
