"""stable-ts forced-alignment wrapper.

The dependency is imported lazily so the rest of the app can be tested without
downloading models or installing heavy audio packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autosrt_aligner.audio import clip_audio_segment, ensure_ffmpeg_on_path, probe_audio_duration
from autosrt_aligner.errors import AlignmentError, DependencyError
from autosrt_aligner.models import AlignmentResult, AlignmentToken, AudioChunk, CleanedText


class StableTsEngine:
    requires_audio_preprocessing = True
    engine_name = "stable-ts"
    enforce_chunk_timing = True

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            import stable_whisper  # type: ignore

            self._model = stable_whisper.load_model(self.model_name)
        return self._model

    def align(
        self,
        audio_path: Path,
        cleaned_text: CleanedText,
        language: str,
        logs: list[str],
    ) -> AlignmentResult:
        ensure_ffmpeg_on_path()
        try:
            import stable_whisper  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local install
            raise DependencyError(
                "未安装 stable-ts。请先运行 pip install -r requirements.txt。"
            ) from exc

        logs.append(f"加载 stable-ts 模型: {self.model_name}")
        try:
            model = self._load_model()
            language_arg = "zh" if language == "zh-TW" else language
            logs.append(f"开始 stable-ts forced alignment，language={language_arg}")
            result = model.align(str(audio_path), cleaned_text.align_text, language=language_arg)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise AlignmentError(f"stable-ts 对齐失败: {exc}") from exc

        tokens = _extract_tokens(result)
        if not tokens:
            raise AlignmentError("stable-ts 未返回可用 token/word 时间戳")

        raw = _summarize_result(result)
        raw["confidence_available"] = True
        raw["requested_language"] = language
        raw["stable_ts_language"] = language_arg
        return AlignmentResult(tokens=tokens, raw=raw, audio_duration=raw.get("duration"), language=language)

    def realign_fragment(
        self,
        audio_path: Path,
        cleaned_text: CleanedText,
        language: str,
        audio_start: float,
        audio_end: float,
        work_dir: Path,
        logs: list[str],
        attempt_id: str,
    ) -> AlignmentResult:
        clip_path = clip_audio_segment(
            audio_path,
            work_dir / f"timeline_realign_{attempt_id}.wav",
            audio_start,
            audio_end,
        )
        logs.append(
            "局部重对齐: "
            f"{audio_start:.3f}s-{audio_end:.3f}s, 文本字符数 {len(cleaned_text.display_text)}"
        )
        return self.align(clip_path, cleaned_text, language, logs)

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
        """Align one bounded audio window and return global timestamps."""

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


def _extract_tokens(result: Any) -> list[AlignmentToken]:
    tokens: list[AlignmentToken] = []
    segments = _get_attr_or_item(result, "segments", [])
    for segment in segments or []:
        words = _get_attr_or_item(segment, "words", None)
        if words:
            for word in words:
                text = (
                    _get_attr_or_item(word, "word", None)
                    or _get_attr_or_item(word, "text", "")
                    or ""
                )
                start = _as_float(_get_attr_or_item(word, "start", 0.0))
                end = _as_float(_get_attr_or_item(word, "end", start))
                probability = _get_attr_or_item(word, "probability", None)
                if text and end >= start:
                    tokens.append(
                        AlignmentToken(
                            text=str(text),
                            start=start,
                            end=end,
                            confidence=_as_optional_float(probability),
                            unit_type="word",
                        )
                    )
        else:
            text = _get_attr_or_item(segment, "text", "") or ""
            start = _as_float(_get_attr_or_item(segment, "start", 0.0))
            end = _as_float(_get_attr_or_item(segment, "end", start))
            if text and end >= start:
                tokens.append(
                    AlignmentToken(
                        text=str(text),
                        start=start,
                        end=end,
                        unit_type="segment",
                    )
                )
    return tokens


def _summarize_result(result: Any) -> dict[str, Any]:
    segments = _get_attr_or_item(result, "segments", [])
    duration = _get_attr_or_item(result, "duration", None)
    return {
        "engine": "stable-ts",
        "duration": _as_optional_float(duration),
        "segment_count": len(segments or []),
    }


def _get_attr_or_item(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _clip_duration_or_default(audio_path: Path, fallback: float) -> float:
    try:
        duration = probe_audio_duration(audio_path)
    except Exception:
        return fallback
    return duration if duration > 0 else fallback


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
