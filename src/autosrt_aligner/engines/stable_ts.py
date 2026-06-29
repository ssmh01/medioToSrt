"""stable-ts forced-alignment wrapper.

The dependency is imported lazily so the rest of the app can be tested without
downloading models or installing heavy audio packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autosrt_aligner.audio import ensure_ffmpeg_on_path
from autosrt_aligner.errors import AlignmentError, DependencyError
from autosrt_aligner.models import AlignmentResult, AlignmentToken, CleanedText


class StableTsEngine:
    requires_audio_preprocessing = True

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name

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
            model = stable_whisper.load_model(self.model_name)
            language_arg = None if language == "auto" else ("zh" if language == "zh-TW" else language)
            logs.append("开始 stable-ts forced alignment")
            result = model.align(str(audio_path), cleaned_text.align_text, language=language_arg)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise AlignmentError(f"stable-ts 对齐失败: {exc}") from exc

        tokens = _extract_tokens(result)
        if not tokens:
            raise AlignmentError("stable-ts 未返回可用 token/word 时间戳")

        raw = _summarize_result(result)
        return AlignmentResult(tokens=tokens, raw=raw, audio_duration=raw.get("duration"), language=language)


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
                        )
                    )
        else:
            text = _get_attr_or_item(segment, "text", "") or ""
            start = _as_float(_get_attr_or_item(segment, "start", 0.0))
            end = _as_float(_get_attr_or_item(segment, "end", start))
            if text and end >= start:
                tokens.append(AlignmentToken(text=str(text), start=start, end=end))
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
