"""Alignment-engine selection with an explicit, dependency-safe fallback."""

from __future__ import annotations

import os
import platform
from typing import Any

from autosrt_aligner.errors import InputError

from .qwen_mlx import QwenMlxEngine, qwen_mlx_runtime_available
from .stable_ts import StableTsEngine


ENGINE_CHOICES = ("auto", "qwen-mlx", "stable-ts")


def create_alignment_engine(name: str | None = None) -> Any:
    """Create the configured engine without importing heavy ML packages.

    ``auto`` is intentionally conservative: Qwen/MLX is selected only when
    the optional package is installed on Apple Silicon. An explicit
    ``qwen-mlx`` request never falls back to stable-ts, so a missing dependency
    is visible instead of producing a result with an unexpected backend.
    """

    requested = (name or os.getenv("AUTOSRT_ALIGNMENT_ENGINE", "auto")).strip().lower()
    aliases = {
        "default": "auto",
        "qwen": "qwen-mlx",
        "qwen3": "qwen-mlx",
        "qwen3-mlx": "qwen-mlx",
        "stable": "stable-ts",
        "stable_ts": "stable-ts",
    }
    requested = aliases.get(requested, requested)
    if requested == "auto":
        if _qwen_mlx_available():
            return QwenMlxEngine()
        return StableTsEngine()
    if requested == "qwen-mlx":
        return QwenMlxEngine()
    if requested == "stable-ts":
        return StableTsEngine()
    supported = ", ".join(ENGINE_CHOICES)
    raise InputError(f"不支持的对齐引擎 {name!r}，可选: {supported}")


def alignment_engine_status() -> dict[str, Any]:
    selected = os.getenv("AUTOSRT_ALIGNMENT_ENGINE", "auto").strip().lower() or "auto"
    qwen_available = _qwen_mlx_available()
    return {
        "selected": selected,
        "choices": list(ENGINE_CHOICES),
        "qwen_mlx_available": qwen_available,
        "auto_backend": "qwen-mlx" if qwen_available else "stable-ts",
    }


def _qwen_mlx_available() -> bool:
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return False
    return qwen_mlx_runtime_available()
