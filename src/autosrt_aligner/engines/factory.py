"""Qwen MLX alignment-engine selection."""

from __future__ import annotations

import platform
from typing import Any

from autosrt_aligner.errors import InputError

from .qwen_mlx import QwenMlxEngine, qwen_mlx_runtime_available


ENGINE_CHOICES = ("qwen-mlx",)


def create_alignment_engine(name: str | None = None) -> Any:
    """Create the only supported alignment engine.

    Legacy ``auto``/``qwen`` aliases are kept as Qwen aliases so an old local
    setting cannot reintroduce a different backend. Any other engine name is
    rejected instead of falling back silently.
    """

    requested = (name or "qwen-mlx").strip().lower()
    aliases = {
        "default": "qwen-mlx",
        "auto": "qwen-mlx",
        "qwen": "qwen-mlx",
        "qwen3": "qwen-mlx",
        "qwen3-mlx": "qwen-mlx",
    }
    requested = aliases.get(requested, requested)
    if requested == "qwen-mlx":
        return QwenMlxEngine()
    supported = ", ".join(ENGINE_CHOICES)
    raise InputError(
        f"对齐引擎已固定为 Qwen MLX，不支持 {name!r}；可选: {supported}"
    )


def alignment_engine_status() -> dict[str, Any]:
    qwen_available = _qwen_mlx_available()
    return {
        "selected": "qwen-mlx",
        "choices": list(ENGINE_CHOICES),
        "qwen_mlx_available": qwen_available,
        "backend": "qwen-mlx",
    }


def _qwen_mlx_available() -> bool:
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return False
    return qwen_mlx_runtime_available()
