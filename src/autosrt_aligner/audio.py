"""Audio dependency checks and ffmpeg-based preprocessing."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import DependencyError, InputError


@dataclass(frozen=True)
class AudioInfo:
    original_path: Path
    wav_path: Path
    duration: float | None


def require_ffmpeg() -> tuple[str, str | None]:
    ffmpeg = ensure_ffmpeg_on_path()
    ffprobe = shutil.which("ffprobe")
    return ffmpeg, ffprobe


def ensure_ffmpeg_on_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    fallback = _imageio_ffmpeg_path()
    if fallback:
        local_bin = _local_ffmpeg_link(fallback)
        if local_bin:
            os.environ["PATH"] = f"{local_bin.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return str(local_bin)
        return fallback

    raise DependencyError(
        "未找到 ffmpeg。请安装系统 ffmpeg，或运行 pip install imageio-ffmpeg 后重试。"
    )


def probe_audio_duration(audio_path: Path) -> float:
    ffmpeg, ffprobe = require_ffmpeg()
    if ffprobe:
        duration = _probe_duration_with_ffprobe(ffprobe, audio_path)
    else:
        duration = _probe_duration_with_ffmpeg(ffmpeg, audio_path)
    if duration <= 0:
        raise InputError("音频时长无效或音频为空")
    return duration


def preprocess_audio(audio_path: str | Path, work_dir: str | Path) -> AudioInfo:
    source = Path(audio_path)
    if not source.exists():
        raise InputError(f"音频文件不存在: {source}")
    ffmpeg, _ = require_ffmpeg()
    duration = probe_audio_duration(source)
    if duration < 0.5:
        raise InputError("音频太短，无法可靠对齐")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "preprocessed_16k_mono.wav"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "loudnorm",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise InputError(f"音频预处理失败: {result.stderr.strip()}")
    return AudioInfo(original_path=source, wav_path=wav_path, duration=duration)


def _probe_duration_with_ffprobe(ffprobe: str, audio_path: Path) -> float:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise InputError(f"音频文件无法读取: {result.stderr.strip() or audio_path}")
    payload = json.loads(result.stdout or "{}")
    return float(payload.get("format", {}).get("duration") or 0)


def _probe_duration_with_ffmpeg(ffmpeg: str, audio_path: Path) -> float:
    command = [ffmpeg, "-hide_banner", "-i", str(audio_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    if "No such file" in output or "Invalid data" in output:
        raise InputError(f"音频文件无法读取: {audio_path}")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise InputError("无法读取音频时长")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _imageio_ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg  # type: ignore

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def _local_ffmpeg_link(source: str) -> Path | None:
    executable_dir = Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin")
    if not executable_dir.exists():
        executable_dir = Path(sys.executable).resolve().parent
    if not os.access(executable_dir, os.W_OK):
        return None
    target = executable_dir / "ffmpeg"
    if target.exists():
        return target
    try:
        target.symlink_to(source)
    except OSError:
        return None
    return target
