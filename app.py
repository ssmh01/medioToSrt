"""FastAPI web UI entrypoint for the AutoSRT aligner."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from autosrt_aligner.errors import AutosrtError, InputError
from autosrt_aligner.pipeline import run_alignment_job
from autosrt_aligner.profiles import PROFILE_LABELS, SUPPORTED_LANGUAGES


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LANGUAGES = ["auto", "zh", "zh-TW", "ja", "en"]
FILE_LABELS = {
    "srt": "字幕文件.srt",
    "vtt": "字幕文件.vtt",
    "quality_report": "quality_report.json",
    "alignment": "alignment.json",
}


@dataclass
class JobRecord:
    job_id: str
    audio_filename: str = "audio"
    status: str = "queued"
    stage: str = "queued"
    logs: list[str] = field(default_factory=list)
    preview_rows: list[list[Any]] = field(default_factory=list)
    quality_report: dict[str, Any] | None = None
    files: dict[str, Path] = field(default_factory=dict)
    error: str | None = None
    output_dir: Path | None = None


app = FastAPI(title="AI 语音原文案对齐 SRT 工具")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/")
def index_head() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
def api_options() -> dict[str, Any]:
    return {
        "languages": [value for value in LANGUAGES if value in SUPPORTED_LANGUAGES],
        "profiles": [
            {"key": key, "label": label}
            for key, label in PROFILE_LABELS.items()
        ],
        "defaults": {
            "language": "auto",
            "subtitle_profile": "youtube_long",
            "min_duration": 1.2,
            "max_duration": 6.5,
            "max_chars_per_line": 18,
            "generate_vtt": True,
            "preserve_punctuation": True,
        },
    }


@app.post("/api/jobs")
def create_job(
    audio_file: UploadFile | None = File(default=None),
    script_file: UploadFile | None = File(default=None),
    script_text: str = Form(default=""),
    language: str = Form(default="auto"),
    subtitle_profile: str = Form(default="youtube_long"),
    min_duration: float = Form(default=1.2),
    max_duration: float = Form(default=6.5),
    max_chars_per_line: int = Form(default=18),
    generate_vtt: bool = Form(default=True),
    preserve_punctuation: bool = Form(default=True),
) -> dict[str, str]:
    if audio_file is None or not audio_file.filename:
        raise HTTPException(status_code=400, detail="请先上传音频文件")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言参数: {language}")
    if subtitle_profile not in PROFILE_LABELS:
        raise HTTPException(status_code=400, detail=f"不支持的字幕风格: {subtitle_profile}")
    if min_duration <= 0:
        raise HTTPException(status_code=400, detail="每条字幕最短时长必须大于 0")
    if max_duration <= min_duration:
        raise HTTPException(status_code=400, detail="每条字幕最长时长必须大于最短时长")
    if max_chars_per_line <= 0:
        raise HTTPException(status_code=400, detail="字幕切分参考字符数必须大于 0")

    text = _read_script_text(script_file, script_text)
    job_id = str(uuid.uuid4())
    output_dir = Path(tempfile.mkdtemp(prefix=f"autosrt_web_{job_id}_"))
    upload_dir = output_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    audio_filename = _safe_filename(audio_file.filename, "audio")
    audio_path = upload_dir / audio_filename
    _save_upload(audio_file, audio_path)
    if script_file is not None and script_file.filename:
        _save_upload(script_file, upload_dir / _safe_filename(script_file.filename, "script.txt"))
    (upload_dir / "script.txt").write_text(text, encoding="utf-8")

    record = JobRecord(
        job_id=job_id,
        audio_filename=audio_filename,
        status="queued",
        stage="queued",
        logs=["已接收上传文件", "任务已进入队列"],
        output_dir=output_dir,
    )
    with _jobs_lock:
        _jobs[job_id] = record

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "audio_path": audio_path,
            "script_text": text,
            "language": language,
            "subtitle_profile": subtitle_profile,
            "output_dir": output_dir,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "max_chars_per_line": max_chars_per_line,
            "generate_vtt": generate_vtt,
            "preserve_punctuation": preserve_punctuation,
        },
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_payload(_get_job(job_id))


@app.get("/api/jobs/{job_id}/files/{kind}")
def download_file(job_id: str, kind: str) -> FileResponse:
    record = _get_job(job_id)
    if kind not in FILE_LABELS:
        raise HTTPException(status_code=404, detail="不支持的下载文件类型")
    path = record.files.get(kind)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="文件尚未生成")
    media_type = "application/json" if kind in {"quality_report", "alignment"} else "text/plain"
    return FileResponse(path, media_type=media_type, filename=_download_filename(record, kind))


def _run_job(
    job_id: str,
    audio_path: Path,
    script_text: str,
    language: str,
    subtitle_profile: str,
    output_dir: Path,
    min_duration: float,
    max_duration: float,
    max_chars_per_line: int,
    generate_vtt: bool,
    preserve_punctuation: bool,
) -> None:
    _update_job(job_id, status="running", stage="aligning", logs=["开始语音识别与字幕对齐"])
    try:
        result = run_alignment_job(
            audio_path=audio_path,
            script_text=script_text,
            language=language,
            subtitle_profile=subtitle_profile,
            output_dir=output_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            max_chars_per_line=max_chars_per_line,
            generate_vtt=generate_vtt,
            preserve_punctuation=preserve_punctuation,
        )
        files = {
            "srt": result.srt_path,
            "quality_report": result.quality_report_path,
            "alignment": result.alignment_json_path,
        }
        if result.vtt_path is not None:
            files["vtt"] = result.vtt_path
        _update_job(
            job_id,
            status="succeeded",
            stage="succeeded",
            logs=result.logs,
            preview_rows=result.preview_rows(),
            quality_report=result.quality_report,
            files=files,
        )
    except (AutosrtError, InputError) as exc:
        _update_job(job_id, status="failed", stage="failed", logs=[str(exc)], error=str(exc))
    except Exception as exc:  # pragma: no cover - UI safety net
        message = f"未知错误: {exc}"
        _update_job(job_id, status="failed", stage="failed", logs=[message], error=message)


def _read_script_text(script_file: UploadFile | None, script_text: str | None) -> str:
    if script_file is not None and script_file.filename:
        content = script_file.file.read()
        script_file.file.seek(0)
        return content.decode("utf-8-sig")
    if script_text and script_text.strip():
        return script_text
    raise HTTPException(status_code=400, detail="请上传 TXT 文案或粘贴原始文案")


def _save_upload(upload: UploadFile, path: Path) -> None:
    upload.file.seek(0)
    with path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    upload.file.seek(0)


def _safe_filename(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name.strip()
    return name or fallback


def _get_job(job_id: str) -> JobRecord:
    with _jobs_lock:
        record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return record


def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    logs: list[str] | None = None,
    preview_rows: list[list[Any]] | None = None,
    quality_report: dict[str, Any] | None = None,
    files: dict[str, Path] | None = None,
    error: str | None = None,
) -> None:
    with _jobs_lock:
        record = _jobs[job_id]
        if status is not None:
            record.status = status
        if stage is not None:
            record.stage = stage
        if logs:
            record.logs.extend(logs)
        if preview_rows is not None:
            record.preview_rows = preview_rows
        if quality_report is not None:
            record.quality_report = quality_report
        if files is not None:
            record.files = files
        if error is not None:
            record.error = error


def _job_payload(record: JobRecord) -> dict[str, Any]:
    return {
        "job_id": record.job_id,
        "status": record.status,
        "stage": record.stage,
        "logs": record.logs,
        "preview_rows": record.preview_rows,
        "quality_report": record.quality_report,
        "downloads": _download_payload(record),
        "error": record.error,
    }


def _download_payload(record: JobRecord) -> list[dict[str, str]]:
    if record.status != "succeeded":
        return []
    return [
        {
            "kind": kind,
            "label": _download_filename(record, kind),
            "url": f"/api/jobs/{record.job_id}/files/{kind}",
        }
        for kind in ("srt", "vtt")
        if kind in record.files and record.files[kind].exists()
    ]


def _download_filename(record: JobRecord, kind: str) -> str:
    if kind in {"srt", "vtt"}:
        stem = Path(record.audio_filename).stem.strip() or "audio"
        return f"{stem}.{kind}"
    return FILE_LABELS[kind]


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="127.0.0.1", port=port)
