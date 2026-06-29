"""Gradio web UI entrypoint for the AutoSRT aligner MVP."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from autosrt_aligner.errors import AutosrtError, InputError
from autosrt_aligner.pipeline import run_alignment_job
from autosrt_aligner.profiles import PROFILE_LABELS, SUPPORTED_LANGUAGES


PROFILE_LABEL_TO_KEY = {label: key for key, label in PROFILE_LABELS.items()}
LANGUAGES = ["auto", "zh", "zh-TW", "ja", "en"]


def generate_from_ui(
    audio_file: str | None,
    script_file: str | None,
    script_text: str | None,
    language: str,
    profile_label: str,
    min_duration: float,
    max_duration: float,
    max_chars_per_line: int,
    generate_vtt: bool,
    preserve_punctuation: bool,
) -> tuple[list[list[Any]], dict[str, Any], str | None, str | None, str | None, str | None, str]:
    if not audio_file:
        return _error_result("请先上传音频文件")
    try:
        text = _read_script_text(script_file, script_text)
        output_dir = tempfile.mkdtemp(prefix="autosrt_ui_")
        result = run_alignment_job(
            audio_path=audio_file,
            script_text=text,
            language=language,
            subtitle_profile=PROFILE_LABEL_TO_KEY[profile_label],
            output_dir=output_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            max_chars_per_line=int(max_chars_per_line),
            generate_vtt=generate_vtt,
            preserve_punctuation=preserve_punctuation,
        )
        return (
            result.preview_rows(),
            result.quality_report,
            str(result.srt_path),
            str(result.vtt_path) if result.vtt_path else None,
            str(result.quality_report_path),
            str(result.alignment_json_path),
            "\n".join(result.logs),
        )
    except AutosrtError as exc:
        return _error_result(str(exc))
    except Exception as exc:  # pragma: no cover - UI safety net
        return _error_result(f"未知错误: {exc}")


def build_app():
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("未安装 Gradio。请先运行 pip install -r requirements.txt。") from exc

    with gr.Blocks(title="AI 语音原文案对齐 SRT 工具") as demo:
        gr.Markdown("# AI 语音原文案对齐 SRT 工具")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(label="上传音频", type="filepath")
                script_file = gr.File(label="上传 txt 文案", file_types=[".txt"], type="filepath")
                script_text = gr.Textbox(label="或粘贴原始文案", lines=12)
                language = gr.Dropdown(
                    choices=[value for value in LANGUAGES if value in SUPPORTED_LANGUAGES],
                    value="auto",
                    label="语言",
                )
                profile = gr.Dropdown(
                    choices=list(PROFILE_LABEL_TO_KEY.keys()),
                    value=PROFILE_LABELS["youtube_long"],
                    label="字幕风格",
                )
                with gr.Accordion("高级参数", open=False):
                    min_duration = gr.Number(value=1.2, label="每条字幕最短时长")
                    max_duration = gr.Number(value=6.5, label="每条字幕最长时长")
                    max_chars_per_line = gr.Number(value=18, label="字幕切分参考字符数")
                    generate_vtt = gr.Checkbox(value=True, label="生成 VTT")
                    preserve_punctuation = gr.Checkbox(value=True, label="保留原文标点")
                button = gr.Button("开始生成", variant="primary")
            with gr.Column(scale=2):
                logs = gr.Textbox(label="日志", lines=10)
                preview = gr.Dataframe(
                    headers=["序号", "start", "end", "字幕文本", "持续时间", "异常"],
                    label="字幕预览",
                    wrap=True,
                )
                report = gr.JSON(label="quality_report.json")
                with gr.Row():
                    srt = gr.File(label="下载 SRT")
                    vtt = gr.File(label="下载 VTT")
                with gr.Row():
                    quality = gr.File(label="下载 quality_report.json")
                    alignment = gr.File(label="下载 alignment.json")

        button.click(
            fn=generate_from_ui,
            inputs=[
                audio,
                script_file,
                script_text,
                language,
                profile,
                min_duration,
                max_duration,
                max_chars_per_line,
                generate_vtt,
                preserve_punctuation,
            ],
            outputs=[preview, report, srt, vtt, quality, alignment, logs],
        )
    return demo


def _read_script_text(script_file: str | None, script_text: str | None) -> str:
    if script_file:
        return Path(script_file).read_text(encoding="utf-8-sig")
    if script_text and script_text.strip():
        return script_text
    raise InputError("请上传 txt 文案或粘贴原始文案")


def _error_result(message: str) -> tuple[list[list[Any]], dict[str, Any], None, None, None, None, str]:
    return [], {"error": message}, None, None, None, None, message


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    build_app().launch(server_name="127.0.0.1", server_port=port)
