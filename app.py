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


APP_CSS = """
:root {
    --surface: #ffffff;
    --surface-soft: #f7f9fb;
    --surface-muted: #eef3f7;
    --border: #dfe7ee;
    --border-strong: #c8d5df;
    --text: #17212b;
    --text-muted: #647381;
    --accent: #0f766e;
    --accent-soft: #e3f4f1;
    --accent-strong: #115e59;
    --warning: #b7791f;
}

.gradio-container {
    max-width: none !important;
    min-height: 100vh;
    background:
        linear-gradient(180deg, rgba(15, 118, 110, 0.05), rgba(15, 118, 110, 0) 260px),
        #f4f7fa !important;
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.app-frame {
    max-width: 1500px;
    margin: 0 auto;
    padding: 22px 24px 28px;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.brand-mark {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 8px;
    background: #113a44;
    color: #ffffff;
    font-weight: 760;
    letter-spacing: 0;
}

.brand-title {
    color: var(--text);
    font-size: 22px;
    line-height: 1.2;
    font-weight: 760;
    letter-spacing: 0;
}

.brand-subtitle {
    margin-top: 3px;
    color: var(--text-muted);
    font-size: 13px;
}

.header-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 34px;
    padding: 0 12px;
    border: 1px solid #b9ded8;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    font-size: 13px;
    font-weight: 650;
    white-space: nowrap;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--accent);
}

.main-grid {
    gap: 16px;
    align-items: stretch;
}

.tool-panel {
    min-width: 0;
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 18px 48px rgba(23, 33, 43, 0.06);
}

.tool-panel .block {
    border-color: var(--border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.panel-heading h2,
.panel-heading h3 {
    margin: 0 0 12px !important;
    color: var(--text);
    font-size: 15px !important;
    line-height: 1.25 !important;
    font-weight: 760 !important;
    letter-spacing: 0 !important;
}

.input-panel textarea {
    min-height: 126px !important;
}

.audio-compact {
    height: 156px !important;
    min-height: 156px !important;
    overflow: hidden !important;
}

.input-panel label,
.output-panel label,
.workspace-panel label {
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 650 !important;
}

.primary-action,
.primary-action button,
button.primary-action {
    min-height: 44px;
    border: 0 !important;
    border-radius: 8px !important;
    background: var(--accent) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 720 !important;
}

.primary-action:hover,
.primary-action button:hover,
button.primary-action:hover {
    background: var(--accent-strong) !important;
}

.advanced-box {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface-soft) !important;
}

.process-strip {
    display: grid;
    grid-template-columns: 1fr;
    gap: 12px;
    align-items: center;
    margin-bottom: 14px;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: linear-gradient(180deg, #fbfdfe, #f6fafb);
}

.process-label {
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 650;
}

.process-title {
    margin-top: 4px;
    color: var(--text);
    font-size: 16px;
    font-weight: 760;
}

.timeline {
    display: grid;
    grid-template-columns: repeat(4, minmax(58px, 1fr));
    gap: 8px;
}

.timeline-step {
    position: relative;
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    justify-content: center;
    padding: 8px 9px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #ffffff;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 650;
}

.timeline-step::before {
    content: "";
    width: 9px;
    height: 9px;
    flex: 0 0 auto;
    border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-soft);
}

.waveform {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 34px;
    margin-top: 10px;
}

.waveform span {
    display: block;
    width: 5px;
    border-radius: 999px;
    background: #8bb8b2;
}

.waveform span:nth-child(1) { height: 12px; }
.waveform span:nth-child(2) { height: 24px; }
.waveform span:nth-child(3) { height: 18px; }
.waveform span:nth-child(4) { height: 30px; }
.waveform span:nth-child(5) { height: 16px; }
.waveform span:nth-child(6) { height: 27px; }
.waveform span:nth-child(7) { height: 20px; }
.waveform span:nth-child(8) { height: 32px; }
.waveform span:nth-child(9) { height: 14px; }
.waveform span:nth-child(10) { height: 25px; }
.waveform span:nth-child(11) { height: 17px; }
.waveform span:nth-child(12) { height: 28px; }

.preview-table {
    min-height: 520px;
}

.preview-table table {
    font-size: 13px !important;
}

.preview-table thead th {
    background: var(--surface-muted) !important;
    color: #40515f !important;
    font-weight: 720 !important;
}

.preview-table tbody tr:nth-child(even) td {
    background: #fafcfd !important;
}

.log-box textarea {
    min-height: 190px !important;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
    font-size: 12px !important;
}

.download-title {
    margin: 14px 0 8px;
    color: var(--text);
    font-size: 15px;
    font-weight: 760;
}

.download-row {
    gap: 10px;
}

.quality-note {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 10px 0 8px;
    padding: 10px 12px;
    border-radius: 8px;
    background: #fff8e8;
    color: var(--warning);
    font-size: 12px;
    font-weight: 650;
}

.quality-note span:last-child {
    color: #7c5d18;
    white-space: nowrap;
}

@media (max-width: 1100px) {
    .app-frame {
        padding: 16px;
    }

    .app-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .timeline {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""


HEADER_HTML = """
<div class="app-header">
    <div class="brand-lockup">
        <div class="brand-mark">SRT</div>
        <div>
            <div class="brand-title">AI 语音原文案对齐 SRT 工具</div>
            <div class="brand-subtitle">音频转字幕对齐工作台</div>
        </div>
    </div>
    <div class="header-status"><span class="status-dot"></span>本地运行</div>
</div>
"""


PROCESS_HTML = """
<div class="process-strip">
    <div>
        <div class="process-label">当前流程</div>
        <div class="process-title">上传 → 对齐 → 校验 → 导出</div>
        <div class="waveform">
            <span></span><span></span><span></span><span></span><span></span><span></span>
            <span></span><span></span><span></span><span></span><span></span><span></span>
        </div>
    </div>
    <div class="timeline">
        <div class="timeline-step">上传</div>
        <div class="timeline-step">对齐</div>
        <div class="timeline-step">校验</div>
        <div class="timeline-step">导出</div>
    </div>
</div>
"""


QUALITY_NOTE_HTML = """
<div class="quality-note">
    <span>质量报告会在生成后刷新</span>
    <span>覆盖率 / 警告 / 文件</span>
</div>
"""


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

    with gr.Blocks(title="AI 语音原文案对齐 SRT 工具", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-frame"]):
            gr.HTML(HEADER_HTML)
            with gr.Row(elem_classes=["main-grid"]):
                with gr.Column(scale=4, elem_classes=["tool-panel", "input-panel"]):
                    gr.Markdown("## 输入与参数", elem_classes=["panel-heading"])
                    button = gr.Button("开始生成", variant="primary", elem_classes=["primary-action"])
                    audio = gr.Audio(
                        label="上传音频",
                        sources=["upload"],
                        type="filepath",
                        elem_classes=["audio-compact"],
                    )
                    script_file = gr.File(label="上传 TXT 文案", file_types=[".txt"], type="filepath", height=116)
                    script_text = gr.Textbox(label="粘贴原始文案", lines=6)
                    with gr.Row():
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
                    with gr.Accordion("高级参数", open=False, elem_classes=["advanced-box"]):
                        with gr.Row():
                            min_duration = gr.Number(value=1.2, label="每条字幕最短时长")
                            max_duration = gr.Number(value=6.5, label="每条字幕最长时长")
                        max_chars_per_line = gr.Number(value=18, label="字幕切分参考字符数")
                        with gr.Row():
                            generate_vtt = gr.Checkbox(value=True, label="生成 VTT")
                            preserve_punctuation = gr.Checkbox(value=True, label="保留原文标点")

                with gr.Column(scale=7, elem_classes=["tool-panel", "workspace-panel"]):
                    gr.HTML(PROCESS_HTML)
                    preview = gr.Dataframe(
                        headers=["序号", "开始", "结束", "字幕文本", "时长", "异常"],
                        label="字幕预览",
                        wrap=True,
                        interactive=False,
                        elem_classes=["preview-table"],
                    )

                with gr.Column(scale=4, elem_classes=["tool-panel", "output-panel"]):
                    gr.Markdown("## 输出与检查", elem_classes=["panel-heading"])
                    logs = gr.Textbox(label="生成日志", lines=9, show_copy_button=True, elem_classes=["log-box"])
                    gr.HTML(QUALITY_NOTE_HTML)
                    report = gr.JSON(label="quality_report.json")
                    gr.HTML('<div class="download-title">下载文件</div>')
                    with gr.Row(elem_classes=["download-row"]):
                        srt = gr.File(label="SRT")
                        vtt = gr.File(label="VTT")
                    with gr.Row(elem_classes=["download-row"]):
                        quality = gr.File(label="quality_report.json")
                        alignment = gr.File(label="alignment.json")

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
