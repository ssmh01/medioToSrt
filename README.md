# 全自动 AI 语音原文案对齐 SRT 工具

这是一个本地运行的 FastAPI Web 工具。它把“已经生成好的 AI 语音”和“原始文案”做 forced alignment，导出适合 YouTube 使用的 `output.srt`、可选 `output.vtt`、`alignment.json` 和 `quality_report.json`。

核心约束：

- 原始文案是最终字幕文本标准答案。
- 输入文案必须是纯正文；不要把 SRT 序号和时间码混入 forced alignment 文案。
- Qwen3-ForcedAligner 只负责提供时间轴，不用于改写、删减或补充文案。
- 对齐 token 必须严格映射回原文，缺字、错字、跳过原文或时间证据冲突都会停止导出。
- 导出前会生成 `alignment.json` 和 `quality_report.json`；质量门禁未通过时只保留诊断文件，不生成伪造的 SRT。
- V2 默认先使用整段 forced alignment 保留全局上下文，整段无法严格映射时才使用带重叠的音频分块，并对重叠证据去重、检查时间倒退。

## 环境要求

- Python 3.10+，推荐使用 Codex 自带 Python 3.12。
- 推荐系统安装 `ffmpeg` 和 `ffprobe`；如果没有系统 ffmpeg，项目会使用 `imageio-ffmpeg` 提供的本地 ffmpeg fallback。
- 依赖：FastAPI、uvicorn、python-multipart、imageio-ffmpeg、mlx-audio。
- 必须在能访问 Apple GPU/Metal 的 Apple Silicon 桌面进程中运行 Qwen3-ForcedAligner。

macOS 如果已有 Homebrew，也可以安装系统 ffmpeg：

```bash
brew install ffmpeg
```

## 安装

```bash
cd /Users/xieyulong/Documents/Codex/语音srt生成项目
/Users/xieyulong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

在 Apple Silicon 上，`pip install -r requirements.txt` 会安装 MLX 适配器；首次运行时
会从 Hugging Face 下载 `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`。
本工具固定使用 Qwen MLX。若当前进程无法访问 Metal，任务会明确失败，不会切换到其他对齐引擎。

## 启动网页

```bash
source .venv/bin/activate
PYTHONPATH=src python app.py
```

默认地址：

```text
http://127.0.0.1:7860
```

## CLI 用法

```bash
source .venv/bin/activate
PYTHONPATH=src python -m autosrt_aligner.cli \
  --audio input.mp3 \
  --text script.txt \
  --language zh \
  --profile youtube_long \
  --out-dir outputs \
  --engine qwen-mlx \
  --vtt
```

## 当前 MVP 范围

已包含：

- FastAPI + 原生 HTML/CSS/JS 网页 UI。
- 异步任务提交、状态轮询、日志、质量报告和下载列表。
- 音频上传、txt 上传或文本粘贴。
- `zh` / `zh-TW` / `ja` / `en` / `ko`，必须手动选择语言。
- Qwen3-ForcedAligner/MLX 已知原文对齐后端；中文（含繁体）、日语、韩语、英语均
  走同一套严格原文映射和质量门禁。
- 基于 token 时间证据的语言分段：中文、日语、韩语、英语分别处理自然边界。
- SRT/VTT 导出、字幕预览、质量报告、alignment JSON。

未包含：

- 批量任务。
- LLM 断句辅助。
- WhisperX fallback。
- 手动时间轴编辑。
- 历史记录和账号系统。

## 测试

单元测试不下载模型，也不依赖 ffmpeg：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

真实 Qwen MLX 对齐需要可访问 Metal 的 Apple Silicon 进程、模型和 ffmpeg，并准备实际音频文件。
若报告显示“待复核”，应先检查报告中的低置信区间和时间证据，再决定是否重新生成；程序不会自动把不确定时间轴标记为通过。
