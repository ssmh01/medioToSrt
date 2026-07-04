# 全自动 AI 语音原文案对齐 SRT 工具

这是一个本地运行的 FastAPI Web 工具。它把“已经生成好的 AI 语音”和“原始文案”做 forced alignment，导出适合 YouTube 使用的 `output.srt`、可选 `output.vtt`、`alignment.json` 和 `quality_report.json`。

核心约束：

- 原始文案是最终字幕文本标准答案。
- stable-ts 只负责提供时间轴，不用于改写、删减或补充文案。
- 导出前会校验字幕正文是否连续覆盖原文，校验失败会停止导出。

## 环境要求

- Python 3.10+，推荐使用 Codex 自带 Python 3.12。
- 推荐系统安装 `ffmpeg` 和 `ffprobe`；如果没有系统 ffmpeg，项目会使用 `imageio-ffmpeg` 提供的本地 ffmpeg fallback。
- 依赖：FastAPI、uvicorn、python-multipart、stable-ts、imageio-ffmpeg。

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
  --vtt
```

## 当前 MVP 范围

已包含：

- FastAPI + 原生 HTML/CSS/JS 网页 UI。
- 异步任务提交、状态轮询、日志、质量报告和下载列表。
- 音频上传、txt 上传或文本粘贴。
- `zh` / `zh-TW` / `ja` / `en` / `ko`，必须手动选择语言。
- stable-ts forced alignment 引擎封装。
- 规则候选 + 贪心 + validator 的字幕切分。
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

真实 stable-ts 对齐需要先安装依赖和 ffmpeg，并准备实际音频文件。
