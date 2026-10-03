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
- 依赖：FastAPI、uvicorn、python-multipart、imageio-ffmpeg、mlx-audio，以及字幕分词使用的 jieba、Sudachi、spaCy 和英语模型。
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

字幕分词依赖与历史回放实验固定为相同版本，随 `requirements.txt` 安装。
中文简繁大词典随项目打包；日语词典和英语模型在安装依赖时下载。

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

安装依赖后，单元测试不额外下载对齐模型，也不依赖 ffmpeg：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

2026-10-03 的集成验证：115 项测试通过，21 份历史数据的 84 组风格分割与导出约束检查通过，16 种语言与风格组合的界面同步通过。长视频的 21 份 SRT/VTT 与已复核实验结果完全一致。验证使用历史时间证据，未重新请求对齐模型或逐条试听；详细范围和参数见 [验证记录](reports/subtitle-integration-validation-20261003.json)。

英语切分优化应用后：117 项测试通过，6 份英语长视频的 SRT/VTT 与冻结候选完全一致，其他语言的 64 组回放输出与修改前一致。22 份内容、4 种风格共 88 组回放中，85 组可导出；3 组英语老年风格因现有时长、字数与阅读速度约束被阻断，与修改前相同。当前英语样本通过完整导出流程。详见 [英语应用验证记录](reports/english-segmentation-application-20261003.json)。

真实 Qwen MLX 对齐需要可访问 Metal 的 Apple Silicon 进程、模型和 ffmpeg，并准备实际音频文件。
若报告显示“待复核”，应先检查报告中的低置信区间和时间证据，再决定是否重新生成；程序不会自动把不确定时间轴标记为通过。

## 字幕切分和风格设置

中文使用简繁词典保护词语，日语保护复合词、助词和动词组合，英语使用词语及短语关系，韩语使用词间空格和标点边界。切点从已有 token 中选择，时间由对应 token 的起止值计算。

英语在已识别的句界内建立句法单位，保护短核心短语、介词与补足语、程度修饰等关系。满足时长、字数和阅读速度约束后，全局切分优先保留短核心语法单位，再考虑句界完整和较宽短语连接，最后比较阅读节奏和长度分数。

语言与字幕风格共同决定默认设置。切换任意一项都会重置时长和字数；手动修改后显示“自定义设置”，可点击“恢复当前风格默认”。

| 风格 | 最短时长 | 最长时长 | 单条字符上限：中／日／韩／英 |
| --- | --- | --- | --- |
| YouTube 长视频 | 1.2 秒 | 日英 5 秒、中韩 6.5 秒 | 34／26／38／60 |
| 标准字幕 | 1.2 秒 | 英语 5 秒、其他 6 秒 | 34／34／38／60 |
| 短字幕 | 1 秒 | 英语 5 秒、其他 4.2 秒 | 26／26／29／60 |
| 老年频道 | 1.5 秒 | 英语 5 秒、其他 7 秒 | 29／29／32／60 |

长视频的软目标时长为中文、日语 3 秒，韩语 3.1 秒，英语 3.3 秒。完整语意、阅读速度和硬约束共同决定实际长度。
YouTube 风格中，两端都为完整句子或独立话语边界的短句可低至 0.8 秒；句内片段至少 1.2 秒。手动提高最短时长时，完整短句也遵守提高后的值。最后一条保留音频结尾的既有时长规则。
老年频道的阅读速度上限降低 15%。英语各风格默认最长 5 秒、每条最多 60 字符。切点选择会提前检查最后一条延伸至音频末尾后的时长。

“每条字幕最大字符数”直接约束整条字幕，空白不计入字符数。API 和 CLI 使用 `max_chars_total` / `--max-chars-total`；`max_chars_per_line` 继续作为旧客户端的兼容参数。
