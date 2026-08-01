import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.chunking import plan_audio_chunks
from autosrt_aligner.errors import AlignmentError, ExportValidationError
from autosrt_aligner.models import (
    AlignmentResult,
    AlignmentToken,
    AudioChunk,
    ChunkAlignment,
    CleanedText,
    SubtitleCue,
)
from autosrt_aligner.pipeline_v2 import run_alignment_job_v2
from autosrt_aligner.pipeline_v2 import (
    _align_chunk_with_retry,
    _align_source,
    _chunk_alignment,
)
from autosrt_aligner.quality_v2 import _timing_issues, build_v2_quality_report
from autosrt_aligner.reconcile import ReconcileReport, reconcile_chunk_alignments
from autosrt_aligner.segmenter_v2 import segment_cues
from autosrt_aligner.text import build_source_document, map_tokens_to_source_strict
from autosrt_aligner.profiles import resolve_profile


class V2CoreTests(unittest.TestCase):
    def test_strict_mapping_rejects_unmatched_engine_text(self):
        source = build_source_document("这是第一句。", "zh")
        with self.assertRaises(AlignmentError):
            map_tokens_to_source_strict(
                [AlignmentToken("这是", 0.0, 0.4), AlignmentToken("错误", 0.4, 0.8)],
                source,
            )

    def test_strict_mapping_restores_chunk_source_offset(self):
        full = build_source_document("前置文本。这里是正文。", "zh")
        offset = full.display_text.index("这里")
        chunk = build_source_document(full.display_text[offset:], "zh")
        mapped = map_tokens_to_source_strict(
            [AlignmentToken("这里", 0.0, 0.4), AlignmentToken("是正文", 0.4, 1.0), AlignmentToken("。", 1.0, 1.1)],
            chunk,
            source_offset=offset,
            chunk_id="chunk-001",
        )
        self.assertEqual(mapped[0].start_char, offset)
        self.assertEqual(mapped[-1].end_char, len(full.display_text))
        self.assertEqual(mapped[0].chunk_id, "chunk-001")

    def test_chunk_planner_covers_long_source_with_audio_overlap(self):
        text = "这是一个很长的中文段落，用来验证分块规划。" * 80
        source = build_source_document(text, "zh")
        chunks = plan_audio_chunks(source, 180.0, target_seconds=30.0, overlap_seconds=1.0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].source_start, 0)
        self.assertEqual(chunks[-1].source_end, len(source.display_text))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLess(previous.audio_start, current.audio_start)
            self.assertGreaterEqual(previous.audio_end, current.audio_start)
            self.assertLessEqual(previous.audio_end, 180.0)
            self.assertLessEqual(previous.source_start, current.source_start)
            self.assertGreaterEqual(previous.source_end, current.source_start)
            self.assertEqual(previous.core_source_end, current.core_source_start)
            self.assertEqual(previous.core_audio_end, current.core_audio_start)
        self.assertEqual(chunks[0].core_source_start, 0)
        self.assertEqual(chunks[-1].core_source_end, len(source.display_text))
        self.assertEqual(chunks[0].core_audio_start, 0.0)
        self.assertEqual(chunks[-1].core_audio_end, 180.0)

    def test_reconciler_prefers_complete_boundary_token_without_losing_coverage(self):
        source = build_source_document("清楚", "zh")
        selected, report = reconcile_chunk_alignments(
            [
                ChunkAlignment(
                    AudioChunk(
                        "chunk-001",
                        0,
                        1,
                        0.0,
                        1.0,
                        core_source_start=0,
                        core_source_end=1,
                    ),
                    [AlignmentToken("清", 0.0, 0.2, 0, 1, 0.8, "chunk-001")],
                    "test",
                ),
                ChunkAlignment(
                    AudioChunk(
                        "chunk-002",
                        0,
                        2,
                        0.0,
                        1.0,
                        core_source_start=0,
                        core_source_end=2,
                    ),
                    [AlignmentToken("清楚", 0.2, 0.5, 0, 2, 0.8, "chunk-002")],
                    "test",
                ),
            ],
            source,
        )
        self.assertEqual([token.text for token in selected], ["清楚"])
        self.assertEqual(report.source_token_coverage, 1.0)
        self.assertEqual(report.overlap_count, 0)
        self.assertEqual(report.resolved_overlap_count, 1)
        self.assertTrue(report.publishable)

    def test_reconciler_blocks_crossing_overlap_that_cannot_be_lossless(self):
        source = build_source_document("清楚吧", "zh")
        _, report = reconcile_chunk_alignments(
            [
                ChunkAlignment(
                    AudioChunk(
                        "chunk-001",
                        0,
                        2,
                        0.0,
                        1.0,
                        core_source_start=0,
                        core_source_end=2,
                    ),
                    [AlignmentToken("清楚", 0.0, 0.4, 0, 2, 0.9, "chunk-001")],
                    "test",
                ),
                ChunkAlignment(
                    AudioChunk(
                        "chunk-002",
                        1,
                        3,
                        0.0,
                        1.0,
                        core_source_start=1,
                        core_source_end=3,
                    ),
                    [AlignmentToken("楚吧", 0.4, 0.8, 1, 3, 0.9, "chunk-002")],
                    "test",
                ),
            ],
            source,
        )
        self.assertGreater(report.overlap_count, 0)
        self.assertLess(report.source_token_coverage, 1.0)
        self.assertTrue(report.uncovered_source_ranges)
        self.assertFalse(report.publishable)

    def test_retry_records_effective_expanded_audio_window(self):
        source = build_source_document("这是原文", "zh")
        cleaned = CleanedText(
            display_text=source.display_text,
            align_text=source.align_text,
            align_to_display=list(source.align_to_display),
        )
        chunk = AudioChunk(
            "chunk-001",
            0,
            len(source.display_text),
            10.0,
            20.0,
            core_source_start=0,
            core_source_end=len(source.display_text),
            core_audio_start=10.0,
            core_audio_end=20.0,
        )
        calls = []

        def align_chunk(audio_path, cleaned_text, language, attempt_chunk, work_dir, logs, attempt_id):
            calls.append((attempt_chunk.audio_start, attempt_chunk.audio_end))
            if len(calls) == 1:
                return AlignmentResult(
                    tokens=[AlignmentToken("错误", 10.0, 10.2)],
                    raw={"engine": "retry-fixture"},
                )
            return AlignmentResult(
                tokens=[AlignmentToken("这是原文", attempt_chunk.audio_start, attempt_chunk.audio_start + 0.8)],
                raw={"engine": "retry-fixture"},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            mapped, result = _align_chunk_with_retry(
                chunk_aligner=align_chunk,
                align_audio_path=Path("/missing/audio.wav"),
                cleaned=cleaned,
                chunk=chunk,
                chunk_source=source,
                audio_duration=60.0,
                language="zh",
                work_dir=Path(temp_dir),
                logs=[],
                attempt_prefix="001",
            )
        alignment = _chunk_alignment(chunk, mapped, result)
        self.assertEqual(calls, [(10.0, 20.0), (6.0, 24.0)])
        self.assertEqual(alignment.attempts, 2)
        self.assertEqual(alignment.effective_audio_start, 6.0)
        self.assertEqual(alignment.effective_audio_end, 24.0)

    def test_timing_validation_uses_effective_retry_window(self):
        source = build_source_document("原文", "zh")
        chunk = AudioChunk("chunk-001", 0, 2, 10.0, 20.0)
        token = AlignmentToken("原文", 23.5, 23.9, 0, 2, 0.9, "chunk-001")
        alignment = ChunkAlignment(
            chunk,
            [token],
            "test",
            effective_audio_start=6.0,
            effective_audio_end=24.0,
        )
        self.assertNotIn(
            "token 时间超出所属音频分块范围",
            _timing_issues(
                [token],
                source.display_text,
                [chunk],
                60.0,
                alignments=[alignment],
            ),
        )
        token_outside = AlignmentToken("原文", 24.5, 24.9, 0, 2, 0.9, "chunk-001")
        self.assertIn(
            "token 时间超出所属音频分块范围",
            _timing_issues(
                [token_outside],
                source.display_text,
                [chunk],
                60.0,
                alignments=[alignment],
            ),
        )

    def test_quality_report_distinguishes_text_from_token_time_evidence(self):
        source = build_source_document("清楚", "zh")
        cues = [SubtitleCue(1, 0.0, 1.5, "清楚", 0, 2)]
        chunk = AudioChunk("chunk-001", 0, 2, 0.0, 2.0)
        reconcile = ReconcileReport(
            source_token_coverage=0.5,
            duplicate_token_count=0,
            overlap_count=0,
            time_reversal_count=0,
            low_confidence_count=0,
            issues=("source token 覆盖率不足: 0.500",),
            uncovered_source_ranges=((1, 2),),
        )
        report = build_v2_quality_report(
            source,
            [AlignmentToken("清", 0.0, 0.4, 0, 1, 0.9, "chunk-001")],
            cues,
            [chunk],
            reconcile,
            resolve_profile("youtube_long", "zh"),
            audio_duration=2.0,
        )
        self.assertIn("原文 token 时间证据未完整覆盖", report["warnings"])
        self.assertNotIn("字幕文字未能连续覆盖原文", report["warnings"])
        self.assertEqual(report["text_integrity_status"], "pass")
        self.assertEqual(report["token_evidence_status"], "fail")
        self.assertEqual(report["uncovered_source_ranges"][0]["text"], "楚")

    def test_reconciler_deduplicates_overlap_evidence_and_reports_clean_result(self):
        source = build_source_document("这是第一句。", "zh")
        tokens = [
            AlignmentToken("这是", 0.0, 0.4, 0, 2, 0.8, "chunk-001", "word"),
            AlignmentToken("第一句", 0.4, 1.0, 2, 5, 0.8, "chunk-001", "word"),
            AlignmentToken("。", 1.0, 1.1, 5, 6, 0.8, "chunk-001", "punctuation"),
        ]
        duplicate = AlignmentToken("第一句", 0.45, 0.95, 2, 5, 0.9, "chunk-002", "word")
        chunk = AudioChunk(
            "chunk-001",
            0,
            len(source.display_text),
            0.0,
            1.2,
            core_source_start=0,
            core_source_end=len(source.display_text),
        )
        chunk2 = AudioChunk(
            "chunk-002",
            2,
            len(source.display_text),
            0.0,
            1.2,
            0.2,
            0.0,
            core_source_start=5,
            core_source_end=len(source.display_text),
        )
        selected, report = reconcile_chunk_alignments(
            [
                ChunkAlignment(chunk, tokens, "test", "fixture", 0.8),
                ChunkAlignment(chunk2, [duplicate], "test", "fixture", 0.9),
            ],
            source,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(report.duplicate_token_count, 1)
        self.assertEqual(report.overlap_count, 0)
        self.assertTrue(report.publishable)
        self.assertEqual(selected[1].confidence, 0.9)

    def test_reconciler_does_not_treat_missing_confidence_as_verified(self):
        source = build_source_document("这是原文。", "zh")
        chunk = AudioChunk("chunk-001", 0, len(source.display_text), 0.0, 1.0)
        tokens = [
            AlignmentToken("这是原文", 0.0, 0.8, 0, 4),
            AlignmentToken("。", 0.8, 0.9, 4, 5),
        ]
        _, report = reconcile_chunk_alignments(
            [ChunkAlignment(chunk, tokens, "test", "fixture")],
            source,
        )
        self.assertEqual(report.missing_confidence_count, 2)
        self.assertFalse(report.publishable)
        self.assertTrue(any("缺少置信度" in issue for issue in report.issues))

    def test_segmenter_uses_evidence_times_and_preserves_english_words(self):
        text = "There's a quiet house. Somebody waits outside, and nobody answers."
        source = build_source_document(text, "en")
        tokens = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            start = index * 0.08
            tokens.append(AlignmentToken(char, start, start + 0.06, index, index + 1, 0.9))
        profile = resolve_profile("youtube_long", "en", min_duration=0.8, max_duration=3.0, max_chars_per_line=42)
        cues = segment_cues(source, tokens, profile)
        self.assertGreaterEqual(len(cues), 2)
        self.assertTrue(all("\n" not in cue.text for cue in cues))
        self.assertTrue(all(previous.end <= current.start for previous, current in zip(cues, cues[1:])))
        self.assertEqual(cues[-1].end, tokens[-1].end)
        for cue in cues[:-1]:
            self.assertFalse(
                text[cue.end_char - 1 : cue.end_char].isalnum()
                and text[cue.end_char : cue.end_char + 1].isalnum()
            )

    def test_segmenter_respects_four_language_boundary_rules(self):
        cases = [
            ("zh", "母亲把钥匙放在桌上，然后慢慢走出房间。窗外开始下雨了。", "把钥匙"),
            ("ja", "母は鍵を机の上に置いて、静かに部屋を出ました。外では雨が降り始めました。", "鍵を"),
            ("ko", "어머니는 열쇠를 책상 위에 놓고 조용히 방을 나갔습니다. 밖에는 비가 내리기 시작했습니다.", "열쇠를"),
        ]
        for language, text, forbidden_fragment in cases:
            with self.subTest(language=language):
                source = build_source_document(text, language)
                tokens = []
                visible = 0
                for index, char in enumerate(text):
                    if char.isspace():
                        continue
                    start = visible * 0.18
                    tokens.append(AlignmentToken(char, start, start + 0.14, index, index + 1, 0.9))
                    visible += 1
                profile = resolve_profile(
                    "youtube_long",
                    language,
                    min_duration=0.8,
                    max_duration=3.0,
                )
                cues = segment_cues(source, tokens, profile)
                self.assertTrue(validate_text(cues, text))
                forbidden_start = text.index(forbidden_fragment)
                for cue in cues[:-1]:
                    boundary = cue.end_char
                    self.assertFalse(forbidden_start < boundary < forbidden_start + len(forbidden_fragment))
                    if language == "ko":
                        previous_index = boundary - 1
                        while previous_index >= 0 and text[previous_index].isspace():
                            previous_index -= 1
                        has_space = any(
                            char.isspace() for char in text[previous_index + 1 : boundary]
                        )
                        self.assertTrue(
                            has_space
                            or text[previous_index : previous_index + 1] in "。!?"
                        )

    def test_v2_pipeline_blocks_unmapped_tokens(self):
        class BadEngine:
            requires_audio_preprocessing = False

            def align(self, audio_path, cleaned_text, language, logs):
                return AlignmentResult(
                    tokens=[AlignmentToken("不是原文", 0.0, 1.0)],
                    raw={"engine": "bad-fixture"},
                    audio_duration=1.0,
                    language=language,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ExportValidationError) as context:
                run_alignment_job_v2(
                    Path("/missing/audio.wav"),
                    "这是原文。",
                    language="zh",
                    generate_vtt=False,
                    output_dir=temp_dir,
                    engine=BadEngine(),
                )
            self.assertTrue((Path(temp_dir) / "quality_report.json").exists())
        self.assertIn("token", str(context.exception))

    def test_v2_pipeline_blocks_silent_missing_tail_instead_of_filling_it(self):
        class IncompleteEngine:
            requires_audio_preprocessing = False

            def align(self, audio_path, cleaned_text, language, logs):
                half = max(1, len(cleaned_text.display_text) // 2)
                tokens = [
                    AlignmentToken(char, index * 0.2, index * 0.2 + 0.12, index, index + 1, 0.9)
                    for index, char in enumerate(cleaned_text.display_text[:half])
                    if not char.isspace()
                ]
                return AlignmentResult(
                    tokens=tokens,
                    raw={"engine": "incomplete-fixture"},
                    audio_duration=10.0,
                    language=language,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ExportValidationError):
                run_alignment_job_v2(
                    Path("/missing/audio.wav"),
                    "这是前半段内容，后半段绝不能被伪造填充。",
                    language="zh",
                    generate_vtt=False,
                    output_dir=temp_dir,
                    engine=IncompleteEngine(),
                )
            report = json.loads(
                (Path(temp_dir) / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["text_status"], "unknown")
            self.assertLess(report["source_token_coverage"], 1.0)
            self.assertFalse((Path(temp_dir) / "output.srt").exists())

    def test_chunk_engine_is_recovery_path_after_full_mapping_failure(self):
        class ChunkFixtureEngine:
            requires_audio_preprocessing = False

            def __init__(self):
                self.chunk_calls = 0

            def align(self, audio_path, cleaned_text, language, logs):
                return AlignmentResult(
                    tokens=[AlignmentToken("不在原文", 0.0, 1.0)],
                    raw={"engine": "chunk-fixture"},
                    audio_duration=180.0,
                    language=language,
                )

            def align_chunk(
                self,
                audio_path,
                cleaned_text,
                language,
                chunk,
                work_dir,
                logs,
                attempt_id,
            ):
                self.chunk_calls += 1
                tokens = []
                visible = 0
                for index, char in enumerate(cleaned_text.display_text):
                    if char.isspace():
                        continue
                    start = chunk.audio_start + visible * 0.18
                    tokens.append(
                        AlignmentToken(char, start, start + 0.12, confidence=0.9)
                    )
                    visible += 1
                return AlignmentResult(
                    tokens=tokens,
                    raw={"engine": "chunk-fixture"},
                    audio_duration=chunk.audio_end - chunk.audio_start,
                    language=language,
                )

        engine = ChunkFixtureEngine()
        source = build_source_document("这是一个用于分块恢复路径的中文句子。" * 100, "zh")
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks, alignments, duration = _align_source(
                engine=engine,
                source=source,
                align_audio_path=Path("/missing/audio.wav"),
                audio_duration=180.0,
                work_dir=Path(temp_dir),
                language="zh",
                preserve_punctuation=True,
                logs=[],
            )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks), len(alignments))
        self.assertEqual(engine.chunk_calls, len(chunks))
        self.assertEqual(duration, 180.0)

    def test_v2_pipeline_uses_source_text_and_evidence_times(self):
        class FixtureEngine:
            requires_audio_preprocessing = False

            def align(self, audio_path, cleaned_text, language, logs):
                tokens = []
                visible = 0
                for index, char in enumerate(cleaned_text.display_text):
                    if char.isspace():
                        continue
                    start = visible * 0.22
                    tokens.append(
                        AlignmentToken(char, start, start + 0.16, index, index + 1, 0.9)
                    )
                    visible += 1
                return AlignmentResult(
                    tokens=tokens,
                    raw={"engine": "fixture"},
                    audio_duration=visible * 0.22,
                    language=language,
                )

        script = "母亲把钥匙放在桌上，认真说出了自己的决定。窗外的雨停了，巷子里慢慢亮了起来。"
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_alignment_job_v2(
                Path("/missing/audio.wav"),
                script,
                language="zh",
                generate_vtt=False,
                output_dir=temp_dir,
                engine=FixtureEngine(),
            )
            self.assertEqual(result.quality_report["schema_version"], "2")
            self.assertEqual(result.quality_report["text_status"], "pass")
            self.assertEqual(result.quality_report["publish_status"], "pass")
            self.assertEqual(result.quality_report["source_token_coverage"], 1.0)
            self.assertEqual(result.quality_report["audio_duration"], 8.36)
            self.assertTrue(result.srt_path.exists())
            self.assertIn("决定。", result.srt_path.read_text(encoding="utf-8"))
            alignment = json.loads(result.alignment_json_path.read_text(encoding="utf-8"))
            self.assertEqual(alignment["chunk_evidence"][0]["token_count"], 38)
            self.assertEqual(alignment["chunks"][0]["core_source_start"], 0)
            self.assertEqual(alignment["chunk_evidence"][0]["effective_audio_start"], 0.0)
            self.assertEqual(alignment["chunk_evidence"][0]["effective_audio_end"], 8.36)

    def test_v2_pipeline_supports_four_required_languages(self):
        cases = {
            "zh": "母亲把钥匙放在桌上。窗外的雨停了。",
            "ja": "母は鍵を机の上に置きました。外の雨が止みました。",
            "ko": "어머니는 열쇠를 책상 위에 놓았습니다. 밖의 비가 그쳤습니다.",
            "en": "Mother put the key on the table. The rain outside stopped.",
        }

        class MultilingualFixtureEngine:
            requires_audio_preprocessing = False

            def align(self, audio_path, cleaned_text, language, logs):
                tokens = []
                visible = 0
                for index, char in enumerate(cleaned_text.display_text):
                    if char.isspace():
                        continue
                    start = visible * 0.2
                    tokens.append(
                        AlignmentToken(char, start, start + 0.14, index, index + 1, 0.95)
                    )
                    visible += 1
                return AlignmentResult(
                    tokens=tokens,
                    raw={"engine": "multilingual-fixture"},
                    audio_duration=visible * 0.2,
                    language=language,
                )

        for language, script in cases.items():
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temp_dir:
                result = run_alignment_job_v2(
                    Path("/missing/audio.wav"),
                    script,
                    language=language,
                    generate_vtt=False,
                    output_dir=temp_dir,
                    engine=MultilingualFixtureEngine(),
                )
                self.assertEqual(result.quality_report["publish_status"], "pass")
                self.assertEqual(result.quality_report["text_status"], "pass")


def validate_text(cues, expected):
    return "".join("".join(cue.text.split()) for cue in cues) == "".join(expected.split())


if __name__ == "__main__":
    unittest.main()
