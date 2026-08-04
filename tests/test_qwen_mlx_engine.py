import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.engines import factory
from autosrt_aligner.engines.qwen_mlx import QwenMlxEngine
import autosrt_aligner.engines.qwen_mlx as qwen_mlx
from autosrt_aligner.errors import AlignmentError, DependencyError, InputError
from autosrt_aligner.models import AlignmentToken, AudioChunk, ChunkAlignment, CleanedText
from autosrt_aligner.reconcile import reconcile_chunk_alignments
from autosrt_aligner.text import build_source_document


class FakeQwenAligner:
    def __init__(self):
        self.calls = []

    def generate(self, audio_path, *, text, language):
        self.calls.append((audio_path, text, language))
        return {
            "items": [
                {"text": "清", "start_time": 0.1, "end_time": 0.3},
                {"text": "楚", "start_time": 0.3, "end_time": 0.6},
            ]
        }


class QwenMlxEngineTests(unittest.TestCase):
    def setUp(self):
        self.ffmpeg_patcher = patch.object(
            qwen_mlx,
            "ensure_ffmpeg_on_path",
            return_value="ffmpeg",
        )
        self.ffmpeg_patcher.start()
        self.aligner = FakeQwenAligner()
        self.cleaned = CleanedText(
            display_text="清楚",
            align_text="清楚",
            align_to_display=[0, 1],
        )

    def tearDown(self):
        self.ffmpeg_patcher.stop()

    def test_supported_languages_are_mapped_without_changing_source_text(self):
        engine = QwenMlxEngine(loader=lambda _name: self.aligner)
        expected = {
            "zh": "Chinese",
            "zh-TW": "Chinese",
            "ja": "Japanese",
            "en": "English",
            "ko": "Korean",
        }

        for language, qwen_language in expected.items():
            result = engine.align(Path("audio.wav"), self.cleaned, language, [])
            self.assertEqual(result.raw["qwen_language"], qwen_language)
            self.assertEqual(self.aligner.calls[-1][1], "清楚")
            self.assertEqual(self.aligner.calls[-1][2], qwen_language)

        self.assertEqual([token.text for token in result.tokens], ["清", "楚"])
        self.assertFalse(result.raw["confidence_available"])
        self.assertEqual(result.tokens[0].unit_type, "character")

    def test_chunk_alignment_restores_global_offset(self):
        engine = QwenMlxEngine(loader=lambda _name: self.aligner)
        chunk = AudioChunk("chunk-002", 0, 2, 12.5, 15.0)
        with patch.object(
            qwen_mlx,
            "clip_audio_segment",
            return_value=Path("clip.wav"),
        ):
            logs = []
            result = engine.align_chunk(
                Path("audio.wav"),
                self.cleaned,
                "zh-TW",
                chunk,
                Path(tempfile.mkdtemp()),
                logs,
                "002-1",
            )

        self.assertEqual(result.tokens[0].start, 12.6)
        self.assertEqual(result.tokens[1].end, 13.1)
        self.assertEqual(result.tokens[0].chunk_id, "chunk-002")
        self.assertTrue(any("裁剪窗口" in log and "实际" in log for log in logs))

    def test_chunk_retry_uses_wider_qwen_context(self):
        self.assertEqual(QwenMlxEngine.retry_padding_seconds, (0.0, 8.0, 16.0))
        self.assertEqual(QwenMlxEngine.chunk_overlap_seconds, 16.0)
        self.assertEqual(QwenMlxEngine.trailing_context_seconds, 8.0)

    def test_unsupported_language_is_rejected(self):
        engine = QwenMlxEngine(loader=lambda _name: self.aligner)
        with self.assertRaises(AlignmentError):
            engine.align(Path("audio.wav"), self.cleaned, "fr", [])

    def test_missing_optional_dependency_is_explicit(self):
        engine = QwenMlxEngine()
        with patch.dict(sys.modules, {"mlx_audio": None, "mlx_audio.stt": None}):
            with self.assertRaises(DependencyError):
                engine.align(Path("audio.wav"), self.cleaned, "zh", [])


class AlignmentEngineFactoryTests(unittest.TestCase):
    def test_only_qwen_engine_is_supported(self):
        self.assertIsInstance(factory.create_alignment_engine("qwen-mlx"), QwenMlxEngine)
        self.assertIsInstance(factory.create_alignment_engine("auto"), QwenMlxEngine)
        with self.assertRaises(InputError):
            factory.create_alignment_engine("stable-ts")

    def test_status_never_reports_stable_ts_as_backend(self):
        with patch.object(factory, "_qwen_mlx_available", return_value=False):
            status = factory.alignment_engine_status()

        self.assertEqual(status["selected"], "qwen-mlx")
        self.assertEqual(status["choices"], ["qwen-mlx"])
        self.assertEqual(status["backend"], "qwen-mlx")
        self.assertFalse(status["qwen_mlx_available"])

    def test_qwen_without_confidence_is_reported_but_not_mislabeled_as_low_confidence(self):
        source = build_source_document("清楚", "zh")
        chunk = AudioChunk("chunk-001", 0, 2, 0.0, 1.0)
        alignment = ChunkAlignment(
            chunk=chunk,
            tokens=[
                AlignmentToken("清", 0.1, 0.3, 0, 1, chunk_id="chunk-001"),
                AlignmentToken("楚", 0.3, 0.6, 1, 2, chunk_id="chunk-001"),
            ],
            engine="qwen3-forced-aligner-mlx",
            confidence_available=False,
        )
        _, report = reconcile_chunk_alignments([alignment], source)

        self.assertEqual(report.missing_confidence_count, 2)
        self.assertEqual(report.low_confidence_count, 0)
        self.assertFalse(report.confidence_available)
        self.assertFalse(any("缺少置信度" in issue for issue in report.issues))

    def test_exact_qwen_boundary_duplicates_can_choose_monotonic_candidate(self):
        source = build_source_document("甲乙丙", "zh")
        first = AudioChunk(
            "chunk-a",
            0,
            2,
            0.0,
            2.0,
            core_source_start=0,
            core_source_end=2,
        )
        second = AudioChunk(
            "chunk-b",
            1,
            3,
            0.0,
            2.0,
            core_source_start=1,
            core_source_end=3,
        )
        _, report = reconcile_chunk_alignments(
            [
                ChunkAlignment(
                    first,
                    [
                        AlignmentToken("甲", 0.0, 0.5, 0, 1, chunk_id="chunk-a"),
                        AlignmentToken("乙", 2.0, 2.5, 1, 2, chunk_id="chunk-a"),
                    ],
                    "qwen3-forced-aligner-mlx",
                ),
                ChunkAlignment(
                    second,
                    [
                        AlignmentToken("乙", 1.0, 1.5, 1, 2, chunk_id="chunk-b"),
                        AlignmentToken("丙", 1.5, 2.0, 2, 3, chunk_id="chunk-b"),
                    ],
                    "qwen3-forced-aligner-mlx",
                ),
            ],
            source,
        )

        self.assertEqual(report.source_token_coverage, 1.0)
        self.assertEqual(report.time_reversal_count, 0)


if __name__ == "__main__":
    unittest.main()
