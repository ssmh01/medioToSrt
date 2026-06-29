import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.models import AlignmentResult, AlignmentToken
from autosrt_aligner.pipeline import run_alignment_job


class FakeEngine:
    requires_audio_preprocessing = False

    def align(self, audio_path, cleaned_text, language, logs):
        tokens = []
        visible_index = 0
        for idx, ch in enumerate(cleaned_text.display_text):
            if ch.isspace():
                continue
            start = visible_index * 0.24
            tokens.append(AlignmentToken(ch, start, start + 0.18, idx, idx + 1))
            visible_index += 1
        logs.append("fake alignment complete")
        return AlignmentResult(tokens=tokens, raw={"engine": "fake"}, audio_duration=visible_index * 0.24)


class PipelineTests(unittest.TestCase):
    def test_run_alignment_job_with_fake_engine_exports_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "out"
            result = run_alignment_job(
                audio_path=Path(temp_dir) / "dummy.mp3",
                script_text="这是第一句话。这是第二句话。",
                language="zh",
                subtitle_profile="youtube_long",
                output_dir=out_dir,
                min_duration=0.8,
                max_duration=3.0,
                max_chars_per_line=12,
                generate_vtt=True,
                engine=FakeEngine(),
            )
            self.assertTrue(result.srt_path.exists())
            self.assertTrue(result.vtt_path and result.vtt_path.exists())
            self.assertTrue(result.quality_report_path.exists())
            self.assertTrue(result.alignment_json_path.exists())
            self.assertIn("这是第一句话", result.srt_path.read_text(encoding="utf-8"))
            self.assertEqual(result.quality_report["unaligned_text_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()

