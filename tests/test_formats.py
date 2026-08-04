import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.formats import export_srt, export_vtt, srt_timestamp, vtt_timestamp
from autosrt_aligner.models import SubtitleCue


class FormatTests(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(srt_timestamp(65.432), "00:01:05,432")
        self.assertEqual(vtt_timestamp(65.432), "00:01:05.432")

    def test_export_srt_and_vtt(self):
        cues = [SubtitleCue(1, 0.0, 1.25, "第一句。", 0, 4)]
        srt = export_srt(cues, language="zh")
        vtt = export_vtt(cues)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,250", srt)
        self.assertIn("\n第一句\n", srt)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.250", vtt)
        self.assertIn("第一句。", vtt)

    def test_export_srt_selects_policy_by_language(self):
        cases = {
            "zh": ("第一句。", "第一句"),
            "zh-TW": ("第一句。", "第一句"),
            "ja": ("最初の文。", "最初の文"),
            "ko": ("첫 문장.", "첫 문장"),
            "en": ("First sentence.", "First sentence."),
        }

        for language, (source, expected) in cases.items():
            with self.subTest(language=language):
                cue = SubtitleCue(1, 0.0, 1.25, source, 0, len(source))
                srt = export_srt([cue], language=language)
                self.assertIn(f"\n{expected}\n", srt)

    def test_export_srt_preserves_semantic_and_closing_punctuation(self):
        cues = [
            SubtitleCue(1, 0.0, 1.25, "他说完了。」", 0, 6),
            SubtitleCue(2, 1.25, 2.5, "真的吗？」", 6, 11),
            SubtitleCue(3, 2.5, 3.75, "太好了！", 11, 15),
            SubtitleCue(4, 3.75, 5.0, "等一下……", 15, 20),
            SubtitleCue(5, 5.0, 6.25, "等一下...", 20, 25),
            SubtitleCue(6, 6.25, 7.5, "他说完了：", 25, 30),
            SubtitleCue(7, 7.5, 8.75, "他说完了；", 30, 35),
        ]
        srt = export_srt(cues, language="zh")

        self.assertIn("\n他说完了」\n\n", srt)
        self.assertIn("\n真的吗？」\n\n", srt)
        self.assertIn("\n太好了！\n\n", srt)
        self.assertIn("\n等一下……\n\n", srt)
        self.assertIn("\n等一下...\n\n", srt)
        self.assertIn("\n他说完了：\n\n", srt)
        self.assertIn("\n他说完了；\n", srt)

    def test_export_srt_cleans_each_visible_line_and_keeps_internal_punctuation(self):
        cues = [
            SubtitleCue(1, 0.0, 1.25, "第一行。\n第二行？", 0, 8),
            SubtitleCue(2, 1.25, 2.5, "第一句，第二句。", 8, 15),
            SubtitleCue(3, 2.5, 3.75, "中间？可以", 15, 20),
        ]
        srt = export_srt(cues, language="zh")

        self.assertIn("\n第一行\n第二行？\n\n", srt)
        self.assertIn("\n第一句，第二句\n\n", srt)
        self.assertIn("\n中间？可以\n", srt)

    def test_export_srt_can_preserve_source_punctuation(self):
        cues = [SubtitleCue(1, 0.0, 1.25, "第一句。", 0, 4)]
        srt = export_srt(cues, strip_trailing_punctuation=False)
        self.assertIn("\n第一句。\n", srt)


if __name__ == "__main__":
    unittest.main()
