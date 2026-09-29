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
            "en": ("First sentence.", "First sentence"),
        }

        for language, (source, expected) in cases.items():
            with self.subTest(language=language):
                cue = SubtitleCue(1, 0.0, 1.25, source, 0, len(source))
                srt = export_srt([cue], language=language)
                self.assertIn(f"\n{expected}\n", srt)

    def test_export_english_keeps_semantic_punctuation_and_cleans_noise(self):
        cues = [
            SubtitleCue(1, 0.0, 1.0, "Don't split well-known words.", 0, 29),
            SubtitleCue(2, 1.0, 2.0, "First, second, and third.", 29, 54),
            SubtitleCue(3, 2.0, 3.0, "Really???", 54, 63),
            SubtitleCue(4, 3.0, 4.0, "Watch out!!!", 63, 75),
            SubtitleCue(5, 4.0, 5.0, "What did you say!?!", 75, 94),
            SubtitleCue(6, 5.0, 6.0, 'He said, "Stop."', 94, 110),
            SubtitleCue(7, 6.0, 7.0, "Wait...", 110, 117),
            SubtitleCue(8, 7.0, 8.0, "I live in the U.S.", 117, 136),
            SubtitleCue(9, 8.0, 9.0, "Ask Dr.", 136, 143),
            SubtitleCue(10, 9.0, 10.0, "The value is 3.5.", 143, 160),
            SubtitleCue(11, 10.0, 11.0, 'He said "U.S."', 160, 174),
            SubtitleCue(12, 11.0, 12.0, "One condition: be ready;", 174, 198),
            SubtitleCue(13, 12.0, 13.0, "Use it (carefully).", 198, 217),
        ]

        srt = export_srt(cues, language="en")
        vtt = export_vtt(cues, language="en")

        for output in (srt, vtt):
            self.assertIn("Don't split well-known words", output)
            self.assertIn("First, second, and third", output)
            self.assertIn("Really?", output)
            self.assertNotIn("Really???", output)
            self.assertIn("Watch out!", output)
            self.assertNotIn("Watch out!!!", output)
            self.assertIn("What did you say?!", output)
            self.assertNotIn("What did you say!?!", output)
            self.assertIn('He said, "Stop"', output)
            self.assertIn("Wait...", output)
            self.assertIn("I live in the U.S.", output)
            self.assertIn("Ask Dr.", output)
            self.assertIn("The value is 3.5", output)
            self.assertIn('He said "U.S."', output)
            self.assertIn("One condition: be ready;", output)
            self.assertIn("Use it (carefully)", output)

    def test_export_english_can_preserve_all_source_punctuation(self):
        cue = SubtitleCue(1, 0.0, 1.25, "Really!!!", 0, 9)

        srt = export_srt(
            [cue],
            strip_trailing_punctuation=False,
            language="en",
        )
        vtt = export_vtt(
            [cue],
            language="en",
            clean_punctuation=False,
        )

        self.assertIn("\nReally!!!\n", srt)
        self.assertIn("\nReally!!!\n", vtt)

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
