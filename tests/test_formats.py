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
        srt = export_srt(cues)
        vtt = export_vtt(cues)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,250", srt)
        self.assertIn("\n第一句\n", srt)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.250", vtt)
        self.assertIn("第一句。", vtt)

    def test_export_srt_strips_all_trailing_punctuation_per_line(self):
        cues = [
            SubtitleCue(1, 0.0, 1.25, "他说完了。」", 0, 6),
            SubtitleCue(2, 1.25, 2.5, "第一句，第二句。", 6, 13),
            SubtitleCue(3, 2.5, 3.75, "中间？可以", 13, 18),
        ]
        srt = export_srt(cues)

        self.assertIn("\n他说完了\n\n", srt)
        self.assertIn("\n第一句，第二句\n\n", srt)
        self.assertIn("\n中间？可以\n", srt)

    def test_export_srt_strips_trailing_punctuation_on_each_line(self):
        cues = [SubtitleCue(1, 0.0, 1.25, "第一行。\n第二行？", 0, 8)]
        srt = export_srt(cues)

        self.assertIn("\n第一行\n第二行\n", srt)


if __name__ == "__main__":
    unittest.main()
