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
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.250", vtt)


if __name__ == "__main__":
    unittest.main()

