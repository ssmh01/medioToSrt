import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.models import AlignmentToken, SubtitleCue
from autosrt_aligner.text import (
    clean_script_text,
    map_tokens_to_display,
    normalize_for_compare,
    validate_subtitle_continuity,
)


class TextTests(unittest.TestCase):
    def test_clean_script_preserves_display_but_cleans_alignment(self):
        cleaned = clean_script_text("\ufeff# 标题\n这是 **原文**。")
        self.assertEqual(cleaned.display_text, "# 标题\n这是 **原文**。")
        self.assertNotIn("#", cleaned.align_text)
        self.assertNotIn("*", cleaned.align_text)
        self.assertIn("这是 原文。", cleaned.align_text)

    def test_map_tokens_to_display(self):
        cleaned = clean_script_text("这是第一句话。")
        tokens = [
            AlignmentToken("这是", 0.0, 0.5),
            AlignmentToken("第一句话", 0.5, 1.5),
            AlignmentToken("。", 1.5, 1.6),
        ]
        mapped = map_tokens_to_display(tokens, cleaned)
        self.assertEqual(mapped[0].start_char, 0)
        self.assertEqual(mapped[-1].end_char, len(cleaned.display_text))

    def test_validate_subtitle_continuity_ignores_layout_whitespace(self):
        display = "第一句。\n第二句。"
        cues = [
            SubtitleCue(1, 0, 1, "第一句。", 0, 4),
            SubtitleCue(2, 1, 2, "第二句。", 4, len(display)),
        ]
        self.assertTrue(validate_subtitle_continuity(cues, display))
        self.assertEqual(normalize_for_compare("A \n B"), "AB")


if __name__ == "__main__":
    unittest.main()

