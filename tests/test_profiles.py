import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.profiles import (
    CANONICAL_LANGUAGES,
    language_group,
    normalize_language,
)
from autosrt_aligner.segmenter_v2 import _has_bad_edge, _is_safe_boundary


class ProfileTests(unittest.TestCase):
    def test_chinese_variants_use_one_canonical_language(self):
        self.assertEqual(CANONICAL_LANGUAGES, ("zh", "ja", "en", "ko"))
        self.assertEqual(normalize_language("zh"), "zh")
        self.assertEqual(normalize_language("zh-TW"), "zh")
        self.assertEqual(language_group("zh"), "cjk")
        self.assertEqual(language_group("zh-TW"), "cjk")

    def test_traditional_alias_uses_chinese_boundary_rules(self):
        text = "我的房子"
        self.assertFalse(_is_safe_boundary(text, 2, "zh"))
        self.assertFalse(_is_safe_boundary(text, 2, "zh-TW"))
        self.assertTrue(_has_bad_edge("我的", "zh"))
        self.assertTrue(_has_bad_edge("我的", "zh-TW"))


if __name__ == "__main__":
    unittest.main()
