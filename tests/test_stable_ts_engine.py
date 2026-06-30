import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.engines.stable_ts import _resolve_stable_ts_language


class StableTsEngineTests(unittest.TestCase):
    def test_auto_language_resolves_to_stable_ts_language_code(self):
        self.assertEqual(_resolve_stable_ts_language("auto", "这是中文文案。"), "zh")
        self.assertEqual(_resolve_stable_ts_language("auto", "これは日本語です。"), "ja")
        self.assertEqual(_resolve_stable_ts_language("auto", "This is English."), "en")

    def test_explicit_traditional_chinese_uses_whisper_chinese_code(self):
        self.assertEqual(_resolve_stable_ts_language("zh-TW", "這是繁體中文。"), "zh")


if __name__ == "__main__":
    unittest.main()
