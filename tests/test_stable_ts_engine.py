import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import autosrt_aligner.engines.stable_ts as stable_ts
from autosrt_aligner.models import CleanedText


class FakeStableTsModel:
    def __init__(self):
        self.languages = []

    def align(self, audio_path, text, language):
        self.languages.append(language)
        return SimpleNamespace(
            duration=0.4,
            segments=[
                {
                    "words": [
                        {
                            "word": text[:1],
                            "start": 0.0,
                            "end": 0.2,
                            "probability": 0.9,
                        }
                    ]
                }
            ],
        )


class StableTsLanguageTests(unittest.TestCase):
    def setUp(self):
        self.original_module = sys.modules.get("stable_whisper")
        self.original_ffmpeg_check = stable_ts.ensure_ffmpeg_on_path

    def tearDown(self):
        if self.original_module is None:
            sys.modules.pop("stable_whisper", None)
        else:
            sys.modules["stable_whisper"] = self.original_module
        stable_ts.ensure_ffmpeg_on_path = self.original_ffmpeg_check

    def test_korean_language_is_passed_to_stable_ts(self):
        language_arg, raw = self._align_language("ko", "테스트")
        self.assertEqual(language_arg, "ko")
        self.assertEqual(raw["requested_language"], "ko")
        self.assertEqual(raw["stable_ts_language"], "ko")

    def test_traditional_chinese_maps_to_stable_ts_zh(self):
        language_arg, raw = self._align_language("zh-TW", "測試")
        self.assertEqual(language_arg, "zh")
        self.assertEqual(raw["requested_language"], "zh-TW")
        self.assertEqual(raw["stable_ts_language"], "zh")

    def _align_language(self, language: str, text: str):
        fake_model = FakeStableTsModel()
        sys.modules["stable_whisper"] = SimpleNamespace(load_model=lambda _model_name: fake_model)
        stable_ts.ensure_ffmpeg_on_path = lambda: "ffmpeg"
        engine = stable_ts.StableTsEngine()
        cleaned = CleanedText(display_text=text, align_text=text, align_to_display=list(range(len(text))))
        logs = []
        result = engine.align(Path("audio.wav"), cleaned, language, logs)
        return fake_model.languages[-1], result.raw


if __name__ == "__main__":
    unittest.main()
