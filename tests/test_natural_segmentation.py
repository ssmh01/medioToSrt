import unittest
from dataclasses import replace

from autosrt_aligner.errors import AlignmentError
from autosrt_aligner.models import AlignmentToken, SubtitleCue
from autosrt_aligner.profiles import resolve_profile
from autosrt_aligner.quality_v2 import _segmentation_issues
from autosrt_aligner.segmenter_v2 import segment_cues
from autosrt_aligner.text import (
    build_source_document,
    is_nonspoken_alignment_char,
    validate_subtitle_continuity,
)


def character_tokens(text, step=0.2):
    tokens = []
    for index, char in enumerate(text):
        if is_nonspoken_alignment_char(char):
            continue
        start = len(tokens) * step
        tokens.append(AlignmentToken(char, start, start + step, index, index + 1, 0.9))
    return tokens


class NaturalSegmentationTests(unittest.TestCase):
    def test_preserves_words_and_phrases_in_four_languages(self):
        cases = [
            ("zh", "我把那封信原封不动放回抽屉，后来才明白她的意思。", "原封不动"),
            ("zh", "我把那封信原封不動放回抽屜，後來才明白她的意思。", "原封不動"),
            ("ja", "主人の顔を見ると、昔の約束を思い出しました。", "顔を"),
            ("ko", "도시락 가방을 들고 조용히 집으로 돌아왔습니다.", "도시락 가방"),
            (
                "en",
                "He looked at the nursing home and quietly walked away.",
                "nursing home",
            ),
        ]
        for language, text, phrase in cases:
            with self.subTest(language=language, phrase=phrase):
                source = build_source_document(text, language)
                tokens = character_tokens(text, 0.12 if language == "en" else 0.2)
                profile = resolve_profile("youtube_long", language, max_duration=3.0)
                cues = segment_cues(source, tokens, profile)
                self.assertGreater(len(cues), 1)
                self.assertTrue(validate_subtitle_continuity(cues, text))
                self.assertFalse(_segmentation_issues(source, cues, profile))
                start = text.index(phrase)
                for cue in cues[:-1]:
                    self.assertFalse(start < cue.end_char < start + len(phrase))
                self.assertTrue(
                    all(
                        cue.start in {t.start for t in tokens}
                        and cue.end in {t.end for t in tokens}
                        for cue in cues
                    )
                )

    def test_complete_short_question_stands_alone_and_passes_export_gate(self):
        text = "真的？我想了很久，终于决定告诉她。"
        source = build_source_document(text, "zh")
        tokens = character_tokens(text)
        tokens[0].start, tokens[0].end = 0.0, 0.45
        tokens[1].start, tokens[1].end = 0.45, 0.9
        for index, token in enumerate(tokens[2:]):
            token.start = 1.0 + index * 0.2
            token.end = token.start + 0.2
        profile = resolve_profile("youtube_long", "zh")
        cues = segment_cues(source, tokens, profile)
        self.assertEqual(cues[0].text, "真的？")
        self.assertEqual(cues[0].end, 0.9)
        self.assertFalse(_segmentation_issues(source, cues, profile))
        fragments = [
            SubtitleCue(1, 0, 0.9, "真的", 0, 2),
            SubtitleCue(2, 1, 4, text[2:], 2, len(text)),
        ]
        self.assertIn(
            "存在短于配置下限的 cue", _segmentation_issues(source, fragments, profile)
        )

    def test_opening_quote_belongs_to_next_utterance(self):
        text = 'She nodded. "I remember that nursing home."'
        source = build_source_document(text, "en")
        profile = resolve_profile("youtube_long", "en")
        cues = segment_cues(source, character_tokens(text, 0.18), profile)
        self.assertEqual(cues[0].text, "She nodded.")
        self.assertTrue(cues[1].text.startswith('"I remember'))
        self.assertTrue(cues[-1].text.endswith('"'))
        self.assertTrue(validate_subtitle_continuity(cues, text))

    def test_infeasible_character_limit_stops_segmentation(self):
        text = "supercalifragilisticexpialidocious"
        source = build_source_document(text, "en")
        profile = resolve_profile("youtube_long", "en", max_chars_total=10)
        tokens = [AlignmentToken(text, 0, 3, 0, len(text), 0.9)]
        with self.assertRaisesRegex(AlignmentError, "无法.*满足"):
            segment_cues(source, tokens, profile)

    def test_custom_minimum_is_respected_for_complete_utterances(self):
        profile = resolve_profile("youtube_long", "zh", min_duration=2.0)
        self.assertEqual(profile.min_complete_duration, 2.0)

    def test_terminal_audio_extension_is_considered_before_selecting_cuts(self):
        text = "我把那封信原封不动放回抽屉，后来才明白她的意思。"
        source = build_source_document(text, "zh")
        tokens = character_tokens(text)
        profile = resolve_profile("short", "zh", max_duration=3.0)
        old_cues = segment_cues(source, tokens, profile)
        audio_duration = tokens[-1].end + 1.4
        self.assertGreater(audio_duration - old_cues[-1].start, profile.max_duration)
        cues = segment_cues(source, tokens, profile, audio_duration=audio_duration)
        cues[-1] = replace(cues[-1], end=audio_duration)
        self.assertTrue(validate_subtitle_continuity(cues, text))
        self.assertFalse(_segmentation_issues(source, cues, profile))


if __name__ == "__main__":
    unittest.main()
