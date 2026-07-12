import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.models import AlignmentToken, SubtitleCue
from autosrt_aligner.profiles import resolve_profile
from autosrt_aligner.quality import build_quality_report
from autosrt_aligner.splitter import _rebalance_korean_timeline, _smooth_timing, split_subtitles, wrap_subtitle_text
from autosrt_aligner.text import validate_subtitle_continuity


def char_tokens(text: str, step: float = 0.22) -> list[AlignmentToken]:
    tokens = []
    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        start = len(tokens) * step
        tokens.append(AlignmentToken(ch, start, start + step * 0.8, idx, idx + 1))
    return tokens


def assert_not_split_inside(test_case: unittest.TestCase, text: str, cues: list[SubtitleCue], phrase: str) -> None:
    start = text.index(phrase)
    protected = set(range(start + 1, start + len(phrase)))
    boundaries = {cue.end_char for cue in cues[:-1]}
    test_case.assertFalse(boundaries & protected, f"split inside {phrase!r}")


class SplitterQualityTests(unittest.TestCase):
    def test_splitter_keeps_text_continuous_and_non_overlapping(self):
        text = "这是第一句话，这是第二句话。然后继续讲故事，字幕要自然。"
        profile = resolve_profile("youtube_long", "zh", min_duration=0.8, max_duration=3.0, max_chars_per_line=12)
        cues = split_subtitles(text, char_tokens(text), "zh", profile)
        self.assertTrue(validate_subtitle_continuity(cues, text))
        self.assertGreater(len(cues), 1)
        for prev, cur in zip(cues, cues[1:]):
            self.assertLessEqual(prev.end, cur.start)

    def test_wrap_keeps_export_text_single_line(self):
        wrapped = wrap_subtitle_text("テーブルに置くことしかできませんでした。", "ja", 12)
        self.assertNotIn("\n", wrapped)
        self.assertEqual(wrapped, "テーブルに置くことしかできませんでした。")

    def test_japanese_splitter_avoids_common_bad_word_boundaries(self):
        text = (
            "最後の出勤の日、私は玄関でお疲れさまでしたと言って送り出したんですけれど、"
            "正則はテレビの前で、チャンネルを変える手がだんだん遅くなっていくのが、"
            "私には気づいていました。以前はスーパーに買い物に行ってくれていました。"
            "何しろ四十年間、「自分たちがどこに住みたいか」と考えたこともなく、"
            "月八万五千円の家賃を見て、二人の関係をどうしたいか、どっちの道を行こうかと考えました。"
        )
        profile = resolve_profile("youtube_long", "ja", min_duration=1.2, max_duration=6.5, max_chars_per_line=12)
        cues = split_subtitles(text, char_tokens(text, step=0.1), "ja", profile)
        self.assertTrue(validate_subtitle_continuity(cues, text))
        for cue in cues:
            self.assertNotIn("\n", cue.text)
        for phrase in [
            "言って",
            "テレビ",
            "だんだん",
            "気づいて",
            "買い物",
            "自分たち",
            "月八万五千円",
            "関係",
            "行こうか",
        ]:
            assert_not_split_inside(self, text, cues, phrase)

    def test_chinese_splitter_protects_quotes_and_sound_words(self):
        text = (
            '那一瞬间，阿珠攥着电话倒在地上的画面，"啪"地一下盖了过来，'
            '她忽然明白，这把钥匙不是为了让建平来"管"她，'
            "而是把他也算进'信得过的人'里头。"
        )
        profile = resolve_profile("youtube_long", "zh", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        cues = split_subtitles(text, char_tokens(text, step=0.12), "zh", profile)
        self.assertTrue(validate_subtitle_continuity(cues, text))
        for cue in cues:
            self.assertNotIn("\n", cue.text)
        for phrase in ['"啪"地一下', '"管"她', "'信得过的人'"]:
            assert_not_split_inside(self, text, cues, phrase)

    def test_chinese_splitter_splits_overlong_cue_at_safe_boundary(self):
        text = (
            "这次……换你,也留一把妈的钥匙。 "
            "她说着,从口袋里,掏出一把崭新的、刚配好的钥匙,放到建平手里。"
            "这把钥匙,不是为了让建平来\"管\"她,是她主动交出去的。"
        )
        profile = resolve_profile("youtube_long", "zh", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        cues = split_subtitles(text, char_tokens(text, step=0.1), "zh", profile)
        self.assertTrue(validate_subtitle_continuity(cues, text))
        self.assertGreater(len(cues), 2)
        self.assertLessEqual(max(len("".join(cue.text.split())) for cue in cues), 38)
        assert_not_split_inside(self, text, cues, '"管"她')

    def test_korean_splitter_avoids_common_word_boundaries(self):
        text = (
            "어머니는 괜찮다고 말했지만 마음속으로는 오래 생각했습니다. "
            "그날 밤에도 가족들은 조용히 서로를 바라보며 다시 괜찮다고 말했습니다. "
            "나는 그 말이 끝나기도 전에 어머니는 웃으면서 천천히 고개를 끄덕였습니다."
        )
        profile = resolve_profile("youtube_long", "ko", min_duration=1.2, max_duration=4.2, max_chars_per_line=20)
        cues = split_subtitles(text, char_tokens(text, step=0.12), "ko", profile)
        self.assertTrue(validate_subtitle_continuity(cues, text))
        for cue in cues:
            self.assertNotIn("\n", cue.text)
        for phrase in ["어머니는", "괜찮다고", "생각했습니다"]:
            assert_not_split_inside(self, text, cues, phrase)

    def test_timing_smoothing_reduces_large_gap_and_fast_cue(self):
        profile = resolve_profile("youtube_long", "ja", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        cues = [
            SubtitleCue(1, 0.0, 2.0, "年金の通知書を横に置いて、", 0, 12),
            SubtitleCue(2, 4.48, 6.06, "何かを計算している。私が聞いたら、", 12, 30),
        ]
        smoothed = _smooth_timing(cues, profile, "ja")
        gap = smoothed[1].start - smoothed[0].end
        cps = len(smoothed[1].text) / smoothed[1].duration
        self.assertLessEqual(gap, 0.201)
        self.assertLess(cps, 10.0)
        self.assertLessEqual(smoothed[0].end, smoothed[1].start)

    def test_timing_smoothing_can_shift_next_cue_when_previous_is_maxed(self):
        profile = resolve_profile("youtube_long", "ja", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        cues = [
            SubtitleCue(1, 0.0, 6.5, "それが、長い長い年月、積もり積もって、運命になっていくんです。", 0, 32),
            SubtitleCue(2, 7.22, 10.22, "愚痴と昔の自慢を、言わない。", 32, 48),
        ]
        smoothed = _smooth_timing(cues, profile, "ja")
        gap = smoothed[1].start - smoothed[0].end
        self.assertLessEqual(gap, 0.201)
        self.assertLessEqual(smoothed[1].duration, profile.max_duration)
        self.assertLessEqual(smoothed[0].end, smoothed[1].start)

    def test_korean_timing_smoothing_uses_soft_max_to_reduce_stranded_gap(self):
        profile = resolve_profile("youtube_long", "ko", min_duration=1.2, max_duration=4.0, max_chars_per_line=20)
        cues = [
            SubtitleCue(
                1,
                83.46,
                87.46,
                "그날 밤 저는 잠이 안 와서 장판 밑에 깔아 둔 봉투를 꺼냈습니다. 고무줄로 감아 둔 꼬깃꼬깃한 지폐를 침",
                0,
                42,
            ),
            SubtitleCue(2, 91.28, 93.46, "발라 가며 세고 또 셌지요.", 42, 53),
        ]
        smoothed = _smooth_timing(cues, profile, "ko")
        gap = smoothed[1].start - smoothed[0].end
        self.assertLessEqual(gap, 0.201)
        self.assertLessEqual(smoothed[0].end, smoothed[1].start)
        report = build_quality_report(
            smoothed,
            "그날 밤 저는 잠이 안 와서 장판 밑에 깔아 둔 봉투를 꺼냈습니다. 고무줄로 감아 둔 꼬깃꼬깃한 지폐를 침발라 가며 세고 또 셌지요.",
            smoothed[-1].end,
            profile,
            "ko",
        )
        self.assertEqual(report["large_gap_count"], 0)
        self.assertEqual(report["too_long_count"], 0)

    def test_korean_rebalances_severe_fast_cue_with_slow_neighbors(self):
        profile = resolve_profile("youtube_long", "ko", min_duration=1.2, max_duration=4.0, max_chars_per_line=20)
        parts = [
            "지갑을 두고 왔다는 사람 눈이 왜 그렇게 절박했을까요. 저부터도 부조 봉투 앞에선 오만 원이냐 십만 원이냐",
            "한나절을",
            "망설이는 처지라,",
        ]
        text = " ".join(parts)
        first_end = len(parts[0])
        second_start = first_end + 1
        second_end = second_start + len(parts[1])
        third_start = second_end + 1
        cues = [
            SubtitleCue(1, 600.945, 602.485, parts[0], 0, first_end),
            SubtitleCue(2, 602.685, 604.806, parts[1], second_start, second_end),
            SubtitleCue(3, 604.886, 609.208, parts[2], third_start, len(text)),
        ]
        rebalanced = _rebalance_korean_timeline(cues, text, "ko", profile)
        report = build_quality_report(rebalanced, text, rebalanced[-1].end, profile, "ko")
        self.assertEqual(len(rebalanced), 3)
        self.assertTrue(validate_subtitle_continuity(rebalanced, text))
        self.assertEqual(report["ko_fast_cue_count"], 0)
        self.assertEqual(report["ko_unsafe_boundary_count"], 0)

    def test_korean_merges_short_tail_cue_when_safe(self):
        profile = resolve_profile("youtube_long", "ko", min_duration=1.2, max_duration=4.0, max_chars_per_line=20)
        text = "찾아뵙겠습니다. 늘 건강하십시오."
        split_at = text.index("건강")
        cues = [
            SubtitleCue(1, 1546.88, 1548.98, text[:split_at].strip(), 0, split_at),
            SubtitleCue(2, 1549.06, 1549.51, text[split_at:], split_at, len(text)),
        ]
        rebalanced = _rebalance_korean_timeline(cues, text, "ko", profile)
        self.assertEqual(len(rebalanced), 1)
        self.assertEqual(rebalanced[0].text, text)
        self.assertTrue(validate_subtitle_continuity(rebalanced, text))

    def test_quality_report_counts_problems(self):
        profile = resolve_profile("standard", "zh", min_duration=1.0, max_duration=2.0, max_chars_per_line=6)
        cues = [
            SubtitleCue(1, 0.0, 0.4, "这是第一句。", 0, 5),
            SubtitleCue(2, 1.6, 3.0, "这是第二句很长很长", 5, 14),
        ]
        report = build_quality_report(cues, "这是第一句。这是第二句很长很长", 3.0, profile)
        self.assertEqual(report["large_gap_count"], 1)
        self.assertEqual(report["too_short_count"], 1)
        self.assertEqual(report["weak_boundary_count"], 1)
        self.assertIn("p95_chars_per_second", report)
        self.assertLess(report["quality_score"], 100)

    def test_quality_report_counts_japanese_word_internal_boundaries(self):
        text = "スーパーに買い物に行きました。"
        profile = resolve_profile("youtube_long", "ja", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        split_at = text.index("物")
        cues = [
            SubtitleCue(1, 0.0, 1.8, text[:split_at], 0, split_at),
            SubtitleCue(2, 1.88, 3.2, text[split_at:], split_at, len(text)),
        ]
        report = build_quality_report(cues, text, 3.2, profile, "ja")
        self.assertEqual(report["weak_boundary_count"], 1)

    def test_quality_report_adds_chinese_risk_metrics(self):
        text = '那一瞬间，阿珠攥着电话倒在地上的画面，"啪"地一下盖了过来。'
        split_at = text.index('"啪"') + 1
        cues = [
            SubtitleCue(1, 0.0, 2.0, text[:split_at], 0, split_at),
            SubtitleCue(2, 4.0, 6.2, text[split_at:], split_at, len(text)),
        ]
        profile = resolve_profile("youtube_long", "zh", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        report = build_quality_report(cues, text, 6.2, profile, "zh")
        self.assertGreater(report["zh_unsafe_boundary_count"], 0)
        self.assertGreater(report["zh_timeline_risk_count"], 0)
        self.assertIn("存在中文不安全切段", report["warnings"])

    def test_quality_report_adds_korean_risk_metrics(self):
        text = "어머니는 괜찮다고 말했습니다."
        split_at = text.index("찮")
        cues = [
            SubtitleCue(1, 0.0, 0.35, text[:split_at], 0, split_at),
            SubtitleCue(2, 0.88, 2.4, text[split_at:], split_at, len(text)),
        ]
        profile = resolve_profile("youtube_long", "ko", min_duration=1.2, max_duration=6.5, max_chars_per_line=20)
        report = build_quality_report(cues, text, 2.4, profile, "ko")
        self.assertEqual(report["weak_boundary_count"], 1)
        self.assertGreater(report["ko_fast_cue_count"], 0)
        self.assertGreater(report["ko_unsafe_boundary_count"], 0)
        self.assertGreater(report["ko_timeline_risk_count"], 0)
        self.assertIn("存在韩语不安全切段", report["warnings"])


if __name__ == "__main__":
    unittest.main()
