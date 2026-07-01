import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner.models import AlignmentResult, AlignmentToken, SubtitleCue
from autosrt_aligner.pipeline import (
    TimelineRepairSummary,
    _apply_timeline_quality,
    _detect_zh_cue_timeline_ranges,
    _register_unresolved_zh_timeline_risks,
    _repair_timeline_if_needed,
    _repair_zh_cue_risks_with_local_realign,
    run_alignment_job,
)
from autosrt_aligner.profiles import resolve_profile
from autosrt_aligner.quality import build_quality_report
from autosrt_aligner.text import validate_subtitle_continuity


class FakeEngine:
    requires_audio_preprocessing = False

    def align(self, audio_path, cleaned_text, language, logs):
        tokens = []
        visible_index = 0
        for idx, ch in enumerate(cleaned_text.display_text):
            if ch.isspace():
                continue
            start = visible_index * 0.24
            tokens.append(AlignmentToken(ch, start, start + 0.18, idx, idx + 1))
            visible_index += 1
        logs.append("fake alignment complete")
        return AlignmentResult(tokens=tokens, raw={"engine": "fake"}, audio_duration=visible_index * 0.24)


class BrokenTimelineEngine:
    requires_audio_preprocessing = False

    def __init__(self, repair_succeeds: bool = True) -> None:
        self.repair_succeeds = repair_succeeds
        self.realign_calls = 0

    def align(self, audio_path, cleaned_text, language, logs):
        break_at = cleaned_text.display_text.index("心の中")
        tokens = []
        visible_index = 0
        for idx, ch in enumerate(cleaned_text.display_text):
            if ch.isspace():
                continue
            if idx < break_at:
                start = visible_index * 0.2
                confidence = 0.9
            else:
                start = 44.0 + visible_index * 1.6
                confidence = 0.01
            tokens.append(AlignmentToken(ch, start, start + 0.12, idx, idx + 1, confidence=confidence))
            visible_index += 1
        return AlignmentResult(tokens=tokens, raw={"engine": "broken"}, audio_duration=90.0, language=language)

    def realign_fragment(self, audio_path, cleaned_text, language, audio_start, audio_end, work_dir, logs, attempt_id):
        self.realign_calls += 1
        tokens = []
        visible_index = 0
        step = 0.18 if self.repair_succeeds else 3.0
        confidence = 0.92 if self.repair_succeeds else 0.01
        for idx, ch in enumerate(cleaned_text.display_text):
            if ch.isspace():
                continue
            start = visible_index * step
            tokens.append(AlignmentToken(ch, start, start + 0.12, idx, idx + 1, confidence=confidence))
            visible_index += 1
        logs.append(f"fake local realign {attempt_id}")
        return AlignmentResult(tokens=tokens, raw={"engine": "fake-local"}, audio_duration=visible_index * step)


class ZhCueRepairEngine:
    requires_audio_preprocessing = False

    def __init__(self, repair_succeeds: bool = True) -> None:
        self.repair_succeeds = repair_succeeds
        self.realign_calls = 0

    def realign_fragment(self, audio_path, cleaned_text, language, audio_start, audio_end, work_dir, logs, attempt_id):
        self.realign_calls += 1
        tokens = []
        visible_index = 0
        step = 0.2 if self.repair_succeeds else 3.0
        confidence = 0.9 if self.repair_succeeds else 0.01
        for idx, ch in enumerate(cleaned_text.display_text):
            if ch.isspace():
                continue
            start = visible_index * step
            tokens.append(AlignmentToken(ch, start, start + 0.12, idx, idx + 1, confidence=confidence))
            visible_index += 1
        return AlignmentResult(tokens=tokens, raw={"engine": "zh-local"}, audio_duration=visible_index * step, language=language)


class PipelineTests(unittest.TestCase):
    def test_run_alignment_job_with_fake_engine_exports_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "out"
            result = run_alignment_job(
                audio_path=Path(temp_dir) / "dummy.mp3",
                script_text="这是第一句话。这是第二句话。",
                language="zh",
                subtitle_profile="youtube_long",
                output_dir=out_dir,
                min_duration=0.8,
                max_duration=3.0,
                max_chars_per_line=12,
                generate_vtt=True,
                engine=FakeEngine(),
            )
            self.assertTrue(result.srt_path.exists())
            self.assertTrue(result.vtt_path and result.vtt_path.exists())
            self.assertTrue(result.quality_report_path.exists())
            self.assertTrue(result.alignment_json_path.exists())
            self.assertIn("这是第一句话", result.srt_path.read_text(encoding="utf-8"))
            self.assertEqual(result.quality_report["unaligned_text_ratio"], 0.0)
            self.assertEqual(result.quality_report["timeline_status"], "ok")
            self.assertIn("cue_diagnostics", result.alignment_payload)
            self.assertIn("before_repair", result.alignment_payload["cue_diagnostics"])
            self.assertIn("after_repair", result.alignment_payload["cue_diagnostics"])

    def test_local_realign_repairs_suspect_timeline(self):
        script = (
            "その時、四つ目の苦しみが、わかったんです。"
            "父親も、息子も、二人とも、心の中では、ずっと、「会いたい」と思っていた。"
            "待っている、うちに、間に合わなく、なってしまったんです。"
            "体が動かなくなるのも、居場所がなくなるのも、これは、避けられません。"
        )
        engine = BrokenTimelineEngine(repair_succeeds=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_alignment_job(
                audio_path=Path(temp_dir) / "dummy.mp3",
                script_text=script,
                language="ja",
                subtitle_profile="youtube_long",
                output_dir=Path(temp_dir) / "out",
                min_duration=1.2,
                max_duration=6.5,
                max_chars_per_line=18,
                generate_vtt=False,
                engine=engine,
            )

        self.assertGreater(engine.realign_calls, 0)
        self.assertEqual(result.quality_report["timeline_status"], "repaired")
        self.assertGreaterEqual(result.quality_report["timeline_confidence_score"], 90)
        self.assertGreater(len(result.quality_report["repaired_ranges"]), 0)
        self.assertEqual(result.quality_report["low_confidence_ranges"], [])
        self.assertIn("before_repair", result.alignment_payload["token_diagnostics"])
        self.assertIn("after_repair", result.alignment_payload["token_diagnostics"])
        before = result.alignment_payload["token_diagnostics"]["before_repair"]
        after = result.alignment_payload["token_diagnostics"]["after_repair"]
        self.assertGreater(before["low_confidence_token_count"], after["low_confidence_token_count"])
        self.assertTrue(validate_subtitle_continuity(result.cues, script))

    def test_failed_local_realign_marks_timeline_needs_review_but_exports(self):
        script = (
            "その時、四つ目の苦しみが、わかったんです。"
            "父親も、息子も、二人とも、心の中では、ずっと、「会いたい」と思っていた。"
            "待っている、うちに、間に合わなく、なってしまったんです。"
            "体が動かなくなるのも、居場所がなくなるのも、これは、避けられません。"
        )
        engine = BrokenTimelineEngine(repair_succeeds=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_alignment_job(
                audio_path=Path(temp_dir) / "dummy.mp3",
                script_text=script,
                language="ja",
                subtitle_profile="youtube_long",
                output_dir=Path(temp_dir) / "out",
                min_duration=1.2,
                max_duration=6.5,
                max_chars_per_line=18,
                generate_vtt=False,
                engine=engine,
            )
            self.assertTrue(result.srt_path.exists())

        self.assertEqual(result.quality_report["timeline_status"], "needs_review")
        self.assertLessEqual(result.quality_report["quality_score"], 75)
        self.assertGreater(len(result.quality_report["low_confidence_ranges"]), 0)
        self.assertIn("低置信时间轴", " ".join(result.quality_report["warnings"]))

    def test_timeline_repair_rebuilds_broken_tail(self):
        parts = [
            "その時、四つ目の苦しみが、わかったんです。",
            "父親も、息子も、二人とも、",
            "心の中",
            "では、ずっと、「会いたい」",
            "「謝りたい」と、思っていた。お互いに、相手からの、一本の電話を、待っていた。",
            "待っている、うち",
            "に、間に合わなく、",
            "なってしまったんです。",
            "体が動かなくなるのも、居場所がなくなるのも、人を見送るのも、これは、避けられません。",
            "実は、私にも、あったんです。四十年、胸に、刺さったままの、棘が。",
            "電話でも、いい。短い、手紙でも、いい。たった、一言で、いいんですよ。",
            "チャンネル登録と、通知のベルも、押しておいて、もらえると、嬉しいです。",
        ]
        text = "".join(parts)
        spans = []
        cursor = 0
        for part in parts:
            spans.append((cursor, cursor + len(part)))
            cursor += len(part)

        cues = [
            SubtitleCue(1, 0.0, 3.5, parts[0], *spans[0]),
            SubtitleCue(2, 3.7, 10.2, parts[1], *spans[1]),
            SubtitleCue(3, 17.4, 33.5, parts[2], *spans[2]),
            SubtitleCue(4, 33.58, 36.2, parts[3], *spans[3]),
            SubtitleCue(5, 36.28, 44.0, parts[4], *spans[4]),
            SubtitleCue(6, 47.2, 115.9, parts[5], *spans[5]),
            SubtitleCue(7, 348.88, 355.3, parts[6], *spans[6]),
            SubtitleCue(8, 356.0, 401.0, parts[7], *spans[7]),
            SubtitleCue(9, 401.1, 405.0, parts[8], *spans[8]),
            SubtitleCue(10, 406.7, 406.8, "".join(parts[9:]), spans[9][0], spans[-1][1]),
        ]
        profile = resolve_profile("youtube_long", "ja", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        logs: list[str] = []

        mapped_tokens = []
        visible_index = 0
        for idx, ch in enumerate(text):
            if ch.isspace():
                continue
            start = visible_index * 0.18
            mapped_tokens.append(AlignmentToken(ch, start, start + 0.12, idx, idx + 1, confidence=0.9))
            visible_index += 1

        repaired, info = _repair_timeline_if_needed(cues, text, mapped_tokens, profile, "ja", 80.0, logs)
        report = build_quality_report(repaired, text, 80.0, profile, "ja")
        gaps = [cur.start - prev.end for prev, cur in zip(repaired, repaired[1:])]
        cps_values = [len("".join(cue.text.split())) / max(cue.duration, 0.1) for cue in repaired]

        self.assertIsNotNone(info)
        self.assertTrue(validate_subtitle_continuity(repaired, text))
        self.assertEqual(report["too_long_count"], 0)
        self.assertEqual(report["overlap_count"], 0)
        self.assertEqual(report["large_gap_count"], 0)
        self.assertLessEqual(max(gaps, default=0.0), 0.201)
        self.assertLess(max(cps_values, default=0.0), 8.5)
        self.assertEqual(info["detected_index"], 3)
        self.assertEqual(info["start_index"], 2)
        self.assertEqual(info["mode"], "anchor_interpolated")
        self.assertEqual(info["confidence"], "low")
        self.assertIn("低置信估算", " ".join(logs))

        heart_cue = next(cue for cue in repaired if "心の中" in cue.text)
        self.assertLess(heart_cue.start, 8.0)

    def test_chinese_cue_risk_local_realign_repairs_tokens(self):
        text = (
            "我只是想,把你,也算进我'信得过的人'里头。"
            "这次……换你,也留一把妈的钥匙。 她说着,从口袋里,掏出一把崭新的、刚配好的钥匙,放到建平手里。"
            "这把钥匙,不是为了让建平来\"管\"她,是她主动交出去的。"
        )
        first_end = text.index("这次")
        second_end = text.index("这把钥匙")
        cues = [
            SubtitleCue(1, 0.0, 4.0, text[:first_end], 0, first_end),
            SubtitleCue(2, 4.2, 10.7, text[first_end:second_end], first_end, second_end),
            SubtitleCue(3, 12.9, 17.8, text[second_end:], second_end, len(text)),
        ]
        profile = resolve_profile("youtube_long", "zh", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        ranges = _detect_zh_cue_timeline_ranges(cues, text, 20.0, profile, "zh")
        self.assertTrue(any("zh_long_cue_gap" in range_.reasons for range_ in ranges))

        mapped_tokens = [
            AlignmentToken(ch, idx * 0.08, idx * 0.08 + 0.04, idx, idx + 1, confidence=0.8)
            for idx, ch in enumerate(text)
            if not ch.isspace()
        ]
        summary = TimelineRepairSummary()
        engine = ZhCueRepairEngine(repair_succeeds=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            repaired_tokens, changed = _repair_zh_cue_risks_with_local_realign(
                mapped_tokens,
                cues,
                text,
                engine,
                Path(temp_dir) / "dummy.wav",
                Path(temp_dir),
                "zh",
                20.0,
                profile,
                [],
                summary,
            )

        self.assertTrue(changed)
        self.assertGreater(engine.realign_calls, 0)
        self.assertGreater(len(summary.repaired_ranges), 0)
        self.assertEqual(summary.low_confidence_ranges, [])
        self.assertGreater(len(repaired_tokens), 0)

    def test_unresolved_chinese_cue_risk_marks_needs_review(self):
        text = (
            "我只是想,把你,也算进我'信得过的人'里头。"
            "这次……换你,也留一把妈的钥匙。 她说着,从口袋里,掏出一把崭新的、刚配好的钥匙,放到建平手里。"
            "这把钥匙,不是为了让建平来\"管\"她,是她主动交出去的。"
        )
        first_end = text.index("这次")
        second_end = text.index("这把钥匙")
        cues = [
            SubtitleCue(1, 0.0, 4.0, text[:first_end], 0, first_end),
            SubtitleCue(2, 4.2, 10.7, text[first_end:second_end], first_end, second_end),
            SubtitleCue(3, 12.9, 17.8, text[second_end:], second_end, len(text)),
        ]
        profile = resolve_profile("youtube_long", "zh", min_duration=1.2, max_duration=6.5, max_chars_per_line=18)
        ranges = _detect_zh_cue_timeline_ranges(cues, text, 20.0, profile, "zh")
        summary = TimelineRepairSummary()
        _register_unresolved_zh_timeline_risks(summary, ranges)
        report = build_quality_report(cues, text, 20.0, profile, "zh")
        _apply_timeline_quality(report, summary)

        self.assertEqual(report["timeline_status"], "needs_review")
        self.assertLessEqual(report["quality_score"], 75)
        self.assertGreater(len(report["low_confidence_ranges"]), 0)


if __name__ == "__main__":
    unittest.main()
