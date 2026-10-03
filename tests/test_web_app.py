import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi.testclient import TestClient

import app as web_app
from autosrt_aligner.models import JobResult, SubtitleCue
from autosrt_aligner.errors import ExportValidationError
from autosrt_aligner.formats import export_srt, export_vtt


def fake_run_alignment_job(
    audio_path,
    script_text,
    language="zh",
    subtitle_profile="youtube_long",
    output_dir=None,
    min_duration=None,
    max_duration=None,
    max_chars_per_line=None,
    generate_vtt=True,
    engine=None,
    max_chars_total=None,
):
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="autosrt_test_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cue_text = script_text.strip()
    cues = [SubtitleCue(1, 0.0, 1.25, cue_text, 0, len(cue_text))]
    srt_path = out_dir / "output.srt"
    vtt_path = out_dir / "output.vtt" if generate_vtt else None
    quality_path = out_dir / "quality_report.json"
    alignment_path = out_dir / "alignment.json"
    srt_path.write_text(export_srt(cues, language=language), encoding="utf-8")
    if vtt_path:
        vtt_path.write_text(export_vtt(cues), encoding="utf-8")
    quality_report = {
        "audio_duration": 1.25,
        "subtitle_count": 1,
        "avg_subtitle_duration": 1.25,
        "min_subtitle_duration": 1.25,
        "max_subtitle_duration": 1.25,
        "avg_chars_per_second": 4.0,
        "max_chars_per_second": 4.0,
        "p95_chars_per_second": 4.0,
        "max_chars_per_cue": len(script_text.strip()),
        "large_gap_count": 0,
        "max_gap_seconds": 0,
        "weak_boundary_count": 0,
        "overlap_count": 0,
        "too_short_count": 0,
        "too_long_count": 0,
        "empty_subtitle_count": 0,
        "unaligned_text_ratio": 0,
        "suspicious_gap_count": 0,
        "quality_score": 100,
        "warnings": [],
    }
    quality_path.write_text("{}", encoding="utf-8")
    alignment_path.write_text("{}", encoding="utf-8")
    return JobResult(
        output_dir=out_dir,
        srt_path=srt_path,
        vtt_path=vtt_path,
        alignment_json_path=alignment_path,
        quality_report_path=quality_path,
        cues=cues,
        quality_report=quality_report,
        alignment_payload={},
        logs=["fake alignment complete", "导出完成"],
    )


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.original_runner = web_app.run_alignment_job
        web_app.run_alignment_job = fake_run_alignment_job
        with web_app._jobs_lock:
            web_app._jobs.clear()
        self.client = TestClient(web_app.app)

    def tearDown(self):
        web_app.run_alignment_job = self.original_runner
        with web_app._jobs_lock:
            web_app._jobs.clear()

    def test_options_endpoint(self):
        response = self.client.get("/api/options")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("auto", payload["languages"])
        self.assertIn("zh", payload["languages"])
        self.assertNotIn("zh-TW", payload["languages"])
        self.assertIn("ko", payload["languages"])
        self.assertNotIn("language", payload["defaults"])
        self.assertEqual(payload["defaults"]["subtitle_profile"], "youtube_long")
        self.assertNotIn("preserve_punctuation", payload["defaults"])
        self.assertEqual(
            payload["language_defaults"],
            {
                "zh": {"min_duration": 1.2, "max_duration": 6.5, "max_chars_per_line": 18},
                "ja": {"min_duration": 1.2, "max_duration": 6.5, "max_chars_per_line": 17},
                "en": {"min_duration": 1.2, "max_duration": 6.0, "max_chars_per_line": 42},
                "ko": {"min_duration": 1.2, "max_duration": 6.5, "max_chars_per_line": 20},
            },
        )

    def test_create_job_requires_language(self):
        response = self.client.post(
            "/api/jobs",
            data={
                "script_text": "测试字幕",
                "subtitle_profile": "youtube_long",
                "min_duration": "1.0",
                "max_duration": "4.0",
                "max_chars_per_line": "12",
                "generate_vtt": "true",
            },
            files={"audio_file": ("audio.mp3", b"fake audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的语言参数", response.json()["detail"])

    def test_options_include_all_language_style_combinations(self):
        presets = self.client.get("/api/options").json()["profile_defaults"]
        expected_caps = {
            "zh": (34, 34, 26, 29), "ja": (34, 34, 26, 29),
            "ko": (38, 38, 29, 32), "en": (84, 84, 65, 84),
        }
        for language, caps in expected_caps.items():
            for style, cap in zip(("youtube_long", "standard", "short", "slow_elder"), caps):
                with self.subTest(language=language, style=style):
                    self.assertEqual(presets[language][style]["max_chars_total"], cap)
        self.assertEqual(presets["en"]["youtube_long"]["max_duration"], 6.0)
        self.assertEqual(presets["ja"]["short"]["max_duration"], 4.2)
        self.assertEqual(presets["ko"]["slow_elder"]["min_duration"], 1.5)

    def test_create_job_resolves_omitted_settings_and_keeps_custom_character_cap(self):
        calls = []

        def capture_job(**kwargs):
            calls.append(kwargs)
            return fake_run_alignment_job(**kwargs)

        web_app.run_alignment_job = capture_job
        for style, extra, expected in (
            ("short", {}, (1.0, 4.2, 26)),
            ("slow_elder", {}, (1.5, 7.0, 29)),
            ("youtube_long", {"max_chars_total": "12", "max_duration": "4.8"}, (1.2, 4.8, 12)),
        ):
            response = self.client.post(
                "/api/jobs", data={"script_text": "测试字幕。", "language": "zh", "subtitle_profile": style, **extra},
                files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
            )
            self.assertEqual(response.status_code, 200)
            self._wait_for_job(response.json()["job_id"])
            call = calls[-1]
            self.assertEqual((call["min_duration"], call["max_duration"], call["max_chars_total"]), expected)

    def test_create_job_rejects_nonpositive_character_cap(self):
        response = self.client.post(
            "/api/jobs", data={"script_text": "测试字幕。", "language": "zh", "max_chars_total": "0"},
            files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("最大字符数", response.json()["detail"])

    def test_create_job_status_and_download(self):
        response = self.client.post(
            "/api/jobs",
            data={
                "script_text": "测试字幕。",
                "language": "zh",
                "subtitle_profile": "youtube_long",
                "min_duration": "1.0",
                "max_duration": "4.0",
                "max_chars_per_line": "18",
                "generate_vtt": "true",
            },
            files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        payload = self._wait_for_job(job_id)
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["quality_report"]["subtitle_count"], 1)
        self.assertEqual(payload["preview_rows"][0][3], "测试字幕。")
        self.assertEqual({item["kind"] for item in payload["downloads"]}, {"srt", "vtt"})
        self.assertIn(
            {"kind": "srt", "label": "audio.srt", "url": f"/api/jobs/{job_id}/files/srt"},
            payload["downloads"],
        )
        self.assertIn(
            {"kind": "vtt", "label": "audio.vtt", "url": f"/api/jobs/{job_id}/files/vtt"},
            payload["downloads"],
        )

        download = self.client.get(f"/api/jobs/{job_id}/files/srt")
        self.assertEqual(download.status_code, 200)
        self.assertIn("audio.srt", download.headers["content-disposition"])
        self.assertIn("\n测试字幕\n", download.text)
        self.assertNotIn("测试字幕。", download.text)

    def test_create_job_accepts_legacy_traditional_language_alias(self):
        response = self.client.post(
            "/api/jobs",
            data={
                "script_text": "测试字幕。",
                "language": "zh-TW",
                "subtitle_profile": "youtube_long",
                "min_duration": "1.0",
                "max_duration": "4.0",
                "max_chars_per_line": "18",
                "generate_vtt": "true",
            },
            files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = self._wait_for_job(response.json()["job_id"])
        self.assertEqual(payload["status"], "succeeded")

    def test_create_job_requires_audio(self):
        response = self.client.post("/api/jobs", data={"script_text": "测试字幕"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("请先上传音频文件", response.json()["detail"])

    def test_create_job_requires_script(self):
        response = self.client.post(
            "/api/jobs",
            files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("请上传 TXT 文案", response.json()["detail"])

    def test_quality_gate_exposes_review_artifacts_without_subtitle_download(self):
        original_runner = web_app.run_alignment_job

        def blocked_run_alignment_job(*args, **kwargs):
            out_dir = Path(kwargs["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "quality_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2",
                        "publish_status": "blocked",
                        "warnings": ["存在时间轴问题"],
                        "subtitle_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "alignment.json").write_text(
                json.dumps(
                    {
                        "cues": [
                            {"index": 1, "start": 0.0, "end": 1.5, "text": "测试字幕"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            raise ExportValidationError("质量门禁阻断")

        web_app.run_alignment_job = blocked_run_alignment_job
        try:
            response = self.client.post(
                "/api/jobs",
                data={
                    "script_text": "测试字幕",
                    "language": "zh",
                    "subtitle_profile": "youtube_long",
                },
                files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
            )
            self.assertEqual(response.status_code, 200)
            payload = self._wait_for_job(response.json()["job_id"])
            self.assertEqual(payload["status"], "needs_review")
            self.assertEqual(payload["quality_report"]["publish_status"], "blocked")
            self.assertEqual({item["kind"] for item in payload["downloads"]}, {"quality_report", "alignment"})
            self.assertEqual(payload["preview_rows"][0][3], "测试字幕")
            quality_download = self.client.get(f"/api/jobs/{response.json()['job_id']}/files/quality_report")
            self.assertEqual(quality_download.status_code, 200)
            self.assertIn("blocked", quality_download.text)
        finally:
            web_app.run_alignment_job = original_runner

    def test_unknown_job_and_file_kind_return_404(self):
        missing = self.client.get("/api/jobs/not-found")
        self.assertEqual(missing.status_code, 404)

    def _wait_for_job(self, job_id):
        deadline = time.time() + 3
        while time.time() < deadline:
            response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"succeeded", "needs_review", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail("job did not finish")


if __name__ == "__main__":
    unittest.main()
