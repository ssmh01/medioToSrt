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


def fake_run_alignment_job(
    audio_path,
    script_text,
    language="auto",
    subtitle_profile="youtube_long",
    output_dir=None,
    min_duration=None,
    max_duration=None,
    max_chars_per_line=None,
    generate_vtt=True,
    preserve_punctuation=True,
    engine=None,
):
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="autosrt_test_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cues = [SubtitleCue(1, 0.0, 1.25, script_text.strip(), 0, len(script_text.strip()))]
    srt_path = out_dir / "output.srt"
    vtt_path = out_dir / "output.vtt" if generate_vtt else None
    quality_path = out_dir / "quality_report.json"
    alignment_path = out_dir / "alignment.json"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,250\n测试字幕\n", encoding="utf-8")
    if vtt_path:
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\n测试字幕\n", encoding="utf-8")
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
        self.assertIn("auto", payload["languages"])
        self.assertEqual(payload["defaults"]["subtitle_profile"], "youtube_long")

    def test_create_job_status_and_download(self):
        response = self.client.post(
            "/api/jobs",
            data={
                "script_text": "测试字幕",
                "language": "zh",
                "subtitle_profile": "youtube_long",
                "min_duration": "1.0",
                "max_duration": "4.0",
                "max_chars_per_line": "18",
                "generate_vtt": "true",
                "preserve_punctuation": "true",
            },
            files={"audio_file": ("audio.mp3", b"audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        payload = self._wait_for_job(job_id)
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["quality_report"]["subtitle_count"], 1)
        self.assertEqual(payload["preview_rows"][0][3], "测试字幕")
        self.assertEqual({item["kind"] for item in payload["downloads"]}, {"srt", "vtt", "quality_report", "alignment"})

        download = self.client.get(f"/api/jobs/{job_id}/files/srt")
        self.assertEqual(download.status_code, 200)
        self.assertIn("测试字幕", download.text)

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

    def test_unknown_job_and_file_kind_return_404(self):
        missing = self.client.get("/api/jobs/not-found")
        self.assertEqual(missing.status_code, 404)

    def _wait_for_job(self, job_id):
        deadline = time.time() + 3
        while time.time() < deadline:
            response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"succeeded", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail("job did not finish")


if __name__ == "__main__":
    unittest.main()
