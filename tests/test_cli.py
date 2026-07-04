import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autosrt_aligner import cli
from autosrt_aligner.models import JobResult


class CliTests(unittest.TestCase):
    def test_cli_accepts_korean_language(self):
        original_runner = cli.run_alignment_job
        calls = []

        def fake_run_alignment_job(**kwargs):
            calls.append(kwargs)
            out_dir = Path(kwargs["output_dir"])
            return JobResult(
                output_dir=out_dir,
                srt_path=out_dir / "output.srt",
                vtt_path=None,
                alignment_json_path=out_dir / "alignment.json",
                quality_report_path=out_dir / "quality_report.json",
                cues=[],
                quality_report={"quality_score": 100},
                alignment_payload={},
                logs=["fake"],
            )

        cli.run_alignment_job = fake_run_alignment_job
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = Path(temp_dir) / "script.txt"
                script_path.write_text("어머니는 괜찮다고 말했습니다.", encoding="utf-8")
                with redirect_stdout(StringIO()):
                    exit_code = cli.main(
                        [
                            "--audio",
                            str(Path(temp_dir) / "audio.mp3"),
                            "--text",
                            str(script_path),
                            "--language",
                            "ko",
                            "--out-dir",
                            str(Path(temp_dir) / "out"),
                        ]
                    )
        finally:
            cli.run_alignment_job = original_runner

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["language"], "ko")


if __name__ == "__main__":
    unittest.main()
