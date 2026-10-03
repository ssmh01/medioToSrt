import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticQualityAssetTests(unittest.TestCase):
    def test_index_loads_versioned_v2_quality_script_and_uses_evidence_label(self):
        index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("/static/app.js?v=20261003-natural-subtitles", index)
        self.assertIn("原文证据覆盖率", index)
        self.assertNotIn("语音覆盖率", index)

    def test_app_uses_source_token_coverage_and_separates_low_confidence_metric(self):
        app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("report.source_token_coverage", app)
        self.assertIn('["低置信 token", String(report.low_confidence_token_count || 0)]', app)
        self.assertIn('payload.status === "needs_review"', app)
        self.assertNotIn("+ Number(report.low_confidence_token_count || 0)", app)
        self.assertNotIn("report.unaligned_text_ratio", app)


if __name__ == "__main__":
    unittest.main()
