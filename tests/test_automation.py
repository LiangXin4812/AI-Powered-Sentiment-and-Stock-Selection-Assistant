"""Daily automation pieces that run without the database: publishing and the report checks."""
import sys
import tempfile
import unittest
from pathlib import Path

from sgx_capstone.configuration import ROOT, load_config, configure_report_modules


class AutomationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, _ = load_config()
        sys.path.insert(0, str(ROOT / "src"))
        configure_report_modules(cfg)

    def test_both_samples_pass_every_report_check(self):
        from sgx_capstone.daily_report import gather_sample, render
        from sgx_capstone.automation import check_report
        for date in ("2026-09-21", "2026-09-17"):
            D = gather_sample(ROOT, date=date)
            D["narrative"] = {"mode": "template", "summary": None, "cards": {}, "flags": []}
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(check_report(render(D, tmp), D), [], date)

    def test_lessons_from_the_multi_date_check(self):
        """check-dates over 20 stored dates (17 Aug - 22 Sep) found three things, fixed here."""
        from sgx_capstone.daily_report import pick_headlines, gather_sample, ATTENTION_FLAG
        from sgx_capstone import narrative as N
        import re
        # 1. quoted headlines that read as advice or a price target are not printed
        heads = [{"title": "DBS Was Expensive at S$59. Now It's S$77.60. Should You Sell?", "sentiment": "bullish",
                  "score": 0.99, "at": "Tue, 01 Sep 2026 08:00:00 GMT"},
                 {"title": "Will DBS Reach S$100 by the End of 2026?", "sentiment": "bullish", "score": 0.98,
                  "at": "Tue, 01 Sep 2026 08:00:00 GMT"},
                 {"title": "DBS reports record quarterly profit", "sentiment": "bullish", "score": 0.90,
                  "at": "Tue, 01 Sep 2026 08:00:00 GMT"}]
        self.assertEqual([h["title"] for h in pick_headlines(heads, "2026-09-02")],
                         ["DBS reports record quarterly profit"])
        # 2. ⚑ means ELEVATED only (divergence or silent news + conviction + usable record)
        self.assertEqual(set(ATTENTION_FLAG), {"ELEVATED"})
        # 3. early in the live period no forecast record exists yet: no 'nan' is printed
        from sgx_capstone.daily_report import research_findings, render
        from sgx_capstone.automation import report_text
        D = gather_sample(ROOT, date="2026-09-21")
        D["findings"] = research_findings(ROOT, "2026-08-17")
        D["narrative"] = {"mode": "template", "summary": None, "cards": {}, "flags": []}
        with tempfile.TemporaryDirectory() as tmp:
            body = report_text(render(D, tmp))
        self.assertNotRegex(body, r"\bnan\b")
        self.assertIn("too few to quote a record", body)
        # 4. the market-read draft never carries a digit outside a placeholder (counts are words)
        for date in ("2026-09-21", "2026-09-17"):
            D = gather_sample(ROOT, date=date)
            f, _, _ = N.build_facts(D)
            for line in N.summary_draft(D, f):
                self.assertNotRegex(N._ALLOWED.sub("", N._PH.sub("", line)), r"\d", line)

    def test_publish_writes_the_day_folder_and_the_index(self):
        from sgx_capstone.daily_report import gather_sample, render
        from sgx_capstone.automation import publish
        D = gather_sample(ROOT, date="2026-09-21")
        D["narrative"] = {"mode": "template", "summary": None, "cards": {}, "flags": []}
        with tempfile.TemporaryDirectory() as tmp:
            docx = render(D, Path(tmp) / "out")
            files = publish(Path(tmp), docx, "2026-09-21")
            index = (Path(tmp) / "reports" / "README.md").read_text()
            self.assertTrue(files["docx"].exists())
            self.assertTrue(files["notes"].exists())
            self.assertIn("| 2026-09-21 | [docx](2026-09-21/", index)

    def test_workflow_runs_after_the_close_on_weekdays_with_secrets_only(self):
        path = ROOT / ".github" / "workflows" / "daily_report.yml"
        if not path.exists():
            self.skipTest("workflow file is not copied into the Docker image")
        wf = path.read_text()
        self.assertIn('cron: "30 10 * * 1-5"', wf)         # 18:30 Singapore time
        self.assertIn("python -m sgx_capstone run-all", wf)
        self.assertIn("secrets.SUPABASE_URL", wf)
        self.assertNotRegex(wf, r"postgres(ql)?://")
        self.assertNotRegex(wf, r"gsk_[A-Za-z0-9]")


if __name__ == "__main__":
    unittest.main()
