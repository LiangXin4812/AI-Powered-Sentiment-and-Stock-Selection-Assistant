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

    def test_what_followed_the_report_is_stated_honestly(self):
        """R15 (23 Oct): flagged stocks did not move more, and the ✓ gain is in 'no big move' calls."""
        from sgx_capstone.research.r15_report_followup import run
        from sgx_capstone.daily_report import gather_sample, render, flag_reason
        from sgx_capstone.automation import report_text
        # on a copy of the frozen inputs: the test never rewrites the packaged R15 file, and a
        # check-dates or republish run in progress (outputs/date_check) cannot change what it reads
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(ROOT / "evidence" / "date_check_inputs", Path(tmp) / "evidence" / "date_check_inputs")
            (Path(tmp) / "outputs" / "research").mkdir(parents=True)
            for name in ("R11_predictions.csv", "R10_suppression_replay.csv"):
                shutil.copy(ROOT / "outputs" / "research" / name, Path(tmp) / "outputs" / "research" / name)
            g = run(Path(tmp)).set_index("group")
        self.assertEqual(int(g.loc["routine", "dates"]), 18)
        self.assertLess(g.loc["flagged (ELEVATED)", "value"], g.loc["routine", "value"])
        self.assertAlmostEqual(g.loc["shown, rise/fall call", "value"], 10 / 46, places=3)
        self.assertGreater(g.loc["shown, no-big-move call", "value"], g.loc["always 'no big move'", "value"])
        D = gather_sample(ROOT, date="2026-09-21")
        D["narrative"] = {"mode": "template", "summary": None, "cards": {}, "flags": []}
        self.assertEqual(D["findings"]["shown_dir_n"], 46)
        with tempfile.TemporaryDirectory() as tmp:
            body = report_text(render(D, tmp))
        self.assertIn("a ✓ never makes a direction likely", body)
        self.assertIn("Does ⚑ mean a bigger move?", body)
        self.assertNotIn("only ✓ calls carry information", body)
        self.assertNotIn("firm view", flag_reason({"archetype": "model_only", "t6_class": "bearish"}))

    def test_republish_rebuilds_every_stored_date_into_the_archive(self):
        """republish = check-dates + publish: each stored date lands in reports/<date>/ in the
        current format, and the index lists it (database mocked with the saved 21 Sep copy)."""
        from unittest.mock import patch
        from sgx_capstone.daily_report import gather_sample
        from sgx_capstone.automation import check_dates

        class Conn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **k): return [("2026-09-21",)]

        class Engine:
            def connect(self): return Conn()

        cfg, _ = load_config()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("sgx_capstone.daily_report.gather_live", lambda ns, c, d: gather_sample(ROOT, date=d)), \
                patch("sgx_capstone.automation.to_pdf", lambda p: None):
            summary = check_dates({"SUPABASE_ENGINE": Engine()}, cfg, "2026-09-21", "2026-09-21", Path(tmp),
                                  publish_all=True)
            day = Path(tmp) / "reports" / "2026-09-21"
            self.assertTrue(any(day.glob("T7_Daily_Signal_Report_*.docx")))
            self.assertTrue(any(day.glob("T7_reviewer_notes_*.md")))
            self.assertIn("2026-09-21", (Path(tmp) / "reports" / "README.md").read_text())
            self.assertIn("True", summary.read_text())

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
