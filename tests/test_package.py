"""Offline checks against the supplied historical evidence."""
import csv
from pathlib import Path
import unittest
from sgx_capstone.configuration import ROOT, load_config


class PackageChecks(unittest.TestCase):
    def test_config_paths_and_versions(self):
        config, sha = load_config()
        self.assertEqual(config["t5"]["model_version"], "v6")
        self.assertEqual(config["t6"]["expected_forward_horizon"], 3)
        self.assertTrue(Path(config["paths"]["t6_artifacts"]).is_file())
        self.assertEqual(len(sha), 12)

    def test_r11_claims_match_retained_results(self):
        path = ROOT / "outputs" / "research" / "R11_summary.csv"
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        selected = [r for r in rows if r["target"] == "relative" and r["model"] == "news_logreg"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected[0]["days"]), 22)
        self.assertAlmostEqual(float(selected[0]["IC"]), 0.111, delta=0.01)
        self.assertAlmostEqual(float(selected[0]["spread%"]), 0.177, delta=0.01)
        with (ROOT / "outputs" / "research" / "R11_verdicts.csv").open(newline="") as f:
            verdicts = list(csv.DictReader(f))
        self.assertTrue(verdicts)
        self.assertTrue(all(r["verdict"] != "PROMOTE-CANDIDATE" for r in verdicts))

    def test_serving_code_excludes_research_and_old_model_training(self):
        entry = (ROOT / "sgx_capstone" / "__main__.py").read_text()
        self.assertNotIn('_execute("r11"', entry)
        self.assertNotIn('_execute("task5_part_a"', entry)
        self.assertIn('t6_preflight', entry)

    def test_r13_consistency_uses_f1_per_stock(self):
        with (ROOT / "outputs" / "research" / "R13_champion_consistency.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 15)
        self.assertEqual({r["pooled_winner"] for r in rows}, {"XGBoost"})
        leaders = sum(r["winner_leads_f1"] == "True" for r in rows)
        self.assertEqual(leaders, 4)
        self.assertLess(leaders / len(rows), 0.8)


if __name__ == "__main__":
    unittest.main()
