import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from creditrisk import config as C
from creditrisk import report as Rp

COMMITTED = C.REPORTS_DIR / "comparison_report.md"
HAVE_RESULTS = all((C.RESULTS_DIR / f).exists() for f in
                   ("cv_summary.csv", "cv_paired_vs_baseline.csv", "run_config.json", "cost_sensitivity.csv",
                    "operational_ranking.csv", "shap_groups.json", "imbalance_comparison.csv")) and COMMITTED.exists()


class TestReport(unittest.TestCase):
    @unittest.skipUnless(HAVE_RESULTS, "results not present")
    def test_committed_report_is_generated_from_committed_results(self):
        """Regenerating the report must reproduce reports/comparison_report.md exactly (no stale or typed numbers)."""
        with tempfile.TemporaryDirectory() as td, mock.patch.object(C, "REPORTS_DIR", Path(td)):
            Rp.main(pdf=False)
            new = (Path(td) / "comparison_report.md").read_text()
        self.assertEqual(new, COMMITTED.read_text())
        for bad in ("/Users/", "file://", "$"):
            self.assertNotIn(bad, new)

    def test_ranking_sentence_distinguishes_detectable_slices(self):
        rk = pd.DataFrame({"top_share": [0.1, 0.2, 0.3], "baseline": "lr_binned", "challenger": "lgbm",
                           "diff_pp": [0.77, 0.62, 0.90], "diff_pp_ci_low": [0.20, -0.17, 0.42],
                           "diff_pp_ci_high": [1.34, 1.41, 1.39], "diff_defaults": [10.2, 8.2, 12.0]})
        s = Rp._ranking_sentence(rk)
        self.assertIn("+0.8 to +0.9 pp", s)
        self.assertIn("top 10% and 30%", s)
        self.assertIn("top-20% difference is not detectable (+0.62 pp", s)
        none_detectable = rk.assign(diff_pp_ci_low=-1.0)
        self.assertIn("no detectable gain", Rp._ranking_sentence(none_detectable))

    def test_pdf_export_writes_nothing_into_repo_when_chrome_missing(self):
        with tempfile.TemporaryDirectory() as td:
            md = Path(td) / "r.md"
            md.write_text("# t\n")
            with mock.patch.object(Rp.os.path, "exists", return_value=False), mock.patch.object(Rp.shutil, "which", return_value=None):
                self.assertFalse(Rp.export_pdf(md, Path(td) / "r.pdf"))
            self.assertEqual(sorted(p.name for p in Path(td).iterdir()), ["r.md"])


if __name__ == "__main__":
    unittest.main()
