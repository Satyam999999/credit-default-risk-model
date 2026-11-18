"""Generate the ~2-page non-technical comparison report from results/*.csv (no hand-typed numbers).

    python -m creditrisk.report
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C


def _fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def main():
    R = C.RESULTS_DIR
    summ = pd.read_csv(R / "cv_summary.csv")
    paired = pd.read_csv(R / "cv_paired_vs_baseline.csv") if (R / "cv_paired_vs_baseline.csv").exists() else pd.DataFrame()
    cfg = json.loads((R / "run_config.json").read_text())
    c_fn, c_fp = cfg["c_fn"], cfg["c_fp"]
    models = list(dict.fromkeys(summ.model))
    g = lambda m, met: summ[(summ.model == m) & (summ.metric == met)].iloc[0]

    lines = ["# Credit-default risk: comparison of scoring models", ""]
    if cfg["source"] == "synthetic":
        lines += ["> **SYNTHETIC DATA: smoke-test output only. These numbers mean nothing.**", ""]
    best = min(models, key=lambda m: g(m, "cost_per_customer")["mean"])
    lines += ["## Summary", "",
              f"We compared the existing scorecard-style model (logistic regression) with {len(models)-1} machine-learning "
              f"alternative{'s' if len(models) > 2 else ''} on {cfg['n_clean']:,} credit-card customers, predicting who will miss next month's payment. "
              f"On the assumed cost of mistakes, the lowest-cost model was **{C.MODEL_LABELS[best]}**."
              if len(models) > 1 else f"Only {C.MODEL_LABELS[models[0]]} was evaluated.", ""]

    lines += ["## What the baseline does", "",
              "Logistic regression gives each customer a score by adding up weighted signals (recent late payments, share of bills "
              "paid, credit use, limit, age). Each weight can be read and audited, which is why scorecards are standard in "
              "lending. It can only add effects up; it cannot capture interactions such as 'late payments matter more for "
              "customers who also use most of their limit' unless someone builds them in by hand.", ""]

    lines += ["## How the models compare", "",
              f"Numbers are the average over {cfg['outer']} cross-validation folds with a 95% interval across folds "
              "(indicative; folds share data, so true uncertainty is somewhat larger). Higher is better except cost.", "",
              "| Model | ROC-AUC | PR-AUC | Recall | Precision | Cost per customer |", "|---|---|---|---|---|---|"]
    for m in models:
        lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(
            _fmt(g(m, k)["mean"], g(m, k)["ci_low"], g(m, k)["ci_high"]) for k in
            ("roc_auc", "pr_auc", "recall", "precision", "cost_per_customer")) + " |")
    lines.append("")
    if len(paired):
        lines += ["**Improvement over the baseline** (same folds, difference with 95% interval; an interval that excludes 0 "
                  "is a consistent difference):", "", "| Model | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |", "|---|---|---|---|---|"]
        for m in [x for x in models if x != "lr"]:
            q = lambda met: paired[(paired.model == m) & (paired.metric == met)].iloc[0]
            lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(
                _fmt(q(k).mean_diff_vs_baseline, q(k).ci_low, q(k).ci_high, 4)
                for k in ("roc_auc", "pr_auc", "recall", "cost_per_customer")) + " |")
        lines.append("")

    lines += ["## Where we draw the line", "",
              f"A model outputs a probability; we need a cut-off above which a customer is flagged. **Assumption (not measured "
              f"from data): a missed default costs {c_fn:g} times as much as a false alarm** (cost {c_fn:g} vs {c_fp:g}). "
              "The cut-off is chosen on training data only to minimise that cost. If the real ratio differs, the cut-off "
              "and the 'cost per customer' column change; the ranking of models is the more robust finding.", "",
              "![Gain and lift](../results/figures/gain_lift.png)", ""]

    shap_p = R / "shap_summary.csv"
    if shap_p.exists():
        t = pd.read_csv(shap_p).head(5)
        lines += ["## What drives risk", ""]
        for r in t.itertuples():
            base = r.feature.split("_missingindicator")[0]
            desc = C.FEATURE_DESCRIPTIONS.get(base, r.feature)
            way = "higher values raise risk" if r.direction_corr > 0.1 else (
                "higher values lower risk" if r.direction_corr < -0.1 else "effect is not one-directional")
            lines.append(f"- **{desc}**: {way}.")
        chk = R / "business_sense_check.json"
        if chk.exists():
            c = json.loads(chk.read_text())
            lines += ["", "Sense check: recent late payments " + ("are among the top three drivers, as expected."
                      if c.get("delinquency_in_top3") else "are NOT among the top three drivers; investigate before trusting the model.")]
        lines.append("")

    if len(models) > 1:
        lines += ["## Recommendation", ""]
        if len(paired):
            r = paired[(paired.model == best) & (paired.metric == "cost_per_customer")]
            if len(r) and r.iloc[0].ci_high < 0:
                lines.append(f"Adopt **{C.MODEL_LABELS[best]}** (at least as a challenger run beside the scorecard): its cost is "
                             "consistently lower than the baseline across folds. Keep the baseline as the audit reference.")
            else:
                lines.append(f"**{C.MODEL_LABELS[best]}** has the lowest average cost, but the gap to the baseline is not "
                             "consistent across folds. Keep the transparent baseline unless a larger validation shows a clear gain.")
        lines.append("")

    lines += ["## Limitations", "",
              "- One dataset from one bank in one country (Taiwan, 2005); results may not transfer.",
              "- Random cross-validation, **no time-based validation**: we have not tested performance on later periods or under economic change.",
              "- **No fairness or regulatory analysis.** Sex, age and marital status are in the data; using them may be restricted in real lending.",
              "- The cost ratio is an assumption; real costs vary by exposure and recovery.",
              "- 'Days past due' is approximated by the dataset's repayment-status codes (months of delay)."]
    C.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    md_path = C.REPORTS_DIR / "comparison_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    export_pdf(md_path, C.REPORTS_DIR / "comparison_report.pdf")


def export_pdf(md_path: C.Path, pdf_path: C.Path):
    import os, shutil, subprocess
    try:
        import markdown
    except ImportError:
        return
    chrome = None
    for cand in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chrome"),
    ]:
        if cand and os.path.exists(cand):
            chrome = cand
            break
    if not chrome:
        return

    md_text = md_path.read_text()
    abs_fig = os.path.abspath(C.FIGURES_DIR / "gain_lift.png")
    md_text = md_text.replace("../results/figures/gain_lift.png", "file://" + abs_fig)
    html_content = markdown.markdown(md_text, extensions=["tables"])
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 12mm 14mm 12mm 14mm; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.5pt; line-height: 1.35; color: #222; margin: 0; padding: 0;
  }}
  h1 {{ font-size: 15pt; margin: 0 0 8px 0; color: #111; }}
  h2 {{ font-size: 11pt; margin: 10px 0 4px 0; color: #1a365d; border-bottom: 1px solid #e2e8f0; padding-bottom: 2px; }}
  p {{ margin: 4px 0 6px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 6px 0 8px 0; font-size: 8.5pt; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 4px 6px; text-align: left; }}
  th {{ background-color: #f1f5f9; font-weight: 600; }}
  ul {{ margin: 4px 0 6px 0; padding-left: 18px; }}
  li {{ margin-bottom: 2px; }}
  img {{ max-width: 80%; max-height: 155px; display: block; margin: 4px auto; }}
  .page-break {{ page-break-before: always; }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""
    html_doc = html_doc.replace("<h2>Where we draw the line</h2>", "<div class=\"page-break\"></div><h2>Where we draw the line</h2>")
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html_doc)
    try:
        subprocess.run([
            chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            "--user-data-dir=/tmp/chrome_user_data",
            f"--print-to-pdf={pdf_path}", str(html_path.resolve())
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {pdf_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
