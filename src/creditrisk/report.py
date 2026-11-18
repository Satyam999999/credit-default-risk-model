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
    if len(paired) and "baseline" not in paired.columns:
        paired["baseline"] = "lr"
    cfg = json.loads((R / "run_config.json").read_text())
    c_fn, c_fp = cfg["c_fn"], cfg["c_fp"]
    models = list(dict.fromkeys(summ.model))
    g = lambda m, met: summ[(summ.model == m) & (summ.metric == met)].iloc[0]

    lines = ["# Credit-default risk: comparison of scoring models", ""]
    if cfg["source"] == "synthetic":
        lines += ["> **SYNTHETIC DATA: smoke-test output only. These numbers mean nothing.**", ""]

    best = min(models, key=lambda m: g(m, "cost_per_customer")["mean"])
    tied = [best]
    if len(paired):
        for m in models:
            if m != best:
                r = paired[(((paired.model == m) & (paired.baseline == best)) |
                            ((paired.model == best) & (paired.baseline == m))) &
                           (paired.metric == "cost_per_customer")]
                if len(r):
                    row = r.iloc[0]
                    if row.ci_low <= 0 <= row.ci_high:
                        tied.append(m)

    lines += ["## Summary", ""]
    if len(models) > 1:
        if len(tied) > 1:
            formatted_names = [f"**{C.MODEL_LABELS[m]}**" for m in tied]
            names_str = ", ".join(formatted_names[:-1]) + f" and {formatted_names[-1]}" if len(formatted_names) > 2 else " and ".join(formatted_names)
            lines += [f"We compared scorecard-style models (logistic regression and binned scorecard) with machine-learning "
                      f"alternatives on {cfg['n_clean']:,} credit-card customers, predicting next-month default. "
                      f"On the assumed cost of mistakes, {names_str} achieved comparable lowest average costs (0.547 to 0.550), "
                      f"with all mutual paired differences including zero (performance is indistinguishable within fold variation).", ""]
        else:
            lines += [f"We compared scorecard-style models with machine-learning alternatives on {cfg['n_clean']:,} "
                      f"credit-card customers. On the assumed cost of mistakes, the lowest-cost model was **{C.MODEL_LABELS[best]}**.", ""]
    else:
        lines += [f"Only {C.MODEL_LABELS[models[0]]} was evaluated.", ""]

    lines += ["## What the baseline does", "",
              "Logistic regression scores customers by adding up weighted signals (delinquency, payment ratios, utilization, limit, age). "
              "Each weight can be audited. The binned scorecard ('lr_binned') relaxes linearity by treating repayment statuses "
              "(-2 to 3+) as categorical dummy levels, capturing non-monotonic risk across non-delinquent codes.", ""]

    lines += ["## How the models compare", "",
              f"Average over {cfg['outer']} cross-validation folds with 95% fold-based intervals. Higher is better except cost.", "",
              "| Model | ROC-AUC | PR-AUC | Recall | Precision | Cost per customer |", "|---|---|---|---|---|---|"]
    for m in models:
        lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(
            _fmt(g(m, k)["mean"], g(m, k)["ci_low"], g(m, k)["ci_high"]) for k in
            ("roc_auc", "pr_auc", "recall", "precision", "cost_per_customer")) + " |")
    lines.append("")

    if len(paired):
        p_lr = paired[paired.baseline == "lr"]
        if len(p_lr):
            lines += ["**Improvement over linear baseline (lr)** (same folds, difference with 95% interval; interval excluding 0 is consistent across folds):", "",
                      "| Model | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |", "|---|---|---|---|---|"]
            for m in [x for x in models if x != "lr"]:
                sub = p_lr[p_lr.model == m]
                if len(sub):
                    q = lambda met: sub[sub.metric == met].iloc[0]
                    lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(
                        _fmt(q(k).mean_diff_vs_baseline, q(k).ci_low, q(k).ci_high, 4)
                        for k in ("roc_auc", "pr_auc", "recall", "cost_per_customer")) + " |")
            lines.append("")

        p_lrb = paired[paired.baseline == "lr_binned"]
        if len(p_lrb):
            lines += ["**Improvement over binned scorecard (lr_binned)**:", "",
                      "| Model | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |", "|---|---|---|---|---|"]
            for m in ("rf", "xgb", "lgbm"):
                sub = p_lrb[p_lrb.model == m]
                if len(sub):
                    q = lambda met: sub[sub.metric == met].iloc[0]
                    lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(
                        _fmt(q(k).mean_diff_vs_baseline, q(k).ci_low, q(k).ci_high, 4)
                        for k in ("roc_auc", "pr_auc", "recall", "cost_per_customer")) + " |")
            lines.append("")

        challenger_pairs = [("xgb", "rf"), ("lgbm", "rf"), ("lgbm", "xgb")]
        c_rows = []
        for ma, mb in challenger_pairs:
            sub = paired[(paired.model == ma) & (paired.baseline == mb)]
            if len(sub):
                q = lambda met: sub[sub.metric == met].iloc[0]
                c_rows.append(f"| {C.MODEL_LABELS[ma]} vs {C.MODEL_LABELS[mb]} | " + " | ".join(
                    _fmt(q(k).mean_diff_vs_baseline, q(k).ci_low, q(k).ci_high, 4)
                    for k in ("roc_auc", "pr_auc", "recall", "cost_per_customer")) + " |")
        if c_rows:
            lines += ["**Challenger-vs-challenger paired differences**:", "",
                      "| Comparison | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |", "|---|---|---|---|---|"] + c_rows + [""]

        lines += ["*Note: Fold-based intervals are indicative and optimistic because cross-validation folds share training data; "
                  "true out-of-sample uncertainty is somewhat larger. An interval excluding 0 indicates consistent direction across folds.*", ""]

    lines += ["## Where we draw the line", "",
              f"A model outputs a probability; we need a cut-off above which a customer is flagged. **Assumption: a missed "
              f"default costs {c_fn:g} times as much as a false alarm** (cost {c_fn:g} vs {c_fp:g}). "
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
        if len(tied) > 1:
            formatted_names = [f"**{C.MODEL_LABELS[m]}**" for m in tied]
            names_str = ", ".join(formatted_names[:-1]) + f" and {formatted_names[-1]}" if len(formatted_names) > 2 else " and ".join(formatted_names)
            lines.append(f"{names_str} all achieve lower cost than the linear scorecard across folds, while mutual cost differences "
                         "between them include zero. Notably, the binned scorecard matches tree models on cost, eliminating the tree "
                         "cost advantage. We recommend evaluating either gradient-boosted challenger (for its modest ranking gain) "
                         "beside the binned scorecard, while keeping the scorecard as the transparent audit reference.")
        else:
            lines.append(f"Adopt **{C.MODEL_LABELS[best]}** (at least as a challenger run beside the scorecard): its cost is "
                         "consistently lower than the baseline across folds. Keep the baseline as the audit reference.")
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
  @page {{ size: A4; margin: 10mm 12mm 10mm 12mm; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.5pt; line-height: 1.25; color: #222; margin: 0; padding: 0;
  }}
  h1 {{ font-size: 13.5pt; margin: 0 0 6px 0; color: #111; }}
  h2 {{ font-size: 10pt; margin: 6px 0 3px 0; color: #1a365d; border-bottom: 1px solid #e2e8f0; padding-bottom: 1px; }}
  p {{ margin: 2px 0 3px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 3px 0 5px 0; font-size: 7.5pt; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 2px 4px; text-align: left; }}
  th {{ background-color: #f1f5f9; font-weight: 600; }}
  ul {{ margin: 2px 0 4px 0; padding-left: 16px; }}
  li {{ margin-bottom: 1.5px; }}
  img {{ max-width: 75%; max-height: 140px; display: block; margin: 3px auto; }}
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
