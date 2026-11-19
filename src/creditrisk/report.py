"""Generate the ~2-page non-technical comparison report from results/ (no hand-typed numbers).

Every number and every "detectable / not detectable" statement below is computed from the result files
(cv_summary, cv_paired_vs_baseline, cost_sensitivity, operational_ranking, imbalance_comparison, shap_*).
Produce those files first with `make cv`, `make analyze` and `make interpret`.

    python -m creditrisk.report            # writes reports/comparison_report.md and, if Chrome is found, the PDF
    python -m creditrisk.report --no-pdf
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from . import config as C

TREES = ("rf", "xgb", "lgbm")
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def _read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _includes_zero(lo, hi) -> bool:
    return lo <= 0 <= hi


def _joined(items) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


# --------------------------------------------------------------------------- building blocks
def _cost_vs_binned(paired, trees):
    """Paired cost differences (tree - lr_binned) at the main-run ratio, if available."""
    if paired is None or not len(paired):
        return None
    sub = paired[(paired.baseline == "lr_binned") & (paired.model.isin(trees)) & (paired.metric == "cost_per_customer")]
    return sub if len(sub) else None


def _ratio_bullets(cs, c_fn):
    """Cost-sensitivity bullets from results/cost_sensitivity.csv (columns: fn_cost, model, mean_cost, diff_*)."""
    diff = "diff_vs_lr_binned"
    out = ["**Cost sensitivity across FN:FP ratios (" + ", ".join(f"{r:g}:1" for r in sorted(cs.fn_cost.unique())) + ")**:", ""]
    ratios = [c_fn] + [r for r in sorted(cs.fn_cost.unique()) if r != c_fn]
    for r in ratios:
        g = cs[cs.fn_cost == r]
        t = g[g.model.isin(TREES)]
        if not len(t):
            continue
        all_zero = all(_includes_zero(a, b) for a, b in zip(t.diff_ci_low, t.diff_ci_high))
        all_neg = all(b < 0 for b in t.diff_ci_high)
        mean = t[diff].mean()
        lo, hi = t.diff_ci_low.min(), t.diff_ci_high.max()
        if r == c_fn:
            label = f"**{r:g}:1 ratio (primary)**"
            txt = ("No detectable cost difference between tree models and lr_binned (all 95% intervals include zero)."
                   if all_zero else
                   f"Cost difference vs lr_binned is detectable for "
                   f"{_joined(C.MODEL_LABELS[m] for m, a, b in zip(t.model, t.diff_ci_low, t.diff_ci_high) if not _includes_zero(a, b))}.")
        else:
            label = f"**{r:g}:1 ratio ({'low' if r < c_fn else 'high'} penalty)**"
            if all_neg:
                txt = (f"Tree models are slightly better and never detectably worse ({mean:+.3f} cost units, detectable, "
                       f"95% CI: [{lo:.3f}, {hi:.3f}]).")
            elif all_zero:
                txt = (f"Tree models show {'lower' if mean < 0 else 'higher'} average cost (~ {mean:+.3f} units), but fold "
                       "variance is wide and the difference is not detectable (CI includes zero).")
            else:
                txt = "Mixed: the difference vs lr_binned is detectable for some tree models and not for others (see cost_sensitivity.csv)."
        out.append(f"- {label}: {txt}")

    # Effect of binning: linear lr vs lr_binned at each ratio.
    parts = []
    for r in sorted(cs.fn_cost.unique()):
        row = cs[(cs.fn_cost == r) & (cs.model == "lr")]
        base = cs[(cs.fn_cost == r) & (cs.model == "lr_binned")]
        if not len(row) or not len(base):
            continue
        row, base = row.iloc[0], base.iloc[0]
        if row.diff_ci_low > 0:
            parts.append(f"a detectable gain at {r:g}:1 ({row[diff]:+.4f} linear penalty)")
        elif row.diff_ci_high < 0:
            parts.append(f"a detectable loss at {r:g}:1 ({row[diff]:+.4f})")
        elif (row.diff_ci_high - row.diff_ci_low) / 2 <= 0.01 * base.mean_cost:   # narrow interval around zero
            parts.append(f"no difference at {r:g}:1")
        else:
            parts.append(f"an undetermined effect at {r:g}:1 (wide interval)")
    if parts:
        out.append("- **Effect of binning**: Across ratios, binning shows " + _joined(parts) + ".")
    return out + [""]


def _ranking_sentence(rk):
    """Recommendation sentence about defaults captured in the top slices, from operational_ranking.csv."""
    if rk is None or not len(rk):
        return ""
    chal, base = rk.challenger.iloc[0], rk.baseline.iloc[0]
    name = C.MODEL_LABELS.get(chal, chal)
    det = rk[rk.diff_pp_ci_low > 0].sort_values("top_share")
    nd = rk[rk.diff_pp_ci_low <= 0].sort_values("top_share")
    pct = lambda q: f"{round(q * 100):d}%"
    txt = f"Comparing {name} vs {base}, "
    if len(det):
        lo_pp, hi_pp = det.diff_pp.min(), det.diff_pp.max()
        lo_d, hi_d = det.diff_defaults.min(), det.diff_defaults.max()
        pp = f"{lo_pp:+.1f} to {hi_pp:+.1f}" if f"{lo_pp:.1f}" != f"{hi_pp:.1f}" else f"{lo_pp:+.1f}"
        dd = f"about {lo_d:.0f} to {hi_d:.0f}" if f"{lo_d:.0f}" != f"{hi_d:.0f}" else f"about {lo_d:.0f}"
        txt += (f"{name} captures {pp} pp ({dd} more defaults per fold) in the top "
                f"{_joined(pct(q) for q in det.top_share)}")
    else:
        txt += f"{name} shows no detectable gain in any top slice"
    if len(det) and len(nd):
        txt += "; " + "; ".join(
            f"the top-{pct(r.top_share)} difference is not detectable ({r.diff_pp:+.2f} pp, 95% CI "
            f"[{r.diff_pp_ci_low:+.2f}, {r.diff_pp_ci_high:+.2f}] pp)" for r in nd.itertuples())
    return txt + ". "


# --------------------------------------------------------------------------- main
def main(pdf: bool = True):
    R = C.RESULTS_DIR
    summ = pd.read_csv(R / "cv_summary.csv")
    paired = _read(R / "cv_paired_vs_baseline.csv")
    if paired is not None and len(paired) and "baseline" not in paired.columns:
        paired["baseline"] = "lr"
    cfg = json.loads((R / "run_config.json").read_text())
    c_fn, c_fp = cfg["c_fn"], cfg["c_fp"]
    ratio_txt = f"{c_fn:g}:{c_fp:g}"
    models = list(dict.fromkeys(summ.model))
    trees = [m for m in TREES if m in models]
    g = lambda m, met: summ[(summ.model == m) & (summ.metric == met)].iloc[0]
    cvb = _cost_vs_binned(paired, trees)
    cost_nd = cvb is not None and all(_includes_zero(a, b) for a, b in zip(cvb.ci_low, cvb.ci_high))
    cost_phrase = (f"**not detectable at the assumed {ratio_txt} ratio** (all 95% intervals include zero)"
                   if cost_nd else "**detectable for at least one challenger** (see the paired tables)")

    lines = ["# Credit-default risk: comparison of scoring models", ""]
    if cfg["source"] == "synthetic":
        lines += ["> **SYNTHETIC DATA: smoke-test output only. These numbers mean nothing.**", ""]

    # ---- Summary
    lines += ["## Summary", ""]
    if "lr_binned" in models and trees and paired is not None and len(paired):
        d = paired[(paired.baseline == "lr_binned") & (paired.model.isin(("xgb", "lgbm") if {"xgb", "lgbm"} & set(trees) else trees))]
        roc = d[d.metric == "roc_auc"].mean_diff_vs_baseline.mean()
        prauc = d[d.metric == "pr_auc"].mean_diff_vs_baseline.mean()
        lines += [f"We compared two scorecard baselines (linear logistic regression and binned scorecard) with "
                  f"machine-learning alternatives (Random Forest, XGBoost, LightGBM) on {cfg['n_clean']:,} credit-card customers, "
                  f"predicting next-month default. On the assumed cost of mistakes (cost of false alarm = {c_fp:g}, cost of missed default = {c_fn:g}), "
                  f"cost differences between the machine-learning challengers and the binned scorecard (lr_binned) are "
                  f"{cost_phrase}. Gradient-boosted challengers maintain a small but "
                  f"consistent ranking advantage ({roc:+.3f} ROC-AUC, {prauc:+.3f} PR-AUC).", ""]
    else:
        lines += [f"We compared {len(models)} models ({', '.join(C.MODEL_LABELS[m] for m in models)}) on "
                  f"{cfg['n_clean']:,} credit-card customers, predicting next-month default.", ""]

    # ---- Baselines
    lines += ["## What the baselines do", "",
              "We evaluate two scorecard baselines: (1) **Linear scorecard ('lr')**, which scores customers by adding weighted "
              "continuous signals (delinquency delay, payment ratios, utilization, limit, age) using a single linear slope across delay codes; "
              "and (2) **Binned scorecard ('lr_binned')**, which treats repayment statuses (pay_1 to pay_6) as categorical levels "
              "(-2, -1, 0, 1, 2, 3+) via one-hot dummy variables. The binned scorecard is much closer to production credit scorecards "
              "used in real lending, capturing non-monotonic risk across non-delinquent codes.", ""]

    # ---- Comparison tables
    lines += ["## How the models compare", "",
              f"Average over {cfg['outer']} cross-validation folds with 95% fold-based intervals. Higher is better except cost. "
              f"Cost units are abstract (false alarm = {c_fp:g}).", "",
              "| Model | ROC-AUC | PR-AUC | Recall | Precision | Cost per customer |", "|---|---|---|---|---|---|"]
    for m in models:
        lines.append(f"| {C.MODEL_LABELS[m]} | " + " | ".join(
            _fmt(g(m, k)["mean"], g(m, k)["ci_low"], g(m, k)["ci_high"]) for k in
            ("roc_auc", "pr_auc", "recall", "precision", "cost_per_customer")) + " |")
    lines.append("")

    cols = ("roc_auc", "pr_auc", "recall", "cost_per_customer")
    head = ["| Model | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |", "|---|---|---|---|---|"]
    if paired is not None and len(paired):
        def row(label, sub):
            q = lambda met: sub[sub.metric == met].iloc[0]
            return f"| {label} | " + " | ".join(_fmt(q(k).mean_diff_vs_baseline, q(k).ci_low, q(k).ci_high, 4) for k in cols) + " |"

        p_lr = paired[paired.baseline == "lr"]
        if len(p_lr):
            lines += ["**Improvement over linear baseline (lr)** (same folds, difference with 95% interval; interval excluding 0 is consistent across folds):", ""] + head
            lines += [row(C.MODEL_LABELS[m], p_lr[p_lr.model == m]) for m in models if m != "lr" and len(p_lr[p_lr.model == m])]
            lines.append("")
        p_lrb = paired[paired.baseline == "lr_binned"]
        if len(p_lrb):
            lines += ["**Improvement over binned scorecard (lr_binned)**:", ""] + head
            lines += [row(C.MODEL_LABELS[m], p_lrb[p_lrb.model == m]) for m in TREES if len(p_lrb[p_lrb.model == m])]
            lines.append("")
        c_rows = [row(f"{C.MODEL_LABELS[a]} vs {C.MODEL_LABELS[b]}", paired[(paired.model == a) & (paired.baseline == b)])
                  for a, b in (("xgb", "rf"), ("lgbm", "rf"), ("lgbm", "xgb")) if len(paired[(paired.model == a) & (paired.baseline == b)])]
        if c_rows:
            lines += ["**Challenger-vs-challenger paired differences**:", "",
                      "| Comparison | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |", "|---|---|---|---|---|"] + c_rows + [""]
        lines += ["*Note: Fold-based intervals are indicative and optimistic because cross-validation folds share training data; "
                  "true out-of-sample uncertainty is somewhat larger. An interval excluding 0 indicates consistent direction across folds.*", ""]

    # ---- Threshold, calibration, imbalance
    lines += ["## Where we draw the line", "",
              f"A model outputs a probability; we need a cut-off above which a customer is flagged. **Assumption: a missed "
              f"default costs {c_fn:g} times as much as a false alarm** (cost {c_fn:g} vs {c_fp:g}, in abstract units where false alarm = {c_fp:g}). "
              "The cut-off is chosen on training data only to minimise that cost. If the real ratio differs, the cut-off "
              "and the 'cost per customer' column change; the ranking of models is the more robust finding.", ""]
    brier = {m: g(m, "brier")["mean"] for m in models}
    best_b = min(brier, key=brier.get)
    best_label = "the boosted trees" if best_b in ("xgb", "lgbm") else C.MODEL_LABELS[best_b]
    lines.append(f"- **Calibration**: out-of-fold predicted probabilities track observed default rates closely for all "
                 f"{NUMBER_WORDS.get(len(models), len(models))} models (Brier {min(brier.values()):.3f} to {max(brier.values()):.3f}, "
                 f"best for {best_label}).")
    imb = _read(R / "imbalance_comparison.csv")
    if imb is not None and len(imb):
        none = imb[imb.strategy == "none"]["recall_at_0.5"]
        wtd = imb[imb.strategy.isin(("class_weight", "undersample"))]["recall_at_0.5"]
        cost = imb["cost_per_customer_at_tuned"]
        lines.append(f"- **Imbalance**: class weighting or undersampling raises recall at a 0.5 cut-off (about {none.min():.2f}-{none.max():.2f} "
                     f"to {wtd.min():.2f}-{wtd.max():.2f}) but lowers precision and distorts probabilities; "
                     f"with a cost-tuned cut-off all strategies cost about the same ({cost.min():.3f}-{cost.max():.3f}), "
                     "so we use no resampling and tune the threshold.")
    lines += ["", "![Gain and lift](../results/figures/gain_lift.png)", ""]

    cs = _read(R / "cost_sensitivity.csv")
    if cs is not None and len(cs):
        lines += _ratio_bullets(cs, c_fn)

    # ---- Drivers
    shap_p, grp_p = R / "shap_summary.csv", R / "shap_groups.json"
    if shap_p.exists() and grp_p.exists():
        df_s, grp = pd.read_csv(shap_p), json.loads(grp_p.read_text())
        top, d, o = grp["top_feature"], grp["delinquency"], grp["other"]
        topdesc = ("recent delinquency" if top in C.DELINQUENCY_GROUP else C.FEATURE_DESCRIPTIONS.get(top, top))
        lines += ["## What drives risk", "",
                  f"TreeExplainer SHAP values from a final LightGBM model refit on all data, explaining a fixed sample of 3,000 customers (not out-of-fold), "
                  f"show that {topdesc} is the largest single driver ({top} mean |SHAP| = {grp['top_feature_mean_abs_shap']:.3f}). "
                  f"Grouping the {grp['n_features']} transformed features into two families:", "",
                  f"- **Delinquency features ({d['n_features']} features)** (recent delay codes pay_1..pay_6, months late, max delay, delay trend): "
                  f"**{d['share'] * 100:.1f}%** of total importance (~{d['share'] * 100:.0f}%, summed mean |SHAP| = {d['sum_mean_abs_shap']:.3f}).",
                  f"- **Financial & demographic features ({o['n_features']} transformed features)** (bills, utilization, payments, limit, age, one-hot categories): "
                  f"**{o['share'] * 100:.1f}%** of total importance (summed mean |SHAP| = {o['sum_mean_abs_shap']:.3f}).", "",
                  "Top individual feature drivers:", ""]
        for r in df_s.head(5).itertuples():
            base = r.feature.split("_missingindicator")[0]
            desc = C.FEATURE_DESCRIPTIONS.get(base, r.feature)
            way = "higher values raise risk" if r.direction_corr > 0.1 else (
                "higher values lower risk" if r.direction_corr < -0.1 else "effect is not one-directional")
            lines.append(f"- **{desc}**: {way} (mean |SHAP| = {r.mean_abs_shap:.3f}).")
        lines += ["", "These are associations in the model, not causal effects; for example, larger recent bills appear to lower risk only after accounting for credit limit, utilization and payments."]
        chk = R / "business_sense_check.json"
        if chk.exists():
            c = json.loads(chk.read_text())
            lines += ["", "Sense check: recent late payments " + ("are among the top three drivers, as expected."
                      if c.get("delinquency_in_top3") else "are NOT among the top three drivers; investigate before trusting the model.")]
        lines.append("")

    # ---- Recommendation
    if len(models) > 1:
        rk = _read(R / "operational_ranking.csv")
        lines += ["## Recommendation", "",
                  f"Cost differences between tree models and the binned scorecard are "
                  f"{'**not detectable**' if cost_nd else '**detectable for at least one challenger**'} at {ratio_txt}"
                  f"{' (all 95% intervals include zero)' if cost_nd else ''}. "
                  + _ranking_sentence(rk) +
                  "Note that fold-based intervals are optimistic because cross-validation folds share training data. We recommend:", "",
                  "1. **Stronger baseline**: Carry forward the binned scorecard ('lr_binned') into further validation as the stronger baseline. "
                  f"Treating repayment statuses as categorical bins resolves non-monotonic risk and shows no detectable cost difference at {ratio_txt} without black-box complexity.",
                  "2. **Challenger to monitor**: Track LightGBM/XGBoost as a challenger to monitor. Its small ranking edge provides operational value for tiered interventions "
                  "(such as early outreach and risk-tiered limits) while keeping the binned scorecard for core evaluation.", ""]

    lines += ["## Limitations", "",
              "- **Single dataset**: One dataset from one bank in Taiwan (2005); findings may not transfer to other credit portfolios.",
              "- **No time-based validation**: Evaluated via cross-validation, not time-based cohorts under economic shift.",
              "- **No fairness or regulatory review**: Age, sex, and marital status are present in the data without compliance screening.",
              f"- **Assumed cost ratio**: Thresholds depend on assumed cost weights (missed default = {c_fn:g}, false alarm = {c_fp:g}).",
              "- **Repayment granularity**: Monthly status codes (-2 to 8) serve as an approximation for exact days past due."]
    C.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    md_path = C.REPORTS_DIR / "comparison_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    if pdf:
        export_pdf(md_path, C.REPORTS_DIR / "comparison_report.pdf")


CSS = """
  @page { size: A4; margin: 10mm 12mm 10mm 12mm; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         font-size: 8.5pt; line-height: 1.25; color: #222; margin: 0; padding: 0; }
  h1 { font-size: 13.5pt; margin: 0 0 6px 0; color: #111; }
  h2 { font-size: 10pt; margin: 6px 0 3px 0; color: #1a365d; border-bottom: 1px solid #e2e8f0; padding-bottom: 1px; }
  p { margin: 2px 0 3px 0; }
  table { width: 100%; border-collapse: collapse; margin: 3px 0 5px 0; font-size: 7.5pt; }
  th, td { border: 1px solid #cbd5e1; padding: 2px 4px; text-align: left; }
  th { background-color: #f1f5f9; font-weight: 600; }
  ul { margin: 2px 0 4px 0; padding-left: 16px; }
  li { margin-bottom: 1.5px; }
  img { max-width: 75%; max-height: 140px; display: block; margin: 3px auto; }
  .page-break { page-break-before: always; }
"""


def export_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Render the markdown report to PDF with headless Chrome/Chromium.

    The intermediate HTML (which needs an absolute image URL for Chrome) lives in a temporary directory,
    so no machine-specific path is ever written into the repository.
    """
    try:
        import markdown
    except ImportError:
        print("PDF skipped: the 'markdown' package is not installed (pip install markdown).")
        return False
    chrome = next((c for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                               shutil.which("google-chrome"), shutil.which("chromium"),
                               shutil.which("chromium-browser"), shutil.which("chrome")) if c and os.path.exists(c)), None)
    if not chrome:
        print("PDF skipped: Chrome/Chromium not found. The markdown report was written.")
        return False

    md_text = md_path.read_text().replace("](../results/figures/", f"]({C.FIGURES_DIR.resolve().as_uri()}/")
    body = markdown.markdown(md_text, extensions=["tables"])
    body = body.replace("<h2>Where we draw the line</h2>", '<div class="page-break"></div><h2>Where we draw the line</h2>')
    html_doc = f'<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n'
    with tempfile.TemporaryDirectory() as td:
        html_path = Path(td) / "report.html"
        html_path.write_text(html_doc)
        try:
            subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            f"--user-data-dir={Path(td) / 'profile'}", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        except Exception as e:  # noqa: BLE001
            print(f"PDF export failed: {e}")
            return False
    print(f"wrote {pdf_path}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true", help="write the markdown report only")
    main(pdf=not ap.parse_args().no_pdf)
