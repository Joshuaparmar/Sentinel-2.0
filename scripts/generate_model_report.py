
#!/usr/bin/env python3
"""Generate a detailed SENTINEL 2.0 model comparison report."""

from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "model_comparison"
FIG_DIR = REPORT_DIR / "figures"

# Look for the comparison CSV produced by the existing project.
CSV_CANDIDATES = [
    REPORT_DIR / "model_comparison.csv",
    ROOT / "reports" / "model_comparison.csv",
    ROOT / "reports" / "model_comparison" / "metrics.csv",
]

METRIC_ALIASES = {
    "model": ["model", "model_name", "algorithm"],
    "precision": ["precision"],
    "recall": ["recall"],
    "f1": ["f1", "f1_score"],
    "average_precision": [
        "pr_auc_average_precision",
        "average_precision",
        "pr_auc",
        "pr_auc_score",
    ],
    "roc_auc": ["roc_auc", "roc_auc_score", "auc"],
    "false_positive_rate": ["false_positive_rate", "fpr"],
    "tp": ["true_positives", "tp"],
    "fp": ["false_positives", "fp"],
    "fn": ["false_negatives", "fn"],
    "tn": ["true_negatives", "tn"],
    "threshold": ["threshold", "decision_threshold"],
}


def find_column(df, aliases):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def fmt(value, digits=4):
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        searched = "\n".join(str(p) for p in CSV_CANDIDATES)
        raise FileNotFoundError(
            "Could not find the model-comparison CSV.\n"
            "Searched:\n" + searched +
            "\nRun your existing model-comparison script first, "
            "or update CSV_CANDIDATES to match its output."
        )

    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"The results CSV is empty: {csv_path}")

    # Normalize common column names without changing the original CSV.
    data = pd.DataFrame(index=raw.index)
    for standard, aliases in METRIC_ALIASES.items():
        source = find_column(raw, aliases)
        if source is not None:
            data[standard] = raw[source]

    if "model" not in data:
        raise ValueError(
            "No model-name column found. Available columns: "
            + ", ".join(map(str, raw.columns))
        )

    numeric_metrics = [
        "precision", "recall", "f1", "average_precision",
        "roc_auc", "false_positive_rate", "tp", "fp", "fn",
        "tn", "threshold",
    ]
    for col in numeric_metrics:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["model"] = data["model"].astype(str)

    # Prefer average precision for rare-anomaly problems. Use F1,
    # precision, and recall as tie-breakers where those values exist.
    ranking_columns = [
        c for c in ["average_precision", "f1", "precision", "recall"]
        if data[c].notna().any()
    ]
    if not ranking_columns:
        raise ValueError("No usable evaluation metrics were found in the CSV.")

    # Sequence and row-based models may use different evaluation units.
    # If a unit/type column exists, preserve it and rank within each group.
    unit_source = find_column(
        raw, ["score_unit", "evaluation_unit", "model_type", "model_family"]
    )
    if unit_source:
        data["evaluation_group"] = raw[unit_source].fillna("unknown").astype(str)
    else:
        data["evaluation_group"] = np.where(
            data["model"].str.contains("lstm|cnn|sequence", case=False),
            "sequence models",
            "row-based models",
        )

    data["average_precision_rank"] = np.nan
    for _, indices in data.groupby("evaluation_group").groups.items():
        subset = data.loc[indices]
        if subset["average_precision"].notna().any():
            data.loc[indices, "average_precision_rank"] = (
                subset["average_precision"].rank(
                    ascending=False, method="min"
                )
            )

    # Select from row-based models when available. A model with a missing
    # primary metric is not silently assigned a zero score.
    row_models = data[
        data["evaluation_group"].str.contains(
            "row|all test|point", case=False, regex=True
        )
    ]
    candidates = row_models if not row_models.empty else data

    if candidates["average_precision"].notna().any():
        candidates = candidates[candidates["average_precision"].notna()]

    sort_columns = [
        c for c in [
            "average_precision", "f1", "precision", "recall"
        ] if candidates[c].notna().any()
    ]
    if sort_columns:
        candidates = candidates.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
            na_position="last",
        )

    winner = candidates.iloc[0]
    winner_name = winner["model"]

    # Save metric charts.
    charts = [
        ("average_precision", "Average precision (PR-AUC)",
         "Higher is better; useful for rare anomalies."),
        ("f1", "F1 score", "Balances precision and recall."),
        ("precision", "Precision", "Fraction of alerts that were true anomalies."),
        ("recall", "Recall", "Fraction of labelled anomalies detected."),
        ("roc_auc", "ROC-AUC", "Ranking quality across thresholds."),
        ("false_positive_rate", "False-positive rate",
         "Fraction of normal rows incorrectly flagged."),
    ]

    chart_paths = []
    for metric, title, explanation in charts:
        plot = data[["model", metric]].dropna().sort_values(metric)
        if plot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(4, 0.48 * len(plot))))
        ax.barh(plot["model"], plot[metric])
        ax.set_title(title)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
        fig.tight_layout()

        path = FIG_DIR / f"{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append((title, explanation, path))

    # Confusion-count chart, if the source provides these values.
    if data[["tp", "fp", "fn"]].notna().all(axis=1).any():
        plot = data.dropna(subset=["tp", "fp", "fn"]).copy()
        fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(plot))))
        y = np.arange(len(plot))
        left = np.zeros(len(plot))
        for col, label in [
            ("tp", "True positives"),
            ("fp", "False positives"),
            ("fn", "Missed anomalies"),
        ]:
            values = plot[col].to_numpy()
            ax.barh(y, values, left=left, label=label)
            left += values
        ax.set_yticks(y, plot["model"])
        ax.set_xlabel("Number of rows")
        ax.set_title("Detection outcomes")
        ax.legend()
        ax.invert_yaxis()
        fig.tight_layout()
        path = FIG_DIR / "detection_counts.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append((
            "Detection outcomes",
            "True detections, false alarms, and missed anomalies.",
            path,
        ))

    display_cols = [
        c for c in [
            "model", "evaluation_group", "precision", "recall", "f1",
            "average_precision", "roc_auc", "false_positive_rate",
            "tp", "fp", "fn", "threshold",
        ] if c in data
    ]
    table = data[display_cols].copy()
    for col in display_cols:
        if col not in ("model", "evaluation_group"):
            table[col] = table[col].map(fmt)
    table_html = table.to_html(index=False, escape=True, border=0)

    figures_html = "\n".join(
        f"<section><h3>{title}</h3><p>{note}</p>"
        f'<img src="figures/{path.name}" alt="{title}">'
        f"</section>"
        for title, note, path in chart_paths
    )

    metric_explanations = """
    <ul>
      <li><b>Precision:</b> Of the rows flagged, how many were truly anomalous.</li>
      <li><b>Recall:</b> Of the labelled anomalies, how many were detected.</li>
      <li><b>F1:</b> Combines precision and recall into one score.</li>
      <li><b>Average precision:</b> Summarizes the precision-recall curve and is
      particularly informative when anomalies are rare.</li>
      <li><b>ROC-AUC:</b> Measures how well scores rank positives above negatives
      over different thresholds.</li>
      <li><b>False-positive rate:</b> The fraction of normal rows incorrectly
      flagged as anomalous.</li>
    </ul>
    """

    reason_parts = []
    for metric, label in [
        ("average_precision", "average precision"),
        ("f1", "F1"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("false_positive_rate", "false-positive rate"),
    ]:
        if pd.notna(winner.get(metric)):
            reason_parts.append(f"{label} = {fmt(winner[metric])}")
    reason = "; ".join(reason_parts) or "highest-ranked available evaluation metrics"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SENTINEL 2.0 Model Comparison Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 30px auto;
padding: 0 22px; color: #202124; line-height: 1.55; }}
h1 {{ border-bottom: 3px solid #333; padding-bottom: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 18px; margin: 18px 0; }}
.warning {{ background: #fff8e6; padding: 12px; border-left: 4px solid #d99b00; }}
table {{ border-collapse: collapse; width: 100%; display: block; overflow-x: auto; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ width: 100%; height: auto; }}
section {{ margin: 28px 0; }}
</style>
</head>
<body>
<h1>SENTINEL 2.0 — Model Comparison Report</h1>
<p>Source results: <code>{csv_path.relative_to(ROOT)}</code></p>

<div class="warning"><b>Important:</b> The selected model is the top candidate
under the recorded metrics, not automatically production-ready. Results depend
on the test data, labels, training procedure, and threshold policy. Sequence
and row-based models may be evaluated on different units and should not be
compared directly.</div>

<h2>1. Selected model</h2>
<div class="card">
<h3>{winner_name}</h3>
<p><b>Selection basis:</b> {reason}</p>
<p>The default ranking prioritizes average precision for rare anomalies, then
uses F1, precision, and recall as supporting metrics where available. Row-based
models are preferred for the main selection when those results are identifiable.</p>
</div>

<h2>2. All-model comparison</h2>
{table_html}

<h2>3. Understanding the metrics</h2>
{metric_explanations}

<h2>4. Visualizations</h2>
{figures_html}

<h2>5. Conclusion</h2>
<p>Based on this run, <b>{winner_name}</b> is the provisional benchmark choice
under the ranking policy above. The strongest model depends on the project's
actual goal: catching as many anomalies as possible, reducing false alarms,
or balancing the two. Review its false-positive burden and missed anomalies
before calling it the best operational model.</p>

<h2>6. Limitations and next steps</h2>
<ul>
<li>Verify that the test set is independent of training and validation data.</li>
<li>Choose thresholds on validation data rather than tuning on final test labels.</li>
<li>Inspect false positives and missed anomalies by host, time, and scenario if
those fields are available in the original predictions.</li>
<li>Confirm whether sequence and row-based models score the same test population.</li>
<li>For supervised models, check whether training anomalies resemble real
deployment anomalies.</li>
<li>Save the chosen model, preprocessing configuration, feature list, and
threshold together for reproducibility.</li>
</ul>
<p>Generated by <code>scripts/generate_model_report.py</code>.</p>
</body>
</html>"""

    html_path = REPORT_DIR / "final_model_report.html"
    html_path.write_text(html, encoding="utf-8")

    summary = {
        "project": "SENTINEL 2.0",
        "source_csv": str(csv_path.relative_to(ROOT)),
        "selected_model": str(winner_name),
        "ranking_policy": "Average precision first, then F1, precision, recall.",
        "selected_metrics": {
            key: (
                None if pd.isna(winner.get(key))
                else float(winner[key])
            )
            for key in [
                "precision", "recall", "f1", "average_precision",
                "roc_auc", "false_positive_rate", "tp", "fp", "fn", "tn",
                "threshold",
            ]
        },
        "report": str(html_path.relative_to(ROOT)),
    }
    summary_path = REPORT_DIR / "model_selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSENTINEL 2.0 report generated successfully.")
    print(f"Input results: {csv_path}")
    print(f"HTML report:   {html_path}")
    print(f"JSON summary:  {summary_path}")
    print(f"Charts:        {FIG_DIR}")
    print(f"\nProvisional selected model: {winner_name}")
    print("Open final_model_report.html in your browser.")
    print("You can print the HTML report to PDF from the browser.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
