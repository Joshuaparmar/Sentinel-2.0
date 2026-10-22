#!/usr/bin/env python
"""Diagnose synthetic anomaly feature signals and Dense Autoencoder scores without retraining."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "synthetic_output_v2"
REPORT = ROOT / "reports" / "model_comparison"
MODEL = ROOT / "models" / "comparison"

def main():
    manifest = json.loads((DATA / "feature_manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    prep = json.loads((MODEL / "preprocessing.json").read_text(encoding="utf-8"))
    means = pd.Series(prep["scaler_mean"], index=features)
    scales = pd.Series(prep["scaler_scale"], index=features).replace(0, np.nan)

    test = pd.read_parquet(DATA / "synthetic_test.parquet")
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
    scores_path = REPORT / "row_scores_test.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}. Run sentinel_model_comparison.py first.")
    scores = pd.read_csv(scores_path)
    dense = scores[scores["model"] == "Dense Autoencoder"].copy()
    dense["timestamp"] = pd.to_datetime(dense["timestamp"], utc=True)

    # Merge on stable host/timestamp key, not row order.
    key = ["hostname", "timestamp"]
    use_cols = key + ["label", "anomaly_type"]
    dense = dense.drop(columns=[c for c in ["label", "anomaly_type"] if c in dense.columns])
    joined = test[use_cols + features].merge(
        dense[key + ["score", "threshold", "prediction"]],
        on=key, how="left", validate="one_to_one"
    )
    joined["label"] = joined["label"].fillna(0).astype(int)
    joined["prediction"] = joined["prediction"].fillna(0).astype(int)
    joined["score"] = pd.to_numeric(joined["score"], errors="coerce")
    joined["threshold"] = pd.to_numeric(joined["threshold"], errors="coerce")

    # Standardized feature deviation from training distribution.
    z = (joined[features] - means) / scales
    z = z.replace([np.inf, -np.inf], np.nan)

    print("\n=== DATA AND SCORE ALIGNMENT ===")
    print(f"Test rows: {len(test):,}")
    print(f"Rows with Dense AE scores: {joined['score'].notna().sum():,}")
    print(f"Test anomalous rows: {int(joined['label'].sum())}")
    print(f"Dense AE threshold: {joined['threshold'].dropna().iloc[0]:.6f}")

    print("\n=== DENSE AUTOENCODER: SCORE SEPARATION BY ANOMALY TYPE ===")
    rows = []
    for kind, g in joined[joined["label"] == 1].groupby("anomaly_type", dropna=False):
        s = g["score"].dropna()
        threshold = g["threshold"].dropna()
        rows.append({
            "anomaly_type": str(kind),
            "rows": len(g),
            "detected": int(g["prediction"].sum()),
            "recall": float(g["prediction"].mean()),
            "score_median": float(s.median()) if len(s) else np.nan,
            "score_p90": float(s.quantile(.90)) if len(s) else np.nan,
            "threshold": float(threshold.iloc[0]) if len(threshold) else np.nan,
        })
    score_table = pd.DataFrame(rows).sort_values("recall")
    print(score_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    score_table.to_csv(REPORT / "dense_ae_type_score_separation.csv", index=False)

    print("\n=== TOP FEATURE DEVIATIONS BY ANOMALY TYPE ===")
    feature_rows = []
    for kind, g in joined[joined["label"] == 1].groupby("anomaly_type", dropna=False):
        zg = z.loc[g.index, features]
        # Rank by average absolute z-score; retain signed mean to show direction.
        mean_abs = zg.abs().mean()
        mean_signed = zg.mean()
        for feat in mean_abs.sort_values(ascending=False).head(10).index:
            feature_rows.append({
                "anomaly_type": str(kind),
                "feature": feat,
                "mean_abs_train_z": float(mean_abs[feat]),
                "mean_signed_train_z": float(mean_signed[feat]),
                "anomaly_rows": len(g),
            })
        print(f"\n[{kind}] ({len(g)} rows)")
        top = pd.DataFrame({
            "feature": mean_abs.sort_values(ascending=False).head(8).index,
            "mean_abs_train_z": mean_abs.sort_values(ascending=False).head(8).values,
        })
        top["mean_signed_train_z"] = top["feature"].map(mean_signed)
        print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(REPORT / "feature_deviation_by_anomaly_type.csv", index=False)

    print("\n=== MISSED ABRUPT-POWER-DROP ROWS ===")
    abrupt = joined[(joined["label"] == 1) & (joined["anomaly_type"] == "abrupt_power_drop")]
    if abrupt.empty:
        print("No abrupt_power_drop rows found; check anomaly_type labels.")
    else:
        cols = ["hostname", "timestamp", "score", "threshold", "prediction"]
        print(abrupt[cols].sort_values(["hostname", "timestamp"]).to_string(index=False))
        abrupt[cols + ["anomaly_type"]].to_csv(REPORT / "missed_abrupt_power_drop_rows.csv", index=False)

    print("\n=== OUTPUTS ===")
    print(REPORT / "dense_ae_type_score_separation.csv")
    print(REPORT / "feature_deviation_by_anomaly_type.csv")
    print(REPORT / "missed_abrupt_power_drop_rows.csv")
    print("\nInterpretation note: feature z-scores are relative to global training means/scales. "
          "They are diagnostic signals, not proof of causal importance or a model explanation.")

if __name__ == "__main__":
    main()
