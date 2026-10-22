#!/usr/bin/env python
"""SENTINEL 2.0: isolated V3 supervised XGBoost experiment."""

from pathlib import Path
import hashlib
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_v3"
OUT = ROOT / "experiments" / "v3_xgboost_run_001"
MODEL_DIR = OUT / "model"
REPORT_DIR = OUT / "reports"
PRED_DIR = OUT / "predictions"

TRAIN_FILE = DATA / "synthetic_supervised_train.parquet"
VAL_FILE = DATA / "synthetic_val.parquet"
TEST_FILE = DATA / "synthetic_test.parquet"
EVENT_FILE = DATA / "synthetic_events.csv"

SEED = 20261022
TARGET_FPR = 0.01

FEATURES = [
    "p0_gpu0_power", "p0_gpu1_power", "p0_gpu2_power",
    "p1_gpu0_power", "p1_gpu1_power", "p1_gpu2_power",
    "p0_power", "p1_power", "ps0_input_power", "ps1_input_power",
    "gpu0_core_temp", "gpu0_mem_temp",
    "gpu1_core_temp", "gpu1_mem_temp",
    "gpu2_core_temp", "gpu2_mem_temp",
    "gpu3_core_temp", "gpu3_mem_temp",
    "gpu4_core_temp", "gpu4_mem_temp",
    "gpu5_core_temp", "gpu5_mem_temp",
    "p0_temp_mean", "p0_temp_min", "p0_temp_max",
    "p1_temp_mean", "p1_temp_min", "p1_temp_max",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path, require_anomalies):
    df = pd.read_parquet(path)

    missing = [c for c in FEATURES + ["label", "event_id"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path}: missing required columns: {missing}")

    if df["label"].isna().any():
        raise RuntimeError(f"{path}: null labels found")

    labels = set(df["label"].unique().tolist())
    if not labels.issubset({0, 1}):
        raise RuntimeError(f"{path}: labels are not binary: {labels}")

    if require_anomalies and int(df["label"].sum()) == 0:
        raise RuntimeError(f"{path}: expected positive examples, found none")

    if df[FEATURES].isna().any().any():
        raise RuntimeError(f"{path}: missing telemetry feature values")

    X = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(X).all():
        raise RuntimeError(f"{path}: non-finite telemetry feature values")

    y = df["label"].to_numpy(dtype=np.int8, copy=True)
    return df, X, y


def choose_threshold_at_fpr(y, scores, target_fpr):
    """Choose a score threshold using validation negatives only."""
    negatives = np.sort(scores[y == 0])[::-1]
    if len(negatives) == 0:
        raise RuntimeError("Cannot choose threshold without validation negatives")

    max_fp = int(np.floor(target_fpr * len(negatives)))
    # A threshold just above this negative score permits at most max_fp
    # false positives, accounting conservatively for tied scores.
    index = min(max_fp, len(negatives) - 1)
    threshold = float(np.nextafter(negatives[index], np.inf))

    pred = (scores >= threshold).astype(np.int8)
    actual_fpr = float(pred[y == 0].mean())
    if actual_fpr > target_fpr + 1e-12:
        raise RuntimeError(
            f"Threshold FPR constraint failed: {actual_fpr:.6f} > {target_fpr}"
        )
    return threshold, actual_fpr, max_fp


def classification_metrics(y, scores, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "rows": int(len(y)),
        "actual_normal": int((y == 0).sum()),
        "actual_anomaly": int((y == 1).sum()),
        "predicted_normal": int((pred == 0).sum()),
        "predicted_anomaly": int((pred == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "average_precision_pr_auc": (
            float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None
        ),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None,
        "confusion_matrix": {"labels": ["normal", "anomaly"],
                             "rows_actual_cols_predicted": [[int(tn), int(fp)],
                                                            [int(fn), int(tp)]]},
    }
    return result


def main():
    start = time.time()

    # Refuse to overwrite model, report, or prediction artifacts from an earlier run.
    if OUT.exists():
        raise SystemExit(
            f"STOP: output directory already exists: {OUT}\n"
            "Choose a new run directory explicitly; existing results were not changed."
        )

    for p in [TRAIN_FILE, VAL_FILE, TEST_FILE, EVENT_FILE]:
        if not p.is_file():
            raise SystemExit(f"STOP: required input file is missing: {p}")

    print("SENTINEL V3 XGBOOST EXPERIMENT")
    print("=" * 64)
    print("XGBoost:", xgb.__version__, "| scikit-learn:", sklearn.__version__)
    print("Data:", DATA)
    print("Output:", OUT)
    print("Features:", len(FEATURES))

    train_df, X_train, y_train = load_split(TRAIN_FILE, True)
    val_df, X_val, y_val = load_split(VAL_FILE, True)
    test_df, X_test, y_test = load_split(TEST_FILE, True)

    if not np.array_equal(np.unique(y_train), np.array([0, 1])):
        raise RuntimeError("Training split must contain both classes")

    print("\nSplit sizes:")
    for name, df, y in [
        ("train", train_df, y_train),
        ("validation", val_df, y_val),
        ("test", test_df, y_test),
    ]:
        print(f"  {name:10s}: {len(df):,} rows; "
              f"{int((y == 0).sum()):,} normal; {int((y == 1).sum()):,} anomaly")

    # Class weighting is based on the training split only.
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        random_state=SEED,
        early_stopping_rounds=50,
    )

    print("\nTraining XGBoost (early stopping uses validation only)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("\nScoring validation and selecting threshold...")
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold, validation_fpr, allowed_fp = choose_threshold_at_fpr(
        y_val, val_scores, TARGET_FPR
    )
    val_pred = (val_scores >= threshold).astype(np.int8)
    val_metrics = classification_metrics(y_val, val_scores, val_pred)

    # Test is scored only after the model and threshold have been fixed.
    print("Scoring the held-out test split once...")
    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(np.int8)
    test_metrics = classification_metrics(y_test, test_scores, test_pred)

    # Join event manifest only for reporting, never for model inputs.
    events = pd.read_csv(EVENT_FILE)
    test_events = events.loc[events["split"].eq("test")].copy()
    if len(test_events) != 30:
        raise RuntimeError(f"Expected 30 test manifest events, got {len(test_events)}")

    test_report = test_df.copy()
    test_report["score"] = test_scores
    test_report["prediction"] = test_pred

    event_results = []
    for _, event in test_events.iterrows():
        event_id = str(event["event_id"])
        mask = test_report["event_id"].astype(str).eq(event_id)
        rows = test_report.loc[mask]
        if rows.empty:
            raise RuntimeError(f"Manifest event has no matching test rows: {event_id}")

        event_results.append({
            "event_id": event_id,
            "anomaly_type": str(event["anomaly_type"]),
            "severity_band": str(event["severity_band"]),
            "held_out": bool(event["held_out"]),
            "event_rows_in_manifest": int(event["n_rows"]),
            "matched_rows": int(len(rows)),
            "detected_any_row": bool(rows["prediction"].eq(1).any()),
            "detected_row_count": int(rows["prediction"].sum()),
            "anomalous_row_recall": float(rows["prediction"].mean()),
            "max_row_score": float(rows["score"].max()),
            "threshold": threshold,
        })

    event_df = pd.DataFrame(event_results)
    if not event_df["matched_rows"].eq(event_df["event_rows_in_manifest"]).all():
        raise RuntimeError("Manifest event row counts do not match test Parquet")

    event_metrics = {
        "test_events": int(len(event_df)),
        "events_detected": int(event_df["detected_any_row"].sum()),
        "event_detection_rate": float(event_df["detected_any_row"].mean()),
        "by_severity_band": {},
        "by_anomaly_type": {},
    }
    for band, group in event_df.groupby("severity_band"):
        event_metrics["by_severity_band"][str(band)] = {
            "events": int(len(group)),
            "detected": int(group["detected_any_row"].sum()),
            "detection_rate": float(group["detected_any_row"].mean()),
        }
    for kind, group in event_df.groupby("anomaly_type"):
        event_metrics["by_anomaly_type"][str(kind)] = {
            "events": int(len(group)),
            "detected": int(group["detected_any_row"].sum()),
            "detection_rate": float(group["detected_any_row"].mean()),
        }

    # Per-event row recall is useful, but event detection means at least one
    # row in that event crossed the threshold.
    anomaly_row_results = {}
    for band in ["baseline", "held_out"]:
        ids = set(test_events.loc[test_events["severity_band"].eq(band), "event_id"].astype(str))
        mask = test_report["event_id"].astype(str).isin(ids) & test_report["label"].eq(1)
        y = test_report.loc[mask, "label"].to_numpy(dtype=np.int8)
        pred = test_report.loc[mask, "prediction"].to_numpy(dtype=np.int8)
        anomaly_row_results[band] = {
            "anomalous_rows": int(mask.sum()),
            "detected_anomalous_rows": int(pred.sum()),
            "anomalous_row_recall": float(pred.mean()) if len(pred) else None,
        }

    OUT.mkdir(parents=True)
    MODEL_DIR.mkdir()
    REPORT_DIR.mkdir()
    PRED_DIR.mkdir()

    model_path = MODEL_DIR / "xgboost_v3.json"
    model.save_model(model_path)

    # Keep metadata with predictions, but not as model features.
    keep_cols = [
        c for c in [
            "timestamp", "hostname", "label", "anomaly_type",
            "event_id", "source_row_id", "source_label"
        ] if c in test_df.columns
    ]
    pred_df = test_df[keep_cols].copy()
    pred_df["anomaly_score"] = test_scores
    pred_df["prediction"] = test_pred
    pred_df.to_parquet(PRED_DIR / "test_predictions.parquet", index=False)

    event_df.to_csv(REPORT_DIR / "test_event_detection.csv", index=False)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    report = {
        "experiment": "SENTINEL V3 supervised XGBoost",
        "seed": SEED,
        "versions": {
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "inputs_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [TRAIN_FILE, VAL_FILE, TEST_FILE, EVENT_FILE]
        },
        "features": FEATURES,
        "excluded_metadata_columns": [
            "timestamp", "node_state", "hostname", "yymmdd",
            "job_begin_time", "job_end_time", "allocation_id",
            "label", "xid", "source_row_id", "source_label",
            "anomaly_type", "event_id",
        ],
        "training": {
            "rows": int(len(y_train)),
            "normal_rows": negatives,
            "anomaly_rows": positives,
            "scale_pos_weight": float(scale_pos_weight),
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
            "parameters": model.get_params(),
        },
        "threshold_selection": {
            "method": "validation negative-score order statistic",
            "target_false_positive_rate": TARGET_FPR,
            "validation_false_positive_rate": validation_fpr,
            "maximum_allowed_validation_false_positives": allowed_fp,
            "threshold": threshold,
            "validation_metrics_at_threshold": val_metrics,
        },
        "test": {
            "metrics_at_fixed_validation_threshold": test_metrics,
            "event_detection": event_metrics,
            "anomalous_row_recall_by_severity_band": anomaly_row_results,
            "warning": (
                "Test metrics were calculated once using the threshold selected "
                "on validation. Do not tune model or threshold from these results."
            ),
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }

    with (REPORT_DIR / "experiment_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print("Best iteration:", model.best_iteration)
    print(f"Threshold: {threshold:.10g}")
    print(f"Validation FPR: {validation_fpr:.4%} (target <= {TARGET_FPR:.2%})")
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nTest event detection:")
    print(json.dumps(event_metrics, indent=2))
    print("\nAnomalous-row recall by severity:")
    print(json.dumps(anomaly_row_results, indent=2))
    print("\nArtifacts saved under:", OUT)
    print("Report:", REPORT_DIR / "experiment_report.json")
    print("Elapsed seconds:", round(time.time() - start, 2))


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
