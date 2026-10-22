
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Project paths and reuse of the existing benchmark functions
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import sentinel_model_comparison as base


OUT_DIR = PROJECT_ROOT / "reports" / "feature_ablation"
SEED = base.SEED
THRESHOLD_QUANTILE = base.THRESHOLD_QUANTILE


# Exact power-feature names from synthetic_output_v2/feature_manifest.json
POWER_FEATURES = [
    "p0_gpu0_power",
    "p0_gpu1_power",
    "p0_gpu2_power",
    "p0_power",
    "p1_gpu0_power",
    "p1_gpu1_power",
    "p1_gpu2_power",
    "p1_power",
    "ps0_input_power",
    "ps1_input_power",
]


def calculate_metrics(y, scores, threshold):
    """Calculate row-level classification metrics at a fixed threshold."""
    valid = np.isfinite(scores)
    y = np.asarray(y)[valid].astype(int)
    scores = np.asarray(scores)[valid]

    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "n_scored": int(len(y)),
        "test_anomalies": int(y.sum()),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "pr_auc_average_precision": (
            float(average_precision_score(y, scores))
            if y.sum() > 0 else float("nan")
        ),
        "false_positive_rate": (
            float(fp / (fp + tn)) if fp + tn else float("nan")
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the feature manifest configured by the existing benchmark.
    all_features = base.load_manifest()

    missing = sorted(set(POWER_FEATURES) - set(all_features))
    if missing:
        raise SystemExit(
            f"STOP: power features missing from manifest: {missing}"
        )

    temperature_features = [
        feature for feature in all_features
        if feature not in POWER_FEATURES
    ]

    variants = [
        ("A_all_28", all_features),
        ("B_temperature_only_18", temperature_features),
        ("C_power_only_10", POWER_FEATURES),
    ]

    print("=" * 65)
    print("SENTINEL 2.0 - DENSE AUTOENCODER FEATURE ABLATION")
    print("=" * 65)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Data directory: {base.DATA_DIR}")
    print(f"Threshold quantile: {THRESHOLD_QUANTILE}")
    print(f"Features: all={len(all_features)}, "
          f"temperature={len(temperature_features)}, "
          f"power={len(POWER_FEATURES)}")

    print("\nLoading data splits...")
    train = base.load_split("train", all_features)
    val = base.load_split("val", all_features)
    test = base.load_split("test", all_features)

    # Choose one training-row sample and reuse it for every variant.
    # This avoids giving one feature set a different training sample.
    rng = np.random.default_rng(SEED)
    train_indices = base.sample_indices(
        len(train), base.MAX_DENSE_TRAIN, rng
    )

    summary_rows = []
    score_frames = []

    for variant_name, features in variants:
        print(f"\n--- {variant_name}: {len(features)} features ---")

        # Imputation statistics come from training data only.
        medians = (
            train[features]
            .replace([np.inf, -np.inf], np.nan)
            .median()
        )

        X_train_raw, _ = base.point_arrays(train, features, medians)
        X_val_raw, y_val = base.point_arrays(val, features, medians)
        X_test_raw, y_test = base.point_arrays(test, features, medians)

        # Fit scaler on the sampled training rows only.
        scaler = StandardScaler()
        scaler.fit(X_train_raw[train_indices])

        X_train = scaler.transform(
            X_train_raw[train_indices]
        ).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        # Reuse the exact Dense Autoencoder implementation.
        model, val_scores, test_scores = base.fit_dense_autoencoder(
            X_train, X_val, X_test
        )

        # Calibrate each variant on normal validation rows only.
        normal_val = val_scores[
            (y_val == 0) & np.isfinite(val_scores)
        ]
        if len(normal_val) == 0:
            raise SystemExit(
                f"STOP: no normal validation scores for {variant_name}"
            )

        threshold = float(
            np.quantile(normal_val, THRESHOLD_QUANTILE)
        )

        metrics = calculate_metrics(
            y_test, test_scores, threshold
        )

        metrics.update({
            "variant": variant_name,
            "feature_count": len(features),
            "threshold": threshold,
            "threshold_quantile": THRESHOLD_QUANTILE,
            "threshold_source": "normal_validation_only",
            "training_rows_used": len(train_indices),
            "training_iterations": int(
                getattr(model, "n_iter_", -1)
            ),
            "features": ";".join(features),
        })

        test_types = (
            test["anomaly_type"].fillna("").astype(str).to_numpy()
        )
        test_labels = test["label"].astype(int).to_numpy()
        predictions = (test_scores > threshold).astype(int)

        # Recall for each synthetic anomaly type.
        for anomaly_type in sorted(set(test_types) - {""}):
            mask = (
                (test_types == anomaly_type)
                & (test_labels == 1)
                & np.isfinite(test_scores)
            )
            if mask.any():
                metrics[f"recall__{anomaly_type}"] = float(
                    predictions[mask].mean()
                )
                metrics[f"rows__{anomaly_type}"] = int(mask.sum())

        summary_rows.append(metrics)

        # Store row-level scores for later inspection.
        score_frame = test[
            ["hostname", "timestamp", "label", "anomaly_type"]
        ].copy()

        score_frame.insert(0, "variant", variant_name)
        score_frame["score"] = test_scores
        score_frame["threshold"] = threshold
        score_frame["prediction"] = predictions
        score_frame["is_true_positive"] = (
            (test_labels == 1) & (predictions == 1)
        ).astype(int)
        score_frame["is_false_negative"] = (
            (test_labels == 1) & (predictions == 0)
        ).astype(int)
        score_frame["is_false_positive"] = (
            (test_labels == 0) & (predictions == 1)
        ).astype(int)

        score_frames.append(score_frame)

        print(f"Threshold: {threshold:.6f}")
        print(
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"PR-AUC={metrics['pr_auc_average_precision']:.4f}"
        )
        print(
            f"TP={metrics['true_positives']} | "
            f"FP={metrics['false_positives']} | "
            f"FN={metrics['false_negatives']} | "
            f"TN={metrics['true_negatives']}"
        )
        print(f"Training iterations: {metrics['training_iterations']}")

        for anomaly_type in sorted(set(test_types) - {""}):
            recall_key = f"recall__{anomaly_type}"
            if recall_key in metrics:
                print(
                    f"  {anomaly_type}: "
                    f"recall={metrics[recall_key]:.3f} "
                    f"({metrics[f'rows__{anomaly_type}']} rows)"
                )

    summary = pd.DataFrame(summary_rows)
    all_scores = pd.concat(score_frames, ignore_index=True)

    summary_path = OUT_DIR / "ablation_summary.csv"
    scores_path = OUT_DIR / "ablation_scores_test.csv"
    metadata_path = OUT_DIR / "ablation_metadata.json"

    summary.to_csv(summary_path, index=False)
    all_scores.to_csv(scores_path, index=False)

    metadata = {
        "project_root": str(PROJECT_ROOT),
        "data_directory": str(base.DATA_DIR),
        "variants": {
            name: features for name, features in variants
        },
        "power_features": POWER_FEATURES,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold_protocol": (
            "Each variant uses its own threshold, calibrated at the "
            "99th percentile of normal validation reconstruction scores."
        ),
        "training_rows_used": int(len(train_indices)),
        "random_seed": SEED,
        "note": (
            "Exploratory ablation. Do not retune the benchmark based "
            "on test results. Compare abrupt-event recall and false "
            "positives alongside overall metrics."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("ABLATION SUMMARY")
    print("=" * 65)

    columns = [
        "variant",
        "feature_count",
        "precision",
        "recall",
        "f1",
        "pr_auc_average_precision",
        "false_positive_rate",
        "true_positives",
        "false_positives",
        "false_negatives",
        "recall__abrupt_power_drop",
    ]
    columns = [c for c in columns if c in summary.columns]
    print(summary[columns].to_string(index=False))

    print("\nSaved files:")
    print(summary_path)
    print(scores_path)
    print(metadata_path)
    print("\nExisting benchmark reports were not overwritten.")


if __name__ == "__main__":
    main()
