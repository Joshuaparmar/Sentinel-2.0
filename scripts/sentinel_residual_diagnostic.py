from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sentinel_model_comparison as base

OUT = ROOT / "reports" / "feature_residual_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    seed = base.SEED
    rng = np.random.default_rng(seed)
    features = base.load_manifest()

    train = base.load_split("train", features)
    val = base.load_split("val", features)
    test = base.load_split("test", features)

    medians = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0.0)
    )

    X_train_raw, y_train = base.point_arrays(train, features, medians)
    X_val_raw, y_val = base.point_arrays(val, features, medians)
    X_test_raw, y_test = base.point_arrays(test, features, medians)

    if y_train.sum() != 0:
        raise ValueError("Training split contains labelled anomalies.")

    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Reproduce the original benchmark's RNG consumption:
    # Isolation Forest sample -> GMM sample -> Dense AE sample.
    _ = base.sample_indices(
        len(X_train), base.MAX_CLASSICAL_TRAIN, rng
    )
    _ = base.sample_indices(
        len(X_train), base.MAX_GMM_TRAIN, rng
    )
    dense_idx = base.sample_indices(
        len(X_train), base.MAX_DENSE_TRAIN, rng
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 16, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=3,
        random_state=seed,
        verbose=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train[dense_idx], X_train[dense_idx])

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # Squared reconstruction residual for each feature, not yet averaged.
    resid_val = np.square(X_val - pred_val)
    resid_test = np.square(X_test - pred_test)

    power_features = [f for f in features if "power" in f.lower()]
    temp_features = [
        f for f in features
        if "temp" in f.lower() or "temperature" in f.lower()
    ]

    if not power_features or not temp_features:
        raise ValueError(
            "Could not identify power and temperature features from manifest."
        )

    power_idx = [features.index(f) for f in power_features]
    temp_idx = [features.index(f) for f in temp_features]

    val_power = resid_val[:, power_idx].mean(axis=1)
    test_power = resid_test[:, power_idx].mean(axis=1)
    val_temp = resid_val[:, temp_idx].mean(axis=1)
    test_temp = resid_test[:, temp_idx].mean(axis=1)
    val_all = resid_val.mean(axis=1)
    test_all = resid_test.mean(axis=1)

    normal = y_val == 0
    if not normal.any():
        raise ValueError("No normal validation rows available.")

    def percentile_of_normal(score):
        # Empirical percentile relative to normal validation scores.
        reference = np.sort(score[normal])
        return 100.0 * np.searchsorted(
            reference, score, side="right"
        ) / len(reference)

    val_scores = pd.DataFrame({
        "row_index": np.arange(len(val)),
        "label": y_val,
        "score_all": val_all,
        "score_power": val_power,
        "score_temperature": val_temp,
    })
    test_scores = pd.DataFrame({
        "row_index": np.arange(len(test)),
        "label": y_test,
        "score_all": test_all,
        "score_power": test_power,
        "score_temperature": test_temp,
        "all_percentile_vs_normal_val": percentile_of_normal(val_all)
            if len(val) == len(test) else np.nan,
    })

    # Percentiles must use the matching normal-validation score channel.
    test_scores["all_percentile_vs_normal_val"] = percentile_of_normal(
        np.concatenate([val_all, test_all])[len(val_all):]
    )
    test_scores["power_percentile_vs_normal_val"] = percentile_of_normal(
        np.concatenate([val_power, test_power])[len(val_power):]
    )
    test_scores["temperature_percentile_vs_normal_val"] = percentile_of_normal(
        np.concatenate([val_temp, test_temp])[len(val_temp):]
    )

    # Attach host/time/event metadata where present.
    metadata_candidates = [
        "hostname", "timestamp", "anomaly_type", "event_type",
        "event_id", "label", "is_anomaly"
    ]
    for col in metadata_candidates:
        if col in val.columns:
            val_scores[col] = val[col].to_numpy()
        if col in test.columns:
            test_scores[col] = test[col].to_numpy()

    # Preserve the individual per-feature residuals for further investigation.
    for j, feature in enumerate(features):
        test_scores[f"resid_{feature}"] = resid_test[:, j]

    val_scores.to_csv(OUT / "validation_residual_scores.csv", index=False)
    test_scores.to_csv(OUT / "test_residual_scores.csv", index=False)

    # Compare each injected test row against normal validation distributions.
    rows = []
    for name, val_score, test_score in [
        ("all_features", val_all, test_all),
        ("power_only_residual_channel", val_power, test_power),
        ("temperature_only_residual_channel", val_temp, test_temp),
    ]:
        ref = np.sort(val_score[normal])
        anomalous = test_score[y_test == 1]
        rows.append({
            "channel": name,
            "normal_validation_rows": int(normal.sum()),
            "test_anomaly_rows": int(len(anomalous)),
            "anomaly_median_score": float(np.median(anomalous))
                if len(anomalous) else np.nan,
            "anomaly_p90_score": float(np.quantile(anomalous, 0.90))
                if len(anomalous) else np.nan,
            "anomaly_median_percentile": float(
                np.median(100.0 * np.searchsorted(
                    ref, anomalous, side="right"
                ) / len(ref))
            ) if len(anomalous) else np.nan,
            "normal_validation_p95": float(np.quantile(ref, 0.95)),
            "normal_validation_p99": float(np.quantile(ref, 0.99)),
        })

    pd.DataFrame(rows).to_csv(
        OUT / "channel_separation_summary.csv", index=False
    )

    metadata = {
        "seed": seed,
        "features": features,
        "training_sample_rows": int(len(dense_idx)),
        "training_sample_index_sha256": __import__("hashlib").sha256(
            dense_idx.astype(np.int64).tobytes()
        ).hexdigest(),
        "n_iter": int(model.n_iter_),
        "final_training_loss": float(model.loss_),
        "loss_curve": [float(x) for x in model.loss_curve_],
        "best_validation_score": getattr(
            model, "best_validation_score_", None
        ),
        "power_features": power_features,
        "temperature_features": temp_features,
        "warnings": [str(w.message) for w in caught],
        "note": (
            "Reproduces original RNG sampling order for the Dense AE. "
            "Exact scores can still vary with package versions or numerical "
            "environment."
        ),
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved diagnostics to: {OUT}")
    print(f"Training rows: {len(dense_idx):,}")
    print(f"Autoencoder iterations: {model.n_iter_}")
    print(f"Power features: {len(power_features)}")
    print(f"Temperature features: {len(temp_features)}")
    print("\nChannel separation summary:")
    print(pd.DataFrame(rows).to_string(index=False))
    if caught:
        print("\nCaptured warnings:")
        for w in caught:
            print(f"- {w.message}")

if __name__ == "__main__":
    main()

