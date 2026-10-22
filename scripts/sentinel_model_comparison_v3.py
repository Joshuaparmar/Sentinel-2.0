#!/usr/bin/env python
"""SENTINEL 2.0: V3 anomaly-detection comparison.

Compares eight newly fitted model approaches with the separately saved,
frozen V3 XGBoost baseline. Model inputs are restricted to the 28 features
declared by data/synthetic_v3/feature_manifest.json.

Thresholds for new models are selected from normal validation scores to target
at most 1% false positives. Test labels are used for evaluation only.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic_v3"
REPORT_DIR = ROOT / "reports" / "model_comparison_v3_run_001"
MODEL_DIR = ROOT / "models" / "comparison_v3_run_001"
BASELINE_DIR = ROOT / "experiments" / "v3_xgboost_run_001"

SEED = 42
TARGET_VALIDATION_FPR = 0.01
MAX_CLASSICAL_TRAIN = 100_000
MAX_DENSE_TRAIN = 100_000
MAX_GMM_TRAIN = 30_000
MAX_SEQUENCE_TRAIN = 25_000
SEQUENCE_LENGTH = 30
BATCH_SIZE = 256
EPOCHS = 8
PATIENCE = 2


def load_manifest() -> list[str]:
    path = DATA_DIR / "feature_manifest.json"
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    features = manifest.get("model_feature_columns")
    if not isinstance(features, list) or len(features) != 28:
        raise ValueError(
            "V3 feature_manifest.json must contain exactly 28 "
            "'model_feature_columns'."
        )
    forbidden = {"label", "event_id", "hostname", "timestamp", "anomaly_type",
                 "source_row_id", "source_label", "xid"}
    bad = forbidden.intersection(features)
    if bad:
        raise ValueError(f"Non-feature metadata found in model inputs: {sorted(bad)}")
    return features


def load_split(name: str, features: list[str]) -> pd.DataFrame:
    path = DATA_DIR / f"synthetic_{name}.parquet"
    df = pd.read_parquet(path)
    required = set(features) | {
        "hostname", "timestamp", "label", "anomaly_type", "event_id"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} split missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["hostname", "timestamp"], kind="stable").reset_index(drop=True)
    values = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = values.groupby(df["hostname"], sort=False).ffill()
    return df


def sample_indices(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=cap, replace=False))


def point_arrays(
    df: pd.DataFrame, features: list[str], medians: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    X = (
        df[features].replace([np.inf, -np.inf], np.nan)
        .fillna(medians).to_numpy(dtype=np.float32)
    )
    y = df["label"].astype(int).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature values remain after median imputation.")
    return X, y


def threshold_at_target_fpr(
    normal_scores: np.ndarray, target_fpr: float = TARGET_VALIDATION_FPR
) -> float:
    """Select an order-statistic threshold with no more than target FPR above it."""
    scores = np.asarray(normal_scores, dtype=float)
    scores = np.sort(scores[np.isfinite(scores)])
    if not len(scores):
        raise ValueError("No finite normal validation scores available.")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    max_fp = int(np.floor(target_fpr * len(scores)))
    index = max(0, len(scores) - max_fp - 1)
    return float(scores[index])


def score_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores):
        raise ValueError(f"Label/score length mismatch: {len(y)} vs {len(scores)}")
    valid = np.isfinite(scores)
    yv, sv = y[valid], scores[valid]
    pred = (sv > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(yv)) > 1
    return {
        "n_scored": int(len(yv)),
        "coverage_fraction": float(len(yv) / len(y)) if len(y) else float("nan"),
        "unscored_rows": int(len(y) - len(yv)),
        "test_anomalies": int(yv.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(yv, pred, zero_division=0)),
        "recall": float(recall_score(yv, pred, zero_division=0)),
        "f1": float(f1_score(yv, pred, zero_division=0)),
        "pr_auc_average_precision": (
            float(average_precision_score(yv, sv)) if has_both_classes else float("nan")
        ),
        "roc_auc": float(roc_auc_score(yv, sv)) if has_both_classes else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def event_metrics(
    test_df: pd.DataFrame, scored_rows: np.ndarray,
    scores: np.ndarray, threshold: float
) -> dict:
    """Count each injected event_id once, including multi-host events."""
    frame = test_df.reset_index(drop=True)
    score_by_row = dict(zip(
        np.asarray(scored_rows, dtype=int).tolist(),
        np.asarray(scores, dtype=float).tolist()
    ))
    event_rows = frame[
        (frame["label"].astype(int) == 1)
        & frame["event_id"].fillna("").astype(str).ne("")
    ]
    manifest_path = DATA_DIR / "synthetic_events.csv"
    event_manifest = pd.read_csv(manifest_path)
    event_manifest = event_manifest[event_manifest["split"].astype(str).eq("test")]
    manifest_by_id = event_manifest.drop_duplicates("event_id").set_index("event_id")

    records = []
    for event_id, group in event_rows.groupby("event_id", sort=False):
        rows = group.index.to_numpy(dtype=int)
        available = [score_by_row[r] for r in rows if r in score_by_row]
        finite = [v for v in available if np.isfinite(v)]
        meta = manifest_by_id.loc[event_id] if event_id in manifest_by_id.index else None
        records.append({
            "event_id": str(event_id),
            "anomaly_type": str(meta["anomaly_type"]) if meta is not None else str(group["anomaly_type"].iloc[0]),
            "severity_band": str(meta["severity_band"]) if meta is not None else "unknown",
            "detected": bool(any(v > threshold for v in finite)),
            "anomalous_rows": int(len(rows)),
            "scored_anomalous_rows": int(len(finite)),
            "n_hosts": int(group["hostname"].nunique()),
        })
    events = pd.DataFrame(records)
    if events.empty:
        return {
            "test_events": 0, "events_detected": 0,
            "event_detection_rate": float("nan"),
            "event_detection_by_type": {}, "event_detection_by_severity": {},
            "event_detection_note": "No labelled event IDs found."
        }

    def breakdown(column: str) -> dict:
        result = {}
        for value, group in events.groupby(column, dropna=False, sort=True):
            result[str(value)] = {
                "events": int(len(group)),
                "detected": int(group["detected"].sum()),
                "detection_rate": float(group["detected"].mean()),
            }
        return result

    return {
        "test_events": int(len(events)),
        "events_detected": int(events["detected"].sum()),
        "event_detection_rate": float(events["detected"].mean()),
        "event_detection_by_type": breakdown("anomaly_type"),
        "event_detection_by_severity": breakdown("severity_band"),
        "event_detection_note": "Counted by manifest event_id; multi-host events count once.",
    }


def make_sequences(
    df: pd.DataFrame, features: list[str], medians: pd.Series,
    scaler: StandardScaler, stride: int, max_windows: int | None = None,
    normal_only: bool = False, rng: np.random.Generator | None = None
):
    """Build host-local windows; returned row IDs point to endpoint rows in df."""
    seqs, labels_out, rows_out = [], [], []
    for _, group in df.groupby("hostname", sort=False):
        group = group.sort_values("timestamp", kind="stable")
        raw = group[features].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(np.float32)
        scaled = scaler.transform(raw).astype(np.float32)
        labels = group["label"].astype(int).to_numpy()
        rows = group.index.to_numpy(dtype=int)
        if len(group) < SEQUENCE_LENGTH:
            continue
        for end in range(SEQUENCE_LENGTH - 1, len(group), stride):
            if normal_only and labels[end] != 0:
                continue
            seqs.append(scaled[end - SEQUENCE_LENGTH + 1:end + 1])
            labels_out.append(labels[end])
            rows_out.append(rows[end])
    if not seqs:
        return (
            np.empty((0, SEQUENCE_LENGTH, len(features)), dtype=np.float32),
            np.empty(0, dtype=int), np.empty(0, dtype=int)
        )
    X = np.asarray(seqs, dtype=np.float32)
    y = np.asarray(labels_out, dtype=int)
    rows = np.asarray(rows_out, dtype=int)
    if max_windows is not None and len(X) > max_windows:
        if rng is None:
            rng = np.random.default_rng(SEED)
        chosen = np.sort(rng.choice(len(X), size=max_windows, replace=False))
        X, y, rows = X[chosen], y[chosen], rows[chosen]
    return X, y, rows


def fit_dense_autoencoder(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 16, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
        max_iter=25, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=3, random_state=SEED, verbose=False,
    )
    model.fit(X_train, X_train)

    def errors(X):
        return np.mean(np.square(X - model.predict(X)), axis=1)

    return model, errors(X_val), errors(X_test)


def get_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, nn, DataLoader, TensorDataset, device


def train_sequence_model(kind: str, train_seq: np.ndarray,
                         val_seq: np.ndarray, test_seq: np.ndarray):
    torch, nn, DataLoader, TensorDataset, device = get_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_features = train_seq.shape[-1]

    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, n_features, kernel_size=3, padding=1),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x.transpose(1, 2))).transpose(1, 2)

    class LSTMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(n_features, 48, batch_first=True)
            self.to_latent = nn.Linear(48, 24)
            self.from_latent = nn.Linear(24, 48)
            self.decoder = nn.LSTM(48, 48, batch_first=True)
            self.output = nn.Linear(48, n_features)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            z = torch.relu(self.to_latent(h[-1]))
            repeated = self.from_latent(z).unsqueeze(1).repeat(1, x.shape[1], 1)
            decoded, _ = self.decoder(repeated)
            return self.output(decoded)

    model = ConvAE() if kind == "1D-CNN Autoencoder" else LSTMAE()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_seq)), batch_size=BATCH_SIZE,
        shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_seq)), batch_size=BATCH_SIZE, shuffle=False
    )
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                losses.append(loss_fn(model(batch), batch).item())
        val_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"    {kind}: epoch {epoch + 1}/{EPOCHS}, val reconstruction loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def errors(X):
        output = []
        loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                            batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                err = torch.mean(
                    (reconstructed[:, -1, :] - batch[:, -1, :]) ** 2, dim=1
                )
                output.append(err.cpu().numpy())
        return np.concatenate(output) if output else np.empty(0)

    return model, errors(val_seq), errors(test_seq), str(device)


def load_frozen_baseline(test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Load and strictly align the pre-existing baseline; never fit or overwrite it."""
    pred_path = BASELINE_DIR / "predictions" / "test_predictions.parquet"
    report_path = BASELINE_DIR / "reports" / "experiment_report.json"
    if not pred_path.exists() or not report_path.exists():
        raise FileNotFoundError("Frozen XGBoost baseline predictions/report are missing.")
    pred = pd.read_parquet(pred_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keys = ["timestamp", "hostname"]
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    joined = test[keys + ["label", "event_id", "anomaly_type"]].merge(
        pred[keys + ["label", "event_id", "anomaly_type", "anomaly_score", "prediction"]],
        on=keys, how="outer", suffixes=("_test", "_baseline"),
        indicator=True, validate="one_to_one"
    )
    if len(joined) != len(test) or not joined["_merge"].eq("both").all():
        raise ValueError("Frozen baseline predictions do not align one-to-one with V3 test rows.")
    if not (joined["label_test"] == joined["label_baseline"]).all():
        raise ValueError("Frozen baseline labels disagree with V3 test labels.")
    if not (joined["event_id_test"].fillna("") == joined["event_id_baseline"].fillna("")).all():
        raise ValueError("Frozen baseline event IDs disagree with V3 test event IDs.")
    joined = joined.sort_values(keys, kind="stable").reset_index(drop=True)
    # Rebuild the aligned frame in test row order to preserve row-index mapping.
    baseline = test[keys + ["label", "event_id", "anomaly_type"]].merge(
        pred[keys + ["anomaly_score", "prediction"]],
        on=keys, how="left", validate="one_to_one"
    )
    baseline["timestamp"] = pd.to_datetime(baseline["timestamp"], utc=True)
    if baseline["anomaly_score"].isna().any():
        raise ValueError("Frozen baseline has missing scores after alignment.")
    return report, baseline


def main():
    started = time.time()
    for directory in (REPORT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    features = load_manifest()
    print(f"Project: {ROOT}")
    print(f"Data: {DATA_DIR}")
    print(f"Features: {len(features)} | Sequence length: {SEQUENCE_LENGTH}")
    train = load_split("train", features)
    val = load_split("val", features)
    test = load_split("test", features)
    supervised_train = load_split("supervised_train", features)

    medians = train[features].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    X_train_raw, y_train = point_arrays(train, features, medians)
    X_val_raw, y_val = point_arrays(val, features, medians)
    X_test_raw, y_test = point_arrays(test, features, medians)
    X_supervised_raw, y_supervised = point_arrays(supervised_train, features, medians)
    if y_train.sum() != 0:
        raise ValueError("Normal-only training split unexpectedly contains anomalies.")
    if y_val.sum() == 0 or y_test.sum() == 0:
        raise ValueError("Validation and test splits must contain anomalies.")
    if y_supervised.sum() == 0 or y_supervised.sum() == len(y_supervised):
        raise ValueError("Supervised training split must contain both classes.")
    if not np.isfinite(X_supervised_raw).all():
        raise ValueError("Supervised training features contain non-finite values.")

    scale_pos_weight = float((y_supervised == 0).sum() / y_supervised.sum())
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    with (MODEL_DIR / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "features": features,
            "target_validation_fpr": TARGET_VALIDATION_FPR,
            "sequence_length": SEQUENCE_LENGTH,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "train_medians": medians.to_dict(),
        }, f, indent=2, default=str)

    print(f"Rows: train={len(train):,} (normal-only), supervised_train={len(supervised_train):,}, "
          f"validation={len(val):,}, test={len(test):,}")
    print(f"Supervised anomalies={int(y_supervised.sum()):,}; scale_pos_weight={scale_pos_weight:.3f}")

    results = []
    # name -> (validation labels/scores, test labels/scores, validation row IDs, test row IDs)
    raw_scores: dict[str, tuple] = {}
    val_rows_all = np.arange(len(val), dtype=int)
    test_rows_all = np.arange(len(test), dtype=int)

    train_idx = sample_indices(len(X_train), MAX_CLASSICAL_TRAIN, rng)
    print("\n[1/8] Isolation Forest")
    isolation = IsolationForest(
        n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1
    ).fit(X_train[train_idx])
    raw_scores["Isolation Forest"] = (
        y_val, -isolation.score_samples(X_val), y_test, -isolation.score_samples(X_test),
        val_rows_all, test_rows_all
    )
    joblib.dump(isolation, MODEL_DIR / "isolation_forest.joblib")

    print("[2/8] PCA reconstruction")
    pca_n = min(20, len(features), max(2, len(train_idx) - 1))
    pca = PCA(n_components=pca_n, svd_solver="randomized", random_state=SEED).fit(X_train[train_idx])
    raw_scores["PCA Reconstruction"] = (
        y_val,
        np.mean((X_val - pca.inverse_transform(pca.transform(X_val))) ** 2, axis=1),
        y_test,
        np.mean((X_test - pca.inverse_transform(pca.transform(X_test))) ** 2, axis=1),
        val_rows_all, test_rows_all
    )
    joblib.dump(pca, MODEL_DIR / "pca_reconstruction.joblib")

    print("[3/8] Gaussian Mixture Model")
    gmm_idx = sample_indices(len(X_train), MAX_GMM_TRAIN, rng)
    gmm = GaussianMixture(
        n_components=8, covariance_type="diag", reg_covar=1e-4,
        max_iter=100, random_state=SEED
    ).fit(X_train[gmm_idx])
    raw_scores["Gaussian Mixture Model"] = (
        y_val, -gmm.score_samples(X_val), y_test, -gmm.score_samples(X_test),
        val_rows_all, test_rows_all
    )
    joblib.dump(gmm, MODEL_DIR / "gaussian_mixture.joblib")

    print("[4/8] LightGBM")
    lgbm = LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.05,
        num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=SEED,
        n_jobs=-1, verbosity=-1,
    ).fit(X_supervised_raw, y_supervised)
    raw_scores["LightGBM"] = (
        y_val, lgbm.predict_proba(X_val_raw)[:, 1],
        y_test, lgbm.predict_proba(X_test_raw)[:, 1],
        val_rows_all, test_rows_all
    )
    joblib.dump(lgbm, MODEL_DIR / "lightgbm_supervised.joblib")

    print("[5/8] Comparison-run XGBoost")
    xgb = XGBClassifier(
        objective="binary:logistic", n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1,
    ).fit(X_supervised_raw, y_supervised)
    raw_scores["XGBoost Comparison Run"] = (
        y_val, xgb.predict_proba(X_val_raw)[:, 1],
        y_test, xgb.predict_proba(X_test_raw)[:, 1],
        val_rows_all, test_rows_all
    )
    joblib.dump(xgb, MODEL_DIR / "xgboost_comparison_run.joblib")

    print("[6/8] Dense Autoencoder")
    dense_idx = sample_indices(len(X_train), MAX_DENSE_TRAIN, rng)
    dense, dense_val, dense_test = fit_dense_autoencoder(X_train[dense_idx], X_val, X_test)
    raw_scores["Dense Autoencoder"] = (
        y_val, dense_val, y_test, dense_test, val_rows_all, test_rows_all
    )
    joblib.dump(dense, MODEL_DIR / "dense_autoencoder.joblib")

    print("\nBuilding host-aware temporal windows...")
    train_seq, _, _ = make_sequences(
        train, features, medians, scaler, stride=10, max_windows=MAX_SEQUENCE_TRAIN,
        normal_only=True, rng=rng
    )
    val_seq, val_seq_y, val_seq_rows = make_sequences(
        val, features, medians, scaler, stride=1
    )
    test_seq, test_seq_y, test_seq_rows = make_sequences(
        test, features, medians, scaler, stride=1
    )
    if min(len(train_seq), len(val_seq), len(test_seq)) == 0:
        raise ValueError("Temporal windows are empty; cannot fit sequence models.")
    print(f"Windows: train={len(train_seq):,}, val={len(val_seq):,}, test={len(test_seq):,}")

    print("[7/8] 1D-CNN Autoencoder")
    cnn, cnn_val, cnn_test, cnn_device = train_sequence_model(
        "1D-CNN Autoencoder", train_seq, val_seq, test_seq
    )
    raw_scores["1D-CNN Autoencoder"] = (
        val_seq_y, cnn_val, test_seq_y, cnn_test, val_seq_rows, test_seq_rows
    )
    import torch
    torch.save(cnn.state_dict(), MODEL_DIR / "cnn_autoencoder_state_dict.pt")
    print("    device:", cnn_device)

    print("[8/8] LSTM Autoencoder")
    lstm, lstm_val, lstm_test, lstm_device = train_sequence_model(
        "LSTM Autoencoder", train_seq, val_seq, test_seq
    )
    raw_scores["LSTM Autoencoder"] = (
        val_seq_y, lstm_val, test_seq_y, lstm_test, val_seq_rows, test_seq_rows
    )
    torch.save(lstm.state_dict(), MODEL_DIR / "lstm_autoencoder_state_dict.pt")
    print("    device:", lstm_device)

    # Select each newly fitted model's threshold from normal validation scores only.
    score_frames = {"validation": [], "test": []}
    type_rows, alert_rows = [], []
    for name, (yv, sv, yt, st, val_row_ids, test_row_ids) in raw_scores.items():
        normal_scores = sv[(yv == 0) & np.isfinite(sv)]
        threshold = threshold_at_target_fpr(normal_scores)
        metrics = score_metrics(yt, st, threshold)
        metrics.update({
            "model": name,
            "threshold_source": "normal_validation_order_statistic_target_1pct_fpr",
            "target_validation_fpr": TARGET_VALIDATION_FPR,
            "threshold_quantile": None,
            "score_unit": "sequence endpoint rows" if name in (
                "1D-CNN Autoencoder", "LSTM Autoencoder"
            ) else "all test rows",
        })
        metrics.update(event_metrics(test, test_row_ids, st, threshold))
        results.append(metrics)
        print(f"{name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc_average_precision']:.4f}, "
              f"ROC-AUC={metrics['roc_auc']:.4f}, FPR={metrics['false_positive_rate']:.4%}, "
              f"events={metrics['events_detected']}/{metrics['test_events']}")

        for split_name, frame, labels, scores, row_ids in (
            ("validation", val, yv, sv, val_row_ids),
            ("test", test, yt, st, test_row_ids),
        ):
            valid = np.isfinite(scores)
            aligned_rows = np.asarray(row_ids)[valid].astype(int)
            aligned_scores = np.asarray(scores)[valid]
            aligned_labels = np.asarray(labels)[valid].astype(int)
            scored_frame = frame.iloc[aligned_rows][
                ["hostname", "timestamp", "label", "anomaly_type", "event_id"]
            ].copy()
            scored_frame.insert(0, "row_index", aligned_rows)
            scored_frame.insert(0, "split", split_name)
            scored_frame.insert(0, "model", name)
            scored_frame["score"] = aligned_scores
            scored_frame["threshold"] = threshold
            scored_frame["prediction"] = (aligned_scores > threshold).astype(int)
            score_frames[split_name].append(scored_frame)

            if split_name == "test":
                scored_frame["is_true_positive"] = (
                    (aligned_labels == 1) & (aligned_scores > threshold)
                ).astype(int)
                scored_frame["is_false_negative"] = (
                    (aligned_labels == 1) & (aligned_scores <= threshold)
                ).astype(int)
                scored_frame["is_false_positive"] = (
                    (aligned_labels == 0) & (aligned_scores > threshold)
                ).astype(int)
                anomalous = scored_frame[scored_frame["label"].astype(int) == 1]
                for anomaly_type, group in anomalous.groupby("anomaly_type", dropna=False):
                    type_rows.append({
                        "model": name, "anomaly_type": str(anomaly_type),
                        "anomaly_rows": int(len(group)),
                        "detected_rows": int(group["prediction"].sum()),
                        "row_recall": float(group["prediction"].mean()),
                        "mean_score": float(group["score"].mean()),
                        "max_score": float(group["score"].max()),
                        "threshold": threshold,
                    })
                normal_rows = int((scored_frame["label"].astype(int) == 0).sum())
                fp_rows = int(scored_frame["is_false_positive"].sum())
                alert_rows.append({
                    "model": name, "scored_rows": int(len(scored_frame)),
                    "scored_normal_rows": normal_rows, "false_positive_rows": fp_rows,
                    "false_positives_per_100_host_hours": (
                        float(fp_rows / (normal_rows / 360.0) * 100.0) if normal_rows else float("nan")
                    ),
                    "note": "Approximate false-positive rows per 100 normal host-hours, not deduplicated alert incidents.",
                })

    # Add the frozen baseline as a separate entry; do not refit or save over it.
    print("\nLoading frozen XGBoost V3 baseline...")
    baseline_report, baseline = load_frozen_baseline(test)
    baseline_threshold = float(baseline_report["threshold_selection"]["threshold"])
    baseline_scores = baseline["anomaly_score"].to_numpy(dtype=float)
    baseline_labels = baseline["label"].astype(int).to_numpy()
    baseline_rows = np.arange(len(test), dtype=int)
    baseline_metrics = score_metrics(baseline_labels, baseline_scores, baseline_threshold)
    baseline_metrics.update({
        "model": "XGBoost V3 Frozen Baseline",
        "threshold_source": "saved_original_validation_selected_threshold",
        "target_validation_fpr": float(
            baseline_report["threshold_selection"]["target_false_positive_rate"]
        ),
        "threshold_quantile": None,
        "score_unit": "all test rows",
        "baseline_best_iteration": baseline_report["training"].get("best_iteration"),
        "baseline_note": "Separate pre-existing experiment; saved threshold and test scores reused unchanged.",
    })
    baseline_metrics.update(event_metrics(test, baseline_rows, baseline_scores, baseline_threshold))
    results.append(baseline_metrics)

    baseline_export = baseline[
        ["hostname", "timestamp", "label", "anomaly_type", "event_id"]
    ].copy()
    baseline_export.insert(0, "row_index", baseline_rows)
    baseline_export.insert(0, "split", "test")
    baseline_export.insert(0, "model", "XGBoost V3 Frozen Baseline")
    baseline_export["score"] = baseline_scores
    baseline_export["threshold"] = baseline_threshold
    baseline_export["prediction"] = (baseline_scores > baseline_threshold).astype(int)
    baseline_export["is_true_positive"] = (
        (baseline_export["label"].astype(int) == 1) & (baseline_export["prediction"] == 1)
    ).astype(int)
    baseline_export["is_false_negative"] = (
        (baseline_export["label"].astype(int) == 1) & (baseline_export["prediction"] == 0)
    ).astype(int)
    baseline_export["is_false_positive"] = (
        (baseline_export["label"].astype(int) == 0) & (baseline_export["prediction"] == 1)
    ).astype(int)
    score_frames["test"].append(baseline_export)
    for anomaly_type, group in baseline_export[
        baseline_export["label"].astype(int) == 1
    ].groupby("anomaly_type", dropna=False):
        type_rows.append({
            "model": "XGBoost V3 Frozen Baseline", "anomaly_type": str(anomaly_type),
            "anomaly_rows": int(len(group)), "detected_rows": int(group["prediction"].sum()),
            "row_recall": float(group["prediction"].mean()),
            "mean_score": float(group["score"].mean()), "max_score": float(group["score"].max()),
            "threshold": baseline_threshold,
        })
    baseline_normal_rows = int((baseline_export["label"].astype(int) == 0).sum())
    baseline_fp_rows = int(baseline_export["is_false_positive"].sum())
    alert_rows.append({
        "model": "XGBoost V3 Frozen Baseline",
        "scored_rows": int(len(baseline_export)),
        "scored_normal_rows": baseline_normal_rows,
        "false_positive_rows": baseline_fp_rows,
        "false_positives_per_100_host_hours": (
            float(baseline_fp_rows / (baseline_normal_rows / 360.0) * 100.0)
            if baseline_normal_rows else float("nan")
        ),
        "note": "Frozen baseline; approximate false-positive rows per 100 normal host-hours.",
    })
    print(f"XGBoost V3 Frozen Baseline: precision={baseline_metrics['precision']:.4f}, "
          f"recall={baseline_metrics['recall']:.4f}, F1={baseline_metrics['f1']:.4f}, "
          f"PR-AUC={baseline_metrics['pr_auc_average_precision']:.4f}, "
          f"ROC-AUC={baseline_metrics['roc_auc']:.4f}, "
          f"FPR={baseline_metrics['false_positive_rate']:.4%}, "
          f"events={baseline_metrics['events_detected']}/{baseline_metrics['test_events']}")

    # Persist outputs only under the isolated V3 run directories.
    for split_name, frames in score_frames.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                REPORT_DIR / f"row_scores_{split_name}.csv", index=False
            )
    pd.DataFrame(type_rows).to_csv(REPORT_DIR / "anomaly_type_diagnostics.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(REPORT_DIR / "false_positive_burden.csv", index=False)

    result_df = pd.DataFrame(results)
    columns = [
        "model", "score_unit", "threshold", "threshold_source", "target_validation_fpr",
        "threshold_quantile", "n_scored", "coverage_fraction", "unscored_rows",
        "test_anomalies", "precision", "recall", "f1", "pr_auc_average_precision",
        "roc_auc", "false_positive_rate", "true_positives", "false_positives",
        "false_negatives", "true_negatives", "test_events", "events_detected",
        "event_detection_rate", "event_detection_by_type", "event_detection_by_severity",
        "event_detection_note", "baseline_best_iteration", "baseline_note",
    ]
    result_df = result_df.reindex(columns=[c for c in columns if c in result_df.columns])
    csv_path = REPORT_DIR / "model_comparison.csv"
    json_path = REPORT_DIR / "model_comparison.json"
    result_df.to_csv(csv_path, index=False)
    result_df.to_json(json_path, orient="records", indent=2)

    summary = {
        "experiment": "SENTINEL V3 model comparison",
        "seed": SEED,
        "features": features,
        "sequence_length": SEQUENCE_LENGTH,
        "threshold_protocol": (
            "New models: normal-validation order statistic targeting at most 1% FPR. "
            "Frozen XGBoost baseline reuses its original validation-selected threshold."
        ),
        "target_validation_fpr": TARGET_VALIDATION_FPR,
        "train_rows": int(len(train)),
        "supervised_train_rows": int(len(supervised_train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_anomalies": int(y_train.sum()),
        "supervised_train_anomalies": int(y_supervised.sum()),
        "validation_anomalies": int(y_val.sum()),
        "test_anomalies": int(y_test.sum()),
        "test_event_count": int(test.loc[test["label"].astype(int) == 1, "event_id"].nunique()),
        "models": result_df.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": [
            "The benchmark evaluates generated synthetic scenarios, not real-world incident performance.",
            "LightGBM and the comparison-run XGBoost train on injected synthetic anomalies; real-world generalization is unverified.",
            "The frozen XGBoost baseline is a separately trained experiment with its own saved validation-selected threshold.",
            "Temporal models score sequence endpoints only; early rows per host are not scored.",
            "False-positive burden is an approximate row-based rate, not a deduplicated alert-incident count.",
        ],
    }
    with (REPORT_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n========== COMPARISON COMPLETE ==========")
    display_cols = [
        "model", "n_scored", "precision", "recall", "f1",
        "pr_auc_average_precision", "roc_auc", "false_positive_rate",
        "events_detected", "test_events", "event_detection_rate",
    ]
    print(result_df[[c for c in display_cols if c in result_df.columns]].to_string(index=False))
    print(f"\nReports: {REPORT_DIR}")
    print(f"Models:  {MODEL_DIR}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()
