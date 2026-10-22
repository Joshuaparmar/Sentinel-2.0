"""Tests for SENTINEL synthetic telemetry generation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import generate_synthetic as gs


def make_mock_source(n_times=600, n_hosts=10):
    rng = np.random.default_rng(123)
    rows = []
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    for ti in range(n_times):
        for hi in range(n_hosts):
            rows.append({
                "timestamp": start + pd.Timedelta(seconds=10 * ti),
                "hostname": f"gpu-{hi:02d}",
                "node_state": "up",
                "p0_gpu0_power": float(max(20, rng.normal(120, 12))),
                "p0_gpu1_power": float(max(20, rng.normal(115, 11))),
                "gpu0_core_temp": float(rng.normal(35, 2)),
                "gpu1_core_temp": float(rng.normal(36, 2)),
                "p0_temp_min": float(30 + rng.random()),
                "p0_temp_mean": float(35 + rng.random()),
                "p0_temp_max": float(40 + rng.random()),
                "label": 0,
                "xid": -1,
                "allocation_id": 0,
                "extra_unmodeled": "preserve-me",
            })
    return pd.DataFrame(rows)


def test_validate_raw_columns_rejects_reserved_collision():
    df = make_mock_source().assign(source_row_id=0)
    with pytest.raises(ValueError, match="reserved generator"):
        gs.validate_raw_columns(df)


def test_feature_manifest_excludes_labels_and_metadata():
    df = make_mock_source()
    features = gs.identify_features(df, "timestamp", "hostname")
    assert "label" not in features["features"]
    assert "xid" not in features["features"]
    assert "hostname" not in features["features"]
    assert "p0_gpu0_power" in features["features"]
    assert "gpu0_core_temp" in features["features"]


def test_chronological_split_has_purge_gap():
    df = make_mock_source(n_times=100, n_hosts=3)
    train, val, test, purged, bounds = gs.split_chronologically(
        df, "timestamp", 0.6, 0.2, 5
    )
    assert train.timestamp.max() < val.timestamp.min()
    assert val.timestamp.max() < test.timestamp.min()
    assert purged > 0
    assert bounds["purge_steps"] == 5


def test_generator_end_to_end_preserves_source_and_train_normal(tmp_path):
    source = tmp_path / "train.parquet"
    out = tmp_path / "out"
    make_mock_source(n_times=600, n_hosts=10).to_parquet(source, index=False)
    before = gs.sha256_file(source)
    report = gs.generate(source, out, seed=42, window_steps=5, events_per_split=5)
    after = gs.sha256_file(source)
    assert before == after
    train = pd.read_parquet(out / "synthetic_train.parquet")
    val = pd.read_parquet(out / "synthetic_val.parquet")
    test = pd.read_parquet(out / "synthetic_test.parquet")
    assert (train["label"] == 0).all()
    assert (train["anomaly_type"] == "normal").all()
    assert (val["label"] == 1).any()
    assert (test["label"] == 1).any()
    assert (val.loc[val.label == 1, "event_id"] != "").all()
    assert (test.loc[test.label == 1, "event_id"] != "").all()
    manifest = __import__("json").loads((out / "feature_manifest.json").read_text())
    assert "label" not in manifest["features"]
    assert "xid" not in manifest["features"]
    assert report["source_unchanged"] is True
    assert (val.loc[val.label == 1, "anomaly_type"] != "normal").all()
    assert (test.loc[test.label == 1, "anomaly_type"] != "normal").all()

    # A second run must not overwrite the existing generated artifacts.
    with pytest.raises(FileExistsError, match="Refusing to overwrite existing output files"):
        gs.generate(source, out, seed=42, window_steps=5, events_per_split=5)


def test_split_validation_rejects_anomaly_without_metadata():
    df = make_mock_source(n_times=20, n_hosts=2)
    df["anomaly_type"] = "normal"
    df["event_id"] = ""
    df.loc[df.index[0], "label"] = 1
    features = gs.identify_features(df, "timestamp", "hostname")
    limits = gs._derive_limits(df, features, 0.1)
    with pytest.raises(ValueError, match="missing event metadata"):
        gs.validate_split(df, "validation", "timestamp", "hostname", features, limits)



def test_event_records_report_realized_changes(tmp_path):
    import json
    import numpy as np
    import pandas as pd
    from sentinel import generate_synthetic as gs

    source = make_mock_source().drop(columns=[gs.SOURCE_ROW_COL], errors="ignore")
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)

    output_dir = tmp_path / "output"
    gs.generate(
        source_path, output_dir, seed=42,
        window_steps=5, events_per_split=5,
    )

    events = pd.read_csv(output_dir / "synthetic_events.csv")
    assert len(events) == 10

    power_events = events[events["anomaly_type"].isin([
        "abrupt_power_drop", "transient_spike",
        "gradual_degradation", "multi_host_event",
    ])]
    temp_events = events[events["anomaly_type"].isin([
        "thermal_anomaly", "multi_host_event",
    ])]

    assert power_events["realized_median_power_change_pct"].notna().all()
    assert temp_events["realized_median_temp_change"].notna().all()

    assert events.loc[
        events["anomaly_type"] == "thermal_anomaly",
        "realized_median_power_change_pct",
    ].isna().all()

    assert events.loc[
        events["anomaly_type"] == "abrupt_power_drop",
        "realized_median_temp_change",
    ].isna().all()

    report = json.loads((output_dir / "synthetic_report.json").read_text())
    assert report

def test_temp_summary_ordering_helper_finds_group():
    groups = gs._temp_summary_groups(["p0_temp_min", "p0_temp_mean", "p0_temp_max"])
    assert len(groups) == 1
