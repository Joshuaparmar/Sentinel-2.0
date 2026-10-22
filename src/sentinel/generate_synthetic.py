
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SENTINEL 2.0 - SYNTHETIC TELEMETRY GENERATOR
# ============================================================

SEED = 42
SAMPLE_SECONDS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_ROOT
    / ".."
    / "Sentinel"
    / "dataset"
    / "ClusterWise_mini"
    / "olcf_task"
    / "train.parquet"
).resolve()

OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

LABEL_COL = "label"
XID_COL = "xid"
HOST_COL = "hostname"
TIME_COL = "timestamp"

POWER_COLS = [
    "p0_gpu0_power",
    "p0_gpu1_power",
    "p0_gpu2_power",
    "p1_gpu0_power",
    "p1_gpu1_power",
    "p1_gpu2_power",
]

SYSTEM_POWER_COLS = ["p0_power", "p1_power"]

TEMP_COLS = [
    f"gpu{i}_{kind}_temp"
    for i in range(6)
    for kind in ("core", "mem")
]

PSU_POWER_COLS = ["ps0_input_power", "ps1_input_power"]

SUMMARY_TEMP_COLS = [
    "p0_temp_mean",
    "p0_temp_min",
    "p0_temp_max",
    "p1_temp_mean",
    "p1_temp_min",
    "p1_temp_max",
]

TELEMETRY_COLS = (
    POWER_COLS
    + SYSTEM_POWER_COLS
    + TEMP_COLS
    + PSU_POWER_COLS
    + SUMMARY_TEMP_COLS
)

ANOMALY_TYPES = [
    "normal",
    "abrupt_power_drop",
    "thermal_anomaly",
    "transient_spike",
    "gradual_degradation",
    "multi_host_event",
]


def load_source():
    """Load the original normal-only training telemetry."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    df = pd.read_parquet(SOURCE_FILE)

    required = (
        [TIME_COL, HOST_COL, LABEL_COL]
        + TELEMETRY_COLS
    )
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required source columns: {missing}"
        )

    if not (df[LABEL_COL] == 0).all():
        raise ValueError(
            "Expected train.parquet to contain only label 0."
        )

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values([TIME_COL, HOST_COL]).reset_index(
        drop=True
    )

    return df


def split_chronologically(df):
    """Split by timestamp, never by individual rows."""
    timestamps = np.sort(df[TIME_COL].unique())

    train_end = int(len(timestamps) * 0.60)
    val_end = int(len(timestamps) * 0.80)

    train_times = timestamps[:train_end]
    val_times = timestamps[train_end:val_end]
    test_times = timestamps[val_end:]

    train = df[df[TIME_COL].isin(train_times)].copy()
    val = df[df[TIME_COL].isin(val_times)].copy()
    test = df[df[TIME_COL].isin(test_times)].copy()

    return train, val, test


def initialize_output(df):
    """Add separate synthetic ground-truth metadata."""
    out = df.copy()

    # Preserve the source label for auditing.
    out["source_label"] = out[LABEL_COL].astype("int8")

    # Replace label with synthetic ground truth in this copy.
    out[LABEL_COL] = np.zeros(len(out), dtype=np.int8)
    out["anomaly_type"] = "normal"
    out["event_id"] = ""

    # Original XID is not used to create synthetic labels.
    # Exclude it from model input features later.
    return out


def get_event_rows(df, start_time, end_time, hosts):
    """Find rows for selected hosts and an inclusive time range."""
    return (
        df[HOST_COL].isin(hosts)
        & (df[TIME_COL] >= start_time)
        & (df[TIME_COL] <= end_time)
    )


def mark_event(df, mask, anomaly_type, event_id):
    """Assign ground-truth labels to injected rows."""
    if mask.sum() == 0:
        raise ValueError(
            f"No rows found for event {event_id}."
        )

    df.loc[mask, LABEL_COL] = 1
    df.loc[mask, "anomaly_type"] = anomaly_type
    df.loc[mask, "event_id"] = event_id


def inject_power_drop(df, rng, event_id):
    """Abrupt, sustained drop in power on selected hosts."""
    hosts = rng.choice(
        df[HOST_COL].unique(),
        size=min(8, df[HOST_COL].nunique()),
        replace=False,
    )

    start_idx = int(len(df[TIME_COL].unique()) * 0.25)
    times = np.sort(df[TIME_COL].unique())
    start = pd.Timestamp(times[start_idx])
    end = start + pd.Timedelta(minutes=3)

    mask = get_event_rows(df, start, end, hosts)

    for col in POWER_COLS + SYSTEM_POWER_COLS:
        df.loc[mask, col] *= 0.45

    for col in PSU_POWER_COLS:
        df.loc[mask, col] *= 0.70

    mark_event(df, mask, "abrupt_power_drop", event_id)


def inject_thermal_anomaly(df, rng, event_id):
    """Sustained temperature rise on selected hosts."""
    hosts = rng.choice(
        df[HOST_COL].unique(),
        size=min(6, df[HOST_COL].nunique()),
        replace=False,
    )

    times = np.sort(df[TIME_COL].unique())
    start_idx = int(len(times) * 0.45)
    start = pd.Timestamp(times[start_idx])
    end = start + pd.Timedelta(minutes=4)

    mask = get_event_rows(df, start, end, hosts)

    for col in TEMP_COLS:
        df.loc[mask, col] += 8.0

    for col in SUMMARY_TEMP_COLS:
        df.loc[mask, col] += 8.0

    mark_event(df, mask, "thermal_anomaly", event_id)


def inject_transient_spike(df, rng, event_id):
    """Short-lived telemetry spike."""
    hosts = rng.choice(
        df[HOST_COL].unique(),
        size=min(5, df[HOST_COL].nunique()),
        replace=False,
    )

    times = np.sort(df[TIME_COL].unique())
    start_idx = int(len(times) * 0.60)
    start = pd.Timestamp(times[start_idx])
    end = start + pd.Timedelta(seconds=30)

    mask = get_event_rows(df, start, end, hosts)

    for col in POWER_COLS + SYSTEM_POWER_COLS:
        df.loc[mask, col] *= 1.8

    mark_event(df, mask, "transient_spike", event_id)


def inject_gradual_degradation(df, rng, event_id):
    """Progressive power reduction over a longer interval."""
    hosts = rng.choice(
        df[HOST_COL].unique(),
        size=min(6, df[HOST_COL].nunique()),
        replace=False,
    )

    times = np.sort(df[TIME_COL].unique())
    start_idx = int(len(times) * 0.70)
    start = pd.Timestamp(times[start_idx])
    end = start + pd.Timedelta(minutes=5)

    mask = get_event_rows(df, start, end, hosts)

    if mask.sum() == 0:
        raise ValueError("No rows found for gradual event.")

    elapsed = (
        df.loc[mask, TIME_COL] - start
    ).dt.total_seconds().to_numpy()

    progress = np.clip(elapsed / 300.0, 0.0, 1.0)
    factor = 1.0 - 0.55 * progress

    
    for col in POWER_COLS + SYSTEM_POWER_COLS:
        original_values = df.loc[mask, col].to_numpy(
            dtype=np.float64
        )
    adjusted_values = original_values * factor

    df.loc[mask, col] = adjusted_values.astype(
        df[col].dtype
    )


    mark_event(
        df, mask, "gradual_degradation", event_id
    )


def inject_multi_host_event(df, rng, event_id):
    """Correlated power event affecting a larger host group."""
    hosts = rng.choice(
        df[HOST_COL].unique(),
        size=min(20, df[HOST_COL].nunique()),
        replace=False,
    )

    times = np.sort(df[TIME_COL].unique())
    start_idx = int(len(times) * 0.35)
    start = pd.Timestamp(times[start_idx])
    end = start + pd.Timedelta(minutes=2)

    mask = get_event_rows(df, start, end, hosts)

    for col in POWER_COLS + SYSTEM_POWER_COLS:
        df.loc[mask, col] *= 0.65

    for col in TEMP_COLS:
        df.loc[mask, col] += 3.0

    mark_event(df, mask, "multi_host_event", event_id)


def inject_scenarios(df, split_name, seed):
    """Inject controlled events into a validation or test split."""
    rng = np.random.default_rng(seed)
    out = initialize_output(df)

    # Use separate event IDs per split.
    inject_power_drop(
        out, rng, f"{split_name}_E01"
    )
    inject_thermal_anomaly(
        out, rng, f"{split_name}_E02"
    )
    inject_transient_spike(
        out, rng, f"{split_name}_E03"
    )
    inject_gradual_degradation(
        out, rng, f"{split_name}_E04"
    )
    inject_multi_host_event(
        out, rng, f"{split_name}_E05"
    )

    return out


def validate_split(df, split_name, require_normal_only=False):
    """Check labels, events, duplicates and telemetry validity."""
    if df["event_id"].isna().any():
        raise ValueError(
            f"{split_name}: missing event IDs."
        )

    if require_normal_only and (df[LABEL_COL] != 0).any():
        raise ValueError(
            f"{split_name}: training data is not normal-only."
        )

    if not set(df[LABEL_COL].unique()).issubset({0, 1}):
        raise ValueError(
            f"{split_name}: unexpected label values."
        )

    if df.duplicated([HOST_COL, TIME_COL]).any():
        raise ValueError(
            f"{split_name}: duplicate host-timestamp rows."
        )

    bad_features = [
        c for c in TELEMETRY_COLS
        if df[c].isna().any()
        or not np.isfinite(df[c].to_numpy(dtype=float)).all()
    ]

    if bad_features:
        raise ValueError(
            f"{split_name}: invalid telemetry values: "
            f"{bad_features}"
        )

    if split_name != "train" and not (
        df[LABEL_COL] == 1
    ).any():
        raise ValueError(
            f"{split_name}: no injected anomalies found."
        )


def save_split(df, filename):
    path = OUTPUT_DIR / filename
    df.to_parquet(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")


def write_report(train, val, test):
    """Write a reproducibility and data quality report."""
    report_path = OUTPUT_DIR / "synthetic_dataset_report.txt"

    lines = [
        "SENTINEL 2.0 SYNTHETIC DATASET REPORT",
        "=" * 48,
        f"Random seed: {SEED}",
        f"Source file: {SOURCE_FILE}",
        f"Sample interval: {SAMPLE_SECONDS} seconds",
        f"Telemetry features: {len(TELEMETRY_COLS)}",
        "",
    ]

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:
        lines.extend([
            f"{name}",
            "-" * 48,
            f"Rows: {len(df):,}",
            f"Hosts: {df[HOST_COL].nunique():,}",
            f"Time start: {df[TIME_COL].min()}",
            f"Time end: {df[TIME_COL].max()}",
            f"Label counts: {df[LABEL_COL].value_counts().to_dict()}",
            "Anomaly type counts:",
            df["anomaly_type"].value_counts().to_string(),
            "",
        ])

    lines.extend([
        "IMPORTANT LIMITATIONS",
        "-" * 48,
        "Injected anomalies have known synthetic ground truth.",
        "They are not confirmed real hardware failures.",
        "XID and identifiers must not be model input features.",
        "The test set contains controlled synthetic scenarios.",
        "Synthetic performance does not establish real-world",
        "failure-detection performance.",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {report_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original normal telemetry...")
    source = load_source()

    print(f"Source rows: {len(source):,}")
    print(f"Source hosts: {source[HOST_COL].nunique():,}")
    print(
        f"Source time range: {source[TIME_COL].min()} "
        f"to {source[TIME_COL].max()}"
    )

    train_raw, val_raw, test_raw = split_chronologically(
        source
    )

    train = initialize_output(train_raw)
    val = inject_scenarios(
        val_raw, "validation", SEED + 1
    )
    test = inject_scenarios(
        test_raw, "test", SEED + 2
    )

    validate_split(
        train, "train", require_normal_only=True
    )
    validate_split(val, "validation")
    validate_split(test, "test")

    save_split(train, "synthetic_train.parquet")
    save_split(val, "synthetic_val.parquet")
    save_split(test, "synthetic_test.parquet")

    write_report(train, val, test)

    print("\nAll initial validation checks passed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
