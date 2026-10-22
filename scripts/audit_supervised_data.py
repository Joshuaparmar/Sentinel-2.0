from pathlib import Path
import pandas as pd
import hashlib

root = Path("synthetic_output_v2")
train_path = root / "synthetic_train.parquet"
supervised_path = root / "synthetic_supervised_train.parquet"
events_path = root / "synthetic_supervised_train_events.csv"

for p in (train_path, supervised_path, events_path):
    if not p.is_file():
        raise SystemExit("STOP: Missing expected file: " + str(p))

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

normal = pd.read_parquet(train_path)
sup = pd.read_parquet(supervised_path)
events = pd.read_csv(events_path)

print("NORMAL TRAIN:", normal.shape)
print("SUPERVISED TRAIN:", sup.shape)
print("Original train SHA-256:", sha256(train_path))

print("\nLabel counts:")
print(sup["label"].value_counts(dropna=False).sort_index())

print("\nAnomaly types:")
print(sup["anomaly_type"].value_counts(dropna=False))

print("\nEvent records:", len(events))

print("\nIntegrity checks:")
print("Same row count:", len(normal) == len(sup))
print("Original training labels all normal:",
      "label" in normal.columns and normal["label"].eq(0).all())
print("Supervised labels are binary:",
      set(sup["label"].dropna().unique()).issubset({0, 1}))
print("No missing labels:", sup["label"].notna().all())
print("Every anomalous row has a non-normal type:",
      sup.loc[sup["label"] == 1, "anomaly_type"].ne("normal").all())
print("Every anomalous row has an event ID:",
      sup.loc[sup["label"] == 1, "event_id"].notna().all()
      and sup.loc[sup["label"] == 1, "event_id"].astype(str).ne("").all())
print("Original columns present in supervised data:",
      all(c in sup.columns for c in normal.columns))
print("Original train SHA-256 after check:", sha256(train_path))
