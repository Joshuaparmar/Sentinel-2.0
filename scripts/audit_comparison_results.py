#!/usr/bin/env python
"""Read-only audit of saved SENTINEL V3 model-comparison results."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "reports" / "model_comparison_v3_run_001"
CSV_PATH = RUN_DIR / "model_comparison.csv"
JSON_PATH = RUN_DIR / "model_comparison.json"
META_PATH = RUN_DIR / "run_metadata.json"


def load_json(path):
    if not path.is_file():
        print(f"MISSING: {path.relative_to(ROOT)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"ERROR reading {path.name}: {exc}")
        return None


def print_csv(path):
    print(f"\n{'=' * 78}\nCSV: {path.relative_to(ROOT)}\n{'=' * 78}")
    if not path.is_file():
        print("MISSING")
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Rows: {len(rows)}")
        print("Columns:", " | ".join(reader.fieldnames or []))
        for i, row in enumerate(rows, 1):
            print(f"\n--- Result {i} ---")
            for key, value in row.items():
                print(f"{key}: {value}")
    except Exception as exc:
        print(f"ERROR reading CSV: {exc}")


def show_json(path, label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    data = load_json(path)
    if data is None:
        return
    if isinstance(data, dict):
        print("Top-level keys:", ", ".join(data.keys()))
    print(json.dumps(data, indent=2, ensure_ascii=False)[:18000])
    if isinstance(data, (dict, list)) and len(json.dumps(data)) > 18000:
        print("\n[Output truncated; original file was not changed.]")


def main():
    print("SENTINEL V3 SAVED-RESULTS AUDIT (READ ONLY)")
    print(f"Project: {ROOT}")
    print(f"Run directory exists: {RUN_DIR.is_dir()}")

    for path in (CSV_PATH, JSON_PATH, META_PATH):
        if path.is_file():
            print(f"Artifact: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
        else:
            print(f"Artifact MISSING: {path.relative_to(ROOT)}")

    print_csv(CSV_PATH)
    show_json(JSON_PATH, "COMPARISON JSON")
    show_json(META_PATH, "RUN METADATA")

    print("\nAudit complete. No models were trained and no experiment files were modified.")


if __name__ == "__main__":
    main()
