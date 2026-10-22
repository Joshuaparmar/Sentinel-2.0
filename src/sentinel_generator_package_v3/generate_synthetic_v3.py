"""SENTINEL 2.0 V3 synthetic telemetry generator.

Reads a source Parquet file (read-only), splits it chronologically with a purge
gap, and writes four datasets plus event manifests, a feature manifest, the
resolved configuration, validation results and reports:

    synthetic_train.parquet             normal-only training data (unsupervised models)
    synthetic_supervised_train.parquet  copy of the above with labelled events (XGBoost/LightGBM)
    synthetic_val.parquet               validation, baseline-severity events (threshold selection)
    synthetic_test.parquet              test, baseline + separately tagged held-out severity events

Design summary
--------------
* Events modify existing rows in place (no rows added or removed). All feature
  columns keep one dtype across all datasets (float32 by default).
* Event parameters come from ONE configuration object (`GeneratorConfig`); the
  reports are generated from that same object and from the generated data.
* Randomness: `SeedSequence(seed)` with fixed named streams (host pools,
  supervised_train, validation, test) plus one seed per event (stored in the
  manifest). Same source + same config => same output.
* Nothing is silently truncated or clipped. If events do not fit, generation fails
  with a capacity message. Realized effects are measured on the final float32
  values and must match the configured targets within tolerance.
* Output is built in a staging directory, validated, and only then renamed to the
  final directory. Existing output is never overwritten.

Scope: a passing run certifies the DATA-GENERATION process (reproducible, correctly
labelled, internally consistent). It does not show that synthetic anomalies resemble
real GPU faults, that any model will perform well, or that a validation FPR target
is met - those are model-evaluation questions.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GENERATOR_VERSION = "3.0.0"

# ----------------------------------------------------------------------------
# Column conventions (unchanged from the existing generator)
# ----------------------------------------------------------------------------
TIME_CANDIDATES = ("timestamp", "time", "datetime", "ts")
HOST_CANDIDATES = ("hostname", "host", "node", "node_name")
LABEL_COL = "label"
TYPE_COL = "anomaly_type"
EVENT_COL = "event_id"
SOURCE_LABEL_COL = "source_label"
SOURCE_ROW_COL = "source_row_id"
ADDED_COLS = (SOURCE_LABEL_COL, SOURCE_ROW_COL, TYPE_COL, EVENT_COL)
METADATA_EXCLUDE = {
    LABEL_COL, TYPE_COL, EVENT_COL, SOURCE_LABEL_COL, SOURCE_ROW_COL,
    "xid", "hostname", "host", "node", "node_name", "timestamp", "time",
    "datetime", "ts", "node_state", "allocation_id", "job_id",
    "job_begin_time", "job_end_time",
}
POWER_HINTS = ("power", "watt", "pwr")
TEMP_HINTS = ("temp", "temperature")
SUMMARY_TOLERANCE = 1e-4  # tolerance for min <= mean <= max summary checks
NORMAL = "normal"

SCENARIO_NAMES = (
    "abrupt_power_drop", "thermal_anomaly", "transient_spike",
    "gradual_degradation", "multi_host_event",
)
POWER_SCENARIOS = {"abrupt_power_drop", "transient_spike", "gradual_degradation", "multi_host_event"}
TEMP_SCENARIOS = {"thermal_anomaly", "multi_host_event"}
CONST_MULTIPLIER_SCENARIOS = {"abrupt_power_drop", "transient_spike", "multi_host_event"}

# Severity dimensions: name -> kind. "multiplier" scales values, "final_factor" is the
# end point of a linear 1.0 -> factor trajectory, "additive_offset" adds degrees.
DIMENSION_KINDS = {
    "power_multiplier": "multiplier",
    "final_power_factor": "final_factor",
    "temp_offset": "additive_offset",
}
NEUTRAL_VALUE = {"multiplier": 1.0, "final_factor": 1.0, "additive_offset": 0.0}
REQUIRED_DIMENSIONS = {
    "abrupt_power_drop": ("power_multiplier",),
    "thermal_anomaly": ("temp_offset",),
    "transient_spike": ("power_multiplier",),
    "gradual_degradation": ("final_power_factor",),
    "multi_host_event": ("power_multiplier", "temp_offset"),
}

SPLIT_NAMES = ("supervised_train", "validation", "test")
DATASET_FILES = {
    "train": "synthetic_train.parquet",
    "supervised_train": "synthetic_supervised_train.parquet",
    "validation": "synthetic_val.parquet",
    "test": "synthetic_test.parquet",
}
EVENTS_FILE = "synthetic_events.csv"
SUPERVISED_EVENTS_FILE = "synthetic_supervised_train_events.csv"
OUTPUT_FILES = tuple(DATASET_FILES.values()) + (
    EVENTS_FILE, SUPERVISED_EVENTS_FILE, "feature_manifest.json", "generation_config.json",
    "validation_results.json", "synthetic_report.json", "synthetic_report.txt",
)
STREAM_IDS = {"host_pools": 0, "supervised_train": 1, "validation": 2, "test": 3}
SPLIT_PREFIX = {"supervised_train": "SUPTRAIN", "validation": "VAL", "test": "TEST"}


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------
class GeneratorError(Exception):
    """Base class for all expected generator failures."""


class ConfigError(GeneratorError, ValueError):
    pass


class SourceError(GeneratorError, ValueError):
    pass


class CapacityError(GeneratorError, ValueError):
    pass


class EventError(GeneratorError, ValueError):
    pass


class OutputExistsError(GeneratorError, FileExistsError):
    pass


class ValidationFailed(GeneratorError):
    pass


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Dimension:
    name: str       # power_multiplier | final_power_factor | temp_offset
    kind: str       # multiplier | final_factor | additive_offset
    direction: str  # decrease | increase (relative to the neutral value)


@dataclass(frozen=True)
class SeverityBand:
    name: str                                  # e.g. "baseline", "held_out"
    held_out: bool                             # withheld from supervised training
    status: str                                # e.g. "baseline_existing", "proposed_unvalidated"
    ranges: dict[str, tuple[float, float]]     # dimension name -> (low, high), inclusive


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    dimensions: tuple[Dimension, ...]
    duration_steps: tuple[int, int]            # inclusive; 1 step = sample_interval_seconds
    hosts: tuple[int, int]                     # hosts affected per event, inclusive
    bands: tuple[SeverityBand, ...]


@dataclass(frozen=True)
class SplitPlan:
    placement: str                              # "global" (no two events overlap in time) | "per_host"
    event_counts: dict[str, dict[str, int]]     # scenario -> band name -> number of events


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    train_frac: float = 0.60
    val_frac: float = 0.20
    purge_steps: int = 30             # timestamps dropped between partitions
    edge_margin_steps: int = 30       # no events this close to a partition edge
    guard_steps: int = 30             # minimum normal steps between events (same time / same host)
    sample_interval_seconds: int = 10
    feature_dtype: str = "float32"
    max_placement_attempts: int = 1000
    allow_band_overlap: bool = False  # allow held-out ranges to overlap baseline ranges
    realized_ratio_tol: float = 1e-4  # tolerance on realized/baseline power ratios
    realized_offset_tol: float = 1e-3  # tolerance on realized temperature offsets (degrees)
    power_floor: float = 1e-6         # baseline power values at/below this are excluded from ratios
    envelope_quantiles: tuple[float, float] = (0.001, 0.999)
    envelope_headroom: float = 0.10   # descriptive only; never used for clipping
    scenarios: dict[str, ScenarioSpec] = field(default_factory=dict)
    split_plans: dict[str, SplitPlan] = field(default_factory=dict)


def default_scenarios() -> dict[str, ScenarioSpec]:
    """Baseline = the existing generator's ranges. held_out = proposed, NOT yet validated."""
    pm_dec = Dimension("power_multiplier", "multiplier", "decrease")
    pm_inc = Dimension("power_multiplier", "multiplier", "increase")
    to_inc = Dimension("temp_offset", "additive_offset", "increase")
    ff_dec = Dimension("final_power_factor", "final_factor", "decrease")

    def band(held, ranges):
        return SeverityBand(
            name="held_out" if held else "baseline", held_out=held,
            status="proposed_unvalidated" if held else "baseline_existing", ranges=ranges,
        )

    return {
        "abrupt_power_drop": ScenarioSpec(
            "abrupt_power_drop", (pm_dec,), (12, 18), (1, 1),
            (band(False, {"power_multiplier": (0.35, 0.50)}),
             band(True, {"power_multiplier": (0.20, 0.30)}))),
        "thermal_anomaly": ScenarioSpec(
            "thermal_anomaly", (to_inc,), (18, 24), (1, 1),
            (band(False, {"temp_offset": (5.0, 8.0)}),
             band(True, {"temp_offset": (9.0, 12.0)}))),
        "transient_spike": ScenarioSpec(
            "transient_spike", (pm_inc,), (2, 4), (1, 1),
            (band(False, {"power_multiplier": (1.35, 1.65)}),
             band(True, {"power_multiplier": (1.70, 1.90)}))),
        "gradual_degradation": ScenarioSpec(
            "gradual_degradation", (ff_dec,), (24, 36), (1, 1),
            (band(False, {"final_power_factor": (0.45, 0.65)}),
             band(True, {"final_power_factor": (0.25, 0.40)}))),
        "multi_host_event": ScenarioSpec(
            "multi_host_event", (pm_dec, to_inc), (9, 14), (4, 4),
            (band(False, {"power_multiplier": (0.60, 0.75), "temp_offset": (2.0, 2.0)}),
             band(True, {"power_multiplier": (0.40, 0.55), "temp_offset": (3.0, 5.0)}))),
    }


def _counts(**by_band: int) -> dict[str, dict[str, int]]:
    return {name: dict(by_band) for name in SCENARIO_NAMES}


def default_split_plans() -> dict[str, SplitPlan]:
    return {
        "supervised_train": SplitPlan("per_host", _counts(baseline=60)),
        "validation": SplitPlan("global", _counts(baseline=6)),
        "test": SplitPlan("global", _counts(baseline=3, held_out=3)),
    }


def default_config(**overrides: Any) -> GeneratorConfig:
    base = GeneratorConfig(scenarios=default_scenarios(), split_plans=default_split_plans())
    return dataclasses.replace(base, **overrides) if overrides else base


def _range_ok(rng: tuple[float, float]) -> bool:
    return len(rng) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in rng) and rng[0] <= rng[1]


def _disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[1] < b[0] or b[1] < a[0]


def validate_config(cfg: GeneratorConfig) -> None:
    """Reject invalid or incompatible configurations with a specific message."""
    errs: list[str] = []
    if not (0 < cfg.train_frac < 1 and 0 < cfg.val_frac < 1 and cfg.train_frac + cfg.val_frac < 1):
        errs.append("train_frac and val_frac must be in (0,1) with train_frac + val_frac < 1")
    for name in ("purge_steps", "edge_margin_steps", "guard_steps"):
        v = getattr(cfg, name)
        if not isinstance(v, int) or v < 0:
            errs.append(f"{name} must be a non-negative integer")
    if cfg.sample_interval_seconds <= 0:
        errs.append("sample_interval_seconds must be positive")
    if cfg.max_placement_attempts < 1:
        errs.append("max_placement_attempts must be >= 1")
    if cfg.feature_dtype not in ("float32", "float64"):
        errs.append("feature_dtype must be 'float32' or 'float64'")
    for name in ("realized_ratio_tol", "realized_offset_tol", "power_floor"):
        if not getattr(cfg, name) > 0:
            errs.append(f"{name} must be positive")

    missing = [s for s in SCENARIO_NAMES if s not in cfg.scenarios]
    unknown = [s for s in cfg.scenarios if s not in SCENARIO_NAMES]
    if missing:
        errs.append(f"missing scenario definitions: {missing}")
    if unknown:
        errs.append(f"unknown scenarios: {unknown}")

    for sname, spec in cfg.scenarios.items():
        if sname not in SCENARIO_NAMES:
            continue
        dims = {d.name: d for d in spec.dimensions}
        if set(dims) != set(REQUIRED_DIMENSIONS[sname]):
            errs.append(f"{sname}: dimensions must be {list(REQUIRED_DIMENSIONS[sname])}, got {list(dims)}")
            continue
        for d in dims.values():
            if DIMENSION_KINDS.get(d.name) != d.kind:
                errs.append(f"{sname}.{d.name}: kind must be {DIMENSION_KINDS.get(d.name)!r}, got {d.kind!r}")
            if d.direction not in ("decrease", "increase"):
                errs.append(f"{sname}.{d.name}: direction must be 'decrease' or 'increase'")
        for label, (lo, hi) in (("duration_steps", spec.duration_steps), ("hosts", spec.hosts)):
            if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi):
                errs.append(f"{sname}.{label}: need integers with 1 <= low <= high, got {(lo, hi)}")
        names = [b.name for b in spec.bands]
        if len(set(names)) != len(names) or not names:
            errs.append(f"{sname}: band names must be unique and non-empty")
        for b in spec.bands:
            if set(b.ranges) != set(dims):
                errs.append(f"{sname}/{b.name}: ranges must cover exactly {sorted(dims)}")
                continue
            for dname, rng in b.ranges.items():
                if not _range_ok(tuple(rng)):
                    errs.append(f"{sname}/{b.name}.{dname}: invalid range {rng} (need finite low <= high)")
                    continue
                d = dims[dname]
                neutral = NEUTRAL_VALUE[d.kind]
                if d.kind != "additive_offset" and rng[0] <= 0:
                    errs.append(f"{sname}/{b.name}.{dname}: factors must be > 0, got {rng}")
                if d.direction == "decrease" and not rng[1] < neutral:
                    errs.append(f"{sname}/{b.name}.{dname}: a 'decrease' range must lie below {neutral}, got {rng}")
                if d.direction == "increase" and not rng[0] > neutral:
                    errs.append(f"{sname}/{b.name}.{dname}: an 'increase' range must lie above {neutral}, got {rng}")
        if not cfg.allow_band_overlap:
            for h in (b for b in spec.bands if b.held_out):
                for base in (b for b in spec.bands if not b.held_out):
                    overlapping = [
                        dn for dn in dims
                        if dn in h.ranges and dn in base.ranges
                        and not _disjoint(tuple(h.ranges[dn]), tuple(base.ranges[dn]))
                    ]
                    if overlapping:
                        errs.append(
                            f"{sname}: held-out band '{h.name}' overlaps baseline band '{base.name}' in "
                            f"{overlapping}; shrink the ranges or set allow_band_overlap=True (documented)")

    if set(cfg.split_plans) != set(SPLIT_NAMES):
        errs.append(f"split_plans must define exactly {list(SPLIT_NAMES)}")
    else:
        for split, plan in cfg.split_plans.items():
            if plan.placement not in ("global", "per_host"):
                errs.append(f"{split}: placement must be 'global' or 'per_host'")
            for sname in SCENARIO_NAMES:
                counts = plan.event_counts.get(sname)
                spec = cfg.scenarios.get(sname)
                if counts is None:
                    errs.append(f"{split}: no event counts for scenario {sname}")
                    continue
                if spec is None:
                    continue
                band_map = {b.name: b for b in spec.bands}
                for bname, n in counts.items():
                    if bname not in band_map:
                        errs.append(f"{split}/{sname}: unknown band '{bname}'")
                    elif not isinstance(n, int) or n < 0:
                        errs.append(f"{split}/{sname}/{bname}: event count must be a non-negative integer")
                    elif band_map[bname].held_out and split != "test" and n > 0:
                        errs.append(f"{split}/{sname}/{bname}: held-out bands may only be used in the test split")
                if sum(n for n in counts.values() if isinstance(n, int)) < 1:
                    errs.append(f"{split}/{sname}: needs at least one event (all five scenarios must be exercised)")
        test_plan = cfg.split_plans["test"]
        held_total = base_total = 0
        for sname, counts in test_plan.event_counts.items():
            spec = cfg.scenarios.get(sname)
            if spec is None:
                continue
            for b in spec.bands:
                n = counts.get(b.name, 0)
                if isinstance(n, int):
                    held_total += n if b.held_out else 0
                    base_total += 0 if b.held_out else n
        if held_total < 1 or base_total < 1:
            errs.append("test split needs at least one baseline and one held-out event")
    if errs:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(errs))


def config_to_dict(cfg: GeneratorConfig) -> dict[str, Any]:
    d = dataclasses.asdict(cfg)
    d["scenarios"] = {k: dataclasses.asdict(v) for k, v in cfg.scenarios.items()}
    return json.loads(json.dumps(d))  # tuples -> lists, plain JSON types


def config_from_dict(d: dict[str, Any]) -> GeneratorConfig:
    d = dict(d)
    allowed = {f.name for f in dataclasses.fields(GeneratorConfig)}
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"Unknown configuration keys: {sorted(unknown)}")
    scenarios = {}
    for name, s in d.pop("scenarios", {}).items():
        scenarios[name] = ScenarioSpec(
            name=s["name"],
            dimensions=tuple(Dimension(**x) for x in s["dimensions"]),
            duration_steps=tuple(s["duration_steps"]), hosts=tuple(s["hosts"]),
            bands=tuple(SeverityBand(
                name=b["name"], held_out=b["held_out"], status=b["status"],
                ranges={k: tuple(v) for k, v in b["ranges"].items()}) for b in s["bands"]),
        )
    plans = {k: SplitPlan(v["placement"], {s: dict(c) for s, c in v["event_counts"].items()})
             for k, v in d.pop("split_plans", {}).items()}
    if "envelope_quantiles" in d:
        d["envelope_quantiles"] = tuple(d["envelope_quantiles"])
    return GeneratorConfig(scenarios=scenarios, split_plans=plans, **d)


# ----------------------------------------------------------------------------
# Source loading and audit
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_path(path: Path) -> str:
    """SHA-256 of a file, or of every file in a directory (relative path + content)."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Protected path not found: {path}")
    digest = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(f.relative_to(path)).replace("\\", "/").encode())
        digest.update(sha256_file(f).encode())
    return digest.hexdigest()


def frame_digest(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([[str(c), str(t)] for c, t in zip(df.columns, df.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _find_column(columns, candidates, description):
    lookup = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    raise SourceError(f"Could not identify {description} column. Tried: {candidates}")


@dataclass
class FeatureInfo:
    features: list[str]
    power: list[str]
    temperature: list[str]
    power_idx: list[int]
    temp_idx: list[int]
    summary_groups: list[dict[str, str]]


def _temp_summary_groups(cols: list[str]) -> list[dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for col in cols:
        name = str(col).lower()
        for token in ("_min", "_mean", "_avg", "_max"):
            if name.endswith(token):
                groups.setdefault(name[:-len(token)], {})[token[1:]] = col
                break
    return [g for g in groups.values() if "min" in g and ("mean" in g or "avg" in g) and "max" in g]


def identify_features(df: pd.DataFrame, time_col: str, host_col: str) -> FeatureInfo:
    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and str(c).lower() not in METADATA_EXCLUDE and c not in {time_col, host_col}]
    power = [c for c in numeric if any(h in str(c).lower() for h in POWER_HINTS)]
    temps = [c for c in numeric if any(h in str(c).lower() for h in TEMP_HINTS)]
    both = sorted(set(power) & set(temps))
    if both:
        raise SourceError(f"Columns match both power and temperature hints: {both}")
    features = sorted(set(power + temps))
    if not power:
        raise SourceError("No numeric power features found (name hints: power/watt/pwr); "
                          "power scenarios cannot be generated.")
    if not temps:
        raise SourceError("No numeric temperature features found (name hint: temp); "
                          "thermal scenarios cannot be generated.")
    return FeatureInfo(
        features=features, power=sorted(power), temperature=sorted(temps),
        power_idx=[features.index(c) for c in sorted(power)],
        temp_idx=[features.index(c) for c in sorted(temps)],
        summary_groups=_temp_summary_groups(sorted(temps)),
    )


def summary_violations(df: pd.DataFrame, groups: list[dict[str, str]]) -> np.ndarray:
    """Row-wise boolean: some min <= mean <= max relationship is violated."""
    bad = np.zeros(len(df), dtype=bool)
    for g in groups:
        mn = df[g["min"]].to_numpy(dtype=np.float64)
        me = df[g.get("mean", g.get("avg"))].to_numpy(dtype=np.float64)
        mx = df[g["max"]].to_numpy(dtype=np.float64)
        bad |= (mn > me + SUMMARY_TOLERANCE) | (me > mx + SUMMARY_TOLERANCE) | (mn > mx + SUMMARY_TOLERANCE)
    return bad


def prepare_source(raw: pd.DataFrame) -> pd.DataFrame:
    """Tag original row order, sort by (timestamp, host), add provenance columns."""
    clash = sorted(set(raw.columns) & set(ADDED_COLS))
    if clash:
        raise SourceError(f"Source contains reserved generator column(s): {clash}. Rename them first.")
    time_col = _find_column(raw.columns, TIME_CANDIDATES, "timestamp")
    host_col = _find_column(raw.columns, HOST_CANDIDATES, "host")
    if raw[time_col].isna().any() or raw[host_col].isna().any():
        raise SourceError("Timestamp or host column contains missing values.")
    df = raw.copy()
    df[SOURCE_ROW_COL] = np.arange(len(df), dtype=np.int64)
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
        except (ValueError, TypeError) as exc:
            raise SourceError(f"Timestamp column cannot be parsed: {exc}") from exc
    df = df.sort_values([time_col, host_col], kind="mergesort").reset_index(drop=True)
    if LABEL_COL in df.columns:
        if df[LABEL_COL].isna().any():
            raise SourceError("Source label column contains missing values.")
        df[SOURCE_LABEL_COL] = df[LABEL_COL].astype("int8")
    else:
        df[LABEL_COL] = np.zeros(len(df), dtype=np.int8)
        df[SOURCE_LABEL_COL] = np.zeros(len(df), dtype=np.int8)
    return df


@dataclass
class SourceAudit:
    facts: dict[str, Any]
    feat: FeatureInfo
    time_col: str
    host_col: str
    host_names: list[str]
    n_hosts: int
    n_steps: int


def audit_source(df: pd.DataFrame, cfg: GeneratorConfig) -> SourceAudit:
    """Hard source requirements; returns the facts that are recorded in the report."""
    time_col = _find_column(df.columns, TIME_CANDIDATES, "timestamp")
    host_col = _find_column(df.columns, HOST_CANDIDATES, "host")
    if (df[SOURCE_LABEL_COL] != 0).any():
        raise SourceError(f"Source has {(df[SOURCE_LABEL_COL] != 0).sum()} non-zero labels; "
                          "V3 requires a normal-only source (anomalies would be relabelled as normal).")
    if df.duplicated([host_col, time_col]).any():
        raise SourceError("Source contains duplicate host/timestamp rows.")
    n_hosts, n_steps = int(df[host_col].nunique()), int(df[time_col].nunique())
    if len(df) != n_hosts * n_steps:
        raise SourceError(f"Incomplete host/timestamp coverage: {len(df)} rows but {n_hosts} hosts x "
                          f"{n_steps} timestamps = {n_hosts * n_steps}. Event placement needs a complete panel.")
    host_names = df[host_col].iloc[:n_hosts].astype(str).tolist()
    codes = pd.Categorical(df[host_col].astype(str), categories=host_names).codes
    if not np.array_equal(codes, np.tile(np.arange(n_hosts), n_steps)):
        raise SourceError("Rows are not ordered as (timestamp, host) blocks with a constant host order.")
    times = df[time_col].iloc[::n_hosts].reset_index(drop=True)
    deltas = times.diff().dropna().dt.total_seconds()
    if len(deltas) == 0 or float(deltas.median()) != float(cfg.sample_interval_seconds):
        raise SourceError(f"Median timestamp interval is {deltas.median() if len(deltas) else 'n/a'} s; "
                          f"expected {cfg.sample_interval_seconds} s.")
    feat = identify_features(df, time_col, host_col)
    dtypes = sorted({str(df[c].dtype) for c in feat.features})
    if dtypes != [cfg.feature_dtype]:
        raise SourceError(f"Feature dtypes are {dtypes}; configured feature_dtype is {cfg.feature_dtype!r}. "
                          "Features are never silently cast.")
    if not np.isfinite(df[feat.features].to_numpy(dtype=np.float64)).all():
        raise SourceError("Source features contain NaN/inf values.")
    interval_counts = {str(int(k)): int(v) for k, v in deltas.value_counts().head(5).items()}
    facts = {
        "rows": int(len(df)), "hosts": n_hosts, "timestamps": n_steps, "complete_panel": True,
        "duplicate_host_time_rows": 0, "source_label_nonzero": 0,
        "interval_seconds_top_counts": interval_counts,
        "irregular_intervals": int((deltas != cfg.sample_interval_seconds).sum()),
        "feature_dtype": cfg.feature_dtype, "feature_count": len(feat.features),
        "power_features": len(feat.power), "temperature_features": len(feat.temperature),
        "preexisting_summary_violations": int(summary_violations(df, feat.summary_groups).sum()),
        "time_start": str(times.iloc[0]), "time_end": str(times.iloc[-1]),
    }
    return SourceAudit(facts, feat, time_col, host_col, host_names, n_hosts, n_steps)


# ----------------------------------------------------------------------------
# Splitting and host pools
# ----------------------------------------------------------------------------
def split_bounds(n_steps: int, cfg: GeneratorConfig) -> dict[str, tuple[int, int]]:
    """Timestamp-index bounds [start, end) of each partition; `purge_steps` are dropped between them."""
    if n_steps < 10:
        raise SourceError("Need at least 10 distinct timestamps to split.")
    i1 = int(math.floor(n_steps * cfg.train_frac + 1e-9))
    i2 = int(math.floor(n_steps * (cfg.train_frac + cfg.val_frac) + 1e-9))
    p = cfg.purge_steps
    bounds = {"train": (0, i1), "validation": (i1 + p, i2), "test": (i2 + p, n_steps)}
    empty = [k for k, (a, b) in bounds.items() if b - a < 1]
    if empty:
        raise SourceError(f"Dataset too small for the requested fractions and purge gap: empty split(s) {empty}.")
    return bounds


def allocate_host_pools(n_hosts: int, seed: int) -> dict[str, tuple[int, ...]]:
    """Three disjoint pools of hosts that may receive anomalies (train / validation / test)."""
    if n_hosts < 3:
        raise CapacityError(f"Only {n_hosts} host(s): three disjoint anomaly-host pools need at least 3 hosts.")
    perm = [int(i) for i in stream_rng(seed, "host_pools").permutation(n_hosts)]
    n_each = n_hosts // 3
    n_train = n_hosts - 2 * n_each
    return {
        "supervised_train": tuple(sorted(perm[:n_train])),
        "validation": tuple(sorted(perm[n_train:n_train + n_each])),
        "test": tuple(sorted(perm[n_train + n_each:])),
    }


# ----------------------------------------------------------------------------
# Event planning (pure function of config, partition sizes, host pools and seed)
# ----------------------------------------------------------------------------
def stream_rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(STREAM_IDS[stream],)))


def event_seed(seed: int, stream: str, draft_index: int) -> int:
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(STREAM_IDS[stream], 1 + draft_index))
    return int(ss.generate_state(1)[0])


@dataclass(frozen=True)
class PlannedEvent:
    event_id: str
    split: str
    scenario: str
    band: str
    held_out: bool
    band_status: str
    params: dict[str, float]      # severity dimension -> target value
    duration: int                 # steps
    hosts: tuple[int, ...]        # host indices
    start_step: int               # first step within the partition
    event_seed: int


@dataclass
class _Draft:
    index: int
    scenario: str
    band: SeverityBand
    params: dict[str, float]
    duration: int
    n_hosts: int


def _stratified(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    """n values covering [lo, hi] evenly (one per stratum), in random order."""
    if lo == hi:
        return np.full(n, float(lo))
    u = (rng.permutation(n) + rng.random(n)) / n
    return np.clip(lo + u * (hi - lo), lo, hi)


def _stratified_int(rng: np.random.Generator, n: int, lo: int, hi: int) -> np.ndarray:
    if lo == hi:
        return np.full(n, lo, dtype=int)
    u = (rng.permutation(n) + rng.random(n)) / n
    return np.minimum(lo + np.floor(u * (hi - lo + 1)).astype(int), hi)


def _draft_events(cfg: GeneratorConfig, plan: SplitPlan, rng: np.random.Generator) -> list[_Draft]:
    drafts: list[_Draft] = []
    for sname in sorted(plan.event_counts):
        spec = cfg.scenarios[sname]
        bands = {b.name: b for b in spec.bands}
        for bname in sorted(plan.event_counts[sname]):
            n = plan.event_counts[sname][bname]
            if n == 0:
                continue
            band = bands[bname]
            values = {d.name: _stratified(rng, n, *band.ranges[d.name]) for d in spec.dimensions}
            durations = _stratified_int(rng, n, *spec.duration_steps)
            for i in range(n):
                n_hosts = spec.hosts[0] if spec.hosts[0] == spec.hosts[1] else int(rng.integers(spec.hosts[0], spec.hosts[1] + 1))
                drafts.append(_Draft(len(drafts), sname, band, {k: float(v[i]) for k, v in values.items()},
                                     int(durations[i]), n_hosts))
    return drafts


def _pick_hosts(rng: np.random.Generator, pool: tuple[int, ...], usage: dict[int, int], k: int) -> tuple[int, ...]:
    """Least-used hosts first (random tie-break) so events spread over the pool."""
    keys = rng.random(len(pool))
    order = sorted(range(len(pool)), key=lambda j: (usage[pool[j]], keys[j]))
    return tuple(sorted(pool[j] for j in order[:k]))


def _host_free(busy: list[tuple[int, int]], start: int, end: int, guard: int) -> bool:
    return all(not (s <= end + guard and e >= start - guard) for s, e in busy)


def plan_split(cfg: GeneratorConfig, split: str, n_steps: int, pool: tuple[int, ...], seed: int) -> list[PlannedEvent]:
    """Decide scenario, severity, duration, hosts and start step for every event of one split."""
    plan = cfg.split_plans[split]
    rng = stream_rng(seed, split)
    drafts = _draft_events(cfg, plan, rng)
    if not drafts:
        return []
    margin, guard = cfg.edge_margin_steps, cfg.guard_steps
    usable = n_steps - 2 * margin
    need_hosts = max(d.n_hosts for d in drafts)
    if need_hosts > len(pool):
        raise CapacityError(f"{split}: events need {need_hosts} hosts but the host pool has {len(pool)}.")
    if usable < max(d.duration for d in drafts):
        raise CapacityError(f"{split}: only {usable} usable steps ({n_steps} minus 2x{margin} edge margin); "
                            f"longest event needs {max(d.duration for d in drafts)}.")
    usage = {h: 0 for h in pool}
    order = [int(i) for i in rng.permutation(len(drafts))]
    placed: list[tuple[_Draft, tuple[int, ...], int, int]] = []

    if plan.placement == "global":
        total = sum(drafts[i].duration for i in order) + guard * (len(order) - 1)
        slack = usable - total
        if slack < 0:
            raise CapacityError(
                f"{split}: {len(order)} events need {total} steps (event durations "
                f"{total - guard * (len(order) - 1)} + {len(order) - 1} guard gaps of {guard}) but only "
                f"{usable} usable steps exist ({n_steps} minus 2x{margin} edge margin). Reduce event counts "
                f"or guard_steps/edge_margin_steps; events are never shortened.")
        cuts = np.sort(rng.integers(0, slack + 1, size=len(order)))
        extras = np.diff(np.concatenate(([0], cuts)))
        cursor = margin
        for j, i in enumerate(order):
            d = drafts[i]
            cursor += int(extras[j])
            seed_e = event_seed(seed, split, d.index)
            hosts = _pick_hosts(np.random.default_rng(seed_e), pool, usage, d.n_hosts)
            for h in hosts:
                usage[h] += 1
            placed.append((d, hosts, cursor, seed_e))
            cursor += d.duration + guard
    else:
        need = sum(d.n_hosts * (d.duration + guard) for d in drafts)
        capacity = len(pool) * (usable + guard)
        if need > capacity:
            raise CapacityError(f"{split}: events need {need} host-steps (incl. guards) but the host pool offers "
                                f"at most {capacity}. Reduce event counts or enlarge the pool.")
        busy: dict[int, list[tuple[int, int]]] = {h: [] for h in pool}
        for i in order:
            d = drafts[i]
            seed_e = event_seed(seed, split, d.index)
            rng_e = np.random.default_rng(seed_e)
            lo, hi = margin, n_steps - margin - d.duration
            for _ in range(cfg.max_placement_attempts):
                hosts = _pick_hosts(rng_e, pool, usage, d.n_hosts)
                start = int(rng_e.integers(lo, hi + 1))
                end = start + d.duration - 1
                if all(_host_free(busy[h], start, end, guard) for h in hosts):
                    for h in hosts:
                        busy[h].append((start, end))
                        usage[h] += 1
                    placed.append((d, hosts, start, seed_e))
                    break
            else:
                raise EventError(f"{split}: could not place a {d.scenario} event of {d.duration} steps on "
                                 f"{d.n_hosts} host(s) after {cfg.max_placement_attempts} attempts; "
                                 "reduce event counts or guard_steps.")

    placed.sort(key=lambda t: (t[2], t[1]))
    return [
        PlannedEvent(
            event_id=f"{SPLIT_PREFIX[split]}_E{n:03d}", split=split, scenario=d.scenario, band=d.band.name,
            held_out=d.band.held_out, band_status=d.band.status, params=d.params, duration=d.duration,
            hosts=hosts, start_step=start, event_seed=seed_e)
        for n, (d, hosts, start, seed_e) in enumerate(placed, start=1)
    ]


def plan_all(cfg: GeneratorConfig, steps: dict[str, int], pools: dict[str, tuple[int, ...]]) -> dict[str, list[PlannedEvent]]:
    return {s: plan_split(cfg, s, steps[s], pools[s], cfg.seed) for s in SPLIT_NAMES}


# ----------------------------------------------------------------------------
# Event injection and realized-effect measurement
# ----------------------------------------------------------------------------
def event_rows(ev: PlannedEvent, n_hosts: int) -> np.ndarray:
    """Row positions of an event in its partition, host-major (all steps of host 1, then host 2 ...)."""
    steps = np.arange(ev.start_step, ev.start_step + ev.duration)
    return (steps[None, :] * n_hosts + np.asarray(ev.hosts)[:, None]).ravel()


def gradual_factors(final_factor: float, duration: int) -> np.ndarray:
    """factor_k = 1 + (final - 1) * (k + 1) / n, k = 0..n-1: the first labelled row already changes."""
    return 1.0 + (final_factor - 1.0) * (np.arange(duration) + 1.0) / duration


def _apply_transform(work: np.ndarray, rows: np.ndarray, ev: PlannedEvent, feat: FeatureInfo) -> None:
    p, t = feat.power_idx, feat.temp_idx
    if ev.scenario in CONST_MULTIPLIER_SCENARIOS:
        work[np.ix_(rows, p)] = work[np.ix_(rows, p)] * ev.params["power_multiplier"]
    elif ev.scenario == "gradual_degradation":
        factors = np.tile(gradual_factors(ev.params["final_power_factor"], ev.duration), len(ev.hosts))
        work[np.ix_(rows, p)] = work[np.ix_(rows, p)] * factors[:, None]
    if ev.scenario in TEMP_SCENARIOS:
        work[np.ix_(rows, t)] = work[np.ix_(rows, t)] + ev.params["temp_offset"]


def measure_event(ev: PlannedEvent, before: np.ndarray, after: np.ndarray, feat: FeatureInfo,
                  cfg: GeneratorConfig, limit_max: np.ndarray) -> dict[str, Any]:
    """Measure what an event actually did (final stored values vs baseline); raise if it is wrong.

    `before`/`after` hold the event rows (host-major) of all feature columns.
    """
    k, d = len(ev.hosts), ev.duration
    b = before.astype(np.float64).reshape(k, d, -1)
    a = after.astype(np.float64).reshape(k, d, -1)
    p, t = feat.power_idx, feat.temp_idx
    changes_power, changes_temp = ev.scenario in POWER_SCENARIOS, ev.scenario in TEMP_SCENARIOS
    out: dict[str, Any] = {}

    if not changes_power and not np.array_equal(a[:, :, p], b[:, :, p]):
        raise EventError(f"{ev.event_id}: power features changed in a {ev.scenario} event")
    if not changes_temp and not np.array_equal(a[:, :, t], b[:, :, t]):
        raise EventError(f"{ev.event_id}: temperature features changed in a {ev.scenario} event")
    changed_rows = (a != b).any(axis=2)
    if not changed_rows.all():
        raise EventError(f"{ev.event_id}: {int((~changed_rows).sum())} labelled rows have no feature change")

    exceed_cols: list[int] = []
    if changes_power:
        exceed_cols += p
        pb, pa = b[:, :, p], a[:, :, p]
        valid = np.abs(pb) > cfg.power_floor
        if not valid.any():
            raise EventError(f"{ev.event_id}: no baseline power values above the floor {cfg.power_floor}")
        ratio = np.full(pb.shape, np.nan)
        np.divide(pa, pb, out=ratio, where=valid)
        if ev.scenario == "gradual_degradation":
            expected = gradual_factors(ev.params["final_power_factor"], d)[None, :, None]
            err = float(np.nanmax(np.abs(ratio - expected)))
            step_median = np.nanmedian(ratio, axis=(0, 2))
            if np.any(np.diff(step_median) > cfg.realized_ratio_tol):
                raise EventError(f"{ev.event_id}: gradual trajectory is not monotonically non-increasing")
            out["realized_final_power_factor"] = float(np.nanmedian(ratio[:, -1, :]))
            target = ev.params["final_power_factor"]
            if abs(out["realized_final_power_factor"] - target) > cfg.realized_ratio_tol:
                raise EventError(f"{ev.event_id}: realized final factor {out['realized_final_power_factor']:.6f} "
                                 f"!= target {target:.6f}")
        else:
            target = ev.params["power_multiplier"]
            err = float(np.nanmax(np.abs(ratio - target)))
            out["realized_power_multiplier"] = float(np.nanmedian(ratio))
        if err > cfg.realized_ratio_tol:
            raise EventError(f"{ev.event_id}: realized power effect deviates from the target by {err:.3g} "
                             f"(tolerance {cfg.realized_ratio_tol})")
        near_zero = ~valid
        if near_zero.any():  # absolute criterion where ratios are unstable
            factor = gradual_factors(ev.params["final_power_factor"], d)[None, :, None] \
                if ev.scenario == "gradual_degradation" else ev.params["power_multiplier"]
            if np.max(np.abs(pa - pb * factor)[near_zero]) > cfg.power_floor:
                raise EventError(f"{ev.event_id}: near-zero power cells are not scaled as configured")
        out["power_max_abs_error"] = err
    if changes_temp:
        exceed_cols += t
        diff = a[:, :, t] - b[:, :, t]
        target = ev.params["temp_offset"]
        err = float(np.max(np.abs(diff - target)))
        if err > cfg.realized_offset_tol:
            raise EventError(f"{ev.event_id}: realized temperature offset deviates from {target} by {err:.3g}")
        out["realized_temp_offset"] = float(np.median(diff))
        out["temp_max_abs_error"] = err
    cols = sorted(set(exceed_cols))
    out["envelope_exceed_fraction"] = float((a[:, :, cols] > limit_max[cols]).mean())
    return out


def _string_column(values: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=object))


def with_label_columns(frame: pd.DataFrame, labels=None, types=None, ids=None) -> pd.DataFrame:
    """Set the three synthetic ground-truth columns (all-normal by default) with fixed dtypes."""
    n = len(frame)
    out = frame.copy()
    out[LABEL_COL] = np.zeros(n, dtype=np.int8) if labels is None else np.asarray(labels, dtype=np.int8)
    out[TYPE_COL] = _string_column(np.full(n, NORMAL, dtype=object) if types is None else types)
    out[EVENT_COL] = _string_column(np.full(n, "", dtype=object) if ids is None else ids)
    return out


def derive_limits(normal: pd.DataFrame, feat: FeatureInfo, cfg: GeneratorConfig) -> dict[str, dict[str, float]]:
    """Descriptive envelope from the normal training partition only. Never used for clipping."""
    arr = normal[feat.features].to_numpy(dtype=np.float64)
    lo_q, hi_q = np.quantile(arr, cfg.envelope_quantiles, axis=0)
    limits = {}
    for j, col in enumerate(feat.features):
        lo, hi = float(lo_q[j]), float(hi_q[j])
        span = max(hi - lo, abs(hi) * 0.05, 1e-6)
        limits[col] = {"min": lo - cfg.envelope_headroom * span, "max": hi + cfg.envelope_headroom * span}
    return limits


def inject_events(normal: pd.DataFrame, planned: list[PlannedEvent], feat: FeatureInfo, n_hosts: int,
                  time_col: str, host_names: list[str], limits: dict, cfg: GeneratorConfig,
                  ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return (frame with events injected, event records). `normal` is never modified."""
    n_steps = len(normal) // n_hosts
    dtype = np.dtype(cfg.feature_dtype)
    base = normal[feat.features].to_numpy(dtype=dtype, copy=True)
    work = base.astype(np.float64)
    labels = np.zeros(len(normal), dtype=np.int8)
    types = np.full(len(normal), NORMAL, dtype=object)
    ids = np.full(len(normal), "", dtype=object)
    placed: list[tuple[PlannedEvent, np.ndarray]] = []
    for ev in planned:
        if ev.start_step < 0 or ev.start_step + ev.duration > n_steps:
            raise EventError(f"{ev.event_id}: window [{ev.start_step}, {ev.start_step + ev.duration}) leaves the partition")
        rows = event_rows(ev, n_hosts)
        if labels[rows].any():
            raise EventError(f"{ev.event_id}: conflicts with an earlier event on the same host/timestamp")
        _apply_transform(work, rows, ev, feat)
        labels[rows], types[rows], ids[rows] = 1, ev.scenario, ev.event_id
        placed.append((ev, rows))
    final = work.astype(dtype)  # single rounding step, after all events

    limit_max = np.array([limits[c]["max"] for c in feat.features])
    times = normal[time_col].iloc[::n_hosts].reset_index(drop=True)
    records = []
    for ev, rows in placed:
        m = measure_event(ev, base[rows], final[rows], feat, cfg, limit_max)
        end_step = ev.start_step + ev.duration - 1
        t0, t1 = times.iloc[ev.start_step], times.iloc[end_step]
        records.append({
            "event_id": ev.event_id, "split": ev.split, "anomaly_type": ev.scenario,
            "severity_band": ev.band, "held_out": ev.held_out, "band_status": ev.band_status,
            "start_time": pd.Timestamp(t0).isoformat(), "end_time": pd.Timestamp(t1).isoformat(),
            "start_step": ev.start_step, "end_step": end_step, "duration_steps": ev.duration,
            "duration_seconds": ev.duration * cfg.sample_interval_seconds,
            "observed_span_seconds": float((t1 - t0).total_seconds()),
            "hosts": ";".join(host_names[h] for h in ev.hosts), "n_hosts": len(ev.hosts),
            "n_rows": int(len(rows)), "event_seed": ev.event_seed,
            "placement_policy": cfg.split_plans[ev.split].placement,
            "target_power_multiplier": ev.params.get("power_multiplier"),
            "target_final_power_factor": ev.params.get("final_power_factor"),
            "target_temp_offset": ev.params.get("temp_offset"),
            "realized_power_multiplier": m.get("realized_power_multiplier"),
            "realized_final_power_factor": m.get("realized_final_power_factor"),
            "realized_temp_offset": m.get("realized_temp_offset"),
            "power_max_abs_error": m.get("power_max_abs_error"),
            "temp_max_abs_error": m.get("temp_max_abs_error"),
            "envelope_exceed_fraction": m["envelope_exceed_fraction"],
            "status": "validated",
        })
    out = with_label_columns(normal, labels, types, ids)
    for j, col in enumerate(feat.features):
        out[col] = final[:, j]
    return out, records


# ----------------------------------------------------------------------------
# Build (pure, in memory)
# ----------------------------------------------------------------------------
@dataclass
class Build:
    cfg: GeneratorConfig
    source: pd.DataFrame
    audit: SourceAudit
    split_meta: dict[str, Any]
    pools: dict[str, tuple[int, ...]]
    plans: dict[str, list[PlannedEvent]]
    frames: dict[str, pd.DataFrame]       # train, supervised_train, validation, test
    bases: dict[str, pd.DataFrame]        # normal-labelled partitions: supervised_train, validation, test
    events: pd.DataFrame
    limits: dict[str, dict[str, float]]
    steps: dict[str, int]


def build_datasets(source_raw: pd.DataFrame, cfg: GeneratorConfig) -> Build:
    """Split, plan and inject. Takes the raw source frame (as read from Parquet); no file I/O."""
    validate_config(cfg)
    source = prepare_source(source_raw)
    audit = audit_source(source, cfg)
    H, T, tcol = audit.n_hosts, audit.n_steps, audit.time_col
    bounds = split_bounds(T, cfg)
    parts = {name: source.iloc[a * H:b * H].reset_index(drop=True) for name, (a, b) in bounds.items()}
    steps = {"supervised_train": bounds["train"][1] - bounds["train"][0],
             "validation": bounds["validation"][1] - bounds["validation"][0],
             "test": bounds["test"][1] - bounds["test"][0]}
    pools = allocate_host_pools(H, cfg.seed)
    plans = plan_all(cfg, steps, pools)

    normal = with_label_columns(parts["train"])
    limits = derive_limits(normal, audit.feat, cfg)
    bases = {"supervised_train": normal, "validation": with_label_columns(parts["validation"]),
             "test": with_label_columns(parts["test"])}
    frames = {"train": normal}
    records: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        frames[split], recs = inject_events(bases[split], plans[split], audit.feat, H, tcol,
                                            audit.host_names, limits, cfg)
        records += recs
    events = pd.DataFrame(records)
    times = source[tcol].iloc[::H].reset_index(drop=True)
    meta = {
        "n_timestamps": T, "purge_steps": cfg.purge_steps,
        "step_bounds": {k: [int(a), int(b)] for k, (a, b) in bounds.items()},
        "purged_step_ranges": [[bounds["train"][1], bounds["validation"][0]],
                               [bounds["validation"][1], bounds["test"][0]]],
        "purged_rows": int(len(source) - sum(len(p) for p in parts.values())),
        "boundaries": {
            "train_start": str(times.iloc[bounds["train"][0]]), "train_end": str(times.iloc[bounds["train"][1] - 1]),
            "validation_start": str(times.iloc[bounds["validation"][0]]),
            "validation_end": str(times.iloc[bounds["validation"][1] - 1]),
            "test_start": str(times.iloc[bounds["test"][0]]), "test_end": str(times.iloc[bounds["test"][1] - 1]),
        },
    }
    return Build(cfg, source, audit, meta, pools, plans, frames, bases, events, limits, steps)


# ----------------------------------------------------------------------------
# Validation gates. Any FAIL stops publication.
# ----------------------------------------------------------------------------
@dataclass
class CheckResult:
    id: str
    name: str
    status: str  # PASS | FAIL | NOT_APPLICABLE
    details: str
    measured: dict[str, Any] = field(default_factory=dict)


def _result(cid: str, name: str, problems: list[str], summary: str, measured=None, na: str | None = None) -> CheckResult:
    if na is not None:
        return CheckResult(cid, name, "NOT_APPLICABLE", na, {})
    if problems:
        more = f" (+{len(problems) - 8} more)" if len(problems) > 8 else ""
        return CheckResult(cid, name, "FAIL", "; ".join(problems[:8]) + more, measured or {})
    return CheckResult(cid, name, "PASS", summary, measured or {})


def _feat_array(frame: pd.DataFrame, feat: FeatureInfo) -> np.ndarray:
    return frame[feat.features].to_numpy()


def check_source_audit(b: Build) -> CheckResult:
    f = b.audit.facts
    return _result("C01", "source_audit", [], "complete panel, regular interval, normal-only labels, finite uniform-dtype "
                   f"features; {f['preexisting_summary_violations']} pre-existing temperature-summary violations recorded",
                   f)


def check_source_unchanged(before: str | None, after: str | None) -> CheckResult:
    if before is None or after is None:
        return _result("C02", "source_file_unchanged", [], "", na="source hashes not supplied (in-memory build)")
    return _result("C02", "source_file_unchanged", [] if before == after else ["source SHA-256 changed during generation"],
                   "SHA-256 identical before and after", {"sha256": after})


def check_splits(b: Build) -> CheckResult:
    H, P, tcol = b.audit.n_hosts, b.cfg.purge_steps, b.audit.time_col
    src_times = b.source[tcol].iloc[::H].reset_index(drop=True)
    bounds = b.split_meta["step_bounds"]
    problems: list[str] = []
    ranges = {}
    for name, key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        fr = b.frames[name]
        a, e = bounds[key]
        if len(fr) != (e - a) * H:
            problems.append(f"{name}: {len(fr)} rows, expected {(e - a) * H}")
        t = fr[tcol].iloc[::H].reset_index(drop=True)
        if not t.equals(src_times.iloc[a:e].reset_index(drop=True)):
            problems.append(f"{name}: timestamps differ from the source slice [{a},{e})")
        ranges[name] = (t.iloc[0], t.iloc[-1])
        if not (t.diff().dropna() > pd.Timedelta(0)).all():
            problems.append(f"{name}: timestamps are not strictly increasing")
    for x, y in (("train", "validation"), ("validation", "test")):
        gap = int(((src_times > ranges[x][1]) & (src_times < ranges[y][0])).sum())
        if gap != P:
            problems.append(f"purge gap between {x} and {y} is {gap} timestamps, expected {P}")
    if not (ranges["train"][1] < ranges["validation"][0] and ranges["validation"][1] < ranges["test"][0]):
        problems.append("partitions are not chronologically ordered")
    ids = np.concatenate([b.frames[n][SOURCE_ROW_COL].to_numpy() for n in ("train", "validation", "test")])
    if len(np.unique(ids)) != len(ids):
        problems.append("partitions share source rows")
    if b.split_meta["purged_rows"] != 2 * P * H:
        problems.append(f"purged rows {b.split_meta['purged_rows']} != {2 * P * H}")
    for name in ("supervised_train",):
        if not b.frames[name][tcol].equals(b.frames["train"][tcol]):
            problems.append("supervised_train timestamps differ from normal train")
    return _result("C03", "split_boundaries_and_purge", problems,
                   f"3 chronological partitions, {P}-timestamp purge gaps, no shared rows or timestamps",
                   {"steps": b.steps, "purged_rows": b.split_meta["purged_rows"]})


def check_host_pools(b: Build) -> CheckResult:
    problems: list[str] = []
    pools = {k: set(v) for k, v in b.pools.items()}
    names = {n: set(b.audit.host_names[h] for h in pools[n]) for n in pools}
    for x in SPLIT_NAMES:
        for y in SPLIT_NAMES:
            if x < y and pools[x] & pools[y]:
                problems.append(f"host pools {x} and {y} overlap")
    used: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        ev = b.events[b.events["split"] == split]
        used[split] = {h for s in ev["hosts"] for h in s.split(";")}
        if not used[split] <= names[split]:
            problems.append(f"{split}: events use hosts outside the {split} pool")
    return _result("C04", "host_pool_separation", problems,
                   "anomaly hosts are disjoint across supervised_train / validation / test (every host still "
                   "appears as normal data in every partition)",
                   {"pool_sizes": {k: len(v) for k, v in b.pools.items()},
                    "hosts_used": {k: len(v) for k, v in used.items()}})


def _source_lookup(b: Build) -> pd.DataFrame:
    return b.source.set_index(SOURCE_ROW_COL)


def check_normal_train(b: Build) -> CheckResult:
    fr, problems = b.frames["train"], []
    if (fr[LABEL_COL] != 0).any() or (fr[TYPE_COL] != NORMAL).any() or (fr[EVENT_COL] != "").any():
        problems.append("normal-only training data contains labels, anomaly types or event IDs")
    ref = _source_lookup(b).loc[fr[SOURCE_ROW_COL].to_numpy()].reset_index(drop=True)
    for col in b.source.columns:
        if col in (SOURCE_ROW_COL, LABEL_COL):
            continue
        if not ref[col].equals(fr[col].reset_index(drop=True)):
            problems.append(f"column {col} differs from the source rows")
    return _result("C05", "normal_train_preserved", problems,
                   f"{len(fr):,} rows, all labels 0, every source column identical to the source rows")


def check_supervised_copy(b: Build) -> CheckResult:
    normal, sup, problems = b.frames["train"], b.frames["supervised_train"], []
    if list(normal.columns) != list(sup.columns):
        problems.append("column names/order differ from normal train")
    for col in (SOURCE_ROW_COL, b.audit.host_col, b.audit.time_col):
        if not normal[col].equals(sup[col]):
            problems.append(f"{col} differs from normal train (rows not identical in identity/order)")
    lab = sup[LABEL_COL].to_numpy() == 1
    feat = b.audit.feat
    other = [c for c in normal.columns if c not in feat.features and c not in (LABEL_COL, TYPE_COL, EVENT_COL)]
    for col in other:
        if not normal[col].equals(sup[col]):
            problems.append(f"non-feature column {col} differs from normal train")
    fa, fb = _feat_array(normal, feat), _feat_array(sup, feat)
    changed = (fa != fb).any(axis=1)
    if changed[~lab].any():
        problems.append(f"{int(changed[~lab].sum())} unlabelled rows were modified")
    if not changed[lab].all():
        problems.append(f"{int((~changed[lab]).sum())} labelled rows are unchanged")
    if (b.frames["train"][LABEL_COL] != 0).any():
        problems.append("normal train was mutated")
    return _result("C06", "supervised_copy_integrity", problems,
                   f"same {len(sup):,} rows/order as normal train; only the {int(lab.sum()):,} labelled rows differ")


def check_labels(b: Build) -> CheckResult:
    problems = []
    valid_types = set(SCENARIO_NAMES) | {NORMAL}
    for name in SPLIT_NAMES:
        fr = b.frames[name]
        lab = fr[LABEL_COL].to_numpy()
        if not set(np.unique(lab)) <= {0, 1}:
            problems.append(f"{name}: label values outside {{0,1}}")
        if ((lab == 1) != (fr[EVENT_COL].to_numpy() != "")).any():
            problems.append(f"{name}: label disagrees with event_id")
        if ((lab == 1) != (fr[TYPE_COL].to_numpy() != NORMAL)).any():
            problems.append(f"{name}: label disagrees with anomaly_type")
        if not set(fr[TYPE_COL].unique()) <= valid_types:
            problems.append(f"{name}: unknown anomaly types")
        per_event = fr.loc[lab == 1].groupby(EVENT_COL)[TYPE_COL].nunique()
        if (per_event > 1).any():
            problems.append(f"{name}: an event_id carries several anomaly types")
        if not lab.any():
            problems.append(f"{name}: no labelled rows")
    return _result("C07", "label_consistency", problems, "label, anomaly_type and event_id agree in all three labelled datasets",
                   {n: int((b.frames[n][LABEL_COL] == 1).sum()) for n in SPLIT_NAMES})


def check_manifest(b: Build) -> CheckResult:
    problems, H = [], b.audit.n_hosts
    ev_all = b.events
    if ev_all["event_id"].duplicated().any():
        problems.append("duplicate event IDs in the manifest")
    index = {h: i for i, h in enumerate(b.audit.host_names)}
    for split in SPLIT_NAMES:
        fr, ev = b.frames[split], ev_all[ev_all["split"] == split]
        lab_pos = np.flatnonzero(fr[LABEL_COL].to_numpy() == 1)  # positions in the full dataset
        groups = {k: lab_pos[v] for k, v in fr.loc[fr[LABEL_COL] == 1].groupby(EVENT_COL).indices.items()}
        if set(groups) != set(ev["event_id"]):
            problems.append(f"{split}: event IDs in the data and manifest differ")
        for r in ev.itertuples(index=False):
            hosts = [index[h] for h in r.hosts.split(";")]
            steps = np.arange(r.start_step, r.end_step + 1)
            expected = np.sort((steps[None, :] * H + np.asarray(hosts)[:, None]).ravel())
            actual = np.sort(np.asarray(groups.get(r.event_id, []), dtype=int))
            if not np.array_equal(expected, actual):
                problems.append(f"{r.event_id}: labelled rows differ from hosts x window in the manifest")
                continue
            sub = fr.iloc[actual]
            if (sub[TYPE_COL] != r.anomaly_type).any():
                problems.append(f"{r.event_id}: anomaly_type differs from the manifest")
            if r.n_rows != len(actual) or r.duration_steps != len(steps):
                problems.append(f"{r.event_id}: row count or duration differs from the manifest")
            if pd.Timestamp(sub[b.audit.time_col].min()).isoformat() != r.start_time or \
               pd.Timestamp(sub[b.audit.time_col].max()).isoformat() != r.end_time:
                problems.append(f"{r.event_id}: start/end time differ from the data")
        if int(ev["n_rows"].sum()) != int((fr[LABEL_COL] == 1).sum()):
            problems.append(f"{split}: manifest row total != labelled rows")
    return _result("C08", "event_manifest_matches_data", problems,
                   f"{len(ev_all)} events: labelled rows equal hosts x window for every event",
                   {"events": int(len(ev_all))})


def check_params_and_placement(b: Build) -> CheckResult:
    cfg, problems = b.cfg, []
    m, g = cfg.edge_margin_steps, cfg.guard_steps
    for split in SPLIT_NAMES:
        plan, ev = cfg.split_plans[split], b.events[b.events["split"] == split]
        n_steps = b.steps[split]
        for sname in SCENARIO_NAMES:
            spec = cfg.scenarios[sname]
            for band in spec.bands:
                sub = ev[(ev["anomaly_type"] == sname) & (ev["severity_band"] == band.name)]
                want = plan.event_counts[sname].get(band.name, 0)
                if len(sub) != want:
                    problems.append(f"{split}/{sname}/{band.name}: {len(sub)} events, configured {want}")
                for r in sub.itertuples(index=False):
                    if not spec.duration_steps[0] <= r.duration_steps <= spec.duration_steps[1]:
                        problems.append(f"{r.event_id}: duration {r.duration_steps} outside {spec.duration_steps}")
                    if not spec.hosts[0] <= r.n_hosts <= spec.hosts[1]:
                        problems.append(f"{r.event_id}: {r.n_hosts} hosts outside {spec.hosts}")
                    if bool(r.held_out) != band.held_out or r.band_status != band.status:
                        problems.append(f"{r.event_id}: held_out/status differ from band '{band.name}'")
                    for dim, rng in band.ranges.items():
                        v = getattr(r, f"target_{dim}")
                        if v is None or pd.isna(v) or not rng[0] - 1e-12 <= v <= rng[1] + 1e-12:
                            problems.append(f"{r.event_id}: target {dim}={v} outside band range {rng}")
                    if r.start_step < m or r.end_step > n_steps - 1 - m:
                        problems.append(f"{r.event_id}: violates the {m}-step edge margin")
        occupied: dict[Any, list[tuple[int, int, str]]] = {}
        for r in ev.itertuples(index=False):
            keys = ["*"] if plan.placement == "global" else r.hosts.split(";")
            for k in keys:
                occupied.setdefault(k, []).append((r.start_step, r.end_step, r.event_id))
        for k, spans in occupied.items():
            spans.sort()
            for (s0, e0, i0), (s1, e1, i1) in zip(spans, spans[1:]):
                if s1 - e0 - 1 < g:
                    problems.append(f"{split}: {i0} and {i1} are closer than the {g}-step guard "
                                    f"({'any host' if k == '*' else k})")
    return _result("C09", "event_parameters_and_placement", problems,
                   "counts, durations, host counts, band ranges, edge margins and overlap/guard rules all respected",
                   {"policies": {s: cfg.split_plans[s].placement for s in SPLIT_NAMES}})


def check_realized_effects(b: Build) -> CheckResult:
    problems, feat, cfg, H = [], b.audit.feat, b.cfg, b.audit.n_hosts
    limit_max = np.array([b.limits[c]["max"] for c in feat.features])
    for split in SPLIT_NAMES:
        fr, base = b.frames[split], b.bases[split]
        fa, fb = _feat_array(fr, feat), _feat_array(base, feat)
        changed = (fa != fb).any(axis=1)
        lab = fr[LABEL_COL].to_numpy() == 1
        if changed[~lab].any():
            problems.append(f"{split}: {int(changed[~lab].sum())} unlabelled rows differ from baseline")
        ev = b.events[b.events["split"] == split].set_index("event_id")
        for pe in b.plans[split]:
            rows = event_rows(pe, H)
            try:
                m = measure_event(pe, fb[rows], fa[rows], feat, cfg, limit_max)
            except EventError as exc:
                problems.append(str(exc))
                continue
            for key, value in m.items():
                recorded = ev.loc[pe.event_id, key]
                if pd.isna(recorded) or abs(float(recorded) - value) > 1e-9:
                    problems.append(f"{pe.event_id}: manifest {key}={recorded} != re-measured {value}")
    return _result("C10", "realized_effects", problems,
                   "every event re-measured on the final frames: correct direction/magnitude, unrelated features and "
                   "unlabelled rows unchanged, every labelled row changed, manifest values match")


def check_schema(b: Build) -> CheckResult:
    problems, feat = [], b.audit.feat
    ref = b.frames["train"]
    for name, fr in b.frames.items():
        if list(fr.columns) != list(ref.columns):
            problems.append(f"{name}: columns differ from train")
        if not fr.dtypes.equals(ref.dtypes):
            diff = [c for c in ref.columns if c in fr.columns and fr[c].dtype != ref[c].dtype]
            problems.append(f"{name}: dtypes differ from train in {diff}")
        arr = fr[feat.features].to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            problems.append(f"{name}: non-finite feature values")
        if {str(fr[c].dtype) for c in feat.features} != {b.cfg.feature_dtype}:
            problems.append(f"{name}: features are not all {b.cfg.feature_dtype}")
        if fr[[LABEL_COL, TYPE_COL, EVENT_COL]].isna().any().any():
            problems.append(f"{name}: missing values in label columns")
    return _result("C11", "schema_dtype_finite", problems,
                   f"identical columns/dtypes in all 4 datasets; {len(feat.features)} finite {b.cfg.feature_dtype} features")


def check_temp_summary(b: Build) -> CheckResult:
    feat, problems, measured = b.audit.feat, [], {}
    lookup = _source_lookup(b)
    for name, fr in b.frames.items():
        ref = lookup.loc[fr[SOURCE_ROW_COL].to_numpy()]
        out_bad = summary_violations(fr, feat.summary_groups)
        src_bad = summary_violations(ref, feat.summary_groups)
        measured[name] = {"violations": int(out_bad.sum()), "in_source_rows": int(src_bad.sum())}
        if (out_bad & ~src_bad).any():
            problems.append(f"{name}: {int((out_bad & ~src_bad).sum())} new temperature-summary violations")
    na = None if feat.summary_groups else "no min/mean/max temperature summary columns in the source"
    return _result("C12", "temperature_summary_no_new_violations", problems,
                   "no new min<=mean<=max violations (pre-existing source violations are recorded, not altered)",
                   measured, na=na)


def check_held_out(b: Build) -> CheckResult:
    cfg, problems = b.cfg, []
    if cfg.allow_band_overlap:
        return _result("C13", "held_out_separation", [], "", na="allow_band_overlap=True: overlap explicitly permitted")
    ev = b.events
    for split in ("supervised_train", "validation"):
        if ev[(ev["split"] == split) & ev["held_out"]].shape[0]:
            problems.append(f"{split} contains held-out events")
    for r in ev.itertuples(index=False):
        spec = cfg.scenarios[r.anomaly_type]
        for hb in (x for x in spec.bands if x.held_out):
            in_held = all(hb.ranges[d][0] - 1e-12 <= getattr(r, f"target_{d}") <= hb.ranges[d][1] + 1e-12 for d in hb.ranges)
            if in_held and not r.held_out:
                problems.append(f"{r.event_id}: non-held-out event has parameters inside held-out band '{hb.name}'")
    held = ev[ev["held_out"]]
    return _result("C13", "held_out_separation", problems,
                   "held-out events appear only in test; no other event has parameters inside a held-out range",
                   {"held_out_events": int(len(held)), "held_out_rows": int(held["n_rows"].sum())})


def check_plan_reproducibility(b: Build) -> CheckResult:
    again = plan_all(b.cfg, b.steps, allocate_host_pools(b.audit.n_hosts, b.cfg.seed))
    same = again == b.plans
    return _result("C14", "plan_reproducibility", [] if same else ["re-deriving the event plan from the seed gave a different plan"],
                   "event plan re-derived from the seed is identical (full-content determinism is covered by the test suite)")


def check_capacity(b: Build) -> CheckResult:
    measured = {}
    for split in SPLIT_NAMES:
        usable = b.steps[split] - 2 * b.cfg.edge_margin_steps
        measured[split] = {"usable_steps": usable, "events": len(b.plans[split]),
                           "event_steps": int(sum(e.duration for e in b.plans[split])),
                           "pool_hosts": len(b.pools[split])}
    return _result("C15", "capacity_feasibility", [], "all configured events were placed without shortening or dropping", measured)


def check_protected(before: dict[str, str] | None, after: dict[str, str] | None) -> CheckResult:
    if not before:
        return _result("C16", "protected_files_unchanged", [], "", na="no --protect paths supplied")
    problems = [f"{p} changed" for p in before if before[p] != (after or {}).get(p)]
    return _result("C16", "protected_files_unchanged", problems, f"{len(before)} protected path(s) unchanged")


def check_readback(b: Build, staged: Path | None) -> CheckResult:
    if staged is None:
        return _result("C17", "artifacts_readback", [], "", na="no staged files (in-memory build)")
    problems = []
    for name, fname in DATASET_FILES.items():
        back = pd.read_parquet(staged / fname)
        if frame_digest(back) != frame_digest(b.frames[name]):
            problems.append(f"{fname}: content or dtypes differ after write/read")
    ev_back = pd.read_csv(staged / EVENTS_FILE)
    ev_sup = pd.read_csv(staged / SUPERVISED_EVENTS_FILE)
    if len(ev_back) != len(b.events) or list(ev_back["event_id"]) != list(b.events["event_id"]):
        problems.append(f"{EVENTS_FILE} does not match the in-memory manifest")
    if list(ev_sup["event_id"]) != list(b.events.loc[b.events["split"] == "supervised_train", "event_id"]):
        problems.append(f"{SUPERVISED_EVENTS_FILE} does not match the supervised-train events")
    for col in ("start_step", "duration_steps", "n_rows", "event_seed"):
        if not np.array_equal(ev_back[col].to_numpy(), b.events[col].to_numpy()):
            problems.append(f"{EVENTS_FILE}: column {col} differs after write/read")
    return _result("C17", "artifacts_readback", problems, "parquet files and event manifests re-read identical to memory")


def run_checks(b: Build, source_hashes: tuple[str, str] | None = None, protected: tuple[dict, dict] | None = None,
               staged: Path | None = None) -> list[CheckResult]:
    return [
        check_source_audit(b),
        check_source_unchanged(*(source_hashes or (None, None))),
        check_splits(b), check_host_pools(b), check_normal_train(b), check_supervised_copy(b),
        check_labels(b), check_manifest(b), check_params_and_placement(b), check_realized_effects(b),
        check_schema(b), check_temp_summary(b), check_held_out(b), check_plan_reproducibility(b),
        check_capacity(b), check_protected(*(protected or (None, None))), check_readback(b, staged),
    ]


# ----------------------------------------------------------------------------
# Reports and artifacts
# ----------------------------------------------------------------------------
ASSUMPTIONS = [
    "Baseline severity ranges are the existing generator's configuration, not measured fault statistics.",
    "Held-out severity bands are PROPOSED and not validated; stronger anomalies test extrapolation to more severe "
    "faults, not generalization to unseen anomaly categories (and are likely easier to detect).",
    "One multiplier is applied to all power features and one offset to all temperature features (including "
    "min/mean/max summaries, which therefore keep their ordering). Real hardware may not behave this way.",
    "Step durations are converted to time with the audited 10 s sampling interval.",
    "Guard and edge margins (30 steps) are configurable design choices, chosen to match the 30-step sequence window.",
    "The training-derived envelope is descriptive; events are never clipped to it.",
]
LIMITATIONS = [
    "A passing run certifies data generation only; it does not show that synthetic anomalies resemble real faults.",
    "It does not show model performance, generalization, or that a validation false-positive-rate target is met.",
    "Host separation covers where anomalies are injected; every host appears as normal data in every partition.",
    "Rows inside an event are strongly autocorrelated; evaluate at event level and do not treat rows as independent.",
]


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _spread(values: pd.Series) -> dict[str, float]:
    v = values.dropna()
    return {"min": float(v.min()), "median": float(v.median()), "max": float(v.max())} if len(v) else {}


def build_report(b: Build, checks: list[CheckResult], hashes: dict[str, Any]) -> dict[str, Any]:
    ev, feat, cfg = b.events, b.audit.feat, b.cfg
    realized: dict[str, Any] = {}
    for (scenario, band), g in ev.groupby(["anomaly_type", "severity_band"]):
        realized.setdefault(scenario, {})[band] = {
            "events": int(len(g)), "rows": int(g["n_rows"].sum()),
            **{c: _spread(g[c]) for c in ("target_power_multiplier", "realized_power_multiplier",
                                          "target_final_power_factor", "realized_final_power_factor",
                                          "target_temp_offset", "realized_temp_offset",
                                          "envelope_exceed_fraction", "duration_steps") if g[c].notna().any()},
        }
    splits = {}
    for name, fr in b.frames.items():
        splits[name] = {
            "rows": int(len(fr)), "steps": int(len(fr) // b.audit.n_hosts), "hosts": int(fr[b.audit.host_col].nunique()),
            "time_start": str(fr[b.audit.time_col].min()), "time_end": str(fr[b.audit.time_col].max()),
            "label_counts": {str(k): int(v) for k, v in fr[LABEL_COL].value_counts().sort_index().items()},
            "anomaly_type_counts": {str(k): int(v) for k, v in fr[TYPE_COL].value_counts().items()},
        }
    counts: dict[str, Any] = {}
    for (split, scenario, band), g in ev.groupby(["split", "anomaly_type", "severity_band"]):
        counts.setdefault(split, {}).setdefault(scenario, {})[band] = int(len(g))
    failed = [c.id for c in checks if c.status == "FAIL"]
    return {
        "title": "SENTINEL 2.0 V3 synthetic dataset report",
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),  # the only non-deterministic field
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "scope": "Generator validation only. Not a model evaluation; not evidence of real-world realism.",
        "seed": cfg.seed,
        "seed_derivation": "np.random.SeedSequence(entropy=seed, spawn_key=(stream_id,)) with stream ids "
                           f"{STREAM_IDS}; per-event seed = SeedSequence(entropy=seed, spawn_key=(stream_id, 1 + draft_index)).",
        "sampling_interval_seconds": cfg.sample_interval_seconds,
        "source": {"path": hashes.get("source_path"), "sha256_before": hashes.get("source_before"),
                   "sha256_after": hashes.get("source_after"),
                   "unchanged": hashes.get("source_before") == hashes.get("source_after") if hashes.get("source_before") else None,
                   "audit": b.audit.facts},
        "split": {**b.split_meta, "fractions": {"train": cfg.train_frac, "validation": cfg.val_frac,
                                                "test": round(1 - cfg.train_frac - cfg.val_frac, 10)}},
        "host_pools": {k: {"hosts": len(v), "names": [b.audit.host_names[i] for i in v]} for k, v in b.pools.items()},
        "host_separation": "anomaly hosts disjoint across supervised_train, validation and test (check C04)",
        "features": {"count": len(feat.features), "dtype": cfg.feature_dtype, "power": feat.power,
                     "temperature": feat.temperature, "summary_groups": feat.summary_groups},
        "datasets": splits,
        "event_counts": counts,
        "severity_bands_used": {s: {sc: {b_.name: {"held_out": b_.held_out, "status": b_.status, "ranges": b_.ranges}
                                         for b_ in cfg.scenarios[sc].bands
                                         if cfg.split_plans[s].event_counts[sc].get(b_.name, 0) > 0}
                                    for sc in SCENARIO_NAMES} for s in SPLIT_NAMES},
        "realized_effects": realized,
        "envelope_descriptive": b.limits,
        "validation": {"overall": "PASS" if not failed else "FAIL",
                       "checks": [dataclasses.asdict(c) for c in checks]},
        "content_digests": hashes.get("digests", {}),
        "assumptions": ASSUMPTIONS, "limitations": LIMITATIONS,
    }


def render_text_report(r: dict[str, Any]) -> str:
    L = [r["title"], "=" * 60, f"Generator version: {r['generator_version']}  |  seed: {r['seed']}",
         f"Scope: {r['scope']}", "",
         f"Source: {r['source']['path']}", f"  SHA-256 before: {r['source']['sha256_before']}",
         f"  SHA-256 after : {r['source']['sha256_after']}", f"  unchanged: {r['source']['unchanged']}",
         f"  rows {r['source']['audit']['rows']:,} | hosts {r['source']['audit']['hosts']} | timestamps "
         f"{r['source']['audit']['timestamps']:,} | interval {r['sampling_interval_seconds']} s",
         f"  pre-existing temperature-summary violations: {r['source']['audit']['preexisting_summary_violations']}", "",
         f"Split boundaries (purge {r['split']['purge_steps']} timestamps each; purged rows {r['split']['purged_rows']:,}):"]
    L += [f"  {k}: {v}" for k, v in r["split"]["boundaries"].items()]
    L += ["", f"Features: {r['features']['count']} ({r['features']['dtype']}) - "
              f"{len(r['features']['power'])} power, {len(r['features']['temperature'])} temperature", "",
          "Datasets:"]
    for name, d in r["datasets"].items():
        L.append(f"  {name}: {d['rows']:,} rows, labels {d['label_counts']}, types {d['anomaly_type_counts']}")
    L += ["", "Anomaly-host pools: " + ", ".join(f"{k}={v['hosts']}" for k, v in r["host_pools"].items()),
          "", "Event counts (split / scenario / band):"]
    for split, sc in r["event_counts"].items():
        for scenario, bands in sc.items():
            L.append(f"  {split:17s} {scenario:20s} {bands}")
    L += ["", "Realized effects (target vs realized, by scenario and band):"]
    for scenario, bands in r["realized_effects"].items():
        for band, d in bands.items():
            L.append(f"  {scenario} / {band}: events={d['events']} rows={d['rows']}")
            for key, val in d.items():
                if isinstance(val, dict):
                    L.append(f"      {key}: min {val['min']:.4g}, median {val['median']:.4g}, max {val['max']:.4g}")
    L += ["", f"Validation checks: {r['validation']['overall']}"]
    L += [f"  [{c['status']:14s}] {c['id']} {c['name']}: {c['details']}" for c in r["validation"]["checks"]]
    L += ["", "Assumptions:"] + [f"  - {a}" for a in r["assumptions"]]
    L += ["", "Limitations:"] + [f"  - {a}" for a in r["limitations"]]
    return "\n".join(L) + "\n"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def write_artifacts(b: Build, staged: Path, hashes: dict[str, Any], checks_fn) -> list[CheckResult]:
    """Write every artifact into `staged`; data files first, then run checks, then the reports."""
    for name, fname in DATASET_FILES.items():
        b.frames[name].to_parquet(staged / fname, index=False)
    b.events.to_csv(staged / EVENTS_FILE, index=False)
    b.events[b.events["split"] == "supervised_train"].to_csv(staged / SUPERVISED_EVENTS_FILE, index=False)
    feat = b.audit.feat
    _write_json(staged / "feature_manifest.json", {
        "generator_version": GENERATOR_VERSION, "model_feature_columns": feat.features, "dtype": b.cfg.feature_dtype,
        "power_features": feat.power, "temperature_features": feat.temperature,
        "temperature_summary_groups": feat.summary_groups,
        "excluded_columns": [c for c in b.frames["train"].columns if c not in feat.features],
        "note": "Only these columns are model inputs. Labels, event IDs, identifiers, timestamps and every other column "
                "must be excluded.",
    })
    _write_json(staged / "generation_config.json", config_to_dict(b.cfg))
    hashes["digests"] = {**{n: frame_digest(f) for n, f in b.frames.items()},
                         "events_csv_sha256": hashlib.sha256(b.events.to_csv(index=False).encode()).hexdigest()}
    checks = checks_fn(staged)
    report = build_report(b, checks, hashes)
    _write_json(staged / "validation_results.json", report["validation"])
    _write_json(staged / "synthetic_report.json", report)
    (staged / "synthetic_report.txt").write_text(render_text_report(report), encoding="utf-8")
    return checks


def _check_output_target(output_dir: Path, source_path: Path, protect: list[Path]) -> None:
    if output_dir == source_path.parent or source_path.is_relative_to(output_dir) or output_dir.is_relative_to(source_path.parent):
        raise OutputExistsError(f"Output directory {output_dir} overlaps the source dataset location.")
    for p in protect:
        if output_dir == p or output_dir.is_relative_to(p) or p.is_relative_to(output_dir):
            raise OutputExistsError(f"Output directory {output_dir} overlaps protected path {p}.")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise OutputExistsError(f"Output path exists and is not a directory: {output_dir}")
        existing = sorted(p.name for p in output_dir.iterdir())
        if existing:
            raise OutputExistsError(f"Refusing to use non-empty output directory {output_dir} "
                                    f"(contains: {', '.join(existing[:10])}). V3 never overwrites existing data.")


def run(cfg: GeneratorConfig, source_path: Path, output_dir: Path, protect: list[Path] | None = None,
        verbose: bool = True) -> dict[str, Any]:
    """Generate V3 into a new directory. Returns the report dict. Raises GeneratorError on any failure."""
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    protect = [Path(p).resolve() for p in (protect or [])]
    log = print if verbose else (lambda *a, **k: None)
    validate_config(cfg)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source Parquet file not found: {source_path}")
    _check_output_target(output_dir, source_path, protect)

    hash_before = sha256_file(source_path)
    protected_before = {str(p): hash_path(p) for p in protect}
    log(f"Source: {source_path}\nLoading (read-only) and building datasets...")
    build = build_datasets(pd.read_parquet(source_path), cfg)
    hash_after = sha256_file(source_path)
    protected_after = {str(p): hash_path(p) for p in protect}
    hashes: dict[str, Any] = {"source_path": str(source_path), "source_before": hash_before, "source_after": hash_after}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex[:8]}"
    staged.mkdir()
    try:
        checks = write_artifacts(
            build, staged, hashes,
            lambda s: run_checks(build, (hash_before, hash_after), (protected_before, protected_after), staged=s))
        failed = [c for c in checks if c.status == "FAIL"]
        if failed:
            raise ValidationFailed("Critical validation checks failed:\n" +
                                   "\n".join(f"- {c.id} {c.name}: {c.details}" for c in failed))
        missing = [f for f in OUTPUT_FILES if not (staged / f).is_file()]
        if missing:
            raise ValidationFailed(f"Artifacts missing from staging: {missing}")
        for f in OUTPUT_FILES:
            if f.endswith(".json"):
                json.loads((staged / f).read_text(encoding="utf-8"))
        _check_output_target(output_dir, source_path, protect)  # re-check just before publishing
        if output_dir.exists():
            output_dir.rmdir()  # empty directory only (checked above)
        staged.rename(output_dir)
    except BaseException as exc:
        failed_dir = output_dir.parent / f"{output_dir.name}_FAILED_{uuid.uuid4().hex[:6]}"
        if staged.exists():
            (staged / "FAILED.txt").write_text(f"V3 generation failed; nothing was published.\n{exc}\n", encoding="utf-8")
            staged.rename(failed_dir)
            log(f"Generation failed; partial output kept for inspection at {failed_dir} (not a valid V3 output).")
        raise
    report = json.loads((output_dir / "synthetic_report.json").read_text(encoding="utf-8"))
    if verbose:
        for c in checks:
            print(f"[{c.status:14s}] {c.id} {c.name}")
        print(f"\nPublished V3 output: {output_dir}")
        print("Generator checks passed. This is NOT a model evaluation and does not mean V3 is frozen until you review the report.")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate SENTINEL 2.0 V3 synthetic telemetry (new directory only).")
    ap.add_argument("--source", required=True, type=Path, help="Original source Parquet (read-only).")
    ap.add_argument("--output-dir", required=True, type=Path, help="New output directory; must not exist or be empty.")
    ap.add_argument("--seed", type=int, default=None, help="Override the seed in the configuration.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional generation_config.json to reuse (e.g. from a previous run).")
    ap.add_argument("--protect", type=Path, action="append", default=[],
                    help="File or directory (e.g. original generator, V2 outputs) to hash before/after; repeatable.")
    args = ap.parse_args(argv)
    try:
        cfg = config_from_dict(json.loads(args.config.read_text(encoding="utf-8"))) if args.config else default_config()
        if args.seed is not None:
            cfg = dataclasses.replace(cfg, seed=args.seed)
        run(cfg, args.source, args.output_dir, args.protect)
    except GeneratorError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
