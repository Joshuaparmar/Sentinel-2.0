"""Tests for generate_synthetic_v3.py.

All tests use small temporary mock datasets shaped like the audited source (complete
host x timestamp panel, 10 s interval, float32 features, all-zero source labels).
No production-sized data is generated and the real source file is never touched.

Run from the project root:
    python -m pytest src\\sentinel_generator_package\\sentinel_generator_package\\tests\\test_generate_synthetic_v3.py -q
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sentinel"))
import generate_synthetic_v3 as gs  # noqa: E402

N_HOSTS, N_STEPS = 12, 1500

POWER = [f"p{s}_gpu{g}_power" for s in (0, 1) for g in range(3)] + ["p0_power", "p1_power",
                                                                     "ps0_input_power", "ps1_input_power"]
TEMPS = [f"gpu{i}_{k}_temp" for i in range(6) for k in ("core", "mem")]
SUMMARY = [f"p{s}_temp_{k}" for s in (0, 1) for k in ("mean", "min", "max")]


def make_source(n_hosts=N_HOSTS, n_steps=N_STEPS, seed=0, interval="10s", dtype="float32") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n_steps, freq=interval)
    hosts = [f"h{i:03d}" for i in range(n_hosts)]
    df = pd.MultiIndex.from_product([times, hosts], names=["timestamp", "hostname"]).to_frame(index=False)
    n = len(df)
    for c in POWER:
        df[c] = (80 + 170 * rng.beta(2, 5, n)).astype(dtype)
    for c in TEMPS:
        df[c] = rng.uniform(30, 62, n).astype(dtype)
    for p in ("p0", "p1"):
        mean = rng.uniform(35, 55, n)
        df[f"{p}_temp_mean"] = mean.astype(dtype)
        df[f"{p}_temp_min"] = (mean - rng.uniform(1, 4, n)).astype(dtype)
        df[f"{p}_temp_max"] = (mean + rng.uniform(1, 4, n)).astype(dtype)
    df["label"] = np.zeros(n, dtype="int64")
    df["xid"] = np.zeros(n, dtype="int64")
    df["job_id"] = pd.array(rng.integers(0, 50, n), dtype="Int64")
    df.loc[::7, "job_id"] = pd.NA
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffled file order


def small_plans(per_train=3, per_val=2, test_base=1, test_held=1):
    def c(**k):
        return {s: dict(k) for s in gs.SCENARIO_NAMES}
    return {"supervised_train": gs.SplitPlan("per_host", c(baseline=per_train)),
            "validation": gs.SplitPlan("global", c(baseline=per_val)),
            "test": gs.SplitPlan("global", c(baseline=test_base, held_out=test_held))}


def small_cfg(**kw) -> gs.GeneratorConfig:
    base = dict(purge_steps=5, edge_margin_steps=5, guard_steps=5, split_plans=small_plans())
    base.update(kw)
    return gs.default_config(**base)


@pytest.fixture(scope="module")
def source():
    return make_source()


@pytest.fixture(scope="module")
def build(source):
    return gs.build_datasets(source, small_cfg())


def set_band(cfg, scenario, band, **ranges):
    spec = cfg.scenarios[scenario]
    bands = tuple(dataclasses.replace(b, ranges={**b.ranges, **ranges}) if b.name == band else b for b in spec.bands)
    return dataclasses.replace(cfg, scenarios={**cfg.scenarios, scenario: dataclasses.replace(spec, bands=bands)})


def host_index(b):
    return {h: i for i, h in enumerate(b.audit.host_names)}


def rows_of(b, ev):
    idx = host_index(b)
    hosts = np.asarray([idx[h] for h in ev.hosts.split(";")])
    steps = np.arange(ev.start_step, ev.end_step + 1)
    return (steps[None, :] * b.audit.n_hosts + hosts[:, None]).ravel()


# ----------------------------------------------------------------------------
# A. Configuration
# ----------------------------------------------------------------------------
def test_default_config_is_valid_and_matches_agreed_design():
    cfg = gs.default_config()
    gs.validate_config(cfg)
    assert cfg.purge_steps == 30 and cfg.sample_interval_seconds == 10 and cfg.feature_dtype == "float32"
    assert {s: p.event_counts["abrupt_power_drop"] for s, p in cfg.split_plans.items()} == {
        "supervised_train": {"baseline": 60}, "validation": {"baseline": 6}, "test": {"baseline": 3, "held_out": 3}}
    base = {s: {b.name: b.ranges for b in spec.bands} for s, spec in cfg.scenarios.items()}
    assert base["abrupt_power_drop"]["baseline"]["power_multiplier"] == (0.35, 0.50)
    assert base["abrupt_power_drop"]["held_out"]["power_multiplier"] == (0.20, 0.30)
    assert base["multi_host_event"]["held_out"] == {"power_multiplier": (0.40, 0.55), "temp_offset": (3.0, 5.0)}
    assert all(b.status == "proposed_unvalidated" for s in cfg.scenarios.values() for b in s.bands if b.held_out)


@pytest.mark.parametrize("mutate", [
    lambda c: set_band(c, "abrupt_power_drop", "baseline", power_multiplier=(0.5, 0.35)),       # inverted
    lambda c: set_band(c, "transient_spike", "baseline", power_multiplier=(0.5, 0.9)),          # wrong direction
    lambda c: set_band(c, "abrupt_power_drop", "baseline", power_multiplier=(float("nan"), 0.4)),
    lambda c: set_band(c, "thermal_anomaly", "baseline", temp_offset=(-3.0, -1.0)),             # wrong direction
    lambda c: dataclasses.replace(c, scenarios={**c.scenarios, "thermal_anomaly": dataclasses.replace(
        c.scenarios["thermal_anomaly"], duration_steps=(10, 5))}),
    lambda c: dataclasses.replace(c, scenarios={**c.scenarios, "thermal_anomaly": dataclasses.replace(
        c.scenarios["thermal_anomaly"], duration_steps=(0, 5))}),
    lambda c: dataclasses.replace(c, train_frac=0.9, val_frac=0.2),
    lambda c: dataclasses.replace(c, purge_steps=-1),
    lambda c: dataclasses.replace(c, split_plans={**c.split_plans, "validation": gs.SplitPlan("global", {
        **c.split_plans["validation"].event_counts, "transient_spike": {"baseline": -1}})}),
    lambda c: dataclasses.replace(c, split_plans={**c.split_plans, "validation": gs.SplitPlan("sideways", c.split_plans["validation"].event_counts)}),
    lambda c: dataclasses.replace(c, split_plans={**c.split_plans, "validation": gs.SplitPlan("global", {
        **c.split_plans["validation"].event_counts, "transient_spike": {"baseline": 0}})}),     # scenario not exercised
    lambda c: dataclasses.replace(c, split_plans={**c.split_plans, "validation": gs.SplitPlan("global", {
        s: {"baseline": 2, "held_out": 1} for s in gs.SCENARIO_NAMES})}),                       # held-out outside test
])
def test_invalid_configuration_rejected(mutate):
    with pytest.raises(gs.ConfigError):
        gs.validate_config(mutate(gs.default_config()))


def test_missing_scenario_definition_rejected():
    cfg = gs.default_config()
    scen = {k: v for k, v in cfg.scenarios.items() if k != "multi_host_event"}
    with pytest.raises(gs.ConfigError, match="multi_host_event"):
        gs.validate_config(dataclasses.replace(cfg, scenarios=scen))


def test_held_out_overlap_with_baseline_needs_explicit_flag():
    cfg = set_band(gs.default_config(), "abrupt_power_drop", "held_out", power_multiplier=(0.30, 0.45))
    with pytest.raises(gs.ConfigError, match="overlaps"):
        gs.validate_config(cfg)
    gs.validate_config(dataclasses.replace(cfg, allow_band_overlap=True))


def test_config_json_round_trip():
    cfg = gs.default_config(seed=7)
    again = gs.config_from_dict(json.loads(json.dumps(gs.config_to_dict(cfg))))
    assert again == cfg
    with pytest.raises(gs.ConfigError):
        gs.config_from_dict({**gs.config_to_dict(cfg), "surprise": 1})


# ----------------------------------------------------------------------------
# Source audit / edge cases
# ----------------------------------------------------------------------------
def _build(raw, cfg=None):
    return gs.build_datasets(raw, cfg or small_cfg())


def test_nonfinite_features_rejected(source):
    raw = source.copy()
    raw.loc[10, POWER[0]] = np.float32("nan")
    with pytest.raises(gs.SourceError, match="NaN/inf"):
        _build(raw)


def test_duplicate_host_timestamp_rejected(source):
    raw = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(gs.SourceError, match="duplicate"):
        _build(raw)


def test_nonzero_source_label_rejected(source):
    raw = source.copy()
    raw.loc[3, "label"] = 1
    with pytest.raises(gs.SourceError, match="non-zero labels"):
        _build(raw)


def test_incomplete_panel_rejected(source):
    with pytest.raises(gs.SourceError, match="Incomplete host/timestamp coverage"):
        _build(source.iloc[1:].reset_index(drop=True))


def test_one_host_source_fails_clearly():
    with pytest.raises(gs.CapacityError, match="at least 3 hosts"):
        _build(make_source(n_hosts=1))


def test_missing_power_or_temperature_features_rejected(source):
    with pytest.raises(gs.SourceError, match="power"):
        _build(source.drop(columns=POWER))
    with pytest.raises(gs.SourceError, match="temperature"):
        _build(source.drop(columns=TEMPS + SUMMARY))


def test_wrong_interval_rejected():
    with pytest.raises(gs.SourceError, match="interval"):
        _build(make_source(interval="20s"))


def test_feature_dtype_mismatch_rejected_not_cast():
    with pytest.raises(gs.SourceError, match="never silently cast"):
        _build(make_source(dtype="float64"))


def test_reserved_column_collision_rejected(source):
    raw = source.copy()
    raw["event_id"] = ""
    with pytest.raises(gs.SourceError, match="reserved"):
        _build(raw)


def test_short_dataset_fails_clearly():
    with pytest.raises(gs.SourceError, match="at least 10"):
        _build(make_source(n_steps=8))
    with pytest.raises((gs.CapacityError, gs.EventError), match="place|usable|need"):  # splits fine, events do not fit
        _build(make_source(n_steps=300))


def test_zero_power_column_is_handled_by_absolute_criterion(source):
    raw = source.copy()
    raw[POWER[0]] = np.float32(0.0)
    b = _build(raw)
    assert (b.events["status"] == "validated").all()


def test_preexisting_summary_violation_is_recorded_not_blamed(source):
    raw = source.copy()
    bad = raw.index[:25]
    raw.loc[bad, "p0_temp_min"] = raw.loc[bad, "p0_temp_mean"] + np.float32(5)
    b = _build(raw)
    assert b.audit.facts["preexisting_summary_violations"] == 25
    assert gs.check_temp_summary(b).status == "PASS"


# ----------------------------------------------------------------------------
# B. Splits
# ----------------------------------------------------------------------------
def test_split_bounds_for_the_audited_source_shape():
    bounds = gs.split_bounds(8640, gs.default_config())
    assert bounds == {"train": (0, 5184), "validation": (5214, 6912), "test": (6942, 8640)}
    assert (bounds["validation"][1] - bounds["validation"][0]) * 100 == 169_800


def test_splits_are_chronological_disjoint_and_purged(build):
    c = gs.check_splits(build)
    assert c.status == "PASS", c.details
    assert build.split_meta["purged_rows"] == 2 * 5 * N_HOSTS
    tr, va, te = (build.frames[n]["timestamp"] for n in ("train", "validation", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert set(tr) & set(va) == set() and set(va) & set(te) == set()


# ----------------------------------------------------------------------------
# C. Dataset architecture
# ----------------------------------------------------------------------------
def test_all_checks_pass_on_a_clean_build(build):
    results = gs.run_checks(build)
    failed = [(c.id, c.details) for c in results if c.status == "FAIL"]
    assert failed == []
    na = {c.id for c in results if c.status == "NOT_APPLICABLE"}
    assert na == {"C02", "C16", "C17"}  # hashes, protect paths and staged files are only supplied by run()


def test_normal_train_is_label_free_and_equals_source_rows(build, source):
    tr = build.frames["train"]
    assert (tr["label"] == 0).all() and (tr["anomaly_type"] == "normal").all() and (tr["event_id"] == "").all()
    ref = build.source.set_index("source_row_id").loc[tr["source_row_id"]].reset_index(drop=True)
    for col in POWER + TEMPS + SUMMARY:
        assert np.array_equal(ref[col].to_numpy(), tr[col].to_numpy())


def test_supervised_train_is_a_separate_labelled_copy(build):
    norm, sup = build.frames["train"], build.frames["supervised_train"]
    assert norm is not sup and len(norm) == len(sup)
    assert norm["source_row_id"].equals(sup["source_row_id"]) and norm["hostname"].equals(sup["hostname"])
    assert (norm["label"] == 0).all() and (sup["label"] == 1).sum() > 0
    feats = build.audit.feat.features
    unchanged = ~(norm[feats].to_numpy() != sup[feats].to_numpy()).any(axis=1)
    assert unchanged[sup["label"].to_numpy() == 0].all()
    assert not unchanged[sup["label"].to_numpy() == 1].any()


def test_labels_agree_with_event_records(build):
    assert gs.check_labels(build).status == "PASS"
    assert gs.check_manifest(build).status == "PASS"
    for split in gs.SPLIT_NAMES:
        fr, ev = build.frames[split], build.events[build.events["split"] == split]
        assert (fr["label"] == 1).sum() == ev["n_rows"].sum()


def test_schema_and_dtypes_identical_across_datasets(build):
    ref = build.frames["train"]
    for fr in build.frames.values():
        assert list(fr.columns) == list(ref.columns) and fr.dtypes.equals(ref.dtypes)
    assert {str(fr[c].dtype) for fr in build.frames.values() for c in build.audit.feat.features} == {"float32"}
    assert {"source_row_id", "source_label", "label", "anomaly_type", "event_id", "xid", "job_id"} <= set(ref.columns)


# ----------------------------------------------------------------------------
# D. Scenario transformations
# ----------------------------------------------------------------------------
def test_all_five_scenarios_exercised_in_every_injected_split(build):
    for split in gs.SPLIT_NAMES:
        got = set(build.events.loc[build.events["split"] == split, "anomaly_type"])
        assert got == set(gs.SCENARIO_NAMES), split


def test_each_scenario_changes_only_its_features_with_realized_effect_matching_target(build):
    feat, H = build.audit.feat, build.audit.n_hosts
    cfg = build.cfg
    for split in gs.SPLIT_NAMES:
        fr, base = build.frames[split], build.bases[split]
        for ev in build.events[build.events["split"] == split].itertuples(index=False):
            rows = rows_of(build, ev)
            b = base[feat.features].to_numpy()[rows].astype(np.float64)
            a = fr[feat.features].to_numpy()[rows].astype(np.float64)
            P, T = feat.power_idx, feat.temp_idx
            if ev.anomaly_type == "thermal_anomaly":
                assert np.array_equal(a[:, P], b[:, P])
            else:
                assert not np.array_equal(a[:, P], b[:, P])
            if ev.anomaly_type in ("thermal_anomaly", "multi_host_event"):
                assert np.abs((a[:, T] - b[:, T]) - ev.target_temp_offset).max() < cfg.realized_offset_tol
            else:
                assert np.array_equal(a[:, T], b[:, T])
            if ev.anomaly_type in ("abrupt_power_drop", "transient_spike", "multi_host_event"):
                ratio = a[:, P] / b[:, P]
                assert np.abs(ratio - ev.target_power_multiplier).max() < cfg.realized_ratio_tol
                assert (ratio < 1).all() == (ev.target_power_multiplier < 1)
            assert not (np.isnan(a).any())


def test_non_event_rows_and_other_columns_unchanged(build):
    for split in gs.SPLIT_NAMES:
        fr, base = build.frames[split], build.bases[split]
        lab = fr["label"].to_numpy() == 1
        for col in fr.columns:
            if col in build.audit.feat.features or col in ("label", "anomaly_type", "event_id"):
                continue
            assert fr[col].equals(base[col]), col
        f = build.audit.feat.features
        assert np.array_equal(fr[f].to_numpy()[~lab], base[f].to_numpy()[~lab])


def test_gradual_degradation_changes_first_row_and_is_monotonic(build):
    feat = build.audit.feat
    checked = 0
    for split in gs.SPLIT_NAMES:
        for ev in build.events[(build.events["split"] == split) & (build.events["anomaly_type"] == "gradual_degradation")].itertuples(index=False):
            rows = rows_of(build, ev)
            b = build.bases[split][feat.features].to_numpy()[rows].astype(np.float64).reshape(1, ev.duration_steps, -1)
            a = build.frames[split][feat.features].to_numpy()[rows].astype(np.float64).reshape(1, ev.duration_steps, -1)
            ratio = (a[:, :, feat.power_idx] / b[:, :, feat.power_idx]).mean(axis=2)[0]
            n, f = ev.duration_steps, ev.target_final_power_factor
            expected = 1 + (f - 1) * (np.arange(n) + 1) / n
            assert ratio[0] < 1 - 1e-3 and abs(ratio[0] - expected[0]) < 1e-4      # first labelled row already changed
            assert abs(ratio[-1] - f) < 1e-4                                      # last row reaches the final factor
            assert np.all(np.diff(ratio) < 0)
            assert abs(ev.realized_final_power_factor - f) < 1e-4
            checked += 1
    assert checked == len(build.events[build.events["anomaly_type"] == "gradual_degradation"])


def test_multi_host_events_cover_all_hosts_with_one_window_and_both_effects(build):
    multi = build.events[build.events["anomaly_type"] == "multi_host_event"]
    assert len(multi) > 0 and (multi["n_hosts"] == 4).all()
    for ev in multi.itertuples(index=False):
        fr = build.frames[ev.split]
        sub = fr[fr["event_id"] == ev.event_id]
        assert sub["hostname"].nunique() == 4
        assert sub.groupby("hostname")["timestamp"].agg(["min", "max"]).nunique().max() == 1  # identical window per host
        assert len(sub) == 4 * ev.duration_steps
        assert not pd.isna(ev.target_power_multiplier) and not pd.isna(ev.target_temp_offset)
        assert not pd.isna(ev.realized_power_multiplier) and not pd.isna(ev.realized_temp_offset)


def test_temperature_summary_ordering_is_preserved_by_thermal_events(build):
    feat = build.audit.feat
    for split in gs.SPLIT_NAMES:
        fr = build.frames[split]
        sub = fr[fr["anomaly_type"].isin(["thermal_anomaly", "multi_host_event"])]
        assert not gs.summary_violations(sub, feat.summary_groups).any()
    assert gs.check_temp_summary(build).status == "PASS"


# ----------------------------------------------------------------------------
# E. Event placement
# ----------------------------------------------------------------------------
def test_placement_rules_hold(build):
    c = gs.check_params_and_placement(build)
    assert c.status == "PASS", c.details


def test_event_ids_unique_durations_exact_windows_inside_split(build):
    ev = build.events
    assert ev["event_id"].is_unique
    assert (ev["duration_steps"] == ev["end_step"] - ev["start_step"] + 1).all()
    assert (ev["duration_seconds"] == ev["duration_steps"] * 10).all()
    assert (ev["n_rows"] == ev["n_hosts"] * ev["duration_steps"]).all()
    for split in gs.SPLIT_NAMES:
        sub = ev[ev["split"] == split]
        assert sub["start_step"].min() >= build.cfg.edge_margin_steps
        assert sub["end_step"].max() <= build.steps[split] - 1 - build.cfg.edge_margin_steps


def test_global_policy_never_overlaps_and_per_host_policy_allows_concurrency_without_conflicts(build):
    for split in ("validation", "test"):
        sub = build.events[build.events["split"] == split].sort_values("start_step")
        gaps = sub["start_step"].to_numpy()[1:] - sub["end_step"].to_numpy()[:-1] - 1
        assert (gaps >= build.cfg.guard_steps).all()
    sup = build.events[build.events["split"] == "supervised_train"].sort_values("start_step")
    concurrent = (sup["start_step"].to_numpy()[1:] <= sup["end_step"].to_numpy()[:-1]).any()
    assert concurrent  # different hosts overlap in time in supervised training
    # but no host/timestamp is claimed twice
    fr = build.frames["supervised_train"]
    assert int((fr["label"] == 1).sum()) == int(sup["n_rows"].sum())


def test_insufficient_capacity_fails_instead_of_dropping_or_shortening_events(source):
    cfg = small_cfg(split_plans=small_plans(per_val=40))
    with pytest.raises(gs.CapacityError, match="never shortened"):
        gs.build_datasets(source, cfg)
    cfg = small_cfg(split_plans=small_plans(per_train=400))
    with pytest.raises(gs.CapacityError):
        gs.build_datasets(source, cfg)


def test_event_counts_equal_configuration_exactly(build):
    for split, plan in build.cfg.split_plans.items():
        sub = build.events[build.events["split"] == split]
        for scenario, bands in plan.event_counts.items():
            for band, n in bands.items():
                got = ((sub["anomaly_type"] == scenario) & (sub["severity_band"] == band)).sum()
                assert got == n, (split, scenario, band)


def test_severity_and_duration_ranges_are_respected_and_spread_across_the_band(build):
    cfg = build.cfg
    for ev in build.events.itertuples(index=False):
        spec = cfg.scenarios[ev.anomaly_type]
        band = next(b for b in spec.bands if b.name == ev.severity_band)
        assert spec.duration_steps[0] <= ev.duration_steps <= spec.duration_steps[1]
        for dim, (lo, hi) in band.ranges.items():
            assert lo - 1e-12 <= getattr(ev, f"target_{dim}") <= hi + 1e-12
    # stratified sampling: 3 baseline training events per scenario land in different thirds of the band
    sup = build.events[(build.events["split"] == "supervised_train") & (build.events["anomaly_type"] == "abrupt_power_drop")]
    lo, hi = 0.35, 0.50
    thirds = sorted(int(min(2, (v - lo) / (hi - lo) * 3)) for v in sup["target_power_multiplier"])
    assert thirds == [0, 1, 2]


# ----------------------------------------------------------------------------
# F. Held-out severity
# ----------------------------------------------------------------------------
def test_held_out_events_only_in_test_and_tagged(build):
    ev = build.events
    assert not ev[ev["split"] != "test"]["held_out"].any()
    test = ev[ev["split"] == "test"]
    assert test["held_out"].any() and (~test["held_out"]).any()
    assert set(test.loc[test["held_out"], "severity_band"]) == {"held_out"}
    assert set(test.loc[test["held_out"], "band_status"]) == {"proposed_unvalidated"}
    assert set(ev.loc[~ev["held_out"], "severity_band"]) == {"baseline"}
    assert gs.check_held_out(build).status == "PASS"


def test_held_out_parameters_never_appear_in_training_or_validation(build):
    for r in build.events[build.events["split"] != "test"].itertuples(index=False):
        held = next(b for b in build.cfg.scenarios[r.anomaly_type].bands if b.held_out)
        inside = all(lo <= getattr(r, f"target_{d}") <= hi for d, (lo, hi) in held.ranges.items())
        assert not inside
    sup_vals = build.events[(build.events["split"] == "supervised_train") & (build.events["anomaly_type"] == "transient_spike")]["target_power_multiplier"]
    assert sup_vals.max() <= 1.65 and build.events[(build.events["held_out"]) & (build.events["anomaly_type"] == "transient_spike")]["target_power_multiplier"].min() >= 1.70


# ----------------------------------------------------------------------------
# G. Reproducibility and integrity
# ----------------------------------------------------------------------------
def test_same_seed_gives_identical_datasets_and_manifests(source):
    a, b = gs.build_datasets(source, small_cfg()), gs.build_datasets(source, small_cfg())
    for name in a.frames:
        assert gs.frame_digest(a.frames[name]) == gs.frame_digest(b.frames[name])
    pd.testing.assert_frame_equal(a.events, b.events)
    assert a.plans == b.plans and a.split_meta == b.split_meta and a.pools == b.pools


def test_different_seed_gives_different_plan_and_host_pools(source, build):
    other = gs.build_datasets(source, small_cfg(seed=43))
    assert other.plans != build.plans and other.pools != build.pools
    assert gs.frame_digest(other.frames["test"]) != gs.frame_digest(build.frames["test"])


def test_validation_and_test_schedules_are_independent(build):
    v = build.events[build.events["split"] == "validation"].sort_values("start_step")
    t = build.events[build.events["split"] == "test"].sort_values("start_step")
    assert list(v["start_step"]) != list(t["start_step"])
    assert list(v["anomaly_type"]) != list(t["anomaly_type"])
    assert set(v["event_seed"]).isdisjoint(set(t["event_seed"]))


def test_dropping_a_scenario_from_the_plan_cannot_hide_inside_a_valid_run(build):
    # corrupting the plan makes the reproducibility gate fail
    broken = copy.copy(build)
    broken.plans = {**build.plans, "test": build.plans["test"][:-1]}
    assert gs.check_plan_reproducibility(broken).status == "FAIL"


# ----------------------------------------------------------------------------
# Validation gates must FAIL on corrupted data
# ----------------------------------------------------------------------------
def _corrupt(build, split, fn):
    b = copy.copy(build)
    b.frames = {k: v.copy() for k, v in build.frames.items()}
    fn(b.frames[split])
    return b


def test_gate_detects_wrong_label(build):
    def f(fr):
        fr.loc[fr.index[fr["label"] == 1][0], "label"] = 0
    b = _corrupt(build, "validation", f)
    assert gs.check_labels(b).status == "FAIL" and gs.check_manifest(b).status == "FAIL"


def test_gate_detects_modified_unlabelled_row(build):
    def f(fr):
        i = fr.index[fr["label"] == 0][3]
        fr.loc[i, POWER[0]] = fr.loc[i, POWER[0]] + np.float32(1)
    assert gs.check_realized_effects(_corrupt(build, "test", f)).status == "FAIL"


def test_gate_detects_overlapping_events(build):
    b = copy.copy(build)
    ev = build.events.copy()
    idx = ev.index[ev["split"] == "validation"]
    ev.loc[idx[1], "start_step"] = ev.loc[idx[0], "start_step"] + 1
    b.events = ev
    assert gs.check_params_and_placement(b).status == "FAIL"


def test_gate_detects_new_temperature_summary_violation(build):
    def f(fr):
        i = fr.index[fr["anomaly_type"] == "thermal_anomaly"][0]
        fr.loc[i, "p0_temp_min"] = fr.loc[i, "p0_temp_mean"] + np.float32(5)
    assert gs.check_temp_summary(_corrupt(build, "test", f)).status == "FAIL"


def test_gate_detects_dtype_and_schema_drift(build):
    def f(fr):
        fr[POWER[0]] = fr[POWER[0]].astype("float64")
    assert gs.check_schema(_corrupt(build, "test", f)).status == "FAIL"


def test_gate_detects_non_normal_training_data(build):
    def f(fr):
        fr.loc[fr.index[0], "label"] = 1
    assert gs.check_normal_train(_corrupt(build, "train", f)).status == "FAIL"


def test_gate_detects_mutated_normal_train_vs_supervised(build):
    def f(fr):
        fr.loc[fr.index[5], "hostname"] = "tampered"
    assert gs.check_supervised_copy(_corrupt(build, "supervised_train", f)).status == "FAIL"


def test_measure_event_rejects_ineffective_or_wrong_events(build):
    feat = build.audit.feat
    pe = build.plans["validation"][0]
    rows = gs.event_rows(pe, build.audit.n_hosts)
    base = build.bases["validation"][feat.features].to_numpy()[rows]
    limit = np.array([build.limits[c]["max"] for c in feat.features])
    with pytest.raises(gs.EventError, match="no feature change"):
        gs.measure_event(pe, base, base.copy(), feat, build.cfg, limit)
    wrong = build.frames["validation"][feat.features].to_numpy()[rows].copy()
    wrong[:, feat.power_idx[0]] *= np.float32(1.05)
    with pytest.raises(gs.EventError):
        gs.measure_event(pe, base, wrong, feat, build.cfg, limit)


# ----------------------------------------------------------------------------
# I/O: staging, publication, overwrite protection
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def source_file(tmp_path_factory, source):
    p = tmp_path_factory.mktemp("src") / "dataset" / "train.parquet"
    p.parent.mkdir()
    source.to_parquet(p, index=False)
    return p


@pytest.fixture(scope="module")
def published(tmp_path_factory, source_file):
    out = tmp_path_factory.mktemp("out") / "synthetic_v3"
    before = gs.sha256_file(source_file)
    report = gs.run(small_cfg(), source_file, out, verbose=False)
    return out, report, before


def test_run_publishes_the_complete_artifact_set_and_leaves_source_untouched(published, source_file):
    out, report, before = published
    assert sorted(p.name for p in out.iterdir()) == sorted(gs.OUTPUT_FILES)
    assert gs.sha256_file(source_file) == before
    assert report["source"]["unchanged"] is True and report["validation"]["overall"] == "PASS"
    statuses = {c["id"]: c["status"] for c in report["validation"]["checks"]}
    assert statuses["C16"] == "NOT_APPLICABLE" and all(s in ("PASS", "NOT_APPLICABLE") for s in statuses.values())
    assert statuses["C17"] == "PASS" and statuses["C02"] == "PASS"
    assert not list(out.parent.glob(".*staging*")) and not list(out.parent.glob("*_FAILED_*"))


def test_reports_and_manifests_agree_with_written_parquet(published):
    out, report, _ = published
    ev = pd.read_csv(out / gs.EVENTS_FILE)
    for name, fname in gs.DATASET_FILES.items():
        fr = pd.read_parquet(out / fname)
        counts = {str(k): int(v) for k, v in fr["label"].value_counts().sort_index().items()}
        assert report["datasets"][name]["label_counts"] == counts and report["datasets"][name]["rows"] == len(fr)
    assert sum(sum(b.values()) for s in report["event_counts"].values() for b in s.values()) == len(ev)
    sup = pd.read_csv(out / gs.SUPERVISED_EVENTS_FILE)
    assert list(sup["event_id"]) == list(ev.loc[ev["split"] == "supervised_train", "event_id"])
    cfg_saved = gs.config_from_dict(json.loads((out / "generation_config.json").read_text()))
    assert cfg_saved == small_cfg()
    manifest = json.loads((out / "feature_manifest.json").read_text())
    assert len(manifest["model_feature_columns"]) == 28 and "label" not in manifest["model_feature_columns"]
    assert "synthetic dataset report" in (out / "synthetic_report.txt").read_text().lower()
    assert report["severity_bands_used"]["supervised_train"]["abrupt_power_drop"].keys() == {"baseline"}
    assert report["severity_bands_used"]["test"]["abrupt_power_drop"].keys() == {"baseline", "held_out"}


def test_second_run_to_a_new_directory_is_content_identical(published, source_file, tmp_path):
    out, report, _ = published
    again = gs.run(small_cfg(), source_file, tmp_path / "v3b", verbose=False)
    assert again["content_digests"] == report["content_digests"]
    a, b = dict(report), dict(again)
    for key in ("generated_at_utc",):
        a.pop(key), b.pop(key)
    assert a == b


def test_existing_output_is_never_overwritten(published, source_file):
    out, _, _ = published
    marker = {p.name: gs.sha256_file(p) for p in out.iterdir()}
    with pytest.raises(gs.OutputExistsError):
        gs.run(small_cfg(), source_file, out, verbose=False)
    assert {p.name: gs.sha256_file(p) for p in out.iterdir()} == marker


def test_non_empty_directory_with_foreign_file_is_refused(source_file, tmp_path):
    target = tmp_path / "v2_like"
    target.mkdir()
    (target / "synthetic_report.json").write_text("{}")
    with pytest.raises(gs.OutputExistsError, match="non-empty"):
        gs.run(small_cfg(), source_file, target, verbose=False)
    assert [p.name for p in target.iterdir()] == ["synthetic_report.json"]


def test_output_inside_source_folder_or_protected_path_is_refused(source_file, tmp_path):
    with pytest.raises(gs.OutputExistsError):
        gs.run(small_cfg(), source_file, source_file.parent / "v3", verbose=False)
    protected = tmp_path / "v2"
    protected.mkdir()
    with pytest.raises(gs.OutputExistsError):
        gs.run(small_cfg(), source_file, protected / "v3", protect=[protected], verbose=False)


def test_protected_paths_are_hashed_and_reported(source_file, tmp_path):
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "a.txt").write_text("v2 data")
    report = gs.run(small_cfg(), source_file, tmp_path / "v3p", protect=[v2], verbose=False)
    status = {c["id"]: c["status"] for c in report["validation"]["checks"]}
    assert status["C16"] == "PASS" and (v2 / "a.txt").read_text() == "v2 data"


def test_failed_validation_publishes_nothing(source_file, tmp_path, monkeypatch):
    real = gs.run_checks

    def sabotaged(*a, **k):
        res = real(*a, **k)
        res[5] = gs.CheckResult("C06", "supervised_copy_integrity", "FAIL", "injected failure", {})
        return res

    monkeypatch.setattr(gs, "run_checks", sabotaged)
    out = tmp_path / "v3_bad"
    with pytest.raises(gs.ValidationFailed, match="injected failure"):
        gs.run(small_cfg(), source_file, out, verbose=False)
    assert not out.exists()
    failed = list(tmp_path.glob("v3_bad_FAILED_*"))
    assert len(failed) == 1 and (failed[0] / "FAILED.txt").is_file()
    assert not list(tmp_path.glob(".*staging*"))


def test_generation_error_before_writing_leaves_no_directory(source_file, tmp_path):
    cfg = small_cfg(split_plans=small_plans(per_val=40))
    with pytest.raises(gs.CapacityError):
        gs.run(cfg, source_file, tmp_path / "v3_cap", verbose=False)
    assert list(tmp_path.iterdir()) == []


def test_cli_returns_error_code_for_refused_output(source_file, tmp_path, capsys):
    target = tmp_path / "taken"
    target.mkdir()
    (target / "x").write_text("1")
    code = gs.main(["--source", str(source_file), "--output-dir", str(target)])
    assert code == 1 and "ERROR" in capsys.readouterr().err
