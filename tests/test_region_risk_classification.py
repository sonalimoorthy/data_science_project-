"""Tests for region / country risk classification helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MOD = _ROOT / "analysis" / "predictive" / "classification" / "region_risk_classification.py"
_spec = importlib.util.spec_from_file_location("region_risk_classification", _MOD)
assert _spec and _spec.loader
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def test_aggregate_region_year_sums_attacks() -> None:
    raw = pd.DataFrame(
        {
            "Country": ["A", "B", "B"],
            "Year": [2000, 2000, 2000],
            "Region": ["R1", "R1", "R1"],
            "Num_Attacks": [3, 5, 7],
            "Fatalities": [1, 0, 2],
            "Injuries": [0, 1, 0],
            "Conflict_Level": [1.0, 2.0, 3.0],
            "GDP_per_capita": [100.0, 200.0, 300.0],
            "Population": [1e6, 2e6, 3e6],
            "Conflict": [0, 1, 0],
        }
    )
    out = rc.aggregate_region_year(raw)
    assert len(out) == 1
    assert out.iloc[0]["Total_Attacks"] == 15


def test_risk_score_formula() -> None:
    s = rc.risk_score_from_attacks_fatalities(
        pd.Series([10.0, 0.0]), pd.Series([4.0, 0.0])
    )
    assert float(s.iloc[0]) == 12.0
    assert float(s.iloc[1]) == 0.0


def test_expand_balanced_fills_missing_year_with_zero_attacks() -> None:
    obs = pd.DataFrame(
        {
            "Region": ["R1", "R1"],
            "Year": [2000, 2002],
            "Total_Attacks": [10.0, 30.0],
            "Total_Fatalities": [1.0, 3.0],
            "Total_Injuries": [0.0, 0.0],
            "Mean_Conflict_Level": [0.0, 1.0],
            "Countries_Reporting": [1.0, 1.0],
            "Mean_GDP": [100.0, 110.0],
            "Mean_Pop": [1e6, 1e6],
            "Conflict_Share": [0.0, 0.0],
        }
    )
    bal = rc.expand_balanced_region_year(obs)
    row2001 = bal[(bal["Region"] == "R1") & (bal["Year"] == 2001)].iloc[0]
    assert row2001["Total_Attacks"] == 0


def test_attacks_roll3_excludes_current_year() -> None:
    p = pd.DataFrame(
        {
            "Region": ["R"] * 5,
            "Year": [2010, 2011, 2012, 2013, 2014],
            "Total_Attacks": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Total_Fatalities": [0.0] * 5,
            "Total_Injuries": [0.0] * 5,
            "Mean_Conflict_Level": [0.0] * 5,
            "Countries_Reporting": [1.0] * 5,
            "Mean_GDP": [1.0] * 5,
            "Mean_Pop": [1.0] * 5,
            "Conflict_Share": [0.0] * 5,
        }
    )
    out = rc.add_region_lag_features(p)
    row2012 = out[out["Year"] == 2012].iloc[0]
    assert row2012["attacks_lag1"] == 20.0
    assert row2012["attacks_roll3"] == pytest.approx((10.0 + 20.0) / 2)


def test_apply_high_risk_train_quantile() -> None:
    df = pd.DataFrame(
        {
            "risk_score": [1.0, 2.0, 3.0, 100.0],
            "Year": [2000, 2001, 2002, 2003],
        }
    )
    train = df["Year"] <= 2001
    out, thr = rc.apply_high_risk_from_train_quantile(df, train, quantile=0.75)
    assert thr == pytest.approx(1.75)
    assert out.loc[out["Year"] == 2003, "high_risk"].iloc[0] == 1


def test_pipelines_fit_and_predict() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for reg in ["R0", "R1"]:
        for y in range(2005, 2015):
            rows.append(
                {
                    "Region": reg,
                    "Year": y,
                    "attacks_lag1": float(rng.random()),
                    "attacks_lag2": float(rng.random()),
                    "attacks_roll3": float(rng.random()),
                    "fatalities_lag1": float(rng.random()),
                    "injuries_lag1": float(rng.random()),
                    "conflict_level_lag1": float(rng.random()),
                    "countries_lag1": float(rng.integers(1, 4)),
                    "log_attacks_lag1": float(rng.random()),
                    "high_risk": int(rng.integers(0, 2)),
                }
            )
    df = pd.DataFrame(rows)
    num, cat = rc.feature_columns_region()
    feat = num + cat
    tr = df.iloc[:12]
    te = df.iloc[12:16]
    pipes = rc.build_pipelines(num, cat)
    for name, pipe in pipes.items():
        pipe.fit(tr[feat], tr["high_risk"].values)
        pred = pipe.predict(te[feat])
        assert len(pred) == len(te)


def test_best_f1_threshold_uses_pr_curve() -> None:
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.55, 0.7, 0.9])
    t = rc.best_f1_threshold_from_precision_recall_curve(y, proba)
    assert 0.0 <= t <= 1.0


def test_require_data_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Data CSV not found"):
        rc.require_data(Path("/nonexistent/master_country_year.csv"))
