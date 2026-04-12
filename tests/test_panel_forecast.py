"""Tests for panel expansion and feature helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MOD = _ROOT / "analysis" / "predictive" / "panelModel" / "panel_forecast_and_risk.py"
_spec = importlib.util.spec_from_file_location("panel_forecast_and_risk", _MOD)
assert _spec and _spec.loader
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


def test_expand_balanced_panel_fills_missing_year_with_zero_attacks() -> None:
    df = pd.DataFrame(
        {
            "Country": ["A", "A"],
            "Year": [2000, 2002],
            "Region": ["R", "R"],
            "Num_Attacks": [10, 30],
            "Fatalities": [1, 3],
            "Injuries": [0, 0],
            "Conflict_Level": [0, 1],
            "GDP_per_capita": [1000, 1100],
            "Population": [1e6, 1e6],
            "Conflict": [0, 0],
        }
    )
    out = pm.expand_balanced_panel(df)
    assert len(out) == 3  # A x {2000,2001,2002}
    row2001 = out[(out["Country"] == "A") & (out["Year"] == 2001)].iloc[0]
    assert row2001["Num_Attacks"] == 0


def test_lag_and_targets_within_country() -> None:
    df = pd.DataFrame(
        {
            "Country": ["C"] * 5,
            "Year": [2010, 2011, 2012, 2013, 2014],
            "Region": ["X"] * 5,
            "Num_Attacks": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Fatalities": [0.0] * 5,
            "Injuries": [0.0] * 5,
            "Conflict_Level": [0.0] * 5,
            "GDP_per_capita": [100.0] * 5,
            "Population": [1e6] * 5,
            "Conflict": [0] * 5,
        }
    )
    p = pm.expand_balanced_panel(df)
    p = pm.add_panel_features(p)
    row = p[(p["Country"] == "C") & (p["Year"] == 2012)].iloc[0]
    assert row["num_attacks_lag1"] == 2.0
    assert row["y_plus1"] == 4.0


def test_require_panel_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Panel data CSV not found"):
        pm.require_panel_csv(Path("/nonexistent/path/master_country_year.csv"))


def test_feature_columns_exclude_same_year_leakage() -> None:
    numeric, _ = pm.feature_columns()
    banned = {
        "Fatalities",
        "Injuries",
        "GDP_per_capita",
        "Population",
        "Conflict_Level",
        "Conflict",
        "log_gdp",
        "log_fatalities",
        "log_injuries",
        "log_pop",
        "gdp_x_conflict",
    }
    assert banned.isdisjoint(set(numeric))


def test_structural_covariates_are_lagged() -> None:
    df = pd.DataFrame(
        {
            "Country": ["C"] * 4,
            "Year": [2010, 2011, 2012, 2013],
            "Region": ["X"] * 4,
            "Num_Attacks": [1.0, 2.0, 3.0, 4.0],
            "Fatalities": [0.0] * 4,
            "Injuries": [0.0] * 4,
            "Conflict_Level": [1.0, 2.0, 3.0, 4.0],
            "GDP_per_capita": [100.0, 200.0, 300.0, 400.0],
            "Population": [1e6] * 4,
            "Conflict": [0, 1, 0, 1],
        }
    )
    p = pm.expand_balanced_panel(df)
    p = pm.add_panel_features(p)
    row = p[(p["Country"] == "C") & (p["Year"] == 2012)].iloc[0]
    assert row["gdp_per_capita_lag1"] == 200.0
    assert row["conflict_level_lag1"] == 2.0
    assert row["conflict_lag1"] == 1.0


def test_log1p_target_inverse_roundtrip() -> None:
    y = np.array([0.0, 1.0, 99.0, 1000.0])
    z = pm.target_log1p(y)
    back = pm.inverse_target_log1p(z)
    np.testing.assert_allclose(back, y, rtol=1e-12)


def test_train_eval_direct_horizon_log_target_finite_rmse() -> None:
    rng = np.random.default_rng(0)
    n = 80
    years_c0 = list(range(2000, 2000 + n // 2))
    years_c1 = list(range(2000, 2000 + n // 2))
    train_df = pd.DataFrame(
        {
            "Country": ["C0"] * (n // 2) + ["C1"] * (n // 2),
            "Year": years_c0 + years_c1,
            "Region": ["R"] * n,
            "num_attacks_lag1": rng.poisson(2, n).astype(float),
            "num_attacks_lag2": rng.poisson(2, n).astype(float),
            "num_attacks_roll3": rng.poisson(2, n).astype(float),
            "log_gdp_lag1": rng.random(n) * 5,
            "log_pop_lag1": rng.random(n) * 3,
            "gdp_x_conflict_lag1": rng.random(n),
            "conflict_level_lag1": rng.random(n),
            "conflict_lag1": rng.integers(0, 2, n).astype(float),
            "y_plus1": rng.poisson(3, n).astype(float),
        }
    )
    test_df = train_df.iloc[:20].copy()
    train_df = train_df.iloc[20:].copy()
    numeric, categorical = pm.feature_columns()
    out = pm.train_eval_direct_horizon(
        train_df,
        test_df,
        "y_plus1",
        numeric,
        categorical,
        log_target=True,
    )
    assert np.isfinite(out["rmse_rf"])
    assert np.isfinite(out["rmse_ridge"])
    assert (out["pred_rf"] >= 0).all()


def test_groupkfold_runs_on_multi_country_panel() -> None:
    rows = []
    for c in ["C0", "C1", "C2", "C3", "C4"]:
        for y in range(2010, 2016):
            rows.append(
                {
                    "Country": c,
                    "Year": y,
                    "Region": "R",
                    "Num_Attacks": float((y - 2010) + 1),
                    "Fatalities": 1.0,
                    "Injuries": 0.0,
                    "Conflict_Level": 0.5,
                    "GDP_per_capita": 1000.0 + y,
                    "Population": 1e6,
                    "Conflict": 0,
                }
            )
    raw = pd.DataFrame(rows)
    panel = pm.add_panel_features(pm.expand_balanced_panel(raw))
    numeric, categorical = pm.feature_columns()
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric)
    out = pm.train_eval_grouped_cv(panel, "y_plus1", numeric, categorical, n_splits=5)
    assert out is not None
    assert out["n_splits"] == 5
    assert out["n_rows"] > 0
