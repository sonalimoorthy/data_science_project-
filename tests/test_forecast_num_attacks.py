"""Tests for analysis/predictive/regressionModel/forecast_num_attacks.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_MOD = _ROOT / "analysis" / "predictive" / "regressionModel" / "forecast_num_attacks.py"
_spec = importlib.util.spec_from_file_location("forecast_num_attacks", _MOD)
assert _spec and _spec.loader
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)


def test_global_yearly_fills_gap_with_zero() -> None:
    df = pd.DataFrame({"Year": [2000, 2002], "Num_Attacks": [100, 300]})
    s = fc.global_yearly_totals(df)
    assert s.loc[2001] == 0
    assert s.loc[2000] == 100
    assert s.loc[2002] == 300


def test_supervised_lag_aligns() -> None:
    s = pd.Series([10.0, 20.0, 40.0], index=[2000, 2001, 2002])
    sup = fc.supervised_frame(s)
    assert list(sup["lag1"]) == [10.0, 20.0]
    assert list(sup["y"]) == [20.0, 40.0]
    assert list(sup["year"]) == [2001, 2002]


def test_horizon_one_matches_direct_predict() -> None:
    rng = np.random.default_rng(0)
    years = np.arange(1980, 2000)
    y = 1000 + 10 * (years - 1980) + rng.normal(0, 50, size=len(years))
    s = pd.Series(y, index=years)
    sup = fc.supervised_frame(s)
    anchor = 1995
    train = sup[(sup["year"] < anchor) & (sup["year"] >= 1982)]
    lag0 = float(sup.loc[sup["year"] == anchor, "y"].iloc[0])
    model = fc.Pipeline(
        [
            ("scaler", fc.StandardScaler()),
            ("ridge", fc.RidgeCV(alphas=[1.0])),
        ]
    )
    model.fit(train[["lag1", "year"]], train["y"])
    multi = fc.iterative_predict(model, lag0, anchor, 1)
    direct = float(
        model.predict(pd.DataFrame({"lag1": [lag0], "year": [anchor + 1]}))[0]
    )
    assert np.isclose(multi, direct)
