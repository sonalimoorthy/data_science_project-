"""Tests for analysis/diagnostic/rootCause/root_cause_drivers.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = _ROOT / "analysis" / "diagnostic" / "rootCause" / "root_cause_drivers.py"
_spec = importlib.util.spec_from_file_location("root_cause_drivers", _MOD_PATH)
assert _spec and _spec.loader
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def test_region_group_table_orders_by_mean_attacks() -> None:
    df = pd.DataFrame(
        {
            "Region": ["A", "A", "B", "B"],
            "Num_Attacks": [10, 10, 1, 1],
            "Conflict_Level": [1, 1, 0, 0],
            "Conflict": [1, 0, 0, 0],
            "GDP_per_capita": [1000] * 4,
            "Population": [1e6] * 4,
            "Fatalities": [0] * 4,
            "Injuries": [0] * 4,
            "Year": [2000] * 4,
            "Attack_Type_Mode": ["X"] * 4,
        }
    )
    out = rc.region_group_table(df)
    assert out.index[0] == "A"
    assert out.loc["A", "mean_attacks"] > out.loc["B", "mean_attacks"]


def test_linear_ranking_puts_dominant_numeric_first() -> None:
    rng = np.random.default_rng(1)
    n = 200
    fatalities = rng.normal(0, 1, size=n)
    noise = rng.normal(0, 3, size=n)
    attacks = 5 * fatalities + noise
    df = pd.DataFrame(
        {
            "Region": ["R1"] * (n // 2) + ["R2"] * (n // 2),
            "GDP_per_capita": rng.normal(1, 0.1, size=n),
            "Population": rng.normal(0, 1, size=n),
            "Conflict_Level": rng.integers(0, 3, size=n),
            "Fatalities": fatalities,
            "Injuries": rng.normal(0, 1, size=n),
            "Year": 2000 + rng.integers(0, 10, size=n),
            "Conflict": rng.integers(0, 2, size=n),
            "Num_Attacks": attacks,
            "Attack_Type_Mode": ["X"] * n,
        }
    )
    _model, ranking = rc.fit_scaled_linear_with_region(df)
    assert ranking.iloc[0]["feature"] == "num__Fatalities"
