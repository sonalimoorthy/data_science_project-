"""Unit tests for analysis/diagnostic/hypothesis/hypothesis_tests.py (synthetic panels)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_HYP_PATH = _ROOT / "analysis" / "diagnostic" / "hypothesis" / "hypothesis_tests.py"
_spec = importlib.util.spec_from_file_location("hypothesis_tests", _HYP_PATH)
assert _spec and _spec.loader
ht = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ht)


def test_region_anova_detects_shift() -> None:
    rng = np.random.default_rng(0)
    low = rng.normal(1.0, 0.1, size=40)
    high = rng.normal(8.0, 0.2, size=40)
    df = pd.DataFrame(
        {
            "Region": ["A"] * 40 + ["B"] * 40,
            "Num_Attacks": np.concatenate([low, high]),
            "Attack_Type_Mode": ["X"] * 80,
        }
    )
    f_stat, p_anova, _h, p_kw, names = ht.compute_region_attack_tests(df)
    assert set(names) == {"A", "B"}
    assert f_stat > 0
    assert p_anova < 0.001
    assert p_kw < 0.001


def test_attack_type_gof_uniform_fails_when_skewed() -> None:
    df = pd.DataFrame(
        {
            "Region": ["R"] * 100,
            "Num_Attacks": [1] * 50 + [50] * 50,
            "Attack_Type_Mode": ["Mild"] * 50 + ["Severe"] * 50,
        }
    )
    chi2, p_gof, df_freedom, totals = ht.compute_attack_type_gof(df)
    assert df_freedom == 1
    assert totals.sum() == df["Num_Attacks"].sum()
    assert chi2 > 0
    assert p_gof < 0.05


def test_contingency_independence_vs_association() -> None:
    n = 60
    df_ok = pd.DataFrame(
        {
            "Region": ["N"] * n + ["S"] * n,
            "Num_Attacks": [1] * (2 * n),
            "Attack_Type_Mode": (["A"] * (n // 2) + ["B"] * (n // 2)) * 2,
        }
    )
    _chi2, p, dof, shape = ht.compute_region_attacktype_independence(df_ok)
    assert shape == (2, 2)
    assert dof == 1
    assert p > 0.05

    df_assoc = pd.DataFrame(
        {
            "Region": ["N"] * n + ["S"] * n,
            "Num_Attacks": [1] * (2 * n),
            "Attack_Type_Mode": ["A"] * n + ["B"] * n,
        }
    )
    _chi2b, p2, _dof2, _shape2 = ht.compute_region_attacktype_independence(df_assoc)
    assert p2 < 1e-10
