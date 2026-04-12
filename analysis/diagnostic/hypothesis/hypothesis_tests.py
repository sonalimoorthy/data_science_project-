"""
Hypothesis tests on master_country_year.csv:
  - Region vs Num_Attacks: one-way ANOVA + Kruskal-Wallis (non-parametric)
  - Attack type: chi-square goodness-of-fit vs uniform distribution of total attacks
  - Region x Attack type: chi-square test of independence (row = country-year records)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"


def load_panel() -> pd.DataFrame:
    usecols = ["Region", "Num_Attacks", "Attack_Type_Mode"]
    return pd.read_csv(DATA_PATH, usecols=usecols).dropna()


def compute_region_attack_tests(
    df: pd.DataFrame,
) -> tuple[float, float, float, float, list[str]]:
    groups = [g["Num_Attacks"].values for _, g in df.groupby("Region", sort=True)]
    region_names = [name for name, _ in df.groupby("Region", sort=True)]
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kw = stats.kruskal(*groups)
    return f_stat, p_anova, h_stat, p_kw, region_names


def compute_attack_type_gof(
    df: pd.DataFrame,
) -> tuple[float, float, int, pd.Series]:
    totals = df.groupby("Attack_Type_Mode", observed=True)["Num_Attacks"].sum().sort_index()
    observed = totals.values.astype(float)
    n_cat = len(observed)
    expected = np.full(n_cat, observed.sum() / n_cat)
    chi2, p_gof = stats.chisquare(observed, expected)
    return chi2, p_gof, n_cat - 1, totals


def compute_region_attacktype_independence(df: pd.DataFrame) -> tuple[float, float, int, tuple[int, int]]:
    tab = pd.crosstab(df["Region"], df["Attack_Type_Mode"])
    chi2, p, dof, _expected = stats.chi2_contingency(tab)
    return chi2, p, dof, tab.shape


def main() -> None:
    df = load_panel()

    f_stat, p_anova, h_stat, p_kw, region_names = compute_region_attack_tests(df)
    print("=== A. Region vs attack frequency (Num_Attacks per country-year row) ===")
    print("H0: Mean Num_Attacks is the same across all regions.")
    print("H1: At least one region has a different mean Num_Attacks.")
    print(f"  One-way ANOVA  F = {f_stat:.6f},  p-value = {p_anova:.6e}")
    print(
        "  Kruskal-Wallis  H = {:.6f},  p-value = {:.6e}  (rank-based; robust to skew)".format(
            h_stat, p_kw
        )
    )
    print(f"  Regions (k = {len(region_names)}): {', '.join(region_names)}")
    print(f"  N rows: {len(df):,}")

    chi2_gof, p_gof, df_gof, totals = compute_attack_type_gof(df)
    print("\n=== B. Attack type distribution (total Num_Attacks by Attack_Type_Mode) ===")
    print("H0: Total attacks are evenly distributed across attack types (uniform).")
    print("H1: Some attack types account for a disproportionate share of attacks.")
    print(f"  Chi-square goodness-of-fit  chi2 = {chi2_gof:.4f},  df = {df_gof},  p-value = {p_gof:.6e}")
    print("  Observed totals (attacks) by type:")
    for name, val in totals.items():
        print(f"    {name}: {int(val):,}")

    chi2_ct, p_ct, dof_ct, shape = compute_region_attacktype_independence(df)
    print("\n=== C. Region x Attack type (independence on country-year rows) ===")
    print("H0: Region and attack type are independent (no association in the table).")
    print("H1: There is a statistical association between region and attack type.")
    print(f"  Chi-square contingency  chi2 = {chi2_ct:.4f},  df = {dof_ct},  p-value = {p_ct:.6e}")
    print(f"  Table shape: {shape[0]} regions x {shape[1]} attack types")


if __name__ == "__main__":
    main()
