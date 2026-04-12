"""
Associational "driver" analysis for Num_Attacks:
  - Group summaries by Region (and attack type)
  - Linear regression coefficients after scaling numerics + one-hot Region;
    rank by absolute coefficient to highlight strongest linear predictors.

This does not establish causal root causes (confounding, reverse causation, panel structure).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"
OUT_DIR = Path(__file__).resolve().parent

TARGET = "Num_Attacks"
NUMERIC_FEATURES = [
    "GDP_per_capita",
    "Population",
    "Conflict_Level",
    "Fatalities",
    "Injuries",
    "Year",
    "Conflict",
]
CATEGORICAL_FEATURES = ["Region"]


def load_frame() -> pd.DataFrame:
    usecols = [TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "Attack_Type_Mode"]
    usecols = list(dict.fromkeys(usecols))
    df = pd.read_csv(DATA_PATH, usecols=usecols).dropna()
    return df


def region_group_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("Region", observed=True)
        .agg(
            n_rows=(TARGET, "count"),
            mean_attacks=(TARGET, "mean"),
            median_attacks=(TARGET, "median"),
            mean_conflict_level=("Conflict_Level", "mean"),
            share_conflict=("Conflict", "mean"),
        )
        .sort_values("mean_attacks", ascending=False)
    )
    return g


def attack_type_group_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Attack_Type_Mode", observed=True)
        .agg(
            n_rows=(TARGET, "count"),
            mean_attacks=(TARGET, "mean"),
            median_attacks=(TARGET, "median"),
        )
        .sort_values("mean_attacks", ascending=False)
    )


def fit_scaled_linear_with_region(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        [("prep", preprocessor), ("lin", LinearRegression())]
    )
    model.fit(X, y)

    feature_names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["lin"].coef_
    ranking = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    return model, ranking


def main() -> None:
    df = load_frame()

    region_stats = region_group_table(df)
    type_stats = attack_type_group_table(df)
    _model, ranking = fit_scaled_linear_with_region(df)

    region_stats.to_csv(OUT_DIR / "region_group_stats.csv")
    type_stats.to_csv(OUT_DIR / "attack_type_group_stats.csv")
    ranking.to_csv(OUT_DIR / "linear_coef_ranking.csv", index=False)

    print(f"Rows used: {len(df):,}")
    print("\n=== Group comparison: Region (mean Num_Attacks, conflict proxies) ===")
    print(region_stats.round(4).to_string())
    print("\n=== Group comparison: Attack_Type_Mode (modal type in row) ===")
    print(type_stats.round(4).head(15).to_string())
    print("\n=== Linear model: standardized numerics + Region one-hot (full sample) ===")
    print(f"Intercept: { _model.named_steps['lin'].intercept_:.6f}")
    print(f"R^2 (in-sample): {_model.score(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]):.6f}")
    print("\nRanked |coefficient| (stronger linear association on scaled numerics / OHE):")
    print(ranking.to_string(index=False))
    print("\nStrongest single term (by |coef|):")
    top = ranking.iloc[0]
    print(f"  {top['feature']}: {top['coefficient']:.6f}")

    print("\nCSV outputs:")
    print(f"  {OUT_DIR / 'region_group_stats.csv'}")
    print(f"  {OUT_DIR / 'attack_type_group_stats.csv'}")
    print(f"  {OUT_DIR / 'linear_coef_ranking.csv'}")


if __name__ == "__main__":
    main()
