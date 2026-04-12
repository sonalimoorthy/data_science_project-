"""
Regression analysis for Num_Attacks:
  - Linear baselines (5 features, +Year, +Region OHE)
  - PolynomialFeatures(degree=2) + LinearRegression (numeric 5 + Year, scaled)
  - HistGradientBoosting on enriched numeric + categorical encodings (stronger fit)
  - Same model with log1p target transform (often better RMSE on skewed counts)
  - Time-only linear / quadratic on Year for coarse trend read

Same train/test split (80/20, random_state=42) for all models.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"

TARGET = "Num_Attacks"
FEATURES_BASE = [
    "GDP_per_capita",
    "Population",
    "Conflict_Level",
    "Fatalities",
    "Injuries",
]
FEATURES_WITH_YEAR = [*FEATURES_BASE, "Year"]

NUMERIC_EXTRA = [
    "Conflict",
    "Conflict_Intensity_deaths",
    "has_conflict_data",
    "has_population_data",
    "has_gdp_data",
]
CATEGORICAL_FOR_TREE = ["Region", "Attack_Type_Mode", "Target_Type_Mode"]
NUMERIC_FOR_TREE = [*FEATURES_WITH_YEAR, *NUMERIC_EXTRA]

USECOLS = list(
    dict.fromkeys(
        [
            TARGET,
            *FEATURES_WITH_YEAR,
            *CATEGORICAL_FOR_TREE,
            *NUMERIC_EXTRA,
        ]
    )
)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    clip_nonnegative: bool = False,
) -> dict:
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    if clip_nonnegative:
        pred_train = np.clip(pred_train, 0, None)
        pred_test = np.clip(pred_test, 0, None)
    return {
        "model": name,
        "r2_train": r2_score(y_train, pred_train),
        "r2_test": r2_score(y_test, pred_test),
        "rmse_train": rmse(y_train, pred_train),
        "rmse_test": rmse(y_test, pred_test),
    }


def make_gradient_boosting_pipeline(*, log_target: bool) -> Pipeline | TransformedTargetRegressor:
    prep = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FOR_TREE,
            ),
        ],
        remainder="passthrough",
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=12,
        learning_rate=0.05,
        l2_regularization=0.3,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    pipe = Pipeline([("prep", prep), ("hgb", hgb)])
    if not log_target:
        return pipe
    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories",
        category=UserWarning,
    )

    df = pd.read_csv(DATA_PATH, usecols=USECOLS)
    df["Conflict_Intensity_deaths"] = df["Conflict_Intensity_deaths"].fillna(0)
    df = df.dropna()

    y = df[TARGET]
    region_dummies = pd.get_dummies(df["Region"], prefix="Region", drop_first=True)
    X_tab = pd.concat([df[FEATURES_WITH_YEAR], region_dummies], axis=1)
    X_tree = df[NUMERIC_FOR_TREE + CATEGORICAL_FOR_TREE]

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=0.2, random_state=42
    )
    X_tree_train = X_tree.loc[X_train.index]
    X_tree_test = X_tree.loc[X_test.index]

    rows = []

    lin5 = LinearRegression()
    rows.append(
        evaluate(
            "Linear (5 features, no Year)",
            lin5,
            X_train[FEATURES_BASE],
            X_test[FEATURES_BASE],
            y_train,
            y_test,
        )
    )

    lin6 = LinearRegression()
    rows.append(
        evaluate(
            "Linear (5 features + Year)",
            lin6,
            X_train[FEATURES_WITH_YEAR],
            X_test[FEATURES_WITH_YEAR],
            y_train,
            y_test,
        )
    )

    lin_region = LinearRegression()
    rows.append(
        evaluate(
            "Linear (5 + Year + Region OHE)",
            lin_region,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    poly2 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression()),
        ]
    )
    rows.append(
        evaluate(
            "Polynomial deg=2 (5 + Year, scaled)",
            poly2,
            X_train[FEATURES_WITH_YEAR],
            X_test[FEATURES_WITH_YEAR],
            y_train,
            y_test,
        )
    )

    rows.append(
        evaluate(
            "HistGradientBoosting (enriched + OHE, raw target)",
            make_gradient_boosting_pipeline(log_target=False),
            X_tree_train,
            X_tree_test,
            y_train,
            y_test,
            clip_nonnegative=True,
        )
    )

    rows.append(
        evaluate(
            "HistGradientBoosting (enriched + OHE, log1p target)",
            make_gradient_boosting_pipeline(log_target=True),
            X_tree_train,
            X_tree_test,
            y_train,
            y_test,
            clip_nonnegative=True,
        )
    )

    summary = pd.DataFrame(rows)

    year_train = X_train[["Year"]]
    year_test = X_test[["Year"]]

    time_lin = LinearRegression()
    time_lin.fit(year_train, y_train)

    time_poly = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression()),
        ]
    )
    time_poly.fit(year_train, y_train)

    poly_step = time_poly.named_steps["poly"]
    feat_names = list(poly_step.get_feature_names_out(["Year"]))
    time_poly_coefs = pd.DataFrame(
        {"term": feat_names, "coefficient": time_poly.named_steps["lin"].coef_}
    )

    print(f"Rows used (complete cases): {len(df):,}")
    print(f"Train size: {len(X_train):,}  Test size: {len(X_test):,}")
    print("\n=== Model comparison (test metrics are primary) ===")
    print(
        summary[
            ["model", "r2_test", "rmse_test", "r2_train", "rmse_train"]
        ].to_string(index=False)
    )

    print("\n=== Linear (5 + Year): coefficients (raw-scale features) ===")
    coef6 = pd.DataFrame(
        {
            "feature": ["intercept", *FEATURES_WITH_YEAR],
            "coefficient": np.concatenate([[lin6.intercept_], lin6.coef_]),
        }
    )
    print(coef6.to_string(index=False))

    print("\n=== Time-only: linear Num_Attacks ~ Year (raw Year) ===")
    print(f"  intercept: {time_lin.intercept_:.6f}")
    print(f"  Year coef: {time_lin.coef_[0]:.6f}  (positive -> higher years predict more attacks)")

    print("\n=== Time-only: quadratic Num_Attacks ~ Year + Year^2 (raw Year) ===")
    print(f"  intercept: {time_poly.named_steps['lin'].intercept_:.6f}")
    print(time_poly_coefs.to_string(index=False))
    r2_time_lin = r2_score(y_test, time_lin.predict(year_test))
    r2_time_poly = r2_score(y_test, time_poly.predict(year_test))
    print(f"\n  R^2 (test) time-only linear:    {r2_time_lin:.6f}")
    print(f"  R^2 (test) time-only quadratic: {r2_time_poly:.6f}")

    year_sq_coef = time_poly_coefs.loc[
        time_poly_coefs["term"] == "Year^2", "coefficient"
    ]
    year_sq_val = float(year_sq_coef.iloc[0]) if len(year_sq_coef) else float("nan")

    print("\n=== Plain-language answers (time-only models, descriptive) ===")
    if time_lin.coef_[0] > 0:
        print(
            "- Attacks vs calendar year (linear fit on train): the slope on Year is positive, "
            "so a straight-line summary of the global panel trend is *upward* over time."
        )
    else:
        print(
            "- Attacks vs calendar year (linear fit on train): the slope on Year is not positive "
            "in this crude time-only model."
        )

    if np.isfinite(year_sq_val) and year_sq_val > 0:
        print(
            "- Acceleration: the time-only quadratic term Year^2 has a *positive* coefficient, "
            "so the fitted curve bends *upward* (growth speeding up in that simple parabola)."
        )
    elif np.isfinite(year_sq_val) and year_sq_val < 0:
        print(
            "- Acceleration: Year^2 is *negative*, so the simple parabola bends *downward* "
            "(eventually slower growth or decline in that toy fit)."
        )
    else:
        print("- Acceleration: could not read Year^2 coefficient.")

    print(
        "\nNote: Time-only models ignore country, GDP, conflict, etc. Multivariate models above "
        "are better for *prediction*; time-only fits help read coarse global time shape only."
    )
    print(
        "\nBest test metrics here usually come from HistGradientBoosting with extra columns and "
        "optionally a log1p target (skewed counts); that is still associational / predictive, not causal."
    )


if __name__ == "__main__":
    main()
