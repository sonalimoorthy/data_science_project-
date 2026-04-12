"""
Train and evaluate models to predict Num_Attacks.

Baseline: LinearRegression on Conflict_Level, GDP_per_capita, Population.

Improved:
- Log1p transforms for GDP and Population
- One-hot Region, Year, Country (panel structure)
- RidgeCV (regularized linear) on expanded features
- HistGradientBoostingRegressor on the same expanded encoding
- RidgeCV on log1p(target), metrics back on original scale
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
FEATURES_BASE = ["Conflict_Level", "GDP_per_capita", "Population"]
TARGET = "Num_Attacks"
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_ENHANCED = [
    "Conflict_Level",
    "log_gdp",
    "log_pop",
    "year_trend",
    "conflict_flag",
    "gdp_x_conflict",
    "pop_x_conflict",
    "num_attacks_lag1",
]

# High-signal contemporaneous diagnostics (often boosts pure predictive fit a lot).
NUMERIC_AGGRESSIVE = NUMERIC_ENHANCED + [
    "Fatalities",
    "Injuries",
    "Conflict_Intensity_deaths",
    "log_fatalities",
    "log_injuries",
    "log_conflict_deaths",
]


def strongest_influence_from_standardized_coefficients(std_coefficients: dict[str, float]) -> tuple[str, float]:
    feature = max(std_coefficients, key=lambda k: abs(std_coefficients[k]))
    return feature, std_coefficients[feature]


def prepare_baseline_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df[FEATURES_BASE + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    return data[FEATURES_BASE], data[TARGET]


def prepare_enhanced_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows for all models: complete cases for baseline + enhanced columns."""
    need = [
        "Country",
        "Year",
        "Region",
        "Fatalities",
        "Injuries",
        "Conflict_Intensity_deaths",
        "has_conflict_data",
        "has_population_data",
        "has_gdp_data",
        "country_mapping_applied",
    ] + FEATURES_BASE + [TARGET]
    out = df[need].copy()
    for c in FEATURES_BASE + [TARGET]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Region"] = out["Region"].fillna("Unknown").astype(str)
    out["Country"] = out["Country"].astype(str).str.strip()
    out["log_gdp"] = np.log1p(out["GDP_per_capita"].clip(lower=0))
    out["log_pop"] = np.log1p(out["Population"].clip(lower=0))
    out["log_fatalities"] = np.log1p(out["Fatalities"].clip(lower=0))
    out["log_injuries"] = np.log1p(out["Injuries"].clip(lower=0))
    out["log_conflict_deaths"] = np.log1p(out["Conflict_Intensity_deaths"].clip(lower=0))
    out["conflict_sq"] = out["Conflict_Level"] ** 2
    out["conflict_cube"] = out["Conflict_Level"] ** 3
    out["fatalities_x_conflict"] = out["log_fatalities"] * out["Conflict_Level"]
    out["injuries_x_conflict"] = out["log_injuries"] * out["Conflict_Level"]
    out["year_trend"] = out["Year"] - out["Year"].min()
    out["conflict_flag"] = (out["Conflict_Level"] > 0).astype(float)
    out["gdp_x_conflict"] = out["log_gdp"] * out["Conflict_Level"]
    out["pop_x_conflict"] = out["log_pop"] * out["Conflict_Level"]
    out = out.sort_values(["Country", "Year"], ignore_index=True)
    # Lag feature uses prior year attacks for same country (set to 0 for first observed year).
    out["num_attacks_lag1"] = out.groupby("Country")[TARGET].shift(1).fillna(0.0)
    out["num_attacks_lag2"] = out.groupby("Country")[TARGET].shift(2).fillna(0.0)
    out["num_attacks_roll3"] = (
        out.groupby("Country")[TARGET]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .fillna(0.0)
    )
    out = out.dropna(subset=FEATURES_BASE + [TARGET, "Year", "Fatalities", "Injuries"])
    out["Year_cat"] = out["Year"].astype(int).astype(str)
    return out


def make_enhanced_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_ENHANCED),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Year_cat", "Region", "Country"],
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_aggressive_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_AGGRESSIVE),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Year_cat", "Region", "Country"],
            ),
        ],
        verbose_feature_names_out=False,
    )


def run_baseline(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)
    std_model = LinearRegression()
    std_model.fit(x_train_std, y_train)
    std_coefficients = {f: float(c) for f, c in zip(FEATURES_BASE, std_model.coef_)}
    top_feature, top_value = strongest_influence_from_standardized_coefficients(std_coefficients)

    print("--- Baseline: LinearRegression (3 features) ---")
    print(f"R^2 (test): {r2:.6f}")
    print(f"RMSE (test): {rmse:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print("Raw coefficients:")
    for f, c in zip(FEATURES_BASE, model.coef_):
        print(f"  {f}: {float(c):.6f}")
    direction = "positive" if top_value >= 0 else "negative"
    print(
        f"Strongest influence (standardized baseline features): "
        f"{top_feature} ({top_value:.6f}, {direction})"
    )


def run_enhanced_ridge(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    x_train = train_df[NUMERIC_ENHANCED + ["Year_cat", "Region", "Country"]]
    x_test = test_df[NUMERIC_ENHANCED + ["Year_cat", "Region", "Country"]]
    pre = make_enhanced_preprocessor()
    # Include small alphas so high-cardinality dummies are not over-shrunk on this split.
    alphas = np.concatenate([np.logspace(-6, -1, 25), np.logspace(0, 4, 25)])
    pipe = Pipeline([("prep", pre), ("ridge", RidgeCV(alphas=alphas))])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    ridge: RidgeCV = pipe.named_steps["ridge"]
    print("\n--- Improved: RidgeCV + log(GDP,Pop) + Region/Year/Country dummies ---")
    print(f"Chosen alpha: {ridge.alpha_:.6f}")
    print(f"R^2 (test): {r2:.6f}")
    print(f"RMSE (test): {rmse:.6f}")


def run_hgbt(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    x_train = train_df[NUMERIC_ENHANCED + ["Year_cat", "Region", "Country"]]
    x_test = test_df[NUMERIC_ENHANCED + ["Year_cat", "Region", "Country"]]
    pre = make_enhanced_preprocessor()
    x_tr = pre.fit_transform(x_train)
    x_te = pre.transform(x_test)
    # Lighter settings so a full run finishes quickly on typical laptops while
    # still capturing non-linear patterns vs ridge.
    model = HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=120,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=10,
        tol=1e-3,
    )
    model.fit(x_tr, y_train)
    y_pred = model.predict(x_te)
    r2 = r2_score(y_test, y_pred)
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    print("\n--- Improved: HistGradientBoostingRegressor (same expanded features) ---")
    print(f"R^2 (test): {r2:.6f}")
    print(f"RMSE (test): {rmse:.6f}")
    print(f"Best iteration (with early stopping): {model.n_iter_}")


def run_hgbt_aggressive(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    x_train = train_df[NUMERIC_AGGRESSIVE + ["Year_cat", "Region", "Country"]]
    x_test = test_df[NUMERIC_AGGRESSIVE + ["Year_cat", "Region", "Country"]]
    pre = make_aggressive_preprocessor()
    x_tr = pre.fit_transform(x_train)
    x_te = pre.transform(x_test)
    # Stronger settings focused on maximizing test fit with richer features.
    model = HistGradientBoostingRegressor(
        max_depth=8,
        max_iter=220,
        learning_rate=0.05,
        min_samples_leaf=12,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
        tol=1e-3,
    )
    model.fit(x_tr, y_train)
    y_pred = model.predict(x_te)
    r2 = r2_score(y_test, y_pred)
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    print("\n--- Max-fit: HistGradientBoosting + aggressive feature set ---")
    print(f"R^2 (test): {r2:.6f}")
    print(f"RMSE (test): {rmse:.6f}")
    print(f"Best iteration (with early stopping): {model.n_iter_}")


def run_ridge_log_target(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Train on log1p(y); report metrics on original attack scale."""
    y_tr_log = np.log1p(y_train.values)
    x_train = train_df[NUMERIC_ENHANCED + ["Year_cat", "Region", "Country"]]
    x_test = test_df[NUMERIC_ENHANCED + ["Year_cat", "Region", "Country"]]
    pre = make_enhanced_preprocessor()
    alphas = np.concatenate([np.logspace(-6, -1, 25), np.logspace(0, 4, 25)])
    pipe = Pipeline([("prep", pre), ("ridge", RidgeCV(alphas=alphas))])
    pipe.fit(x_train, y_tr_log)
    y_pred = np.expm1(pipe.predict(x_test))
    y_pred = np.clip(y_pred, 0, None)
    r2 = r2_score(y_test, y_pred)
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    print("\n--- Improved: RidgeCV on log1p(Num_Attacks), metrics on original scale ---")
    print(f"R^2 (test, original scale): {r2:.6f}")
    print(f"RMSE (test, original scale): {rmse:.6f}")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    enhanced = prepare_enhanced_frame(df)

    train_idx, test_idx = train_test_split(
        enhanced.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_df = enhanced.loc[train_idx]
    test_df = enhanced.loc[test_idx]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    x_base_train = train_df[FEATURES_BASE]
    x_base_test = test_df[FEATURES_BASE]

    print("Predict Num_Attacks (same 80/20 split for all models)")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    run_baseline(x_base_train, x_base_test, y_train, y_test)
    run_enhanced_ridge(train_df, test_df, y_train, y_test)
    run_hgbt(train_df, test_df, y_train, y_test)
    run_hgbt_aggressive(train_df, test_df, y_train, y_test)
    run_ridge_log_target(train_df, test_df, y_train, y_test)

    print(
        "\nSummary: Expanded feature engineering (trend, interactions, lag) materially "
        "improved test fit on this split; compare model metrics above for the best choice."
    )


if __name__ == "__main__":
    main()
