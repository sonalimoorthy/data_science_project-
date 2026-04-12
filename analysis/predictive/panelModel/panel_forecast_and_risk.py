"""
Country-year panel: engineered features, direct multi-horizon forecasting (RF vs Ridge),
terrorism risk classification, feature importance, country-level errors, KMeans on country profiles.

Regression targets may be trained on ``log1p(Num_Attacks)`` with ``expm1`` inverse; log-space
predictions are clipped to ``[0, log1p(max training y)]`` before inverse so linear models stay
stable on the original count scale.

Data: data/final/master_country_year.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Regression: train through this year; test windows chosen so y_plus10 exists (max Year 2017).
TRAIN_MAX_YEAR_REG = 2005
# Classification: more recent split for relevance.
TRAIN_MAX_YEAR_CLF = 2010
RANDOM_STATE = 42


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def target_log1p(y) -> np.ndarray:
    """y' = log(Num_Attacks + 1); stable for nonnegative counts."""
    return np.log1p(np.asarray(y, dtype=float))


def inverse_target_log1p(y_log) -> np.ndarray:
    """Inverse of log1p; clip at 0 for nonnegative attack counts."""
    return np.maximum(0.0, np.expm1(np.asarray(y_log, dtype=float)))


def require_panel_csv(path: Path = DATA_PATH) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Panel data CSV not found: {path}. "
            "Expected data/final/master_country_year.csv under the repo root."
        )


def load_raw() -> pd.DataFrame:
    require_panel_csv()
    cols = [
        "Country",
        "Year",
        "Region",
        "Num_Attacks",
        "Fatalities",
        "Injuries",
        "Conflict_Level",
        "GDP_per_capita",
        "Population",
        "Conflict",
    ]
    df = pd.read_csv(DATA_PATH, usecols=cols).dropna(
        subset=["Country", "Year", "Region", "Num_Attacks"]
    )
    return df


def expand_balanced_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Each (Country, Year) for full calendar span; missing attack years -> 0; other cols ffill/bfill."""
    y0, y1 = int(df["Year"].min()), int(df["Year"].max())
    years = pd.RangeIndex(y0, y1 + 1)
    countries = np.sort(df["Country"].unique())
    mi = pd.MultiIndex.from_product([countries, years], names=["Country", "Year"])
    region_map = df.groupby("Country")["Region"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]
    )
    base = df.set_index(["Country", "Year"]).sort_index()
    out = base.reindex(mi)
    out["Num_Attacks"] = out["Num_Attacks"].fillna(0.0)
    out = out.reset_index()
    out["Region"] = out["Country"].map(region_map).fillna("Unknown")

    num_fill = [
        "Fatalities",
        "Injuries",
        "Conflict_Level",
        "GDP_per_capita",
        "Population",
        "Conflict",
    ]
    for c in num_fill:
        if c not in out.columns:
            continue
        out[c] = out.groupby("Country", sort=False)[c].transform(
            lambda s: s.ffill().bfill()
        )
        out[c] = out[c].fillna(0.0)

    return out.sort_values(["Country", "Year"]).reset_index(drop=True)


def add_panel_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["Country", "Year"]).copy()
    p["num_attacks_lag1"] = (
        p.groupby("Country", sort=False)["Num_Attacks"].shift(1).fillna(0.0)
    )
    p["num_attacks_lag2"] = (
        p.groupby("Country", sort=False)["Num_Attacks"].shift(2).fillna(0.0)
    )
    p["num_attacks_roll3"] = (
        p.groupby("Country", sort=False)["Num_Attacks"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )
    g = p.groupby("Country", sort=False)
    p["gdp_per_capita_lag1"] = g["GDP_per_capita"].shift(1)
    p["population_lag1"] = g["Population"].shift(1)
    p["conflict_level_lag1"] = g["Conflict_Level"].shift(1)
    p["conflict_lag1"] = g["Conflict"].shift(1)
    for c in ("gdp_per_capita_lag1", "population_lag1", "conflict_level_lag1", "conflict_lag1"):
        p[c] = p[c].fillna(0.0)
    p["log_gdp_lag1"] = np.log1p(p["gdp_per_capita_lag1"].clip(lower=0))
    p["log_pop_lag1"] = np.log1p(p["population_lag1"].clip(lower=0))
    p["gdp_x_conflict_lag1"] = p["gdp_per_capita_lag1"] * p["conflict_level_lag1"]

    p["y_plus1"] = p.groupby("Country", sort=False)["Num_Attacks"].shift(-1)
    p["y_plus5"] = p.groupby("Country", sort=False)["Num_Attacks"].shift(-5)
    p["y_plus10"] = p.groupby("Country", sort=False)["Num_Attacks"].shift(-10)
    return p


def feature_columns() -> tuple[list[str], list[str]]:
    # No same-year harm or same-year attack consequences (Fatalities/Injuries/etc.).
    # Structural covariates enter only with a one-year lag within country.
    numeric = [
        "num_attacks_lag1",
        "num_attacks_lag2",
        "num_attacks_roll3",
        "log_gdp_lag1",
        "log_pop_lag1",
        "gdp_x_conflict_lag1",
        "conflict_level_lag1",
        "conflict_lag1",
        "Year",
    ]
    categorical = ["Region"]
    return numeric, categorical


def make_preprocessors(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                categorical,
            ),
        ]
    )


def random_forest_regressor_pipe(
    numeric: list[str], categorical: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessors(numeric, categorical)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_eval_direct_horizon(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    numeric: list[str],
    categorical: list[str],
    log_target: bool = False,
) -> dict:
    X_tr = train_df[numeric + categorical]
    X_te = test_df[numeric + categorical]
    y_tr = train_df[target_col].astype(float)
    y_te = test_df[target_col].astype(float)

    y_tr_fit = target_log1p(y_tr) if log_target else y_tr
    log_hi = float(np.log1p(np.max(y_tr))) if log_target else None

    rf = random_forest_regressor_pipe(numeric, categorical)
    rf.fit(X_tr, y_tr_fit)
    pred_rf_raw = rf.predict(X_te)
    if log_target:
        pred_rf_raw = np.clip(pred_rf_raw, 0.0, log_hi)
    pred_rf = (
        inverse_target_log1p(pred_rf_raw) if log_target else pred_rf_raw
    )

    prep_r = make_preprocessors(numeric, categorical)
    ridge = Pipeline(
        [
            ("prep", prep_r),
            (
                "ridge",
                RidgeCV(alphas=np.logspace(-3, 4, 25)),
            ),
        ]
    )
    ridge.fit(X_tr, y_tr_fit)
    pred_ridge_raw = ridge.predict(X_te)
    if log_target:
        pred_ridge_raw = np.clip(pred_ridge_raw, 0.0, log_hi)
    pred_ridge = (
        inverse_target_log1p(pred_ridge_raw) if log_target else pred_ridge_raw
    )

    return {
        "rmse_rf": rmse(y_te, pred_rf),
        "rmse_ridge": rmse(y_te, pred_ridge),
        "rf_pipe": rf,
        "ridge_pipe": ridge,
        "y_test": y_te,
        "pred_rf": pred_rf,
        "test_df": test_df,
    }


def train_eval_grouped_cv(
    panel: pd.DataFrame,
    target_col: str,
    numeric: list[str],
    categorical: list[str],
    n_splits: int = 5,
    log_target: bool = False,
) -> dict | None:
    """GroupKFold by Country: no country appears in both train and test in a fold."""
    df = panel.dropna(subset=numeric + [target_col])
    n_countries = df["Country"].nunique()
    k = min(n_splits, n_countries)
    if k < 2:
        return None
    X = df[numeric + categorical]
    y = df[target_col].astype(float)
    groups = df["Country"].values
    gkf = GroupKFold(n_splits=k)
    fold_rf: list[float] = []
    fold_ridge: list[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        y_tr_fit = target_log1p(y_tr) if log_target else y_tr
        log_hi = float(np.log1p(np.max(y_tr))) if log_target else None
        rf = random_forest_regressor_pipe(numeric, categorical)
        rf.fit(X_tr, y_tr_fit)
        pred_rf_raw = rf.predict(X_te)
        if log_target:
            pred_rf_raw = np.clip(pred_rf_raw, 0.0, log_hi)
        pred_rf = (
            inverse_target_log1p(pred_rf_raw) if log_target else pred_rf_raw
        )
        prep_r = make_preprocessors(numeric, categorical)
        ridge = Pipeline(
            [
                ("prep", prep_r),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 4, 25))),
            ]
        )
        ridge.fit(X_tr, y_tr_fit)
        pred_ridge_raw = ridge.predict(X_te)
        if log_target:
            pred_ridge_raw = np.clip(pred_ridge_raw, 0.0, log_hi)
        pred_ridge = (
            inverse_target_log1p(pred_ridge_raw) if log_target else pred_ridge_raw
        )
        fold_rf.append(rmse(y_te, pred_rf))
        fold_ridge.append(rmse(y_te, pred_ridge))
    return {
        "n_splits": k,
        "n_rows": len(df),
        "rmse_rf_mean": float(np.mean(fold_rf)),
        "rmse_rf_std": float(np.std(fold_rf)),
        "rmse_ridge_mean": float(np.mean(fold_ridge)),
        "rmse_ridge_std": float(np.std(fold_ridge)),
    }


def plot_feature_importance(
    rf_pipe: Pipeline,
    top_n: int,
    out_path: Path,
    *,
    title: str | None = None,
) -> None:
    model = rf_pipe.named_steps["model"]
    prep = rf_pipe.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(top_n), imp[order][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([names[i] for i in order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(
        title or f"Random forest: top {top_n} features (direct t+1 model)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Risk classifier confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def country_level_rmse(te: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    err = pd.DataFrame(
        {
            "Country": te["Country"].values,
            "sq_err": (y_true.values - y_pred) ** 2,
        }
    )
    return (
        err.groupby("Country")["sq_err"]
        .mean()
        .apply(np.sqrt)
        .sort_values(ascending=False)
        .rename("rmse_h1")
        .reset_index()
    )


def safe_country_filename(country: str) -> str:
    bad = '\\/:*?"<>|'
    s = "".join("_" if c in bad else c for c in str(country)).strip()
    return (s[:120] if s else "unknown").replace(" ", "_")


def plot_country_history_and_forecasts(
    country: str,
    years_hist: np.ndarray,
    y_hist: np.ndarray,
    forecast_years: list[int],
    forecast_vals: list[float],
    anchor_year: int,
    out_path: Path,
    *,
    log_target: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(years_hist, y_hist)
    ax.axvline(anchor_year, linestyle="--", alpha=0.75)
    fv = np.clip(np.asarray(forecast_vals, dtype=float), 0, None)
    ax.scatter(forecast_years, fv, zorder=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Num_Attacks")
    title_suffix = " (log target, expm1 inverse)" if log_target else ""
    ax.set_title(f"{country}: history to {anchor_year} + RF direct forecasts{title_suffix}")
    ax.text(
        0.02,
        0.98,
        "Markers: out-of-sample projection (not observed)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fit_deploy_rf_models(
    panel: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    max_train_year: int,
    log_target: bool = False,
) -> tuple[dict[str, Pipeline], dict[str, float]]:
    """Train one RF per horizon using only rows up to max_train_year (no future-year fit).

    When ``log_target`` is True, returns per-horizon upper bounds ``log1p(max train y)`` for
    clipping linear-model-style extrapolation before ``expm1`` (used by forward predictions).
    """
    base = panel.loc[panel["Year"] <= max_train_year]
    out: dict[str, Pipeline] = {}
    log_clip_hi: dict[str, float] = {}
    for key, col in [("h1", "y_plus1"), ("h5", "y_plus5"), ("h10", "y_plus10")]:
        tr = base.dropna(subset=[col] + numeric)
        pipe = random_forest_regressor_pipe(numeric, categorical)
        y_raw = tr[col].astype(float)
        y_fit = target_log1p(y_raw) if log_target else y_raw
        pipe.fit(tr[numeric + categorical], y_fit)
        out[key] = pipe
        if log_target:
            log_clip_hi[key] = float(np.log1p(np.max(y_raw)))
    return out, log_clip_hi


def run_forward_forecasts_and_plots(
    panel: pd.DataFrame,
    deploy: dict[str, Pipeline],
    numeric: list[str],
    categorical: list[str],
    anchor_year: int,
    plot_dir: Path,
    log_target: bool = False,
    log_clip_hi: dict[str, float] | None = None,
) -> pd.DataFrame:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fyears = [anchor_year + 1, anchor_year + 5, anchor_year + 10]
    rows = []
    for country in sorted(panel["Country"].unique()):
        row_df = panel[
            (panel["Country"] == country) & (panel["Year"] == anchor_year)
        ]
        if row_df.empty:
            continue
        Xb = row_df.iloc[:1][numeric + categorical]

        def _pred_inv(key: str, pipe: Pipeline) -> float:
            raw = float(pipe.predict(Xb)[0])
            if not log_target:
                return raw
            hi = (log_clip_hi or {}).get(key)
            if hi is not None:
                raw = float(np.clip(raw, 0.0, hi))
            return float(np.maximum(0.0, np.expm1(raw)))

        p1 = max(0.0, _pred_inv("h1", deploy["h1"]))
        p5 = max(0.0, _pred_inv("h5", deploy["h5"]))
        p10 = max(0.0, _pred_inv("h10", deploy["h10"]))
        rows.append(
            {
                "Country": country,
                "anchor_year": anchor_year,
                "forecast_year_plus1": fyears[0],
                "pred_rf_plus1": p1,
                "forecast_year_plus5": fyears[1],
                "pred_rf_plus5": p5,
                "forecast_year_plus10": fyears[2],
                "pred_rf_plus10": p10,
            }
        )
        hist = panel[panel["Country"] == country].sort_values("Year")
        plot_country_history_and_forecasts(
            country,
            hist["Year"].values,
            hist["Num_Attacks"].values,
            fyears,
            [p1, p5, p10],
            anchor_year,
            plot_dir / f"{safe_country_filename(country)}.png",
            log_target=log_target,
        )
    return pd.DataFrame(rows)


def country_profiles_for_clustering(panel: pd.DataFrame) -> pd.DataFrame:
    """Per country: mean attacks, mean fatalities, attack growth (OLS slope vs year)."""
    rows = []
    for country, g in panel.groupby("Country", sort=True):
        g = g.sort_values("Year")
        y = g["Num_Attacks"].values
        years = g["Year"].values.astype(float)
        if len(y) < 3:
            slope = 0.0
        else:
            slope, _ = np.polyfit(years, y, 1)
        rows.append(
            {
                "Country": country,
                "mean_attacks": float(np.mean(y)),
                "mean_fatalities": float(g["Fatalities"].mean()),
                "attack_slope": float(slope),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    panel = expand_balanced_panel(raw)
    panel = add_panel_features(panel)

    numeric, categorical = feature_columns()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    n_before = len(panel)
    countries_before = panel["Country"].nunique()
    row_bad = panel[numeric].isna().any(axis=1)
    n_countries_any_bad = int(panel.loc[row_bad, "Country"].nunique())
    panel = panel.dropna(subset=numeric)
    n_removed = n_before - len(panel)
    if n_removed:
        print(
            f"Panel NaN drop: removed {n_removed:,} of {n_before:,} rows "
            f"(numeric feature columns). Countries with >=1 dropped row: "
            f"{n_countries_any_bad} of {countries_before} (pre-drop country count)."
        )
    else:
        print(
            f"Panel rows after numeric feature check: {len(panel):,} "
            f"(no rows removed; {countries_before} countries)."
        )

    last_y = int(panel["Year"].max())
    train_reg = panel.loc[panel["Year"] <= TRAIN_MAX_YEAR_REG].copy()

    test_masks = {
        "t+1": (panel["Year"] > TRAIN_MAX_YEAR_REG) & (panel["Year"] <= last_y - 1),
        "t+5": (panel["Year"] > TRAIN_MAX_YEAR_REG) & (panel["Year"] <= last_y - 5),
        "t+10": (panel["Year"] > TRAIN_MAX_YEAR_REG) & (panel["Year"] <= last_y - 10),
    }
    target_cols = {
        "t+1": "y_plus1",
        "t+5": "y_plus5",
        "t+10": "y_plus10",
    }

    # Median on *observed* GTD rows only (balanced panel stacks many true zeros).
    med = float(raw["Num_Attacks"].median())
    panel["high_risk"] = (panel["Num_Attacks"] > med).astype(int)

    results = {}
    group_cv_results: dict[str, dict] = {}
    group_cv_log_results: dict[str, dict] = {}
    rf_h1 = None
    rf_h1_log = None

    for name in ["t+1", "t+5", "t+10"]:
        col = target_cols[name]
        tr = train_reg.dropna(subset=[col])
        te = panel.loc[test_masks[name]].dropna(subset=[col])
        if len(te) < 20:
            print(f"Skip {name}: too few test rows ({len(te)})")
            continue
        out = train_eval_direct_horizon(tr, te, col, numeric, categorical, log_target=False)
        out_log = train_eval_direct_horizon(tr, te, col, numeric, categorical, log_target=True)
        results[name] = {
            "rmse_rf": out["rmse_rf"],
            "rmse_ridge": out["rmse_ridge"],
            "rmse_rf_log_target": out_log["rmse_rf"],
            "rmse_ridge_log_target": out_log["rmse_ridge"],
            "n_train": len(tr),
            "n_test": len(te),
        }
        print(f"\n=== Direct horizon {name} (separate model; no iterative chaining) ===")
        print(f"  Train: Year <= {TRAIN_MAX_YEAR_REG}  |  Test: overlapping years with valid {col}")
        print(f"  Train rows: {len(tr):,}  Test rows: {len(te):,}")
        print(
            "  RMSE on original attack scale (log target: expm1 inverse; clip log pred to "
            "[0, log1p(max train y)] before inverse):"
        )
        print(f"    RandomForest — raw target: {out['rmse_rf']:.4f} | log1p target: {out_log['rmse_rf']:.4f}")
        print(f"    RidgeCV      — raw target: {out['rmse_ridge']:.4f} | log1p target: {out_log['rmse_ridge']:.4f}")
        gout = train_eval_grouped_cv(panel, col, numeric, categorical, log_target=False)
        gout_log = train_eval_grouped_cv(panel, col, numeric, categorical, log_target=True)
        if gout:
            group_cv_results[name] = gout
            print(
                f"  GroupKFold (n={gout['n_splits']}, group=Country, rows={gout['n_rows']:,}) "
                "raw target — "
                f"RMSE RF mean {gout['rmse_rf_mean']:.4f} (sd {gout['rmse_rf_std']:.4f}) | "
                f"Ridge mean {gout['rmse_ridge_mean']:.4f} (sd {gout['rmse_ridge_std']:.4f})"
            )
        else:
            print("  GroupKFold: skipped (need >=2 countries with complete rows).")
        if gout_log:
            group_cv_log_results[name] = gout_log
            print(
                f"  GroupKFold log1p target — "
                f"RMSE RF mean {gout_log['rmse_rf_mean']:.4f} (sd {gout_log['rmse_rf_std']:.4f}) | "
                f"Ridge mean {gout_log['rmse_ridge_mean']:.4f} (sd {gout_log['rmse_ridge_std']:.4f})"
            )
        if name == "t+1":
            rf_h1 = out["rf_pipe"]
            rf_h1_log = out_log["rf_pipe"]
            c_rmse = country_level_rmse(te, out["y_test"], out["pred_rf"])
            c_rmse.to_csv(OUT_DIR / "country_rmse_h1_test.csv", index=False)
            top5 = ", ".join(c_rmse["Country"].head(5).tolist())
            print(f"  Worst 5 countries by RMSE (h1 test, raw target RF): {top5}")
            c_rmse_log = country_level_rmse(te, out_log["y_test"], out_log["pred_rf"])
            c_rmse_log.to_csv(OUT_DIR / "country_rmse_h1_test_log_target.csv", index=False)
            top5_log = ", ".join(c_rmse_log["Country"].head(5).tolist())
            print(f"  Worst 5 countries by RMSE (h1 test, log-target RF): {top5_log}")

    if rf_h1 is not None:
        plot_feature_importance(
            rf_h1, top_n=12, out_path=OUT_DIR / "feature_importance_h1.png"
        )
        print(f"\nWrote {OUT_DIR / 'feature_importance_h1.png'}")
    if rf_h1_log is not None:
        plot_feature_importance(
            rf_h1_log,
            top_n=12,
            out_path=OUT_DIR / "feature_importance_h1_log_target.png",
            title="Random forest: top 12 features (direct t+1, log1p target)",
        )
        print(f"Wrote {OUT_DIR / 'feature_importance_h1_log_target.png'}")

    # Forward pass: RF fit only on Year <= TRAIN_MAX_YEAR_REG (same informational cutoff as backtest).
    plot_dir = OUT_DIR / "country_forecast_plots"
    deploy, deploy_log_clip = fit_deploy_rf_models(
        panel,
        numeric,
        categorical,
        max_train_year=TRAIN_MAX_YEAR_REG,
        log_target=True,
    )
    fc_df = run_forward_forecasts_and_plots(
        panel,
        deploy,
        numeric,
        categorical,
        last_y,
        plot_dir,
        log_target=True,
        log_clip_hi=deploy_log_clip,
    )
    fc_path = OUT_DIR / "country_forward_forecasts_rf.csv"
    fc_df.to_csv(fc_path, index=False)
    print("\n=== Forward forecasts (RF, direct horizons from last panel year) ===")
    print(
        f"  Deploy models: log1p(Num_Attacks+1) target, predictions inverse-transformed (expm1, clip 0)."
    )
    print(
        f"  Trained on Year <= {TRAIN_MAX_YEAR_REG} only (no refit on later years)."
    )
    print(
        f"  Anchor year: {last_y}. Predictions for {last_y + 1}, {last_y + 5}, {last_y + 10} "
        "are **not** in the dataset (illustrative projections only)."
    )
    print(f"  Countries with plots: {len(fc_df):,}  ->  {plot_dir}")
    print(f"  Wrote {fc_path}")

    tr_r = panel.loc[panel["Year"] <= TRAIN_MAX_YEAR_CLF].dropna(subset=numeric)
    te_r = panel.loc[panel["Year"] > TRAIN_MAX_YEAR_CLF].dropna(subset=numeric)
    X_tr, y_tr = tr_r[numeric + categorical], tr_r["high_risk"]
    X_te, y_te = te_r[numeric + categorical], te_r["high_risk"]

    clf_prep = make_preprocessors(numeric, categorical)
    clf = Pipeline(
        [
            ("prep", clf_prep),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    pred_c = clf.predict(X_te)
    acc = accuracy_score(y_te, pred_c)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, pred_c, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_te, pred_c)

    print("\n=== Risk classification (high = Num_Attacks > median of raw GTD rows) ===")
    print(f"  Threshold = median Num_Attacks in original country-year rows: {med:.4f}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Test precision (high risk): {prec:.4f}  recall: {rec:.4f}  F1: {f1:.4f}")
    print("  Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)
    print("\n  Classification report:")
    print(classification_report(y_te, pred_c, zero_division=0))

    plot_confusion(cm, ["low", "high"], OUT_DIR / "confusion_matrix_risk.png")
    print(f"\nWrote {OUT_DIR / 'confusion_matrix_risk.png'}")

    profiles = country_profiles_for_clustering(panel)
    Xc = profiles[["mean_attacks", "mean_fatalities", "attack_slope"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)

    def _kmeans_elbow_plot(X: np.ndarray, k0: int, k1: int, out_png: Path) -> None:
        ks = list(range(k0, k1 + 1))
        inertias: list[float] = []
        for k in ks:
            km_e = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            km_e.fit(X)
            inertias.append(float(km_e.inertia_))
        pd.DataFrame({"k": ks, "inertia": inertias}).to_csv(
            out_png.with_suffix(".csv"), index=False
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, inertias, marker="o")
        ax.set_xlabel("k (clusters)")
        ax.set_ylabel("Inertia")
        ax.set_title("KMeans elbow (scaled country profiles)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    elbow_path = OUT_DIR / "kmeans_elbow.png"
    _kmeans_elbow_plot(Xs, 2, 10, elbow_path)
    print(f"\nWrote {elbow_path} and {elbow_path.with_suffix('.csv')}")

    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    profiles["cluster"] = km.fit_predict(Xs)
    profiles.to_csv(OUT_DIR / "country_clusters_kmeans.csv", index=False)
    print("\n=== KMeans (k=4) on country profiles ===")
    print(profiles.groupby("cluster").size().to_string())
    print(f"\nWrote {OUT_DIR / 'country_clusters_kmeans.csv'}")

    if results:
        summ = pd.DataFrame(results).T
        summ[["rmse_rf", "rmse_ridge", "n_train", "n_test"]].to_csv(
            OUT_DIR / "direct_horizon_rmse_comparison.csv"
        )
        print(f"\nWrote {OUT_DIR / 'direct_horizon_rmse_comparison.csv'}")
        (
            summ.reset_index()
            .rename(columns={"index": "horizon"})
            .to_csv(OUT_DIR / "direct_horizon_rmse_original_vs_log.csv", index=False)
        )
        print(f"Wrote {OUT_DIR / 'direct_horizon_rmse_original_vs_log.csv'}")
    if group_cv_results:
        gc_df = pd.DataFrame([{"horizon": h, **d} for h, d in group_cv_results.items()])
        gc_path = OUT_DIR / "direct_horizon_groupkfold_rmse.csv"
        gc_df.to_csv(gc_path, index=False)
        print(f"Wrote {gc_path}")
    if group_cv_results and group_cv_log_results:
        rows_gc = []
        for h in group_cv_results:
            if h not in group_cv_log_results:
                continue
            o = group_cv_results[h]
            lg = group_cv_log_results[h]
            rows_gc.append(
                {
                    "horizon": h,
                    "n_splits": o["n_splits"],
                    "n_rows": o["n_rows"],
                    "rmse_rf_mean_original": o["rmse_rf_mean"],
                    "rmse_rf_std_original": o["rmse_rf_std"],
                    "rmse_ridge_mean_original": o["rmse_ridge_mean"],
                    "rmse_ridge_std_original": o["rmse_ridge_std"],
                    "rmse_rf_mean_log_target": lg["rmse_rf_mean"],
                    "rmse_rf_std_log_target": lg["rmse_rf_std"],
                    "rmse_ridge_mean_log_target": lg["rmse_ridge_mean"],
                    "rmse_ridge_std_log_target": lg["rmse_ridge_std"],
                }
            )
        if rows_gc:
            pd.DataFrame(rows_gc).to_csv(
                OUT_DIR / "direct_horizon_groupkfold_rmse_original_vs_log.csv",
                index=False,
            )
            print(f"Wrote {OUT_DIR / 'direct_horizon_groupkfold_rmse_original_vs_log.csv'}")
    print(f"Balanced panel rows: {len(panel):,}")


if __name__ == "__main__":
    main()
