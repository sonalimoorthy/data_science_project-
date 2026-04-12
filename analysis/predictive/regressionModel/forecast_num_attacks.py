"""
Lag-based linear regression on *global* yearly attack totals to compare forecast reliability
at horizons 1, 5, and 10 years (iterative autoregressive use of predictions as lags).

Data: sum(Num_Attacks) by Year from master_country_year.csv; missing calendar years reindexed
with 0 attacks (only 1993 is missing in the current file).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def global_yearly_totals(df: pd.DataFrame) -> pd.Series:
    totals = df.groupby("Year", sort=True)["Num_Attacks"].sum()
    full_years = pd.RangeIndex(int(totals.index.min()), int(totals.index.max()) + 1)
    return totals.reindex(full_years, fill_value=0).rename("total_attacks")


def supervised_frame(series: pd.Series) -> pd.DataFrame:
    s = pd.DataFrame({"year": series.index.astype(int), "y": series.values.astype(float)})
    s["lag1"] = s["y"].shift(1)
    return s.dropna().reset_index(drop=True)


def iterative_predict(
    model: Pipeline, start_lag: float, start_year: int, steps: int
) -> float:
    """Predict year (start_year + steps) using previous prediction as lag1 each step."""
    lag = float(start_lag)
    for k in range(1, steps + 1):
        year = start_year + k
        X = pd.DataFrame({"lag1": [lag], "year": [year]})
        pred = float(model.predict(X)[0])
        lag = pred
    return lag


def horizon_rmse_backtest(
    sup: pd.DataFrame,
    anchors: list[int],
    horizon: int,
    min_train_year: int,
) -> tuple[float, int]:
    """Train with all rows year < anchor; forecast horizon steps from actual total at anchor."""
    sq_errors: list[float] = []
    used = 0
    for anchor in anchors:
        train = sup[(sup["year"] < anchor) & (sup["year"] >= min_train_year)]
        if len(train) < 5:
            continue
        actual_future = sup.loc[sup["year"] == anchor + horizon, "y"]
        if actual_future.empty:
            continue
        y_true = float(actual_future.iloc[0])
        lag0 = float(sup.loc[sup["year"] == anchor, "y"].iloc[0])

        X = train[["lag1", "year"]]
        y = train["y"]
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    RidgeCV(alphas=np.logspace(-3, 6, 30)),
                ),
            ]
        )
        model.fit(X, y)
        y_pred = iterative_predict(model, lag0, anchor, horizon)
        sq_errors.append((y_pred - y_true) ** 2)
        used += 1
    if not sq_errors:
        return float("nan"), 0
    return float(np.sqrt(np.mean(sq_errors))), used


def save_visualization(
    sup: pd.DataFrame,
    rmse1: float,
    rmse5: float,
    rmse10: float,
    n1: int,
    n5: int,
    n10: int,
    last_year: int,
    last_total: float,
    y1: float,
    y5: float,
    y10: float,
    ref_rmse: float,
    out_path: Path,
) -> None:
    """Two-panel figure: global time series + demo forecasts; bar chart of backtest RMSE by horizon."""
    fig, (ax_series, ax_rmse) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        height_ratios=[2.2, 1],
    )

    ax_series.plot(sup["year"], sup["y"])
    ax_series.axvline(last_year, linestyle="--")
    demo_years = [last_year + 1, last_year + 5, last_year + 10]
    demo_vals = [y1, y5, y10]
    ax_series.scatter(demo_years, demo_vals)
    ax_series.set_xlabel("Year")
    ax_series.set_ylabel("Global total Num_Attacks")
    ax_series.set_title(
        "Observed yearly global totals (line) and illustrative iterative forecasts (points)"
    )

    labels = [f"1 yr\n(n={n1})", f"5 yr\n(n={n5})", f"10 yr\n(n={n10})"]
    rmses = [rmse1, rmse5, rmse10]
    bars = ax_rmse.bar(labels, rmses)
    ax_rmse.set_ylabel("Backtest RMSE")
    ax_rmse.set_title(
        f"Forecast horizon reliability (lower is better). Random-split ref. RMSE ~ {ref_rmse:.0f}."
    )
    for bar, val in zip(bars, rmses):
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, usecols=["Year", "Num_Attacks"])
    series = global_yearly_totals(df)
    sup = supervised_frame(series)

    last_year = int(sup["year"].max())
    first_year = int(sup["year"].min())
    min_train_year = first_year + 2

    max_anchor_h10 = last_year - 10
    anchors_h10 = list(range(min_train_year + 5, max_anchor_h10 + 1))
    anchors_h5 = list(range(min_train_year + 5, last_year - 5 + 1))
    anchors_h1 = list(range(min_train_year + 5, last_year))

    rmse1, n1 = horizon_rmse_backtest(sup, anchors_h1, 1, min_train_year)
    rmse5, n5 = horizon_rmse_backtest(sup, anchors_h5, 5, min_train_year)
    rmse10, n10 = horizon_rmse_backtest(sup, anchors_h10, 10, min_train_year)

    # In-sample reference: single train/test split on supervised rows (not time-ordered CV)
    X_all = sup[["lag1", "year"]]
    y_all = sup["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42
    )
    ref = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 6, 30))),
        ]
    )
    ref.fit(X_train, y_train)
    ref_rmse = float(np.sqrt(mean_squared_error(y_test, ref.predict(X_test))))

    # Forward extrapolation from last observed year (illustrative; outside training support)
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 6, 30))),
        ]
    )
    final_model.fit(X_all, y_all)
    last_total = float(sup.loc[sup["year"] == last_year, "y"].iloc[0])
    y1 = iterative_predict(final_model, last_total, last_year, 1)
    y5 = iterative_predict(final_model, last_total, last_year, 5)
    y10 = iterative_predict(final_model, last_total, last_year, 10)

    print("=== Global yearly totals: lag1 + year -> Ridge regression ===")
    print(f"Year span (supervised rows): {first_year}-{last_year} (n={len(sup)})")
    print("\n=== Backtest RMSE by iterative horizon (lower = more reliable here) ===")
    print(f"  h=1 year:  RMSE ~ {rmse1:.2f}  (anchors used: {n1})")
    print(f"  h=5 years: RMSE ~ {rmse5:.2f}  (anchors used: {n5})")
    print(f"  h=10 years: RMSE ~ {rmse10:.2f}  (anchors used: {n10})")
    print(
        "\nTypical pattern: h=1 is smallest; error grows as horizon lengthens because each step "
        "feeds model output back as the lag (error compounding)."
    )
    print(f"\nRandom split reference RMSE (same features, not pure time CV): {ref_rmse:.2f}")

    print("\n=== Illustrative forward iterations from last data year ===")
    print(f"Last observed year {last_year}, total attacks ~ {last_total:,.0f}")
    print(f"Iterated point forecast total attacks in {last_year + 1}: {y1:,.2f}")
    print(f"Iterated point forecast total attacks in {last_year + 5}: {y5:,.2f}")
    print(f"Iterated point forecast total attacks in {last_year + 10}: {y10:,.2f}")
    print(
        "These extrapolations are not calibrated for true out-of-sample years; use only as demos."
    )

    out = pd.DataFrame(
        {
            "horizon_years": [1, 5, 10],
            "backtest_rmse": [rmse1, rmse5, rmse10],
            "anchors_used": [n1, n5, n10],
        }
    )
    out.to_csv(OUT_DIR / "horizon_backtest_rmse.csv", index=False)
    fwd = pd.DataFrame(
        {
            "forecast_year": [last_year + 1, last_year + 5, last_year + 10],
            "horizon": [1, 5, 10],
            "predicted_total_attacks": [y1, y5, y10],
        }
    )
    fwd.to_csv(OUT_DIR / "forward_iterative_forecasts.csv", index=False)

    plot_path = OUT_DIR / "forecast_visualization.png"
    save_visualization(
        sup,
        rmse1,
        rmse5,
        rmse10,
        n1,
        n5,
        n10,
        last_year,
        last_total,
        y1,
        y5,
        y10,
        ref_rmse,
        plot_path,
    )

    print(f"\nWrote {OUT_DIR / 'horizon_backtest_rmse.csv'}")
    print(f"Wrote {OUT_DIR / 'forward_iterative_forecasts.csv'}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
