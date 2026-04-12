"""
Time series analysis of Num_Attacks: yearly totals, trend plot, peak years, simple trend summary.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"
OUT_DIR = Path(__file__).resolve().parent


def yearly_total_attacks(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Year", as_index=True)["Num_Attacks"].sum().sort_index()


def linear_trend_slope(years: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(years.astype(float), values.astype(float), 1)
    return float(slope), float(intercept)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    yearly = yearly_total_attacks(df)

    # 1. Trend plot; highlight years with highest global totals (peaks)
    top_k = 5
    peak_years = yearly.nlargest(top_k).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly.index, yearly.values)
    ax.scatter(peak_years.index, peak_years.values, zorder=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Num_Attacks")
    ax.set_title("Total attacks by year (markers: top 5 years)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trend_num_attacks_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    years_arr = yearly.index.values
    vals_arr = yearly.values
    slope, intercept = linear_trend_slope(years_arr, vals_arr)

    print("=== Yearly total Num_Attacks (first / last) ===")
    print(f"{yearly.index[0]}: {yearly.iloc[0]:,}  |  {yearly.index[-1]}: {yearly.iloc[-1]:,}")
    print("\n=== Top years by total attacks (peaks) ===")
    print(yearly.nlargest(10).to_string())
    print("\n=== Linear trend (OLS on Year vs total attacks, full span) ===")
    print(f"Slope (attacks per calendar year): {slope:,.2f}")
    print(f"Intercept (at year 0, not interpretable): {intercept:,.2f}")
    if slope > 0:
        print(
            "Direction: positive slope -> higher totals in later years on average (line of best fit)."
        )
    elif slope < 0:
        print("Direction: negative slope -> lower totals in later years on average.")
    else:
        print("Direction: flat best-fit line over the window.")


if __name__ == "__main__":
    main()
