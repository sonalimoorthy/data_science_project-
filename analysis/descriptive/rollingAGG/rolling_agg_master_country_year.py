"""
Time-based aggregations on master_country_year.csv:
yearly global attack totals, 3-row rolling mean of Num_Attacks per Country, YoY growth on yearly totals.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["Country", "Year"])

    # 1. Yearly total number of attacks (sum across all countries)
    yearly_attacks = (
        df.groupby("Year", as_index=False)["Num_Attacks"]
        .sum()
        .rename(columns={"Num_Attacks": "Total_Attacks"})
        .sort_values("Year")
    )

    # 2. 3-year rolling average of Num_Attacks per Country (groupby + rolling)
    df_roll = df.copy()
    df_roll["Rolling3Y_Avg_Attacks"] = (
        df_roll.groupby("Country", group_keys=False)["Num_Attacks"]
        .rolling(window=3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 3. Year-over-year growth rate of global yearly attack totals
    yearly_attacks["YoY_Growth_Rate"] = yearly_attacks["Total_Attacks"].pct_change()

    print("=== 1. Yearly total attacks & 3. YoY growth (global) ===")
    print(yearly_attacks.to_string(index=False))

    print("\n=== 2. Sample: 3-year rolling avg Num_Attacks per Country (Afghanistan) ===")
    sample_country = "Afghanistan"
    cols = ["Country", "Year", "Num_Attacks", "Rolling3Y_Avg_Attacks"]
    print(
        df_roll.loc[df_roll["Country"] == sample_country, cols]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
