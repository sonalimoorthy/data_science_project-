"""
Descriptive statistics for master_country_year.csv (Num_Attacks, Fatalities, Injuries)
and mean Num_Attacks by Country and Region.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    cols = ["Num_Attacks", "Fatalities", "Injuries"]

    summary = df[cols].agg(["mean", "median", "std", "min", "max"])
    print("=== Summary statistics (all rows) ===")
    print(summary.round(4).to_string())
    print(f"\nRow count: {len(df):,}")

    avg_by_country = (
        df.groupby("Country", as_index=False)["Num_Attacks"]
        .mean()
        .rename(columns={"Num_Attacks": "Avg_Num_Attacks"})
        .sort_values("Avg_Num_Attacks", ascending=False)
    )
    avg_by_region = (
        df.groupby("Region", as_index=False)["Num_Attacks"]
        .mean()
        .rename(columns={"Num_Attacks": "Avg_Num_Attacks"})
        .sort_values("Avg_Num_Attacks", ascending=False)
    )

    print("\n=== Average Num_Attacks by Country ===")
    print(avg_by_country.to_string(index=False))
    print("\n=== Average Num_Attacks by Region ===")
    print(avg_by_region.to_string(index=False))


if __name__ == "__main__":
    main()
