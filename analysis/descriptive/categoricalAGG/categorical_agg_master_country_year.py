"""
Aggregation analysis on master_country_year.csv:
total attacks by country/region, mean fatalities by region, top 10 countries by total attacks.
All result tables are sorted in descending order by the main metric.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # 1. Total attacks per country (descending)
    attacks_by_country = (
        df.groupby("Country", as_index=False)["Num_Attacks"]
        .sum()
        .rename(columns={"Num_Attacks": "Total_Attacks"})
        .sort_values("Total_Attacks", ascending=False)
    )

    # 2. Total attacks per region (descending)
    attacks_by_region = (
        df.groupby("Region", as_index=False)["Num_Attacks"]
        .sum()
        .rename(columns={"Num_Attacks": "Total_Attacks"})
        .sort_values("Total_Attacks", ascending=False)
    )

    # 3. Average fatalities per region (descending)
    avg_fatalities_by_region = (
        df.groupby("Region", as_index=False)["Fatalities"]
        .mean()
        .rename(columns={"Fatalities": "Avg_Fatalities"})
        .sort_values("Avg_Fatalities", ascending=False)
    )

    # 4. Top 10 countries by total attacks (subset of #1, already sorted)
    top10_countries = attacks_by_country.head(10)

    print("=== 1. Total attacks per Country (descending) ===")
    print(attacks_by_country.to_string(index=False))
    print(f"\nCountries: {len(attacks_by_country):,}")

    print("\n=== 2. Total attacks per Region (descending) ===")
    print(attacks_by_region.to_string(index=False))

    print("\n=== 3. Average fatalities per Region (descending) ===")
    print(avg_fatalities_by_region.round(6).to_string(index=False))

    print("\n=== 4. Top 10 countries by total attacks ===")
    print(top10_countries.to_string(index=False))


if __name__ == "__main__":
    main()
