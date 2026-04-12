"""
Create a Region-Year heatmap for Num_Attacks from
data/final/master_country_year.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
OUTPUT_PNG = PROJECT_ROOT / "data" / "final" / "plots" / "region_year_attack_heatmap.png"


def build_region_year_pivot(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["Region", "Year"], as_index=False)
        .agg(Num_Attacks=("Num_Attacks", "sum"))
        .sort_values(["Region", "Year"], ignore_index=True)
    )
    pivot = grouped.pivot(index="Region", columns="Year", values="Num_Attacks").fillna(0)
    return pivot


def save_heatmap(pivot: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 7))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Num_Attacks"},
    )
    plt.title("Region vs Year Heatmap of Num_Attacks")
    plt.xlabel("Year")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    pivot = build_region_year_pivot(df)
    save_heatmap(pivot, OUTPUT_PNG)
    print(f"Saved heatmap: {OUTPUT_PNG}")
    print(f"Pivot shape (regions x years): {pivot.shape}")


if __name__ == "__main__":
    main()
