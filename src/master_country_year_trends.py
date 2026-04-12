"""
Create Year-wise trend plots and brief summary statistics
from data/final/master_country_year.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final" / "plots"

PLOT_CONFIG = [
    ("Num_Attacks", "Year vs Num_Attacks", "Number of Attacks", "year_vs_num_attacks.png"),
    ("Fatalities", "Year vs Fatalities", "Fatalities", "year_vs_fatalities.png"),
    (
        "Conflict_Level",
        "Year vs Average Conflict_Level",
        "Average Conflict Level",
        "year_vs_avg_conflict_level.png",
    ),
    (
        "GDP_per_capita",
        "Year vs Average GDP_per_capita",
        "Average GDP per Capita",
        "year_vs_avg_gdp_per_capita.png",
    ),
]


def build_year_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a yearly table for plotting."""
    grouped = (
        df.groupby("Year", as_index=False)
        .agg(
            Num_Attacks=("Num_Attacks", "sum"),
            Fatalities=("Fatalities", "sum"),
            Conflict_Level=("Conflict_Level", "mean"),
            GDP_per_capita=("GDP_per_capita", "mean"),
        )
        .sort_values("Year", ignore_index=True)
    )
    return grouped


def save_line_plot(
    years: pd.Series,
    values: pd.Series,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(years, values, marker="o", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_brief_summary_stats(yearly_df: pd.DataFrame) -> None:
    metrics = ["Num_Attacks", "Fatalities", "Conflict_Level", "GDP_per_capita"]
    print("\nBrief summary statistics (year-level series):")
    for col in metrics:
        series = yearly_df[col].dropna()
        print(
            f"- {col}: count={int(series.count())}, "
            f"mean={series.mean():.2f}, median={series.median():.2f}, "
            f"min={series.min():.2f}, max={series.max():.2f}"
        )


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    yearly_df = build_year_aggregates(df)

    for column, title, y_label, filename in PLOT_CONFIG:
        save_line_plot(
            years=yearly_df["Year"],
            values=yearly_df[column],
            title=title,
            y_label=y_label,
            output_path=OUTPUT_DIR / filename,
        )

    print_brief_summary_stats(yearly_df)
    print("\nSaved plots:")
    for _, _, _, filename in PLOT_CONFIG:
        print(f"- {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
