"""
Create top-10 country rankings and bar charts from
data/final/master_country_year.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final" / "plots"
TOP_N = 10

PLOT_CONFIG = [
    ("Num_Attacks", "Top 10 Countries by Num_Attacks", "Num_Attacks", "top10_num_attacks.png"),
    ("Fatalities", "Top 10 Countries by Fatalities", "Fatalities", "top10_fatalities.png"),
    (
        "Conflict_Level",
        "Top 10 Countries by Conflict_Level",
        "Average Conflict_Level",
        "top10_conflict_level.png",
    ),
]


def build_country_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate country-level metrics used for top-10 rankings."""
    return (
        df.groupby("Country", as_index=False)
        .agg(
            Num_Attacks=("Num_Attacks", "sum"),
            Fatalities=("Fatalities", "sum"),
            Conflict_Level=("Conflict_Level", "mean"),
        )
        .sort_values("Country", ignore_index=True)
    )


def top_n_table(df: pd.DataFrame, metric: str, n: int = TOP_N) -> pd.DataFrame:
    return df[["Country", metric]].sort_values(metric, ascending=False).head(n).reset_index(drop=True)


def save_bar_chart(table: pd.DataFrame, metric: str, title: str, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    plt.bar(table["Country"], table[metric])
    plt.title(title)
    plt.xlabel("Country")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_top_table(table: pd.DataFrame, metric: str) -> None:
    formatted = table.copy()
    if metric == "Conflict_Level":
        formatted[metric] = formatted[metric].map(lambda x: f"{x:.2f}")
    else:
        formatted[metric] = formatted[metric].map(lambda x: f"{x:,.0f}")
    print(f"\nTop {TOP_N} countries by {metric}:")
    print(formatted.to_string(index=False))


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    country_df = build_country_aggregates(df)

    for metric, title, y_label, filename in PLOT_CONFIG:
        table = top_n_table(country_df, metric)
        save_bar_chart(table, metric, title, y_label, OUTPUT_DIR / filename)
        print_top_table(table, metric)

    print("\nSaved plots:")
    for _, _, _, filename in PLOT_CONFIG:
        print(f"- {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
