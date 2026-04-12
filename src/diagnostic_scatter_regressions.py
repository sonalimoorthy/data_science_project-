"""
Create diagnostic scatter plots with regression lines for:
- Conflict_Level vs Num_Attacks
- GDP_per_capita vs Num_Attacks
- Population vs Fatalities
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final" / "plots"

PLOT_CONFIG = [
    (
        "Conflict_Level",
        "Num_Attacks",
        "Conflict_Level vs Num_Attacks",
        "diagnostic_conflict_level_vs_num_attacks.png",
    ),
    (
        "GDP_per_capita",
        "Num_Attacks",
        "GDP_per_capita vs Num_Attacks",
        "diagnostic_gdp_per_capita_vs_num_attacks.png",
    ),
    (
        "Population",
        "Fatalities",
        "Population vs Fatalities",
        "diagnostic_population_vs_fatalities.png",
    ),
]


def save_scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.45, "s": 20},
        line_kws={"color": "red", "linewidth": 2},
    )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    for x_col, y_col, title, filename in PLOT_CONFIG:
        save_scatter_with_regression(
            df=df,
            x_col=x_col,
            y_col=y_col,
            title=title,
            output_path=OUTPUT_DIR / filename,
        )

    print("Saved plots:")
    for _, _, _, filename in PLOT_CONFIG:
        print(f"- {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
