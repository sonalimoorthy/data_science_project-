"""
Compute correlation diagnostics and plot an annotated heatmap for:
Num_Attacks, Fatalities, Conflict_Level, GDP_per_capita, Population.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
OUTPUT_PNG = PROJECT_ROOT / "data" / "final" / "plots" / "diagnostic_correlation_heatmap.png"

CORR_COLUMNS = ["Num_Attacks", "Fatalities", "Conflict_Level", "GDP_per_capita", "Population"]


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = df[CORR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return corr_df.corr(numeric_only=True)


def strongest_pairwise_correlation(corr: pd.DataFrame) -> tuple[str, str, float]:
    best_pair: tuple[str, str, float] | None = None
    for left, right in combinations(corr.columns, 2):
        value = float(corr.loc[left, right])
        if best_pair is None or abs(value) > abs(best_pair[2]):
            best_pair = (left, right, value)
    if best_pair is None:
        raise ValueError("Could not compute pairwise correlation.")
    return best_pair


def save_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Diagnostic Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    corr = compute_correlation_matrix(df)
    save_heatmap(corr, OUTPUT_PNG)

    left, right, value = strongest_pairwise_correlation(corr)
    print("Correlation matrix:")
    print(corr.to_string())
    print(f"\nSaved heatmap: {OUTPUT_PNG}")

    # Strongest-correlation interpretation:
    # We pick the off-diagonal pair with the largest absolute coefficient.
    # Positive means both variables move together; negative means inverse movement.
    print(
        f"Strongest pair (absolute): {left} vs {right} = {value:.3f} "
        f"({'positive' if value >= 0 else 'negative'})"
    )


if __name__ == "__main__":
    main()
