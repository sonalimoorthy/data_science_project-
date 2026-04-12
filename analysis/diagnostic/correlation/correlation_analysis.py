"""
Pearson and Spearman correlation among selected numeric columns in master_country_year.csv.
Writes CSV matrices, pair-plot, and heatmaps (default matplotlib/seaborn styling).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "final" / "master_country_year.csv"
OUT_DIR = Path(__file__).resolve().parent

NUMERIC_COLS = [
    "Num_Attacks",
    "Fatalities",
    "Injuries",
    "GDP_per_capita",
    "Population",
    "Conflict_Level",
]


def top_off_diagonal_pairs(corr: pd.DataFrame, n: int = 8) -> pd.Series:
    """Absolute correlation for upper triangle (excluding diagonal), sorted descending."""
    c = corr.copy()
    mask = pd.DataFrame(False, index=c.index, columns=c.columns)
    for i, row in enumerate(c.index):
        for j, col in enumerate(c.columns):
            if i >= j:
                mask.loc[row, col] = True
    upper = c.where(~mask).stack()
    return upper.abs().sort_values(ascending=False).head(n)


def main() -> None:
    df = pd.read_csv(DATA_PATH, usecols=NUMERIC_COLS)

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    pearson.to_csv(OUT_DIR / "correlation_pearson.csv")
    spearman.to_csv(OUT_DIR / "correlation_spearman.csv")

    print("=== Pearson correlation matrix ===")
    print(pearson.round(4).to_string())
    print("\n=== Spearman correlation matrix ===")
    print(spearman.round(4).to_string())
    print("\n=== Largest |r| (Pearson, off-diagonal) ===")
    print(top_off_diagonal_pairs(pearson).round(4).to_string())
    print("\n=== Largest |r| (Spearman, off-diagonal) ===")
    print(top_off_diagonal_pairs(spearman).round(4).to_string())

    # 3. Pairwise scatter / density (sample for speed)
    sample_n = min(2000, len(df))
    plot_df = df.sample(n=sample_n, random_state=42)
    pair = sns.pairplot(plot_df, corner=True, diag_kind="kde", plot_kws={"alpha": 0.35, "s": 12})
    pair.figure.savefig(
        OUT_DIR / "pairplot_correlation.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    # 4. Heatmaps
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(pearson, ax=ax, annot=True, fmt=".2f", square=True)
    ax.set_title("Pearson correlation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_pearson.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(spearman, ax=ax, annot=True, fmt=".2f", square=True)
    ax.set_title("Spearman correlation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
