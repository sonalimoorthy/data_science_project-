"""
Visualizations for master_country_year.csv (matplotlib / seaborn defaults only; no manual colors).
Writes PNGs next to this script.
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


def plot_line_total_attacks_over_time(df: pd.DataFrame) -> None:
    yearly = df.groupby("Year", as_index=True)["Num_Attacks"].sum().sort_index()
    fig, ax = plt.subplots()
    ax.plot(yearly.index, yearly.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total attacks")
    ax.set_title("Total attacks over time")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "line_total_attacks_over_time.png", dpi=150)
    plt.close(fig)


def plot_bar_top10_countries(df: pd.DataFrame) -> None:
    by_country = df.groupby("Country", as_index=True)["Num_Attacks"].sum()
    top10 = by_country.nlargest(10).sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.barh(top10.index.astype(str), top10.values)
    ax.set_xlabel("Total attacks")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 countries by total attacks")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bar_top10_countries_attacks.png", dpi=150)
    plt.close(fig)


def plot_hist_num_attacks(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.hist(df["Num_Attacks"].dropna(), bins=50)
    ax.set_xlabel("Num_Attacks")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Num_Attacks")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hist_num_attacks_distribution.png", dpi=150)
    plt.close(fig)


def plot_heatmap_country_year(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        index="Country",
        columns="Year",
        values="Num_Attacks",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.sort_index(axis=1)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig_h = max(8, 0.12 * len(pivot))
    fig_w = max(10, 0.25 * pivot.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, ax=ax, cbar_kws={"label": "Attacks"})
    ax.set_title("Attacks by country and year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Country")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_country_year_attacks.png", dpi=150)
    plt.close(fig)


def plot_pie_attack_type_proportions(df: pd.DataFrame) -> None:
    by_type = (
        df.groupby("Attack_Type_Mode", as_index=True)["Num_Attacks"]
        .sum()
        .sort_values(ascending=False)
    )
    total = float(by_type.sum())
    pct = (by_type.values / total * 100).round(1)
    legend_labels = [
        f"{name}: {p:g}%"
        for name, p in zip(by_type.index.astype(str), pct)
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _ = ax.pie(by_type.values, labels=None)
    ax.legend(
        wedges,
        legend_labels,
        title="Attack type",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.set_title("Attack type proportions (share of total Num_Attacks)")
    ax.axis("equal")
    fig.savefig(
        OUT_DIR / "pie_attack_type_proportions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    plot_line_total_attacks_over_time(df)
    plot_bar_top10_countries(df)
    plot_hist_num_attacks(df)
    plot_heatmap_country_year(df)
    plot_pie_attack_type_proportions(df)


if __name__ == "__main__":
    main()
