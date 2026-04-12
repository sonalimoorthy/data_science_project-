"""
Create histograms with mean/median lines for key variables from
data/final/master_country_year.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final" / "plots"

HIST_CONFIG = [
    ("Fatalities", "Histogram of Fatalities", "Fatalities", "hist_fatalities.png", 40),
    (
        "Conflict_Level",
        "Histogram of Conflict_Level",
        "Conflict Level",
        "hist_conflict_level.png",
        30,
    ),
    (
        "GDP_per_capita",
        "Histogram of GDP_per_capita",
        "GDP per Capita",
        "hist_gdp_per_capita.png",
        35,
    ),
    ("Population", "Histogram of Population", "Population", "hist_population.png", 40),
]


def get_skewness_comment(skew_value: float) -> str:
    # Skewness interpretation:
    # positive => right tail (few very high values),
    # negative => left tail (few very low values),
    # near 0   => roughly symmetric.
    if skew_value > 0.5:
        return "right-skewed (long right tail)"
    if skew_value < -0.5:
        return "left-skewed (long left tail)"
    return "approximately symmetric"


def save_histogram_with_reference_lines(
    series: pd.Series,
    title: str,
    x_label: str,
    output_path: Path,
    bins: int,
) -> tuple[float, float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    mean_value = float(clean.mean())
    median_value = float(clean.median())
    skew_value = float(clean.skew())

    plt.figure(figsize=(10, 6))
    plt.hist(clean, bins=bins, alpha=0.8)
    plt.axvline(mean_value, linestyle="--", linewidth=2, label=f"Mean = {mean_value:,.2f}")
    plt.axvline(median_value, linestyle="-.", linewidth=2, label=f"Median = {median_value:,.2f}")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return mean_value, median_value, skew_value


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print("Histogram summary:")
    for column, title, x_label, filename, bins in HIST_CONFIG:
        mean_value, median_value, skew_value = save_histogram_with_reference_lines(
            series=df[column],
            title=title,
            x_label=x_label,
            output_path=OUTPUT_DIR / filename,
            bins=bins,
        )
        skew_comment = get_skewness_comment(skew_value)
        print(
            f"- {column}: mean={mean_value:,.2f}, median={median_value:,.2f}, "
            f"skewness={skew_value:.3f} -> {skew_comment}"
        )

    print("\nSaved plots:")
    for _, _, _, filename, _ in HIST_CONFIG:
        print(f"- {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
