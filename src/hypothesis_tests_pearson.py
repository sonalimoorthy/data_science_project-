"""
Run Pearson correlation hypothesis tests on final dataset:
H1: Conflict_Level vs Num_Attacks
H2: GDP_per_capita vs Num_Attacks
H3: Population vs Fatalities
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
ALPHA = 0.05

TESTS = [
    ("H1", "Conflict_Level", "Num_Attacks"),
    ("H2", "GDP_per_capita", "Num_Attacks"),
    ("H3", "Population", "Fatalities"),
]


def run_pearson_test(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, int]:
    pair = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError(f"Not enough data points for Pearson test: {x_col} vs {y_col}")
    r, p = pearsonr(pair[x_col], pair[y_col])
    return float(r), float(p), int(len(pair))


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print("Pearson Correlation Hypothesis Testing Results")
    print(f"Significance threshold: p < {ALPHA}")

    for hypothesis_id, x_col, y_col in TESTS:
        r, p, n = run_pearson_test(df, x_col, y_col)
        significant = "Yes" if p < ALPHA else "No"
        print(f"\n{hypothesis_id}: {x_col} vs {y_col}")
        print(f"- Sample size (n): {n}")
        print(f"- Correlation coefficient (r): {r:.6f}")
        print(f"- p-value: {p:.6e}")
        print(f"- Statistically significant (p < 0.05): {significant}")


if __name__ == "__main__":
    main()
