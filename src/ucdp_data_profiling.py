"""
UCDP organized violence data profiling script (no cleaning).

This script:
1) Loads UCDP data from raw folder.
2) Selects only required profiling columns.
3) Prints and writes profiling checks.
4) Saves subset and report to data/interim.
"""

from pathlib import Path
import pandas as pd


SELECTED_COLUMNS = [
    "country_cy",
    "year_cy",
    "region_cy",
    "cumulative_total_deaths_in_orgvio_best_cy",
]


def resolve_input_path(project_root: Path) -> Path:
    """
    Resolve UCDP input path in either location:
    - data/raw/organizedviolencecy_v25_1.csv
    - data/raw/organizedviolencecy-251-csv/organizedviolencecy_v25_1.csv
    """
    candidates = [
        project_root / "data" / "raw" / "organizedviolencecy_v25_1.csv",
        project_root
        / "data"
        / "raw"
        / "organizedviolencecy-251-csv"
        / "organizedviolencecy_v25_1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find organizedviolencecy_v25_1.csv in expected raw locations."
    )


def format_series(series: pd.Series, indent: str = "  - ") -> list[str]:
    """Format series values for a plain-text report."""
    return [f"{indent}{idx}: {val}" for idx, val in series.items()]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    subset_output_path = interim_dir / "conflict_profile_subset.csv"
    report_output_path = interim_dir / "conflict_profile_report.txt"

    df = pd.read_csv(input_path, usecols=SELECTED_COLUMNS, low_memory=False)

    # Save exact profiling subset (no cleaning/modification).
    df.to_csv(subset_output_path, index=False)

    deaths_col = "cumulative_total_deaths_in_orgvio_best_cy"
    year_col = "year_cy"
    country_col = "country_cy"

    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())
    unique_countries = int(df[country_col].nunique(dropna=True))

    year_numeric = pd.to_numeric(df[year_col], errors="coerce")
    year_min = year_numeric.min()
    year_max = year_numeric.max()

    deaths_numeric = pd.to_numeric(df[deaths_col], errors="coerce")
    deaths_mean = deaths_numeric.mean()
    deaths_min = deaths_numeric.min()
    deaths_max = deaths_numeric.max()
    negative_deaths_count = int((deaths_numeric < 0).sum(skipna=True))

    missing_country_or_year = df[df[country_col].isna() | df[year_col].isna()]
    zero_conflict_years = int((deaths_numeric == 0).sum(skipna=True))
    nonzero_conflict_years = int((deaths_numeric > 0).sum(skipna=True))

    report_lines: list[str] = []
    report_lines.append("UCDP ORGANIZED VIOLENCE - DATA PROFILING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Subset output: {subset_output_path}")
    report_lines.append("")

    report_lines.append("1) BASIC DATASET INFO")
    report_lines.append("-" * 80)
    report_lines.append(f"Shape (rows, columns): {df.shape}")
    report_lines.append(f"Column names: {list(df.columns)}")
    report_lines.append("Data types:")
    report_lines.extend(format_series(df.dtypes.astype(str)))
    report_lines.append("")

    report_lines.append("2) DATA QUALITY CHECKS")
    report_lines.append("-" * 80)
    report_lines.append("Missing values (count):")
    report_lines.extend(format_series(missing_count))
    report_lines.append("Missing values (%):")
    report_lines.extend(format_series(missing_pct))
    report_lines.append(f"Duplicate rows: {duplicate_rows}")
    report_lines.append(f"Unique countries: {unique_countries}")
    report_lines.append(f"Year range: {year_min} to {year_max}")
    report_lines.append("")

    report_lines.append("3) CONFLICT DEATHS COLUMN CHECKS")
    report_lines.append("-" * 80)
    report_lines.append(f"Column: {deaths_col}")
    report_lines.append(f"Mean: {deaths_mean}")
    report_lines.append(f"Min: {deaths_min}")
    report_lines.append(f"Max: {deaths_max}")
    report_lines.append(f"Negative values count: {negative_deaths_count}")
    report_lines.append("")

    report_lines.append("4) POTENTIAL DATA ISSUES")
    report_lines.append("-" * 80)
    report_lines.append(
        f"Rows with missing country or year: {len(missing_country_or_year)}"
    )
    report_lines.append(f"Zero conflict years (deaths == 0): {zero_conflict_years}")
    report_lines.append(
        f"Non-zero conflict years (deaths > 0): {nonzero_conflict_years}"
    )
    report_lines.append("")

    report_lines.append("NOTE")
    report_lines.append("-" * 80)
    report_lines.append(
        "This script is profiling only. No cleaning, dropping, filling, renaming, "
        "or other data modifications were applied."
    )

    report_text = "\n".join(report_lines)
    print(report_text)
    report_output_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
