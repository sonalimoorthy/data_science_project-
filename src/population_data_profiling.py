"""
Population dataset profiling script (no cleaning).

This script:
1) Loads population data from raw folder.
2) Selects required profiling columns.
3) Runs profile checks and issue checks.
4) Saves subset CSV and text report in data/interim.
"""

from pathlib import Path
import pandas as pd


def resolve_input_path(project_root: Path) -> Path:
    """
    Resolve population input path in either location:
    - data/raw/population-with-un-projections.csv
    - data/raw/population-with-un-projections/population-with-un-projections.csv
    """
    candidates = [
        project_root / "data" / "raw" / "population-with-un-projections.csv",
        project_root
        / "data"
        / "raw"
        / "population-with-un-projections"
        / "population-with-un-projections.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find population-with-un-projections.csv.")


def resolve_population_column(columns: list[str]) -> str:
    """
    Pick the best available population column.
    Preference order keeps compatibility with OWID exports that vary by version.
    """
    candidates = [
        "Population (historical estimates)",
        "Population, total",
        "Population (historical)",
        "Population",
    ]
    for col in candidates:
        if col in columns:
            return col
    raise KeyError(
        "Population column not found. Expected one of: "
        + ", ".join(candidates)
    )


def format_series(series: pd.Series, indent: str = "  - ") -> list[str]:
    """Format key/value series into report lines."""
    return [f"{indent}{idx}: {val}" for idx, val in series.items()]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    subset_output_path = interim_dir / "population_profile_subset.csv"
    report_output_path = interim_dir / "population_profile_report.txt"

    full_df = pd.read_csv(input_path, low_memory=False)
    population_col = resolve_population_column(full_df.columns.tolist())
    selected_columns = ["Entity", "Year", population_col]
    df = full_df[selected_columns].copy()

    # Save exact subset only (no cleaning/modification).
    df.to_csv(subset_output_path, index=False)

    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())
    unique_countries = int(df["Entity"].nunique(dropna=True))

    year_numeric = pd.to_numeric(df["Year"], errors="coerce")
    year_min = year_numeric.min()
    year_max = year_numeric.max()

    population_numeric = pd.to_numeric(df[population_col], errors="coerce")
    missing_population_count = int(population_numeric.isna().sum())

    # OWID typically tags aggregate entries (regions/income groups/world) as OWID_* codes.
    # Use the original full_df["Code"] to avoid false positives from country names.
    code_series = full_df["Code"].astype("string")
    non_country_mask = code_series.str.startswith("OWID_", na=False)
    non_country_count = int(non_country_mask.sum())
    non_country_examples = (
        full_df.loc[non_country_mask, "Entity"].dropna().drop_duplicates().head(15)
    )

    report_lines: list[str] = []
    report_lines.append("POPULATION DATASET - DATA PROFILING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Subset output: {subset_output_path}")
    report_lines.append(f"Selected population column: {population_col}")
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
    report_lines.append(f"Unique countries/entities: {unique_countries}")
    report_lines.append(f"Year range: {year_min} to {year_max}")
    report_lines.append("")

    report_lines.append("3) POTENTIAL DATA ISSUES")
    report_lines.append("-" * 80)
    report_lines.append(
        f"Potential non-country entries detected: {non_country_count}"
    )
    report_lines.append(
        "Examples of non-country entries (heuristic): "
        + ", ".join(non_country_examples.astype(str).tolist())
    )
    report_lines.append(f"Missing population values: {missing_population_count}")
    report_lines.append("")

    report_lines.append("NOTE")
    report_lines.append("-" * 80)
    report_lines.append(
        "This script is profiling only. No cleaning, dropping, filling, renaming, "
        "or filtering is applied to the data."
    )

    report_text = "\n".join(report_lines)
    print(report_text)
    report_output_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
