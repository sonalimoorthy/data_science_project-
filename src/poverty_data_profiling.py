"""
Poverty dataset profiling script (no cleaning).

This script:
1) Loads poverty data from raw folder.
2) Selects required profiling columns.
3) Runs profile checks and issue checks.
4) Saves subset CSV and text report in data/interim.
"""

from pathlib import Path
import pandas as pd


def resolve_input_path(project_root: Path) -> Path:
    """
    Resolve poverty input path in either location:
    - data/raw/share-of-population-in-extreme-poverty.csv
    - data/raw/share-of-population-in-extreme-poverty/share-of-population-in-extreme-poverty.csv
    """
    candidates = [
        project_root / "data" / "raw" / "share-of-population-in-extreme-poverty.csv",
        project_root
        / "data"
        / "raw"
        / "share-of-population-in-extreme-poverty"
        / "share-of-population-in-extreme-poverty.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find share-of-population-in-extreme-poverty.csv.")


def resolve_poverty_column(columns: list[str]) -> str:
    """
    Resolve poverty percentage column name across export variants.
    """
    candidates = [
        "Share of population in extreme poverty (%)",
        "Share of population in poverty ($3 a day)",
        "Share of population in extreme poverty",
    ]
    for col in candidates:
        if col in columns:
            return col
    raise KeyError("Poverty column not found in dataset.")


def format_series(series: pd.Series, indent: str = "  - ") -> list[str]:
    return [f"{indent}{idx}: {val}" for idx, val in series.items()]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    subset_output_path = interim_dir / "poverty_profile_subset.csv"
    report_output_path = interim_dir / "poverty_profile_report.txt"

    full_df = pd.read_csv(input_path, low_memory=False)
    poverty_col = resolve_poverty_column(full_df.columns.tolist())

    selected_cols = ["Entity", "Year", poverty_col]
    df = full_df[selected_cols].copy()

    # Save profiling subset only (no cleaning/modification).
    df.to_csv(subset_output_path, index=False)

    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())
    unique_countries = int(df["Entity"].nunique(dropna=True))

    year_numeric = pd.to_numeric(df["Year"], errors="coerce")
    year_min = year_numeric.min()
    year_max = year_numeric.max()

    # Use Code column from source when available to detect OWID aggregate rows.
    if "Code" in full_df.columns:
        non_country_mask = full_df["Code"].astype("string").str.startswith("OWID_", na=False)
    else:
        non_country_mask = pd.Series(False, index=full_df.index)

    non_country_count = int(non_country_mask.sum())
    non_country_examples = (
        full_df.loc[non_country_mask, "Entity"].dropna().drop_duplicates().head(15)
    )

    missing_poverty_count = int(pd.to_numeric(df[poverty_col], errors="coerce").isna().sum())

    report_lines: list[str] = []
    report_lines.append("POVERTY DATASET - DATA PROFILING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Subset output: {subset_output_path}")
    report_lines.append(f"Selected poverty column: {poverty_col}")
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
    report_lines.append(f"Potential non-country entries detected: {non_country_count}")
    report_lines.append(
        "Examples of non-country entries (heuristic): "
        + ", ".join(non_country_examples.astype(str).tolist())
    )
    report_lines.append(f"Missing poverty values: {missing_poverty_count}")
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
