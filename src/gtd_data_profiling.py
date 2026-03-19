"""
GTD data profiling script (no cleaning).

This script:
1) Loads GTD data from raw folder.
2) Selects only the required columns.
3) Runs profiling checks and prints results.
4) Saves subset data and text report to data/interim.
"""

from pathlib import Path
import pandas as pd


def get_gtd_path(project_root: Path) -> Path:
    """
    Resolve GTD raw file path.
    Checks both:
    - data/raw/globalterrorismdb_0718dist.csv
    - data/raw/archive/globalterrorismdb_0718dist.csv
    """
    candidates = [
        project_root / "data" / "raw" / "globalterrorismdb_0718dist.csv",
        project_root / "data" / "raw" / "archive" / "globalterrorismdb_0718dist.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find GTD file. Expected one of: "
        f"{candidates[0]} or {candidates[1]}"
    )


def format_series_as_lines(series: pd.Series, indent: str = "  - ") -> list[str]:
    """Format a series into report-friendly bullet lines."""
    lines: list[str] = []
    for key, value in series.items():
        lines.append(f"{indent}{key}: {value}")
    return lines


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    output_subset_path = interim_dir / "gtd_profile_subset.csv"
    output_report_path = interim_dir / "gtd_profile_report.txt"

    selected_columns = [
        "iyear",
        "country_txt",
        "region_txt",
        "attacktype1_txt",
        "targtype1_txt",
        "gname",
        "nkill",
        "nwound",
    ]

    gtd_path = get_gtd_path(project_root)
    # GTD files are often Latin-1 encoded; fall back if UTF-8 fails.
    try:
        df = pd.read_csv(gtd_path, usecols=selected_columns, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(
            gtd_path,
            usecols=selected_columns,
            low_memory=False,
            encoding="latin1",
        )

    # Save subset only (no modifications to values)
    df.to_csv(output_subset_path, index=False)

    report_lines: list[str] = []
    report_lines.append("GTD DATA PROFILING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Input file: {gtd_path}")
    report_lines.append(f"Subset output: {output_subset_path}")
    report_lines.append("")

    # 1) Basic dataset info
    report_lines.append("1) BASIC DATASET INFO")
    report_lines.append("-" * 80)
    report_lines.append(f"Shape (rows, columns): {df.shape}")
    report_lines.append(f"Column names: {list(df.columns)}")
    report_lines.append("Data types:")
    report_lines.extend(format_series_as_lines(df.dtypes.astype(str)))
    report_lines.append("")

    # 2) Data quality checks
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())
    unique_values = df.nunique(dropna=False)

    report_lines.append("2) DATA QUALITY CHECKS")
    report_lines.append("-" * 80)
    report_lines.append("Missing values (count):")
    report_lines.extend(format_series_as_lines(missing_count))
    report_lines.append("Missing values (%):")
    report_lines.extend(format_series_as_lines(missing_pct))
    report_lines.append(f"Duplicate rows: {duplicate_rows}")
    report_lines.append("Unique values per column:")
    report_lines.extend(format_series_as_lines(unique_values))
    report_lines.append("")

    # 3) Top values for categorical columns
    categorical_columns = [
        "country_txt",
        "region_txt",
        "attacktype1_txt",
        "targtype1_txt",
        "gname",
    ]
    report_lines.append("3) TOP 10 MOST FREQUENT VALUES (CATEGORICAL COLUMNS)")
    report_lines.append("-" * 80)
    for col in categorical_columns:
        report_lines.append(f"{col}:")
        top_10 = df[col].value_counts(dropna=False).head(10)
        report_lines.extend(format_series_as_lines(top_10, indent="  - "))
        report_lines.append("")

    # 4) Numeric profiling
    numeric_columns = ["nkill", "nwound"]
    report_lines.append("4) NUMERIC COLUMNS SUMMARY")
    report_lines.append("-" * 80)
    for col in numeric_columns:
        report_lines.append(f"{col}:")
        stats = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "min": df[col].min(),
            "max": df[col].max(),
        }
        for k, v in stats.items():
            report_lines.append(f"  - {k}: {v}")
        negative_count = int((df[col] < 0).sum(skipna=True))
        report_lines.append(f"  - negative values count: {negative_count}")
        report_lines.append("")

    # 5) Potential data issues
    missing_country_or_year = df[df["country_txt"].isna() | df["iyear"].isna()]
    fatalities_or_injuries_null = df[df["nkill"].isna() | df["nwound"].isna()]
    perpetrator_unknown = df[
        df["gname"].astype(str).str.strip().str.lower().eq("unknown")
    ]

    report_lines.append("5) POTENTIAL DATA ISSUES")
    report_lines.append("-" * 80)
    report_lines.append(
        f"Rows with missing country or year: {len(missing_country_or_year)}"
    )
    report_lines.append(
        f"Rows where fatalities or injuries are null: {len(fatalities_or_injuries_null)}"
    )
    report_lines.append(f"Rows where perpetrator is 'Unknown': {len(perpetrator_unknown)}")
    report_lines.append("")

    report_lines.append("NOTE")
    report_lines.append("-" * 80)
    report_lines.append(
        "This is profiling only. No data cleaning, filling, dropping, or renaming was performed."
    )

    report_text = "\n".join(report_lines)
    print(report_text)
    output_report_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
