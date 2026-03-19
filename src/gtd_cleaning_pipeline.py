"""
GTD cleaning + structuring pipeline.

This script performs:
1) Column selection and renaming.
2) Event-level cleaning (no modeling features).
3) Country-year aggregation using safe mode handling.
4) Output export to CSV and Parquet.
"""

from pathlib import Path
from typing import Any

import pandas as pd


RAW_COLUMNS = [
    "iyear",
    "country_txt",
    "region_txt",
    "attacktype1_txt",
    "targtype1_txt",
    "gname",
    "nkill",
    "nwound",
]

COLUMN_RENAME_MAP = {
    "iyear": "Year",
    "country_txt": "Country",
    "region_txt": "Region",
    "attacktype1_txt": "Attack_Type",
    "targtype1_txt": "Target_Type",
    "gname": "Perpetrator",
    "nkill": "Fatalities",
    "nwound": "Injuries",
}

FINAL_AGG_COLUMNS = [
    "Country",
    "Year",
    "Region",
    "Num_Attacks",
    "Fatalities",
    "Injuries",
    "Attack_Type_Mode",
    "Target_Type_Mode",
    "Perpetrator_Mode",
]


def resolve_input_path(project_root: Path) -> Path:
    """
    Resolve GTD input path.
    Supports both requested and existing project layout.
    """
    candidates = [
        project_root / "data" / "raw" / "globalterrorismdb_0718dist.csv",
        project_root / "data" / "raw" / "archive" / "globalterrorismdb_0718dist.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "GTD raw file not found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}"
    )


def safe_mode(series: pd.Series) -> Any:
    """
    Return the first mode of a series.
    If all values are null/empty after cleanup, return pd.NA.
    """
    cleaned = series.dropna()
    if cleaned.empty:
        return pd.NA
    modes = cleaned.mode(dropna=True)
    if modes.empty:
        return pd.NA
    return modes.iloc[0]


def load_selected_columns(input_path: Path) -> pd.DataFrame:
    """Load only the required GTD columns."""
    try:
        return pd.read_csv(input_path, usecols=RAW_COLUMNS, low_memory=False)
    except UnicodeDecodeError:
        # GTD distribution files frequently use Latin-1.
        return pd.read_csv(
            input_path,
            usecols=RAW_COLUMNS,
            low_memory=False,
            encoding="latin1",
        )


def clean_event_level(df: pd.DataFrame) -> pd.DataFrame:
    """Apply event-level cleaning rules exactly as specified."""
    cleaned = df.rename(columns=COLUMN_RENAME_MAP).copy()

    text_columns = ["Country", "Region", "Attack_Type", "Target_Type", "Perpetrator"]
    for col in text_columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce")
    cleaned["Fatalities"] = pd.to_numeric(cleaned["Fatalities"], errors="coerce")
    cleaned["Injuries"] = pd.to_numeric(cleaned["Injuries"], errors="coerce")

    # Drop rows where Country or Year is missing.
    cleaned = cleaned.dropna(subset=["Country", "Year"])

    # Convert Year to integer after removing missing values.
    cleaned["Year"] = cleaned["Year"].astype("int64")

    # Replace missing Fatalities and Injuries with 0.
    cleaned["Fatalities"] = cleaned["Fatalities"].fillna(0)
    cleaned["Injuries"] = cleaned["Injuries"].fillna(0)

    # Replace Unknown/empty perpetrator values with NaN.
    perpetrator_normalized = cleaned["Perpetrator"].str.casefold()
    unknown_or_empty = perpetrator_normalized.eq("unknown") | cleaned["Perpetrator"].eq("")
    cleaned.loc[unknown_or_empty, "Perpetrator"] = pd.NA
    cleaned["Perpetrator"] = cleaned["Perpetrator"].fillna("N/A")

    return cleaned


def aggregate_country_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned event-level GTD data to country-year level."""
    grouped = (
        df.groupby(["Country", "Year"], as_index=False)
        .agg(
            Region=("Region", safe_mode),
            Num_Attacks=("Country", "size"),
            Fatalities=("Fatalities", "sum"),
            Injuries=("Injuries", "sum"),
            Attack_Type_Mode=("Attack_Type", safe_mode),
            Target_Type_Mode=("Target_Type", safe_mode),
            Perpetrator_Mode=("Perpetrator", safe_mode),
        )
        .loc[:, FINAL_AGG_COLUMNS]
        .sort_values(["Country", "Year"], ignore_index=True)
    )
    grouped["Perpetrator_Mode"] = grouped["Perpetrator_Mode"].fillna("N/A")
    return grouped


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    cleaned_output_path = interim_dir / "gtd_cleaned.csv"
    agg_csv_output_path = interim_dir / "gtd_country_year.csv"
    agg_parquet_output_path = interim_dir / "gtd_country_year.parquet"

    raw_subset = load_selected_columns(input_path)
    cleaned_events = clean_event_level(raw_subset)
    country_year = aggregate_country_year(cleaned_events)

    cleaned_events.to_csv(cleaned_output_path, index=False)
    country_year.to_csv(agg_csv_output_path, index=False)
    country_year.to_parquet(agg_parquet_output_path, index=False)

    print("GTD cleaning pipeline completed successfully.")
    print(f"Input: {input_path}")
    print(f"Saved cleaned event-level data: {cleaned_output_path}")
    print(f"Saved country-year CSV: {agg_csv_output_path}")
    print(f"Saved country-year Parquet: {agg_parquet_output_path}")
    print(f"Cleaned shape: {cleaned_events.shape}")
    print(f"Country-year shape: {country_year.shape}")


if __name__ == "__main__":
    main()
