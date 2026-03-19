"""
UCDP organized violence cleaning + structuring pipeline.

Outputs:
- data/interim/conflict_cleaned.csv
"""

from pathlib import Path

import pandas as pd


RAW_COLUMNS = [
    "country_cy",
    "year_cy",
    "region_cy",
    "cumulative_total_deaths_in_orgvio_best_cy",
]

RENAME_MAP = {
    "country_cy": "Country",
    "year_cy": "Year",
    "region_cy": "Region",
    "cumulative_total_deaths_in_orgvio_best_cy": "Conflict_Intensity_deaths",
}

FINAL_COLUMNS = [
    "Country",
    "Year",
    "Region",
    "Conflict",
    "Conflict_Level",
    "Conflict_Intensity_deaths",
]


def resolve_input_path(project_root: Path) -> Path:
    """
    Resolve UCDP raw file location from expected candidates.
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
        "UCDP raw file not found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}"
    )


def classify_conflict_level(x: float) -> int:
    """Classify conflict intensity into 0-10 ordinal scale."""
    if x == 0:
        return 0
    elif x <= 10:
        return 1
    elif x <= 25:
        return 2
    elif x <= 50:
        return 3
    elif x <= 100:
        return 4
    elif x <= 250:
        return 5
    elif x <= 500:
        return 6
    elif x <= 1000:
        return 7
    elif x <= 2500:
        return 8
    elif x <= 5000:
        return 9
    else:
        return 10


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    output_path = interim_dir / "conflict_cleaned.csv"

    # 1-3) Load selected columns and rename.
    df = pd.read_csv(input_path, usecols=RAW_COLUMNS, low_memory=False)
    df = df.rename(columns=RENAME_MAP)

    # 4) Clean core fields.
    df["Country"] = df["Country"].astype("string").str.strip()
    df["Region"] = df["Region"].astype("string").str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Conflict_Intensity_deaths"] = pd.to_numeric(
        df["Conflict_Intensity_deaths"], errors="coerce"
    )

    # Drop rows where Country or Year is missing.
    df = df.dropna(subset=["Country", "Year"])

    # Convert Year to integer and fill missing conflict intensity with 0.
    df["Year"] = df["Year"].astype("int64")
    df["Conflict_Intensity_deaths"] = df["Conflict_Intensity_deaths"].fillna(0)

    # 5) Binary conflict flag.
    df["Conflict"] = (df["Conflict_Intensity_deaths"] > 0).astype(int)

    # 6-8) Conflict level classification.
    df["Conflict_Level"] = df["Conflict_Intensity_deaths"].apply(
        classify_conflict_level
    )

    # 9) Handle duplicates by Country-Year with requested aggregations.
    df_cleaned = (
        df.groupby(["Country", "Year"], as_index=False)
        .agg(
            Region=("Region", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else pd.NA),
            Conflict_Intensity_deaths=("Conflict_Intensity_deaths", "sum"),
            Conflict=("Conflict", "max"),
            Conflict_Level=("Conflict_Level", "max"),
        )
        .loc[:, FINAL_COLUMNS]
        .sort_values(["Country", "Year"], ignore_index=True)
    )

    # 11) Save cleaned structured output.
    df_cleaned.to_csv(output_path, index=False)

    print("UCDP cleaning pipeline completed successfully.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Final shape: {df_cleaned.shape}")


if __name__ == "__main__":
    main()
