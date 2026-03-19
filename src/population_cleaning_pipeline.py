"""
Population cleaning + structuring pipeline.

Output:
- data/interim/population_cleaned.csv
"""

from pathlib import Path
import re

import pandas as pd


def resolve_input_path(project_root: Path) -> Path:
    """
    Resolve population input path from expected locations.
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
    Resolve population column name across OWID export variants.
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
        "Population column not found. Expected one of: " + ", ".join(candidates)
    )


def is_country_code(code: object) -> bool:
    """
    Check if a code looks like a country code (ISO-like 3 uppercase letters),
    excluding OWID aggregate codes.
    """
    if pd.isna(code):
        return False
    code_str = str(code).strip()
    if code_str.startswith("OWID_"):
        return False
    return bool(re.fullmatch(r"[A-Z]{3}", code_str))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    output_path = interim_dir / "population_cleaned.csv"

    full_df = pd.read_csv(input_path, low_memory=False)
    population_col = resolve_population_column(full_df.columns.tolist())

    # Select required columns (plus Code for non-country filtering).
    required_cols = ["Entity", "Year", population_col]
    read_cols = required_cols + (["Code"] if "Code" in full_df.columns else [])
    df = full_df[read_cols].copy()

    # Rename to final field names.
    rename_map = {
        "Entity": "Country",
        "Year": "Year",
        population_col: "Population",
    }
    df = df.rename(columns=rename_map)

    # Strip whitespace from country names.
    df["Country"] = df["Country"].astype("string").str.strip()

    # Drop rows with missing Country or Year.
    df = df.dropna(subset=["Country", "Year"])
    df = df[df["Country"] != ""]

    # Convert Year to integer.
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype("int64")

    # Convert Population to numeric.
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    df = df.dropna(subset=["Population"])

    # Remove non-country rows (World, continents, income groups, etc.).
    # Prefer code-based filtering when available.
    if "Code" in df.columns:
        df = df[df["Code"].apply(is_country_code)]
        df = df.drop(columns=["Code"])
    else:
        non_country_entities = {
            "World",
            "Africa",
            "Asia",
            "Europe",
            "North America",
            "South America",
            "Oceania",
        }
        df = df[~df["Country"].isin(non_country_entities)]

    # Ensure one row per Country-Year and remove duplicates.
    df = df.drop_duplicates(subset=["Country", "Year"], keep="first")
    df = df.sort_values(["Country", "Year"], ignore_index=True)

    df.to_csv(output_path, index=False)

    print("Population cleaning pipeline completed successfully.")
    print(f"Input: {input_path}")
    print(f"Selected population column: {population_col}")
    print(f"Output: {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
