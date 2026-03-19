"""
GDP per capita (Maddison Project Database) cleaning + structuring pipeline.

Output:
- data/interim/population+gdp_dataclean/gdp_cleaned.csv
"""

from pathlib import Path
import re

import pandas as pd


# Choose GDP missing-value handling:
# - "interpolate": interpolate within each country over time
# - "none": leave missing values as NaN
GDP_MISSING_STRATEGY = "interpolate"


def resolve_input_path(project_root: Path) -> Path:
    """Resolve GDP input path from expected project layouts."""
    candidates = [
        project_root / "data" / "raw" / "gdp-per-capita-maddison-project-database.csv",
        project_root
        / "data"
        / "raw"
        / "gdp-per-capita-maddison-project-database"
        / "gdp-per-capita-maddison-project-database.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find gdp-per-capita-maddison-project-database.csv.")


def is_country_code(code: object) -> bool:
    """
    Identify country rows using ISO-like 3-letter uppercase codes.
    Excludes OWID aggregate rows.
    """
    if pd.isna(code):
        return False
    code_str = str(code).strip()
    if code_str.startswith("OWID_"):
        return False
    return bool(re.fullmatch(r"[A-Z]{3}", code_str))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "interim" / "population+gdp_dataclean"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(project_root)
    output_path = output_dir / "gdp_cleaned.csv"

    # Load source with Code retained for non-country filtering.
    df = pd.read_csv(input_path, usecols=["Entity", "Code", "Year", "GDP per capita"], low_memory=False)

    # Rename columns.
    df = df.rename(
        columns={
            "Entity": "Country",
            "Year": "Year",
            "GDP per capita": "GDP_per_capita",
        }
    )

    # Strip country names.
    df["Country"] = df["Country"].astype("string").str.strip()

    # Drop rows with missing Country or Year.
    df = df.dropna(subset=["Country", "Year"])
    df = df[df["Country"] != ""]

    # Convert types.
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype("int64")
    df["GDP_per_capita"] = pd.to_numeric(df["GDP_per_capita"], errors="coerce")

    # Remove non-country rows (World, regions, aggregates).
    df = df[df["Code"].apply(is_country_code)]
    df = df.drop(columns=["Code"])

    # Ensure one row per Country-Year and remove duplicates.
    # If duplicates exist, keep first after stable sorting.
    df = df.sort_values(["Country", "Year"], kind="stable")
    df = df.drop_duplicates(subset=["Country", "Year"], keep="first")

    # Missing GDP handling.
    if GDP_MISSING_STRATEGY == "interpolate":
        df["GDP_per_capita"] = (
            df.sort_values(["Country", "Year"], kind="stable")
            .groupby("Country")["GDP_per_capita"]
            .transform(
                lambda s: s.interpolate(
                    method="linear",
                    limit_direction="both",
                )
            )
        )
    elif GDP_MISSING_STRATEGY == "none":
        pass
    else:
        raise ValueError(
            "GDP_MISSING_STRATEGY must be either 'interpolate' or 'none'."
        )

    # Final column order.
    df = df[["Country", "Year", "GDP_per_capita"]]
    df = df.sort_values(["Country", "Year"], ignore_index=True)

    df.to_csv(output_path, index=False)

    print("GDP cleaning pipeline completed successfully.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"GDP missing strategy: {GDP_MISSING_STRATEGY}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
