# GTD Data Cleaning (Raw -> Interim)

This folder contains cleaned and aggregated Global Terrorism Database (GTD) interim outputs.

## Raw source

- `data/raw/archive/globalterrorismdb_0718dist.csv`
  - (Fallback path used by code if present: `data/raw/globalterrorismdb_0718dist.csv`)

## Pipeline used

- Cleaning logic is implemented in `src/gtd_cleaning_pipeline.py`.

## What was selected from raw

- `iyear`
- `country_txt`
- `region_txt`
- `attacktype1_txt`
- `targtype1_txt`
- `gname`
- `nkill`
- `nwound`

## Cleaning and transformation steps

1. Renamed columns to standardized names:
   - `iyear` -> `Year`
   - `country_txt` -> `Country`
   - `region_txt` -> `Region`
   - `attacktype1_txt` -> `Attack_Type`
   - `targtype1_txt` -> `Target_Type`
   - `gname` -> `Perpetrator`
   - `nkill` -> `Fatalities`
   - `nwound` -> `Injuries`
2. Trimmed whitespace in text columns.
3. Converted `Year`, `Fatalities`, and `Injuries` to numeric.
4. Dropped rows missing `Country` or `Year`.
5. Converted `Year` to integer type.
6. Filled missing `Fatalities` and `Injuries` with `0`.
7. Normalized perpetrator values:
   - `Unknown` or empty -> missing
   - Missing -> `N/A`
8. Built country-year aggregation with:
   - `Num_Attacks` (event count)
   - Sum of fatalities and injuries
   - Mode for region, attack type, target type, perpetrator
9. Sorted by `Country`, `Year`.

## Files in this folder

- `gtd_cleaned.csv` - cleaned event-level GTD records.
- `gtd_country_year.csv` - country-year aggregated GTD metrics.
- `gtd_country_year.parquet` - same country-year dataset in Parquet format.

## Notes

- The code attempts UTF-8 read first, then falls back to Latin-1 for GTD source encoding compatibility.
