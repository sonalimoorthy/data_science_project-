# Conflict Data Cleaning (Raw -> Interim)

This folder contains cleaned UCDP organized violence data at country-year level.

## Raw source

- `data/raw/organizedviolencecy-251-csv/organizedviolencecy_v25_1.csv`
  - (Fallback path used by code if present: `data/raw/organizedviolencecy_v25_1.csv`)

## Pipeline used

- Cleaning logic is implemented in `src/ucdp_cleaning_pipeline.py`.

## What was selected from raw

- `country_cy`
- `year_cy`
- `region_cy`
- `cumulative_total_deaths_in_orgvio_best_cy`

## Cleaning and transformation steps

1. Renamed fields to standardized names:
   - `country_cy` -> `Country`
   - `year_cy` -> `Year`
   - `region_cy` -> `Region`
   - `cumulative_total_deaths_in_orgvio_best_cy` -> `Conflict_Intensity_deaths`
2. Trimmed whitespace in text columns (`Country`, `Region`).
3. Converted `Year` and `Conflict_Intensity_deaths` to numeric.
4. Dropped rows with missing `Country` or `Year`.
5. Converted `Year` to integer type.
6. Filled missing `Conflict_Intensity_deaths` with `0`.
7. Created binary `Conflict` flag (`1` when deaths > 0, else `0`).
8. Created ordinal `Conflict_Level` (0-10) from intensity buckets.
9. Resolved duplicate country-year rows by grouping and aggregating:
   - `Region`: mode
   - `Conflict_Intensity_deaths`: sum
   - `Conflict`: max
   - `Conflict_Level`: max
10. Sorted final output by `Country`, `Year`.

## Files in this folder

- `conflict_cleaned.csv` - cleaned country-year conflict dataset for downstream merge/model steps.

## Notes

- The script writes `data/interim/conflict_cleaned.csv`; this folder stores the corresponding cleaned interim artifact used in the project workflow.
