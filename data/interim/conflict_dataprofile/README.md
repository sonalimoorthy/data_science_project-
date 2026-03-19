# Conflict Data Profiling (Raw -> Interim)

This folder contains profiling outputs for the UCDP organized violence dataset.

## Raw source

- `data/raw/organizedviolencecy-251-csv/organizedviolencecy_v25_1.csv`
  - (Fallback path used by code if present: `data/raw/organizedviolencecy_v25_1.csv`)

## Pipeline used

- Profiling logic is implemented in `src/ucdp_data_profiling.py`.

## What was selected from raw

- `country_cy`
- `year_cy`
- `region_cy`
- `cumulative_total_deaths_in_orgvio_best_cy`

## Profiling checks performed

1. Basic dataset info (shape, columns, data types).
2. Data quality checks:
   - Missing value counts and percentages
   - Duplicate row count
   - Unique country count
   - Year range
3. Conflict deaths column checks:
   - Mean, min, max
   - Negative value count
4. Potential issue checks:
   - Rows with missing country or year
   - Count of zero-conflict years (`deaths == 0`)
   - Count of non-zero conflict years (`deaths > 0`)

## Files in this folder

- `conflict_profile_subset.csv` - direct selected-column subset from raw (no cleaning).
- `conflict_profile_report.txt` - plain text profiling report with quality and issue diagnostics.

## Notes

- Profiling is read-only: no cleaning, filling, dropping, or renaming is applied in the profiling script.
