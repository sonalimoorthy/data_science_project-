# GTD Data Profiling (Raw -> Interim)

This folder contains profiling outputs for the Global Terrorism Database (GTD).

## Raw source

- `data/raw/archive/globalterrorismdb_0718dist.csv`
  - (Fallback path used by code if present: `data/raw/globalterrorismdb_0718dist.csv`)

## Pipeline used

- Profiling logic is implemented in `src/gtd_data_profiling.py`.

## What was selected from raw

- `iyear`
- `country_txt`
- `region_txt`
- `attacktype1_txt`
- `targtype1_txt`
- `gname`
- `nkill`
- `nwound`

## Profiling checks performed

1. Basic dataset info (shape, selected columns, data types).
2. Data quality checks:
   - Missing value counts and percentages
   - Duplicate row count
   - Unique value count per column
3. Frequency checks:
   - Top 10 values for country, region, attack type, target type, and perpetrator.
4. Numeric checks (`nkill`, `nwound`):
   - Mean, median, min, max
   - Negative value count
5. Potential issue checks:
   - Missing country/year rows
   - Null fatalities/injuries rows
   - Rows with perpetrator marked as `Unknown`

## Files in this folder

- `gtd_profile_subset.csv` - selected raw columns only (no cleaning).
- `gtd_profile_report.txt` - text report summarizing quality checks and potential issues.

## Notes

- Profiling does not modify data values.
- The profiler supports GTD encoding differences by using Latin-1 fallback when UTF-8 read fails.
