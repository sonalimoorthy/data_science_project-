# Population + GDP Data Profiling (Raw -> Interim)

This folder contains profiling-only outputs for population and GDP per capita raw datasets.

## Raw sources

- Population:
  - `data/raw/population-with-un-projections/population-with-un-projections.csv`
  - (Fallback path used by code: `data/raw/population-with-un-projections.csv`)
- GDP per capita:
  - `data/raw/gdp-per-capita-maddison-project-database/gdp-per-capita-maddison-project-database.csv`
  - (Fallback path used by code: `data/raw/gdp-per-capita-maddison-project-database.csv`)

## Pipelines used

- Population profiling: `src/population_data_profiling.py`
- GDP profiling: `src/gdp_data_profiling.py`

## Population profiling checks

1. Selected `Entity`, `Year`, and detected population column variant.
2. Saved exact selected subset (no cleaning).
3. Reported:
   - Shape, columns, data types
   - Missing counts/percentages
   - Duplicate row count
   - Unique entity count
   - Year range
   - Missing population count
   - Potential non-country entries (OWID code heuristic + examples)

## GDP profiling checks

1. Selected `Entity`, `Year`, and detected GDP column variant.
2. Saved exact selected subset (no cleaning).
3. Reported:
   - Shape, columns, data types
   - Missing counts/percentages
   - Duplicate row count
   - Unique entity count
   - Year range
   - GDP summary stats (mean/min/max)
   - Negative GDP value count
   - Potential non-country entries and missing GDP count

## Files in this folder

- `population_profile_subset.csv` - selected population columns from raw.
- `population_profile_report.txt` - population profiling diagnostics report.
- `gdp_profile_subset.csv` - selected GDP columns from raw.
- `gdp_profile_report.txt` - GDP profiling diagnostics report.

## Notes

- Both scripts are profiling-only and intentionally avoid cleaning, filling, filtering, or renaming operations.
