# Population + GDP Data Cleaning (Raw -> Interim)

This folder stores cleaned interim datasets for population and GDP per capita, prepared for country-year integration.

## Raw sources

- Population:
  - `data/raw/population-with-un-projections/population-with-un-projections.csv`
  - (Fallback path used by code: `data/raw/population-with-un-projections.csv`)
- GDP per capita (Maddison):
  - `data/raw/gdp-per-capita-maddison-project-database/gdp-per-capita-maddison-project-database.csv`
  - (Fallback path used by code: `data/raw/gdp-per-capita-maddison-project-database.csv`)

## Pipelines used

- Population cleaning: `src/population_cleaning_pipeline.py`
- GDP cleaning: `src/gdp_cleaning_pipeline.py`

## Population cleaning steps

1. Selected `Entity`, `Year`, and the best available population column variant.
2. Renamed fields to:
   - `Entity` -> `Country`
   - selected population column -> `Population`
3. Trimmed country names.
4. Dropped rows with missing/empty `Country` or missing `Year`.
5. Converted `Year` to integer; converted `Population` to numeric; dropped missing population rows.
6. Removed non-country rows:
   - Prefer ISO-like `Code` filtering when available.
   - Fallback exclusion of aggregate entities (World, continents, etc.).
7. Removed duplicate `Country` + `Year` rows (kept first), then sorted.

## GDP cleaning steps

1. Selected `Entity`, `Code`, `Year`, `GDP per capita`.
2. Renamed fields to:
   - `Entity` -> `Country`
   - `GDP per capita` -> `GDP_per_capita`
3. Trimmed country names.
4. Dropped rows with missing/empty `Country` or missing `Year`.
5. Converted `Year` to integer and `GDP_per_capita` to numeric.
6. Removed non-country aggregate rows using ISO-like `Code` filtering.
7. Removed duplicate `Country` + `Year` rows (kept first after stable sorting).
8. Filled GDP gaps via within-country linear interpolation (both directions), based on configured strategy.
9. Sorted final output by `Country`, `Year`.

## Files in this folder

- `population_cleaned.csv` - cleaned country-year population series.
- `gdp_cleaned.csv` - cleaned country-year GDP per capita series.

## Notes

- GDP missing-value behavior is configurable in code (`GDP_MISSING_STRATEGY`), currently set to interpolation.
