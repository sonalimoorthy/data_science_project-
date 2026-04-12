# Final Table Documentation (`data/final`)

## Project output summary

The final output of this workflow is a merged, analysis-ready table named `master_country_year`. It is stored in:

- `master_country_year.csv`
- `master_country_year.parquet`

The table is a Country-Year panel, where each row represents one country in one year. The primary key is:

- `Country` + `Year`

This structure is suitable for trend analysis, country comparisons, and panel modeling.

## Source datasets used

The final table is created from cleaned interim datasets:

1. GTD base table: `data/interim/gtd_dataclean/gtd_country_year.csv`
2. Conflict table: `data/interim/conflict_dataclean/conflict_cleaned.csv`
3. Population table: `data/interim/population+gdp_dataclean/population_cleaned.csv`
4. GDP table: `data/interim/population+gdp_dataclean/gdp_cleaned.csv`
5. Country harmonization map: `data/interim/country_name_mapping.csv`

## Integration and join method

The pipeline builds the final table in four steps:

1. Standardize `Country` and `Year` fields in all sources.
2. Harmonize country names using `country_name_mapping.csv`.
3. Use GTD Country-Year as the base table.
4. Merge other datasets with LEFT JOIN on `Country` + `Year`.

Join order:

- `GTD LEFT JOIN Conflict`
- `LEFT JOIN Population`
- `LEFT JOIN GDP`

Reason for LEFT JOIN:

- Keeps all GTD Country-Year rows.
- Adds values from other datasets when a match exists.
- Avoids dropping GTD observations due to missing side-table matches.

## Final columns by source

### GTD-derived columns

- `Country`
- `Year`
- `Region`
- `Num_Attacks`
- `Fatalities`
- `Injuries`
- `Attack_Type_Mode`
- `Target_Type_Mode`
- `Perpetrator_Mode`

### Conflict-derived columns

- `Conflict`
- `Conflict_Level`
- `Conflict_Intensity_deaths`
- `Conflict_Region`

### Population/GDP columns

- `Population`
- `GDP_per_capita`

### Pipeline-derived helper columns

- `has_conflict_data`: `1` if a conflict row matched, else `0`
- `has_population_data`: `1` if a population row matched, else `0`
- `has_gdp_data`: `1` if a GDP row matched, else `0`
- `country_mapping_applied`: `1` if country name was standardized via mapping, else `0`

These helper columns are quality/traceability indicators for merge coverage.

## Supporting files in this folder

- `final_table_schema.json`: expected schema (columns, order, dtypes)
- `merge_diagnostics.json`: quality summary (rows, key checks, match rates)

## How to read `final_table_schema.json`

The schema file is the data contract for `master_country_year`. It tells you exactly what the final table should look like.

Main keys in the schema:

- `table_name`: expected final table name.
- `primary_key`: uniqueness definition (`Country` + `Year`).
- `column_order`: exact output column sequence.
- `required_columns`: columns that must exist in every valid build.
- `dtypes`: expected data type for each column (for example `string`, `int64`, `float64`).

Why this matters:

- Prevents accidental column changes between runs.
- Makes handoff and reproducibility easier.
- Gives a clear reference for analysis code and model inputs.

## How to read `merge_diagnostics.json`

The diagnostics file is a quick health report of the final merge run.

Fields and meaning:

- `row_count`: total number of rows in `master_country_year`.
- `distinct_countries`: number of unique countries in the final table.
- `year_min` and `year_max`: year coverage of the final panel.
- `duplicate_country_year_keys`: should be `0`; non-zero means key integrity issue.
- `match_rates.conflict`: share of GTD rows that found a conflict match.
- `match_rates.population`: share of GTD rows that found a population match.
- `match_rates.gdp`: share of GTD rows that found a GDP match.

How to interpret quickly:

- If `duplicate_country_year_keys` is `0`, key integrity is good.
- Higher match rates mean better coverage from side datasets.
- Compare `row_count` with the final CSV row count to confirm output consistency.

## Reproducibility

From project root, run:

```bash
py -3 src/final_table_pipeline.py
```

This regenerates all files in `data/final`.

Plots, correlation checks, hypothesis tests, regressions, and risk classification now live under `analysis/` (see each subfolder’s `README.md` and Python entrypoints there). They read `data/final/master_country_year.csv` the same way this pipeline produces it.

## Validation checklist

Before sharing, verify:

- `master_country_year.csv` and `.parquet` both exist
- `Country + Year` contains no duplicates
- columns match `final_table_schema.json`
- row count in `merge_diagnostics.json` matches the final table

---

When sharing this project, include all files in `data/final` together so the dataset, schema, and diagnostics stay linked.
