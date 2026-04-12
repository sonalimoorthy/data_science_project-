# Descriptive statistics: `master_country_year`

This folder holds a small pandas script and documented results for the cleaned country–year panel (`data/final/master_country_year.csv`).

## How to run

From the repository root:

```bash
python analysis/descriptive/stat/descriptive_stats_master_country_year.py
```

The script loads `master_country_year.csv`, prints summary statistics, then tables of mean `Num_Attacks` by country and by region.

## Data definition

- Each row is one **country in one year**.
- **Mean / median / std / min / max** are computed over **all rows** in the file for `Num_Attacks`, `Fatalities`, and `Injuries`.
- **Average number of attacks by country** = mean of `Num_Attacks` over all country–year rows for that country.
- **Average number of attacks by region** = mean of `Num_Attacks` over all rows whose `Region` matches.

## Results (computed from current `master_country_year.csv`)

**Panel size:** 3,762 rows; **205** distinct countries.

### Summary statistics

|        | Num_Attacks | Fatalities | Injuries |
|--------|------------:|-----------:|---------:|
| mean   | 48.2964 | 109.4811 | 139.2528 |
| median | 5.0000  | 3.0000   | 4.0000   |
| std    | 179.9403 | 551.9795 | 778.6072 |
| min    | 1.0000  | 0.0000   | 0.0000   |
| max    | 3933.0000 | 13965.0000 | 16804.0000 |

### Average `Num_Attacks` by region

| Region | Avg Num_Attacks |
|--------|----------------:|
| South Asia | 193.02 |
| Middle East & North Africa | 89.02 |
| Southeast Asia | 54.76 |
| South America | 50.21 |
| Central America & Caribbean | 38.74 |
| North America | 28.80 |
| Western Europe | 27.69 |
| Sub-Saharan Africa | 22.94 |
| Eastern Europe | 16.59 |
| East Asia | 7.10 |
| Central Asia | 5.31 |
| Australasia & Oceania | 3.81 |

### Average `Num_Attacks` by country

There are 205 countries; the full table is long. Run the script above to print the complete sorted list. Examples:

- **Highest** (by mean attacks per country–year row): Iraq (~665.8), Afghanistan (~397.8), Pakistan (~334.1), India (~284.8), El Salvador (~212.8).
- Many countries with sparse years have averages near **1.0**.

Re-run the script after refreshing `master_country_year.csv` to update these numbers.
