# Categorical aggregation: `master_country_year`

Aggregations on the country–year panel (`data/final/master_country_year.csv`). Each row is one country in one year; `Num_Attacks` and `Fatalities` in that row describe that year (and country). Grouping sums or averages those values across rows in each group.

## How to run

From the repository root:

```bash
python analysis/descriptive/categoricalAGG/categorical_agg_master_country_year.py
```

## What each task measures

1. **Total attacks per country** — `sum(Num_Attacks)` over all years for each `Country`. Sorted by total descending.
2. **Total attacks per region** — `sum(Num_Attacks)` over all country–year rows in each `Region`. Sorted by total descending.
3. **Average fatalities per region** — `mean(Fatalities)` over all rows in each `Region` (equal weight per country–year row). Sorted by average descending.
4. **Top 10 countries by attacks** — same as (1), keep the first 10 rows after sorting descending.

Re-run the script after updating the CSV to refresh numbers.

## Results (from current `master_country_year.csv`)

Panel: **3,762** rows; **205** countries; **12** regions.

### 2. Total attacks per region (descending)

| Region | Total_Attacks |
|--------|--------------:|
| Middle East & North Africa | 50,474 |
| South Asia | 44,974 |
| South America | 18,978 |
| Sub-Saharan Africa | 17,550 |
| Western Europe | 16,639 |
| Southeast Asia | 12,485 |
| Central America & Caribbean | 10,344 |
| Eastern Europe | 5,144 |
| North America | 3,456 |
| East Asia | 802 |
| Central Asia | 563 |
| Australasia & Oceania | 282 |

### 3. Average fatalities per region (descending)

| Region | Avg_Fatalities |
|--------|---------------:|
| South Asia | 434.845494 |
| Middle East & North Africa | 242.754850 |
| Central America & Caribbean | 107.520599 |
| Sub-Saharan Africa | 102.465359 |
| South America | 76.320106 |
| Southeast Asia | 68.583333 |
| North America | 40.966667 |
| Eastern Europe | 23.919355 |
| Western Europe | 11.138103 |
| East Asia | 10.194690 |
| Central Asia | 9.433962 |
| Australasia & Oceania | 2.027027 |

### 4. Top 10 countries by total attacks (descending)

| Country | Total_Attacks |
|---------|--------------:|
| Iraq | 24,636 |
| Pakistan | 14,368 |
| Afghanistan | 12,731 |
| India | 11,960 |
| Colombia | 8,306 |
| Philippines | 6,908 |
| Peru | 6,096 |
| El Salvador | 5,320 |
| United Kingdom | 5,235 |
| Turkey | 4,292 |

### 1. Total attacks per country

The script prints all **205** countries sorted by `Total_Attacks` descending. Highest totals after the top 10 include Nigeria, Thailand, Yemen, and Spain (run the script for the full table).
