# Root cause identification (associational drivers)

This folder summarizes **patterns** linked to **`Num_Attacks`** in `data/final/master_country_year.csv`. Language like “root cause” here means **statistical association** in the merged panel, **not** proof that a feature **causes** terrorism (reverse causation, reporting, and country-level clustering are all plausible).

## Run

```bash
python analysis/diagnostic/rootCause/root_cause_drivers.py
```

## Outputs

| File | Contents |
|------|----------|
| `region_group_stats.csv` | Per **Region**: row count, mean/median `Num_Attacks`, mean `Conflict_Level`, mean `Conflict` (binary) |
| `attack_type_group_stats.csv` | Per **Attack_Type_Mode**: counts and mean/median attacks **in rows** with that modal type |
| `linear_coef_ranking.csv` | All regression terms sorted by **\|coefficient\|** |

## Methods (matches requested steps)

1. **Regression coefficients** — `LinearRegression` on **standardized** numeric inputs plus **one-hot `Region`** (`drop_first=True`). Coefficients on numerics are in **SD units of X** per one SD change; region columns are **relative to the dropped baseline region**.
2. **Rank features** — Sort by **absolute coefficient** (magnitude of linear association in this specification).
3. **Strongest influencing factor** — The top row in `linear_coef_ranking.csv` (also printed). In recent runs this is often **`Fatalities`** or **`Injuries`**, because they move with the **same severe country-years** as attack counts (co-measurement / outcome linkage, not a policy lever).

## Group comparisons (context for “why some regions are higher”)

- **Regions** with higher **mean `Num_Attacks`** per country-year row often also show **higher mean `Conflict_Level`** and **`Conflict` share** in this table. That aligns with a **story** of **political instability / armed conflict exposure** correlating with more GTD-recorded violence — but it is still **descriptive**.
- **Attack_Type_Mode** summaries show which **modal attack categories** sit in **higher-attack rows** (e.g. bombing-heavy contexts). That is a weak proxy for **“active groups”** (perpetrator networks are not modeled here; `Perpetrator_Mode` has very high cardinality and is not used in this script).

## How to read the results

- **Coefficients ≠ causes.** A large |coef| means the linear model **uses** that column to predict attacks **after** holding others constant **in a linear way** — not that changing GDP “reduces terrorism” by that amount.
- **Panel dependence:** Multiple rows per country violate independence assumptions; formal inference would need **cluster-robust** errors or **hierarchical** models.
- **Confounding:** Wealth, population, time (`Year`), and conflict indicators are intertwined; the model is a **lens**, not a full structural explanation.

Re-run after updating `master_country_year.csv` to refresh CSVs and console output.
