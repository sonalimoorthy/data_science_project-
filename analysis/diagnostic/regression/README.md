# Regression analysis: predict `Num_Attacks`

**Target:** `Num_Attacks`  
**Core predictors:** `GDP_per_capita`, `Population`, `Conflict_Level`, `Fatalities`, `Injuries`, plus **`Year`**.  
**Linear extras:** `Region` as **one-hot** (`drop_first=True`).

**Enriched predictors (tree model):** above numerics plus `Conflict`, `Conflict_Intensity_deaths` (missing filled with 0), `has_conflict_data`, `has_population_data`, `has_gdp_data`, and **one-hot** `Region`, `Attack_Type_Mode`, `Target_Type_Mode` (unknown test categories ignored).

**Data:** `data/final/master_country_year.csv` — rows with missing values in any column used by the script are dropped (same **3,298** complete rows as before after `Conflict_Intensity_deaths` imputation).

## Run

```bash
python analysis/diagnostic/regression/regression_num_attacks.py
```

## What the script does

1. **Train/test split:** 80% / 20%, `random_state=42`.
2. **Linear OLS:** 5 features; **+ Year**; **+ Region** dummies.
3. **Polynomial:** `StandardScaler` → `PolynomialFeatures(degree=2, include_bias=False)` → `LinearRegression` on the **six** numeric inputs (5 + Year).
4. **HistGradientBoostingRegressor** (sklearn): gradient-boosted trees on **enriched** numeric + encoded categoricals; optional **`log1p` target** via `TransformedTargetRegressor` (often better on **skewed counts**). Predictions are clipped at 0 before metrics.
5. **Time-only** `Year` linear / quadratic for coarse trend read.
6. **Metrics:** **R²** and **RMSE** on train and test (**test** is primary).

## R² by model and features (example run on current CSV)

Each row is one specification from `regression_num_attacks.py` (80/20 split, `random_state=42`). **Features** lists inputs before any transforms (OHE = one-hot encoded categoricals; polynomial adds squares/interactions on the scaled six numerics).

| Model | Features present | R² (train) | R² (test) |
|--------|------------------|------------|-----------|
| Linear (5, no Year) | `GDP_per_capita`, `Population`, `Conflict_Level`, `Fatalities`, `Injuries` | 0.759 | 0.753 |
| Linear (5 + Year) | above + `Year` | 0.759 | 0.753 |
| Linear (5 + Year + Region OHE) | above + `Region` (dummy columns) | 0.768 | 0.760 |
| Polynomial deg=2 (scaled) | same six numerics as “5 + Year”: `GDP_per_capita`, `Population`, `Conflict_Level`, `Fatalities`, `Injuries`, `Year` — degree-2 expansion after `StandardScaler` | 0.857 | 0.854 |
| HistGradientBoosting (raw *y*) | six numerics + `Conflict`, `Conflict_Intensity_deaths`, `has_conflict_data`, `has_population_data`, `has_gdp_data` + OHE `Region`, `Attack_Type_Mode`, `Target_Type_Mode` | 0.940 | 0.870 |
| HistGradientBoosting (log1p *y*) | same as previous row; target is `log1p(Num_Attacks)` while training | 0.963 | 0.887 |
| Time-only linear | `Year` only | 0.017 | 0.032 |
| Time-only quadratic | `Year`, `Year^2` | 0.025 | 0.052 |

Rounded to three decimals; same **3,298**-row sample and **test** fold as the other models.

## Model comparison (RMSE on test, same run)

| Model | RMSE (test) |
|------|-------------|
| Linear (5 features, no Year) | ~75.31 |
| Linear (5 + Year) | ~75.31 |
| Linear (5 + Year + Region OHE) | ~74.20 |
| Polynomial deg=2 (5 + Year, scaled) | ~57.88 |
| HistGradientBoosting (enriched + OHE, raw target) | **~54.6** |
| HistGradientBoosting (enriched + OHE, **log1p** target) | **~50.9** |

**Takeaways**

- **Trees + more columns** capture **non-linearities and interactions** that OLS/polynomials miss, so **test R² rises** and **test RMSE falls**.
- **`log1p(Num_Attacks)`** as the training target often helps **count-like**, **right-skewed** outcomes; metrics are still computed on **original** attack scale after `expm1`.
- **Train R²** for the boosted models is **much higher** than test — they are **flexible**; for reporting and deployment, **cross-validation**, **stronger regularization**, or **fewer iterations** would be the next checks to guard against **overfitting**.

## Answering the project questions

### 1) Linear regression — attacks from features (and time)

- The **multivariate linear** model with **Year** shows a **small positive coefficient on `Year`** (conditional on the other variables). That is **not** a proof of a global time trend.
- **Stronger fit** comes from **Region**, **polynomial** terms, and especially the **boosted-tree** specs above.

### 2) Polynomial vs richer models

- **Degree-2 polynomial** still improves a lot over plain linear models on the **same six numerics**.
- **HistGradientBoosting** with **extra fields** does **better still** on this split — terrorism dynamics are **not** fully captured by a low-degree polynomial.

### 3) Are attacks increasing over time?

- **Time-only** `Num_Attacks ~ Year` still shows a **positive** slope on **Year** on the training split, but **test R² stays low** (~0.03): year alone is a **weak** summary.

### 4) Is growth accelerating?

- **Time-only quadratic:** **`Year^2` positive** implies **upward curvature** in that toy fit; interpret as **shape**, not causation.

## Interpretation caveats

- **Predictive, not causal:** **Fatalities** / **Injuries** and GTD fields can be **co-determined** with attack counts.
- **Boosted trees** are **black-box** compared with OLS coefficients; they prioritize **accuracy** over simple interpretation.
- Re-run after updating `master_country_year.csv`.
