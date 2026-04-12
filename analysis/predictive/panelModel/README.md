# Panel predictive models (country–year)

`panel_forecast_and_risk.py` upgrades the **global-total** baseline to a **balanced country–year panel**, adds **feature engineering**, uses **Random Forest vs Ridge** on **direct multi-horizon targets** (no iterative error chaining), adds **risk classification**, **feature importance**, **country-level errors**, and **KMeans** country clusters.

## Run

```bash
python analysis/predictive/panelModel/panel_forecast_and_risk.py
```

## 1. Panel construction

- **Expand** every country to every calendar year between the dataset min and max year.
- **`Num_Attacks`:** `0` where the country had no row (true zero years).
- **Other numerics:** `ffill` / `bfill` **within country**, then remaining `NaN` → `0`.
- **Region:** first / mode from observed rows per country.

## 2. Features (at time *t*) — **no same-year leakage**

Predictors use **only information available from prior years** (or calendar `Year` and `Region`). Same-year **Fatalities**, **Injuries**, and same-year **GDP / Population / Conflict_Level / Conflict** are **not** used as features (they would conflate attack outcomes or same-year context with the prediction target).

| Feature | Description |
|--------|-------------|
| `num_attacks_lag1`, `num_attacks_lag2` | Prior-year attack totals (early years → 0) |
| `num_attacks_roll3` | 3-year rolling mean of **past** attacks only: `shift(1)` then `rolling(3)` within country |
| `log_gdp_lag1`, `log_pop_lag1` | `log1p` of **prior-year** GDP per capita and population |
| `conflict_level_lag1`, `conflict_lag1` | Prior-year conflict level and conflict indicator |
| `gdp_x_conflict_lag1` | `gdp_per_capita_lag1 * conflict_level_lag1` |
| `Year` | Calendar year (trend / period effects) |
| `Region` | One-hot (`drop_first`, `handle_unknown='ignore'`) |

## 3. Direct multi-horizon regression (not iterative)

Separate targets, **separate** models (no feeding *t+1* prediction into *t+2*):

- `y_plus1` = next year’s `Num_Attacks` (same country)  
- `y_plus5` = attacks five years ahead  
- `y_plus10` = ten years ahead  

**Train:** all rows with `Year <= 2005` (so future years through **2017** exist for evaluation).  
**Test windows** (overlap with valid shifted targets):

- **t+1:** years **2006–2016**  
- **t+5:** **2006–2012**  
- **t+10:** **2006–2007** only (last year in file is 2017)

Each horizon fits **RandomForestRegressor** and **RidgeCV** on the same `ColumnTransformer` (scaled numerics + OHE).

### Time-based holdout vs **GroupKFold** (by country)

- **Temporal split:** train `Year <= 2005`, test on later years (as above). Countries appear in **both** train and test → RMSE can be **optimistic** because the model indirectly sees country-specific patterns.
- **Grouped CV:** **`GroupKFold`** with `groups=Country` — a country never appears in both train and test **within a fold**. Mean RMSE ± std across folds is printed and saved to `outputs/direct_horizon_groupkfold_rmse.csv`. This is usually **stricter** and better aligned with “generalize to new countries.”

Re-run the script for current numbers after any feature or data change; do not treat old example RMSEs as fixed benchmarks.

## 4. Outputs

All artifacts (**CSV**, **PNG**, and per-country plots under **`outputs/country_forecast_plots/`**) are written when you run the script; the `outputs/` tree is created automatically.

| File | Content |
|------|---------|
| `outputs/direct_horizon_rmse_comparison.csv` | RMSE RF vs Ridge on **raw** target, *n* train/test per horizon (time split) |
| `outputs/direct_horizon_rmse_original_vs_log.csv` | Same split: raw vs **log1p** target (RMSE always on **original** attack scale after `expm1`) |
| `outputs/direct_horizon_groupkfold_rmse.csv` | GroupKFold by **Country**: mean/std RMSE per horizon (RF + Ridge), raw target |
| `outputs/direct_horizon_groupkfold_rmse_original_vs_log.csv` | GroupKFold: raw vs log-target RMSE (original scale) |
| `outputs/country_rmse_h1_test.csv` | Per-country RMSE on **t+1** test (RF, raw target) |
| `outputs/country_rmse_h1_test_log_target.csv` | Per-country RMSE on **t+1** test (RF, log target + inverse) |
| `outputs/feature_importance_h1.png` | Top RF importances for **t+1** model (raw target) |
| `outputs/feature_importance_h1_log_target.png` | Top RF importances for **t+1** (trained on `log1p` target) |
| `outputs/confusion_matrix_risk.png` | Risk classifier confusion matrix |
| `outputs/kmeans_elbow.png` / `outputs/kmeans_elbow.csv` | Inertia vs *k* for elbow (k=2…10) |
| `outputs/country_clusters_kmeans.csv` | Country profiles + `cluster` (k=4) |
| `outputs/country_forward_forecasts_rf.csv` | Per country: anchor year (last panel year, **2017** here) and RF predictions for **t+1, t+5, t+10** calendar years |
| `outputs/country_forecast_plots/*.png` | One figure per country: historical `Num_Attacks`, vertical line at anchor, three forecast points (not observed in the CSV) |

### Forward forecasts (deploy / projection)

The script fits **three Random Forest regressors** using **only rows with `Year <= 2005`** (same cutoff as regression training), with target **`log1p(Num_Attacks)`**, then maps forecasts back to the count scale with **`expm1`** (and clips log-space outputs to the training max before inverse, matching the backtest path). It applies them to the **last panel row per country** (typically **2017**). That answers “what would we project if we froze training in 2005?” — not “retrain on all history including the 2010s.” Projected years **2018 / 2022 / 2027** are **not** in the CSV; **t+1**, **t+5**, **t+10** use **separate** models (not a recursive chain).

## 5. Risk classification

- **Label:** `high_risk = 1` if `Num_Attacks` **>** **median of original GTD rows** (not the expanded panel median, which is dominated by zeros).  
- **Train:** `Year <= 2010` | **Test:** `Year > 2010`  
- **Model:** `RandomForestClassifier` (`class_weight='balanced'`)  
- **Metrics:** accuracy, precision/recall/F1, confusion matrix, `classification_report`

### Example (current run)

- Threshold ≈ **5** attacks (median of raw rows).  
- Test **accuracy ~0.97**, precision/recall on high class **~0.93 / ~0.94**.

## 6. KMeans (unsupervised)

Per country: **mean attacks**, **mean fatalities**, **OLS slope** of attacks vs year. **StandardScaler** + **KMeans k=4**. Elbow files live under **`outputs/`**; one cluster often still holds most countries (sparse violence in many panels).

## 7. Relation to `regressionModel/`

The **global** lag + Ridge script remains a **simple teaching baseline**. This **panel** workflow matches the “make it good” checklist: **country structure**, **richer features**, **trees + linear compare**, **direct horizons**, **risk class**, **importance**, **errors by country**, **clustering**.

## Improvement — log target for regression

For skewed attack counts, the script compares training on the **raw** target versus **`y' = log(Num_Attacks + 1)`**, implemented as **`np.log1p(Num_Attacks)`**, using the same features and splits. Models are fit on `y'`; predicted values in log space are mapped back with **`np.expm1`**, and **RMSE is always computed on the original count scale** so numbers stay comparable.

**Stability (Ridge / linear models):** In log space, Ridge can extrapolate beyond the training range of `y'`, which would make **`expm1`** of those predictions explode. Before inverse transform, predictions are **clipped** to `[0, log(1 + max y_train)]` per horizon and fold. That caps impossible “infinite count” forecasts while keeping the comparison fair; it also implies forecasts cannot exceed the **largest training count** for that horizon unless you change the clip rule.

### Time split (train `Year <= 2005`) — illustrative results

Re-run the script after data changes; values below come from the current `master_country_year.csv` pipeline.

| Horizon | RMSE RF (raw) | RMSE RF (log target) | RMSE Ridge (raw) | RMSE Ridge (log target) |
|--------|---------------|----------------------|------------------|-------------------------|
| t+1 | 191.17 | 212.51 | 95.18 | 145.88 |
| t+5 | 231.56 | 253.28 | 212.59 | 170.51 |
| t+10 | 195.57 | 243.73 | 221.09 | 176.79 |

On this split, **Random Forest does better on the raw target** at every horizon. **Ridge improves with the log target at t+5 and t+10** (lower RMSE on the original scale) but **not** at t+1.

### GroupKFold (hold out whole countries)

| Horizon | RF mean RMSE (raw) | RF mean RMSE (log) | Ridge mean RMSE (raw) | Ridge mean RMSE (log) |
|--------|--------------------|--------------------|-----------------------|------------------------|
| t+1 | 60.55 | 74.37 | 44.72 | 84.20 |
| t+5 | 81.95 | 90.06 | 82.68 | 112.92 |
| t+10 | 100.74 | 103.78 | 107.32 | 117.51 |

Here, **raw-target models win** for both learners on mean RMSE (see `outputs/direct_horizon_groupkfold_rmse_original_vs_log.csv` for standard deviations). So the log transform is **not** a universal win: it is still a useful **experiment** to document, and it can help **linear** models on **longer horizons** under the temporal split while hurting under strict **country** generalization in this run.

Artifacts (under `outputs/`): `direct_horizon_rmse_original_vs_log.csv`, `country_rmse_h1_test_log_target.csv`, `feature_importance_h1_log_target.png`.

## Caveats

- Temporal test RMSE and grouped-CV RMSE answer **different** questions; report **both** where possible.  
- **Train cutoff 2005** is **old** relative to 2010s violence—good for **honest long** *y+10* labels and for a **fixed** deploy story, weak for **recent** policy relevance.  
- Tune **RF / Ridge**, try **quantile** loss for skew, or **other target transforms / link functions** as a next step. Mixed / hierarchical models remain an upgrade path.
