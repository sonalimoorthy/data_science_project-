# Region and country risk classification

`region_risk_classification.py` trains **region-year** and **country-year** panels, uses a **weighted risk score** and **quantile-based** high-risk label, compares models under a **temporal** split and under **GroupKFold** (held-out regions or countries), and adds **baselines**, **ROC** and **precision–recall** curves, **Random Forest feature importance** (plot + CSV + short narrative file), **probability calibration**, and **F1-optimal thresholding** for RF using **`precision_recall_curve`** on train probabilities.

If a rubric still mentions **median on `Total_Attacks` only** or **rolling without `shift`**, that describes an **older** design; the current script documents the mapping at the top of `region_risk_classification.py` (“Rubric alignment”).

## Target (Issue 1)

- **Risk score:** `risk_score = Total_Attacks + 0.5 * Total_Fatalities` (region aggregates) or `Num_Attacks + 0.5 * Fatalities` (country panel).
- **High risk:** rows with `risk_score` in the **top 25%** of the **training** split, i.e. `>=` the **75th percentile of `risk_score` on train** (same cutoff applied to train and test rows for the temporal evaluation so labels stay comparable).

## Leakage fix (Issue 3)

- **`attacks_roll3`** is a rolling mean of **past** totals only: `shift(1)` then `rolling(3).mean()` within each region or country (no current-year attack total in that feature).

## Panels (Issue 2)

- **Region:** `groupby(Region, Year)` sums (and means) from raw country-year rows, then a balanced region×year grid.
- **Country:** balanced country×year grid (same spirit as `panelModel`).

All generated **CSV** and **PNG** files (and the RF interpretation **TXT**) are written under **`outputs/`** when you run the script (the folder is created automatically). Outputs compare **accuracy** and **F1 (high-risk)** for both levels; see `outputs/classification_accuracy_f1_summary.csv` and `outputs/classification_temporal_metrics_region_vs_country.csv`.

## Evaluation (Issues 4–5)

1. **Temporal:** train `Year <= 2010`, test `Year > 2010` (same as before).
2. **GroupKFold:** `groups = Region` or `groups = Country` so **entire regions/countries** are held out per fold. The **75th percentile threshold is recomputed on each train fold** from that fold’s `risk_score` only, then applied to train and test rows in that fold.
3. **Baseline:** `DummyClassifier(strategy="most_frequent")` — included in both temporal and GroupKFold tables.

## Models

- Logistic regression, decision tree, random forest (`class_weight="balanced"`), plus baseline.
- **Random Forest extras (Issue 7 / calibration):**
  - `CalibratedClassifierCV(..., method="isotonic")` on the RF pipeline (reported as `*_random_forest_calibrated`).
  - **F1-tuned threshold** on **train** RF probabilities via `precision_recall_curve` (`*_random_forest_f1_threshold`).

## Feature importance & ROC (Issues 6–7)

- **`outputs/feature_importance_rf_region_top10.png`** (+ `.csv`) and **`outputs/feature_importance_rf_country_top10.png`** (+ `.csv`): RF trained on the **full temporal train** split.
- **`outputs/roc_curves_rf_region_vs_country.png`:** RF on **temporal test**, region vs country.
- **`outputs/precision_recall_curves_rf_region_vs_country.png`:** Same split (PR view for imbalanced high-risk rate).
- **`outputs/rf_feature_importance_interpretation.txt`:** Top-10 lists copied from CSVs plus one-line interpretation.

## KMeans elbow (panel script)

`panel_forecast_and_risk.py` writes elbow artifacts under **`panelModel/outputs/`** (`kmeans_elbow.png` / `.csv`) before fitting **k = 4**. Use the elbow as a guide; cluster sizes can still be imbalanced at any fixed *k*.

## Run

```bash
python analysis/predictive/classification/region_risk_classification.py
```

## Outputs (under `outputs/`)

| File | Description |
|------|-------------|
| `classification_temporal_metrics_region_vs_country.csv` | All temporal metrics (both levels, all models + RF tuned + calibrated) |
| `classification_accuracy_f1_summary.csv` | Short comparison: level, model, accuracy, F1 |
| `groupkfold_metrics_region.csv` | Mean/std accuracy & F1 across folds (unseen regions) |
| `groupkfold_metrics_country.csv` | Same for unseen countries |
| `region_year_test_predictions.csv` | Test window: labels, preds, probabilities |
| `country_year_test_predictions.csv` | Same at country level |
| `regions_ranked_future_high_risk.csv` | Regions by mean RF `P(high risk)` on test years |
| `regions_top_future_risk_rf.png` | Bar chart for that ranking |
| `roc_curves_rf_region_vs_country.png` | ROC comparison |
| `precision_recall_curves_rf_region_vs_country.png` | Precision–recall (test) |
| `rf_feature_importance_interpretation.txt` | Narrative + top-10 importances |
| `feature_importance_rf_*_top10.png` | RF explanations |

## Caveats

- **Region aggregation** hides within-region hotspots; the **country** panel is the fairer stress test and usually **harder** under GroupKFold.
- **High AUC** on the temporal split can still reflect **persistent geography** (Region one-hot) and **autocorrelation** in violence; **GroupKFold** metrics are the stricter generalization check.
- Calibrated probabilities depend on **enough train data** and **isotonic** assumptions; treat as exploratory.
