# Data Sources and Provenance

This file tracks the source, download date, and version details for all raw datasets in `data/raw/`.

## Quick Summary

| Dataset | Raw File | Download Date | Version |
|---|---|---|---|
| GTD | `archive/globalterrorismdb_0718dist.csv` | 2026-03-17 | `globalterrorismdb_0718dist` |
| UCDP | `organizedviolencecy-251-csv/organizedviolencecy_v25_1.csv` | 2026-03-17 | `v25_1` |
| OWID Population | `population-with-un-projections/population-with-un-projections.csv` | 2026-03-17 | UN WPP (2024), via OWID |
| OWID Poverty | `share-of-population-in-extreme-poverty/share-of-population-in-extreme-poverty.csv` | 2026-03-17 | World Bank PIP (2025), via OWID |

## Source Details

### 1) Global Terrorism Database (GTD)
- **Source URL:** https://www.kaggle.com/datasets/START-UMD/gtd
- **Downloaded on:** 2026-03-17
- **Raw file:** `archive/globalterrorismdb_0718dist.csv`
- **Version note:** Kaggle distribution file `globalterrorismdb_0718dist` (2018 release naming).

### 2) UCDP Organized Violence (Country-Year)
- **Source URL:** https://ucdp.uu.se/
- **Downloaded on:** 2026-03-17
- **Raw file:** `organizedviolencecy-251-csv/organizedviolencecy_v25_1.csv`
- **Version note:** `v25_1` (from file name).

### 3) Our World in Data (OWID) - Grapher Exports
- **Source URL:** https://ourworldindata.org/grapher
- **Downloaded on:** 2026-03-17

**Population dataset**
- **Raw file:** `population-with-un-projections/population-with-un-projections.csv`
- **Underlying source/version:** UN World Population Prospects (2024), as recorded in OWID metadata.

**Extreme poverty dataset**
- **Raw file:** `share-of-population-in-extreme-poverty/share-of-population-in-extreme-poverty.csv`
- **Underlying source/version:** World Bank Poverty and Inequality Platform (2025), as recorded in OWID metadata.

## Data Handling Rules

- Treat files in `data/raw/` as immutable (never edit in place).
- Write cleaned source-level outputs to `data/interim/`.
- Write merged analytics-ready outputs to `data/final/`.
