# Time series: `Num_Attacks` by year

Analysis of **global yearly totals** (sum of `Num_Attacks` across all countries) from `data/final/master_country_year.csv`.

## Run

From the repository root:

```bash
python analysis/descriptive/timeSeries/time_series_num_attacks.py
```

Outputs:

- `trend_num_attacks_by_year.png` — line chart of yearly totals; **markers** on the **five highest** years.
- Printed tables: top 10 peak years, OLS slope on year vs total.

## Tasks (brief interpretation)

### 1. Trend over years

The series is **not smooth**: it rises from the 1970s into the early 1990s, **drops sharply** around the late 1990s (e.g. 1998 is a notable low), then climbs again with a strong upswing in the **2010s**. So the story is **regime shifts and volatility**, not a steady straight line—though a **linear fit over the full 1970–2017 window** still tilts **upward** because the start is small and the end is much larger.

### 2. Peaks (highest-attack years)

By **total** `Num_Attacks`, the **top years** in the current file are concentrated in the **2010s**, led by **2014** (~16.9k), then **2015**, **2016**, **2013**, **2017**, etc. Earlier local highs include years such as **1992** relative to surrounding years. See the script output for the exact ranked table.

### 3. Increasing or decreasing?

- **Whole-period linear trend:** OLS slope on `Year` vs yearly total is **positive** (on the order of **~180** extra attacks per calendar year across the full span), so a single straight line describes the long window as **net increasing**.
- **Recent segment:** After the **2014** maximum, totals **edge down** through **2017** in this panel, so the **very end** of the series is a **short-term decrease** from the peak—not a decades-long decline.

Re-run the script after refreshing `master_country_year.csv` to refresh numbers and the figure.
