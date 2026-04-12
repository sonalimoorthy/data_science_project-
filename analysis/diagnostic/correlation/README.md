# Correlation diagnostic: numeric panel variables

Pearson and Spearman correlations among:

- `Num_Attacks`, `Fatalities`, `Injuries`, `GDP_per_capita`, `Population`, `Conflict_Level`

Source: `data/final/master_country_year.csv` (pairwise complete observations per pair).

## Run

```bash
python analysis/diagnostic/correlation/correlation_analysis.py
```

## Outputs

| File | Description |
|------|-------------|
| `correlation_pearson.csv` | Pearson matrix |
| `correlation_spearman.csv` | Spearman matrix |
| `pairplot_correlation.png` | Pairwise scatter (corner) + KDE diagonals (random sample up to 2,000 rows) |
| `heatmap_pearson.png` | Annotated Pearson heatmap |
| `heatmap_spearman.png` | Annotated Spearman heatmap |

## Interpretation: strongest relationships

### Pearson (linear association)

- **Violence outcomes move together:** `Num_Attacks`, `Fatalities`, and `Injuries` are **strongly positively** correlated with each other (**r** roughly **0.79–0.86**). Heavier attack counts, deaths, and injuries tend to occur in the same country-years.
- **`Conflict_Level`** is **moderately positive** with attack counts and harm (**r** about **0.33–0.36**), consistent with more conflict intensity aligning with more GTD-recorded violence in the merged panel.
- **`GDP_per_capita`** shows **weak negative** linear links with violence variables (**r** around **-0.04 to -0.09** except a **moderate negative** with `Conflict_Level`, **r ≈ -0.25**).
- **`Population`** is only **weakly** related in the linear sense (**r** about **0.06–0.14** with attacks/harm).

### Spearman (monotonic / rank association)

- **Attack/harm bundle** remains **strong** (rank correlations often **0.72–0.77** among `Num_Attacks`, `Fatalities`, `Injuries`).
- **`Fatalities` vs `GDP_per_capita`** is **more negative** under Spearman (**≈ -0.30**) than Pearson, suggesting a **rank** pattern (e.g. lower-GDP country-years tending to rank higher on fatalities) that is **not fully linear**.
- **`Population` vs `Num_Attacks`** is **moderately positive** in ranks (**≈ 0.42**) while Pearson was smaller, so **larger countries** may rank higher on attack totals partly through **scale / reporting**, not only a tight linear fit.
- **`Conflict_Level`** remains **moderately** associated with fatalities in particular (**Spearman ≈ 0.53**).

### Caveats

Correlation is **not causation**. Confounding (e.g. country fixed effects, reporting bias, conflict exposure) is not removed here. Re-run the script after changing the CSV to refresh matrices and figures.

## In plain English (what we can take away)

Think of correlation as **“when one number tends to be high or low, does the other tend to follow?”** It does **not** prove that one thing *causes* the other.

**What the data suggests in everyday terms:**

- **When there are more attacks in a country-year, there also tend to be more deaths and more injuries.** That is the clearest pattern: the three harm measures rise and fall together. That matches common sense (bigger events show up in more than one column), but it is still useful to see it quantified.

- **Higher conflict intensity scores tend to show up alongside more terrorism-style violence in the same rows**, but the link is only **moderate**. So conflict level and GTD violence are related, but they are not the same thing and plenty of exceptions exist.

- **Richer country-years (higher GDP per person) do not line up strongly with more attacks or injuries in a straight-line way.** There is a **modest** tendency for **more conflict intensity** to sit with **lower** GDP per capita, but overall the income story is **subtle**—do not read it as “poor places always have more terrorism.”

- **Bigger populations do not strongly predict more attacks in a simple linear way**, but when you look at **rankings** instead, **larger countries more often sit higher on attack counts**. That can reflect many things (more people, more targets, more news coverage), not just “more violence per person.”

**Bottom line:** The numbers mainly tell us that **attack counts, deaths, and injuries are tightly bundled** in this dataset, and that **conflict measures and country context matter at a weaker, muddier level**. Anything you conclude about **why** that happens needs **other methods** (maps, time trends, country comparisons, regression with controls), not correlation alone.
