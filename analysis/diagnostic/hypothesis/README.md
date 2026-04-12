# Hypothesis testing (`master_country_year`)

Tests use `data/final/master_country_year.csv`. Each **row** is one country-year; **`Num_Attacks`** is the attack count in that row. Categorical fields **`Region`** and **`Attack_Type_Mode`** come from the merged panel.

## Run

```bash
python analysis/diagnostic/hypothesis/hypothesis_tests.py
```

## Tests implemented

| # | Question | H0 | H1 | Method |
|---|----------|----|----|--------|
| **A** | Do regions differ in **attack frequency**? | Mean `Num_Attacks` is equal across regions | At least one region’s mean differs | **One-way ANOVA** on `Num_Attacks` by `Region`; **Kruskal–Wallis** (rank-based, less sensitive to skew/outliers) |
| **B** | Are **attack types** evenly represented in total violence? | Total attacks are **uniformly** spread across `Attack_Type_Mode` | Distribution is **not** uniform | **Chi-square goodness-of-fit** on summed `Num_Attacks` per type vs equal expected counts |
| **C** | Are **region** and **attack type** linked? | `Region` and `Attack_Type_Mode` are **independent** (in the contingency table of rows) | **Association** exists | **Chi-square test of independence** on `pd.crosstab(Region, Attack_Type_Mode)` (counts = number of country-year rows) |

## Example output (current CSV)

**A — Region vs mean attacks**

- ANOVA: **F ≈ 22.36**, **p ≈ 7.2×10⁻⁴⁵** → reject H0 at usual α (e.g. 0.05): **strong evidence** that average `Num_Attacks` per row **differs by region**.
- Kruskal–Wallis: **H ≈ 346.26**, **p ≈ 1.5×10⁻⁶⁷** → same conclusion using **ranks** (robust when distributions are skewed).

**B — Uniformity of attack types (by total attacks)**

- Chi-square GOF: **χ² very large**, **df = 8**, **p ≈ 0** → reject H0: **attack types are not evenly distributed**. **Bombing/Explosion** dominates totals; rare types (e.g. **Hijacking**) contribute little—consistent with **H1** (“some types dominate”).

**C — Region × attack type independence**

- Contingency χ²: **χ² ≈ 790.74**, **df = 88**, **p ≈ 1.7×10⁻¹¹³** → reject H0: **region and modal attack type are associated** in this table (mix of types differs across regions).

## How to interpret (insights)

1. **Regional differences (A)** — The panel does **not** look like one homogeneous “global mean” attacks per row: **regions differ** in typical intensity. ANOVA assumes **independent observations** and **roughly normal** group residuals; with heavy tails, **Kruskal–Wallis** is the safer headline for “is there a difference?”

2. **Attack-type mix (B)** — A **uniform** split across types is **not** plausible: **a few categories carry most attacks**. The test only checks the **statistical** departure from uniform; it does not by itself explain **why** (reporting rules, tactics, data coding).

3. **Association (C)** — **Independence** is rejected: **which attack types appear** varies with **region** in this dataset. Again, this is **association**, not proof of mechanism (confounding by country, time, or GTD coding is possible).

## Assumptions and limits

- **Independence:** Country-years for the **same country** are **not** truly independent; **p-values can be optimistic** (too small). A stricter analysis would use **mixed models** or **cluster-robust** inference by country.
- **ANOVA:** Sensitive to **non-normality** and **unequal variances**; Kruskal–Wallis mitigates some of that.
- **Chi-square:** Large **N** makes even **small** deviations significant; read **effect size** (counts, shares) alongside **p-values**.

Re-run the script after refreshing `master_country_year.csv` to update statistics.
