# Prescriptive Analysis — Terrorism Risk Reduction

`prescriptive_analysis.py` is the final analytics tier. It translates diagnostic
and predictive findings into **actionable decisions**: rule-triggered interventions
and optimally allocated security budgets.

## Run

```bash
python analysis/prescriptive/prescriptive_analysis.py
```

---

## 4.1 Rule-Based Intervention Engine

### Design

Eight **IF-THEN rules** are derived directly from the diagnostic and predictive layers:

| Rule ID | Trigger | Priority | Category |
|---------|---------|----------|----------|
| R01 | Bombing-dominant, high-risk region, ≥100 attacks | **CRITICAL** | Counter-IED Capability |
| R02 | Attacks >20% above 3-year rolling average | **HIGH** | Early Warning / Surge Response |
| R03 | Conflict Level ≥ 2 AND low-GDP country | **HIGH** | Stabilisation / Development Aid |
| R04 | Private Citizens dominant target type, ≥10 attacks | **HIGH** | Protective Security |
| R05 | Armed Assault dominant, ≥20 attacks | **MEDIUM** | Armed Forces / Border Security |
| R06 | Kidnapping dominant attack type | **MEDIUM** | Hostage / Ransom Policy |
| R07 | Medium risk region, ≥5 attacks | **MEDIUM** | Intelligence / Monitoring |
| R08 | Diplomatic / Government target spike, ≥15 attacks | **HIGH** | Diplomatic Security |

Rules are applied to the **most recent observed year per country** (≥2005 panel) —
the conditions most relevant to forward-looking policy.

### Outputs

| File | Description |
|------|-------------|
| `rule_engine_fired.csv` | All triggered rules (country, region, year, recommendation) |
| `rule_engine_summary.png` | Bar charts: triggers by priority and by region |

### Key Findings (current data)

- **87 total rule triggers** across **53 countries** and **8 regions**
- **9 CRITICAL**, **46 HIGH**, **32 MEDIUM** priority triggers
- South Asia and Sub-Saharan Africa generate the most HIGH/CRITICAL triggers
- R03 (fragile-state nexus) is the most widely fired rule — poverty × conflict co-occurrence is widespread

---

## 4.2 Optimisation Model — Resource Allocation

### Problem Formulation

**Objective:** Minimise total expected terrorism harm across 12 world regions.

**Decision variables:** `x_i` = percentage of total security budget allocated to region `i`.

**Constraints:**
- `Σ x_i = 100` (full budget deployed)
- `x_i ≥ 2` (minimum 2% coverage per region — no region left unmonitored)
- `x_i ≤ 40` (concentration cap — no single region receives >40%)

**Effectiveness model (diminishing returns):**
```
effectiveness(x_i) = min(0.70, 0.55 × √(x_i / equal_share))
```
Each extra unit of budget has smaller marginal impact — a well-documented property of
security investment (RAND Corporation, 2010; Mueller & Stewart, 2011).

**Harm measure:** Region's mean annual `risk_score = attacks + 0.5 × fatalities` (training period ≤2010).

**Solver:** SciPy SLSQP (Sequential Least Squares Programming) — handles nonlinear
objectives with linear and nonlinear equality/inequality constraints.

### Results

- **Overall expected harm reduction: ~65%** given optimal allocation
- South Asia, MENA, Southeast Asia, South America, and Central America & Caribbean
  all receive maximum efficient allocations (≥13.5%)
- Eastern Europe, Central Asia, and Oceania receive minimum floor allocations (2%)
  because their low baseline risk scores mean marginal returns are insufficient
- **Key insight:** even after optimal allocation, South Asia retains significant
  residual risk (≥70 risk-score units) — indicating that resource allocation alone
  cannot solve the problem; structural interventions (R03) are essential complements

### Outputs

| File | Description |
|------|-------------|
| `resource_allocation.csv` | Optimal budget %, harm reduction % per region |
| `resource_allocation_optimisation.png` | Side-by-side: allocation and harm reduction |
| `harm_before_after.png` | Grouped bar: baseline vs residual risk per region |

---

## Analytical Chain

```
Descriptive  ──▶  Diagnostic  ──▶  Predictive  ──▶  Prescriptive
(What?)           (Why?)           (What next?)       (What to do?)
  summary         correlation       RF classifier      rule engine
  visuals         hypothesis        panel forecast      LP optimiser
  time series     regression        risk scores         recommendations
```

The prescriptive layer closes the loop by converting statistical insights into
prioritised, operationally deployable decisions.

---

## Caveats

1. **Rules fire on the last observed year per country** — temporal lag exists; real-time
   data pipelines would improve responsiveness.
2. **Effectiveness model is illustrative** — real-world security effectiveness depends on
   institutional capacity, political will, and local context not captured here.
3. **Optimisation minimises a proxy harm measure** — direct casualty reduction requires
   richer operational data (threat intelligence, unit costs, response times).
4. **Prescriptive outputs are decision-support tools**, not autonomous decision systems.
   Human expert review of all triggered recommendations is mandatory.
