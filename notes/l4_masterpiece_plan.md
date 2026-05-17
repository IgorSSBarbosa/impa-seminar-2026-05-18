# L4 "masterpiece" plot — design

**Goal.** Single plot for §6: log–log RMSE of $\hat\gamma$ vs total
simulation budget $B$, one curve per $m_0$-regime, showing which regime
gives the steepest decay (the "right" choice of $m_0(m,n)$).

**Constraint.** Reuse `simulation_data/fractal_dim_pool.npz` (65 536
trials × 11 scales = $\rho^k$, $k=1\ldots 11$, $\rho=2$). No new
simulation. Constraint inherited: every kept window must satisfy
$m_0 + m \le 11$.

## Regimes (4)

| key | $m_0(m,n)$                                | rationale                                 |
| --- | ----------------------------------------- | ----------------------------------------- |
| R1  | $1$                                       | baseline — never drops anything           |
| R2  | $\lfloor m/2 \rfloor$                     | linear schedule (current "alpha=1/2")     |
| R3  | $\lceil \tfrac12 \log_2(n m^3) \rceil$    | optimal: matches negligibility threshold $\rho^{m_0} \gg \sqrt{nm^3}/m^2$ |
| R4  | $m-2$                                     | extreme — keep only the largest few scales|

## Budget axis

$B(n,m,m_0) = n \cdot \sum_{k=m_0+1}^{m_0+m} \tau_k$ where $\tau_k$ is
the per-trial BFS cost at scale $\rho^k$. Use the
`mean_L2_time_s` already recorded by `regime_sweep.py` (or recompute
from `sec_per_trial=0.150` and the BFS scaling $\tau_k \propto
\rho^{k\,d_f}$).

## Method

1. **Sweep cells.** For each regime, for each $m \in \{3,\ldots,8\}$,
   for each $n \in \{16, 64, 256, 1024, 4096\}$:
   - Compute $m_0$ from the regime rule.
   - If $m_0 + m > 11$: skip (out of pool range).
   - Partition pool into $R = \lfloor 65536/n \rfloor$ disjoint blocks.
   - For each block: OLS slope of $\log \overline V_{\rho^k}$ on
     $k = m_0+1,\ldots,m_0+m$. Get $\hat\gamma$.
   - Across blocks: $\text{RMSE}^2 = \text{bias}^2 + \text{var}$.

2. **Plot.** Log–log, $x = B$, $y = \text{RMSE}$, one colour per
   regime, marker per regime. Error bars: 95% CI via
   $\pm 1.96 \cdot \text{std}/\sqrt{R}$.

3. **Punchline.** Compare per-regime OLS slope of $\log\text{RMSE}$
   vs $\log B$. R3 should be steepest (matches theory); R1 shallowest
   (bias floor); R2 in between; R4 high-variance / poor for small $B$.

## Pool coverage (the table the user asked for)

`R = 65536 // n` (replicas), `scales used = [ρ^{m_0+1},…,ρ^{m_0+m}]`.
Rows marked **OoR** (out of range) violate $m_0+m \le 11$ and are skipped.

### R1 — $m_0 = 1$

| n     | m | m_0 | scales used                          | R    |
| ----: | -:| --: | ------------------------------------ | ---: |
| 16    | 3 | 1   | 4, 8, 16                             | 4096 |
| 16    | 4 | 1   | 4, 8, 16, 32                         | 4096 |
| 16    | 5 | 1   | 4, 8, 16, 32, 64                     | 4096 |
| 16    | 6 | 1   | 4, 8, 16, 32, 64, 128                | 4096 |
| 16    | 7 | 1   | 4, 8, 16, 32, 64, 128, 256           | 4096 |
| 16    | 8 | 1   | 4, 8, 16, 32, 64, 128, 256, 512      | 4096 |
| 64    | 3 | 1   | 4, 8, 16                             | 1024 |
| 64    | … | …   | (same scales, R=1024)                | 1024 |
| 256   | … | 1   | (m=3..8, R=256)                      |  256 |
| 1024  | … | 1   | (m=3..8, R=64)                       |   64 |
| 4096  | … | 1   | (m=3..8, R=16)                       |   16 |

Full R1 coverage: 5 budget levels × 6 m-values = **30 points**.

### R2 — $m_0 = \lfloor m/2 \rfloor$

| n   | m | m_0 | scales used                  | R    | status |
| --: | -:| --: | ---------------------------- | ---: | -----  |
| any | 3 | 1   | 4, 8, 16                     | …    | ✓      |
| any | 4 | 2   | 8, 16, 32, 64                | …    | ✓      |
| any | 5 | 2   | 8, 16, 32, 64, 128           | …    | ✓      |
| any | 6 | 3   | 16, 32, 64, 128, 256, 512    | …    | ✓      |
| any | 7 | 3   | 16, …, 1024                  | …    | ✓      |
| any | 8 | 4   | 32, …, 4096                  | —    | OoR (m_0+m=12) |

Full R2 coverage: 5 budgets × 5 m-values = **25 points**.

### R3 — $m_0 = \lceil \tfrac12 \log_2(n m^3) \rceil$

Constraint $\tfrac12 \log_2(n m^3) + m \le 11 \;\Leftrightarrow\;
n m^3 \le 4^{11-m}$.

| n    | m | m_0 | nm³     | bound 4^(11−m) | scales used                                | status |
| ---: | -:| --: | ------: | -------------: | ------------------------------------------ | ------ |
| 16   | 3 | 5   |     432 |          65536 | 64, 128, 256                               | ✓      |
| 16   | 4 | 5   |    1024 |          16384 | 64, 128, 256, 512                          | ✓      |
| 16   | 5 | 6   |    2000 |           4096 | 128, 256, 512, 1024, 2048                  | ✓      |
| 64   | 3 | 6   |    1728 |          65536 | 128, 256, 512                              | ✓      |
| 64   | 4 | 6   |    4096 |          16384 | 128, 256, 512, 1024                        | ✓      |
| 256  | 3 | 7   |    6912 |          65536 | 256, 512, 1024                             | ✓      |
| 256  | 4 | 7   |   16384 |          16384 | 256, 512, 1024, 2048                       | ✓ (=) |
| 1024 | 3 | 8   |   27648 |          65536 | 512, 1024, 2048                            | ✓      |
| 4096 | 3 | 9   |  110592 |          65536 | 1024, 2048, 4096                           | OoR    |

R3 coverage: **~8 points** (much sparser — pool's 11 scales bind here).

### R4 — $m_0 = m-2$

| n   | m | m_0 | scales used                        | R     | status |
| --: | -:| --: | ---------------------------------- | ----: | ------ |
| any | 3 | 1   | 4, 8, 16                           | …     | ✓      |
| any | 4 | 2   | 8, 16, 32, 64                      | …     | ✓      |
| any | 5 | 3   | 16, 32, 64, 128, 256               | …     | ✓      |
| any | 6 | 4   | 32, 64, 128, 256, 512, 1024        | …     | ✓      |
| any | 7 | 5   | 64, …, 4096                        | —     | OoR    |

R4 coverage: 5 budgets × 4 m-values = **20 points**.

## Limitations / decisions

- **R3 is pool-limited.** With only 11 scales, R3 has ~8 plottable
  points vs 25–30 for R1, R2, R4. Two ways to handle:
  - (a) plot R3 anyway with sparser markers, flag in the caption;
  - (b) extend the pool to scales $\rho^{12}, \rho^{13}$ — costs ~3h
    extra wall-clock, gives R3 full coverage. **Recommend (a)** for
    Sunday since the relative slope is what matters.
- **R1 and R2/R4 share many scales at small m.** For $m=3$, R1, R2,
  R4 all use $\{4,8,16\}$ with $m_0=1$. They diverge at larger $m$
  (where R2/R4 drop more scales). Expected: curves overlap at low
  budget, separate at high budget — this is fine, makes the
  separation visually meaningful.
- **R=16 (n=4096) std bar is wide.** Either drop the highest budget
  point per regime, or accept wider CI bars. Easier path: cap at
  n=1024 (R=64).

## Implementation sketch

New script `coding/masterpiece_plot.py`:
- Reuses `lib.stats.ols_slope`, `lib.plotting`.
- Loops over `(regime, m, n)`, computes m_0 via regime function,
  partitions pool, computes per-block γ̂, aggregates RMSE.
- Writes `simulation_data/masterpiece.csv` + `images/fig_masterpiece.png`.
- Add to `run_overnight.sh` as step [4/4].

ETA: ~30 min to write + 1 min runtime (no new sim).
