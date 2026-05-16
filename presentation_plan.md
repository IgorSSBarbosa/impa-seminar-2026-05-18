# IMPA Statistics & AI seminar — presentation plan

**Date:** 2026-05-18 (Monday)
**Slot:** 50 min talk + 10 min Q&A
**Source paper:** [`research/article.tex`](../research/article.tex)
**Audience:** mixed statisticians + AI researchers

---

## Format

**Hybrid — Beamer + native GIF/demo windows + chalkboard.**

| Medium | Used for |
|---|---|
| Beamer (LaTeX) | Math-heavy slides: theorem statements, expansion formula, error decomposition, CLT, Wilson CI, minimax bound |
| Native window | Live `perc_interactive.py` demo + the (merged) lattice GIF + mean-field GIF |
| Chalkboard | 3/4/6-connected lattices, log-log linearization, three-part decomposition trick |

Each "demo break" in the deck is a clearly-marked slide ("→ live demo:
interactive percolation") so transitions look intentional.

## Anchor observables for log-log plots

Two are featured throughout:

- **Mean cluster size** $\chi(p) \sim |p-p_c|^{-\gamma}$ — classical
  $|p-p_c|^{-\gamma}$ family.
- **Fractal dimension at criticality** $\langle V(r)\rangle \sim r^{d_f}$
  — $V(r)$ is the number of occupied sites of the origin's cluster inside
  the box of side $r$, at $p = p_c$.
  **This is the form the article's framework actually covers**
  ($\mathbb E Y_i \sim i^\gamma$). The $|p-p_c|^{-\gamma}$ family is
  important but not directly analyzed.

---

## Section outline

### §1 — Phase transitions in percolation (≈ 7 min)
- Live demo: `perc_interactive.py` — sweep $p$, observe the giant cluster appear.
- Chalkboard: 3-, 4-, 6-connected lattices ≡ honeycomb / square / triangular.
- Merged side-by-side GIF of the three lattices near criticality.
- Define $p_c$ as the threshold.

### §2 — Critical exponents, scaling, universality (≈ 10 min)
- Define $\chi(p)$ and the fractal dimension $d_f$ (anchor observables).
- Log-log plots → straight lines.
- **Universality slide:** overlay fractal-dimension log-log plot across
  the three lattices → parallel lines (same $d_f$, different intercepts).
- *(Optional)* `percolation_meanfield.gif` → mean-field as the trivial
  universality class.
- *(Optional, bridge to statistics)* classical CLT $S_n \sim \sqrt{n}$ as
  a toy universality statement: "exponent $1/2$ independent of step distribution".

### §3 — From physics heuristic to statistical question (≈ 8 min)
- Re-use the §2 fractal-dimension log-log plot as running example.
- The problem: physicists read $d_f$ off the slope of $\log\mathbb E Y_r$,
  but only $\overline Y_r$ is observable.
- **Bias–variance trade-off:** large $r$ → small finite-size bias but
  expensive → few samples → high variance. Small $r$ → cheap but biased.
- Chalkboard:
  $\log \mathbb E Y_{\rho^k} = \log a_0 + k\beta\log\rho + a_1\rho^{-k} + \phi_1(\rho^k)\rho^{-2k}$
  showing how log-log linearizes the power law and exposes finite-size correction.
- Slide: the $J=1$ expansion assumption (defer general $J$ to a remark).

### §4 — The log-log estimator and the three-part split (≈ 6 min)
- $\hat\beta = \sum_k w_{k,m}\, \log\overline Y_{\rho^k}$ (OLS slope), with
  $w_{k,m} = \tfrac{12(k-m_0-(m+1)/2)}{m(m^2-1)}$.
- The trick:
  $\log\overline Y_{\rho^k} = \log\mathbb E Y_{\rho^k}
  + \log\!\big(\overline Y_{\rho^k}/\mathbb E Y_{\rho^k}\big)$
  → linear part (signal) + finite-size bias + random noise.
- Linear regression as a linear combination with weights $w_{k,m}$.

### §5 — Main theorem: CLT for $\hat\gamma$ (≈ 10 min)
- **Statement:** $\sqrt{nm^3}\,(\hat\gamma - \gamma) \Rightarrow
  \mathcal N(0,\, 12\sigma_\infty^2/\log^2\rho)$.
- **Proof sketch (one slide each):**
  1. Decomposition $\hat\beta - \beta = R + S_{n,m} + Q_{n,m}$.
  2. $S_{n,m}$ via Lindeberg–Feller; Lyapunov from $(2+\delta)$-moment.
  3. $Q_{n,m}$: Taylor remainder of $\log$, good/bad event split, moment
     bounds → mean and std go to 0 faster than CLT rate.
  4. $R$ (deterministic finite-size bias) handled separately.

### §6 — $m_0$ and Wilson confidence intervals (≈ 7 min)
- Why discard the first $m_0$ scales: kills the leading finite-size term.
- Wilson CI: four explicit terms (finite-size bias, Jensen good-event bias,
  bad-event bias, Gaussian std error) — each decays at a different rate.
- Empirical comparison of $m_0$ regimes:
  - $m_0 = $ const (e.g. 1),
  - $m_0 = \alpha m$ for $\alpha \in (0,1)$,
  - $m_0 \sim \tfrac12 \log_\rho(nm^3)$ (theoretically optimal),
  - $m - m_0 = $ const (e.g. 2 → two-point estimator).

### §7 — Lower bound and optimality (≈ 4 min)
- Minimax: $\mathbb E[\delta_\gamma^2] \gtrsim B^{-2J/(d+2J)}$ via KL between
  two product measures differing in $\gamma$ → Fano-type bound.
- Upper bound matches up to log factors → log-log estimator is near-optimal.

### §8 — Closed-form sanity checks (≈ 3 min, optional)
- Bethe lattice: explicit $\mathbb E Y_i$ via Stirling.
- Simple random walk: closes the loop back to the §2 CLT analogy.

### §9 — Wrap-up (≈ 2 min)
- One slide of takeaways: physicists' log-log plot has a rigorous
  statistical theory; the estimator is asymptotically optimal; $m_0$ is
  the key practical lever.

---

## Asset checklist

**Ready:**
- [`uniform_coupling/perc_interactive.py`](../uniform_coupling/perc_interactive.py) (§1 live demo)
- [`mean_field/percolation_animations/percolation_meanfield.gif`](../mean_field/percolation_animations/percolation_meanfield.gif) (§2 optional)

**To produce:**
- Merged side-by-side GIF of `percolation_{square,hexagonal,triangular}.gif`
  for §1.
- Fresh simulations + log-log plots: fractal dimension and mean cluster
  size, on all three lattices, plus a universality overlay.
- $m_0$-regime comparison plot for §6.
- Beamer skeleton aligned with this section list.

---

## Open items / decisions to revisit

- Whether to keep §8 (closed-form sanity checks) or drop it if §5–§7 run long.
- Whether the mean-field GIF in §2 is in or out (currently optional).
- Whether the classical-CLT bridge slide in §2 lands well with this
  audience — keep if Q&A signals confusion about universality.
