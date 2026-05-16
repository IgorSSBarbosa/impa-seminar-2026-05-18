# Saturday 2026-05-16 — task list

Two days before the IMPA seminar (Mon 2026-05-18).

## Core work — finish §6 and start the deck  (~5–6 h)

1. **Phase 2 regime sweep (§6 masterpiece plot)** — on the existing 2000-trial pool, compare three $m_0$ strategies on the same $(n, m)$ grid:
   - $m_0 = 1$ (constant) — should plateau
   - $m_0 = \alpha m$ with $\alpha \in \{1/3,\,1/2\}$ — partial decay
   - $m_0 \sim \tfrac12 \log_\rho(nm^3)$ — theoretically optimal, steepest slope

   Output: `fig_m0_regimes.png` — one log-log axis, $|\text{bias}|(n)$ per regime.

2. **Bump pool to 8000 trials if needed** (~7 min run). Current pool tops out at $R = 3$ at $n = 512$; for clean Phase 2 curves we want $R \ge 5$ at the largest $n$.

3. **Beamer scaffold** — `presentation18-05-2026/talk.tex`: title slide, section dividers for §1–§9, placeholder slides referencing the existing figures and GIFs.

4. **Draft §1–§5 slides** (math-heavy: $J$-order expansion, three-part decomposition, CLT statement). Leave §6–§9 for Sunday.

## Sanity / logistics  (~1 h)

5. Verify `lattices_merged.gif` (213 MB) plays smoothly in a viewer — if it stutters, re-encode at lower fps / resolution.
6. Test `perc_interactive.py` end-to-end on the actual presentation machine (resolution, slider responsiveness).

## Deferred to Sunday

- §6–§9 slide drafts
- Full rehearsal with timer
- Chalkboard derivation notes (lattices, log-log linearization, decomposition)
