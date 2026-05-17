# Schedule — IMPA seminar (Mon 2026-05-18, 13:30)

## Saturday 2026-05-16 — DONE
- ✅ Pool sim, checkpointed: 65 536 trials at scales 2..2048 (2h44m, square,
  p_c=0.592746). Output: `simulation_data/fractal_dim_pool.npz`.
- ✅ L3 plots regenerated from 65k pool: `fig_regime_sweep.png`,
  `fig_error_vs_time.png` (the user notes the non-m_0=1 regimes still
  don't match expectations — revisit Sunday).
- ✅ Talk skeleton + content drafted §1–§5 (35 frames, biblatex wired,
  IMPA Boadilla theme, slim footer override).
- ✅ §1 Hammersley plain-frame + speaker note; §2 zoo + universality
  + 2-column log-log linearisation; §3 bias-variance + J=1 expansion;
  §4 OLS + three-part split + linear-functional; §5 CLT with (A1)-(A3)
  + 4 proof sketches.

## Sunday 2026-05-17

### Open design decision (before any more §5 work)
- **§5 simplification choice.** §5 is heavy: 5 frames of CLT machinery
  (assumptions, strategy, Lindeberg-Feller, good/bad event, R bias).
  Two options:
  1. **Keep the CLT version** as drafted (technical; ~10 min).
  2. **Fall back to an L² convergence statement** ($\hat\beta \to \beta$
     in L²) which compresses to 2 frames (Chebyshev on Var + bias).
     Cite the CLT as ongoing work (Barbosa–Teixeira–Imbuzeiro 2026⁺).
  Decide first thing Sunday — affects whether §5 collapses or stays.

### Content (~4-5 h)
1. **§6 fill stubs** (5 frames):
   - Why drop the first m_0 scales (prose + maybe a tiny weight-plot).
   - Wilson-style confidence interval (4 explicit error terms).
   - Embed `fig_regime_sweep.png` with verbal punchline.
   - Embed `fig_error_vs_time.png` with verbal punchline.
   - Masterpiece plot (L4 — to be produced, see §6 sims below).
2. **§7 fill stubs** (2 frames): minimax lower bound (Fano-style KL between
   two product measures); upper bound matches up to logs ⇒ near-minimax.
3. **§8 fill stubs** (2 frames): Bethe lattice (closed-form via Stirling);
   simple random walk (return-to-origin closes loop to §2 classical CLT).
4. **§9 take-aways** (1 frame): the four-point summary already sketched.
5. **Appendix** (2 frames): L0-L4 pipeline + BFS speed-up.

### Sims (~1-2 h)
6. **Diagnose the m_0=1 vs other-regimes mismatch.** User flagged the
   non-m_0=1 curves in `fig_regime_sweep.png` look off. Inspect
   `regime_sweep.py` + reload pool, check the alpha=1/4, 1/3, 1/2
   curves' bias decay rates against theory ($\rho^{-\alpha m}/m^2$).
7. **L4 masterpiece plot** — design and run: vary budget B ∈ {1,2,4,8,16}
   in units of pool-equivalent compute; show error vs B per m_0 regime.
   Goal: visually show that the log-ρ regime has the steepest decay.

### Chalkboard notes (~30 min)
8. Write up the 3 chalk derivations to take to the room:
   - Uniform coupling (§1 sim slide).
   - Log-log linearisation (§3 board frame).
   - Three-part split derivation (§4 three-part split frame).

### Mistakes-to-fix list (see audit in chat) (~30 min)
9. Sweep the fixes documented in the Saturday audit (notation
   inconsistencies, sign of the good-event inequality in §4, stale
   TODOs in §2 "Two anchor observables", optional CLT bridge frame).

### Sunday EOD gate
- Full deck compiles clean on Overleaf (no missing refs, no overflow
  warnings on critical frames).
- Dry-run with timer: aim ≤ 50 min including 5 min Q&A buffer.

## Monday 2026-05-18
- Morning: demo dry-run (`perc_interactive.py` + `lattices_merged.gif`
  on presentation machine; slider feels OK).
- Final rehearsal pass with timer.
- Travel to IMPA, arrive 30 min early to test projector/aspect ratio.
- 13:30 talk.
