# `demos/` — live demos for the talk

Three interactive / animated scripts. None of them depend on `lib/` or
on any pre-computed pool — they regenerate everything they need on the
fly, so they work on a clean clone.

## `perc_interactive.py`

Matplotlib GUI for 2D site percolation on the square lattice. A slider
sweeps $p \in [0,1]$; clusters are recoloured on each move. Two
backends:

```bash
python3 coding/demos/perc_interactive.py            # default: Numba union-find
python3 coding/demos/perc_interactive.py --scipy    # SciPy ndimage (slower, no Numba)
```

Grid sizes $\{64, 128, 256, 512\}$ are pre-loaded; click the size buttons
to switch. The view around $p_c = 0.592\,746$ is also prefetched so the
slider stays responsive near criticality.

## `perc_animate_merged.py`

Renders the three-lattice (honeycomb / square / triangular) universality
animation in a single side-by-side figure, sharing the same uniform
field $U \sim \text{Unif}[0,1]^{N\times N}$ across lattices so the only
visible difference between panels is the connectivity rule.

```bash
# Default: GIF at fps=5, 3 grid-size segments, output to images/lattices_merged.gif
python3 coding/demos/perc_animate_merged.py

# MP4 (needs ffmpeg) and a custom grid set:
python3 coding/demos/perc_animate_merged.py --mp4 --grid-sizes 64 256 1024

# Smaller file:
python3 coding/demos/perc_animate_merged.py --frames-per-n 60 --fps 12
```

Pass `-h` for the full flag list.

## `../mean_field/main.py`

Mean-field demonstration: BFS clusters in dimensions $d = 2, \dots, 8$.
Each segment zooms out from the origin and (for $d \geq 3$) rotates the
camera around the full cluster, illustrating the transition to
mean-field behaviour above the upper critical dimension $d = 6$.

```bash
python3 coding/mean_field/main.py                # defaults: seed=42, dims=2..8
python3 coding/mean_field/main.py --seed 7
python3 coding/mean_field/main.py --dims 2 3 4   # subset
```

Output: `coding/mean_field/percolation_animations/percolation_meanfield.gif`.
(Pre-rendered copy is at `images/percolation_meanfield.gif` in the repo
root so the talk can be built without re-running the d=8 BFS.)
