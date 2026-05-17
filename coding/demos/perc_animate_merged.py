"""
Percolation universality — merged side-by-side animation.

Renders all three lattices (honeycomb / square / triangular) into a SINGLE figure,
swept synchronously through p ∈ [0,1] for each grid size in GRID_SIZES, and writes
ONE animation file (GIF or MP4) instead of three.

Same uniform-coupling trick as `perc_animate.py`: a single random matrix U is
shared across the lattices, so each frame compares the *same* random landscape
under three different connectivity rules.

Usage
─────
  python3 perc_animate_merged.py                 # default: GIF, fps=15, 3 segments
  python3 perc_animate_merged.py --mp4           # try MP4 (falls back to GIF)
  python3 perc_animate_merged.py --grid-sizes 256
  python3 perc_animate_merged.py --frames-per-n 60 --fps 12   # smaller file
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import ListedColormap

# ── CLI ────────────────────────────────────────────────────────────────────────

_p = argparse.ArgumentParser(description="Merged three-lattice percolation animation.")
_p.add_argument("--scipy", action="store_true",
                help="Use scipy/BFS backend instead of Numba.")
_p.add_argument("--mp4", action="store_true",
                help="Save as MP4 (ffmpeg). Default writes a GIF.")
_p.add_argument("--fps", type=int, default=5,
                help="Frames per second of the output (default: 15).")
_p.add_argument("--frames-per-n", type=int, default=80,
                help="Animation frames per grid-size segment (default: 80).")
_p.add_argument("--grid-sizes", type=int, nargs="+", default=[64, 256, 1024],
                help="Grid sizes to sweep through (default: 64 256 1024).")
_p.add_argument("--dpi", type=int, default=90,
                help="DPI for rasterization (default: 90).")
_p.add_argument("--out", type=Path,
                default=Path(__file__).resolve().parent.parent.parent
                        / "images" / "lattices_merged",
                help="Output path (extension is set automatically). "
                     "Default writes to <repo>/images/lattices_merged.{gif,mp4}.")
args = _p.parse_args()

BACKEND      = "scipy" if args.scipy else "numba"
SAVE_MP4     = args.mp4
FPS          = args.fps
FRAMES_PER_N = args.frames_per_n
GRID_SIZES   = args.grid_sizes
DPI          = args.dpi
OUT_BASE     = args.out
SEED         = 42

TOTAL_FRAMES = len(GRID_SIZES) * FRAMES_PER_N

# Panel order: 3-connected → 4-connected → 6-connected (matches the chalkboard).
PANEL_ORDER = ["hexagonal", "square", "triangular"]

MODELS = {
    "square": {
        "label": "Square (4-connected)",
        "p_c":   0.592746,
        "color": "#7b68ee",
    },
    "triangular": {
        "label": "Triangular (6-connected)",
        "p_c":   0.5,
        "color": "#ff7f50",
    },
    "hexagonal": {
        "label": "Honeycomb (3-connected)",
        "p_c":   0.6962,
        "color": "#3cb371",
    },
}

DARK_BG    = "#12121e"
TEXT_COLOR = "#e0e0f0"
BASE_CMAP  = plt.cm.gist_ncar
MAX_COLORS = 1024


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLUSTER-LABELING BACKENDS  (same logic as perc_animate.py)
# ══════════════════════════════════════════════════════════════════════════════

def _init_numba():
    import numba

    @numba.njit(cache=True)
    def _find(par, x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    @numba.njit(cache=True)
    def _union(par, rnk, a, b):
        ra, rb = _find(par, a), _find(par, b)
        if ra == rb:
            return
        if rnk[ra] < rnk[rb]:
            ra, rb = rb, ra
        par[rb] = ra
        if rnk[ra] == rnk[rb]:
            rnk[ra] += 1

    @numba.njit(cache=True)
    def _label_sq(m):
        n, N2 = m.shape[0], m.shape[0] ** 2
        par = np.arange(N2); rnk = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if not m[i, j]: continue
                idx = i * n + j
                if j + 1 < n and m[i, j + 1]:   _union(par, rnk, idx, i * n + j + 1)
                if i + 1 < n and m[i + 1, j]:   _union(par, rnk, idx, (i + 1) * n + j)
        sz = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if m[i, j]: sz[_find(par, i * n + j)] += 1
        out = np.zeros((n, n), numba.float32)
        for i in range(n):
            for j in range(n):
                if m[i, j]: out[i, j] = sz[_find(par, i * n + j)]
        return out

    @numba.njit(cache=True)
    def _label_tri(m):
        n, N2 = m.shape[0], m.shape[0] ** 2
        par = np.arange(N2); rnk = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if not m[i, j]: continue
                idx = i * n + j
                if j + 1 < n and m[i, j + 1]:           _union(par, rnk, idx, i * n + j + 1)
                if i + 1 < n and m[i + 1, j]:           _union(par, rnk, idx, (i + 1) * n + j)
                if i + 1 < n and j + 1 < n and m[i + 1, j + 1]:
                    _union(par, rnk, idx, (i + 1) * n + j + 1)
        sz = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if m[i, j]: sz[_find(par, i * n + j)] += 1
        out = np.zeros((n, n), numba.float32)
        for i in range(n):
            for j in range(n):
                if m[i, j]: out[i, j] = sz[_find(par, i * n + j)]
        return out

    @numba.njit(cache=True)
    def _label_hex(m):
        n, N2 = m.shape[0], m.shape[0] ** 2
        par = np.arange(N2); rnk = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if not m[i, j]: continue
                idx = i * n + j
                if j + 1 < n and m[i, j + 1]:
                    _union(par, rnk, idx, i * n + j + 1)
                if (i + j) % 2 == 0 and i + 1 < n and m[i + 1, j]:
                    _union(par, rnk, idx, (i + 1) * n + j)
        sz = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if m[i, j]: sz[_find(par, i * n + j)] += 1
        out = np.zeros((n, n), numba.float32)
        for i in range(n):
            for j in range(n):
                if m[i, j]: out[i, j] = sz[_find(par, i * n + j)]
        return out

    _w = np.ones((4, 4), dtype=bool)
    _label_sq(_w); _label_tri(_w); _label_hex(_w)
    return {"square": _label_sq, "triangular": _label_tri, "hexagonal": _label_hex}


def _label_scipy(m, model):
    if model == "hexagonal":
        return _label_hex_bfs(m)
    from scipy.ndimage import measurements
    structs = {
        "square":     np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        "triangular": np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    }
    lw, _ = measurements.label(m, structure=structs[model])
    area  = measurements.sum(m, lw, index=np.arange(lw.max() + 1))
    return area[lw].astype(np.float32)


def _label_hex_bfs(m):
    from collections import deque
    n      = m.shape[0]
    labels = np.zeros((n, n), dtype=np.int32)
    cur    = 0
    for si in range(n):
        for sj in range(n):
            if not m[si, sj] or labels[si, sj]:
                continue
            cur += 1
            q = deque([(si, sj)])
            labels[si, sj] = cur
            while q:
                ci, cj = q.popleft()
                if cj + 1 < n and m[ci, cj + 1] and not labels[ci, cj + 1]:
                    labels[ci, cj + 1] = cur; q.append((ci, cj + 1))
                if cj - 1 >= 0 and m[ci, cj - 1] and not labels[ci, cj - 1]:
                    labels[ci, cj - 1] = cur; q.append((ci, cj - 1))
                if (ci + cj) % 2 == 0:
                    if ci + 1 < n and m[ci + 1, cj] and not labels[ci + 1, cj]:
                        labels[ci + 1, cj] = cur; q.append((ci + 1, cj))
                else:
                    if ci - 1 >= 0 and m[ci - 1, cj] and not labels[ci - 1, cj]:
                        labels[ci - 1, cj] = cur; q.append((ci - 1, cj))
    if cur == 0:
        return np.zeros((n, n), dtype=np.float32)
    area = np.bincount(labels.ravel(), minlength=cur + 1).astype(np.float32)
    return (area[labels] * m).astype(np.float32)


# ── Startup ────────────────────────────────────────────────────────────────────

print(f"\nMerged percolation animation generator")
print(f"  Backend      : {BACKEND}")
print(f"  Grid sizes   : {GRID_SIZES}")
print(f"  Frames/size  : {FRAMES_PER_N}  →  {TOTAL_FRAMES} total")
print(f"  FPS          : {FPS}   duration ≈ {TOTAL_FRAMES / FPS:.1f}s")
print(f"  Format       : {'MP4' if SAVE_MP4 else 'GIF'}")
print(f"  Output base  : {OUT_BASE}\n")

if BACKEND == "numba":
    print("Compiling Numba kernels...", end=" ", flush=True)
    _numba_fns = _init_numba()
    print("done.")
    def label_fn(m, model): return _numba_fns[model](m)
else:
    def label_fn(m, model): return _label_scipy(m, model)

print("Generating coupling arrays...", end=" ", flush=True)
_U = {n: np.random.default_rng(SEED).random((n, n)).astype(np.float32) for n in GRID_SIZES}
print("done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. COLORMAP HELPER
# ══════════════════════════════════════════════════════════════════════════════

def build_cmap(max_val: int) -> ListedColormap:
    k = min(max(int(max_val) + 1, 2), MAX_COLORS)
    colors = ["black"] + list(BASE_CMAP(np.linspace(0.01, 0.99, k - 1)))
    return ListedColormap(colors)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def make_figure():
    fig = plt.figure(figsize=(15, 6.2))
    fig.patch.set_facecolor(DARK_BG)

    # Layout: title row → 3 image columns → progress bar row.
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[10, 0.7],
        left=0.03, right=0.985,
        top=0.86, bottom=0.07, hspace=0.18, wspace=0.05,
    )
    ax_imgs = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_prog = fig.add_subplot(gs[1, :])

    for ax in ax_imgs:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    # Per-panel title text (set later).
    titles = []
    for c, key in enumerate(PANEL_ORDER):
        info = MODELS[key]
        t = ax_imgs[c].set_title(info["label"], fontsize=12,
                                  color=info["color"], pad=8, fontweight="bold")
        titles.append(t)

    # Shared progress axis: x ∈ [0,1] showing p; vertical markers at each p_c.
    ax_prog.set_facecolor(DARK_BG)
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_yticks([])
    ax_prog.tick_params(axis="x", colors=TEXT_COLOR, labelsize=9)
    for sp in ax_prog.spines.values():
        sp.set_edgecolor("#444466")

    ax_prog.axhspan(0.35, 0.65, color="#2e2e4a", zorder=1)

    pc_lines = []
    for key in PANEL_ORDER:
        info = MODELS[key]
        ln = ax_prog.axvline(info["p_c"], color=info["color"],
                             lw=1.6, alpha=0.9, zorder=4)
        ax_prog.text(info["p_c"], 1.05,
                     f"$p_c$={info['p_c']:.3f}",
                     ha="center", va="bottom",
                     fontsize=8, color=info["color"], zorder=5,
                     transform=ax_prog.get_xaxis_transform())
        pc_lines.append(ln)

    prog_fill = ax_prog.barh(0.5, 0, height=0.30,
                             color="#ffffff", alpha=0.85, zorder=2, left=0)[0]
    prog_lbl  = ax_prog.text(0.5, 0.5, "", ha="center", va="center",
                             fontsize=10, color="#12121e",
                             fontweight="bold", zorder=6)

    suptitle = fig.suptitle("", color=TEXT_COLOR, fontsize=14,
                             fontweight="bold", y=0.965)

    # Image handles populated on first frame.
    im_handles = [None, None, None]

    return dict(
        fig=fig, ax_imgs=ax_imgs, ax_prog=ax_prog,
        prog_fill=prog_fill, prog_lbl=prog_lbl,
        suptitle=suptitle, im_handles=im_handles,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANIMATION RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    ui = make_figure()
    fig         = ui["fig"]
    ax_imgs     = ui["ax_imgs"]
    prog_fill   = ui["prog_fill"]
    prog_lbl    = ui["prog_lbl"]
    suptitle    = ui["suptitle"]
    im_handles  = ui["im_handles"]

    def update(frame_idx):
        seg     = frame_idx // FRAMES_PER_N
        local_f = frame_idx  % FRAMES_PER_N
        n       = GRID_SIZES[seg]
        p       = local_f / max(FRAMES_PER_N - 1, 1)

        mask = _U[n] < p
        for c, key in enumerate(PANEL_ORDER):
            img      = label_fn(mask, key)
            max_val  = int(img.max())
            cmap     = build_cmap(max_val)

            if (im_handles[c] is None
                    or im_handles[c].get_array().shape != img.shape):
                ax_imgs[c].cla()
                ax_imgs[c].set_facecolor("black")
                ax_imgs[c].set_xticks([]); ax_imgs[c].set_yticks([])
                for sp in ax_imgs[c].spines.values():
                    sp.set_edgecolor("#444466")
                info = MODELS[key]
                ax_imgs[c].set_title(info["label"], fontsize=12,
                                     color=info["color"], pad=8,
                                     fontweight="bold")
                im_handles[c] = ax_imgs[c].imshow(
                    img, cmap=cmap, interpolation="none",
                    vmin=0, vmax=max(max_val, 1), aspect="equal",
                )
            else:
                im_handles[c].set_data(img)
                im_handles[c].set_cmap(cmap)
                im_handles[c].set_clim(0, max(max_val, 1))

        suptitle.set_text(
            f"Site percolation on three lattices  |  "
            f"N = {n}   ({n}×{n} = {n*n:,} sites)   |   "
            f"same uniform coupling U across panels"
        )
        prog_fill.set_width(p)
        prog_lbl.set_text(f"p = {p:.3f}")
        return [*im_handles, prog_fill, prog_lbl, suptitle]

    anim = FuncAnimation(
        fig, update,
        frames=TOTAL_FRAMES,
        interval=1000 // FPS,
        blit=False,
    )

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_MP4:
        writer = FFMpegWriter(fps=FPS, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
        out_path = OUT_BASE.with_suffix(".mp4")
    else:
        writer = PillowWriter(fps=FPS)
        out_path = OUT_BASE.with_suffix(".gif")

    print(f"  Saving → {out_path}")
    t0 = time.perf_counter()
    try:
        anim.save(str(out_path), writer=writer, dpi=DPI,
                  progress_callback=_progress_cb)
    except Exception as e:
        if SAVE_MP4 and "ffmpeg" in str(e).lower():
            print(f"\n  ffmpeg not found — retrying as GIF...")
            out_path = OUT_BASE.with_suffix(".gif")
            writer   = PillowWriter(fps=FPS)
            anim.save(str(out_path), writer=writer, dpi=DPI,
                      progress_callback=_progress_cb)
        else:
            raise
    elapsed = time.perf_counter() - t0

    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Done in {elapsed:.1f}s  →  {out_path}  ({sz_mb:.1f} MB)")


def _progress_cb(current_frame, total_frames):
    pct = 100 * current_frame / total_frames
    bar = "#" * int(pct / 2)
    sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  frame {current_frame}/{total_frames}")
    sys.stdout.flush()


if __name__ == "__main__":
    render()
