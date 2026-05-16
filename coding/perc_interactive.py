"""
Site Percolation — Interactive Matplotlib App
Proposal A + selectable backend (Numba union-find  OR  scipy.ndimage)

Usage
─────
  python3 perc_interactive.py              # default: numba
  python3 perc_interactive.py --scipy      # use scipy.ndimage instead
"""

import argparse
import threading
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ── CLI ────────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description='Site percolation interactive app.')
_parser.add_argument(
    '--scipy', action='store_true',
    help='Use scipy.ndimage backend instead of Numba (slower, no extra deps).'
)
_args   = _parser.parse_args()
BACKEND = 'scipy' if _args.scipy else 'numba'

# ── Tuning knobs ───────────────────────────────────────────────────────────────

GRID_SIZES       = [64, 128, 256, 512]
P_INIT           = 0.592746
P_CRIT           = 0.592746
SLIDER_STEPS     = 500
SEEDS            = {n: 42 for n in GRID_SIZES}
PREFETCH_WORKERS = 6
PREFETCH_OFFSETS = [
     0.002, 0.005, 0.01, 0.02, 0.05,
    -0.002,-0.005,-0.01,-0.02,-0.05,
]

# ── Colours ────────────────────────────────────────────────────────────────────

DARK_BG      = '#12121e'
BTN_INACTIVE = '#2e2e4a'
BTN_ACTIVE   = '#7b68ee'
TEXT_COLOR   = '#e0e0f0'
BASE_CMAP    = plt.cm.gist_ncar

# ══════════════════════════════════════════════════════════════════════════════
# 1a. SCIPY BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def label_clusters_scipy(m: np.ndarray) -> np.ndarray:
    """
    Cluster labeling via scipy.ndimage.measurements.
    Returns areaImg (float32): each occupied site gets the size of its cluster.
    Simple and dependency-free beyond scipy, but ~30-60× slower than Numba
    for large N.
    """
    from scipy.ndimage import measurements
    lw, _ = measurements.label(m)
    area  = measurements.sum(m, lw, index=np.arange(lw.max() + 1))
    return area[lw].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1b. NUMBA BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def label_clusters_numba(m: np.ndarray) -> np.ndarray:
    """
    Cluster labeling via JIT-compiled flat union-find (path-halving +
    union-by-rank).  4-connected.
    Returns areaImg (float32): each occupied site gets the size of its cluster.
    ~30-60× faster than scipy for large N.  Compiled once, cached to disk.
    """
    return _label_numba_jit(m)


def _init_numba():
    """Import and JIT-compile Numba kernels.  Called once at startup."""
    import numba

    @numba.njit(cache=True)
    def _find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @numba.njit(cache=True)
    def _union(parent, rank, a, b):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    @numba.njit(cache=True)
    def _label(m):
        n, N2  = m.shape[0], m.shape[0] * m.shape[0]
        parent = np.arange(N2)
        rank   = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if not m[i, j]:
                    continue
                idx = i * n + j
                if j + 1 < n and m[i, j + 1]:
                    _union(parent, rank, idx, i * n + j + 1)
                if i + 1 < n and m[i + 1, j]:
                    _union(parent, rank, idx, (i + 1) * n + j)
        sizes = np.zeros(N2, numba.int32)
        for i in range(n):
            for j in range(n):
                if m[i, j]:
                    sizes[_find(parent, i * n + j)] += 1
        out = np.zeros((n, n), numba.float32)
        for i in range(n):
            for j in range(n):
                if m[i, j]:
                    out[i, j] = sizes[_find(parent, i * n + j)]
        return out

    _label(np.ones((4, 4), dtype=bool))   # trigger compilation
    return _label


# ══════════════════════════════════════════════════════════════════════════════
# 2. BACKEND SELECTION & STARTUP
# ══════════════════════════════════════════════════════════════════════════════

print("Initialising percolation engine...")
print(f"  Backend: {BACKEND}")

if BACKEND == 'numba':
    print("  Compiling Numba kernels...", end=' ', flush=True)
    _label_numba_jit = _init_numba()
    print("done.")
    _label_fn = label_clusters_numba
else:
    _label_fn = label_clusters_scipy

print("  Generating uniform coupling arrays...", end=' ', flush=True)
_U: dict = {}
for _n in GRID_SIZES:
    _rng   = np.random.default_rng(SEEDS[_n])
    _U[_n] = _rng.random((_n, _n)).astype(np.float32)
print("done.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. THREAD-SAFE LAZY CACHE
# ══════════════════════════════════════════════════════════════════════════════

_cache      : dict = {}
_cache_lock        = threading.RLock()
_executor          = concurrent.futures.ThreadPoolExecutor(
                         max_workers=PREFETCH_WORKERS)

def _round_p(p: float) -> float:
    return round(p * SLIDER_STEPS) / SLIDER_STEPS

def _compute(n: int, p: float) -> np.ndarray:
    return _label_fn(_U[n] < p)

def _compute_and_store(n: int, p: float):
    key = (n, _round_p(p))
    with _cache_lock:
        if key in _cache:
            return
    result = _compute(n, p)
    with _cache_lock:
        _cache[key] = result

def get_area_img(n: int, p: float) -> np.ndarray:
    key = (n, _round_p(p))
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    result = _compute(n, p)
    with _cache_lock:
        _cache[key] = result
    return result

def prefetch(n: int, p: float):
    for dp in PREFETCH_OFFSETS:
        pp = float(np.clip(p + dp, 0.0, 1.0))
        key = (n, _round_p(pp))
        with _cache_lock:
            if key in _cache:
                continue
        _executor.submit(_compute_and_store, n, pp)


print("  Pre-warming cache at p_init...", end=' ', flush=True)
for _n in GRID_SIZES:
    get_area_img(_n, P_INIT)
print("done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COLORMAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_cmap(max_val: int) -> ListedColormap:
    n_colors = max(int(max_val) + 1, 2)
    colors   = ['black']
    if n_colors > 1:
        colors += list(BASE_CMAP(np.linspace(0.01, 0.99, n_colors - 1)))
    return ListedColormap(colors)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MATPLOTLIB UI
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(8.2, 8.2))   # slightly wider to fit sidebar
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(
    4, 1, figure=fig,
    height_ratios=[0.7, 8, 0.3, 0.7],
    left=0.08, right=0.82,            # leave right margin for sidebar
    top=0.93, bottom=0.05, hspace=0.15,
)
ax_grid        = fig.add_subplot(gs[1, 0])
ax_slider_host = fig.add_subplot(gs[3, 0])

ax_grid.set_facecolor('black')
for sp in ax_grid.spines.values():
    sp.set_edgecolor('#444466')

fig.suptitle('Site Percolation — Square Lattice',
             fontsize=13, fontweight='bold', color=TEXT_COLOR, y=0.975)

# ── Legend sidebar ─────────────────────────────────────────────────────────────
#
#   [top label]   "large clusters"
#   [ gradient ]  gist_ncar strip, top = large, bottom = small
#   [bot label]   "small clusters"
#   [  spacer  ]
#   [black rect]  closed (unoccupied) site
#

# Gradient strip  [left, bottom, width, height]  in figure coordinates
ax_grad  = fig.add_axes([0.87, 0.30, 0.045, 0.45])
ax_black = fig.add_axes([0.87, 0.18, 0.045, 0.06])

# Draw the gradient: 256 rows, value increases downward → flip so large=top
_grad_data = np.linspace(1, 0, 256).reshape(256, 1)
ax_grad.imshow(_grad_data, aspect='auto', cmap=BASE_CMAP,
               vmin=0, vmax=1, origin='upper')
ax_grad.set_xticks([])
ax_grad.set_yticks([])
for sp in ax_grad.spines.values():
    sp.set_edgecolor('#444466')

# Labels beside the gradient
fig.text(0.915, 0.755, 'large\nclusters', ha='left', va='center',
         fontsize=8.5, color=TEXT_COLOR, linespacing=1.4)
fig.text(0.915, 0.305, 'small\nclusters', ha='left', va='center',
         fontsize=8.5, color=TEXT_COLOR, linespacing=1.4)

# Black patch for closed sites
ax_black.set_facecolor('black')
ax_black.set_xticks([])
ax_black.set_yticks([])
for sp in ax_black.spines.values():
    sp.set_edgecolor('#444466')

fig.text(0.915, 0.210, 'closed\n(unoccupied)', ha='left', va='center',
         fontsize=8.5, color=TEXT_COLOR, linespacing=1.4)

# Section title
fig.text(0.893, 0.795, 'colour\nguide', ha='center', va='bottom',
         fontsize=8.5, color='#aaaacc', fontstyle='italic', linespacing=1.4)

# ── N-selector buttons ─────────────────────────────────────────────────────────

_btn_axes, _btns = [], []
_btn_w = 0.155
_gap   = (0.74 - len(GRID_SIZES) * _btn_w) / (len(GRID_SIZES) - 1)
_x0    = 0.08
state  = {'n': GRID_SIZES[1]}

for _i, _n in enumerate(GRID_SIZES):
    _bax = fig.add_axes([_x0 + _i * (_btn_w + _gap), 0.905, _btn_w, 0.045])
    _btn = Button(_bax, f'N = {_n}', color=BTN_INACTIVE, hovercolor='#4a4a7a')
    _btn.label.set_color(TEXT_COLOR)
    _btn.label.set_fontsize(10)
    _btn_axes.append(_bax)
    _btns.append(_btn)

def _highlight_buttons():
    for i, n in enumerate(GRID_SIZES):
        _btns[i].ax.set_facecolor(BTN_ACTIVE if n == state['n'] else BTN_INACTIVE)
    fig.canvas.draw_idle()

# ── Slider ─────────────────────────────────────────────────────────────────────

ax_slider_host.set_facecolor(DARK_BG)
slider = Slider(
    ax=ax_slider_host, label='p',
    valmin=0.0, valmax=1.0, valinit=P_INIT,
    valstep=1.0 / SLIDER_STEPS, color=BTN_ACTIVE,
)
slider.label.set_color(TEXT_COLOR)
slider.label.set_fontsize(11)
slider.valtext.set_color(TEXT_COLOR)
ax_slider_host.axvline(x=P_CRIT, color='yellow', lw=1.5, alpha=0.6, zorder=5)

# _regime_text = fig.text(0.45, 0.033, '', ha='center',
#                         fontsize=10, color=TEXT_COLOR)

# def _regime(p):
#     if   p < P_CRIT - 0.01: return 'subcritical  —  disconnected phase'
#     elif p > P_CRIT + 0.01: return 'supercritical  —  connected phase'
#     else:                    return 'critical  —  p ≈ p_c'

# ── Grid draw ──────────────────────────────────────────────────────────────────

_im = [None]

def draw_grid(p: float):
    n       = state['n']
    img     = get_area_img(n, p)
    max_val = int(img.max())
    cmap    = build_cmap(max_val)

    if _im[0] is None:
        ax_grid.cla()
        _im[0] = ax_grid.imshow(
            img, cmap=cmap, interpolation='none',
            vmin=0, vmax=max(max_val, 1), aspect='equal',
        )
    else:
        _im[0].set_data(img)
        _im[0].set_cmap(cmap)
        _im[0].set_clim(0, max(max_val, 1))

    ax_grid.set_xticks([]); ax_grid.set_yticks([])
    ax_grid.set_title(
        f'N = {n}   |   {n}×{n} = {n*n:,} sites',
        fontsize=11, color=TEXT_COLOR, pad=6,
    )
    # _regime_text.set_text(_regime(p))
    prefetch(n, p)
    fig.canvas.draw_idle()

# ── Callbacks ──────────────────────────────────────────────────────────────────

slider.on_changed(lambda val: draw_grid(slider.val))

def _make_btn_cb(n):
    def cb(event):
        state['n'] = n
        _im[0] = None
        _highlight_buttons()
        draw_grid(slider.val)
        prefetch(n, slider.val)
    return cb

for _btn, _n in zip(_btns, GRID_SIZES):
    _btn.on_clicked(_make_btn_cb(_n))

# ── Initial render ─────────────────────────────────────────────────────────────

draw_grid(P_INIT)
_highlight_buttons()
for _n in GRID_SIZES:
    prefetch(_n, P_INIT)

plt.show()
_executor.shutdown(wait=False)
