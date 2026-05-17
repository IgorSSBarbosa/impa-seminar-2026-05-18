"""
3D scatter rendering for d=3 percolation cluster.

Animation strategy:
  - Reveal sites progressively by increasing L-inf radius (simulates zoom-out).
  - During the freeze phase, slowly rotate the camera.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers projection
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from percolation.critical_params import (
    ANIM_3D, FIGURE_SIZE, FIGURE_DPI, CMAP_NAME, BG_COLOR,
    TARGET_SIZE, FRACTAL_DIMS,
)


# ---------------------------------------------------------------------------
# Shared utilities (imported by render_graph.py as well)
# ---------------------------------------------------------------------------

def fig_to_pil(fig):
    """Save a matplotlib figure to a PIL RGB image, then close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=FIGURE_DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img.convert('RGB').resize(TARGET_SIZE, Image.LANCZOS)


def style_3d_axes(ax):
    """Apply consistent dark styling to a 3D axes object."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#1a1a3a')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_3d_frames(cluster, cfg):
    """
    Generate all animation frames for d=3 as a list of PIL RGB images.

    Shows sites within an increasing L-inf radius to simulate zoom-out.
    """
    dim         = cluster['dim']
    max_display = cfg.get('max_display_nodes', 3000)

    # Take the first max_display nodes (sorted by BFS depth = closest to origin)
    coords_list = cluster['coords'][:max_display]
    n           = len(coords_list)

    coords      = np.array(coords_list)                  # (N, 3)
    depths_arr  = np.array([cluster['depths'][c] for c in coords_list])
    max_depth   = depths_arr.max()

    x, y, z     = coords[:, 0], coords[:, 1], coords[:, 2]
    linf_radii  = np.max(np.abs(coords), axis=1)

    cmap        = plt.get_cmap(CMAP_NAME)
    colors      = cmap(depths_arr / max(max_depth, 1))   # (N, 4)

    zoom_start_r = float(cfg.get('zoom_start_r', 12))
    full_r       = float(linf_radii.max()) + 2.0

    d_f     = FRACTAL_DIMS.get(dim, '?')
    d_f_str = f'{d_f:.3f}' if isinstance(d_f, float) else str(d_f)

    timing = ANIM_3D
    total  = timing['total']

    print(f"  [d=3] {n:,} sites, {total} frames")

    frames = []
    for i in tqdm(range(total), desc='  d=3  '):
        # Current reveal radius
        if i < timing['zoom_in_frames']:
            r = zoom_start_r
        elif i < timing['zoom_in_frames'] + timing['zoom_out_frames']:
            t      = (i - timing['zoom_in_frames']) / timing['zoom_out_frames']
            t_ease = 1.0 - (1.0 - t) ** 3
            r      = zoom_start_r + t_ease * (full_r - zoom_start_r)
        else:
            r = full_r

        # Camera azimuth: fixed during zoom, slow spin during freeze
        if i < timing['zoom_in_frames'] + timing['zoom_out_frames']:
            azim = 40.0
        else:
            freeze_i = i - timing['zoom_in_frames'] - timing['zoom_out_frames']
            azim     = 40.0 + freeze_i * 3.5

        mask = linf_radii <= r

        fig = plt.figure(figsize=FIGURE_SIZE, facecolor=BG_COLOR)
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(BG_COLOR)

        if mask.any():
            ax.scatter(
                x[mask], y[mask], z[mask],
                c=colors[mask],
                s=6,
                alpha=0.85,
                linewidths=0,
                depthshade=True,
                rasterized=True,
            )

        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        style_3d_axes(ax)
        ax.view_init(elev=22, azim=azim)

        fig.suptitle(
            f'Dimension 3  —  Site Percolation at Criticality\n'
            f'$p_c = {cfg["p_c"]:.4f}$'
            f'   |   {int(mask.sum()):,} / {n:,} sites shown'
            f'   |   $d_f \\approx {d_f_str}$  (non-mean-field)',
            color='white', fontsize=10, y=0.97,
        )

        frames.append(fig_to_pil(fig))

    return frames