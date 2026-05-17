"""
Fast 2D frame generation using a PIL bitmap approach.

No matplotlib per frame. Pipeline:
  1. Render entire cluster into one RGBA numpy array (canvas).
  2. For each animation frame, crop a window centred on the origin,
     then resize to TARGET_SIZE with NEAREST-neighbour (preserves the
     discrete lattice look — pixels zoom in/out).
  3. Overlay a title bar drawn with PIL ImageDraw.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from percolation.critical_params import (
    ANIM_2D, CMAP_NAME, BG_COLOR, TARGET_SIZE, FRACTAL_DIMS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _get_fonts():
    """Return (font_large, font_small) — falls back to default if TTF missing."""
    candidates_bold = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ]
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/TTF/DejaVuSans.ttf',
    ]
    fl = fs = None
    for p in candidates_bold:
        try:
            fl = ImageFont.truetype(p, 18);  break
        except Exception:
            pass
    for p in candidates:
        try:
            fs = ImageFont.truetype(p, 12);  break
        except Exception:
            pass
    if fl is None:
        fl = ImageFont.load_default()
    if fs is None:
        fs = ImageFont.load_default()
    return fl, fs


def _draw_title(img, line1, line2, line3=''):
    """Paint a dark title bar at the top and write three text lines."""
    draw       = ImageDraw.Draw(img)
    W, _H      = img.size
    fl, fs     = _get_fonts()
    bar_height = 56
    draw.rectangle([(0, 0), (W, bar_height)], fill=(5, 5, 20))
    draw.text((10,  4), line1, fill=(255, 255, 255), font=fl)
    draw.text((10, 27), line2, fill=(170, 170, 180), font=fs)
    if line3:
        draw.text((10, 42), line3, fill=(255, 200, 60),  font=fs)
    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_2d_frames(cluster, cfg):
    """
    Generate all animation frames for d=2 as a list of PIL RGB images.

    Fast path: build one large canvas, then crop + resize per frame.
    """
    print("  [d=2] Building bitmap canvas ... ", end='', flush=True)

    coords    = cluster['coords']          # list of (x, y) tuples
    depths    = cluster['depths']
    max_depth = cluster['max_depth']

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad = 10
    W   = max_x - min_x + 2 * pad + 1
    H   = max_y - min_y + 2 * pad + 1

    bg     = _hex_to_rgb(BG_COLOR)
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)

    cmap = plt.get_cmap(CMAP_NAME)
    for coord in coords:
        x, y = coord
        col  = x - min_x + pad
        row  = (max_y - y) + pad           # flip y so +y is up
        d    = depths[coord]
        r, g, b, _ = cmap(d / max(max_depth, 1))
        canvas[row, col] = (int(r * 255), int(g * 255), int(b * 255))

    canvas_pil = Image.fromarray(canvas, mode='RGB')

    # Origin position in canvas (column, row)
    ox = 0 - min_x + pad
    oy = (max_y - 0) + pad

    # Zoom extents
    zoom_start_r = float(cfg.get('zoom_start_r', 30))
    full_r       = float(max(max_x - 0, 0 - min_x,
                              max_y - 0, 0 - min_y) + pad)

    timing = ANIM_2D
    total  = timing['total']

    d_f     = FRACTAL_DIMS.get(2, '?')
    d_f_str = f'{d_f:.3f}' if isinstance(d_f, float) else str(d_f)

    print(f"canvas {W}x{H} px, {cluster['size']:,} sites, {total} frames")

    frames = []
    for i in range(total):
        # Current crop radius
        if i < timing['zoom_in_frames']:
            r = zoom_start_r
        elif i < timing['zoom_in_frames'] + timing['zoom_out_frames']:
            t      = (i - timing['zoom_in_frames']) / timing['zoom_out_frames']
            t_ease = 1.0 - (1.0 - t) ** 3            # cubic ease-out
            r      = zoom_start_r + t_ease * (full_r - zoom_start_r)
        else:
            r = full_r

        left   = max(0,  int(ox - r))
        right  = min(W,  int(ox + r) + 1)
        top    = max(0,  int(oy - r))
        bottom = min(H,  int(oy + r) + 1)

        crop  = canvas_pil.crop((left, top, right, bottom))
        frame = crop.resize(TARGET_SIZE, Image.NEAREST)

        regime  = 'non-mean-field'
        frame = _draw_title(
            frame,
            line1=f'Dimension 2  —  Site Percolation at Criticality',
            line2=(f'p_c = {cfg["p_c"]:.6f}   |   '
                   f'Cluster: {cluster["size"]:,} sites   |   '
                   f'Color = BFS depth from origin'),
            line3=f'Fractal dimension  d_f = {d_f_str}  ({regime})',
        )
        frames.append(frame)

    return frames